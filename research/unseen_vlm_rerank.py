"""VLM top-K rerank on the pseudo-unseen proxy (single-pipeline loophole A).

Uniform cascade for every image:
  1) score with deployed unseen recipe: dbnorm(ctft@taxon)+0.5*dbnorm(L@name)
  2) take top-K from scores (NOT from splits)
  3) Qwen2.5-VL-7B picks one of the K names

Compliance: no splits/*.pkl membership; same recipe for all; disclose VLM.

  python research/unseen_vlm_rerank.py --n 300 --k 10
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time

import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import FishData, dbnorm, OUT, DATA  # noqa: E402

IMG = os.path.join(DATA, 'dl', 'images')


def parse_choice(text: str, k: int) -> int | None:
    """Return 0-based index or None."""
    t = text.strip()
    m = re.search(r'\b([1-9][0-9]*)\b', t)
    if not m:
        return None
    v = int(m.group(1))
    if 1 <= v <= k:
        return v - 1
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=300, help='subsample size')
    ap.add_argument('--k', type=int, default=10)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--model', default='Qwen/Qwen2.5-VL-7B-Instruct')
    ap.add_argument('--nodesc', action='store_true')
    a = ap.parse_args()

    D = FishData()
    desc = json.load(open(os.path.join(DATA, 'dl', 'descriptions.json')))
    cn = json.load(open(os.path.join(OUT, 'common_names.json')))

    # deployed-ish scores: ctftbig @ taxon + 0.5 L @ name
    ct = torch.load(os.path.join(OUT, 'emb_train_ctftbig.pt'), weights_only=False)
    cidx = {fn: i for i, fn in enumerate(ct['files'])}
    cF = F.normalize(ct['feats'].float(), dim=-1)

    files, gold_cls = [], []
    for c in D.pseudo:
        for fn in D.by[c]:
            if fn in cidx and fn in D.LtI:
                files.append(fn)
                gold_cls.append(D.ci[c])
    gold_cls = torch.tensor(gold_cls)
    Qc = torch.stack([cF[cidx[fn]] for fn in files])
    Ql = torch.stack([D.LtF[D.LtI[fn]] for fn in files])
    cand = D.cand
    S = dbnorm(Qc @ D.TtH[cand].t()) + 0.5 * dbnorm(Ql @ D.TnL[cand].t())
    gold = torch.tensor([D.cand_pos[int(g)] for g in gold_cls.tolist()])

    base_pred = S.argmax(1)
    topk = S.topk(a.k, dim=1).indices  # [N,K] into cand columns

    random.seed(a.seed)
    idxs = list(range(len(files)))
    random.shuffle(idxs)
    idxs = idxs[: a.n]

    def acc(pred_idx, subset):
        g = gold[subset]
        p = pred_idx[subset]
        return (p == g).float().mean().item() * 100

    subset = torch.tensor(idxs)
    base_top1 = acc(base_pred, subset)
    in_k = (topk[subset] == gold[subset].unsqueeze(1)).any(dim=1).float().mean().item() * 100
    print(f'subsample n={len(idxs)} K={a.k}', flush=True)
    print(f'text baseline top-1: {base_top1:.2f}', flush=True)
    print(f'gold in top-K:       {in_k:.2f}  (oracle ceiling if perfect VLM)', flush=True)

    def cand_label(sp: str) -> str:
        name = cn.get(sp) or sp
        if a.nodesc:
            return f'{name} ({sp})'
        d = desc.get(sp, '')
        d = (d[:180] + '…') if len(d) > 180 else d
        return f'{name} ({sp}): {d}'

    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info

    print(f'loading {a.model} ...', flush=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        a.model, torch_dtype=torch.bfloat16, device_map='auto'
    ).eval()
    proc = AutoProcessor.from_pretrained(a.model)

    vlm_pred = base_pred.clone()
    parsed = 0
    in_short_correct = 0
    in_short_n = 0
    t0 = time.time()

    for j, i in enumerate(idxs):
        fn = files[i]
        path = os.path.join(IMG, fn)
        if not os.path.exists(path):
            continue
        cols = topk[i].tolist()
        names = [D.classes[int(cand[c])] for c in cols]
        listing = '\n'.join(f'{t+1}. {cand_label(nm)}' for t, nm in enumerate(names))
        prompt = (
            f'You are identifying a fish species from a photo. '
            f'Exactly one of the {a.k} candidates is correct. '
            f'Reply with ONLY the number 1-{a.k}.\n\n{listing}'
        )
        msgs = [{'role': 'user', 'content': [
            {'type': 'image', 'image': path},
            {'type': 'text', 'text': prompt},
        ]}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        imgs, vids = process_vision_info(msgs)
        inputs = proc(text=[text], images=imgs, videos=vids, padding=True, return_tensors='pt')
        inputs = {k: v.to(model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=8, do_sample=False)
        gen = out[0, inputs['input_ids'].shape[1]:]
        resp = proc.decode(gen, skip_special_tokens=True)
        choice = parse_choice(resp, a.k)
        gold_i = int(gold[i])
        in_list = gold_i in cols
        if in_list:
            in_short_n += 1
        if choice is not None:
            parsed += 1
            pick = cols[choice]
            vlm_pred[i] = pick
            if in_list and pick == gold_i:
                in_short_correct += 1
        if (j + 1) % 25 == 0 or j == 0:
            print(f'  {j+1}/{len(idxs)} [{time.time()-t0:.0f}s] '
                  f'resp={resp!r} choice={choice}', flush=True)

    vlm_top1 = acc(vlm_pred, subset)
    # conditional pick rate when gold in shortlist and parsed
    pick_rate = (100.0 * in_short_correct / in_short_n) if in_short_n else 0.0
    chance = 100.0 / a.k
    delta = vlm_top1 - base_top1
    promote = delta >= 2.0 and pick_rate > chance + 5.0

    results = {
        'n': len(idxs), 'k': a.k, 'seed': a.seed, 'model': a.model,
        'text_top1': base_top1,
        'gold_in_topk': in_k,
        'vlm_top1': vlm_top1,
        'delta': delta,
        'parsed': parsed,
        'in_shortlist_n': in_short_n,
        'in_shortlist_vlm_correct': in_short_correct,
        'in_shortlist_pick_rate': pick_rate,
        'chance_1_over_k': chance,
        'promote': promote,
        'verdict': (
            f'PROMOTE: delta={delta:+.2f}pt, pick_rate={pick_rate:.1f}% vs chance {chance:.1f}%'
            if promote else
            f'KILL: delta={delta:+.2f}pt (need >=+2), pick_rate={pick_rate:.1f}% vs chance {chance:.1f}%'
        ),
    }
    print('\n' + results['verdict'], flush=True)
    print(json.dumps({k: results[k] for k in results if k != 'verdict'}, indent=1), flush=True)
    p = os.path.join(OUT, f'unseen_vlm_rerank_n{a.n}_k{a.k}_results.json')
    json.dump(results, open(p, 'w'), indent=1)
    print('wrote', p, flush=True)


if __name__ == '__main__':
    main()
