"""VLM top-K rerank on the pseudo-unseen proxy — DeepSeek vision variant.

Same harness/candidates/sample as research/unseen_vlm_rerank.py (Qwen2.5-VL-7B,
KILLED: delta -16.00pt, pick_rate 16.8% ~ chance). Two things differ, both
free to test since deepseek-v4-flash-vision-exp has no price premium over
text-only V4-Flash (launched 2026-08-21):
  1) a frontier-tier closed VLM instead of a 7B open one
  2) a few hundred tokens of reasoning room before the final answer, instead
     of an 8-token forced greedy guess

Compliance: no splits/*.pkl membership; same recipe for all; disclose VLM
(general-purpose VLM, allowed under COMPETITION_RULES.md SS3.1).

  DEEPSEEK_API_KEY=... python research/unseen_vlm_rerank_deepseek.py --n 300 --k 10
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import random
import re
import sys
import time

import requests
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import FishData, dbnorm, OUT, DATA  # noqa: E402

IMG = os.path.join(DATA, 'dl', 'images')
API_URL = 'https://api.deepseek.com/chat/completions'
MODEL = 'deepseek-v4-flash-vision-exp'


def parse_choice(text: str, k: int) -> int | None:
    m = re.search(r'ANSWER:\s*([1-9][0-9]*)', text, re.IGNORECASE)
    if not m:
        nums = re.findall(r'\b([1-9][0-9]*)\b', text)
        m = nums[-1] if nums else None
        if m is None:
            return None
        v = int(m)
    else:
        v = int(m.group(1))
    return v - 1 if 1 <= v <= k else None


def ask_deepseek(api_key: str, path: str, prompt: str) -> str:
    with open(path, 'rb') as f:
        b64 = base64.b64encode(f.read()).decode('ascii')
    body = {
        'model': MODEL,
        'messages': [{'role': 'user', 'content': [
            {'type': 'text', 'text': prompt},
            {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{b64}'}},
        ]}],
        'max_tokens': 900,
        'temperature': 0,
    }
    r = requests.post(
        API_URL,
        headers={'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'},
        json=body, timeout=60,
    )
    r.raise_for_status()
    msg = r.json()['choices'][0]['message']
    # reasoning model: final answer lands in `content`, but on `finish_reason
    # == length` (ran out of budget mid-thought) content is empty and only
    # reasoning_content has text -- fall back to that.
    return msg.get('content') or msg.get('reasoning_content') or ''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=300, help='subsample size')
    ap.add_argument('--k', type=int, default=10)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()

    api_key = os.environ.get('DEEPSEEK_API_KEY')
    if not api_key:
        sys.exit('set DEEPSEEK_API_KEY')

    D = FishData()
    desc = json.load(open(os.path.join(DATA, 'dl', 'descriptions.json')))
    cn = json.load(open(os.path.join(OUT, 'common_names.json')))

    # same deployed-ish scores as the Qwen run: ctftbig @ taxon + 0.5 L @ name
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
    topk = S.topk(a.k, dim=1).indices  # [N,K] into cand columns, same order as Qwen run

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
    print(f'subsample n={len(idxs)} K={a.k} model={MODEL}', flush=True)
    print(f'text baseline top-1: {base_top1:.2f}', flush=True)
    print(f'gold in top-K:       {in_k:.2f}  (oracle ceiling if perfect VLM)', flush=True)

    def cand_label(sp: str) -> str:
        name = cn.get(sp) or sp
        d = desc.get(sp, '')
        d = (d[:180] + '…') if len(d) > 180 else d
        return f'{name} ({sp}): {d}'

    vlm_pred = base_pred.clone()
    parsed = 0
    in_short_correct = 0
    in_short_n = 0
    errors = 0
    t0 = time.time()

    for j, i in enumerate(idxs):
        fn = files[i]
        path = os.path.join(IMG, fn)
        if not os.path.exists(path):
            continue
        cols = topk[i].tolist()
        names = [D.classes[int(cand[c])] for c in cols]
        listing = '\n'.join(f'{t + 1}. {cand_label(nm)}' for t, nm in enumerate(names))
        prompt = (
            f'You are identifying a fish species from a photo. '
            f'Exactly one of the {a.k} candidates below is correct.\n\n{listing}\n\n'
            f'Briefly reason about distinguishing visual features (fin shape, body form, '
            f'coloration, markings) that match the photo -- keep it under 100 words -- then '
            f'on the FINAL line reply with exactly:\nANSWER: <number>'
        )
        gold_i = int(gold[i])
        in_list = gold_i in cols
        if in_list:
            in_short_n += 1
        try:
            resp = ask_deepseek(api_key, path, prompt)
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f'  [{j+1}] API error: {e}', flush=True)
            continue
        choice = parse_choice(resp, a.k)
        if choice is not None:
            parsed += 1
            pick = cols[choice]
            vlm_pred[i] = pick
            if in_list and pick == gold_i:
                in_short_correct += 1
        if (j + 1) % 25 == 0 or j == 0:
            print(f'  {j+1}/{len(idxs)} [{time.time()-t0:.0f}s] '
                  f'tail={resp[-60:]!r} choice={choice}', flush=True)

    vlm_top1 = acc(vlm_pred, subset)
    pick_rate = (100.0 * in_short_correct / in_short_n) if in_short_n else 0.0
    chance = 100.0 / a.k
    delta = vlm_top1 - base_top1
    promote = delta >= 2.0 and pick_rate > chance + 5.0

    results = {
        'n': len(idxs), 'k': a.k, 'seed': a.seed, 'model': MODEL,
        'text_top1': base_top1,
        'gold_in_topk': in_k,
        'vlm_top1': vlm_top1,
        'delta': delta,
        'parsed': parsed,
        'errors': errors,
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
    p = os.path.join(OUT, f'unseen_vlm_rerank_deepseek_n{a.n}_k{a.k}_results.json')
    json.dump(results, open(p, 'w'), indent=1)
    print('wrote', p, flush=True)


if __name__ == '__main__':
    main()
