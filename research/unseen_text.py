"""Text-source ranking on the pseudo-unseen proxy (HANDOFF §8 item 2).

Question: can BioCLIP-native short trait text lift zero-shot over bare taxon?
Provided raw emb_desc is dead (12.25). This tests LLM-compressed traits from the
SAME descriptions, alone and blended with taxon (mirror taxctx — do not replace).

Apples-to-apples: frozen BioCLIP-2.5-H image feats, same pseudo-unseen queries /
candidates / DBNorm. Absolute numbers overstate real (HANDOFF §7).

  python research/unseen_text.py
  python research/unseen_text.py --traits_key traits_external   # after FishBase merge
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import FishData, dbnorm, OUT, DATA  # noqa: E402

MODEL = 'hf-hub:imageomics/bioclip-2.5-vith14'


def top1(S, gold):
    return (S.argmax(1) == gold).float().mean().item() * 100


def trait_prompts(classes, traits: dict) -> list[str]:
    out = []
    for c in classes:
        t = (traits.get(c) or '').strip()
        if t:
            out.append(f'a photo of {c}. {t}.')
        else:
            out.append(f'a photo of {c}.')
    return out


def encode_texts(texts, cache_path, bs=256):
    if os.path.exists(cache_path):
        d = torch.load(cache_path, weights_only=False)
        if d.get('texts') == texts:
            print(f'cache hit: {cache_path}', flush=True)
            return F.normalize(d['T'].float(), dim=-1)
    import open_clip
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, _, _ = open_clip.create_model_and_transforms(MODEL)
    tok = open_clip.get_tokenizer(MODEL)
    model = model.to(dev).eval()
    out = []
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            with torch.autocast('cuda', dtype=torch.bfloat16):
                f = model.encode_text(tok(texts[i:i + bs]).to(dev)).float()
            out.append(F.normalize(f, dim=-1).cpu())
            if i % (bs * 20) == 0:
                print(f'  encode {i}/{len(texts)} [{time.time()-t0:.0f}s]', flush=True)
    T = torch.cat(out)
    torch.save({'T': T, 'texts': texts, 'model': MODEL}, cache_path)
    print(f'wrote {cache_path}', flush=True)
    return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--traits', default=os.path.join(OUT, 'traits_from_desc.json'))
    ap.add_argument('--tag', default='traits_provided')
    ap.add_argument('--ws', default='0,0.25,0.5,1.0',
                    help='blend weights for taxon + w*traits')
    a = ap.parse_args()

    D = FishData()
    classes = list(pickle.load(open(os.path.join(DATA, 'dl', 'all_classes.pkl'), 'rb')))
    assert classes == D.classes

    files, gold = [], []
    for c in D.pseudo:
        for fn in D.by[c]:
            files.append(fn)
            gold.append(D.cand_pos[D.ci[c]])
    gold = torch.tensor(gold)

    # frozen-H image queries
    d = torch.load(os.path.join(OUT, 'emb_train_h.pt'), weights_only=False)
    idx = {fn: i for i, fn in enumerate(d['files'])}
    feats = F.normalize(d['feats'].float(), dim=-1)
    keep = [i for i, fn in enumerate(files) if fn in idx]
    Q = torch.stack([feats[idx[files[i]]] for i in keep])
    gold = gold[keep]
    print(f'pseudo-unseen: {len(D.pseudo)} classes, {len(keep)} images, '
          f'{len(D.cand)} candidates', flush=True)

    cand = D.cand
    # --- baselines from cached text ---
    Tt = D.TtH[cand]
    Td = D.TdH[cand] if D.TdH is not None else None
    St = dbnorm(Q @ Tt.t())
    a_taxon = top1(St, gold)
    print(f'\ntaxon (bare)     : {a_taxon:.2f}  (ref alone_h ~23.38)', flush=True)
    a_desc = None
    if Td is not None:
        a_desc = top1(dbnorm(Q @ Td.t()), gold)
        print(f'raw emb_desc     : {a_desc:.2f}  (ref ~12.25 on deployed path)', flush=True)

    # taxctx for reference
    pe = torch.load(os.path.join(OUT, 'text_emb_h_promptens.pt'), weights_only=False)
    Ttx = F.normalize(pe['emb_taxctx'].float(), dim=-1)[cand]
    a_taxctx = top1(dbnorm(Q @ Ttx.t()), gold)
    print(f'taxctx           : {a_taxctx:.2f}', flush=True)

    traits = json.load(open(a.traits))
    n_hit = sum(1 for c in classes if traits.get(c))
    print(f'traits file: {a.traits}  coverage {n_hit}/{len(classes)}', flush=True)
    prompts = trait_prompts(classes, traits)
    cache = os.path.join(OUT, f'text_emb_h_{a.tag}.pt')
    # store full-space emb for builders; score on cand only
    if os.path.exists(cache):
        dT = torch.load(cache, weights_only=False)
        if dT.get('prompts') == prompts and 'emb_traits' in dT:
            Ttr_all = F.normalize(dT['emb_traits'].float(), dim=-1)
            print(f'cache hit emb: {cache}', flush=True)
        else:
            Ttr_all = encode_texts(prompts, os.path.join(OUT, f'_tmp_{a.tag}_enc.pt'))
            torch.save({'classes': classes, 'emb_traits': Ttr_all, 'prompts': prompts,
                        'model': MODEL, 'tag': a.tag}, cache)
            print(f'wrote {cache}', flush=True)
    else:
        Ttr_all = encode_texts(prompts, os.path.join(OUT, f'_tmp_{a.tag}_enc.pt'))
        torch.save({'classes': classes, 'emb_traits': Ttr_all, 'prompts': prompts,
                    'model': MODEL, 'tag': a.tag}, cache)
        print(f'wrote {cache}', flush=True)

    Ttr = Ttr_all[cand]
    Str = dbnorm(Q @ Ttr.t())
    a_traits = top1(Str, gold)
    print(f'{a.tag:16}: {a_traits:.2f}  (delta vs taxon {a_traits-a_taxon:+.2f})', flush=True)

    results = {
        'taxon': a_taxon,
        'taxctx': a_taxctx,
        'raw_desc': a_desc,
        a.tag: a_traits,
        'n_images': len(keep),
        'n_cand': int(len(cand)),
        'traits_path': a.traits,
        'tag': a.tag,
        'blends': {},
    }
    print('--- blends: dbnorm(taxon) + w * dbnorm(traits) ---', flush=True)
    for w in [float(x) for x in a.ws.split(',')]:
        if w == 0:
            continue
        acc = top1(St + w * Str, gold)
        results['blends'][str(w)] = acc
        print(f'  taxon + {w:g}*traits : {acc:.2f}  (delta {acc-a_taxon:+.2f})', flush=True)

    # kill criteria (plan)
    best_blend = max(results['blends'].values()) if results['blends'] else a_traits
    clear_margin = 0.5  # pts; ignore ~0.1 noise
    verdict = {
        'traits_beat_raw_desc': (a_desc is None) or (a_traits > a_desc + 2.0),
        'traits_approach_taxon': a_traits >= a_taxon - 5.0,
        'blend_beats_taxon': best_blend > a_taxon + clear_margin,
        'best_blend': best_blend,
        'best_delta': best_blend - a_taxon,
    }
    results['verdict'] = verdict
    go = verdict['blend_beats_taxon']
    results['promote'] = bool(go)
    print('\nVERDICT:', json.dumps(verdict, indent=1), flush=True)
    print('PROMOTE to external traits / submission gate:', go, flush=True)

    p = os.path.join(OUT, f'unseen_text_{a.tag}_results.json')
    json.dump(results, open(p, 'w'), indent=1)
    print('wrote', p, flush=True)


if __name__ == '__main__':
    main()
