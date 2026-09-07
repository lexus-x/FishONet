"""Proxy A/B: text-only vs + external image prototypes (loophole B).

Loads outputs/external_image_files.json (class -> local image paths), embeds with
BioCLIP-2.5-H, builds per-class mean prototypes, scores frozen-H / ctft queries.

  python research/unseen_ext_image_proxy.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import FishData, dbnorm, OUT  # noqa: E402

MODEL = 'hf-hub:imageomics/bioclip-2.5-vith14'


def top1(S, gold):
    return (S.argmax(1) == gold).float().mean().item() * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--files', default=os.path.join(OUT, 'external_image_files.json'))
    ap.add_argument('--cache', default=os.path.join(OUT, 'external_image_protos_h.pt'))
    a = ap.parse_args()

    D = FishData()
    meta = json.load(open(a.files))
    # embed external images
    if os.path.exists(a.cache):
        d = torch.load(a.cache, weights_only=False)
        protos = F.normalize(d['protos'].float(), dim=-1)
        covered = set(d['covered'])
        print(f'cache hit {a.cache} covered={len(covered)}', flush=True)
    else:
        import open_clip
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        model, _, pre = open_clip.create_model_and_transforms(MODEL)
        model = model.to(dev).eval()
        protos = torch.zeros(len(D.classes), 1024)
        cnt = torch.zeros(len(D.classes))
        covered = []
        t0 = time.time()
        with torch.no_grad():
            for j, c in enumerate(D.classes):
                paths = meta.get(c) or []
                if not paths:
                    continue
                feats = []
                for p in paths:
                    if not os.path.exists(p):
                        continue
                    try:
                        im = pre(Image.open(p).convert('RGB')).unsqueeze(0).to(dev)
                        with torch.autocast('cuda', dtype=torch.bfloat16):
                            f = model.encode_image(im).float()
                        feats.append(F.normalize(f, dim=-1).cpu()[0])
                    except Exception:
                        continue
                if not feats:
                    continue
                i = D.ci[c]
                protos[i] = F.normalize(torch.stack(feats).mean(0), dim=-1)
                cnt[i] = len(feats)
                covered.append(c)
                if len(covered) % 100 == 0:
                    print(f'  embedded {len(covered)} classes [{time.time()-t0:.0f}s]', flush=True)
        torch.save({'protos': protos, 'covered': covered, 'cnt': cnt, 'model': MODEL}, a.cache)
        print(f'wrote {a.cache} covered={len(covered)}', flush=True)
        protos = F.normalize(protos, dim=-1)

    # queries: frozen H on pseudo-unseen
    d = torch.load(os.path.join(OUT, 'emb_train_h.pt'), weights_only=False)
    idx = {fn: i for i, fn in enumerate(d['files'])}
    feats = F.normalize(d['feats'].float(), dim=-1)
    files, gold = [], []
    for c in D.pseudo:
        for fn in D.by[c]:
            if fn in idx:
                files.append(fn)
                gold.append(D.cand_pos[D.ci[c]])
    Q = torch.stack([feats[idx[fn]] for fn in files])
    gold = torch.tensor(gold)
    cand = D.cand

    # baselines
    St = dbnorm(Q @ D.TtH[cand].t())
    a_taxon = top1(St, gold)
    print(f'taxon text:           {a_taxon:.2f}', flush=True)

    # external protos on cand (missing -> -inf via zero vec that won't win after dbnorm? better mask)
    Pe = protos[cand]
    has = (Pe.norm(dim=-1) > 0.5)
    Se = Q @ Pe.t()
    Se = Se.masked_fill(~has.unsqueeze(0), -1e4)
    Se = dbnorm(Se)
    a_ext = top1(Se, gold)
    cov_cand = int(has.sum())
    print(f'external protos alone:{a_ext:.2f}  (cand covered {cov_cand}/{len(cand)})', flush=True)

    results = {'taxon': a_taxon, 'ext_alone': a_ext, 'cand_covered': cov_cand,
               'n_images': len(files), 'blends': {}}
    print('--- blends taxon + w * ext ---', flush=True)
    best = a_taxon
    for w in [0.25, 0.5, 1.0, 2.0]:
        acc = top1(St + w * Se, gold)
        results['blends'][str(w)] = acc
        best = max(best, acc)
        print(f'  taxon+{w:g}*ext: {acc:.2f} (delta {acc-a_taxon:+.2f})', flush=True)

    delta = best - a_taxon
    promote = delta >= 2.0 and cov_cand >= 0.3 * len(cand)
    results['best_delta'] = delta
    results['promote'] = promote
    results['verdict'] = (
        f'PROMOTE ext images: best_delta={delta:+.2f}' if promote
        else f'KILL ext images: best_delta={delta:+.2f} or sparse cover'
    )
    print(results['verdict'], flush=True)
    p = os.path.join(OUT, 'unseen_ext_image_proxy_results.json')
    json.dump(results, open(p, 'w'), indent=1)
    print('wrote', p, flush=True)


if __name__ == '__main__':
    main()
