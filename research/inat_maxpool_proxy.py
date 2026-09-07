"""Max-pool / top-m-mean iNat photo scoring vs mean prototype (v37 proxy baseline).

Builds (or loads) per-class photo embedding banks from downloaded iNat images,
then scores queries with max cosine sim over photos per class.

  python research/inat_maxpool_proxy.py --build-bank
  python research/inat_maxpool_proxy.py --bank outputs/inat_photo_bank_ctftshift.pt
  python research/inat_maxpool_proxy.py --enc b2 --build-bank --bank outputs/inat_photo_bank_bioclip2.pt
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import FishData, dbnorm, load_emb, OUT  # noqa: E402
from embed_inat import DS, inject_lora, set_scale, MODEL, DEV, MEAN, STD  # noqa: E402

MODEL_B2 = 'hf-hub:imageomics/bioclip-2'


def top1(S, gold):
    return (S.argmax(1) == gold).float().mean().item() * 100


def build_bank(files_path, ckpt, squash_tta, scale, bs, workers, out_path, max_photos):
    D = FishData()
    meta = json.load(open(files_path))
    items = []
    per_ci = {}
    for c, paths in meta.items():
        if c not in D.ci:
            continue
        ci = D.ci[c]
        ok = [p for p in (paths or []) if os.path.exists(p) and os.path.getsize(p) > 1000][:max_photos]
        if not ok:
            continue
        per_ci[ci] = ok
        for p in ok:
            items.append((ci, p))
    print(f'bank: {len(items)} images, {len(per_ci)} classes', flush=True)

    model, _, pp = open_clip.create_model_and_transforms(MODEL)
    ck = torch.load(ckpt, weights_only=False)
    inject_lora(model, ck['rank'], ck['alpha'], ck['top_k'])
    own = dict(model.named_parameters())
    for n, v in ck['state'].items():
        own[n].data.copy_(v)
    model = model.to(DEV).eval()
    set_scale(model, scale)

    dl = DataLoader(DS(items, pp, squash_tta), batch_size=bs, num_workers=workers, pin_memory=True)
    banks = {}
    t0 = time.time()
    with torch.no_grad():
        for bi, (cis, x, xsq) in enumerate(dl):
            x = x.to(DEV)
            xsq = xsq.to(DEV)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                f = F.normalize(model.encode_image(x).float(), dim=-1)
                f = f + F.normalize(model.encode_image(torch.flip(x, dims=[-1])).float(), dim=-1)
                if squash_tta:
                    f = f + F.normalize(model.encode_image(xsq).float(), dim=-1)
                    f = f + F.normalize(model.encode_image(torch.flip(xsq, dims=[-1])).float(), dim=-1)
                f = F.normalize(f, dim=-1).cpu()
            for j, ci in enumerate(cis.tolist()):
                if ci < 0:
                    continue
                banks.setdefault(ci, []).append(f[j])
            if bi % 50 == 0:
                print(f'  embed batch {bi} [{time.time()-t0:.0f}s]', flush=True)

    stacked = {}
    for ci, vecs in banks.items():
        stacked[ci] = torch.stack(vecs)
    torch.save({'bank': stacked, 'squash_tta': squash_tta, 'ckpt': ckpt, 'enc': 'ctft'}, out_path)
    print(f'wrote {out_path}: {len(stacked)} classes', flush=True)
    return stacked


def build_bank_b2(files_path, bs, workers, out_path, max_photos):
    """Per-class photo embeddings with frozen BioCLIP-2 (matches emb_train_bioclip2.pt)."""
    D = FishData()
    meta = json.load(open(files_path))
    items = []
    per_ci = {}
    for c, paths in meta.items():
        if c not in D.ci:
            continue
        ci = D.ci[c]
        ok = [p for p in (paths or []) if os.path.exists(p) and os.path.getsize(p) > 1000][:max_photos]
        if not ok:
            continue
        per_ci[ci] = ok
        for p in ok:
            items.append((ci, p))
    print(f'b2 bank: {len(items)} images, {len(per_ci)} classes', flush=True)

    class DSB2(Dataset):
        def __init__(self, items, pp):
            self.items = items
            self.pp = pp

        def __len__(self):
            return len(self.items)

        def __getitem__(self, i):
            ci, path = self.items[i]
            try:
                img = Image.open(path).convert('RGB')
                return ci, self.pp(img)
            except Exception:
                return -1, torch.zeros(3, 224, 224)

    model, _, pp = open_clip.create_model_and_transforms(MODEL_B2)
    model = model.to(DEV).eval()
    dl = DataLoader(DSB2(items, pp), batch_size=bs, num_workers=workers, pin_memory=True)
    banks = {}
    t0 = time.time()
    with torch.no_grad():
        for bi, (cis, x) in enumerate(dl):
            x = x.to(DEV)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                f = F.normalize(model.encode_image(x).float(), dim=-1).cpu()
            for j, ci in enumerate(cis.tolist()):
                if ci < 0:
                    continue
                banks.setdefault(ci, []).append(f[j])
            if bi % 50 == 0:
                print(f'  b2 embed batch {bi} [{time.time()-t0:.0f}s]', flush=True)

    stacked = {ci: torch.stack(vecs) for ci, vecs in banks.items()}
    torch.save({'bank': stacked, 'enc': 'bioclip2', 'model': MODEL_B2}, out_path)
    print(f'wrote {out_path}: {len(stacked)} classes', flush=True)
    return stacked


def score_maxpool(Q, bank, cand, topm: int):
    """Q [N,D], bank dict ci->[k,D], cand [C] class indices in full taxonomy."""
    N, D = Q.shape
    C = len(cand)
    S = torch.full((N, C), -1e4)
    cand_list = cand.tolist()
    for j, gidx in enumerate(cand_list):
        photos = bank.get(gidx)
        if photos is None or photos.numel() == 0:
            continue
        sim = Q @ photos.t()
        if topm <= 1:
            S[:, j] = sim.max(dim=1).values
        else:
            k = min(topm, sim.shape[1])
            S[:, j] = sim.topk(k, dim=1).values.mean(dim=1)
    return dbnorm(S)


def v36_stack(D, cand):
    def queries(idx, feats):
        q, y = [], []
        for c in D.pseudo:
            for fn in D.by[c]:
                if fn in idx:
                    q.append(feats[idx[fn]])
                    y.append(D.cand_pos[D.ci[c]])
        return torch.stack(q), torch.tensor(y)

    TtH_c = D.TtH[cand]
    TnL_c = D.TnL[cand]
    TTX = F.normalize(torch.load(os.path.join(OUT, 'text_emb_h_promptens.pt'),
                                 weights_only=False)['emb_taxctx'].float(), dim=-1)
    TTX_c = TTX[cand]
    legs = {}
    for tag in ['ctftshift', 'ftshift', 'fullft336_v2']:
        idx, feats, _ = load_emb(os.path.join(OUT, f'emb_train_{tag}.pt'))
        q, gold = queries(idx, feats)
        legs[tag] = q
    qL, gold = queries(D.LtI, D.LtF)
    S0 = (dbnorm(legs['ctftshift'] @ TtH_c.t()) + 0.5 * dbnorm(qL @ TnL_c.t())
          + 0.75 * dbnorm(legs['fullft336_v2'] @ TtH_c.t())
          + 1.0 * dbnorm(legs['ftshift'] @ TtH_c.t())
          + 1.0 * dbnorm(legs['ctftshift'] @ TTX_c.t()))
    return S0, gold, legs['ctftshift']


def mean_proto_baseline(cand, qenc):
    d = torch.load(os.path.join(OUT, 'inat_protos_ctftshift_full.pt'), weights_only=False)
    P = F.normalize(d['protos'].float(), dim=-1)
    Pc = P[cand]
    has = Pc.norm(dim=-1) > 0.5
    S = qenc @ Pc.t()
    S = S.masked_fill(~has.unsqueeze(0), -1e4)
    return dbnorm(S), has


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--enc', choices=['ctft', 'b2'], default='ctft',
                    help='ctft=LoRA ctftshift bank; b2=frozen BioCLIP-2 bank')
    ap.add_argument('--bank', default=None)
    ap.add_argument('--build-bank', action='store_true')
    ap.add_argument('--files', default=os.path.join(OUT, 'inat_image_files.json'))
    ap.add_argument('--ckpt', default='outputs/ctft_shift.pt')
    ap.add_argument('--squash_tta', type=int, default=1)
    ap.add_argument('--scale', type=float, default=0.4)
    ap.add_argument('--max-photos', type=int, default=24)
    ap.add_argument('--bs', type=int, default=96)
    ap.add_argument('--workers', type=int, default=6)
    a = ap.parse_args()

    if a.bank is None:
        a.bank = (os.path.join(OUT, 'inat_photo_bank_bioclip2.pt') if a.enc == 'b2'
                  else os.path.join(OUT, 'inat_photo_bank_ctftshift.pt'))

    if a.build_bank or not os.path.exists(a.bank):
        if a.enc == 'b2':
            bank = build_bank_b2(a.files, a.bs, a.workers, a.bank, a.max_photos)
        else:
            bank = build_bank(a.files, a.ckpt, a.squash_tta, a.scale, a.bs, a.workers, a.bank, a.max_photos)
    else:
        bank = torch.load(a.bank, weights_only=False)['bank']

    D = FishData()
    cand = D.cand
    S0, gold, qH = v36_stack(D, cand)
    base = top1(S0, gold)
    Smean, has = mean_proto_baseline(cand, qH)
    mean_at_w3 = top1(S0 + 3.0 * Smean, gold)
    print(f'v36 stack baseline={base:.2f}  mean-proto@w3={mean_at_w3:.2f} (ref 43.53)', flush=True)

    res = {'baseline': base, 'mean_proto_w3': mean_at_w3, 'modes': {}}
    best = mean_at_w3
    best_cfg = ('mean', 3, 1)

    for topm in (1, 2, 3, 4):
        Smp = score_maxpool(qH, bank, cand, topm=topm)
        alone = top1(Smp, gold)
        res['modes'][f'maxpool_top{topm}_alone'] = alone
        print(f'maxpool topm={topm} alone={alone:.2f}', flush=True)
        for w in (2.0, 2.5, 3.0, 3.5):
            acc = top1(S0 + w * Smp, gold)
            key = f'maxpool_top{topm}_w{w:g}'
            res['modes'][key] = acc
            print(f'  S0+{w:g}*maxpool_top{topm}: {acc:.2f} ({acc-mean_at_w3:+.2f} vs mean@3)', flush=True)
            if acc > best:
                best, best_cfg = acc, ('maxpool', w, topm)

    rng = random.Random(0)
    cov_classes = [c for c in D.pseudo if has[D.cand_pos[D.ci[c]]]]
    rng.shuffle(cov_classes)
    half = len(cov_classes) // 2
    tune_cls, val_cls = set(cov_classes[:half]), set(cov_classes[half:])
    cand_list = D.cand.tolist()
    q_cls = [D.classes[cand_list[g]] for g in gold.tolist()]
    val_m = torch.tensor([c in val_cls for c in q_cls])

    Smp3 = score_maxpool(qH, bank, cand, topm=3)
    best_w, best_tune = 3.0, top1((S0 + 3.0 * Smean)[~val_m], gold[~val_m])
    for w in (2.0, 2.5, 3.0, 3.5, 4.0):
        for Sm in (Smean, Smp3):
            acc = top1((S0 + w * Sm)[~val_m], gold[~val_m])
            if acc > best_tune:
                best_tune, best_w = acc, w
    val_mean = top1((S0 + 3.0 * Smean)[val_m], gold[val_m])
    val_mp = top1((S0 + best_w * Smp3)[val_m], gold[val_m])
    res['cv'] = {'val_mean_w3': val_mean, 'val_maxpool_top3_w': best_w, 'val_maxpool': val_mp,
                 'val_delta': val_mp - val_mean}
    print(f'CV val mean@3={val_mean:.2f} maxpool@w={best_w}={val_mp:.2f} ({val_mp-val_mean:+.2f})', flush=True)

    res['best'] = best
    res['best_cfg'] = best_cfg
    res['delta_vs_mean_w3'] = best - mean_at_w3
    out = os.path.join(OUT, 'inat_maxpool_proxy_results.json')
    json.dump(res, open(out, 'w'), indent=1)
    print(f'BEST {best:.2f} delta_vs_mean_w3={best-mean_at_w3:+.2f}; wrote {out}', flush=True)


if __name__ == '__main__':
    main()
