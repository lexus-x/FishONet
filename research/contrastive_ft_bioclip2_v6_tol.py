"""B2 LoRA v6: taxctx CE + attract image emb toward TreeOfLife B2 class mean protos.

Warm-start from v2. Extra loss pulls train/iNat views toward disclosed ToL visual
centroids (outputs/tol_protos_bioclip2.pt) — no new images required.

  conda activate onet
  python research/contrastive_ft_bioclip2_v6_tol.py --shift_aug 1 --inat 1 --max_steps 2000 \\
    --init_ckpt outputs/b2_shift_lora_v2.pt --save outputs/b2_shift_lora_v6_tol.pt \\
    2>&1 | tee outputs/b2_shift_lora_v6_tol_train.log
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import defaultdict

import open_clip
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
from contrastive_ft import inject_lora, index_images  # noqa: E402

MODEL = 'hf-hub:imageomics/bioclip-2'
DEV = 'cuda'
OUT = os.path.join(ROOT, 'outputs')


def lora_sd(m):
    return {n: p.detach().cpu() for n, p in m.named_parameters() if n.endswith('.A') or n.endswith('.B')}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--top_k', type=int, default=12)
    ap.add_argument('--rank', type=int, default=16)
    ap.add_argument('--alpha', type=int, default=32)
    ap.add_argument('--bs', type=int, default=96)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--max_steps', type=int, default=2000)
    ap.add_argument('--val_every', type=int, default=100)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--temp', type=float, default=30.0)
    ap.add_argument('--tol_w', type=float, default=0.5, help='weight on ToL proto attraction')
    ap.add_argument('--val_imgs', type=int, default=2318)
    ap.add_argument('--text_emb', default=os.path.join(OUT, 'text_emb_b2_taxctx.pt'))
    ap.add_argument('--tol_proto', default=os.path.join(OUT, 'tol_protos_bioclip2.pt'))
    ap.add_argument('--inat', type=int, default=1)
    ap.add_argument('--inat_cap', type=int, default=8)
    ap.add_argument('--inat_files', default=os.path.join(OUT, 'inat_image_files.json'))
    ap.add_argument('--init_ckpt', default='outputs/b2_shift_lora_v2.pt')
    ap.add_argument('--shift_aug', type=int, default=1)
    ap.add_argument('--save', default='outputs/b2_shift_lora_v6_tol.pt')
    a = ap.parse_args()

    if a.init_ckpt and os.path.isfile(a.init_ckpt):
        _ic = torch.load(a.init_ckpt, weights_only=False)
        a.rank = int(_ic.get('rank', a.rank))
        a.alpha = int(_ic.get('alpha', a.alpha))
        a.top_k = int(_ic.get('top_k', a.top_k))
        print(f'LoRA hparams from init: rank={a.rank} alpha={a.alpha} top_k={a.top_k}', flush=True)

    D = os.path.join(ROOT, 'data', 'dl')
    import pickle
    classes = list(pickle.load(open(f'{D}/all_classes.pkl', 'rb')))
    ci = {c: i for i, c in enumerate(classes)}
    lab = json.load(open(f'{D}/label_train.json'))
    imgidx = index_images(f'{D}/images')
    te = torch.load(a.text_emb, weights_only=False)
    key = 'emb_taxctx' if 'emb_taxctx' in te else 'emb_taxon'
    Tall = F.normalize(te[key].float(), dim=-1)
    Tol = F.normalize(torch.load(a.tol_proto, weights_only=False)['protos'].float(), dim=-1)
    has_tol = Tol.norm(dim=-1) > 0.5
    print(f'text={a.text_emb} tol_cov={int(has_tol.sum())}/{len(classes)} tol_w={a.tol_w}', flush=True)

    tr = torch.load(os.path.join(OUT, 'emb_train_bioclip2.pt'), weights_only=False)
    by = defaultdict(list)
    for fn in tr['files']:
        if fn in lab and lab[fn] in ci and fn in imgidx:
            by[lab[fn]].append(fn)
    seen = sorted(by.keys())
    order = sorted(seen, key=lambda c: len(by[c]))
    pseudo = set(order[: int(len(seen) * 0.2)])
    known = [c for c in seen if c not in pseudo]
    known_set = set(known)

    train_items = []
    for c in known:
        for fn in by[c]:
            train_items.append((fn, False, ci[c]))

    safe_cand = [c for c in classes if c not in known_set and c not in pseudo]
    if a.inat and os.path.isfile(a.inat_files):
        inat_map = json.load(open(a.inat_files))
        for c in known:
            if c in inat_map:
                for p in inat_map[c][: a.inat_cap]:
                    if os.path.isfile(p):
                        train_items.append((p, True, ci[c]))
        for c in safe_cand:
            if c not in inat_map:
                continue
            for p in inat_map[c][: a.inat_cap]:
                if os.path.isfile(p):
                    train_items.append((p, True, ci[c]))
        print(f'iNat mix ON cap={a.inat_cap} train_items={len(train_items)}', flush=True)

    train_classes = sorted({gci for _, _, gci in train_items})
    cls2loc = {g: i for i, g in enumerate(train_classes)}
    Ttrain = Tall[torch.tensor(train_classes)].to(DEV)
    Tol_train = Tol[torch.tensor(train_classes)].to(DEV)
    tol_mask = has_tol[torch.tensor(train_classes)].to(DEV)
    train_y = [(path, ab, cls2loc[gci]) for path, ab, gci in train_items if gci in cls2loc]

    cand = [i for i, c in enumerate(classes) if c not in known_set]
    cand_pos = {cii: j for j, cii in enumerate(cand)}
    Tcand = Tall[torch.tensor(cand)]

    val_items = [(fn, False, cand_pos[ci[c]]) for c in pseudo for fn in by[c]]
    random.seed(0)
    random.shuffle(val_items)
    val_items = val_items[: a.val_imgs]

    print(
        f'model={MODEL} known={len(known)} pseudo={len(pseudo)} '
        f'train={len(train_y)} train_classes={len(train_classes)} val={len(val_items)}',
        flush=True,
    )

    model, _, pp = open_clip.create_model_and_transforms(MODEL)
    inject_lora(model, a.rank, a.alpha, a.top_k)
    if a.init_ckpt and os.path.isfile(a.init_ckpt):
        ck = torch.load(a.init_ckpt, weights_only=False)
        own = dict(model.named_parameters())
        for n, v in ck['state'].items():
            if n in own:
                own[n].data.copy_(v)
        print(f'init LoRA from {a.init_ckpt} db={ck.get("db")}', flush=True)
    model = model.to(DEV)
    for n, p in model.named_parameters():
        p.requires_grad_(n.endswith('.A') or n.endswith('.B'))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)

    mean = (0.48145466, 0.4578275, 0.40821073)
    std = (0.26862954, 0.26130258, 0.27577711)
    eval_pp = pp
    if a.shift_aug:
        train_tf = T.Compose([
            T.RandomResizedCrop(224, scale=(0.35, 1.0), ratio=(0.5, 2.0),
                                interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(), T.ColorJitter(0.2, 0.2, 0.2),
            T.ToTensor(), T.Normalize(mean, std),
        ])
        print('shift_aug ON', flush=True)
    else:
        train_tf = eval_pp

    class DS(Dataset):
        def __init__(s, it, tf, imgidx):
            s.it = it
            s.tf = tf
            s.imgidx = imgidx

        def __len__(s):
            return len(s.it)

        def __getitem__(s, i):
            key, ab, y = s.it[i]
            try:
                path = key if ab else s.imgidx[key]
                return s.tf(Image.open(path).convert('RGB')), y
            except Exception:
                return torch.zeros(3, 224, 224), y

    dl = DataLoader(DS(train_y, train_tf, imgidx), batch_size=a.bs, shuffle=True,
                    num_workers=a.workers, pin_memory=True, drop_last=True)
    vdl = DataLoader(DS(val_items, eval_pp, imgidx), batch_size=256,
                     num_workers=a.workers, pin_memory=True)

    def db(M):
        return (M - M.mean(0, keepdim=True)) / (M.std(0, keepdim=True) + 1e-6)

    @torch.no_grad()
    def validate():
        model.eval()
        feats, ys = [], []
        for x, y in vdl:
            x = x.to(DEV)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                f = F.normalize(model.encode_image(x).float(), dim=-1)
            feats.append(f.cpu())
            ys.append(y)
        Fm = torch.cat(feats)
        Y = torch.cat(ys)
        S = Fm @ Tcand.t()
        model.train()
        raw = round((S.argmax(1) == Y).float().mean().item() * 100, 2)
        d = round((db(S).argmax(1) == Y).float().mean().item() * 100, 2)
        return raw, d

    model.train()
    base = validate()
    print(f'BASELINE raw,db={base}', flush=True)
    warmup = max(1, int(0.08 * a.max_steps))

    def lr_at(s):
        if s < warmup:
            return a.lr * s / warmup
        p = (s - warmup) / max(1, a.max_steps - warmup)
        return a.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))

    step = 0
    t0 = time.time()
    best_db = -1.0
    done = False
    while not done:
        for x, y in dl:
            x = x.to(DEV)
            y = y.to(DEV)
            for g in opt.param_groups:
                g['lr'] = lr_at(step)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                f = F.normalize(model.encode_image(x).float(), dim=-1)
                loss_ce = F.cross_entropy(a.temp * (f @ Ttrain.t()), y)
                # attract toward ToL class mean when available
                tgt = Tol_train[y]
                m = tol_mask[y]
                if m.any() and a.tol_w > 0:
                    cos = (f[m] * tgt[m]).sum(dim=-1)
                    loss_tol = (1.0 - cos).mean()
                    loss = loss_ce + a.tol_w * loss_tol
                else:
                    loss_tol = torch.zeros((), device=DEV)
                    loss = loss_ce
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            if step % 50 == 0:
                print(
                    f'step {step}/{a.max_steps} loss={loss.item():.3f} '
                    f'ce={loss_ce.item():.3f} tol={float(loss_tol):.3f} '
                    f'{(time.time() - t0) / step:.2f}s/it',
                    flush=True,
                )
            if step % a.val_every == 0:
                r, d = validate()
                print(f'  VAL @ {step}: raw,db=({r}, {d})', flush=True)
                if d > best_db:
                    best_db = d
                    torch.save({
                        'state': lora_sd(model), 'top_k': a.top_k, 'rank': a.rank,
                        'alpha': a.alpha, 'db': d, 'step': step, 'model': MODEL,
                        'text_emb': a.text_emb, 'init': a.init_ckpt, 'inat': a.inat,
                        'tol_w': a.tol_w, 'tol_proto': a.tol_proto,
                    }, a.save)
                    print(f'    saved best db={d}', flush=True)
            if step >= a.max_steps:
                done = True
                break
    r, d = validate()
    print(f'FINAL raw,db=({r}, {d}) best_db={best_db}', flush=True)
    if best_db < 16.0:
        print('NO_EXTRACT: best_db below 16.0 gate (need beat v1 15.53 toward stack lift)', flush=True)


if __name__ == '__main__':
    main()
