"""BioCLIP-2 LoRA with hard-negative contrastive FT (taxctx text + optional iNat mix).

Mines hard text negatives: top-k global wrong classes + same-family confusers.
Warm-start from v1 LoRA (outputs/b2_shift_lora.pt, db 15.53).

  conda activate onet
  python research/contrastive_ft_bioclip2_hardneg.py --shift_aug 1 --max_steps 1800 \\
      --init_ckpt outputs/b2_shift_lora.pt --inat 1 \\
      --save outputs/b2_shift_lora_hardneg.pt 2>&1 | tee outputs/b2_shift_lora_hardneg_train.log
"""
import argparse
import json
import math
import os
import pickle
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


def build_family_map(classes, train_class_indices):
    tax = json.load(open(os.path.join(OUT, 'taxonomy_full.json')))
    fam_to_loc = defaultdict(list)
    for loc, gci in enumerate(train_class_indices):
        fam = (tax.get(classes[gci]) or {}).get('family') or ''
        if fam:
            fam_to_loc[fam].append(loc)
    return fam_to_loc


def hard_neg_loss(logits, y, hard_idx, temp, weight):
    """InfoNCE-style pull on positive vs mined hard text negatives only."""
    if weight <= 0 or hard_idx.numel() == 0:
        return logits.new_zeros(())
    B = logits.size(0)
    pos = (temp * logits.gather(1, y.unsqueeze(1))).squeeze(1)
    losses = []
    for i in range(B):
        idx = hard_idx[i]
        idx = idx[idx >= 0]
        if idx.numel() == 0:
            continue
        neg = temp * logits[i, idx]
        losses.append(-(pos[i] - torch.logaddexp(pos[i], torch.logsumexp(neg, dim=0))))
    if not losses:
        return logits.new_zeros(())
    return weight * torch.stack(losses).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--top_k', type=int, default=12)
    ap.add_argument('--rank', type=int, default=16)
    ap.add_argument('--alpha', type=int, default=32)
    ap.add_argument('--bs', type=int, default=96)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--max_steps', type=int, default=1800)
    ap.add_argument('--val_every', type=int, default=100)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--temp', type=float, default=30.0)
    ap.add_argument('--val_imgs', type=int, default=2318)
    ap.add_argument('--text_emb', default=os.path.join(OUT, 'text_emb_b2_taxctx.pt'))
    ap.add_argument('--inat', type=int, default=0)
    ap.add_argument('--inat_cap', type=int, default=8)
    ap.add_argument('--inat_files', default=os.path.join(OUT, 'inat_image_files.json'))
    ap.add_argument('--init_ckpt', default='outputs/b2_shift_lora.pt')
    ap.add_argument('--shift_aug', type=int, default=1)
    ap.add_argument('--save', default='outputs/b2_shift_lora_hardneg.pt')
    ap.add_argument('--hard_k', type=int, default=32, help='top wrong text classes per sample')
    ap.add_argument('--family_k', type=int, default=8, help='extra same-family hard negatives')
    ap.add_argument('--hard_weight', type=float, default=0.35)
    ap.add_argument('--bank_size', type=int, default=4096, help='FIFO memory bank for image hard negs')
    ap.add_argument('--bank_weight', type=float, default=0.15)
    a = ap.parse_args()

    if a.init_ckpt and os.path.isfile(a.init_ckpt):
        _ic = torch.load(a.init_ckpt, weights_only=False)
        a.rank = int(_ic.get('rank', a.rank))
        a.alpha = int(_ic.get('alpha', a.alpha))
        a.top_k = int(_ic.get('top_k', a.top_k))
        print(f'LoRA hparams from init: rank={a.rank} alpha={a.alpha} top_k={a.top_k}', flush=True)

    D = os.path.join(ROOT, 'data', 'dl')
    classes = list(pickle.load(open(f'{D}/all_classes.pkl', 'rb')))
    ci = {c: i for i, c in enumerate(classes)}
    lab = json.load(open(f'{D}/label_train.json'))
    imgidx = index_images(f'{D}/images')
    te = torch.load(a.text_emb, weights_only=False)
    key = 'emb_taxctx' if 'emb_taxctx' in te else 'emb_taxon'
    Tall = F.normalize(te[key].float(), dim=-1)
    print(f'text={a.text_emb} key={key} dim={Tall.shape[1]}', flush=True)

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
    train_y = [(path, ab, cls2loc[gci]) for path, ab, gci in train_items if gci in cls2loc]
    fam_to_loc = build_family_map(classes, train_classes)
    tax_cache = json.load(open(os.path.join(OUT, 'taxonomy_full.json')))

    cand = [i for i, c in enumerate(classes) if c not in known_set]
    cand_pos = {cii: j for j, cii in enumerate(cand)}
    Tcand = Tall[torch.tensor(cand)]

    val_items = [(fn, False, cand_pos[ci[c]]) for c in pseudo for fn in by[c]]
    random.seed(0)
    random.shuffle(val_items)
    val_items = val_items[: a.val_imgs]

    print(
        f'model={MODEL} known={len(known)} pseudo={len(pseudo)} '
        f'train={len(train_y)} train_classes={len(train_classes)} val={len(val_items)} '
        f'hard_k={a.hard_k} family_k={a.family_k} hard_w={a.hard_weight}',
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

    def family_hard_idx(logits, y):
        B, C = logits.shape
        device = logits.device
        hard_lists = []
        for i in range(B):
            yi = y[i].item()
            wrong = logits[i].clone()
            wrong[yi] = -1e9
            top = wrong.topk(min(a.hard_k, C - 1)).indices.tolist()
            gci = train_classes[yi]
            fam = (tax_cache.get(classes[gci]) or {}).get('family')
            extra = []
            if fam and fam in fam_to_loc:
                for loc in fam_to_loc[fam]:
                    if loc != yi and loc not in top:
                        extra.append(loc)
                extra.sort(key=lambda loc: wrong[loc].item(), reverse=True)
                extra = extra[: a.family_k]
            merged = list(dict.fromkeys(top + extra))
            hard_lists.append(merged)
        max_len = max(len(h) for h in hard_lists)
        pad = torch.full((B, max_len), -1, dtype=torch.long, device=device)
        for i, h in enumerate(hard_lists):
            pad[i, : len(h)] = torch.tensor(h, device=device)
        return pad

    bank_f = torch.zeros(a.bank_size, Ttrain.size(1), device=DEV)
    bank_y = torch.full((a.bank_size,), -1, dtype=torch.long, device=DEV)
    bank_ptr = 0
    bank_filled = 0

    def bank_hard_loss(f, y):
        nonlocal bank_ptr, bank_filled
        if a.bank_weight <= 0 or a.bank_size <= 0:
            return f.new_zeros(())
        with torch.no_grad():
            n = f.size(0)
            end = bank_ptr + n
            if end <= a.bank_size:
                bank_f[bank_ptr:end] = f.detach()
                bank_y[bank_ptr:end] = y.detach()
            else:
                part = a.bank_size - bank_ptr
                bank_f[bank_ptr:] = f[:part].detach()
                bank_y[bank_ptr:] = y[:part].detach()
                bank_f[: n - part] = f[part:].detach()
                bank_y[: n - part] = y[part:].detach()
            bank_ptr = (bank_ptr + n) % a.bank_size
            bank_filled = min(a.bank_size, bank_filled + n)
            if bank_filled < 64:
                return f.new_zeros(())
            valid = bank_y >= 0
            bf = bank_f[valid]
            by_ = bank_y[valid]
        sim = f @ bf.t()
        losses = []
        for i in range(f.size(0)):
            mask = by_ != y[i]
            if not mask.any():
                continue
            s = sim[i].masked_fill(~mask, -1e9)
            hard = s.topk(min(16, int(mask.sum()))).values
            pos = (a.temp * (f[i] @ Ttrain[y[i]])).squeeze()
            losses.append(-(pos - torch.logaddexp(pos, torch.logsumexp(a.temp * hard, dim=0))))
        if not losses:
            return f.new_zeros(())
        return a.bank_weight * torch.stack(losses).mean()

    model.train()
    base = validate()
    print(f'BASELINE raw,db={base} (frozen B2 vs taxctx text)', flush=True)
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
                logits = f @ Ttrain.t()
                loss_ce = F.cross_entropy(a.temp * logits, y)
                hard_idx = family_hard_idx(logits.detach(), y)
                loss_hard = hard_neg_loss(logits, y, hard_idx, a.temp, a.hard_weight)
                loss_bank = bank_hard_loss(f, y)
                loss = loss_ce + loss_hard + loss_bank
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            if step % 50 == 0:
                print(
                    f'step {step}/{a.max_steps} loss={loss.item():.3f} '
                    f'ce={loss_ce.item():.3f} hard={loss_hard.item():.3f} bank={loss_bank.item():.3f} '
                    f'{(time.time()-t0)/step:.2f}s/it',
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
                        'hard_k': a.hard_k, 'family_k': a.family_k, 'hard_weight': a.hard_weight,
                    }, a.save)
                    print(f'    saved best db={d}', flush=True)
            if step >= a.max_steps:
                done = True
                break
    r, d = validate()
    print(f'FINAL raw,db=({r}, {d}) best_db={best_db}', flush=True)


if __name__ == '__main__':
    main()
