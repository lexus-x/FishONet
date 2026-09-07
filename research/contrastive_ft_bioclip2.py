"""LoRA fine-tune BioCLIP-2 ViT-L/14 image tower on competition train (frozen taxon text, 768-d).

Disclosed: provided training images only; shift_aug matches test framing.
  conda activate onet
  python research/contrastive_ft_bioclip2.py --shift_aug 1 --max_steps 800 \\
      --save outputs/b2_shift_lora.pt 2>&1 | tee outputs/b2_shift_lora_train.log
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
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
from contrastive_ft import LoRALinear, inject_lora, index_images  # noqa: E402

MODEL = 'hf-hub:imageomics/bioclip-2'
DEV = 'cuda'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--top_k', type=int, default=12)
    ap.add_argument('--rank', type=int, default=16)
    ap.add_argument('--alpha', type=int, default=32)
    ap.add_argument('--bs', type=int, default=96)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--max_steps', type=int, default=800)
    ap.add_argument('--val_every', type=int, default=100)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--temp', type=float, default=30.0)
    ap.add_argument('--val_imgs', type=int, default=2318)
    ap.add_argument('--save', default='outputs/b2_shift_lora.pt')
    ap.add_argument('--shift_aug', type=int, default=1)
    ap.add_argument('--init_ckpt', default='')
    a = ap.parse_args()

    D = 'data/dl'
    classes = list(pickle.load(open(f'{D}/all_classes.pkl', 'rb')))
    ci = {c: i for i, c in enumerate(classes)}
    lab = json.load(open(f'{D}/label_train.json'))
    imgidx = index_images(f'{D}/images')
    Tall = F.normalize(
        torch.load('outputs/text_emb.pt', weights_only=False)['emb_name'].float(), dim=-1
    )
    tr = torch.load('outputs/emb_train_bioclip2.pt', weights_only=False)
    by = defaultdict(list)
    for fn in tr['files']:
        if fn in lab and lab[fn] in ci and fn in imgidx:
            by[lab[fn]].append(fn)
    seen = sorted(by.keys())
    order = sorted(seen, key=lambda c: len(by[c]))
    pseudo = set(order[: int(len(seen) * 0.2)])
    known = [c for c in seen if c not in pseudo]
    k2i = {c: i for i, c in enumerate(known)}
    Tknown = Tall[torch.tensor([ci[c] for c in known])].to(DEV)
    cand = [i for i, c in enumerate(classes) if c not in set(known)]
    cand_pos = {cii: j for j, cii in enumerate(cand)}
    Tcand = Tall[torch.tensor(cand)]
    train_items = [(fn, k2i[lab[fn]]) for c in known for fn in by[c]]
    val_items = [(fn, cand_pos[ci[c]]) for c in pseudo for fn in by[c]]
    random.seed(0)
    random.shuffle(val_items)
    val_items = val_items[: a.val_imgs]
    print(
        f'model={MODEL} known={len(known)} pseudo={len(pseudo)} '
        f'train={len(train_items)} val={len(val_items)}',
        flush=True,
    )

    model, _, pp = open_clip.create_model_and_transforms(MODEL)
    inject_lora(model, a.rank, a.alpha, a.top_k)
    model = model.to(DEV)
    if a.init_ckpt:
        ck = torch.load(a.init_ckpt, weights_only=False)
        model.load_state_dict(ck['state'], strict=False)
        print(f'init from {a.init_ckpt} db={ck.get("db")}', flush=True)
    for n, p in model.named_parameters():
        p.requires_grad_(n.endswith('.A') or n.endswith('.B'))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr)

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
        def __init__(s, it, tf):
            s.it = it
            s.tf = tf

        def __len__(s):
            return len(s.it)

        def __getitem__(s, i):
            fn, y = s.it[i]
            try:
                return s.tf(Image.open(imgidx[fn]).convert('RGB')), y
            except Exception:
                return torch.zeros(3, 224, 224), y

    dl = DataLoader(DS(train_items, train_tf), batch_size=a.bs, shuffle=True,
                    num_workers=a.workers, pin_memory=True, drop_last=True)
    vdl = DataLoader(DS(val_items, eval_pp), batch_size=256, num_workers=a.workers, pin_memory=True)

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

    def lora_sd(m):
        return {n: p.detach().cpu() for n, p in m.named_parameters() if n.endswith('.A') or n.endswith('.B')}

    model.train()
    base = validate()
    print(f'BASELINE raw,db={base} (frozen B2 vs name text)', flush=True)
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
                loss = F.cross_entropy(a.temp * (f @ Tknown.t()), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            if step % 50 == 0:
                print(f'step {step}/{a.max_steps} loss={loss.item():.3f}', flush=True)
            if step % a.val_every == 0:
                r, d = validate()
                print(f'  VAL @ {step}: raw,db=({r}, {d})', flush=True)
                if d > best_db:
                    best_db = d
                    torch.save({'state': lora_sd(model), 'top_k': a.top_k, 'rank': a.rank,
                                'alpha': a.alpha, 'db': d, 'step': step, 'model': MODEL}, a.save)
                    print(f'    saved best db={d}', flush=True)
            if step >= a.max_steps:
                done = True
                break
    r, d = validate()
    print(f'FINAL raw,db=({r}, {d}) best_db={best_db}', flush=True)


if __name__ == '__main__':
    main()
