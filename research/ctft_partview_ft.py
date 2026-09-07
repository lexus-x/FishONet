"""Part-view LiT: continue ctft_shift so head/mid/tail strips still match taxon text.

v56 real +1.23 unseen came from overlap strips at inference, but the encoder was
trained on squash/RRC only — parts are out of distribution. This trains LoRA on
(squash, random long-axis strip) both classified to the same frozen taxon text.

Init: outputs/ctft_shift.pt (db=24.76). Kill if best db never beats 24.76.

  conda activate onet && python -u research/ctft_partview_ft.py
"""
from __future__ import annotations

import json
import math
import os
import pickle
import random
import sys
import time
from collections import defaultdict

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT
from crop_views_proxy import DEV, MEAN, RES, STD, load_ctft

CKPT = os.path.join(OUT, 'ctft_shift.pt')
SAVE = os.path.join(OUT, 'ctft_partview.pt')
IMG_ROOT = 'data/dl/images'
MAX_STEPS = 2500
VAL_EVERY = 200
BS = 48
LR = 2e-4
TEMP = 30.0
WORKERS = 8


def index_images(root):
    idx = {}
    for dp, _, fs in os.walk(root):
        for f in fs:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                idx[f] = os.path.join(dp, f)
    return idx


def random_strip(img, res=RES):
    w, h = img.size
    if w >= h:
        side = h
        x = 0 if w <= side else random.randint(0, w - side)
        crop = img.crop((x, 0, x + side, side))
    else:
        side = w
        y = 0 if h <= side else random.randint(0, h - side)
        crop = img.crop((0, y, side, y + side))
    return crop.resize((res, res), Image.BICUBIC)


to_tensor = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])
squash_tf = T.Compose([
    T.Resize((RES, RES), interpolation=T.InterpolationMode.BICUBIC),
    T.RandomHorizontalFlip(),
    T.ColorJitter(0.15, 0.15, 0.15),
    T.ToTensor(),
    T.Normalize(MEAN, STD),
])
eval_tf = T.Compose([
    T.Resize(RES, interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(RES),
    T.ToTensor(),
    T.Normalize(MEAN, STD),
])


class PairDS(Dataset):
    def __init__(self, items, imgidx):
        self.items = items
        self.imgidx = imgidx

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        fn, y = self.items[i]
        try:
            img = Image.open(self.imgidx[fn]).convert('RGB')
            return squash_tf(img), to_tensor(random_strip(img)), y
        except Exception:
            z = torch.zeros(3, RES, RES)
            return z, z, y


class EvalDS(Dataset):
    def __init__(self, items, imgidx):
        self.items = items
        self.imgidx = imgidx

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        fn, y = self.items[i]
        try:
            return eval_tf(Image.open(self.imgidx[fn]).convert('RGB')), y
        except Exception:
            return torch.zeros(3, RES, RES), y


def lora_sd(model):
    return {n: p.detach().cpu() for n, p in model.named_parameters() if n.endswith('.A') or n.endswith('.B')}


def main():
    print(f'DEV={DEV}', flush=True)
    D = 'data/dl'
    classes = list(pickle.load(open(f'{D}/all_classes.pkl', 'rb')))
    ci = {c: i for i, c in enumerate(classes)}
    lab = json.load(open(f'{D}/label_train.json'))
    imgidx = index_images(IMG_ROOT)
    Tall = F.normalize(torch.load('outputs/text_emb_h_taxon.pt', weights_only=False)['emb_taxon'].float(), dim=-1)
    tr = torch.load('outputs/emb_train_h.pt', weights_only=False)
    by = defaultdict(list)
    for fn in tr['files']:
        if fn in lab and lab[fn] in ci and fn in imgidx:
            by[lab[fn]].append(fn)
    seen = sorted(by.keys())
    order = sorted(seen, key=lambda c: len(by[c]))
    pseudo = set(order[:int(len(seen) * 0.2)])
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
    val_items = val_items[:2318]
    print(f'known={len(known)} pseudo={len(pseudo)} train={len(train_items)} val={len(val_items)}', flush=True)

    model = load_ctft()
    for n, p in model.named_parameters():
        p.requires_grad_(n.endswith('.A') or n.endswith('.B'))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=0.0)
    dl = DataLoader(PairDS(train_items, imgidx), batch_size=BS, shuffle=True, num_workers=WORKERS,
                    pin_memory=True, drop_last=True)
    vdl = DataLoader(EvalDS(val_items, imgidx), batch_size=128, num_workers=WORKERS, pin_memory=True)

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
        return round((S.argmax(1) == Y).float().mean().item() * 100, 2), round(
            (db(S).argmax(1) == Y).float().mean().item() * 100, 2)

    warmup = max(1, int(0.08 * MAX_STEPS))

    def lr_at(s):
        if s < warmup:
            return LR * s / warmup
        p = (s - warmup) / max(1, MAX_STEPS - warmup)
        return LR * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))

    model.train()
    print('BASELINE raw,db =', validate(), '  [ctft_shift db=24.76]', flush=True)
    step = 0
    t0 = time.time()
    best_db = -1.0
    done = False
    while not done:
        for x1, x2, y in dl:
            x1, x2, y = x1.to(DEV), x2.to(DEV), y.to(DEV)
            for g in opt.param_groups:
                g['lr'] = lr_at(step)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                f1 = F.normalize(model.encode_image(x1).float(), dim=-1)
                f2 = F.normalize(model.encode_image(x2).float(), dim=-1)
                loss = F.cross_entropy(TEMP * (f1 @ Tknown.t()), y) + F.cross_entropy(
                    TEMP * (f2 @ Tknown.t()), y)
                loss = loss + 0.2 * (1.0 - (f1 * f2).sum(-1).mean())
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            if step % 50 == 0:
                print(f'step {step}/{MAX_STEPS} loss={loss.item():.3f} {(time.time()-t0)/step:.2f}s/it',
                      flush=True)
            if step % VAL_EVERY == 0:
                r, d = validate()
                print(f'  VAL @ {step}: raw,db = ({r}, {d})', flush=True)
                if d > best_db:
                    best_db = d
                    torch.save({
                        'state': {n: p.detach().cpu() for n, p in model.named_parameters()},
                        'lora': lora_sd(model),
                        'top_k': 16, 'rank': 48, 'alpha': 96,
                        'db': d, 'step': step, 'scale': 0.4,
                    }, SAVE)
                    print(f'    * saved best db={d} -> {SAVE}', flush=True)
            if step >= MAX_STEPS:
                done = True
                break
    r, d = validate()
    print(f'FINAL raw,db = ({r}, {d}) best_db={best_db}', flush=True)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
