"""LiT-style contrastive LoRA fine-tune of the TaxaBind image tower toward its
FROZEN taxonomic-text space (shift-matched aug). Same recipe that took ctftbig
26.62 -> ctftshift 31.10 on the pseudo-unseen proxy; applied to TaxaBind, the
strongest frozen unseen leg (+1.86 in v36). Highest-EV remaining `b` lever.

Text frozen (outputs/text_emb_taxabind_taxctx.pt), train ONLY on known classes,
validate transfer to pseudo-unseen (rarest-20% held out). Rules-legal: fine-tuning
a general-biology model (MVRL/taxabind-vit-b-16) on the PROVIDED training images.

  python research/contrastive_ft_taxabind.py --shift_aug 1 --save outputs/tb_ctftshift.pt
"""
import argparse
import json
import math
import os
import pickle
import random
import time
from collections import defaultdict

import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset

MODEL = 'hf-hub:MVRL/taxabind-vit-b-16'
DEV = 'cuda'


class LoRALinear(nn.Module):
    def __init__(s, base, r=16, alpha=32, dropout=0.05):
        super().__init__()
        s.base = base
        for p in s.base.parameters():
            p.requires_grad_(False)
        s.r = r
        s.scaling = alpha / max(r, 1)
        s.A = nn.Parameter(torch.zeros(r, base.in_features))
        s.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(s.A, a=math.sqrt(5))
        s.drop = nn.Dropout(dropout)
        s.lora_scale = 1.0

    def forward(s, x):
        return s.base(x) + (s.drop(x) @ s.A.t() @ s.B.t()) * s.scaling * s.lora_scale


def inject_lora(model, r, alpha, top_k=0):
    blocks = model.visual.transformer.resblocks
    sel = range(len(blocks)) if top_k <= 0 else range(len(blocks) - top_k, len(blocks))
    n = 0
    for i in sel:
        blk = blocks[i]
        blk.mlp.c_fc = LoRALinear(blk.mlp.c_fc, r, alpha)
        blk.mlp.c_proj = LoRALinear(blk.mlp.c_proj, r, alpha)
        n += 2
    return n


def index_images(root):
    idx = {}
    for dp, _, fs in os.walk(root):
        for f in fs:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                idx[f] = os.path.join(dp, f)
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--top_k', type=int, default=8)
    ap.add_argument('--rank', type=int, default=16)
    ap.add_argument('--alpha', type=int, default=32)
    ap.add_argument('--bs', type=int, default=128)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--max_steps', type=int, default=800)
    ap.add_argument('--val_every', type=int, default=100)
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--temp', type=float, default=30.0)
    ap.add_argument('--val_imgs', type=int, default=2318)
    ap.add_argument('--save', default='outputs/tb_ctftshift.pt')
    ap.add_argument('--shift_aug', type=int, default=1)
    a = ap.parse_args()
    D = 'data/dl'
    classes = list(pickle.load(open(f'{D}/all_classes.pkl', 'rb')))
    ci = {c: i for i, c in enumerate(classes)}
    lab = json.load(open(f'{D}/label_train.json'))
    imgidx = index_images(f'{D}/images')
    Tall = F.normalize(torch.load('outputs/text_emb_taxabind_taxctx.pt', weights_only=False)['emb_taxctx'].float(), dim=-1)
    # membership: use files that exist on disk
    by = defaultdict(list)
    for fn in lab:
        if lab[fn] in ci and fn in imgidx:
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
    val_items = val_items[:a.val_imgs]
    print(f'known={len(known)} pseudo={len(pseudo)} train_imgs={len(train_items)} val_imgs={len(val_items)}', flush=True)

    model, _, pp = open_clip.create_model_and_transforms(MODEL)
    nl = inject_lora(model, a.rank, a.alpha, a.top_k)
    model = model.to(DEV)
    for n, p in model.named_parameters():
        p.requires_grad_(n.endswith('.A') or n.endswith('.B'))
    lora_params = [p for p in model.parameters() if p.requires_grad]
    print(f'injected {nl} LoRA layers, trainable params:', sum(p.numel() for p in lora_params), flush=True)
    opt = torch.optim.AdamW(lora_params, lr=a.lr, weight_decay=0.0)

    mean = (0.48145466, 0.4578275, 0.40821073)
    std = (0.26862954, 0.26130258, 0.27577711)
    eval_pp = pp
    if a.shift_aug:
        train_tf = T.Compose([
            T.RandomResizedCrop(224, scale=(0.35, 1.0), ratio=(0.5, 2.0),
                                interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(), T.ColorJitter(0.2, 0.2, 0.2),
            T.ToTensor(), T.Normalize(mean, std)])
        print('SHIFT-MATCHED aug ON', flush=True)
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
        return round((S.argmax(1) == Y).float().mean().item() * 100, 2), round((db(S).argmax(1) == Y).float().mean().item() * 100, 2)

    def lora_sd(m):
        return {n: p.detach().cpu() for n, p in m.named_parameters() if n.endswith('.A') or n.endswith('.B')}

    model.train()
    print('BASELINE (frozen TaxaBind img tower) raw,db =', validate(), flush=True)
    warmup = max(1, int(0.08 * a.max_steps))

    def lr_at(s):
        if s < warmup:
            return a.lr * s / warmup
        p = (s - warmup) / max(1, a.max_steps - warmup)
        return a.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))

    step = 0
    t0 = time.time()
    done = False
    best_db = -1
    while not done:
        for x, y in dl:
            x = x.to(DEV)
            y = y.to(DEV)
            for g in opt.param_groups:
                g['lr'] = lr_at(step)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                f = F.normalize(model.encode_image(x).float(), dim=-1)
                logits = a.temp * (f @ Tknown.t())
                loss = F.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            if step % 50 == 0:
                print(f'step {step}/{a.max_steps} loss={loss.item():.3f} {(time.time()-t0)/step:.2f}s/it', flush=True)
            if step % a.val_every == 0:
                r, d = validate()
                print(f'  VAL @ {step}: raw,db = ({r}, {d})', flush=True)
                if d > best_db:
                    best_db = d
                    torch.save({'state': lora_sd(model), 'top_k': a.top_k, 'rank': a.rank,
                                'alpha': a.alpha, 'db': d, 'step': step}, a.save)
                    print(f'    * saved best db={d} -> {a.save}', flush=True)
            if step >= a.max_steps:
                done = True
                break
    r, d = validate()
    print(f'FINAL raw,db = ({r}, {d})  best_db={best_db}', flush=True)
    if d > best_db:
        torch.save({'state': lora_sd(model), 'top_k': a.top_k, 'rank': a.rank,
                    'alpha': a.alpha, 'db': d, 'step': step}, a.save)
        print(f'    * saved FINAL db={d}', flush=True)


if __name__ == '__main__':
    main()
