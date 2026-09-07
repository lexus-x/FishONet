"""Generic ctft-style checkpoint extractor (WiSE-FT scale + hflip TTA fusion),
mirroring extract_ctftbig_full.py's enc_tta exactly. Saves {files, feats}.

Run on box: python research/extract_ctft_any.py --ckpt outputs/ctft_cleanse.pt \
  --scale 0.4 --split train --out outputs/emb_train_ctftclean.pt
"""
import os, sys, time, argparse, pickle
import torch, torch.nn as nn, torch.nn.functional as F
import open_clip
from PIL import Image
from torch.utils.data import Dataset, DataLoader

MODEL = 'hf-hub:imageomics/bioclip-2.5-vith14'; DEV = 'cuda'


class LoRALinear(nn.Module):
    def __init__(s, base, r=16, alpha=32):
        super().__init__(); s.base = base
        for p in s.base.parameters(): p.requires_grad_(False)
        s.r = r; s.scaling = alpha / max(r, 1)
        s.A = nn.Parameter(torch.zeros(r, base.in_features)); s.B = nn.Parameter(torch.zeros(base.out_features, r))
        s.drop = nn.Dropout(0.0); s.lora_scale = 1.0

    def forward(s, x): return s.base(x) + (s.drop(x) @ s.A.t() @ s.B.t()) * s.scaling * s.lora_scale


def inject_lora(model, r, alpha, top_k=0):
    blocks = model.visual.transformer.resblocks
    for i in (range(len(blocks)) if top_k <= 0 else range(len(blocks) - top_k, len(blocks))):
        blk = blocks[i]; blk.mlp.c_fc = LoRALinear(blk.mlp.c_fc, r, alpha); blk.mlp.c_proj = LoRALinear(blk.mlp.c_proj, r, alpha)


def set_scale(model, s):
    for blk in model.visual.transformer.resblocks:
        for m in (blk.mlp.c_fc, blk.mlp.c_proj):
            if isinstance(m, LoRALinear): m.lora_scale = s


def index_images(root):
    idx = {}
    for dp, _, fs in os.walk(root):
        for f in fs:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')): idx[f] = os.path.join(dp, f)
    return idx


ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', required=True)
ap.add_argument('--scale', type=float, default=0.4)
ap.add_argument('--split', default='train')
ap.add_argument('--out', required=True)
ap.add_argument('--bs', type=int, default=256)
ap.add_argument('--workers', type=int, default=4)
ap.add_argument('--squash_tta', type=int, default=0,
                help='1 = add squash-view TTA (for shift_aug ckpts)')
a = ap.parse_args()

ck = torch.load(a.ckpt, weights_only=False)
print('ckpt', a.ckpt, 'db', ck.get('db'), 'step', ck.get('step'), 'top_k', ck.get('top_k'), 'rank', ck.get('rank'), flush=True)
model, _, pp = open_clip.create_model_and_transforms(MODEL)
inject_lora(model, ck['rank'], ck['alpha'], ck['top_k'])
own = dict(model.named_parameters())
for n, v in ck['state'].items(): own[n].data.copy_(v)
model = model.to(DEV).eval()
set_scale(model, a.scale)

imgidx = index_images('data/dl/images')
files = list(pickle.load(open(f'data/dl/splits/{a.split}.pkl', 'rb')))
print(f'split={a.split} files={len(files)} squash_tta={a.squash_tta}', flush=True)

import torchvision.transforms as T
mean = (0.48145466, 0.4578275, 0.40821073); std = (0.26862954, 0.26130258, 0.27577711)
squash_tf = T.Compose([T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
                       T.ToTensor(), T.Normalize(mean, std)])


class DS(Dataset):
    def __init__(s, it): s.it = it
    def __len__(s): return len(s.it)
    def __getitem__(s, i):
        fn = s.it[i]
        try:
            img = Image.open(imgidx[fn]).convert('RGB')
            if a.squash_tta:
                return pp(img), squash_tf(img), fn
            return pp(img), fn
        except Exception:
            z = torch.zeros(3, 224, 224)
            return (z, z, fn) if a.squash_tta else (z, fn)


dl = DataLoader(DS(files), batch_size=a.bs, num_workers=a.workers, pin_memory=True)
fs, kept = [], []
t0 = time.time()
with torch.no_grad():
    for bi, batch in enumerate(dl):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            if a.squash_tta:
                x, xsq, fns = batch
                x, xsq = x.to(DEV), xsq.to(DEV)
                f = F.normalize(model.encode_image(x).float(), dim=-1)
                f = f + F.normalize(model.encode_image(torch.flip(x, dims=[-1])).float(), dim=-1)
                f = f + F.normalize(model.encode_image(xsq).float(), dim=-1)
                f = f + F.normalize(model.encode_image(torch.flip(xsq, dims=[-1])).float(), dim=-1)
                f = F.normalize(f, dim=-1)
            else:
                x, fns = batch
                x = x.to(DEV)
                f = F.normalize(model.encode_image(x).float(), dim=-1)
                f2 = F.normalize(model.encode_image(torch.flip(x, dims=[-1])).float(), dim=-1)
                f = F.normalize(f + f2, dim=-1)
        fs.append(f.cpu()); kept += list(fns)
        if bi % 50 == 0: print(f'{bi * a.bs}/{len(files)} {(time.time() - t0):.0f}s', flush=True)
torch.save({'files': kept, 'feats': torch.cat(fs)}, a.out)
print('saved', a.out, 'n=', len(kept), flush=True)
