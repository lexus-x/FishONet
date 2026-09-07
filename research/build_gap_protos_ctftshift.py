"""Gap-fill protos: ctftshift embed of coverage images for classes with ZERO iNat proto.

Quality filter per class: keep if >= min_imgs valid embeds and mean pairwise cosine >= min_pair_cos.
Writes sparse full-class matrix to outputs/gap_protos_ctftshift.pt (iNat classes stay zero).
"""
from __future__ import annotations

import argparse
import json
import os
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
from common import FishData, OUT  # noqa: E402

MODEL = 'hf-hub:imageomics/bioclip-2.5-vith14'
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)


class LoRALinear(nn.Module):
    def __init__(s, base, r=16, alpha=32):
        super().__init__()
        s.base = base
        for p in s.base.parameters():
            p.requires_grad_(False)
        s.r = r
        s.scaling = alpha / max(r, 1)
        s.A = nn.Parameter(torch.zeros(r, base.in_features))
        s.B = nn.Parameter(torch.zeros(base.out_features, r))
        s.drop = nn.Dropout(0.0)
        s.lora_scale = 1.0

    def forward(s, x):
        return s.base(x) + (s.drop(x) @ s.A.t() @ s.B.t()) * s.scaling * s.lora_scale


def inject_lora(model, r, alpha, top_k=0):
    blocks = model.visual.transformer.resblocks
    rng = range(len(blocks)) if top_k <= 0 else range(len(blocks) - top_k, len(blocks))
    for i in rng:
        blk = blocks[i]
        blk.mlp.c_fc = LoRALinear(blk.mlp.c_fc, r, alpha)
        blk.mlp.c_proj = LoRALinear(blk.mlp.c_proj, r, alpha)


def set_scale(model, s):
    for blk in model.visual.transformer.resblocks:
        for m in (blk.mlp.c_fc, blk.mlp.c_proj):
            if isinstance(m, LoRALinear):
                m.lora_scale = s


class DS(Dataset):
    def __init__(s, items, pp, squash):
        s.items = items
        s.pp = pp
        s.squash = squash
        s.sq = T.Compose([
            T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(), T.Normalize(MEAN, STD),
        ])

    def __len__(s):
        return len(s.items)

    def __getitem__(s, i):
        ci, path = s.items[i]
        try:
            img = Image.open(path).convert('RGB')
            if s.squash:
                return ci, s.pp(img), s.sq(img)
            return ci, s.pp(img), s.pp(img)
        except Exception:
            z = torch.zeros(3, 224, 224)
            return -1, z, z


def collect_gap_paths(D, inat_cov: set[int], file_lists: list[str], root: str) -> dict[str, list[str]]:
    paths: dict[str, list[str]] = {}
    for fl in file_lists:
        if not os.path.exists(fl):
            continue
        for c, ps in json.load(open(fl)).items():
            if c not in D.ci or D.ci[c] in inat_cov:
                continue
            for pth in ps:
                ap = os.path.join(root, pth) if not os.path.isabs(pth) else pth
                if os.path.exists(ap) and os.path.getsize(ap) > 1000:
                    paths.setdefault(c, []).append(ap)
    return paths


def filter_class(feats: torch.Tensor, min_imgs: int, min_pair_cos: float) -> torch.Tensor | None:
    n = feats.shape[0]
    if n < min_imgs:
        return None
    if n >= 2:
        sim = feats @ feats.t()
        triu = sim.triu(diagonal=1)
        vals = triu[triu != 0]
        if vals.numel() and vals.mean().item() < min_pair_cos:
            return None
    proto = F.normalize(feats.mean(0), dim=-1)
    if proto.norm().item() < 0.9:
        return None
    return proto


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=os.path.join(OUT, 'ctft_shift.pt'))
    ap.add_argument('--squash_tta', type=int, default=1)
    ap.add_argument('--scale', type=float, default=0.4)
    ap.add_argument('--min_imgs', type=int, default=2)
    ap.add_argument('--min_pair_cos', type=float, default=0.15)
    ap.add_argument('--out', default=os.path.join(OUT, 'gap_protos_ctftshift.pt'))
    ap.add_argument('--files_out', default=os.path.join(OUT, 'gap_image_files.json'))
    ap.add_argument('--bs', type=int, default=128)
    a = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    D = FishData()
    inat = torch.load(os.path.join(OUT, 'inat_protos_ctftshift_full.pt'), weights_only=False)
    Pin = F.normalize(inat['protos'].float(), dim=-1)
    inat_cov = {i for i in range(len(Pin)) if Pin[i].norm() > 0.5}

    flists = [
        os.path.join(OUT, 'coverage_image_files.json'),
        os.path.join(OUT, 'coverage_image_files_extra.json'),
    ]
    gap_paths = collect_gap_paths(D, inat_cov, flists, root)
    rel = {c: [os.path.relpath(p, root) for p in ps] for c, ps in gap_paths.items()}
    json.dump(rel, open(a.files_out, 'w'), indent=0)
    print(f'gap classes with images: {len(gap_paths)} -> {a.files_out}', flush=True)

    items = []
    for c, ps in gap_paths.items():
        for p in ps:
            items.append((D.ci[c], p))
    print(f'embedding {len(items)} images', flush=True)

    model, _, pp = open_clip.create_model_and_transforms(MODEL)
    ck = torch.load(a.ckpt, weights_only=False)
    inject_lora(model, ck['rank'], ck['alpha'], ck['top_k'])
    own = dict(model.named_parameters())
    for n, v in ck['state'].items():
        own[n].data.copy_(v)
    model = model.to(DEV).eval()
    set_scale(model, a.scale)

    by_ci: dict[int, list[torch.Tensor]] = {}
    dl = DataLoader(DS(items, pp, a.squash_tta), batch_size=a.bs, num_workers=4, pin_memory=True)
    t0 = time.time()
    with torch.no_grad():
        for bi, (cis, x, xsq) in enumerate(dl):
            x = x.to(DEV)
            xsq = xsq.to(DEV)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                f = F.normalize(model.encode_image(x).float(), dim=-1)
                f = f + F.normalize(model.encode_image(torch.flip(x, dims=[-1])).float(), dim=-1)
                if a.squash_tta:
                    f = f + F.normalize(model.encode_image(xsq).float(), dim=-1)
                    f = f + F.normalize(model.encode_image(torch.flip(xsq, dims=[-1])).float(), dim=-1)
                f = F.normalize(f, dim=-1).cpu()
            for j, ci in enumerate(cis.tolist()):
                if ci < 0:
                    continue
                by_ci.setdefault(ci, []).append(f[j])
            if bi % 20 == 0:
                print(f'  batch {bi} [{time.time()-t0:.0f}s]', flush=True)

    C = len(D.classes)
    protos = torch.zeros(C, 1024)
    cnt = torch.zeros(C)
    kept, dropped = 0, 0
    for ci, flist in by_ci.items():
        feats = torch.stack(flist)
        proto = filter_class(feats, a.min_imgs, a.min_pair_cos)
        if proto is None:
            dropped += 1
            continue
        protos[ci] = proto
        cnt[ci] = len(flist)
        kept += 1

    covered = [D.classes[i] for i in range(C) if cnt[i] > 0]
    torch.save({
        'protos': protos, 'cnt': cnt, 'covered': covered,
        'enc': 'ctft', 'ckpt': a.ckpt, 'squash_tta': a.squash_tta,
        'model': MODEL, 'classes': D.classes,
        'filter': {'min_imgs': a.min_imgs, 'min_pair_cos': a.min_pair_cos},
        'dropped_classes': dropped,
    }, a.out)
    print(f'wrote {a.out}: kept {kept} classes (dropped {dropped}), {int(cnt.sum())} imgs', flush=True)


if __name__ == '__main__':
    main()
