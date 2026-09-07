"""Build iNat photo banks under ftshift and fullft336shift encoders.

Missing banks called out in HANDOFF 2026-08-08: unseen route only had the
ctftshift bank. This extracts per-class photo embeddings (max 24 / class)
with squash-TTA + hflip, matching outputs/inat_photo_bank_ctftshift.pt.

  conda activate onet
  python research/embed_inat_shift_banks.py
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time

import open_clip
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, 'src')
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from ft import MODEL, inject_lora, load_lora_state, set_lora_scale

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)
FILES_PATH = os.path.join(ROOT, 'outputs', 'inat_image_files.json')
CLASSES_PATH = os.path.join(ROOT, 'data', 'dl', 'all_classes.pkl')
MAX_PHOTOS = 24
SQUASH_TTA = 1
HFLIP = 1
WORKERS = 6
BS_224 = 48
BS_336 = 16
CKPT_FT = os.path.join(ROOT, 'outputs', 'ft_lora_shift.pt')
CKPT_336 = os.path.join(ROOT, 'outputs', 'fullft336_shift.pt')
OUT_FT = os.path.join(ROOT, 'outputs', 'inat_photo_bank_ftshift.pt')
OUT_336 = os.path.join(ROOT, 'outputs', 'inat_photo_bank_fullft336shift.pt')


class PhotoDS(Dataset):
    def __init__(self, items, preprocess, res, squash):
        self.items = items
        self.pp = preprocess
        self.res = res
        self.squash = squash
        self.sq = T.Compose([
            T.Resize((res, res), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(MEAN, STD),
        ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        ci, path = self.items[i]
        z = torch.zeros(3, self.res, self.res)
        try:
            img = Image.open(path).convert('RGB')
            x = self.pp(img)
            xsq = self.sq(img) if self.squash else x
            return ci, x, xsq
        except Exception:
            return -1, z, z


def collect_items(files_path, classes, max_photos):
    name_to_ci = {c: i for i, c in enumerate(classes)}
    meta = json.load(open(files_path))
    items = []
    n_cls = 0
    n_skip = 0
    for name, paths in meta.items():
        ci = name_to_ci.get(name)
        if ci is None:
            n_skip += 1
            continue
        ok = [p for p in (paths or []) if os.path.exists(p) and os.path.getsize(p) > 1000][:max_photos]
        if not ok:
            continue
        n_cls += 1
        for p in ok:
            items.append((ci, p))
    print(f'items={len(items)} classes={n_cls} skipped_unmapped={n_skip} max_photos={max_photos}',
          flush=True)
    return items


@torch.no_grad()
def encode_batch(model, x, xsq, hflip, squash_tta):
    def enc(t):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            f = model.encode_image(t).float()
        return F.normalize(f, dim=-1)

    f = enc(x)
    if hflip:
        f = f + enc(torch.flip(x, dims=[-1]))
    if squash_tta:
        f = f + enc(xsq)
        if hflip:
            f = f + enc(torch.flip(xsq, dims=[-1]))
    return F.normalize(f, dim=-1).cpu()


def run_embed(model, preprocess, items, res, bs, workers, squash_tta, hflip, tag):
    ds = PhotoDS(items, preprocess, res, squash_tta)
    dl = DataLoader(
        ds, batch_size=bs, num_workers=workers, pin_memory=True,
        persistent_workers=workers > 0, prefetch_factor=2 if workers > 0 else None,
    )
    banks = {}
    t0 = time.time()
    n_ok = 0
    print(f'{tag}: batches={len(dl)} bs={bs} res={res} squash_tta={squash_tta} hflip={hflip}',
          flush=True)
    for bi, (cis, x, xsq) in enumerate(dl):
        x = x.to(DEV, non_blocking=True)
        xsq = xsq.to(DEV, non_blocking=True)
        f = encode_batch(model, x, xsq, hflip, squash_tta)
        for j, ci in enumerate(cis.tolist()):
            if ci < 0:
                continue
            banks.setdefault(ci, []).append(f[j])
            n_ok += 1
        if bi % 50 == 0:
            print(f'  {tag} batch {bi}/{len(dl)} photos={n_ok} [{time.time() - t0:.0f}s]',
                  flush=True)
    stacked = {ci: torch.stack(vecs).float() for ci, vecs in banks.items()}
    ns = [v.shape[0] for v in stacked.values()]
    dim = next(iter(stacked.values())).shape[1] if stacked else -1
    print(f'{tag} done: classes={len(stacked)} photos={sum(ns) if ns else 0} '
          f'minmax={min(ns) if ns else 0}/{max(ns) if ns else 0} dim={dim} '
          f'[{time.time() - t0:.0f}s]', flush=True)
    return stacked


def load_ftshift(ckpt_path):
    print(f'loading LoRA ckpt {ckpt_path}', flush=True)
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    a = ck.get('args') or {}
    rank = a.get('rank', 16)
    alpha = a.get('alpha', 32)
    top_k = a.get('top_k_blocks', 0)
    lora_scale = a.get('lora_scale', 1.0)
    print(f'ft_lora_shift rank={rank} alpha={alpha} top_k_blocks={top_k} '
          f'lora_scale={lora_scale} val_acc={ck.get("val_acc")} lora_n={len(ck.get("lora", {}))}',
          flush=True)
    print('creating BioCLIP-2.5 ViT-H/14 @ 224', flush=True)
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL)
    inject_lora(model, rank, alpha, top_k)
    w0 = model.visual.ln_post.weight.detach().clone()
    b0 = model.visual.ln_post.bias.detach().clone()
    proj0 = model.visual.proj.detach().clone() if getattr(model.visual, 'proj', None) is not None else None
    model = model.to(DEV).eval()
    load_lora_state(model, ck['lora'])
    set_lora_scale(model, float(lora_scale))
    if abs(float(lora_scale) - 1.0) > 1e-6:
        al = float(lora_scale)
        model.visual.ln_post.weight.data.copy_(((1 - al) * w0 + al * model.visual.ln_post.weight.data.cpu()).to(DEV))
        model.visual.ln_post.bias.data.copy_(((1 - al) * b0 + al * model.visual.ln_post.bias.data.cpu()).to(DEV))
        if proj0 is not None:
            model.visual.proj.data.copy_(((1 - al) * proj0 + al * model.visual.proj.data.cpu()).to(DEV))
        print(f'WiSE-FT lora_scale={al}', flush=True)
    return model, preprocess


def load_fullft336(ckpt_path):
    print(f'loading fullft ckpt {ckpt_path}', flush=True)
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    print(f'fullft336_shift res={ck.get("res")} val_acc={ck.get("val_acc")}', flush=True)
    print('creating BioCLIP-2.5 ViT-H/14 @ 336', flush=True)
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL, force_image_size=336)
    model.load_state_dict(ck['model'])
    model = model.to(DEV).eval()
    return model, preprocess


def save_bank(stacked, out_path, enc, ckpt):
    payload = {
        'bank': stacked,
        'enc': enc,
        'ckpt': ckpt,
        'squash_tta': SQUASH_TTA,
        'hflip': HFLIP,
        'max_photos': MAX_PHOTOS,
    }
    torch.save(payload, out_path)
    print(f'wrote {out_path}', flush=True)


def main():
    print(f'DEV={DEV} cuda={torch.cuda.is_available()}', flush=True)
    if torch.cuda.is_available():
        print(f'gpu={torch.cuda.get_device_name(0)}', flush=True)
    classes = pickle.load(open(CLASSES_PATH, 'rb'))
    print(f'all_classes n={len(classes)}', flush=True)
    items = collect_items(FILES_PATH, classes, MAX_PHOTOS)

    if os.path.exists(OUT_FT):
        print(f'skip encoder A, exists {OUT_FT}', flush=True)
    else:
        print('=== encoder A ftshift 224 ===', flush=True)
        model, preprocess = load_ftshift(CKPT_FT)
        stacked = run_embed(model, preprocess, items, 224, BS_224, WORKERS,
                            SQUASH_TTA, HFLIP, 'ftshift')
        save_bank(stacked, OUT_FT, 'ftshift', 'outputs/ft_lora_shift.pt')
        del model, stacked
        torch.cuda.empty_cache()

    if os.path.exists(OUT_336):
        print(f'skip encoder B, exists {OUT_336}', flush=True)
    else:
        print('=== encoder B fullft336shift 336 ===', flush=True)
        model, preprocess = load_fullft336(CKPT_336)
        stacked = run_embed(model, preprocess, items, 336, BS_336, WORKERS,
                            SQUASH_TTA, HFLIP, 'fullft336shift')
        save_bank(stacked, OUT_336, 'fullft336shift', 'outputs/fullft336_shift.pt')
        del model, stacked
        torch.cuda.empty_cache()

    print('DONE', flush=True)


if __name__ == '__main__':
    main()
