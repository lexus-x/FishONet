"""v90 phase-1: shift-rendered holdout embeddings for the ctftshift member.

Mechanism test for Shift-Adapted Learned Routing: the encoders got shift-matched
augmentation (+0.4 real, HANDOFF SS3) but the learned gate trains on train-framing
features (emb_train_*) while deployment reads eval-framing images (aspect medians:
train 1.46 -> test 1.53 / unseen 1.92, outputs/shift_diag.json). HANDOFF:1447 names
framing shift as "the residual suspect" for the holdout->real routing gap
(AUC 0.981 holdout -> 0.911 eval); nobody has ever shown the gate eval-framed
training data. This re-extracts the 14,184 holdout rows with a LABEL-BLIND
horizontal pre-stretch; the deployed 4-view squash-TTA recipe is otherwise
unchanged (center + squash + both hflips, mean-normalized).

NOTE the squash views are stretch-invariant by construction (Resize squashes to
224x224 regardless of input aspect), so the perturbation enters via the center
views only -> this UNDERSTATES the real shift. A null here kills the
geometric-stretch mechanism at the gate, not content-level shift.

  conda activate onet && python research/extract_holdout_shift_v90.py
"""
from __future__ import annotations

import os
import sys
import time

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import open_clip
import torchvision.transforms as T
from common import DATA, OUT
from embed_inat import MODEL, inject_lora, set_scale
from learned_gate_v77 import MEMBERS, TRAIN, holdout_split, load


def load_train_embs_v90():
    """v77 loader, but emb_train_bioclip2_lora_v2.pt vanished from outputs/
    (2026-09-01) — fall back to the v1 file. The b2l column is never shifted in
    this experiment, so it is constant across renders and cannot bias the verdict."""
    train = {t: load(f'{OUT}/{TRAIN[t]}.pt') for t, _, _ in MEMBERS}
    b2l = f'{OUT}/emb_train_bioclip2_lora_v2.pt'
    if not os.path.exists(b2l):
        b2l = f'{OUT}/emb_train_bioclip2_lora.pt'
        print(f'WARNING: bioclip2_lora_v2 missing; using {b2l}', flush=True)
    return train, load(f'{OUT}/emb_train_taxabind.pt'), \
        load(f'{OUT}/emb_train_bioclip2.pt'), load(b2l)

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
IMG_ROOT = os.path.join(DATA, 'dl', 'images')
MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)
center_tf = T.Compose([T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
                       T.CenterCrop(224), T.ToTensor(), T.Normalize(MEAN, STD)])
squash_tf = T.Compose([T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
                       T.ToTensor(), T.Normalize(MEAN, STD)])
STRETCH = [1.35, 1.75]
BS = 48


def index_images(root):
    idx = {}
    for dp, _, fs in os.walk(root):
        for f in fs:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                idx[f] = os.path.join(dp, f)
    return idx


def load_ctft():
    ck = torch.load(os.path.join(OUT, 'ctft_shift.pt'), map_location='cpu', weights_only=False)
    model, _, _ = open_clip.create_model_and_transforms(MODEL)
    inject_lora(model, ck['rank'], ck['alpha'], ck['top_k'])
    own = dict(model.named_parameters())
    for n, v in ck['state'].items():
        own[n].data.copy_(v)
    model = model.to(DEV).eval()
    set_scale(model, 0.4)
    print(f'ctft_shift rank={ck["rank"]} alpha={ck["alpha"]} scale=0.4', flush=True)
    return model


class DS(Dataset):
    def __init__(s, files, idx, k):
        s.files, s.idx, s.k = files, idx, k

    def __len__(s):
        return len(s.files)

    def __getitem__(s, i):
        fn = s.files[i]
        try:
            img = Image.open(s.idx[fn]).convert('RGB')
            w, h = img.size
            img = img.resize((max(1, int(round(w * s.k))), h), Image.BICUBIC)
            return center_tf(img), squash_tf(img), fn
        except Exception:
            z = torch.zeros(3, 224, 224)
            return z, z, fn


def main():
    train, tb, b2f, b2l = load_train_embs_v90()
    sp = holdout_split(train, tb, b2f, b2l)
    files = sp['val_files']
    print(f'holdout rows: {len(files)} (seen={sp["n_seen"]})', flush=True)
    imgidx = index_images(IMG_ROOT)
    model = load_ctft()
    for k in STRETCH:
        out = os.path.join(OUT, f'emb_holdout_ctft_stretch{int(k * 100)}.pt')
        if os.path.exists(out):
            print(f'exists: {out}', flush=True)
            continue
        dl = DataLoader(DS(files, imgidx, k), batch_size=BS, num_workers=8, pin_memory=True)
        fs = []
        t0 = time.time()
        with torch.no_grad():
            for bi, (xc, xs, _) in enumerate(dl):
                xc, xs = xc.to(DEV), xs.to(DEV)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    f = F.normalize(model.encode_image(xc).float(), dim=-1)
                    f = f + F.normalize(model.encode_image(torch.flip(xc, dims=[-1])).float(), dim=-1)
                    f = f + F.normalize(model.encode_image(xs).float(), dim=-1)
                    f = f + F.normalize(model.encode_image(torch.flip(xs, dims=[-1])).float(), dim=-1)
                fs.append(F.normalize(f.float(), dim=-1).cpu())
                if bi % 20 == 0:
                    print(f'  k={k} {bi * BS}/{len(files)} {time.time() - t0:.0f}s', flush=True)
        emb = torch.cat(fs)
        torch.save({'files': files, 'feats': emb, 'stretch': k}, out)
        print(f'wrote {out} {tuple(emb.shape)} in {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
