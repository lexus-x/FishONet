"""Re-embed iNat photos with 336 overlap-strip max geometry (gallery matches v56 queries).

Stores per-class stacked photo embeddings: for each photo, mean of center+squash
PLUS 5 overlapping strips concatenated (so scoring can max over photo-views too).
Default: save 7 views concatenated per photo as extra rows in the class bank
(top-4 mean still works; more rows = more max-pool coverage).

  conda activate onet && python -u research/embed_inat_336_cropbank.py
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
sys.path.insert(0, os.path.join(ROOT, 'research'))
from extract_eval_336_crops import CKPT, DEV, MEAN, MODEL, N_STRIPS, RES, STD, overlap_strips, views_for

OUT = os.path.join(ROOT, 'outputs')
FILES_PATH = os.path.join(OUT, 'inat_image_files.json')
CLASSES_PATH = os.path.join(ROOT, 'data', 'dl', 'all_classes.pkl')
MAX_PHOTOS = 24
WORKERS = 6
BS = 3
OUT_PATH = os.path.join(OUT, 'inat_photo_bank_fullft336_cropviews.pt')
VIEW_KEYS = ['center', 'squash'] + [f'ostrip{i}' for i in range(N_STRIPS)]


class CropDS(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        ci, path = self.items[i]
        try:
            img = Image.open(path).convert('RGB')
            vs = views_for(img)
            x = torch.stack([vs[k] for k in VIEW_KEYS])
        except Exception:
            x = torch.zeros(len(VIEW_KEYS), 3, RES, RES)
            ci = -1
        return ci, x


def collate(batch):
    cis = torch.tensor([b[0] for b in batch], dtype=torch.long)
    xs = torch.cat([b[1] for b in batch], dim=0)
    return cis, xs


def main():
    if os.path.exists(OUT_PATH):
        print(f'skip {OUT_PATH}', flush=True)
        print('DONE', flush=True)
        return
    print(f'DEV={DEV}', flush=True)
    classes = pickle.load(open(CLASSES_PATH, 'rb'))
    ci_map = {c: i for i, c in enumerate(classes)}
    meta = json.load(open(FILES_PATH))
    items = []
    for c, paths in meta.items():
        if c not in ci_map:
            continue
        ok = [p for p in (paths or []) if os.path.exists(p) and os.path.getsize(p) > 1000][:MAX_PHOTOS]
        for p in ok:
            items.append((ci_map[c], p))
    print(f'photos={len(items)} classes_touched={len({i[0] for i in items})}', flush=True)

    ck = torch.load(CKPT, map_location='cpu', weights_only=False)
    model, _, _ = open_clip.create_model_and_transforms(MODEL, force_image_size=RES)
    model.load_state_dict(ck['model'] if 'model' in ck else ck)
    model = model.to(DEV).eval()

    dl = DataLoader(CropDS(items), batch_size=BS, num_workers=WORKERS, pin_memory=True,
                    collate_fn=collate)
    banks = {}
    t0 = time.time()
    n_done = 0
    with torch.no_grad():
        for bi, (cis, x) in enumerate(dl):
            x = x.to(DEV, non_blocking=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                f = F.normalize(model.encode_image(x).float(), dim=-1).cpu()
            v = len(VIEW_KEYS)
            nb = cis.shape[0]
            f = f.view(nb, v, -1)
            for j, ci in enumerate(cis.tolist()):
                if ci < 0:
                    continue
                banks.setdefault(ci, []).append(f[j])
            n_done += nb
            if bi % 20 == 0:
                print(f'  {n_done}/{len(items)} [{time.time()-t0:.0f}s]', flush=True)
    stacked = {}
    for ci, rows in banks.items():
        stacked[ci] = F.normalize(torch.cat(rows, dim=0).float(), dim=-1)
    torch.save({
        'bank': stacked,
        'enc': 'fullft336shift_cropviews',
        'ckpt': 'outputs/fullft336_shift.pt',
        'view_keys': VIEW_KEYS,
        'max_photos': MAX_PHOTOS,
    }, OUT_PATH)
    print(f'wrote {OUT_PATH} classes={len(stacked)} [{time.time()-t0:.0f}s]', flush=True)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
