"""Characterize the train->test capture-domain shift with simple image stats
(box only -- needs raw images). Samples N images per split, computes:
resolution, aspect, mean/std RGB, Laplacian-variance (blur), edge density.
Tells us WHAT the shift is (blur? exposure? resolution?) to inform targeted
augmentation for a shift-robust retrain.
"""
import os, sys, pickle, random, json
import numpy as np
from PIL import Image

ROOT = os.path.expanduser('~/onet')
random.seed(0)


def index_images(root):
    idx = {}
    for dp, _, fs in os.walk(root):
        for f in fs:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                idx[f] = os.path.join(dp, f)
    return idx


def stats_of(path):
    im = Image.open(path).convert('RGB')
    w, h = im.size
    im2 = im.resize((128, 128))
    a = np.asarray(im2).astype(np.float32) / 255.0
    g = a.mean(2)
    # laplacian variance (blur proxy)
    lap = (-4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:])
    lapv = float(lap.var())
    # saturation (colorfulness)
    mx, mn = a.max(2), a.min(2)
    sat = float(((mx - mn) / (mx + 1e-5)).mean())
    return dict(w=w, h=h, aspect=w / max(h, 1), mean=float(a.mean()), std=float(a.std()),
                lapv=lapv, sat=sat)


def run(split, n=2500):
    files = list(pickle.load(open(f'{ROOT}/data/dl/splits/{split}.pkl', 'rb')))
    random.shuffle(files)
    imgidx = index_images(f'{ROOT}/data/dl/images')
    rows = []
    for fn in files[:n]:
        p = imgidx.get(fn)
        if not p:
            continue
        try:
            rows.append(stats_of(p))
        except Exception:
            pass
    agg = {}
    for k in rows[0]:
        v = np.array([r[k] for r in rows])
        agg[k] = dict(mean=float(v.mean()), p10=float(np.percentile(v, 10)),
                      p50=float(np.percentile(v, 50)), p90=float(np.percentile(v, 90)))
    return agg, len(rows)


out = {}
for split in ['train', 'test', 'unseen']:
    agg, n = run(split)
    out[split] = agg
    print(f'== {split} (n={n}) ==', flush=True)
    for k, v in agg.items():
        print(f'  {k:>7}: mean={v["mean"]:.4g} p10={v["p10"]:.4g} p50={v["p50"]:.4g} p90={v["p90"]:.4g}', flush=True)

json.dump(out, open(f'{ROOT}/outputs/shift_diag.json', 'w'), indent=1)
print('wrote outputs/shift_diag.json')
