"""CPU diagnostic: aspect ratios of v50↔v56 unseen-head flips.

Hypothesis: hard max over long-axis strips fires false peaks on near-square
images (strips ≈ center, or cut a texture patch). Those flips cost the 129
test-folder corrects. Legal deploy rule would key off image AR, not folders.
"""
from __future__ import annotations

import json
import os
import pickle
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from crop_views_proxy import IMG_ROOT, index_images

D = 'data/dl'


def ar_of(path):
    with Image.open(path) as im:
        w, h = im.size
    return max(w, h) / max(1, min(w, h))


def summarize(xs, name):
    if not xs:
        print(f'{name}: n=0')
        return
    a = np.array(xs, dtype=np.float64)
    qs = np.quantile(a, [0.1, 0.25, 0.5, 0.75, 0.9])
    print(
        f'{name}: n={len(a)} mean={a.mean():.3f} p10={qs[0]:.3f} p25={qs[1]:.3f} '
        f'med={qs[2]:.3f} p75={qs[3]:.3f} p90={qs[4]:.3f} frac>1.5={np.mean(a>=1.5):.3f} '
        f'frac>1.8={np.mean(a>=1.8):.3f} frac>2.0={np.mean(a>=2.0):.3f}',
        flush=True,
    )


def main():
    lab = json.load(open(f'{D}/label_train.json'))
    seen = set(lab.values())
    tf = set(pickle.load(open(f'{D}/splits/test.pkl', 'rb')))
    uf = set(pickle.load(open(f'{D}/splits/unseen.pkl', 'rb')))
    v50 = json.load(open('outputs/prediction_v50_f60_tau18.json'))
    v56 = json.load(open('outputs/prediction_v56_overlap_336_f60_tau18.json'))
    imgidx = index_images(IMG_ROOT)
    print(f'indexed {len(imgidx)} images', flush=True)

    groups = {
        'all_eval': list(v50),
        'test_folder': [fn for fn in v50 if fn in tf],
        'unseen_folder': [fn for fn in v50 if fn in uf],
        'unseen_routed': [fn for fn in v50 if v50[fn] not in seen],
        'flip_all': [fn for fn in v50 if v50[fn] != v56[fn]],
        'flip_test': [fn for fn in v50 if v50[fn] != v56[fn] and fn in tf],
        'flip_unseen': [fn for fn in v50 if v50[fn] != v56[fn] and fn in uf],
        'stable_uns_route': [fn for fn in v50 if v50[fn] not in seen and v50[fn] == v56[fn]],
    }
    out = {}
    for name, fns in groups.items():
        ars = []
        miss = 0
        for fn in fns:
            p = imgidx.get(fn)
            if p is None:
                miss += 1
                continue
            try:
                ars.append(ar_of(p))
            except Exception:
                miss += 1
        summarize(ars, f'{name} miss={miss}')
        out[name] = {
            'n': len(fns),
            'miss': miss,
            'mean': float(np.mean(ars)) if ars else None,
            'median': float(np.median(ars)) if ars else None,
            'frac_ge_15': float(np.mean(np.array(ars) >= 1.5)) if ars else None,
            'frac_ge_18': float(np.mean(np.array(ars) >= 1.8)) if ars else None,
        }
    json.dump(out, open('outputs/v56_flip_aspect.json', 'w'), indent=2)
    print('wrote outputs/v56_flip_aspect.json', flush=True)


if __name__ == '__main__':
    main()
