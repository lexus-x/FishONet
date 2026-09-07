"""Margin-aware soft fusion of two submission JSONs using per-image score dumps.

If score dumps absent, falls back to prefer-A / prefer-B / majority with third.

Expects optional:
  outputs/scores_v44_unseen.pt  — dict with files, idx_uns, logits [n_uns, C]
  outputs/scores_v46_unseen.pt

Without scores: builds hybrid that keeps v46 on disagree (identity) — skip.

Better path implemented here: rebuild lightweight fusion from stored
prediction disagreements + optional third voter (v43).

  conda activate onet && python research/build_v47_margin_fusion.py
"""
from __future__ import annotations

import json
import os
import shutil
import zipfile
from collections import Counter

OUT = 'outputs'
SUB = 'submissions'


def main():
    a = json.load(open(f'{OUT}/prediction_v44_toldenser_w4_wf2_wl1_a05_sink72.json'))
    # v46 may not exist yet
    p46 = f'{OUT}/prediction_v46_loratoldenser_w4_wf2.5_wl2_a05_sink72.json'
    if not os.path.exists(p46):
        # try glob
        import glob
        cands = glob.glob(f'{OUT}/prediction_v46_loratoldenser*.json')
        if not cands:
            raise SystemExit('v46 prediction missing — build v46 first')
        p46 = cands[0]
    b = json.load(open(p46))
    c = json.load(open(f'{OUT}/prediction_v43_b2dual_w4_wf0.5_wl1_sink72.json'))
    files = list(a.keys())
    assert set(files) == set(b) == set(c)

    # Strategy: where a==b keep; where disagree, if either equals c take that;
    # else take b (newer denser). This is a conservative 3-way reconcile.
    out = {}
    n_ab = n_pick_c = n_pick_b = 0
    for fn in files:
        pa, pb, pc = a[fn], b[fn], c[fn]
        if pa == pb:
            out[fn] = pa
            n_ab += 1
        elif pa == pc:
            out[fn] = pa
            n_pick_c += 1
        elif pb == pc:
            out[fn] = pb
            n_pick_c += 1
        else:
            out[fn] = pb  # prefer v46
            n_pick_b += 1

    tag = 'v47_fuse_v46v44v43'
    pred_path = f'{OUT}/prediction_{tag}.json'
    json.dump(out, open(pred_path, 'w'))
    zpath = f'{OUT}/submission_{tag}.zip'
    with zipfile.ZipFile(zpath, 'w', zipfile.ZIP_DEFLATED) as z:
        z.write(pred_path, arcname='prediction.json')
    os.makedirs(SUB, exist_ok=True)
    shutil.copy(zpath, f'{SUB}/submission_{tag}.zip')
    nd = sum(1 for fn in files if out[fn] != a[fn])
    print({
        'agree_ab': n_ab,
        'reconcile_via_v43': n_pick_c,
        'prefer_v46': n_pick_b,
        'diff_vs_v44': nd,
        'diff_pct': round(100 * nd / len(files), 2),
        'zip': f'{SUB}/submission_{tag}.zip',
    }, flush=True)


if __name__ == '__main__':
    main()
