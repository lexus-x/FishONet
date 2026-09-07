"""Better crops than 3-strip / letterbox: tight foreground box + overlapping long-axis windows.

No species model — border-color keyed bbox only. Legal under competition rules.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from crop_views_proxy import RES, FILL, to_tensor, letterbox, long_strips, views_for


def tight_foreground(img, pad=0.06):
    """Crop to pixels unlike the border median. Falls back to the full image."""
    w, h = img.size
    arr = np.asarray(img.convert('RGB'), dtype=np.uint8)
    bw = max(4, min(12, w // 32, h // 32))
    border = np.concatenate(
        [
            arr[:bw].reshape(-1, 3),
            arr[-bw:].reshape(-1, 3),
            arr[:, :bw].reshape(-1, 3),
            arr[:, -bw:].reshape(-1, 3),
        ],
        axis=0,
    )
    bg = np.median(border.astype(np.float32), axis=0)
    dist = np.linalg.norm(arr.astype(np.float32) - bg, axis=2)
    thr = max(22.0, float(np.percentile(dist, 35)))
    mask = dist > thr
    ys, xs = np.where(mask)
    if xs.size < 64:
        return img
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    pw = int(round(pad * (x1 - x0)))
    ph = int(round(pad * (y1 - y0)))
    x0, y0 = max(0, x0 - pw), max(0, y0 - ph)
    x1, y1 = min(w, x1 + pw), min(h, y1 + ph)
    if (x1 - x0) < 32 or (y1 - y0) < 32:
        return img
    if (x1 - x0) * (y1 - y0) > 0.94 * w * h:
        return img
    return img.crop((x0, y0, x1, y1))


def overlap_strips(img, n=5):
    """n overlapping square windows along the long axis."""
    w, h = img.size
    n = max(3, int(n))
    crops = []
    if w >= h:
        side = h
        span = max(0, w - side)
        if span == 0:
            return [img.resize((RES, RES), Image.BICUBIC)]
        xs = [int(round(i * span / (n - 1))) for i in range(n)]
        for x in dict.fromkeys(xs):
            crops.append(img.crop((x, 0, x + side, side)))
    else:
        side = w
        span = max(0, h - side)
        if span == 0:
            return [img.resize((RES, RES), Image.BICUBIC)]
        ys = [int(round(i * span / (n - 1))) for i in range(n)]
        for y in dict.fromkeys(ys):
            crops.append(img.crop((0, y, side, y + side)))
    return [c.resize((RES, RES), Image.BICUBIC) for c in crops]


def views_for_v2(img):
    """Base no-flip views plus tight-box and 5 overlapping strips."""
    v = views_for(img)
    tight = tight_foreground(img)
    v['tight_letterbox'] = to_tensor(letterbox(tight))
    v['tight_squash'] = to_tensor(tight.resize((RES, RES), Image.BICUBIC))
    for i, crop in enumerate(overlap_strips(img, n=5)):
        v[f'ostrip{i}'] = to_tensor(crop)
    for i, crop in enumerate(long_strips(tight)):
        v[f'tstrip{i}'] = to_tensor(crop)
    return v
