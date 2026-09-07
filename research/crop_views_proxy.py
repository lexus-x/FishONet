"""Holdout probe: better crops than center+squash TTA (AWS GPU).

Current shift TTA = center-crop + aspect-squash + hflip (mean). Elongated fish
lose head/tail on center-crop and get distorted on squash. This adds:

  letterbox  — aspect-preserving pad (no cut, no squash)
  longstrips — 3 square windows along the long axis (head / mid / tail)

Scoring: mean TTA vs max-over-crops (raw cosine, then one dbnorm).
Stack = v50-like text+B2+ctft-bank; only the QUERY encoding of the iNat bank leg changes.

  conda activate onet && python research/crop_views_proxy.py
"""
from __future__ import annotations

import json
import os
import sys
import time

import open_clip
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import FishData, dbnorm, load_emb, OUT, DATA
from embed_inat import inject_lora, set_scale, MODEL
from inat_maxpool_proxy import score_maxpool, top1
from v41_taxctx_weight_cv import stack_base
from v50_shiftbank_proxy import queries, proto_leg, IMG_W, B2_WF, B2_WL

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
RES = 224
MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)
FILL = tuple(int(round(m * 255)) for m in MEAN)
KILL = 0.3
CKPT = os.path.join(OUT, 'ctft_shift.pt')
BANK = os.path.join(OUT, 'inat_photo_bank_ctftshift.pt')
FROZEN_PROTO = os.path.join(OUT, 'inat_tol_merged_b2_a05.pt')
FROZEN_Q = os.path.join(OUT, 'emb_train_bioclip2.pt')
LORA_PROTO = os.path.join(OUT, 'inat_tol_merged_b2lora_a0.5.pt')
LORA_Q = os.path.join(OUT, 'emb_train_bioclip2_lora_v2.pt')
CTFT_Q = os.path.join(OUT, 'emb_train_ctftshift.pt')
IMG_ROOT = os.path.join(DATA, 'dl', 'images')

to_tensor = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])
center_tf = T.Compose([
    T.Resize(RES, interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(RES),
    T.ToTensor(),
    T.Normalize(MEAN, STD),
])
squash_tf = T.Compose([
    T.Resize((RES, RES), interpolation=T.InterpolationMode.BICUBIC),
    T.ToTensor(),
    T.Normalize(MEAN, STD),
])


def index_images(root):
    idx = {}
    for dp, _, fs in os.walk(root):
        for f in fs:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                idx[f] = os.path.join(dp, f)
    return idx


def letterbox(img, size=RES, fill=FILL):
    w, h = img.size
    scale = size / max(w, h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = img.resize((nw, nh), Image.BICUBIC)
    canvas = Image.new('RGB', (size, size), fill)
    canvas.paste(resized, ((size - nw) // 2, (size - nh) // 2))
    return canvas


def long_strips(img):
    """Square windows covering the long axis. Unique boxes only."""
    w, h = img.size
    crops = []
    if w >= h:
        side = h
        xs = [0, max(0, (w - side) // 2), max(0, w - side)]
        for x in dict.fromkeys(xs):
            crops.append(img.crop((x, 0, x + side, side)))
    else:
        side = w
        ys = [0, max(0, (h - side) // 2), max(0, h - side)]
        for y in dict.fromkeys(ys):
            crops.append(img.crop((0, y, side, y + side)))
    out = []
    for c in crops:
        out.append(c.resize((RES, RES), Image.BICUBIC))
    return out


def load_ctft():
    ck = torch.load(CKPT, map_location='cpu', weights_only=False)
    model, _, _ = open_clip.create_model_and_transforms(MODEL)
    inject_lora(model, ck['rank'], ck['alpha'], ck['top_k'])
    own = dict(model.named_parameters())
    for n, v in ck['state'].items():
        own[n].data.copy_(v)
    model = model.to(DEV).eval()
    set_scale(model, 0.4)
    print(f'ctft_shift rank={ck["rank"]} alpha={ck["alpha"]} top_k={ck["top_k"]} '
          f'scale=0.4 db={ck.get("db")}', flush=True)
    return model


@torch.no_grad()
def encode_pil_batch(model, tensors):
    x = torch.stack(tensors).to(DEV)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        f = model.encode_image(x).float()
    return F.normalize(f, dim=-1).cpu()


def views_for(img):
    """Named views as tensors [3,H,W]."""
    v = {
        'center': center_tf(img),
        'squash': squash_tf(img),
        'letterbox': to_tensor(letterbox(img)),
    }
    for i, crop in enumerate(long_strips(img)):
        v[f'strip{i}'] = to_tensor(crop)
    v['center_flip'] = torch.flip(v['center'], dims=[-1])
    v['squash_flip'] = torch.flip(v['squash'], dims=[-1])
    v['letterbox_flip'] = torch.flip(v['letterbox'], dims=[-1])
    return v


def mean_stack(feats):
    s = sum(feats)
    return F.normalize(s, dim=-1)


def score_bank_raw(Q, bank, cand, topm=4):
    N = Q.shape[0]
    C = len(cand)
    S = torch.full((N, C), -1e4)
    for j, gidx in enumerate(cand.tolist()):
        photos = bank.get(gidx)
        if photos is None or photos.numel() == 0:
            continue
        sim = Q @ photos.float().t()
        k = min(topm, sim.shape[1])
        S[:, j] = sim.topk(k, dim=1).values.mean(dim=1)
    return S


def main():
    print(f'DEV={DEV}', flush=True)
    D = FishData()
    cand = D.cand
    S0, gold, qH = stack_base(D, cand, 1.0)
    bank = torch.load(BANK, weights_only=False)['bank']
    Sct = score_maxpool(qH, bank, cand, topm=4)
    fi, ff, _ = load_emb(FROZEN_Q)
    Qf, _ = queries(D, fi, ff)
    Sf = proto_leg(Qf, FROZEN_PROTO, cand)
    li, lf, _ = load_emb(LORA_Q)
    Ql, _ = queries(D, li, lf)
    Sl = proto_leg(Ql, LORA_PROTO, cand)
    base_rest = S0 + B2_WF * Sf + B2_WL * Sl
    ref = top1(base_rest + IMG_W * Sct, gold)
    print(f'REF cached ctft+squash_tta bank={ref:.4f}', flush=True)

    idx, feats, files = load_emb(CTFT_Q)
    q_files, q_gold = [], []
    for c in D.pseudo:
        for fn in D.by[c]:
            if fn in idx:
                q_files.append(fn)
                q_gold.append(D.cand_pos[D.ci[c]])
    q_gold = torch.tensor(q_gold)
    assert torch.equal(q_gold, gold)
    imgidx = index_images(IMG_ROOT)
    print(f'queries={len(q_files)} indexed={sum(1 for f in q_files if f in imgidx)}', flush=True)

    model = load_ctft()
    view_keys = [
        'center', 'squash', 'letterbox', 'center_flip', 'squash_flip', 'letterbox_flip',
        'strip0', 'strip1', 'strip2',
    ]
    n = len(q_files)
    dim = int(feats.shape[1])
    packs = {k: [None] * n for k in view_keys}
    miss = 0
    t0 = time.time()
    buf_i, buf_tensors, buf_keys = [], [], []

    def flush():
        if not buf_tensors:
            return
        f = encode_pil_batch(model, buf_tensors)
        for row, key, qi in zip(f, buf_keys, buf_i):
            packs[key][qi] = row
        buf_tensors.clear()
        buf_keys.clear()
        buf_i.clear()

    for i, fn in enumerate(q_files):
        p = imgidx.get(fn)
        cached = feats[idx[fn]]
        if p is None:
            miss += 1
            for k in view_keys:
                packs[k][i] = cached
            continue
        try:
            img = Image.open(p).convert('RGB')
            vs = views_for(img)
        except Exception:
            miss += 1
            for k in view_keys:
                packs[k][i] = cached
            continue
        if 'strip1' not in vs:
            vs['strip1'] = vs['strip0']
        if 'strip2' not in vs:
            vs['strip2'] = vs.get('strip1', vs['strip0'])
        for k in view_keys:
            buf_tensors.append(vs[k])
            buf_keys.append(k)
            buf_i.append(i)
        if len(buf_tensors) >= 72:
            flush()
        if i % 200 == 0:
            print(f'  encode {i}/{n} miss={miss} [{time.time()-t0:.0f}s]', flush=True)
    flush()
    print(f'encoded miss={miss} [{time.time()-t0:.0f}s]', flush=True)

    tensors = {
        k: F.normalize(torch.stack(rows).float(), dim=-1) for k, rows in packs.items()
    }

    recipes = {
        'cached_squash_tta': qH,
        'center+flip': mean_stack([tensors['center'], tensors['center_flip']]),
        'squash+flip': mean_stack([tensors['squash'], tensors['squash_flip']]),
        'letterbox+flip': mean_stack([tensors['letterbox'], tensors['letterbox_flip']]),
        'center+squash+flips (current recipe)': mean_stack([
            tensors['center'], tensors['center_flip'], tensors['squash'], tensors['squash_flip'],
        ]),
        'current+letterbox': mean_stack([
            tensors['center'], tensors['center_flip'], tensors['squash'], tensors['squash_flip'],
            tensors['letterbox'], tensors['letterbox_flip'],
        ]),
        'strips_mean': mean_stack([tensors[k] for k in tensors if k.startswith('strip')]),
        'current+strips': mean_stack([
            tensors['center'], tensors['center_flip'], tensors['squash'], tensors['squash_flip'],
        ] + [tensors[k] for k in tensors if k.startswith('strip')]),
        'current+letterbox+strips': mean_stack([
            tensors['center'], tensors['center_flip'], tensors['squash'], tensors['squash_flip'],
            tensors['letterbox'], tensors['letterbox_flip'],
        ] + [tensors[k] for k in tensors if k.startswith('strip')]),
    }

    rows_out = []
    best, best_name = ref, 'ref_cached'
    for name, Q in recipes.items():
        S = score_maxpool(Q, bank, cand, topm=4)
        acc = top1(base_rest + IMG_W * S, gold)
        d = acc - ref
        rows_out.append({'mode': name, 'pool': 'mean', 'proxy': acc, 'delta': d})
        print(f'mean {name}: {acc:.4f} ({d:+.3f})', flush=True)
        if acc > best:
            best, best_name = acc, f'mean:{name}'

    # max-over-crops on raw bank sims
    groups = {
        'max(center,squash)': ['center', 'squash'],
        'max(center,squash,letterbox)': ['center', 'squash', 'letterbox'],
        'max(center,squash,strips)': ['center', 'squash'] + [k for k in tensors if k.startswith('strip')],
        'max(all_no_flip)': ['center', 'squash', 'letterbox'] + [k for k in tensors if k.startswith('strip')],
        'max(current4+letterbox)': [
            'center', 'center_flip', 'squash', 'squash_flip', 'letterbox', 'letterbox_flip',
        ],
        'max(current4+strips)': [
            'center', 'center_flip', 'squash', 'squash_flip',
        ] + [k for k in tensors if k.startswith('strip')],
        'max(all)': [k for k in tensors],
    }
    for name, keys in groups.items():
        raws = [score_bank_raw(tensors[k], bank, cand, topm=4) for k in keys if k in tensors]
        if not raws:
            continue
        stacked = torch.stack(raws, dim=0)
        S = dbnorm(stacked.max(0).values)
        acc = top1(base_rest + IMG_W * S, gold)
        d = acc - ref
        rows_out.append({'mode': name, 'pool': 'max', 'proxy': acc, 'delta': d})
        print(f'max  {name}: {acc:.4f} ({d:+.3f})', flush=True)
        if acc > best:
            best, best_name = acc, f'max:{name}'

    delta = best - ref
    out = {
        'ref': ref,
        'best': best,
        'best_name': best_name,
        'delta': delta,
        'clears_kill_03': delta >= KILL,
        'n_query': len(q_files),
        'miss': miss,
        'proj_real_020': 51.44259077526987 + 0.20 * delta,
        'proj_real_036': 51.44259077526987 + 0.36 * delta,
        'rows': sorted(rows_out, key=lambda r: -r['proxy']),
    }
    op = os.path.join(OUT, 'crop_views_proxy.json')
    json.dump(out, open(op, 'w'), indent=2)
    print(json.dumps({k: out[k] for k in out if k != 'rows'}, indent=2), flush=True)
    print(f'wrote {op}', flush=True)


if __name__ == '__main__':
    main()
