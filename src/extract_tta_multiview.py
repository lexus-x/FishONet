"""Track B: multi-view TTA extraction to attack the measured train->test shift.

shift_diag.json: test images are more elongated than train (aspect p90 2.12 vs 1.91). The standard
preprocess resizes the short side to 336 then CENTER-CROPS 336 -- on an elongated fish that crops
off the head/tail, losing exactly the discriminative parts. Current cached embeddings use hflip-only
TTA, so they inherit this crop loss.

This extracts each query image as the average of 4 views:
  v1: standard (resize short->336, center-crop 336)          [current single view]
  v2: aspect-squash (resize 336x336, no crop -> whole fish)  [recovers head/tail of elongated fish]
  each x horizontal flip.
All L2-normalized, summed, renormalized. Queries only (test+unseen, ~35665) -- keep existing
single-view train prototypes (cosine-comparable; TTA denoises the query direction). If real A_full
lifts, extend to train + other members.

Usage:
  python src/extract_tta_multiview.py --model fullft336 --ckpt outputs/fullft336.pt --split test
  python src/extract_tta_multiview.py --model fullft336 --ckpt outputs/fullft336.pt --split unseen

Cannot be validated on holdout (holdout has no shift) -- built to be verified by a real submission.
"""
import argparse, os, pickle, time
import torch
import torchvision.transforms as T
from PIL import Image
import open_clip

DEV = 'cuda'
MODEL = 'hf-hub:imageomics/bioclip-2.5-vith14'
RES = 336
MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)

view_std = T.Compose([                                   # v1: resize short side, center crop
    T.Resize(RES, interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(RES),
    T.ToTensor(), T.Normalize(MEAN, STD),
])
view_squash = T.Compose([                                # v2: squash whole image to RESxRES
    T.Resize((RES, RES), interpolation=T.InterpolationMode.BICUBIC),
    T.ToTensor(), T.Normalize(MEAN, STD),
])


def index_images(root):
    idx = {}
    for dp, _, fs in os.walk(root):
        for f in fs:
            idx[f] = os.path.join(dp, f)
    return idx


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='fullft336')
    ap.add_argument('--ckpt', default='outputs/fullft336.pt')
    ap.add_argument('--split', required=True, choices=['test', 'unseen', 'train'])
    ap.add_argument('--bs', type=int, default=96)
    ap.add_argument('--out', default='')
    args = ap.parse_args()
    out = args.out or f'outputs/emb_{args.split}_{args.model}_tta4.pt'

    model, _, _ = open_clip.create_model_and_transforms(MODEL, force_image_size=RES)
    ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    model.load_state_dict(ck['model'] if 'model' in ck else ck)
    model = model.to(DEV).eval()

    files = list(pickle.load(open(f'data/dl/splits/{args.split}.pkl', 'rb')))
    imgidx = index_images('data/dl/images')

    def encode(x):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            f = model.encode_image(x).float()
        return f / f.norm(dim=-1, keepdim=True)

    feats, kept = [], []
    buf1, buf2, bufn, miss = [], [], [], 0
    t0 = time.time()

    def flush():
        if not buf1:
            return
        x1 = torch.stack(buf1).to(DEV)
        x2 = torch.stack(buf2).to(DEV)
        f = (encode(x1) + encode(torch.flip(x1, dims=[-1]))
             + encode(x2) + encode(torch.flip(x2, dims=[-1])))
        f = f / f.norm(dim=-1, keepdim=True)
        feats.append(f.cpu().float())
        kept.extend(bufn)
        buf1.clear(); buf2.clear(); bufn.clear()

    for i, fn in enumerate(files):
        p = imgidx.get(fn)
        if p is None:
            miss += 1
            continue
        try:
            img = Image.open(p).convert('RGB')
            buf1.append(view_std(img)); buf2.append(view_squash(img)); bufn.append(fn)
        except Exception:
            miss += 1
            continue
        if len(buf1) >= args.bs:
            flush()
        if i % 2500 == 0:
            print(f'{args.split} {i}/{len(files)} miss={miss} {time.time()-t0:.0f}s', flush=True)
    flush()
    torch.save({'files': kept, 'feats': torch.cat(feats)}, out)
    print(f'saved {out} n={len(kept)} dim={feats[0].shape[1]} views=4(std+squash)x(orig+hflip) {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
