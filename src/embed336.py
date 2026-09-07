"""Extract FROZEN BioCLIP-2.5 ViT-H features at a custom resolution (pos-emb interpolated).
Fast: parallel DataLoader + optional hflip TTA. Gate test for whether higher res helps the seen route.
  python src/embed336.py --split train --out outputs/emb_train_h336.pt --res 336 --hflip 1
"""
import argparse, os, pickle, torch, open_clip
from PIL import Image
MODEL = 'hf-hub:imageomics/bioclip-2.5-vith14'

def index_images(root):
    idx = {}
    for dp, _, fs in os.walk(root):
        for f in fs:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')): idx[f] = os.path.join(dp, f)
    return idx

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--res', type=int, default=336); ap.add_argument('--bs', type=int, default=128)
    ap.add_argument('--workers', type=int, default=10); ap.add_argument('--hflip', type=int, default=1)
    a = ap.parse_args()
    dev = 'cuda'
    model, _, pp = open_clip.create_model_and_transforms(MODEL, force_image_size=a.res)
    model = model.to(dev).eval()
    files = list(pickle.load(open(f'data/dl/splits/{a.split}.pkl', 'rb')))
    idx = index_images('data/dl/images')
    valid = [fn for fn in files if fn in idx]
    print(f'{a.split}: {len(valid)}/{len(files)} images at res={a.res}', flush=True)
    class DS(torch.utils.data.Dataset):
        def __init__(s, it): s.it = it
        def __len__(s): return len(s.it)
        def __getitem__(s, i):
            fn = s.it[i]
            try: return pp(Image.open(idx[fn]).convert('RGB')), fn
            except Exception: return torch.zeros(3, a.res, a.res), fn
    dl = torch.utils.data.DataLoader(DS(valid), batch_size=a.bs, num_workers=a.workers, pin_memory=True)
    feats, kept = [], []
    for x, fns in dl:
        x = x.to(dev, non_blocking=True)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            f = model.encode_image(x).float(); f = f / f.norm(dim=-1, keepdim=True)
            if a.hflip:
                f2 = model.encode_image(torch.flip(x, dims=[-1])).float(); f2 = f2 / f2.norm(dim=-1, keepdim=True)
                f = f + f2; f = f / f.norm(dim=-1, keepdim=True)
        feats.append(f.cpu().float()); kept.extend(list(fns))
        if len(kept) % 5120 < a.bs: print(f'{a.split} {len(kept)}/{len(valid)}', flush=True)
    torch.save({'files': kept, 'feats': torch.cat(feats)}, a.out)
    print('saved', a.out, 'n=', len(kept), 'dim=', feats[0].shape[1], 'res=', a.res)

if __name__ == '__main__':
    main()
