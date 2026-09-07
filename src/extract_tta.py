"""Multi-crop TTA feature extraction for the fine-tuned BioCLIP-2.5 ViT-H seen tower.
Averages L2-normalized features over: center-crop + hflip, and --crops RandomResizedCrop
views (+their flips). DataLoader workers preprocess in parallel so the GPU stays saturated.
Reuses ft.py's exact LoRA model setup for consistency with v20's emb_*_ft.
  python src/extract_tta.py --extract test  --ckpt outputs/ft_lora_v2.pt --out outputs/emb_test_ft_tta.pt  --crops 6
  python src/extract_tta.py --extract train --ckpt outputs/ft_lora_v2.pt --out outputs/emb_train_ft_tta.pt --crops 6
"""
import argparse, pickle, torch, open_clip
from PIL import Image
from ft import MODEL, inject_lora, load_lora_state, index_images, DEV

class ViewsDS(torch.utils.data.Dataset):
    def __init__(self, items, pp_val, pp_train, crops):
        self.items = items; self.pv = pp_val; self.pt = pp_train; self.crops = crops
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        fn, p = self.items[i]
        im = Image.open(p).convert('RGB')
        views = [self.pv(im)] + [self.pt(im) for _ in range(self.crops)]   # [1+crops] tensors
        return i, torch.stack(views)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--extract', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--crops', type=int, default=6)
    ap.add_argument('--bs', type=int, default=256)
    ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()

    model, pp_train, pp_val = open_clip.create_model_and_transforms(MODEL)
    ck = torch.load(a.ckpt, map_location='cpu', weights_only=False)
    ca = ck.get('args', {})
    inject_lora(model, ca.get('rank', 16), ca.get('alpha', 32), ca.get('top_k_blocks', 0))
    model = model.to(DEV).eval(); load_lora_state(model, ck['lora'])

    files = list(pickle.load(open(f'data/dl/splits/{a.extract}.pkl', 'rb')))
    imgidx = index_images('data/dl/images')
    items = [(fn, imgidx[fn]) for fn in files if fn in imgidx]
    miss = len(files) - len(items)
    ds = ViewsDS(items, pp_val, pp_train, a.crops)
    dl = torch.utils.data.DataLoader(ds, batch_size=a.bs, num_workers=a.workers,
                                     pin_memory=True, persistent_workers=True)

    @torch.no_grad()
    def encode(x):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            f = model.encode_image(x).float()
        return torch.nn.functional.normalize(f, dim=-1)

    V = 1 + a.crops
    out = torch.zeros(len(items), model.visual.proj.shape[1] if getattr(model.visual,'proj',None) is not None else 1024)
    done = 0
    for idx, batch in dl:                     # batch: [B, V, 3, H, W]
        b = batch.to(DEV, non_blocking=True)
        acc = None
        for v in range(V):
            x = b[:, v]
            e = encode(x) + encode(torch.flip(x, dims=[-1]))
            acc = e if acc is None else acc + e
        acc = torch.nn.functional.normalize(acc / (2 * V), dim=-1)
        out[idx] = acc.cpu().float()
        done += len(idx)
        if done % 2048 < a.bs: print(f'{a.extract} {done}/{len(items)} miss={miss}', flush=True)

    torch.save({'files': [fn for fn, _ in items], 'feats': out}, a.out)
    print('saved', a.out, 'n=', len(items), 'dim=', out.shape[1], 'views=', 2 * V)

if __name__ == '__main__':
    main()
