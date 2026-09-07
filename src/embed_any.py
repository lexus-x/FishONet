"""Embed a split with ANY open_clip model. Optional --hflip TTA (avg original+flipped).
Usage: python src/embed_any.py --model M --split train --out outputs/emb_train_h.pt [--hflip 1]"""
import argparse, os, pickle, torch, open_clip
from PIL import Image

def index_images(root):
    idx = {}
    for dp, _, fs in os.walk(root):
        for f in fs:
            if f.lower().endswith((".jpg", ".jpeg", ".png")): idx[f] = os.path.join(dp, f)
    return idx

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--split", required=True)
    ap.add_argument("--imgroot", default="data/dl/images"); ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=256); ap.add_argument("--hflip", type=int, default=0)
    a = ap.parse_args()
    files = list(pickle.load(open(f"data/dl/splits/{a.split}.pkl", "rb")))
    idx = index_images(a.imgroot)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(a.model)
    model = model.to(dev).eval()
    feats, kept, miss, buf, bufn = [], [], 0, [], []
    def flush():
        if not buf: return
        x = torch.stack(buf).to(dev)
        with torch.no_grad():
            f = model.encode_image(x); f = f / f.norm(dim=-1, keepdim=True)
            if a.hflip:
                f2 = model.encode_image(torch.flip(x, dims=[-1])); f2 = f2 / f2.norm(dim=-1, keepdim=True)
                f = f + f2; f = f / f.norm(dim=-1, keepdim=True)
        feats.append(f.cpu().float()); kept.extend(bufn); buf.clear(); bufn.clear()
    for i, fn in enumerate(files):
        p = idx.get(fn)
        if p is None: miss += 1; continue
        try: buf.append(preprocess(Image.open(p).convert("RGB"))); bufn.append(fn)
        except Exception: miss += 1; continue
        if len(buf) >= a.batch: flush()
        if i % 5000 == 0: print(f"{a.split} {i}/{len(files)} miss={miss}", flush=True)
    flush()
    os.makedirs("outputs", exist_ok=True)
    torch.save({"files": kept, "feats": torch.cat(feats)}, a.out)
    print("saved", a.out, "n=", len(kept), "dim=", feats[0].shape[1], "hflip=", a.hflip)

if __name__ == "__main__":
    main()
