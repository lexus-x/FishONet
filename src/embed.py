"""Embed all images of a split with BioCLIP; cache to outputs/emb_<split>.pt.
Usage: python src/embed.py --split train|test|unseen [--imgroot data/dl/images]
"""
import argparse, os, pickle, torch, open_clip
from PIL import Image

MODEL = "hf-hub:imageomics/bioclip-2"

def index_images(root):
    idx = {}
    for dp, _, fs in os.walk(root):
        for f in fs:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                idx[f] = os.path.join(dp, f)
    return idx

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--imgroot", default="data/dl/images")
    ap.add_argument("--out", default=None)
    ap.add_argument("--batch", type=int, default=128)
    a = ap.parse_args()
    out = a.out or f"outputs/emb_{a.split}.pt"

    files = list(pickle.load(open(f"data/dl/splits/{a.split}.pkl", "rb")))
    idx = index_images(a.imgroot)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL)
    model = model.to(dev).eval()

    feats, kept, miss = [], [], 0
    buf, bufn = [], []
    def flush():
        if not buf: return
        x = torch.stack(buf).to(dev)
        with torch.no_grad():
            f = model.encode_image(x); f = f / f.norm(dim=-1, keepdim=True)
        feats.append(f.cpu()); kept.extend(bufn); buf.clear(); bufn.clear()

    for i, fn in enumerate(files):
        p = idx.get(fn)
        if p is None: miss += 1; continue
        try:
            buf.append(preprocess(Image.open(p).convert("RGB"))); bufn.append(fn)
        except Exception:
            miss += 1; continue
        if len(buf) >= a.batch: flush()
        if i % 5000 == 0: print(f"{i}/{len(files)} miss={miss}", flush=True)
    flush()
    os.makedirs("outputs", exist_ok=True)
    torch.save({"files": kept, "feats": torch.cat(feats)}, out)
    print("saved", out, "n=", len(kept), "missing=", miss)

if __name__ == "__main__":
    main()
