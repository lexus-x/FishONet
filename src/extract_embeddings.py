"""Extract BioCLIP image embeddings for a folder of images.

Usage: python src/extract_embeddings.py --images data/train --out outputs/emb_train.pt
"""
import argparse, os, glob
import torch
from PIL import Image

MODEL = "hf-hub:imageomics/bioclip-2"  # or hf-hub:imageomics/bioclip-2.5-vith14

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--model", default=MODEL)
    args = ap.parse_args()

    import open_clip
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(args.model)
    model = model.to(dev).eval()

    paths = sorted(glob.glob(os.path.join(args.images, "**", "*"), recursive=True))
    paths = [p for p in paths if p.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
    feats, kept = [], []
    with torch.no_grad():
        for i in range(0, len(paths), args.batch):
            batch = paths[i:i + args.batch]
            ims = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in batch]).to(dev)
            f = model.encode_image(ims)
            f = f / f.norm(dim=-1, keepdim=True)
            feats.append(f.cpu()); kept.extend(batch)
            print("%d/%d" % (min(i + args.batch, len(paths)), len(paths)))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({"paths": kept, "feats": torch.cat(feats)}, args.out)
    print("saved", args.out)

if __name__ == "__main__":
    main()
