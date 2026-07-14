"""Zero-shot baseline: classify images against species text labels with BioCLIP.

This is the UNKNOWN-species head and a quick first leaderboard number.
Provide a CSV of class names (scientific and/or common). F-name rule:
prefer the common English name where available.

Usage:
  python src/zeroshot_baseline.py --images data/test --labels data/classes.csv \
      --out outputs/zeroshot_preds.csv
"""
import argparse, os, glob, csv
import torch
from PIL import Image

MODEL = "hf-hub:imageomics/bioclip-2"
TEMPLATE = "a photo of {}."

def load_labels(path):
    names = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            # prefer common name, fall back to scientific
            names.append(row.get("common") or row.get("scientific") or row.get("name"))
    return names

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=MODEL)
    args = ap.parse_args()

    import open_clip
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(args.model)
    tokenizer = open_clip.get_tokenizer(args.model)
    model = model.to(dev).eval()

    names = load_labels(args.labels)
    text = tokenizer([TEMPLATE.format(n) for n in names]).to(dev)
    with torch.no_grad():
        tfeat = model.encode_text(text)
        tfeat = tfeat / tfeat.norm(dim=-1, keepdim=True)

    paths = sorted(glob.glob(os.path.join(args.images, "**", "*"), recursive=True))
    paths = [p for p in paths if p.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as fo:
        w = csv.writer(fo); w.writerow(["path", "pred", "score"])
        with torch.no_grad():
            for p in paths:
                im = preprocess(Image.open(p).convert("RGB")).unsqueeze(0).to(dev)
                f = model.encode_image(im); f = f / f.norm(dim=-1, keepdim=True)
                logits = (100.0 * f @ tfeat.T).softmax(dim=-1)[0]
                k = int(logits.argmax())
                w.writerow([p, names[k], float(logits[k])])
    print("saved", args.out)

if __name__ == "__main__":
    main()
