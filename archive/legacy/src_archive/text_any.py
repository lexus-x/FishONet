"""Precompute text embeddings (scientific-name prompt) for all classes with ANY open_clip model.
Usage: python src/text_any.py --model hf-hub:imageomics/bioclip-2.5-vith14 --out outputs/text_emb_h.pt"""
import json, pickle, os, torch, open_clip, argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    D = "data/dl"
    classes = list(pickle.load(open(f"{D}/all_classes.pkl", "rb")))
    desc = json.load(open(f"{D}/descriptions.json"))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, _ = open_clip.create_model_and_transforms(a.model)
    tok = open_clip.get_tokenizer(a.model)
    model = model.to(dev).eval()
    def encode(texts, bs=256):
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), bs):
                t = tok(texts[i:i+bs]).to(dev)
                f = model.encode_text(t); f = f / f.norm(dim=-1, keepdim=True)
                out.append(f.cpu().float())
        return torch.cat(out)
    emb_name = encode([f"a photo of {c}." for c in classes])
    emb_desc = encode([desc.get(c, c) for c in classes])
    os.makedirs("outputs", exist_ok=True)
    torch.save({"classes": classes, "emb_name": emb_name, "emb_desc": emb_desc, "model": a.model}, a.out)
    print("saved", a.out, emb_name.shape, emb_desc.shape)

if __name__ == "__main__":
    main()
