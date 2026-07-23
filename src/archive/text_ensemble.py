"""Prompt-ensemble text embeddings: encode each class name with multiple templates,
average the normalized embeddings. Often improves zero-shot over a single template.
Usage: python src/text_ensemble.py --model hf-hub:imageomics/bioclip-2.5-vith14 --out outputs/text_emb_h_ens.pt"""
import json, pickle, os, torch, open_clip, argparse

TEMPLATES = [
    "a photo of {c}.",
    "a photo of a {c}, a species of fish.",
    "{c}",
    "an underwater photo of {c}.",
    "a photo of {c}, a fish.",
    "a close-up photo of {c}.",
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    D = "data/dl"
    classes = list(pickle.load(open(f"{D}/all_classes.pkl", "rb")))
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

    acc = torch.zeros(len(classes), model.text_projection.shape[1] if hasattr(model, "text_projection") else 1024)
    acc = None
    for tmpl in TEMPLATES:
        emb = encode([tmpl.format(c=c) for c in classes])
        acc = emb if acc is None else acc + emb
        print("encoded template:", tmpl, flush=True)
    emb_name = acc / acc.norm(dim=-1, keepdim=True)  # average then renormalize
    desc = json.load(open(f"{D}/descriptions.json"))
    emb_desc = encode([desc.get(c, c) for c in classes])
    os.makedirs("outputs", exist_ok=True)
    torch.save({"classes": classes, "emb_name": emb_name, "emb_desc": emb_desc, "model": a.model}, a.out)
    print("saved", a.out, emb_name.shape, f"({len(TEMPLATES)} templates)")

if __name__ == "__main__":
    main()
