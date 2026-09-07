"""Precompute BioCLIP text embeddings for ALL 17,393 classes (seen+unseen).
No images needed. Two variants: scientific-name prompt, and (truncated) description.
Output: outputs/text_emb.pt  with keys: classes, emb_name, emb_desc, model.
"""
import json, pickle, os, torch, open_clip

D = "data/dl"
MODEL = "hf-hub:imageomics/bioclip-2"

def main():
    classes = list(pickle.load(open(f"{D}/all_classes.pkl", "rb")))
    desc = json.load(open(f"{D}/descriptions.json"))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, _ = open_clip.create_model_and_transforms(MODEL)
    tok = open_clip.get_tokenizer(MODEL)
    model = model.to(dev).eval()

    def encode(texts, bs=256):
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), bs):
                t = tok(texts[i:i+bs]).to(dev)
                f = model.encode_text(t)
                f = f / f.norm(dim=-1, keepdim=True)
                out.append(f.cpu())
                if (i // bs) % 10 == 0:
                    print(f"{min(i+bs,len(texts))}/{len(texts)}", flush=True)
        return torch.cat(out)

    name_prompts = [f"a photo of {c}." for c in classes]
    desc_prompts = [desc.get(c, c) for c in classes]  # tokenizer truncates to ctx len
    print("encoding name prompts ...", flush=True)
    emb_name = encode(name_prompts)
    print("encoding description prompts ...", flush=True)
    emb_desc = encode(desc_prompts)

    os.makedirs("outputs", exist_ok=True)
    torch.save({"classes": classes, "emb_name": emb_name,
                "emb_desc": emb_desc, "model": MODEL}, "outputs/text_emb.pt")
    print("saved outputs/text_emb.pt", emb_name.shape, emb_desc.shape)

if __name__ == "__main__":
    main()
