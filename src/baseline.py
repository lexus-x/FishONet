"""Predict over all 17,393 classes for test+unseen and write prediction.json.

Default (alpha=0): pure BioCLIP zero-shot using name-prompt text embeddings.
alpha>0: blend in seen-class image prototypes (few-shot) on seen columns.
text variant: --text name|desc.
"""
import argparse, json, pickle, torch
from collections import defaultdict

def build_prototypes(text_classes):
    cls_index = {c: i for i, c in enumerate(text_classes)}
    tr = torch.load("outputs/emb_train.pt")
    lab = json.load(open("data/dl/label_train.json"))
    acc = defaultdict(list)
    for fn, f in zip(tr["files"], tr["feats"]):
        if fn in lab and lab[fn] in cls_index:
            acc[lab[fn]].append(f)
    seen = sorted(acc.keys())
    proto = torch.stack([torch.stack(acc[c]).mean(0) for c in seen])
    proto = proto / proto.norm(dim=-1, keepdim=True)
    seen_idx = torch.tensor([cls_index[c] for c in seen])
    return proto, seen_idx

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.0)
    ap.add_argument("--text", choices=["name", "desc"], default="name")
    ap.add_argument("--out", default="outputs/prediction.json")
    a = ap.parse_args()

    text = torch.load("outputs/text_emb.pt")
    classes = text["classes"]
    T = text["emb_name" if a.text == "name" else "emb_desc"]  # [C,768]

    proto = seen_idx = None
    if a.alpha > 0:
        proto, seen_idx = build_prototypes(classes)

    preds = {}
    for split in ["test", "unseen"]:
        e = torch.load(f"outputs/emb_{split}.pt")
        F = e["feats"]
        scores = F @ T.t()                      # [N, C]
        if a.alpha > 0:
            ps = F @ proto.t()                  # [N, n_seen]
            scores[:, seen_idx] = scores[:, seen_idx] + a.alpha * ps
        top = scores.argmax(1).tolist()
        for fn, k in zip(e["files"], top):
            preds[fn] = classes[k]
    json.dump(preds, open(a.out, "w"))
    print("wrote", a.out, "n=", len(preds))

if __name__ == "__main__":
    main()
