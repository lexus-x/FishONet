"""Best submission: BioCLIP 2.5 ViT-H.
  SEEN  (test.pkl)  -> blended  proto_sim + ALPHA*class_max_sim  over seen classes
  UNSEEN(unseen.pkl)-> DEBIASED name-text argmax over non-seen classes"""
import json, torch, argparse
from collections import defaultdict
import torch.nn.functional as F

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default="outputs/text_emb_h.pt")
    ap.add_argument("--train", default="outputs/emb_train_h.pt")
    ap.add_argument("--test", default="outputs/emb_test_h.pt")
    ap.add_argument("--unseen", default="outputs/emb_unseen_h.pt")
    ap.add_argument("--out", default="outputs/prediction_vith.json")
    ap.add_argument("--alpha", type=float, default=2.0)   # seen blend weight
    ap.add_argument("--debias", type=int, default=1)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    text = torch.load(a.text, weights_only=False)
    classes = text["classes"]; cls_index = {c: i for i, c in enumerate(classes)}
    Tname = F.normalize(text["emb_name"].float(), dim=-1)
    lab = json.load(open("data/dl/label_train.json"))
    tr = torch.load(a.train, weights_only=False)
    by_cls = defaultdict(list)
    for fn, f in zip(tr["files"], tr["feats"]):
        if fn in lab and lab[fn] in cls_index: by_cls[lab[fn]].append(f)
    seen = sorted(by_cls.keys()); s2i = {c: i for i, c in enumerate(seen)}
    P = torch.stack([F.normalize(torch.stack(by_cls[c]).float().mean(0), dim=0) for c in seen]).to(dev)
    TFE, TL = [], []
    for c in seen:
        for f in by_cls[c]: TFE.append(f); TL.append(s2i[c])
    TFE = F.normalize(torch.stack(TFE).float(), dim=-1).to(dev); TL = torch.tensor(TL).to(dev)

    preds = {}
    et = torch.load(a.test, weights_only=False)
    ef = F.normalize(et["feats"].float(), dim=-1).to(dev)
    ti = []
    for i in range(0, ef.shape[0], 4000):
        e = ef[i:i+4000]
        ps = e @ P.t()
        sim = e @ TFE.t()
        cmax = torch.full((e.shape[0], len(seen)), -1e9, device=dev)
        cmax.scatter_reduce_(1, TL.unsqueeze(0).expand(e.shape[0], -1), sim, reduce="amax")
        ti.extend((ps + a.alpha * cmax).argmax(1).cpu().tolist())
    for fn, k in zip(et["files"], ti): preds[fn] = seen[k]

    seen_set = set(seen)
    unseen_cand_idx = torch.tensor([i for i, c in enumerate(classes) if c not in seen_set])
    Tu = Tname[unseen_cand_idx].to(dev)
    eu = torch.load(a.unseen, weights_only=False)
    uf = F.normalize(eu["feats"].float(), dim=-1).to(dev)
    S = uf @ Tu.t()
    if a.debias:
        S = (S - S.mean(0, keepdim=True)) / (S.std(0, keepdim=True) + 1e-6)
    ui = S.argmax(1).cpu().tolist()
    for fn, j in zip(eu["files"], ui): preds[fn] = classes[unseen_cand_idx[j].item()]

    json.dump(preds, open(a.out, "w"))
    print(f"wrote {a.out} n={len(preds)}  seen_test={len(et['files'])} (blend a={a.alpha}) "
          f"unseen={len(eu['files'])} (debias={a.debias})")

if __name__ == "__main__":
    main()
