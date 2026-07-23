"""Compare a backbone on BOTH routes (hard sim): SEEN proto-NCM + UNSEEN debiased text.
Usage: python src/eval_backbone.py --train outputs/emb_train.pt --text outputs/text_emb.pt"""
import json, torch, argparse
from collections import defaultdict
import torch.nn.functional as F

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--holdout", type=float, default=0.2)
    a = ap.parse_args()

    text = torch.load(a.text, weights_only=False)
    classes = text["classes"]; cls_index = {c: i for i, c in enumerate(classes)}
    Tname = F.normalize(text["emb_name"].float(), dim=-1)
    tr = torch.load(a.train, weights_only=False)
    lab = json.load(open("data/dl/label_train.json"))
    by_cls = defaultdict(list)
    for fn, f in zip(tr["files"], tr["feats"]):
        if fn in lab and lab[fn] in cls_index:
            by_cls[lab[fn]].append(f)
    seen = sorted(by_cls.keys())
    feats_by_cls = {c: F.normalize(torch.stack(v).float(), dim=-1) for c, v in by_cls.items()}

    # SEEN: per-class holdout, prototype NCM
    cls_list = [c for c in seen if len(by_cls[c]) >= 2]
    c2i = {c: i for i, c in enumerate(cls_list)}
    protos, valF, valY = [], [], []
    for c in cls_list:
        fs = feats_by_cls[c]
        k = min(max(1, round(a.holdout * len(fs))), len(fs) - 1)
        protos.append(F.normalize(fs[:-k].mean(0), dim=0))
        for j in range(len(fs) - k, len(fs)):
            valF.append(fs[j]); valY.append(c2i[c])
    P = torch.stack(protos); VF = torch.stack(valF); VY = torch.tensor(valY)
    seen_acc = ((VF @ P.t()).argmax(1) == VY).float().mean().item() * 100

    # UNSEEN: hard split (rarest 20%), debiased text
    order = sorted(seen, key=lambda c: len(by_cls[c]))
    pseudo = set(order[:int(len(seen) * 0.2)])
    known_idx = set(cls_index[c] for c in seen if c not in pseudo)
    cand = [i for i in range(len(classes)) if i not in known_idx]
    cand_t = torch.tensor(cand); cand_pos = {ci: j for j, ci in enumerate(cand)}
    qf, qy = [], []
    for c in pseudo:
        for f in by_cls[c]: qf.append(f); qy.append(cls_index[c])
    Fq = F.normalize(torch.stack(qf).float(), dim=-1)
    gold = torch.tensor([cand_pos[y] for y in qy])
    S = Fq @ Tname[cand_t].t()
    Sdb = (S - S.mean(0, keepdim=True)) / (S.std(0, keepdim=True) + 1e-6)
    def t1(M): return (M.argmax(1) == gold).float().mean().item() * 100
    def rk(M, Ks):
        top = M.topk(max(Ks), 1).indices
        return {K: round((top[:, :K] == gold.unsqueeze(1)).any(1).float().mean().item() * 100, 1) for K in Ks}
    dim = tr["feats"].shape[1]
    print(f"=== {a.train} (dim={dim}) / {a.text} ===")
    print(f"SEEN  proto-NCM (holdout) : {seen_acc:.2f}%   (val={len(VY)}, classes={len(cls_list)})")
    print(f"UNSEEN name top1          : {t1(S):.2f}%")
    print(f"UNSEEN name+debias top1   : {t1(Sdb):.2f}%")
    print(f"UNSEEN recall@            : {rk(S,[1,5,10,20,50,100])}")
    # projected overall (real test sizes 20097 seen / 15568 unseen), using debiased unseen
    proj = (20097*seen_acc + 15568*t1(Sdb)) / (20097+15568)
    print(f"PROJECTED overall (sim)   : {proj:.2f}%   [seen {seen_acc:.1f} x.566 + unseenDB {t1(Sdb):.1f} x.434]")

if __name__ == "__main__":
    main()
