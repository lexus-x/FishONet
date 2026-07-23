"""Local validation harness with SIMULATED unknowns (no test labels needed).

Splits the 5,795 seen classes into KNOWN (build prototypes) and PSEUDO-UNSEEN
(hidden -> must be recovered by text zero-shot). Holds out query images per
known class. Then sweeps the prototype-blend alpha and text variant, reporting
known-acc, unseen-acc, and balanced overall — our proxy for the real metric.
"""
import json, torch, argparse
from collections import defaultdict

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unseen_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    text = torch.load("outputs/text_emb.pt", weights_only=False)
    classes = text["classes"]; cls_index = {c: i for i, c in enumerate(classes)}
    Tname, Tdesc = text["emb_name"], text["emb_desc"]

    tr = torch.load("outputs/emb_train.pt", weights_only=False)
    lab = json.load(open("data/dl/label_train.json"))
    by_cls = defaultdict(list)
    for fn, f in zip(tr["files"], tr["feats"]):
        if fn in lab:
            by_cls[lab[fn]].append(f)
    seen = sorted([c for c in by_cls if c in cls_index])

    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(len(seen), generator=g).tolist()
    n_unseen = int(len(seen) * args.unseen_frac)
    pseudo_unseen = set(seen[perm[i]] for i in range(n_unseen))
    known = [c for c in seen if c not in pseudo_unseen]

    # build prototypes from KNOWN classes, holding out 1 query image each
    proto, proto_idx, q_known = [], [], []
    for c in known:
        imgs = by_cls[c]
        if len(imgs) < 2:
            continue  # need >=1 proto and >=1 query
        q_known.append((imgs[0], c))                  # held-out query
        p = torch.stack(imgs[1:]).mean(0); p = p / p.norm()
        proto.append(p); proto_idx.append(cls_index[c])
    proto = torch.stack(proto); proto_idx = torch.tensor(proto_idx)
    # pseudo-unseen queries: ALL their images (no prototype available)
    q_unseen = [(f, c) for c in pseudo_unseen for f in by_cls[c]]

    print(f"known={len(known)} pseudo_unseen={len(pseudo_unseen)} "
          f"q_known={len(q_known)} q_unseen={len(q_unseen)}")

    def acc(queries, T, alpha):
        if not queries: return 0.0
        F = torch.stack([f for f, _ in queries])
        gold = torch.tensor([cls_index[c] for _, c in queries])
        scores = F @ T.t()
        if alpha > 0:
            ps = F @ proto.t()
            scores[:, proto_idx] = scores[:, proto_idx] + alpha * ps
        pred = scores.argmax(1)
        return (pred == gold).float().mean().item()

    print("\n text  alpha  known-acc  unseen-acc  balanced")
    for tname, T in [("name", Tname), ("desc", Tdesc)]:
        for alpha in [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]:
            ka = acc(q_known, T, alpha)
            ua = acc(q_unseen, T, alpha)
            bal = 0.5 * (ka + ua)
            print(f" {tname:4s}  {alpha:4.1f}   {ka:7.4f}    {ua:7.4f}    {bal:7.4f}")

if __name__ == "__main__":
    main()
