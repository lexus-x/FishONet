"""Per-image validation with a proper OPEN-SET GATE (not additive blend).

Fix 1: hold out a FRACTION OF IMAGES (per-image metric, matches the real test),
       not 1 image per class.
Fix 2: gate by max-prototype cosine similarity g = max_k <img, proto_k>.
       If g >= tau  -> predict argmax over KNOWN prototypes (seen route).
       If g <  tau  -> predict argmax over ALL text embeddings (unseen route).
Sweep tau to trace the known/unknown trade-off and find the balanced optimum.
"""
import json, torch
from collections import defaultdict

def main():
    text = torch.load("outputs/text_emb.pt", weights_only=False)
    classes = text["classes"]; cls_index = {c: i for i, c in enumerate(classes)}
    Tname = text["emb_name"]

    tr = torch.load("outputs/emb_train.pt", weights_only=False)
    lab = json.load(open("data/dl/label_train.json"))
    by_cls = defaultdict(list)
    for fn, f in zip(tr["files"], tr["feats"]):
        if fn in lab and lab[fn] in cls_index:
            by_cls[lab[fn]].append(f)
    seen = sorted(by_cls.keys())

    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(len(seen), generator=g).tolist()
    n_unseen = int(len(seen) * 0.2)
    pseudo_unseen = set(seen[perm[i]] for i in range(n_unseen))
    known = [c for c in seen if c not in pseudo_unseen]

    # prototypes from known classes, holding out ~30% of each known class's images (per-image queries)
    proto, proto_cls, q_known = [], [], []
    for c in known:
        imgs = by_cls[c]
        if len(imgs) < 2:
            proto.append((torch.stack(imgs).mean(0)) ); proto_cls.append(c); continue
        k = max(1, int(round(len(imgs) * 0.3)))
        q_known += [(im, c) for im in imgs[:k]]      # held-out query images
        p = torch.stack(imgs[k:]).mean(0); proto.append(p); proto_cls.append(c)
    P = torch.stack([p / p.norm() for p in proto])
    P_idx = torch.tensor([cls_index[c] for c in proto_cls])
    q_unseen = [(im, c) for c in pseudo_unseen for im in by_cls[c]]  # all imgs (per-image)
    print(f"known={len(known)} pseudo_unseen={len(pseudo_unseen)} "
          f"q_known(imgs)={len(q_known)} q_unseen(imgs)={len(q_unseen)}")

    def feats(qs):
        return torch.stack([f for f, _ in qs]), torch.tensor([cls_index[c] for _, c in qs])
    Fk, yk = feats(q_known)
    Fu, yu = feats(q_unseen)

    # precompute gate score g = max proto sim, and both route predictions
    def route(F):
        ps = F @ P.t()                       # [N, n_proto]
        gmax, gi = ps.max(1)                 # gate score + which known proto
        seen_pred = P_idx[gi]                # known-route prediction (global class idx)
        text_pred = (F @ Tname.t()).argmax(1)  # unseen-route prediction
        return gmax, seen_pred, text_pred
    gk, seen_pk, text_pk = route(Fk)
    gu, seen_pu, text_pu = route(Fu)

    print("\n  tau   known-acc  unseen-acc  balanced   (route: g>=tau->proto else text)")
    best = None
    for tau in [0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70]:
        pk = torch.where(gk >= tau, seen_pk, text_pk)
        pu = torch.where(gu >= tau, seen_pu, text_pu)
        ka = (pk == yk).float().mean().item()
        ua = (pu == yu).float().mean().item()
        bal = 0.5*(ka+ua)
        print(f"  {tau:.2f}   {ka:7.4f}    {ua:7.4f}    {bal:7.4f}")
        if best is None or bal > best[3]: best = (tau, ka, ua, bal)
    print(f"\nBEST: tau={best[0]:.2f}  known={best[1]:.4f}  unseen={best[2]:.4f}  balanced={best[3]:.4f}")
    # reference: pure-text (no gate) per-image
    ka0 = (text_pk == yk).float().mean().item()
    ua0 = (text_pu == yu).float().mean().item()
    print(f"pure-text(name) per-image: known={ka0:.4f} unseen={ua0:.4f} balanced={0.5*(ka0+ua0):.4f}")

if __name__ == "__main__":
    main()
