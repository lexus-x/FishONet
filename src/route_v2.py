"""Confirm the seen/unseen test split + measure the ROUTED approach.

(A) Diagnostic on REAL test: is max-proto-sim higher for test.pkl (seen) than
    unseen.pkl (unseen)? If yes, confirms test.pkl=seen, unseen.pkl=unseen.
(B) Simulated-unknown validation of the routed strategy (per-image):
      known query -> prototype argmax over SEEN classes
      unseen query -> text argmax over classes EXCEPT seen (candidate restriction)
"""
import json, torch
from collections import defaultdict

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
seen_set = set(seen)

# ---- (A) real-test diagnostic ----
def proto_all():
    P, idx = [], []
    for c in seen:
        p = torch.stack(by_cls[c]).mean(0); P.append(p/ p.norm()); idx.append(cls_index[c])
    return torch.stack(P), torch.tensor(idx)
P, P_idx = proto_all()
def maxproto(split):
    e = torch.load(f"outputs/emb_{split}.pt", weights_only=False)
    g = (e["feats"] @ P.t()).max(1).values
    q = torch.quantile(g, torch.tensor([0.1,0.5,0.9]))
    print(f"  {split:7s} n={len(e['feats'])}  max-proto-sim  mean={g.mean():.3f}  p10/p50/p90={q[0]:.3f}/{q[1]:.3f}/{q[2]:.3f}")
print("(A) REAL test max-prototype-similarity (higher => more likely a SEEN species):")
maxproto("test"); maxproto("unseen")

# ---- (B) routed validation on simulated unknowns ----
g = torch.Generator().manual_seed(0)
perm = torch.randperm(len(seen), generator=g).tolist()
n_unseen = int(len(seen)*0.2)
pseudo_unseen = set(seen[perm[i]] for i in range(n_unseen))
known = [c for c in seen if c not in pseudo_unseen]
known_idx = torch.tensor([cls_index[c] for c in known])

Pk, Pk_idx, qk = [], [], []
for c in known:
    imgs = by_cls[c]
    if len(imgs) < 2:
        Pk.append(torch.stack(imgs).mean(0)); Pk_idx.append(cls_index[c]); continue
    k = max(1, int(round(len(imgs)*0.3)))
    qk += [(im,c) for im in imgs[:k]]
    p = torch.stack(imgs[k:]).mean(0); Pk.append(p); Pk_idx.append(cls_index[c])
Pk = torch.stack([p/p.norm() for p in Pk]); Pk_idx = torch.tensor(Pk_idx)
qu = [(im,c) for c in pseudo_unseen for im in by_cls[c]]

Fk = torch.stack([f for f,_ in qk]); yk = torch.tensor([cls_index[c] for _,c in qk])
Fu = torch.stack([f for f,_ in qu]); yu = torch.tensor([cls_index[c] for _,c in qu])

# known route: prototype argmax over KNOWN classes
known_acc = (Pk_idx[(Fk @ Pk.t()).argmax(1)] == yk).float().mean().item()
# unseen route: text argmax over classes EXCEPT seen(known) classes (candidate restriction)
mask = torch.ones(len(classes), dtype=torch.bool); mask[known_idx] = False
cand = mask.nonzero(as_tuple=True)[0]
Tu = Tname[cand]
pu = cand[(Fu @ Tu.t()).argmax(1)]
unseen_acc = (pu == yu).float().mean().item()
# unseen via text over ALL classes (no restriction) for comparison
unseen_all = (Tname.argmax(0)*0 + (Fu @ Tname.t()).argmax(1) == yu).float().mean().item()

n_seen_test, n_unseen_test = 20097, 15568
overall = (n_seen_test*known_acc + n_unseen_test*unseen_acc)/(n_seen_test+n_unseen_test)
print("\n(B) ROUTED strategy (per-image, simulated):")
print(f"  known route (proto, seen classes):           {known_acc:.4f}")
print(f"  unseen route (text, restricted to non-seen):  {unseen_acc:.4f}")
print(f"  unseen route (text, ALL classes):             {unseen_all:.4f}")
print(f"  => projected OVERALL (weighted by test sizes): {overall:.4f}")
