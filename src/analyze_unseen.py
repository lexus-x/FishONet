"""Decision-grade unseen-route analysis on a REALISTIC (hard) sim split.
Pseudo-unseen = rarest 20% of seen classes (mimics novel/rare real unseen).
Reports: top-1 baseline (name/desc), recall@K narrowing ceiling, and debiasing variants."""
import json, torch
from collections import defaultdict
import torch.nn.functional as F

text = torch.load("outputs/text_emb.pt", weights_only=False)
classes = text["classes"]; cls_index = {c: i for i, c in enumerate(classes)}
Tname = F.normalize(text["emb_name"].float(), dim=-1)
Tdesc = F.normalize(text["emb_desc"].float(), dim=-1)

tr = torch.load("outputs/emb_train.pt", weights_only=False)
lab = json.load(open("data/dl/label_train.json"))
by_cls = defaultdict(list)
for fn, f in zip(tr["files"], tr["feats"]):
    if fn in lab and lab[fn] in cls_index:
        by_cls[lab[fn]].append(f)
seen = sorted(by_cls.keys())

# HARD split: rarest 20% by image count = pseudo-unseen
order = sorted(seen, key=lambda c: len(by_cls[c]))
n_un = int(len(seen) * 0.2)
pseudo = set(order[:n_un]); known = [c for c in seen if c not in pseudo]
known_idx = set(cls_index[c] for c in known)
cand = [i for i in range(len(classes)) if i not in known_idx]
cand_pos = {ci: j for j, ci in enumerate(cand)}
cand_t = torch.tensor(cand)

qf, qy = [], []
for c in pseudo:
    for f in by_cls[c]:
        qf.append(f); qy.append(cls_index[c])
Fq = F.normalize(torch.stack(qf).float(), dim=-1)
gold = torch.tensor([cand_pos[y] for y in qy])
print(f"pseudo_unseen_classes={len(pseudo)}  queries={len(qy)}  candidates={len(cand)}")
print(f"(avg imgs/pseudo-class={len(qy)/len(pseudo):.1f})")

def t1(S): return (S.argmax(1) == gold).float().mean().item() * 100
def recallK(S, Ks):
    top = S.topk(max(Ks), 1).indices
    return {K: round((top[:, :K] == gold.unsqueeze(1)).any(1).float().mean().item() * 100, 1) for K in Ks}

Sn = Fq @ Tname[cand_t].t()
Sd = Fq @ Tdesc[cand_t].t()
Ks = [1, 5, 10, 20, 50, 100]
print(f"\n[name]      top1={t1(Sn):.2f}%   recall@={recallK(Sn,Ks)}")
print(f"[desc]      top1={t1(Sd):.2f}%   recall@={recallK(Sd,Ks)}")
print(f"[name+desc] top1={t1(0.5*(Sn+Sd)):.2f}%   recall@={recallK(0.5*(Sn+Sd),Ks)}")
print("\n--- debiasing (on name) ---")
print(f"[-colmean]   top1={t1(Sn - Sn.mean(0,keepdim=True)):.2f}%")
print(f"[-colzscore] top1={t1((Sn - Sn.mean(0,keepdim=True))/(Sn.std(0,keepdim=True)+1e-6)):.2f}%")
print(f"[-rowmean]   top1={t1(Sn - Sn.mean(1,keepdim=True)):.2f}%")
mu = Fq.mean(0, keepdim=True)
Sn_mg = F.normalize(Fq - mu, dim=-1) @ Tname[cand_t].t()
print(f"[-imgmean]   top1={t1(Sn_mg):.2f}%   recall@={recallK(Sn_mg,Ks)}")
# combine best debias + name+desc
Sc = 0.5*(Sn+Sd); Sc = Sc - Sc.mean(0,keepdim=True)
print(f"[name+desc -colmean] top1={t1(Sc):.2f}%   recall@={recallK(Sc,Ks)}")
