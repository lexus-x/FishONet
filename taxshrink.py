"""Taxonomic text-anchored prototype shrinkage (seen route), single encoder (ft, H-space).
Shrink each species image-prototype toward a taxonomic anchor, count-adaptively, then RENORMALIZE
(keeps all cosines on one scale -> avoids the score-scale mismatch that killed count-gated text):
    P_shrunk_c = normalize((1-a_c)*P_c + a_c*Anchor_c),   a_c = k/(k+n_c)
Anchors: family-TEXT emb (novel; defined even for 0-img/unseen), family-IMAGE-mean (empirical).
Gate: FT-H-alone holdout (baseline ~87.06). Report overall + stratified by train-count.
"""
import json, torch
import torch.nn.functional as F
from collections import defaultdict
torch.set_num_threads(8)

txtH = torch.load("outputs/text_emb_h.pt", weights_only=False)
classes = txtH["classes"]; ci = {c: i for i, c in enumerate(classes)}
TtH = F.normalize(torch.load("outputs/text_emb_h_taxon.pt", weights_only=False)["emb_taxon"].float(), dim=-1)
fam = torch.load("outputs/text_emb_h_family.pt", weights_only=False)
famkey = "emb_family" if "emb_family" in fam else [k for k in fam if k.startswith("emb")][0]
TfamH = F.normalize(fam[famkey].float(), dim=-1)   # per-species family-text emb, H-space
tax = json.load(open("outputs/taxonomy_full.json"))
lab = json.load(open("data/dl/label_train.json"))
def load(p):
    d = torch.load(p, weights_only=False); return {fn: i for i, fn in enumerate(d["files"])}, F.normalize(d["feats"].float(), dim=-1)
FtI, FtF = load("outputs/emb_train_ft.pt")

by = defaultdict(list)
for fn in FtI:
    if fn in lab and lab[fn] in ci: by[lab[fn]].append(fn)
seen = sorted(by.keys()); s2i = {c: i for i, c in enumerate(seen)}; S = len(seen)
TseenTax = torch.stack([TtH[ci[c]] for c in seen])
Afam_text = torch.stack([TfamH[ci[c]] for c in seen])              # family-text anchor per seen class
famof = {c: (tax.get(c, {}).get("family") or "NA") for c in seen}

# holdout split (same as v20)
trby = defaultdict(list); valrows = []
for c in seen:
    fns = sorted(by[c])
    if len(fns) >= 3:
        k = max(1, round(0.2 * len(fns)))
        for f in fns[:-k]: trby[c].append(f)
        for f in fns[-k:]: valrows.append((f, s2i[c]))
    else:
        for f in fns: trby[c].append(f)
VY = torch.tensor([y for _, y in valrows]); vF = torch.stack([FtF[FtI[f]] for f, _ in valrows])
ncnt = torch.tensor([float(len(trby[c])) for c in seen])           # train count per seen class (in fit split)

# base prototypes + class-max structures (from trby)
P = torch.zeros(S, FtF.shape[1]); TF = []; TL = []
for c in seen:
    fs = [FtF[FtI[f]] for f in trby[c]]
    P[s2i[c]] = torch.stack(fs).sum(0); TF += fs; TL += [s2i[c]] * len(fs)
P = F.normalize(P / ncnt.clamp(min=1).unsqueeze(1), dim=-1)
TF = torch.stack(TF); TL = torch.tensor(TL)
# family-image-mean anchor (mean of member-class prototypes)
byfam = defaultdict(list)
for c in seen: byfam[famof[c]].append(s2i[c])
Afam_img = torch.zeros(S, P.shape[1])
for f, idxs in byfam.items():
    m = F.normalize(P[idxs].mean(0), dim=-1)
    for i in idxs: Afam_img[i] = m

cmax = torch.full((vF.shape[0], S), -1e9)
cmax.scatter_reduce_(1, TL.unsqueeze(0).expand(vF.shape[0], -1), vF @ TF.t(), reduce="amax")
taxon = vF @ TseenTax.t()
bins = [(0, 2, "n<=2"), (3, 4, "3-4"), (5, 9, "5-9"), (10, 10 ** 9, "10+")]
def strat(pred):
    corr = pred.eq(VY)
    out = []
    for lo, hi, name in bins:
        m = (ncnt[VY] >= lo) & (ncnt[VY] <= hi)
        out.append(f"{name}:{100*corr[m].float().mean().item():.1f}%(n={m.sum().item()})" if m.any() else f"{name}:-")
    return "  ".join(out)
def score(Pc): return (vF @ Pc.t()) + 2.0 * cmax + 4.0 * taxon

base = score(P).argmax(1)
print(f"seen={S} val={len(valrows)} monotypic-fam-frac={sum(1 for f in byfam if len(byfam[f])==1)/len(byfam):.2f}")
baseacc = 100 * base.eq(VY).float().mean().item()
print(f"BASELINE (no shrink): {baseacc:.2f}%   | {strat(base)}\n")

for aname, A in [("fam-TEXT", Afam_text), ("fam-IMG", Afam_img)]:
    for k in [1, 2, 5, 10, 20]:
        a = (k / (k + ncnt)).unsqueeze(1)
        Ps = F.normalize((1 - a) * P + a * A, dim=-1)
        pred = score(Ps).argmax(1)
        acc = 100 * pred.eq(VY).float().mean().item()
        flag = "  <== beats base" if acc > 100 * base.eq(VY).float().mean().item() else ""
        print(f"{aname:8s} k={k:<3}: {acc:.2f}%{flag}   | {strat(pred)}")
    print()
