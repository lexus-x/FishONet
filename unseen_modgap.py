"""Modality-gap prototype synthesis for the UNSEEN route.
Idea: for SEEN classes we have both an image-prototype P (mean train img emb) and the
taxon-text emb T. Learn a map f: text-space -> image-space on KNOWN classes, then
SYNTHESIZE a visual prototype Phat=f(T) for unseen classes and classify unseen images
against Phat instead of matching image->text across the modality gap.
Gate: pseudo-unknown holdout (rarest-20% seen hidden), same construction as cyc_*/reconcile.
Baseline to beat = deployed text-match zc(H@Ttaxon)+0.5 zc(L@Tname) = 24.42%.
CPU, cached feats only. Rules-clean (foundation embeds + provided text + train images).
"""
import json, torch
import torch.nn.functional as F
from collections import defaultdict
torch.set_num_threads(8)

txtH = torch.load("outputs/text_emb_h.pt", weights_only=False)
classes = txtH["classes"]; ci = {c: i for i, c in enumerate(classes)}
Ttax = F.normalize(torch.load("outputs/text_emb_h_taxon.pt", weights_only=False)["emb_taxon"].float(), dim=-1)  # H-space text
Tnm  = F.normalize(torch.load("outputs/text_emb.pt", weights_only=False)["emb_name"].float(), dim=-1)           # L-space text
lab = json.load(open("data/dl/label_train.json"))
def loadfeat(p):
    d = torch.load(p, weights_only=False); return {fn: i for i, fn in enumerate(d["files"])}, F.normalize(d["feats"].float(), dim=-1)
HtrI, HtrF = loadfeat("outputs/emb_train_h_tta.pt")   # image emb, H-space
LtrI, LtrF = loadfeat("outputs/emb_train.pt")          # image emb, L-space

by = defaultdict(list)
for fn in HtrI:
    if fn in lab and lab[fn] in ci and fn in LtrI: by[lab[fn]].append(fn)
seen = sorted(by.keys()); order = sorted(seen, key=lambda c: len(by[c]))
pseudo = set(order[:int(len(seen) * 0.2)]); known = [c for c in seen if c not in pseudo]
nonk = [i for i, c in enumerate(classes) if c not in set(known)]; nt = torch.tensor(nonk)
cpos = {v: j for j, v in enumerate(nonk)}

# image prototypes (H and L space) for ALL seen classes
def protos(TrI, TrF):
    P = {}
    for c in seen:
        fs = [TrF[TrI[f]] for f in by[c] if f in TrI]
        P[c] = F.normalize(torch.stack(fs).mean(0), dim=-1)
    return P
PH = protos(HtrI, HtrF)

# fit text->image map on KNOWN classes: X=text(H) , Y=image-proto(H)
Xk = torch.stack([Ttax[ci[c]] for c in known])          # (nk,1024)
Yk = torch.stack([PH[c] for c in known])                # (nk,1024)
Xall = Ttax[nt]                                          # unseen-candidate text (H)

def synth_translation(X, Y, Xq):        # "Mind-the-Gap" global offset
    d = Y.mean(0) - X.mean(0); return F.normalize(Xq + d, dim=-1)
def synth_procrustes(X, Y, Xq):         # orthogonal rotation  min||X R - Y||
    U, S, Vt = torch.linalg.svd(X.t() @ Y, full_matrices=False); R = U @ Vt
    return F.normalize(Xq @ R, dim=-1)
def synth_ridge(X, Y, Xq, lam):         # full linear map, ridge-regularized
    d = X.shape[1]; W = torch.linalg.solve(X.t() @ X + lam * torch.eye(d), X.t() @ Y)
    return F.normalize(Xq @ W, dim=-1)

# pseudo-unknown queries
qfiles = [f for c in pseudo for f in by[c] if f in HtrI and f in LtrI]
gold = torch.tensor([cpos[ci[lab[f]]] for f in qfiles])
QH = torch.stack([HtrF[HtrI[f]] for f in qfiles]); QL = torch.stack([LtrF[LtrI[f]] for f in qfiles])
def t1(M): return (M.argmax(1) == gold).float().mean().item() * 100
def zc(S): return (S - S.mean(0, keepdim=True)) / (S.std(0, keepdim=True) + 1e-6)

# baseline (deployed text-match)
base = zc(QH @ Ttax[nt].t()) + 0.5 * zc(QL @ Tnm[nt].t())
C = t1(base)
print(f"seen={len(seen)} known={len(known)} pseudo-unseen={len(pseudo)} queries={len(qfiles)} cands={len(nonk)}")
print(f"=== BASELINE text-match zc(H@taxon)+0.5zc(L@name): {C:.2f}%  (expect ~24.42) ===\n")

variants = {
    "translation": synth_translation(Xk, Yk, Xall),
    "procrustes":  synth_procrustes(Xk, Yk, Xall),
    "ridge_l1":    synth_ridge(Xk, Yk, Xall, 1.0),
    "ridge_l10":   synth_ridge(Xk, Yk, Xall, 10.0),
    "ridge_l100":  synth_ridge(Xk, Yk, Xall, 100.0),
}
for name, Phat in variants.items():
    Smg = QH @ Phat.t()                       # image -> SYNTHESIZED visual prototype
    solo = t1(zc(Smg))
    # sanity: how close are synth protos to the TRUE image protos, over pseudo classes (only these have protos)
    pseudo_pos = [cpos[ci[c]] for c in pseudo]
    truePH = torch.stack([PH[c] for c in pseudo])
    cos_to_true = F.cosine_similarity(Phat[pseudo_pos], truePH).mean().item()
    best_blend = (solo, 0.0)
    for w in [0.25, 0.5, 1.0, 2.0]:
        a = t1(base + w * zc(Smg))
        if a > best_blend[0]: best_blend = (a, w)
    print(f"{name:12s}: synth-proto-only {solo:5.2f}%  | best blend {best_blend[0]:5.2f}% @w={best_blend[1]}  (Δvs base {best_blend[0]-C:+.2f}) | cos(synth,true_img_proto)={cos_to_true:.3f}")
