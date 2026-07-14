"""v21-improved: v20 multi-encoder + CORAL covariance alignment (4th channel) + genus fallback.
Extends v20: adds CORAL-transformed ft embeddings as a 4th z-scored channel, and
genus-level fallback when species confidence is low.

GO/NO-GO: holdout must beat v20's 87.94%. If it doesn't, CORAL+genus are NO-GO.
  python src/predict_v21_improved.py --generate         # probe
  python src/predict_v21_improved.py --generate --real  # full submission
"""
import json, torch, argparse, zipfile
from collections import defaultdict
import torch.nn.functional as F
torch.set_num_threads(8)

# ---- config ----
LAM = 4.0
CORAL_WEIGHT = 0.3      # weight on CORAL-aligned ft channel in z-scored blend
GENUS_THRESHOLD = 0.3   # percentile threshold: fallback if max_score < this fraction of top score
GENUS_FALLBACK_ENABLED = True

def load(p):
    d = torch.load(p, weights_only=False)
    return {fn: i for i, fn in enumerate(d['files'])}, F.normalize(d['feats'].float(), dim=-1), list(d['files'])

def zc(M): return (M - M.mean()) / (M.std() + 1e-6)

def shrink(C, alpha=0.1):
    D = C.shape[0]; tr = torch.trace(C) / D
    return (1-alpha)*C + alpha*tr*torch.eye(D)

def coral_transform(Xs, Xt, alpha=0.1):
    """CORAL covariance alignment (Sun & Saenko 2016). Whitens Xt then recolors with Xs."""
    mu_s, mu_t = Xs.mean(0), Xt.mean(0)
    Cs = shrink(torch.cov(Xs.T), alpha)
    Ct = shrink(torch.cov(Xt.T), alpha)
    def msqrt(C):
        w, V = torch.linalg.eigh(C); w = w.clamp(min=1e-8)
        return V @ torch.diag(w.sqrt()) @ V.T
    def minv_sqrt(C):
        w, V = torch.linalg.eigh(C); w = w.clamp(min=1e-8)
        return V @ torch.diag(w.rsqrt()) @ V.T
    A = msqrt(Cs) @ minv_sqrt(Ct)
    def apply(X): return (X - mu_t) @ A.T + mu_s
    return apply

ap = argparse.ArgumentParser()
ap.add_argument('--generate', action='store_true')
ap.add_argument('--real', action='store_true', help='write REAL unseen route instead of probe constant')
ap.add_argument('--no-coral', action='store_true', help='disable CORAL channel (pure v20 baseline)')
ap.add_argument('--no-genus', action='store_true', help='disable genus fallback')
A = ap.parse_args()

if A.no_coral: CORAL_WEIGHT = 0.0
if A.no_genus: GENUS_FALLBACK_ENABLED = False

# ---- load data ----
txtH = torch.load('outputs/text_emb_h.pt', weights_only=False)
txtHt = torch.load('outputs/text_emb_h_taxon.pt', weights_only=False)
classes = txtH['classes']; ci = {c: i for i, c in enumerate(classes)}
TtH = F.normalize(txtHt['emb_taxon'].float(), dim=-1)
TnL = F.normalize(torch.load('outputs/text_emb.pt', weights_only=False)['emb_name'].float(), dim=-1)
lab = json.load(open('data/dl/label_train.json'))

FtI, FtF, Ftf = load('outputs/emb_train_ft.pt')
CtI, CtF, _ = load('outputs/emb_train_cap.pt')
LtI, LtF, _ = load('outputs/emb_train.pt')
trainfiles = [fn for fn in Ftf if fn in LtI and fn in CtI and fn in lab and lab[fn] in ci]
by = defaultdict(list)
for fn in trainfiles: by[lab[fn]].append(fn)
seen = sorted(by.keys()); s2i = {c: i for i, c in enumerate(seen)}; S = len(seen)
TseenTax = torch.stack([TtH[ci[c]] for c in seen])

# ---- genus mapping ----
genus_per_class = json.load(open('outputs/genus_per_class.json'))
all_genera = sorted(set(genus_per_class.values()))
g2i = {g: i for i, g in enumerate(all_genera)}; G = len(all_genera)
species_to_genus_idx = torch.zeros(S, dtype=torch.long)
for i, c in enumerate(seen):
    species_to_genus_idx[i] = g2i[genus_per_class.get(c, 'Unknown')]
# Build genus→species mapping for fallback
genus_to_species = defaultdict(list)
for i, c in enumerate(seen):
    genus_to_species[species_to_genus_idx[i].item()].append(i)

def protos_and_train(TrI, TrF, fns_by_cls, order):
    P = torch.zeros(S, TrF.shape[1]); cnt = torch.zeros(S); TF = []; TL = []
    for c in order:
        for fn in fns_by_cls[c]:
            f = TrF[TrI[fn]]; P[s2i[c]] += f; cnt[s2i[c]] += 1; TF.append(f); TL.append(s2i[c])
    P = F.normalize(P / cnt.clamp(min=1).unsqueeze(1), dim=-1); return P, torch.stack(TF), torch.tensor(TL)

def seen_score(qF, P, TF, TL, Tx=None, lam=0.0):
    out = torch.empty(qF.shape[0], S)
    for i in range(0, qF.shape[0], 1000):
        e = qF[i:i + 1000]; ps = e @ P.t(); sim = e @ TF.t()
        cmax = torch.full((e.shape[0], S), -1e9); cmax.scatter_reduce_(1, TL.unsqueeze(0).expand(e.shape[0], -1), sim, reduce='amax')
        sc = ps + 2.0 * cmax
        if Tx is not None and lam > 0: sc = sc + lam * (e @ Tx.t())
        out[i:i + 1000] = sc
    return out

def genus_fallback(sc, top_k=3):
    """For low-confidence predictions, fall back to best genus via species consensus."""
    topk_vals, topk_idx = sc.topk(top_k, dim=1)
    best_species = topk_idx[:, 0]
    best_score = topk_vals[:, 0]
    # Threshold: if max score is below GENUS_THRESHOLD*fraction of top score, use genus
    # Actually: if the top-1 confidence is low relative to top-3 spread, fall back
    margin = (topk_vals[:, 0] - topk_vals[:, 1]) / (topk_vals[:, 0].abs() + 1e-6)
    threshold = 0.05  # if margin < 5%, use genus
    fallback_mask = margin < threshold
    
    if fallback_mask.any():
        # For each fallback: predict genus by majority vote among top-3 species
        for i in range(sc.shape[0]):
            if fallback_mask[i]:
                top_genus_idxs = species_to_genus_idx[topk_idx[i]]
                # Most common genus among top-k
                unique_g, counts = torch.unique(top_genus_idxs, return_counts=True)
                best_g = unique_g[counts.argmax()]
                # Pick highest-scoring species in that genus
                in_genus = species_to_genus_idx == best_g
                # Among species in this genus, pick the one with highest score
                genus_scores = sc[i].clone()
                genus_scores[~in_genus] = -1e9
                best_species[i] = genus_scores.argmax()
    return best_species

# ---- holdout check (v20 baseline + CORAL) ----
trby = defaultdict(list); valrows = []
for c in seen:
    fns = sorted(by[c])
    if len(fns) >= 3:
        k = max(1, round(0.2 * len(fns)))
        for f in fns[:-k]: trby[c].append(f)
        for f in fns[-k:]: valrows.append((f, s2i[c]))
    else:
        for f in fns: trby[c].append(f)
PFh, TFFh, TLFh = protos_and_train(FtI, FtF, trby, seen)
PCh, TFCh, TLCh = protos_and_train(CtI, CtF, trby, seen)
PLh, TFLh, TLLh = protos_and_train(LtI, LtF, trby, seen)
VY = torch.tensor([y for _, y in valrows])
vF = torch.stack([FtF[FtI[f]] for f, _ in valrows])
vC = torch.stack([CtF[CtI[f]] for f, _ in valrows])
vL = torch.stack([LtF[LtI[f]] for f, _ in valrows])

# CORAL: align val (pseudo-test) to train distribution
train_ft_feats = FtF[[FtI[f] for f in trainfiles]]
val_ft_files = [f for f, _ in valrows]
val_ft_feats = FtF[[FtI[f] for f in val_ft_files]]
apply_coral = coral_transform(train_ft_feats, val_ft_feats)
vF_coral = F.normalize(apply_coral(vF), dim=-1)

sF_bl = seen_score(vF, PFh, TFFh, TLFh, TseenTax, LAM)
sC_bl = seen_score(vC, PCh, TFCh, TLCh, TseenTax, LAM)
sL_bl = seen_score(vL, PLh, TFLh, TLLh)
sF_coral = seen_score(vF_coral, PFh, TFFh, TLFh, TseenTax, LAM)

# v20 baseline
v20 = (zc(sF_bl) + 1.0*zc(sC_bl) + 0.5*zc(sL_bl)).argmax(1).eq(VY).float().mean().item()*100
# v21: + CORAL channel
v21_scores = zc(sF_bl) + 1.0*zc(sC_bl) + 0.5*zc(sL_bl) + CORAL_WEIGHT*zc(sF_coral)
v21_preds = v21_scores.argmax(1)
v21_base = v21_preds.eq(VY).float().mean().item()*100

# v21 + genus fallback
if GENUS_FALLBACK_ENABLED:
    v21_gf_preds = genus_fallback(v21_scores)
    v21_gf = v21_gf_preds.eq(VY).float().mean().item()*100
else:
    v21_gf = v21_base

print(f'v20 baseline (ft+1.0cap+0.5L):          {v20:.2f}% (expect ~87.94)')
print(f'v21 +CORAL(w={CORAL_WEIGHT}):              {v21_base:.2f}% (delta {v21_base-v20:+.2f})')
print(f'v21 +CORAL+genus_fallback:              {v21_gf:.2f}% (delta {v21_gf-v20:+.2f})')
print(f'\nGO/NO-GO: {"GO" if v21_gf > v20 else "NO-GO"} ({"beats" if v21_gf > v20 else "does not beat"} v20)')

# ---- generate (real) ----
if A.generate:
    PF, TFF, TLF = protos_and_train(FtI, FtF, by, seen)
    PC, TFC, TLC = protos_and_train(CtI, CtF, by, seen)
    PL, TFL, TLL = protos_and_train(LtI, LtF, by, seen)
    FteI, FteF, Ftef = load('outputs/emb_test_ft.pt')
    CteI, CteF, Ctef = load('outputs/emb_test_cap.pt')
    LteI, LteF, _ = load('outputs/emb_test.pt')
    tf = [fn for fn in Ftef if fn in LteI and fn in CteI]
    qF = torch.stack([FteF[FteI[fn]] for fn in tf])
    qC = torch.stack([CteF[CteI[fn]] for fn in tf])
    qL = torch.stack([LteF[LteI[fn]] for fn in tf])
    
    # CORAL transform on real test
    apply_test_coral = coral_transform(train_ft_feats, qF)
    qF_coral = F.normalize(apply_test_coral(qF), dim=-1)
    
    sc_blend = zc(seen_score(qF, PF, TFF, TLF, TseenTax, LAM)) + 1.0*zc(seen_score(qC, PC, TFC, TLC, TseenTax, LAM)) + 0.5*zc(seen_score(qL, PL, TFL, TLL))
    if CORAL_WEIGHT > 0:
        sc_blend = sc_blend + CORAL_WEIGHT*zc(seen_score(qF_coral, PF, TFF, TLF, TseenTax, LAM))
    
    if GENUS_FALLBACK_ENABLED:
        pred_idx = genus_fallback(sc_blend)
    else:
        pred_idx = sc_blend.argmax(1)
    
    preds = {fn: seen[k] for fn, k in zip(tf, pred_idx.tolist())}
    print(f'seen preds: {len(preds)} / {len(tf)} (expect 20097)')
    
    HunI, HunF, Hunf = load('outputs/emb_unseen_ctftbig.pt'); LunI, LunF, _ = load('outputs/emb_unseen.pt')
    uf = [fn for fn in Hunf if fn in LunI]
    if A.real:
        nonk = torch.tensor([i for i, c in enumerate(classes) if c not in set(seen)])
        uH = torch.stack([HunF[HunI[fn]] for fn in uf]); uL = torch.stack([LunF[LunI[fn]] for fn in uf])
        dis = lambda M, a, b: F.log_softmax(M / a, dim=0) + F.log_softmax(M / b, dim=1)
        MH = uH @ TtH[nonk].t(); ML = uL @ TnL[nonk].t()
        eU = dis(MH, 0.05, 0.5) + 0.5 * dis(ML, 0.05, 0.5)
        for fn, j in zip(uf, eU.argmax(1).tolist()): preds[fn] = classes[nonk[j].item()]
        tag = 'real'
    else:
        for fn in uf: preds[fn] = 'Ostracion cubicum'
        tag = 'probe'
    
    json.dump(preds, open(f'outputs/prediction_v21_improved_{tag}.json', 'w'))
    z = zipfile.ZipFile(f'outputs/submission_v21_improved_{tag}.zip', 'w', zipfile.ZIP_DEFLATED)
    z.write(f'outputs/prediction_v21_improved_{tag}.json', arcname='prediction.json'); z.close()
    print(f'wrote submission_v21_improved_{tag}.zip  entries={len(preds)}  mode={tag}')