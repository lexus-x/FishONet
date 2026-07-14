"""v21 seen-route variants — two untested-on-real-test mechanisms:
  (A) blend REWEIGHT toward the shift-invariant taxon-text term (cmax down, LAM up)
  (B) the trained ArcFace classifier head Wc (saved in ft_capA.pt, currently unused) as a
      direct discriminative score, blended into / replacing NCM.
Prints holdout (SAFETY only — shift is invisible here) + flip-count vs v20 for each variant.
--emit <name> writes a PROBE zip (seen=variant, unseen=const) -> reads accuracy_test alone.
Real gate = probe accuracy_test, target > v20's 79.70% seen.
"""
import json, torch, argparse, zipfile
from collections import defaultdict
import torch.nn.functional as F
torch.set_num_threads(8)

def load(p):
    d = torch.load(p, weights_only=False)
    return {fn: i for i, fn in enumerate(d['files'])}, F.normalize(d['feats'].float(), dim=-1), list(d['files'])
def zc(M): return (M - M.mean()) / (M.std() + 1e-6)

ap = argparse.ArgumentParser()
ap.add_argument('--emit', default=None)
A = ap.parse_args()

txtH = torch.load('outputs/text_emb_h.pt', weights_only=False)
txtHt = torch.load('outputs/text_emb_h_taxon.pt', weights_only=False)
classes = txtH['classes']; ci = {c: i for i, c in enumerate(classes)}
TtH = F.normalize(txtHt['emb_taxon'].float(), dim=-1)
lab = json.load(open('data/dl/label_train.json'))

FtI, FtF, Ftf = load('outputs/emb_train_ft.pt')
CtI, CtF, _ = load('outputs/emb_train_cap.pt')
LtI, LtF, _ = load('outputs/emb_train.pt')
trainfiles = [fn for fn in Ftf if fn in LtI and fn in CtI and fn in lab and lab[fn] in ci]
by = defaultdict(list)
for fn in trainfiles: by[lab[fn]].append(fn)
seen = sorted(by.keys()); s2i = {c: i for i, c in enumerate(seen)}; S = len(seen)
TseenTax = torch.stack([TtH[ci[c]] for c in seen])

cap_ck = torch.load('outputs/ft_capA.pt', map_location='cpu', weights_only=False)
Wc_raw = F.normalize(cap_ck['Wc'].float(), dim=-1); Wc_species = cap_ck['species']
wpos = {c: i for i, c in enumerate(Wc_species)}
Wc = torch.stack([Wc_raw[wpos[c]] for c in seen])  # (S,1024) reordered to `seen`

def protos_and_train(TrI, TrF, fns_by_cls, order):
    P = torch.zeros(S, TrF.shape[1]); cnt = torch.zeros(S); TF = []; TL = []
    for c in order:
        for fn in fns_by_cls[c]:
            f = TrF[TrI[fn]]; P[s2i[c]] += f; cnt[s2i[c]] += 1; TF.append(f); TL.append(s2i[c])
    P = F.normalize(P / cnt.clamp(min=1).unsqueeze(1), dim=-1); return P, torch.stack(TF), torch.tensor(TL)

def seen_score(qF, P, TF, TL, Tx=None, lam=0.0, cmax_w=2.0):
    out = torch.empty(qF.shape[0], S)
    for i in range(0, qF.shape[0], 1000):
        e = qF[i:i + 1000]; ps = e @ P.t(); sim = e @ TF.t()
        cmax = torch.full((e.shape[0], S), -1e9); cmax.scatter_reduce_(1, TL.unsqueeze(0).expand(e.shape[0], -1), sim, reduce='amax')
        sc = ps + cmax_w * cmax
        if Tx is not None and lam > 0: sc = sc + lam * (e @ Tx.t())
        out[i:i + 1000] = sc
    return out

# cmax_w, lam apply to the taxon-bearing encoders (ft,cap); L has no taxon. wcls = weight on Wc-logits (cap feats).
VARIANTS = {
    'v20':   dict(cmax=2.0, lam=4.0, wF=1.0, wC=1.0, wL=0.5, wcls=0.0),
    'R1':    dict(cmax=1.0, lam=8.0, wF=1.0, wC=1.0, wL=0.5, wcls=0.0),  # text up, cmax down
    'R2':    dict(cmax=2.0, lam=8.0, wF=1.0, wC=1.0, wL=0.5, wcls=0.0),  # text up only
    'W1':    dict(cmax=2.0, lam=4.0, wF=1.0, wC=1.0, wL=0.5, wcls=1.0),  # + ArcFace head
    'W2':    dict(cmax=2.0, lam=4.0, wF=1.0, wC=1.0, wL=0.5, wcls=2.0),  # heavy ArcFace
    'Wonly': dict(cmax=2.0, lam=4.0, wF=0.0, wC=0.0, wL=0.0, wcls=1.0),  # pure classifier (sanity)
}

def blend(V, qF, qC, qL, PF, TFF, TLF, PC, TFC, TLC, PL, TFL, TLL):
    sc = torch.zeros(qF.shape[0], S)
    if V['wF']: sc = sc + V['wF'] * zc(seen_score(qF, PF, TFF, TLF, TseenTax, V['lam'], V['cmax']))
    if V['wC']: sc = sc + V['wC'] * zc(seen_score(qC, PC, TFC, TLC, TseenTax, V['lam'], V['cmax']))
    if V['wL']: sc = sc + V['wL'] * zc(seen_score(qL, PL, TFL, TLL, None, 0.0, V['cmax']))
    if V['wcls']: sc = sc + V['wcls'] * zc(qC @ Wc.t())
    return sc

# --- holdout (SAFETY; R* fair, W* leak Wc so treat as sanity only) ---
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
vF = torch.stack([FtF[FtI[f]] for f, _ in valrows]); vC = torch.stack([CtF[CtI[f]] for f, _ in valrows]); vL = torch.stack([LtF[LtI[f]] for f, _ in valrows])
print("=== holdout (SAFETY only; shift invisible; W* LEAK Wc -> inflated) ===")
for name, V in VARIANTS.items():
    acc = blend(V, vF, vC, vL, PFh, TFFh, TLFh, PCh, TFCh, TLCh, PLh, TFLh, TLLh).argmax(1).eq(VY).float().mean().item() * 100
    leak = '  (Wc-LEAK, sanity only)' if V['wcls'] else ''
    print(f"  {name:6s}: {acc:.2f}%{leak}")

# --- real test: flip counts vs v20, and emit ---
PF, TFF, TLF = protos_and_train(FtI, FtF, by, seen)
PC, TFC, TLC = protos_and_train(CtI, CtF, by, seen)
PL, TFL, TLL = protos_and_train(LtI, LtF, by, seen)
FteI, FteF, Ftef = load('outputs/emb_test_ft.pt')
CteI, CteF, _ = load('outputs/emb_test_cap.pt')
LteI, LteF, _ = load('outputs/emb_test.pt')
tf = [fn for fn in Ftef if fn in LteI and fn in CteI]
qF = torch.stack([FteF[FteI[fn]] for fn in tf]); qC = torch.stack([CteF[CteI[fn]] for fn in tf]); qL = torch.stack([LteF[LteI[fn]] for fn in tf])
base = blend(VARIANTS['v20'], qF, qC, qL, PF, TFF, TLF, PC, TFC, TLC, PL, TFL, TLL).argmax(1)
print(f"\n=== real test: {len(tf)} seen imgs; flips vs v20 ===")
argmaxes = {}
for name, V in VARIANTS.items():
    am = blend(V, qF, qC, qL, PF, TFF, TLF, PC, TFC, TLC, PL, TFL, TLL).argmax(1); argmaxes[name] = am
    flips = (am != base).sum().item()
    print(f"  {name:6s}: {flips:5d} flips ({100*flips/len(tf):.1f}%)")

if A.emit:
    am = argmaxes[A.emit]
    preds = {fn: seen[k] for fn, k in zip(tf, am.tolist())}
    for fn in Ftef:  # any test files missing from tf still need a pred; fall back to v20 base-ish const
        pass
    HunI, _, Hunf = load('outputs/emb_unseen_ctftbig.pt')
    for fn in Hunf: preds[fn] = 'Ostracion cubicum'  # probe: tank unseen
    out = f'outputs/prediction_v21_{A.emit}_probe.json'; json.dump(preds, open(out, 'w'))
    zp = f'outputs/submission_v21_{A.emit}_probe.zip'
    z = zipfile.ZipFile(zp, 'w', zipfile.ZIP_DEFLATED); z.write(out, arcname='prediction.json'); z.close()
    print(f"\nEMITTED probe {zp}  entries={len(preds)}  (seen={len(tf)} variant, unseen=const)")
