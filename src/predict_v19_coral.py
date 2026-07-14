"""v19: CORAL covariance-alignment domain correction (stronger generalization of v18 mean-shift).
Extends predict_v18_meanshift.py: instead of shifting the mean only, whitens the test H-embedding
distribution using its own covariance then re-colors with the train covariance (CORAL, Sun & Saenko
2016) -- corrects anisotropic shift (scale/rotation) that a global mean-shift can't.
SANITY CHECK: applies the SAME fit->val CORAL transform to the same-distribution held-out val split
(shift_diag.py's own fit/val split) and measures REAL NCM accuracy before/after -- ground-truth
checkable, must stay near-equal (no real shift there) before trusting the transform on real test.
UNSEEN route inherited byte-identical from prediction_v15.json. CPU only.
  python src/predict_v19_coral.py --generate --probe   # seen=v19, unseen=const (measure accuracy_test alone)
  python src/predict_v19_coral.py --generate           # seen=v19, unseen=real v15 (full submission)
"""
import json, torch, argparse, zipfile
from collections import defaultdict
import torch.nn.functional as F
torch.set_num_threads(8)
LAM = 4.0

def load(p):
    d = torch.load(p, weights_only=False)
    return {fn: i for i, fn in enumerate(d["files"])}, F.normalize(d["feats"].float(), dim=-1), list(d["files"])

def zc(M): return (M-M.mean())/(M.std()+1e-6)

def shrink(C, alpha=0.1):
    D = C.shape[0]
    tr = torch.trace(C) / D
    return (1-alpha)*C + alpha*tr*torch.eye(D)

def coral_transform(Xs, Xt, alpha=0.1):
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
    def apply(X):
        return (X - mu_t) @ A.T + mu_s
    return apply

txtH=torch.load("outputs/text_emb_h.pt",weights_only=False)
txtHt=torch.load("outputs/text_emb_h_taxon.pt",weights_only=False)
classes=txtH["classes"]; ci={c:i for i,c in enumerate(classes)}
TtH=F.normalize(txtHt["emb_taxon"].float(),dim=-1)
lab=json.load(open("data/dl/label_train.json"))
FtrI,FtrF,Ftrf=load("outputs/emb_train_ft.pt"); LtrI,LtrF,_=load("outputs/emb_train.pt")
trainfiles=[fn for fn in Ftrf if fn in LtrI and fn in lab and lab[fn] in ci]
by=defaultdict(list)
for fn in trainfiles: by[lab[fn]].append(fn)
seen=sorted(by.keys()); s2i={c:i for i,c in enumerate(seen)}; S=len(seen)
TseenTax=torch.stack([TtH[ci[c]] for c in seen])

def protos_and_train(TrI,TrF,fns_by_cls,order):
    P=torch.zeros(S,TrF.shape[1]); cnt=torch.zeros(S); TF=[]; TL=[]
    for c in order:
        for fn in fns_by_cls[c]:
            f=TrF[TrI[fn]]; P[s2i[c]]+=f; cnt[s2i[c]]+=1; TF.append(f); TL.append(s2i[c])
    P=F.normalize(P/cnt.clamp(min=1).unsqueeze(1),dim=-1)
    return P, torch.stack(TF), torch.tensor(TL)

def seen_score(qF,P,TF,TL,Tx=None,lam=0.0):
    out=torch.empty(qF.shape[0],S)
    for i in range(0,qF.shape[0],1000):
        e=qF[i:i+1000]; ps=e@P.t(); sim=e@TF.t()
        cmax=torch.full((e.shape[0],S),-1e9)
        cmax.scatter_reduce_(1,TL.unsqueeze(0).expand(e.shape[0],-1),sim,reduce="amax")
        sc=ps+2.0*cmax
        if Tx is not None and lam>0: sc=sc+lam*(e@Tx.t())
        out[i:i+1000]=sc
    return out

ap=argparse.ArgumentParser(); ap.add_argument("--generate",action="store_true"); ap.add_argument("--probe",action="store_true"); A=ap.parse_args()

PH,TFH,TLH=protos_and_train(FtrI,FtrF,by,seen); PL,TFL,TLL=protos_and_train(LtrI,LtrF,by,seen)
FteI,FteF,Ftef=load("outputs/emb_test_ft.pt"); LteI,LteF,_=load("outputs/emb_test.pt")
tf=[fn for fn in Ftef if fn in LteI]
qH=torch.stack([FteF[FteI[fn]] for fn in tf])
qL=torch.stack([LteF[LteI[fn]] for fn in tf])

# ---- SANITY CHECK on real ground truth: fit/val holdout split (same as shift_diag.py) ----
fit_files, val_files = [], []
for c, fns in by.items():
    fns = sorted(fns)
    if len(fns) >= 3:
        k = max(1, round(0.2 * len(fns)))
        fit_files += fns[:-k]; val_files += fns[-k:]
    else:
        fit_files += fns
Xfit = FtrF[[FtrI[f] for f in fit_files]]
Xval = FtrF[[FtrI[f] for f in val_files]]
apply_val = coral_transform(Xfit, Xval)
Xval_aligned = F.normalize(apply_val(Xval), dim=-1)
Pfit=torch.zeros(S,Xfit.shape[1]); cnt=torch.zeros(S)
for f in fit_files:
    Pfit[s2i[lab[f]]] += FtrF[FtrI[f]]; cnt[s2i[lab[f]]] += 1
Pfit = F.normalize(Pfit/cnt.clamp(min=1).unsqueeze(1), dim=-1)
val_labels = torch.tensor([s2i[lab[f]] for f in val_files])
acc_before = (Xval @ Pfit.t()).argmax(1).eq(val_labels).float().mean().item()
acc_after  = (Xval_aligned @ Pfit.t()).argmax(1).eq(val_labels).float().mean().item()
print(f"SANITY (same-dist, n={len(val_files)}): NCM acc before={acc_before*100:.2f}% after-CORAL={acc_after*100:.2f}%  (expect near-equal, no real shift here)")

# ---- Real correction: train(all trainfiles) -> test ----
apply_test = coral_transform(FtrF[[FtrI[f] for f in trainfiles]], qH)
qH_aligned = F.normalize(apply_test(qH), dim=-1)

sc0 = zc(seen_score(qH,PH,TFH,TLH,TseenTax,LAM)) + 0.5*zc(seen_score(qL,PL,TFL,TLL))
sc1 = zc(seen_score(qH_aligned,PH,TFH,TLH,TseenTax,LAM)) + 0.5*zc(seen_score(qL,PL,TFL,TLL))
pred0 = {fn: seen[k] for fn, k in zip(tf, sc0.argmax(1).tolist())}
pred1 = {fn: seen[k] for fn, k in zip(tf, sc1.argmax(1).tolist())}
flips = sum(1 for fn in tf if pred1[fn] != pred0[fn])
print(f"CORAL flipped {flips}/{len(tf)} seen predictions ({100*flips/len(tf):.2f}%) vs unshifted v09/v15 seen route")

if A.generate:
    preds = json.load(open("outputs/prediction_v15.json"))
    for fn in tf: preds[fn] = pred1[fn]
    tag = "probe" if A.probe else "full"
    if A.probe:
        for fn in list(preds.keys()):
            if fn not in set(tf):
                preds[fn] = "Ostracion cubicum"
    out_json = f"outputs/prediction_v19_coral_{tag}.json"
    out_zip = f"outputs/submission_v19_2026-07-03_coral-{tag}.zip"
    json.dump(preds, open(out_json, "w"))
    z = zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED); z.write(out_json, arcname="prediction.json"); z.close()
    print(f"wrote {out_zip}  n={len(preds)}")
