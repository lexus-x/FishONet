# Reconcile: do common-name / neg-prompt channels beat the deployed unseen gate?
# Reuses PRE-BUILT text embeddings (no re-encode). Pseudo-unknown holdout, identical
# construction to cyc_commonname.py / cyc_negprompt.py.
import json, torch
import torch.nn.functional as F
from collections import defaultdict
torch.set_num_threads(8)

txtH=torch.load("outputs/text_emb_h.pt",weights_only=False)
classes=txtH["classes"]; ci={c:i for i,c in enumerate(classes)}
TtaxH=F.normalize(torch.load("outputs/text_emb_h_taxon.pt",weights_only=False)["emb_taxon"].float(),dim=-1)
TnL  =F.normalize(torch.load("outputs/text_emb.pt",weights_only=False)["emb_name"].float(),dim=-1)
Tcn  =F.normalize(torch.load("outputs/text_emb_h_common.pt",weights_only=False)["emb_common"].float(),dim=-1)
Tneg =F.normalize(torch.load("outputs/text_emb_h_neg.pt",weights_only=False)["emb_neg"].float(),dim=-1)
lab=json.load(open("data/dl/label_train.json"))
def loadfeat(p):
    d=torch.load(p,weights_only=False); return {fn:i for i,fn in enumerate(d["files"])}, F.normalize(d["feats"].float(),dim=-1)
HtrI,HtrF=loadfeat("outputs/emb_train_h_tta.pt"); LtrI,LtrF=loadfeat("outputs/emb_train.pt")
by=defaultdict(list)
for fn in HtrI:
    if fn in lab and lab[fn] in ci: by[lab[fn]].append(fn)
seen=sorted(by.keys()); order=sorted(seen,key=lambda c:len(by[c]))
pseudo=set(order[:int(len(seen)*0.2)]); known=set(c for c in seen if c not in pseudo)
nonk=[i for i,c in enumerate(classes) if c not in known]; nt=torch.tensor(nonk); cpos={v:j for j,v in enumerate(nonk)}
qfiles=[f for c in pseudo for f in by[c] if f in HtrI and f in LtrI]
gold=torch.tensor([cpos[ci[lab[f]]] for f in qfiles])
QH=torch.stack([HtrF[HtrI[f]] for f in qfiles]); QL=torch.stack([LtrF[LtrI[f]] for f in qfiles])
TtaxH_=TtaxH[nt]; TnL_=TnL[nt]; Tcn_=Tcn[nt]; Tneg_=Tneg[nt]
def t1(M): return (M.argmax(1)==gold).float().mean().item()*100
def zc(S): return (S-S.mean(0,keepdim=True))/(S.std(0,keepdim=True)+1e-6)
SHt=QH@TtaxH_.t(); SLn=QL@TnL_.t(); Scn=QH@Tcn_.t(); Sneg=QH@Tneg_.t()
C=t1(zc(SHt)+0.5*zc(SLn))
print(f"seen={len(seen)} pseudo(unknown)={len(pseudo)} queries={len(qfiles)} candidates={len(nonk)}")
print(f"=== DEPLOYED unseen gate C (ztaxon+0.5zL): {C:.2f}% ===")
print(f"common-only t1: {t1(zc(Scn)):.2f}%   neg-only t1: {t1(zc(Sneg)):.2f}%")
bc=(C,"none")
for w in [0.1,0.2,0.3,0.5,0.75,1.0]:
    a=t1(zc(SHt)+0.5*zc(SLn)+w*zc(Scn)); 
    if a>bc[0]: bc=(a,f"+{w}*common")
    print(f"  +{w:.2f}*common -> {a:.2f}%  (delta {a-C:+.2f})")
print(f"COMMON best={bc[0]:.2f}% [{bc[1]}] delta {bc[0]-C:+.2f}")
bn=(C,"none")
for w in [0.1,0.2,0.3,0.5,0.75,1.0]:
    a_sub=t1(zc(SHt)+0.5*zc(SLn)-w*zc(Sneg)); a_add=t1(zc(SHt)+0.5*zc(SLn)+w*zc(Sneg))
    if a_sub>bn[0]: bn=(a_sub,f"-{w}*neg")
    if a_add>bn[0]: bn=(a_add,f"+{w}*neg")
    print(f"  -{w:.2f}*neg -> {a_sub:.2f}% ({a_sub-C:+.2f})   +{w:.2f}*neg -> {a_add:.2f}% ({a_add-C:+.2f})")
print(f"NEG best={bn[0]:.2f}% [{bn[1]}] delta {bn[0]-C:+.2f}")
