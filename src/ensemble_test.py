"""Test ViT-H + w*ViT-L score ensemble on BOTH routes (hard sim). Aligned by filename."""
import json, torch
from collections import defaultdict
import torch.nn.functional as F

txtL=torch.load("outputs/text_emb.pt",weights_only=False)
txtH=torch.load("outputs/text_emb_h.pt",weights_only=False)
classes=txtH["classes"]; cls_index={c:i for i,c in enumerate(classes)}
TnL=F.normalize(txtL["emb_name"].float(),dim=-1); TnH=F.normalize(txtH["emb_name"].float(),dim=-1)
trL=torch.load("outputs/emb_train.pt",weights_only=False); trH=torch.load("outputs/emb_train_h.pt",weights_only=False)
L={fn:i for i,fn in enumerate(trL["files"])}; H={fn:i for i,fn in enumerate(trH["files"])}
FL=F.normalize(trL["feats"].float(),dim=-1); FH=F.normalize(trH["feats"].float(),dim=-1)
lab=json.load(open("data/dl/label_train.json"))
common=[fn for fn in trH["files"] if fn in L and fn in lab and lab[fn] in cls_index]
by=defaultdict(list)
for fn in common: by[lab[fn]].append(fn)
seen=sorted(by.keys())
def z(M): return (M-M.mean())/(M.std()+1e-6)

# ---- SEEN: per-class holdout, proto+2cmax per backbone ----
cls_list=[c for c in seen if len(by[c])>=2]; c2i={c:i for i,c in enumerate(cls_list)}
def seen_build(Feat, idx):
    proto=[]; TFE=[]; TL=[]; valrows=[]
    for c in cls_list:
        fns=by[c]; k=min(max(1,round(0.2*len(fns))),len(fns)-1)
        trn=fns[:-k]; val=fns[-k:]
        pf=torch.stack([Feat[idx[f]] for f in trn])
        proto.append(F.normalize(pf.mean(0),dim=0))
        for f in trn: TFE.append(Feat[idx[f]]); TL.append(c2i[c])
        for f in val: valrows.append((f,c2i[c]))
    return torch.stack(proto), torch.stack(TFE), torch.tensor(TL), valrows
PL,TFEL,TLL,valrows=seen_build(FL,L)
PH,TFEH,TLH,_=seen_build(FH,H)
VY=torch.tensor([y for _,y in valrows])
VFL=torch.stack([FL[L[f]] for f,_ in valrows]); VFH=torch.stack([FH[H[f]] for f,_ in valrows])
def seen_score(VF,P,TFE,TL):
    ps=VF@P.t(); sim=VF@TFE.t()
    cmax=torch.full((len(VF),len(cls_list)),-1e9); cmax.scatter_reduce_(1,TL.unsqueeze(0).expand(len(VF),-1),sim,reduce="amax")
    return ps+2.0*cmax
SH=seen_score(VFH,PH,TFEH,TLH); SL=seen_score(VFL,PL,TFEL,TLL)
acc=lambda s:round((s.argmax(1)==VY).float().mean().item()*100,2)
print("SEEN  H only:",acc(SH),"  L only:",acc(SL))
for w in [0.0,0.2,0.3,0.5,1.0]:
    print(f"  SEEN ens z(H)+{w}*z(L):",acc(z(SH)+w*z(SL)))

# ---- UNSEEN: hard split, debiased name per backbone ----
order=sorted(seen,key=lambda c:len(by[c])); pseudo=set(order[:int(len(seen)*0.2)])
known_idx=set(cls_index[c] for c in seen if c not in pseudo)
cand=[i for i in range(len(classes)) if i not in known_idx]; ct=torch.tensor(cand); cpos={ci:j for j,ci in enumerate(cand)}
qfns=[f for c in pseudo for f in by[c]]; gold=torch.tensor([cpos[cls_index[lab[f]]] for f in qfns])
QFL=F.normalize(torch.stack([FL[L[f]] for f in qfns]),dim=-1); QFH=F.normalize(torch.stack([FH[H[f]] for f in qfns]),dim=-1)
def db(S): return (S-S.mean(0,keepdim=True))/(S.std(0,keepdim=True)+1e-6)
UH=db(QFH@TnH[ct].t()); UL=db(QFL@TnL[ct].t())
au=lambda s:round((s.argmax(1)==gold).float().mean().item()*100,2)
print("UNSEEN H only:",au(UH),"  L only:",au(UL))
for w in [0.0,0.2,0.3,0.5,1.0]:
    print(f"  UNSEEN ens H+{w}*L:",au(UH+w*UL))
