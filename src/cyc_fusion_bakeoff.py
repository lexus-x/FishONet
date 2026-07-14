import json, torch
from collections import defaultdict
import torch.nn.functional as F
torch.set_num_threads(8); LAM=4.0

def load(p):
    d=torch.load(p,weights_only=False)
    return {fn:i for i,fn in enumerate(d['files'])}, F.normalize(d['feats'].float(),dim=-1), list(d['files'])

def zc(M): return (M-M.mean())/(M.std()+1e-6)

txtHt=torch.load('outputs/text_emb_h_taxon.pt',weights_only=False)
txtH=torch.load('outputs/text_emb_h.pt',weights_only=False)
classes=txtH['classes']; ci={c:i for i,c in enumerate(classes)}
TtH=F.normalize(txtHt['emb_taxon'].float(),dim=-1)
lab=json.load(open('data/dl/label_train.json'))

FtI,FtF,Ftf=load('outputs/emb_train_ft.pt')
CtI,CtF,_=load('outputs/emb_train_cap.pt')
LtI,LtF,_=load('outputs/emb_train.pt')

trainfiles=[fn for fn in Ftf if fn in LtI and fn in CtI and fn in lab and lab[fn] in ci]
by=defaultdict(list)
for fn in trainfiles: by[lab[fn]].append(fn)
seen=sorted(by.keys()); s2i={c:i for i,c in enumerate(seen)}; S=len(seen)
TseenTax=torch.stack([TtH[ci[c]] for c in seen])

trby=defaultdict(list); valrows=[]
for c in seen:
    fns=sorted(by[c])
    if len(fns)>=3:
        k=max(1,round(0.2*len(fns)))
        for f in fns[:-k]: trby[c].append(f)
        for f in fns[-k:]: valrows.append((f,s2i[c]))
    else:
        for f in fns: trby[c].append(f)

def protos_and_train(TrI,TrF,order):
    P=torch.zeros(S,TrF.shape[1]); cnt=torch.zeros(S); TF=[]; TL=[]
    for c in order:
        for fn in trby[c]:
            f=TrF[TrI[fn]]; P[s2i[c]]+=f; cnt[s2i[c]]+=1; TF.append(f); TL.append(s2i[c])
    P=F.normalize(P/cnt.clamp(min=1).unsqueeze(1),dim=-1); return P,torch.stack(TF),torch.tensor(TL)

def seen_score(qF,P,TF,TL,Tx=None,lam=0.0):
    out=torch.empty(qF.shape[0],S)
    for i in range(0,qF.shape[0],1000):
        e=qF[i:i+1000]; ps=e@P.t(); sim=e@TF.t()
        cmax=torch.full((e.shape[0],S),-1e9); cmax.scatter_reduce_(1,TL.unsqueeze(0).expand(e.shape[0],-1),sim,reduce='amax')
        sc=ps+2.0*cmax
        if Tx is not None and lam>0: sc=sc+lam*(e@Tx.t())
        out[i:i+1000]=sc
    return out

PF,TFF,TLF=protos_and_train(FtI,FtF,seen)
PC,TFC,TLC=protos_and_train(CtI,CtF,seen)
PL,TFL,TLL=protos_and_train(LtI,LtF,seen)

VY=torch.tensor([y for _,y in valrows])
vF=torch.stack([FtF[FtI[f]] for f,_ in valrows]); vC=torch.stack([CtF[CtI[f]] for f,_ in valrows]); vL=torch.stack([LtF[LtI[f]] for f,_ in valrows])

sF=seen_score(vF,PF,TFF,TLF,TseenTax,LAM); sC=seen_score(vC,PC,TFC,TLC,TseenTax,LAM); sL=seen_score(vL,PL,TFL,TLL)

def a(M): return (M.argmax(1)==VY).float().mean().item()*100

base=a(zc(sF)+1.0*zc(sC)+0.5*zc(sL))
print(f"=== v20 baseline (score-fusion ft+1.0cap+0.5L): {base:.2f}%  (should ~= 87.94) ===",flush=True)

vFC = F.normalize(torch.cat([vF,vC],dim=1),dim=-1)
TFFC = F.normalize(torch.cat([TFF,TFC],dim=1),dim=-1)
TxC = F.normalize(torch.cat([TseenTax,TseenTax],dim=1),dim=-1)
PFC = F.normalize(torch.cat([PF,PC],dim=1),dim=-1)
sConcat = seen_score(vFC, PFC, TFFC, TLF, TxC, LAM)
concat_alone = a(sConcat)
print(f"(b) concat(ft,cap) alone: {concat_alone:.2f}%  (delta vs base {concat_alone-base:+.2f})",flush=True)

concat_plusL = a(zc(sConcat)+0.5*zc(sL))
print(f"(b) concat(ft,cap) + 0.5L: {concat_plusL:.2f}%  (delta vs base {concat_plusL-base:+.2f})",flush=True)

hybrid = a(zc(sConcat)+zc(zc(sF)+1.0*zc(sC))+0.5*zc(sL))
print(f"(hybrid) concat + score-fusion + 0.5L: {hybrid:.2f}%  (delta vs base {hybrid-base:+.2f})",flush=True)

mega = a(zc(sF)+zc(sC)+zc(sL))
print(f"(c) mega equal-weight(ft,cap,L): {mega:.2f}%  (delta vs base {mega-base:+.2f})",flush=True)

print(f"BAKEOFF_RESULT base={base:.2f} concat_alone={concat_alone:.2f} concat_plusL={concat_plusL:.2f} hybrid={hybrid:.2f} mega={mega:.2f}",flush=True)
