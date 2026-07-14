import json, torch, time
from collections import defaultdict
import torch.nn.functional as F
dev='cuda'; t0=time.time()
def load(p):
    d=torch.load(p,weights_only=False); return {fn:i for i,fn in enumerate(d['files'])}, F.normalize(d['feats'].float(),dim=-1), list(d['files'])
txtH=torch.load('outputs/text_emb_h.pt',weights_only=False); classes=txtH['classes']; ci={c:i for i,c in enumerate(classes)}
TtH=F.normalize(torch.load('outputs/text_emb_h_taxon.pt',weights_only=False)['emb_taxon'].float(),dim=-1)
lab=json.load(open('data/dl/label_train.json'))
FtrI,FtrF,Ff=load('outputs/emb_train_ft.pt')      # FT-H (deployed)
LtrI,LtrF,_=load('outputs/emb_train.pt')          # L
ZtrI,ZtrF,_=load('outputs/emb_train_h_tta.pt')    # frozen H-TTA
DtrI,DtrF,_=load('outputs/emb_train_dino.pt')     # DINOv2
common=[fn for fn in Ff if fn in LtrI and fn in ZtrI and fn in DtrI and fn in lab and lab[fn] in ci]
by=defaultdict(list)
for fn in common: by[lab[fn]].append(fn)
seen=sorted(by.keys()); s2i={c:i for i,c in enumerate(seen)}; S=len(seen)
TseenTax=torch.stack([TtH[ci[c]] for c in seen]).to(dev)
trby=defaultdict(list); valrows=[]
for c in seen:
    fns=sorted(by[c])
    if len(fns)>=3:
        k=max(1,round(0.2*len(fns)))
        for f in fns[:-k]: trby[c].append(f)
        for f in fns[-k:]: valrows.append((f,s2i[c]))
    else:
        for f in fns: trby[c].append(f)
VY=torch.tensor([y for _,y in valrows]).to(dev)
print(f'seen={S} val={len(valrows)} [{time.time()-t0:.0f}s]',flush=True)
def proto_tr(TrI,TrF,order):
    P=torch.zeros(S,TrF.shape[1]); cnt=torch.zeros(S); TF=[]; TL=[]
    for c in order:
        for fn in trby[c]:
            f=TrF[TrI[fn]]; P[s2i[c]]+=f; cnt[s2i[c]]+=1; TF.append(f); TL.append(s2i[c])
    P=F.normalize(P/cnt.clamp(min=1).unsqueeze(1),dim=-1)
    return P.to(dev), torch.stack(TF).to(dev), torch.tensor(TL).to(dev)
def base_txt(vq,P,TF,TL,Tx):
    # returns base=ps+2cmax  and  txt=vq@Tx.t()  (separately, to sweep lam)
    out=torch.empty(vq.shape[0],S,device=dev)
    for i in range(0,vq.shape[0],2000):
        e=vq[i:i+2000]; ps=e@P.t(); sim=e@TF.t()
        cmax=torch.full((e.shape[0],S),-1e9,device=dev); cmax.scatter_reduce_(1,TL.unsqueeze(0).expand(e.shape[0],-1),sim,reduce='amax')
        out[i:i+2000]=ps+2.0*cmax
    txt=vq@Tx.t() if Tx is not None else None
    return out, txt
def zc(M): return (M-M.mean())/(M.std()+1e-6)
def acc(M): return (M.argmax(1)==VY).float().mean().item()*100
PF,TFF,TLF=proto_tr(FtrI,FtrF,seen); vqF=torch.stack([FtrF[FtrI[f]] for f,_ in valrows]).to(dev)
PL,TFL,TLL=proto_tr(LtrI,LtrF,seen); vqL=torch.stack([LtrF[LtrI[f]] for f,_ in valrows]).to(dev)
PZ,TFZ,TLZ=proto_tr(ZtrI,ZtrF,seen); vqZ=torch.stack([ZtrF[ZtrI[f]] for f,_ in valrows]).to(dev)
PD,TFD,TLD=proto_tr(DtrI,DtrF,seen); vqD=torch.stack([DtrF[DtrI[f]] for f,_ in valrows]).to(dev)
bF,tF=base_txt(vqF,PF,TFF,TLF,TseenTax)
bL,_ =base_txt(vqL,PL,TFL,TLL,None)
bZ,tZ=base_txt(vqZ,PZ,TFZ,TLZ,TseenTax)
bD,_ =base_txt(vqD,PD,TFD,TLD,None)
print(f'features ready [{time.time()-t0:.0f}s]',flush=True)
def sFT(lam): return bF+lam*tF
def sZ(lam):  return bZ+lam*tZ
# DEPLOYED baseline
dep=acc(zc(sFT(4.0))+0.5*zc(bL)); print(f'DEPLOYED v12  zc(FT+4txt)+0.5zc(L)        = {dep:.2f}',flush=True)
print('--- LAM sweep (FT textblend) ---')
for lam in (2,3,4,5,6,8): print(f'  lam={lam}: {acc(zc(sFT(lam))+0.5*zc(bL)):.2f}',flush=True)
print('--- L weight sweep (lam=4) ---')
for wL in (0.3,0.4,0.5,0.6,0.7): print(f'  wL={wL}: {acc(zc(sFT(4))+wL*zc(bL)):.2f}',flush=True)
print('--- + frozen H-TTA stream (lam=4, wL=0.5) ---')
for wZ in (0.25,0.5,0.75,1.0): print(f'  wZ={wZ}: {acc(zc(sFT(4))+0.5*zc(bL)+wZ*zc(sZ(4))):.2f}',flush=True)
print('--- + DINOv2 stream (lam=4, wL=0.5) ---')
for wD in (0.1,0.25,0.5,0.75): print(f'  wD={wD}: {acc(zc(sFT(4))+0.5*zc(bL)+wD*zc(bD)):.2f}',flush=True)
print('--- best combo grid (frozen + dino) ---')
best=(dep,'deployed')
for wZ in (0,0.25,0.5):
    for wD in (0,0.1,0.25):
        for lam in (4,5,6):
            a=acc(zc(sFT(lam))+0.5*zc(bL)+wZ*zc(sZ(lam))+wD*zc(bD))
            if a>best[0]: best=(a,f'lam={lam} wZ={wZ} wD={wD}')
print(f'BEST holdout = {best[0]:.2f}  [{best[1]}]  (deployed {dep:.2f}, +{best[0]-dep:.2f})',flush=True)
print(f'  -> projected real seen ~{best[0]*0.9065:.2f}% (deployed real ~{dep*0.9065:.2f}%)',flush=True)
print('DONE_SEENTUNE',flush=True)
