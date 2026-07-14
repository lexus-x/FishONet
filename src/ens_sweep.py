import json, torch
from collections import defaultdict
import torch.nn.functional as F
torch.set_num_threads(8)
def load(p):
    d=torch.load(p,weights_only=False); return {fn:i for i,fn in enumerate(d['files'])}, F.normalize(d['feats'].float(),dim=-1)
def zc(M): return (M-M.mean())/(M.std()+1e-6)
lab=json.load(open('data/dl/label_train.json'))
srcs={n:load(p) for n,p in [('FTH','outputs/emb_train_ft.pt'),('FrH','outputs/emb_train_h_tta.pt'),('FrL','outputs/emb_train.pt')]}
common=[fn for fn in srcs['FTH'][0] if fn in srcs['FrH'][0] and fn in srcs['FrL'][0] and fn in lab]
by=defaultdict(list)
for fn in common: by[lab[fn]].append(fn)
seen=sorted(by.keys()); s2i={c:i for i,c in enumerate(seen)}; S=len(seen)
trby=defaultdict(list); val=[]
for c in seen:
    fns=sorted(by[c])
    if len(fns)>=3:
        k=max(1,round(0.2*len(fns)))
        for f in fns[:-k]: trby[c].append(f)
        for f in fns[-k:]: val.append((f,s2i[c]))
    else:
        for f in fns: trby[c].append(f)
VY=torch.tensor([y for _,y in val])
def seen_score(I,Ft):
    P=torch.zeros(S,Ft.shape[1]); cnt=torch.zeros(S); TF=[];TL=[]
    for c in seen:
        for fn in trby[c]:
            f=Ft[I[fn]]; P[s2i[c]]+=f; cnt[s2i[c]]+=1; TF.append(f); TL.append(s2i[c])
    P=F.normalize(P/cnt.clamp(min=1).unsqueeze(1),dim=-1); TF=torch.stack(TF); TL=torch.tensor(TL)
    q=torch.stack([Ft[I[f]] for f,_ in val]); out=torch.empty(q.shape[0],S)
    for i in range(0,q.shape[0],1000):
        e=q[i:i+1000]; ps=e@P.t(); sim=e@TF.t()
        cmax=torch.full((e.shape[0],S),-1e9); cmax.scatter_reduce_(1,TL.unsqueeze(0).expand(e.shape[0],-1),sim,reduce='amax')
        out[i:i+1000]=ps+2*cmax
    return out
SC={n:zc(seen_score(I,Ft)) for n,(I,Ft) in srcs.items()}
def acc(M): return (M.argmax(1)==VY).float().mean().item()*100
print('val=',len(VY))
print('FTH alone            :', round(acc(SC['FTH']),3))
print('FTH + 0.5*FrL  (v07) :', round(acc(SC['FTH']+0.5*SC['FrL']),3))
print('FrH alone (frozen)   :', round(acc(SC['FrH']),3))
best=(0,None)
for a in [0,0.15,0.25,0.4,0.6,0.85,1.2]:
    for b in [0,0.15,0.25,0.4,0.6]:
        ac=acc(SC['FTH']+a*SC['FrH']+b*SC['FrL'])
        if ac>best[0]: best=(ac,(a,b))
hb,ab=best; rs=hb*0.9065; ov=(20097*rs/100+15568*0.122)/35665*100
print(f'BEST 3-way: FTH + {ab[0]}*FrH + {ab[1]}*FrL = {round(hb,3)}%  (v07 base 86.17)')
print(f'  -> calibrated real seen ~{rs:.2f}%   overall ~{ov:.2f}%   (need 87.5 holdout / 79.3 real for 50.0)')
