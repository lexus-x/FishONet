import json, torch
import torch.nn.functional as F
from collections import defaultdict
torch.set_num_threads(8)
def load(p):
    d=torch.load(p,weights_only=False)
    return {fn:i for i,fn in enumerate(d["files"])}, F.normalize(d["feats"].float(),dim=-1)
txtH=torch.load("outputs/text_emb_h.pt",weights_only=False)
classes=txtH["classes"]; ci={c:i for i,c in enumerate(classes)}
TnH=F.normalize(txtH["emb_name"].float(),dim=-1)
lab=json.load(open("data/dl/label_train.json"))
HtrI,HtrF=load("outputs/emb_train_h_tta.pt")
by=defaultdict(list)
for fn in HtrI:
    if fn in lab and lab[fn] in ci: by[lab[fn]].append(fn)
seen=sorted(by.keys())
order=sorted(seen,key=lambda c:len(by[c])); pseudo=set(order[:int(len(seen)*0.2)])
known=set(c for c in seen if c not in pseudo)
nonk=[i for i,c in enumerate(classes) if c not in known]
nt=torch.tensor(nonk); cpos={v:j for j,v in enumerate(nonk)}
qf=[f for c in pseudo for f in by[c]]
gold=torch.tensor([cpos[ci[lab[f]]] for f in qf])
Q=torch.stack([HtrF[HtrI[f]] for f in qf])
S=(Q@TnH[nt].t()).double()
print(f"queries={len(qf)} candidates={len(nonk)}")
def t1(M): return round((M.argmax(1)==gold).float().mean().item()*100,2)
print("raw cosine      :", t1(S))
zc=(S-S.mean(0,keepdim=True))/(S.std(0,keepdim=True)+1e-6)
print("z-score debias  :", t1(zc), "  <-- current method")
def sinkhorn(S,temp,iters):
    L=S/temp
    for _ in range(iters):
        L=L-torch.logsumexp(L,1,keepdim=True)
        L=L-torch.logsumexp(L,0,keepdim=True)
    return L
for temp in [0.01,0.02,0.04,0.08]:
    for it in [3,10,30]:
        print(f"sinkhorn temp={temp} iters={it:2d}:", t1(sinkhorn(S,temp,it)))
