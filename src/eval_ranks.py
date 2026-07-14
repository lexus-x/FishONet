"""UNSEEN accuracy by taxonomic rank (species / genus / family) on hard sim, final recipe."""
import json, torch
from collections import defaultdict
import torch.nn.functional as F
dev="cuda" if torch.cuda.is_available() else "cpu"
def load(p): d=torch.load(p,weights_only=False); return {fn:i for i,fn in enumerate(d["files"])}, F.normalize(d["feats"].float(),dim=-1).to(dev), list(d["files"])
def db(M): return (M-M.mean(0,keepdim=True))/(M.std(0,keepdim=True)+1e-6)
txtH=torch.load("outputs/text_emb_h.pt",weights_only=False); txtL=torch.load("outputs/text_emb.pt",weights_only=False)
classes=txtH["classes"]; ci={c:i for i,c in enumerate(classes)}
TnH=F.normalize(txtH["emb_name"].float(),dim=-1).to(dev); TnL=F.normalize(txtL["emb_name"].float(),dim=-1).to(dev)
lab=json.load(open("data/dl/label_train.json"))
tax=json.load(open("outputs/taxonomy.json"))
k0=next(iter(tax)); print("taxonomy sample:",k0,"->",tax[k0])
def fam(sp):
    t=tax.get(sp); 
    return (t.get("family") if isinstance(t,dict) else t) if t else None
gen=lambda sp: sp.split()[0]
HtrI,HtrF,Htrf=load("outputs/emb_train_h_tta.pt"); LtrI,LtrF,_=load("outputs/emb_train.pt")
trf=[fn for fn in Htrf if fn in LtrI and fn in lab and lab[fn] in ci]
by=defaultdict(list)
for fn in trf: by[lab[fn]].append(fn)
seen=sorted(by.keys()); order=sorted(seen,key=lambda c:len(by[c])); pseudo=set(order[:int(len(seen)*0.2)])
nonk=[i for i,c in enumerate(classes) if c not in set(c for c in seen if c not in pseudo)]
nt=torch.tensor(nonk,device=dev); cn=[classes[i] for i in nonk]
qf=[f for c in pseudo for f in by[c]]; gold=[lab[f] for f in qf]
uH=torch.stack([HtrF[HtrI[f]] for f in qf]); uL=torch.stack([LtrF[LtrI[f]] for f in qf])
pid=(db(uH@TnH[nt].t())+0.5*db(uL@TnL[nt].t())).argmax(1).cpu().tolist()
pred=[cn[j] for j in pid]
n=len(qf)
sp=sum(p==g for p,g in zip(pred,gold))/n*100
gn=sum(gen(p)==gen(g) for p,g in zip(pred,gold))/n*100
fp=[(fam(p),fam(g)) for p,g in zip(pred,gold)]; cov=sum(1 for _,b in fp if b)
fm=sum(1 for a,b in fp if a==b and b)/n*100
print(f"\nUNSEEN hard-sim (n={n}) accuracy by rank:")
print(f"  species: {sp:.1f}%   genus: {gn:.1f}%   family: {fm:.1f}%  (family coverage {cov}/{n})")
