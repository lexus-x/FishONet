"""Own-eval: calibrated real-performance estimate (no submission needed).
Multi-seed sim of the FINAL recipe + calibration anchored on the ONE real result we have
(ViT-L submission: real seen 66.5 / unseen 6.5 ; our ViT-L sim was seen 65.6 / unseen 13.5)."""
import json, torch, random
from collections import defaultdict
import numpy as np
import torch.nn.functional as F
dev="cuda" if torch.cuda.is_available() else "cpu"
def load(p):
    d=torch.load(p,weights_only=False); return {fn:i for i,fn in enumerate(d["files"])}, F.normalize(d["feats"].float(),dim=-1).to(dev), list(d["files"])
def zc(M): return (M-M.mean())/(M.std()+1e-6)
def db(M): return (M-M.mean(0,keepdim=True))/(M.std(0,keepdim=True)+1e-6)
txtH=torch.load("outputs/text_emb_h.pt",weights_only=False); txtL=torch.load("outputs/text_emb.pt",weights_only=False)
classes=txtH["classes"]; ci={c:i for i,c in enumerate(classes)}
TnH=F.normalize(txtH["emb_name"].float(),dim=-1).to(dev); TnL=F.normalize(txtL["emb_name"].float(),dim=-1).to(dev)
lab=json.load(open("data/dl/label_train.json"))
HtrI,HtrF,Htrf=load("outputs/emb_train_h_tta.pt"); LtrI,LtrF,_=load("outputs/emb_train.pt")
trainfiles=[fn for fn in Htrf if fn in LtrI and fn in lab and lab[fn] in ci]
by=defaultdict(list)
for fn in trainfiles: by[lab[fn]].append(fn)
seen=sorted(by.keys()); s2i={c:i for i,c in enumerate(seen)}; S=len(seen)
def build(TrI,TrF,trby,order):
    P=torch.zeros(S,TrF.shape[1],device=dev); cnt=torch.zeros(S,device=dev); TF=[]; TL=[]
    for c in order:
        for fn in trby[c]:
            f=TrF[TrI[fn]]; P[s2i[c]]+=f; cnt[s2i[c]]+=1; TF.append(f); TL.append(s2i[c])
    return F.normalize(P/cnt.clamp(min=1).unsqueeze(1),dim=-1), torch.stack(TF), torch.tensor(TL,device=dev)
def sscore(qF,P,TF,TL):
    out=torch.empty(qF.shape[0],S,device=dev)
    for i in range(0,qF.shape[0],2000):
        e=qF[i:i+2000]; ps=e@P.t(); sim=e@TF.t()
        cm=torch.full((e.shape[0],S),-1e9,device=dev); cm.scatter_reduce_(1,TL.unsqueeze(0).expand(e.shape[0],-1),sim,reduce="amax")
        out[i:i+2000]=ps+2.0*cm
    return out
def seen_eval(seed):
    g=random.Random(seed); trby=defaultdict(list); val=[]
    cls2=[c for c in seen if len(by[c])>=2]
    for c in cls2:
        fns=by[c][:]; g.shuffle(fns); k=max(1,round(0.2*len(fns)))
        for f in fns[k:]: trby[c].append(f)
        for f in fns[:k]: val.append((f,s2i[c]))
    PH,TFH,TLH=build(HtrI,HtrF,trby,cls2); PL,TFL,TLL=build(LtrI,LtrF,trby,cls2)
    VY=torch.tensor([y for _,y in val],device=dev)
    qH=torch.stack([HtrF[HtrI[f]] for f,_ in val]); qL=torch.stack([LtrF[LtrI[f]] for f,_ in val])
    return ((zc(sscore(qH,PH,TFH,TLH))+0.5*zc(sscore(qL,PL,TFL,TLL))).argmax(1)==VY).float().mean().item()*100
def unseen_eval(seed):
    g=random.Random(seed); order=sorted(seen,key=lambda c:len(by[c]))
    rare=order[:int(len(seen)*0.4)][:]; g.shuffle(rare); pseudo=set(rare[:int(len(seen)*0.2)])
    nonk_classes=set(c for c in seen if c not in pseudo)
    nonk=[i for i,c in enumerate(classes) if c not in nonk_classes]
    nt=torch.tensor(nonk,device=dev); cpos={x:j for j,x in enumerate(nonk)}
    qf=[f for c in pseudo for f in by[c]]; gold=torch.tensor([cpos[ci[lab[f]]] for f in qf],device=dev)
    uH=torch.stack([HtrF[HtrI[f]] for f in qf]); uL=torch.stack([LtrF[LtrI[f]] for f in qf])
    return ((db(uH@TnH[nt].t())+0.5*db(uL@TnL[nt].t())).argmax(1)==gold).float().mean().item()*100
seeds=[0,1,2,3,4]
se=[seen_eval(s) for s in seeds]; ue=[unseen_eval(s) for s in seeds]
print(f"FINAL recipe SIM (5 seeds):")
print(f"  SEEN  : {np.mean(se):.2f} +/- {np.std(se):.2f}   {[round(x,1) for x in se]}")
print(f"  UNSEEN: {np.mean(ue):.2f} +/- {np.std(ue):.2f}   {[round(x,1) for x in ue]}")
# calibration anchor (ViT-L real submission)
ks, ku = 66.5/65.6, 6.5/13.5
cs, cu = np.mean(se)*ks, np.mean(ue)*ku
print(f"\nCALIBRATION anchor (ViT-L real): seen x{ks:.3f}  unseen x{ku:.3f}")
print(f"CALIBRATED REAL ESTIMATE:")
print(f"  SEEN  ~{cs:.1f}%   UNSEEN ~{cu:.1f}%   OVERALL ~{(20097*cs+15568*cu)/35665:.1f}%")
print(f"  (range w/ unseen +/-30%: {(20097*cs+15568*cu*0.7)/35665:.1f}% - {(20097*cs+15568*cu*1.3)/35665:.1f}%)")
