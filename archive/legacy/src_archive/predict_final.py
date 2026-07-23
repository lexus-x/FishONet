"""FINAL integrated recipe. --generate to write submission; default = sim validation.
SEEN  : ensemble zc(H_blend)+0.5*zc(L_blend), blend = proto_sim + 2*class_max_sim
UNSEEN: ensemble db(H_name)+0.5*db(L_name), db = z-score column debias
H = ViT-H(TTA), L = ViT-L."""
import json, torch, argparse
from collections import defaultdict
import torch.nn.functional as F
dev="cuda" if torch.cuda.is_available() else "cpu"

def load(p):
    d=torch.load(p,weights_only=False)
    return {fn:i for i,fn in enumerate(d["files"])}, F.normalize(d["feats"].float(),dim=-1), list(d["files"])
def zc(M): return (M-M.mean())/(M.std()+1e-6)
def db(M): return (M-M.mean(0,keepdim=True))/(M.std(0,keepdim=True)+1e-6)

txtH=torch.load("outputs/text_emb_h.pt",weights_only=False); txtL=torch.load("outputs/text_emb.pt",weights_only=False)
classes=txtH["classes"]; ci={c:i for i,c in enumerate(classes)}
TnH=F.normalize(txtH["emb_name"].float(),dim=-1); TnL=F.normalize(txtL["emb_name"].float(),dim=-1)
lab=json.load(open("data/dl/label_train.json"))
HtrI,HtrF,Htrf=load("outputs/emb_train_h_tta.pt"); LtrI,LtrF,_=load("outputs/emb_train.pt")
trainfiles=[fn for fn in Htrf if fn in LtrI and fn in lab and lab[fn] in ci]
by=defaultdict(list)
for fn in trainfiles: by[lab[fn]].append(fn)
seen=sorted(by.keys()); s2i={c:i for i,c in enumerate(seen)}; S=len(seen)

def protos_and_train(TrI,TrF,fns_by_cls,order):
    P=torch.zeros(S,TrF.shape[1]); cnt=torch.zeros(S); TF=[]; TL=[]
    for c in order:
        for fn in fns_by_cls[c]:
            f=TrF[TrI[fn]]; P[s2i[c]]+=f; cnt[s2i[c]]+=1; TF.append(f); TL.append(s2i[c])
    P=F.normalize(P/cnt.clamp(min=1).unsqueeze(1),dim=-1)
    return P.to(dev), torch.stack(TF).to(dev), torch.tensor(TL).to(dev)

def seen_score(qF,P,TF,TL):
    out=torch.empty(qF.shape[0],S,device=dev)
    for i in range(0,qF.shape[0],2000):
        e=qF[i:i+2000]; ps=e@P.t(); sim=e@TF.t()
        cmax=torch.full((e.shape[0],S),-1e9,device=dev)
        cmax.scatter_reduce_(1,TL.unsqueeze(0).expand(e.shape[0],-1),sim,reduce="amax")
        out[i:i+2000]=ps+2.0*cmax
    return out

ap=argparse.ArgumentParser(); ap.add_argument("--generate",action="store_true"); A=ap.parse_args()

if not A.generate:
    # ---- SIM VALIDATION (hard split) ----
    # SEEN eval: per-class 20% holdout over classes with >=2 imgs
    cls2=[c for c in seen if len(by[c])>=2]
    trby=defaultdict(list); valrows=[]
    for c in cls2:
        fns=sorted(by[c]); k=min(max(1,round(0.2*len(fns))),len(fns)-1)
        for f in fns[:-k]: trby[c].append(f)
        for f in fns[-k:]: valrows.append((f,s2i[c]))
    # protos use only training portion; restrict S-space to cls2 via masking unused rows (keep full S, empty classes get tiny proto)
    PH,TFH,TLH=protos_and_train(HtrI,HtrF,trby,cls2); PL,TFL,TLL=protos_and_train(LtrI,LtrF,trby,cls2)
    VY=torch.tensor([y for _,y in valrows]).to(dev)
    qH=torch.stack([HtrF[HtrI[f]] for f,_ in valrows]).to(dev); qL=torch.stack([LtrF[LtrI[f]] for f,_ in valrows]).to(dev)
    sH=seen_score(qH,PH,TFH,TLH); sL=seen_score(qL,PL,TFL,TLL)
    seen_acc=((zc(sH)+0.5*zc(sL)).argmax(1)==VY).float().mean().item()*100
    # UNSEEN eval: hard split rarest 20%
    order=sorted(seen,key=lambda c:len(by[c])); pseudo=set(order[:int(len(seen)*0.2)])
    nonk=[i for i,c in enumerate(classes) if c not in set(c for c in seen if c not in pseudo)]
    nt=torch.tensor(nonk); cpos={ci2:j for j,ci2 in enumerate(nonk)}
    qf=[f for c in pseudo for f in by[c]]; gold=torch.tensor([cpos[ci[lab[f]]] for f in qf]).to(dev)
    uH=torch.stack([HtrF[HtrI[f]] for f in qf]).to(dev); uL=torch.stack([LtrF[LtrI[f]] for f in qf]).to(dev)
    eU=db(uH@TnH[nt].to(dev).t())+0.5*db(uL@TnL[nt].to(dev).t())
    un_acc=(eU.argmax(1)==gold).float().mean().item()*100
    proj=(20097*seen_acc+15568*un_acc)/35665
    print(f"SIM VALIDATION (final recipe, TTA+ensemble):")
    print(f"  SEEN={seen_acc:.2f}%  UNSEEN={un_acc:.2f}%  PROJECTED OVERALL={proj:.2f}%")
else:
    # ---- GENERATE submission ----
    PH,TFH,TLH=protos_and_train(HtrI,HtrF,by,seen); PL,TFL,TLL=protos_and_train(LtrI,LtrF,by,seen)
    HteI,HteF,Htef=load("outputs/emb_test_h_tta.pt"); LteI,LteF,_=load("outputs/emb_test.pt")
    tf=[fn for fn in Htef if fn in LteI]
    qH=torch.stack([HteF[HteI[fn]] for fn in tf]).to(dev); qL=torch.stack([LteF[LteI[fn]] for fn in tf]).to(dev)
    ens=zc(seen_score(qH,PH,TFH,TLH))+0.5*zc(seen_score(qL,PL,TFL,TLL))
    preds={fn:seen[k] for fn,k in zip(tf,ens.argmax(1).cpu().tolist())}
    nonk=torch.tensor([i for i,c in enumerate(classes) if c not in set(seen)])
    HunI,HunF,Hunf=load("outputs/emb_unseen_h_tta.pt"); LunI,LunF,_=load("outputs/emb_unseen.pt")
    uf=[fn for fn in Hunf if fn in LunI]
    uH=torch.stack([HunF[HunI[fn]] for fn in uf]).to(dev); uL=torch.stack([LunF[LunI[fn]] for fn in uf]).to(dev)
    eU=db(uH@TnH[nonk].to(dev).t())+0.5*db(uL@TnL[nonk].to(dev).t())
    for fn,j in zip(uf,eU.argmax(1).cpu().tolist()): preds[fn]=classes[nonk[j].item()]
    json.dump(preds,open("outputs/prediction_final.json","w"))
    print(f"wrote prediction_final.json n={len(preds)} test={len(tf)} unseen={len(uf)}")
