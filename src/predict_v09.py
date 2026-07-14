"""v09: v07 + SEEN long-tail TEXT-BLEND. Seen score = proto + 2*cmax + LAM*taxon_text_sim (H route).
Text term (BioCLIP taxonomic-lineage text on the GIVEN class names) rescues low-count seen classes
(holdout 85.88 -> ~87.06). UNSEEN route unchanged from v07. CPU only.
  python src/predict_v09.py            # prints clean holdout (should show ~87.0 for FT-H)
  python src/predict_v09.py --generate # writes outputs/prediction_v09.json"""
import json, torch, argparse
from collections import defaultdict
import torch.nn.functional as F
torch.set_num_threads(8); dev="cpu"
LAM = 4.0   # taxon-text weight (validated peak 3-4)

def load(p):
    d=torch.load(p,weights_only=False)
    return {fn:i for i,fn in enumerate(d["files"])}, F.normalize(d["feats"].float(),dim=-1), list(d["files"])
def zc(M): return (M-M.mean())/(M.std()+1e-6)
def db(M): return (M-M.mean(0,keepdim=True))/(M.std(0,keepdim=True)+1e-6)

txtH=torch.load("outputs/text_emb_h.pt",weights_only=False); txtL=torch.load("outputs/text_emb.pt",weights_only=False)
txtHt=torch.load("outputs/text_emb_h_taxon.pt",weights_only=False)
classes=txtH["classes"]; ci={c:i for i,c in enumerate(classes)}
TnH=F.normalize(txtH["emb_name"].float(),dim=-1); TnL=F.normalize(txtL["emb_name"].float(),dim=-1)
TtH=F.normalize(txtHt["emb_taxon"].float(),dim=-1)
lab=json.load(open("data/dl/label_train.json"))
FtrI,FtrF,Ftrf=load("outputs/emb_train_ft.pt"); LtrI,LtrF,_=load("outputs/emb_train.pt")
trainfiles=[fn for fn in Ftrf if fn in LtrI and fn in lab and lab[fn] in ci]
by=defaultdict(list)
for fn in trainfiles: by[lab[fn]].append(fn)
seen=sorted(by.keys()); s2i={c:i for i,c in enumerate(seen)}; S=len(seen)
TseenTax=torch.stack([TtH[ci[c]] for c in seen])   # [S, dH] taxon text for seen classes

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

ap=argparse.ArgumentParser(); ap.add_argument("--generate",action="store_true"); A=ap.parse_args()

# ---- CLEAN seen sim (ft.py split; FT never trained on val) ----
trby=defaultdict(list); valrows=[]
for c in seen:
    fns=sorted(by[c])
    if len(fns)>=3:
        k=max(1,round(0.2*len(fns)))
        for f in fns[:-k]: trby[c].append(f)
        for f in fns[-k:]: valrows.append((f,s2i[c]))
    else:
        for f in fns: trby[c].append(f)
PH,TFH,TLH=protos_and_train(FtrI,FtrF,trby,seen); PL,TFL,TLL=protos_and_train(LtrI,LtrF,trby,seen)
VY=torch.tensor([y for _,y in valrows])
vqH=torch.stack([FtrF[FtrI[f]] for f,_ in valrows]); vqL=torch.stack([LtrF[LtrI[f]] for f,_ in valrows])
# no-text baseline vs text
sH0=seen_score(vqH,PH,TFH,TLH); sHt=seen_score(vqH,PH,TFH,TLH,TseenTax,LAM); sL=seen_score(vqL,PL,TFL,TLL)
def a(M): return (M.argmax(1)==VY).float().mean().item()*100
print(f"[no text] FT-H={a(zc(sH0)):.2f}  FT-H+0.5L={a(zc(sH0)+0.5*zc(sL)):.2f}")
print(f"[+{LAM}taxon] FT-H={a(zc(sHt)):.2f}  FT-H+0.5L={a(zc(sHt)+0.5*zc(sL)):.2f}")
a_fth=a(zc(sHt)); a_ens=a(zc(sHt)+0.5*zc(sL)); use_ens=a_ens>=a_fth
best=max(a_fth,a_ens); real_seen=best*0.9065
proj=(20097*real_seen/100 + 15568*0.1222)/35665*100
print(f"-> use {'ENSEMBLE' if use_ens else 'FT-H ALONE'}; calibrated real seen~{real_seen:.1f}%  overall~{proj:.1f}%  (v07 was ~49.2)")

if A.generate:
    PH,TFH,TLH=protos_and_train(FtrI,FtrF,by,seen); PL,TFL,TLL=protos_and_train(LtrI,LtrF,by,seen)
    FteI,FteF,Ftef=load("outputs/emb_test_ft.pt"); LteI,LteF,_=load("outputs/emb_test.pt")
    tf=[fn for fn in Ftef if fn in LteI]
    qH=torch.stack([FteF[FteI[fn]] for fn in tf])
    sc=zc(seen_score(qH,PH,TFH,TLH,TseenTax,LAM))
    if use_ens:
        qL=torch.stack([LteF[LteI[fn]] for fn in tf]); sc=sc+0.5*zc(seen_score(qL,PL,TFL,TLL))
    preds={fn:seen[k] for fn,k in zip(tf,sc.argmax(1).tolist())}
    nonk=torch.tensor([i for i,c in enumerate(classes) if c not in set(seen)])
    HunI,HunF,Hunf=load("outputs/emb_unseen_h_tta.pt"); LunI,LunF,_=load("outputs/emb_unseen.pt")
    uf=[fn for fn in Hunf if fn in LunI]
    uH=torch.stack([HunF[HunI[fn]] for fn in uf]); uL=torch.stack([LunF[LunI[fn]] for fn in uf])
    eU=db(uH@TnH[nonk].t())+0.5*db(uL@TnL[nonk].t())
    for fn,j in zip(uf,eU.argmax(1).tolist()): preds[fn]=classes[nonk[j].item()]
    json.dump(preds,open("outputs/prediction_v09.json","w"))
    print(f"wrote prediction_v09.json n={len(preds)} test={len(tf)} unseen={len(uf)} seen_recipe={'ENS' if use_ens else 'FT-H'} LAM={LAM}")
