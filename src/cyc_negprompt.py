import json, torch
import torch.nn.functional as F
from collections import defaultdict
import open_clip
torch.set_num_threads(16); dev='cpu'
MODEL='hf-hub:imageomics/bioclip-2.5-vith14'

txtH=torch.load("outputs/text_emb_h.pt",weights_only=False)
classes=txtH["classes"]; ci={c:i for i,c in enumerate(classes)}
neg_prompts=[f"a photo that is not {c}, a fish species." for c in classes]

model,_,_=open_clip.create_model_and_transforms(MODEL); tok=open_clip.get_tokenizer(MODEL)
model=model.to(dev).eval()
def encode(prompts,bs=256):
    out=[]
    with torch.no_grad():
        for i in range(0,len(prompts),bs):
            e=model.encode_text(tok(prompts[i:i+bs]).to(dev)).float()
            out.append(F.normalize(e,dim=-1).cpu())
            if i % 2560==0: print(f"  encoded {i}/{len(prompts)}", flush=True)
    return torch.cat(out)
Tneg=F.normalize(encode(neg_prompts),dim=-1)
torch.save({'classes':classes,'emb_neg':Tneg}, "outputs/text_emb_h_neg.pt")
print("neg text emb built", tuple(Tneg.shape), flush=True)

TtaxH=F.normalize(torch.load("outputs/text_emb_h_taxon.pt",weights_only=False)["emb_taxon"].float(),dim=-1)
TnL=F.normalize(torch.load("outputs/text_emb.pt",weights_only=False)["emb_name"].float(),dim=-1)
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
TtaxH_=TtaxH[nt]; TnL_=TnL[nt]; Tneg_=Tneg[nt]
print(f"pseudo={len(pseudo)} queries={len(qfiles)} candidates={len(nonk)}", flush=True)
def t1(M): return (M.argmax(1)==gold).float().mean().item()*100
def zc(S): return (S-S.mean(0,keepdim=True))/(S.std(0,keepdim=True)+1e-6)
SHt=QH@TtaxH_.t(); SLn=QL@TnL_.t(); Sneg=QH@Tneg_.t()
C_cur=t1(zc(SHt)+0.5*zc(SLn))
print(f"=== DEPLOYED C (ztaxon+0.5zL): {C_cur:.2f}% ===", flush=True)
print(f"neg-prompt-only member t1 (as positive add): {t1(zc(Sneg)):.2f}%", flush=True)
best=(C_cur,"none")
for w in [0.1,0.2,0.3,0.5,0.75,1.0]:
    a_sub=t1(zc(SHt)+0.5*zc(SLn)-w*zc(Sneg))
    a_add=t1(zc(SHt)+0.5*zc(SLn)+w*zc(Sneg))
    print(f"  -{w:.2f}*neg -> {a_sub:.2f}% (delta {a_sub-C_cur:+.2f})   +{w:.2f}*neg -> {a_add:.2f}% (delta {a_add-C_cur:+.2f})", flush=True)
    if a_sub>best[0]: best=(a_sub,f"-{w}*neg")
    if a_add>best[0]: best=(a_add,f"+{w}*neg")
print(f"CYCLE_NEG_RESULT best={best[0]:.2f}% [{best[1]}] vs deployed {C_cur:.2f} delta {best[0]-C_cur:+.2f}", flush=True)
