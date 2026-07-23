import json, torch
from collections import defaultdict
import torch.nn.functional as F
tr=torch.load("outputs/emb_train_h.pt",weights_only=False)
text=torch.load("outputs/text_emb_h.pt",weights_only=False)
cls_index={c:i for i,c in enumerate(text["classes"])}
lab=json.load(open("data/dl/label_train.json"))
by_cls=defaultdict(list)
for fn,f in zip(tr["files"],tr["feats"]):
    if fn in lab and lab[fn] in cls_index: by_cls[lab[fn]].append(f)
seen=sorted(by_cls.keys())
fbc={c:F.normalize(torch.stack(v).float(),dim=-1) for c,v in by_cls.items()}
cls_list=[c for c in seen if len(by_cls[c])>=2]; c2i={c:i for i,c in enumerate(cls_list)}
proto,TFE,TL,valF,valY=[],[],[],[],[]
for c in cls_list:
    fs=fbc[c]; k=min(max(1,round(0.2*len(fs))),len(fs)-1); trp=fs[:-k]
    proto.append(F.normalize(trp.mean(0),dim=0))
    for t in trp: TFE.append(t); TL.append(c2i[c])
    for j in range(len(fs)-k,len(fs)): valF.append(fs[j]); valY.append(c2i[c])
P=torch.stack(proto); VF=torch.stack(valF); VY=torch.tensor(valY)
TFE=torch.stack(TFE); TL=torch.tensor(TL); C=len(cls_list)
acc=lambda pred:round((pred==VY).float().mean().item()*100,2)
sim=VF@TFE.t()  # [val, Ntrain]
print("proto         :",acc((VF@P.t()).argmax(1)))
print("1-NN          :",acc(TL[sim.argmax(1)]))
for k in [3,5,10]:
    top=sim.topk(k,1).indices
    print(f"{k}-NN major    :",acc(torch.mode(TL[top],1).values))
# class-max-sim (== 1NN by class) and proto+classmax blend
classmax=torch.full((len(VF),C),-1e9)
classmax.scatter_reduce_(1, TL.unsqueeze(0).expand(len(VF),-1), sim, reduce="amax")
print("class-max-sim :",acc(classmax.argmax(1)))
ps=VF@P.t()
for a in [0.3,0.5,1.0]:
    print(f"proto+{a}*cmax :",acc((ps+a*classmax).argmax(1)))
