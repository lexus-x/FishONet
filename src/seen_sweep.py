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
ps=VF@P.t(); sim=VF@TFE.t()
cmax=torch.full((len(VF),C),-1e9)
cmax.scatter_reduce_(1, TL.unsqueeze(0).expand(len(VF),-1), sim, reduce="amax")
print("proto:",acc(ps.argmax(1)),"  class-max(1NN):",acc(cmax.argmax(1)))
best=(0,0)
for a in [0.5,1.0,1.5,2.0,2.5,3.0,4.0,6.0]:
    v=acc((ps+a*cmax).argmax(1))
    print(f"  proto + {a}*cmax : {v}")
    if v>best[1]: best=(a,v)
print("BEST a=",best[0],"->",best[1])
