import json, torch
from collections import defaultdict
import torch.nn.functional as F
text=torch.load("outputs/text_emb_h.pt",weights_only=False)
classes=text["classes"]; cls_index={c:i for i,c in enumerate(classes)}
Tname=F.normalize(text["emb_name"].float(),dim=-1); Tdesc=F.normalize(text["emb_desc"].float(),dim=-1)
tr=torch.load("outputs/emb_train_h.pt",weights_only=False)
lab=json.load(open("data/dl/label_train.json"))
by_cls=defaultdict(list)
for fn,f in zip(tr["files"],tr["feats"]):
    if fn in lab and lab[fn] in cls_index: by_cls[lab[fn]].append(f)
seen=sorted(by_cls.keys())
fbc={c:F.normalize(torch.stack(v).float(),dim=-1) for c,v in by_cls.items()}
cls_list=[c for c in seen if len(by_cls[c])>=2]; c2i={c:i for i,c in enumerate(cls_list)}
proto,proto_trim,trainfeats,trainlbl,valF,valY=[],[],[],[],[],[]
for c in cls_list:
    fs=fbc[c]; k=min(max(1,round(0.2*len(fs))),len(fs)-1); trp=fs[:-k]
    proto.append(F.normalize(trp.mean(0),dim=0))
    if len(trp)>=3:
        m=F.normalize(trp.mean(0),dim=0); keep=(trp@m).argsort(descending=True)[:len(trp)-1]
        proto_trim.append(F.normalize(trp[keep].mean(0),dim=0))
    else: proto_trim.append(F.normalize(trp.mean(0),dim=0))
    for t in trp: trainfeats.append(t); trainlbl.append(c2i[c])
    for j in range(len(fs)-k,len(fs)): valF.append(fs[j]); valY.append(c2i[c])
P=torch.stack(proto); Pt=torch.stack(proto_trim); VF=torch.stack(valF); VY=torch.tensor(valY)
TFE=torch.stack(trainfeats); TL=torch.tensor(trainlbl)
acc=lambda pred:round((pred==VY).float().mean().item()*100,2)
print("SEEN proto       :",acc((VF@P.t()).argmax(1)))
print("SEEN proto-trim  :",acc((VF@Pt.t()).argmax(1)))
print("SEEN 1-NN        :",acc(TL[(VF@TFE.t()).argmax(1)]))
order=sorted(seen,key=lambda c:len(by_cls[c])); pseudo=set(order[:int(len(seen)*0.2)])
known_idx=set(cls_index[c] for c in seen if c not in pseudo)
cand=[i for i in range(len(classes)) if i not in known_idx]; cand_t=torch.tensor(cand); cand_pos={ci:j for j,ci in enumerate(cand)}
qf,qy=[],[]
for c in pseudo:
    for f in by_cls[c]: qf.append(f); qy.append(cls_index[c])
Fq=F.normalize(torch.stack(qf).float(),dim=-1); gold=torch.tensor([cand_pos[y] for y in qy])
def db(S):return (S-S.mean(0,keepdim=True))/(S.std(0,keepdim=True)+1e-6)
def t1(S):return round((S.argmax(1)==gold).float().mean().item()*100,2)
Sn=Fq@Tname[cand_t].t(); Sd=Fq@Tdesc[cand_t].t()
print("UNSEEN name+db   :",t1(db(Sn)))
print("UNSEEN desc+db   :",t1(db(Sd)))
print("UNSEEN n+d/2 +db :",t1(db(0.5*(Sn+Sd))))
print("UNSEEN .8n+.2d+db:",t1(db(0.8*Sn+0.2*Sd)))
