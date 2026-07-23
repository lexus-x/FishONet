"""Text-prompt sweep for the UNSEEN route on frozen ViT-H, hard-sim split.
Tests untested variants: description-first-sentence, name+trait, format variants, no-template.
Cheap: cached image features + fresh text encode. Reports H-alone db and deployed H+0.5L."""
import json, pickle, torch, open_clip, re
from collections import defaultdict
import torch.nn.functional as F
MODEL='hf-hub:imageomics/bioclip-2.5-vith14'; dev='cuda'
D='data/dl'
classes=list(pickle.load(open(f'{D}/all_classes.pkl','rb'))); ci={c:i for i,c in enumerate(classes)}
desc=json.load(open(f'{D}/descriptions.json')); tax=json.load(open('outputs/taxonomy_full.json'))
RANKS=['kingdom','phylum','class','order','family','genus']
def lineage(c):
    t=tax.get(c,{}); parts=[t.get(r) for r in RANKS]; parts=[p for p in parts if p]
    ep=c.split()[1] if len(c.split())>1 else c
    return ' '.join(parts+[ep])
def sent1(c):
    d=desc.get(c,'')
    s=re.split(r'(?<=[.!?])\s+', d.strip())
    return s[0] if s and s[0] else c
def short(c, n=18):  # first n words of description
    return ' '.join(desc.get(c,c).split()[:n])

VARIANTS={
 'name'          : lambda c: f'a photo of {c}.',
 'name_bare'     : lambda c: c,
 'name_fish'     : lambda c: f'a photo of a fish, {c}.',
 'taxon'         : lambda c: f'a photo of {lineage(c)}.',
 'desc_sent1'    : lambda c: sent1(c),
 'desc_short18'  : lambda c: short(c,18),
 'name_trait'    : lambda c: f'a photo of {c}. {short(c,14)}',
 'taxon_trait'   : lambda c: f'{lineage(c)}. {short(c,12)}',
}
model,_,_=open_clip.create_model_and_transforms(MODEL); tok=open_clip.get_tokenizer(MODEL); model=model.to(dev).eval()
def enc(texts,bs=256):
    out=[]
    with torch.no_grad(), torch.autocast('cuda',dtype=torch.bfloat16):
        for i in range(0,len(texts),bs):
            t=tok(texts[i:i+bs]).to(dev); f=model.encode_text(t).float(); out.append(F.normalize(f,dim=-1).cpu())
    return torch.cat(out)
T={k:enc([fn(c) for c in classes]) for k,fn in VARIANTS.items()}
print('encoded variants:',list(T.keys()),flush=True)

# hard-sim
tr=torch.load('outputs/emb_train_h.pt',weights_only=False); HF=F.normalize(tr['feats'].float(),dim=-1); HI={fn:i for i,fn in enumerate(tr['files'])}
L=torch.load('outputs/emb_train.pt',weights_only=False); LF=F.normalize(L['feats'].float(),dim=-1); LI={fn:i for i,fn in enumerate(L['files'])}
TnL=F.normalize(torch.load('outputs/text_emb.pt',weights_only=False)['emb_name'].float(),dim=-1)
lab=json.load(open(f'{D}/label_train.json'))
by=defaultdict(list)
for fn in tr['files']:
    if fn in lab and lab[fn] in ci and fn in LI: by[lab[fn]].append(fn)
seen=sorted(by.keys()); order=sorted(seen,key=lambda c:len(by[c]))
pseudo=set(order[:int(len(seen)*0.2)]); known=set(c for c in seen if c not in pseudo)
cand=[i for i,c in enumerate(classes) if c not in known]; cand_pos={x:j for j,x in enumerate(cand)}; cand_t=torch.tensor(cand)
qfn=[fn for c in pseudo for fn in by[c]]
qH=torch.stack([HF[HI[fn]] for fn in qfn]); qL=torch.stack([LF[LI[fn]] for fn in qfn])
gold=torch.tensor([cand_pos[ci[lab[fn]]] for fn in qfn])
def db(M): return (M-M.mean(0,keepdim=True))/(M.std(0,keepdim=True)+1e-6)
def t1(S): return (S.argmax(1)==gold).float().mean().item()*100
sL=db(qL@TnL[cand_t].t())
print(f'\n{"variant":14} Halone  +0.5L   (ref: name 22.35/23.99, taxon 22.99/23.99)')
for k in VARIANTS:
    Th=T[k][cand_t]; sH=db(qH@Th.t())
    print(f'{k:14} {t1(sH):6.2f}  {t1(sH+0.5*sL):6.2f}',flush=True)
