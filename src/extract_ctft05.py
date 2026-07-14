"""Re-extract unseen-test features from the contrastive-FT ckpt at WiSE-FT scale 0.5 + hflip TTA
(matching the deployed pipeline). Confirm the pseudo-unseen deployed gain, then save for v11."""
import os, json, pickle
from collections import defaultdict
import torch, torch.nn as nn, torch.nn.functional as F
import open_clip
from PIL import Image
from torch.utils.data import Dataset, DataLoader
MODEL='hf-hub:imageomics/bioclip-2.5-vith14'; DEV='cuda'; SCALE=0.5
class LoRALinear(nn.Module):
    def __init__(s,base,r=16,alpha=32):
        super().__init__(); s.base=base
        for p in s.base.parameters(): p.requires_grad_(False)
        s.r=r; s.scaling=alpha/max(r,1)
        s.A=nn.Parameter(torch.zeros(r,base.in_features)); s.B=nn.Parameter(torch.zeros(base.out_features,r))
        s.drop=nn.Dropout(0.0); s.lora_scale=1.0
    def forward(s,x): return s.base(x)+(s.drop(x)@s.A.t()@s.B.t())*s.scaling*s.lora_scale
def inject_lora(model,r,alpha,top_k=0):
    blocks=model.visual.transformer.resblocks
    for i in (range(len(blocks)) if top_k<=0 else range(len(blocks)-top_k,len(blocks))):
        blk=blocks[i]; blk.mlp.c_fc=LoRALinear(blk.mlp.c_fc,r,alpha); blk.mlp.c_proj=LoRALinear(blk.mlp.c_proj,r,alpha)
def set_scale(model,s):
    for blk in model.visual.transformer.resblocks:
        for m in (blk.mlp.c_fc,blk.mlp.c_proj):
            if isinstance(m,LoRALinear): m.lora_scale=s
def index_images(root):
    idx={}
    for dp,_,fs in os.walk(root):
        for f in fs:
            if f.lower().endswith(('.jpg','.jpeg','.png')): idx[f]=os.path.join(dp,f)
    return idx
D='data/dl'
classes=list(pickle.load(open(f'{D}/all_classes.pkl','rb'))); ci={c:i for i,c in enumerate(classes)}
lab=json.load(open(f'{D}/label_train.json')); imgidx=index_images(f'{D}/images')
ck=torch.load('outputs/ctft_lora.pt',weights_only=False)
model,_,pp=open_clip.create_model_and_transforms(MODEL); inject_lora(model,ck['rank'],ck['alpha'],ck['top_k'])
own=dict(model.named_parameters())
for n,v in ck['state'].items(): own[n].data.copy_(v)
model=model.to(DEV).eval(); set_scale(model,SCALE)
print(f'loaded ckpt (db {ck.get("db")}), WiSE-FT scale={SCALE}',flush=True)
class DS(Dataset):
    def __init__(s,it): s.it=it
    def __len__(s): return len(s.it)
    def __getitem__(s,i):
        fn=s.it[i]
        try: return pp(Image.open(imgidx[fn]).convert('RGB')),fn
        except: return torch.zeros(3,224,224),fn
def encode_tta(files):
    dl=DataLoader(DS(files),batch_size=256,num_workers=12,pin_memory=True); fs=[]; kept=[]
    with torch.no_grad():
        for x,fns in dl:
            x=x.to(DEV)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                f=F.normalize(model.encode_image(x).float(),dim=-1)
                f2=F.normalize(model.encode_image(torch.flip(x,dims=[-1])).float(),dim=-1)
                f=F.normalize(f+f2,dim=-1)
            fs.append(f.cpu()); kept+=list(fns)
    return torch.cat(fs),kept
# confirm on pseudo (deployed ensemble, scale 0.5 + hflip)
TtH=F.normalize(torch.load('outputs/text_emb_h_taxon.pt',weights_only=False)['emb_taxon'].float(),dim=-1)
TnL=F.normalize(torch.load('outputs/text_emb.pt',weights_only=False)['emb_name'].float(),dim=-1)
tr=torch.load('outputs/emb_train_h.pt',weights_only=False)
LtrD=torch.load('outputs/emb_train.pt',weights_only=False); LI={fn:i for i,fn in enumerate(LtrD['files'])}; LF=F.normalize(LtrD['feats'].float(),dim=-1)
by=defaultdict(list)
for fn in tr['files']:
    if fn in lab and lab[fn] in ci and fn in imgidx and fn in LI: by[lab[fn]].append(fn)
seen=sorted(by.keys()); order=sorted(seen,key=lambda c:len(by[c]))
pseudo=set(order[:int(len(seen)*0.2)]); known=set(c for c in seen if c not in pseudo)
cand=[i for i,c in enumerate(classes) if c not in known]; cand_pos={cii:j for j,cii in enumerate(cand)}; cand_t=torch.tensor(cand)
qfn=[fn for c in pseudo for fn in by[c]]
qf,kept=encode_tta(qfn)
qL=torch.stack([LF[LI[fn]] for fn in kept]); gold=torch.tensor([cand_pos[ci[lab[fn]]] for fn in kept])
def db(M): return (M-M.mean(0,keepdim=True))/(M.std(0,keepdim=True)+1e-6)
S=db(qf@TtH[cand_t].t())+0.5*db(qL@TnL[cand_t].t())
print(f'CONFIRM pseudo deployed @scale{SCALE}+hflip: {(S.argmax(1)==gold).float().mean().item()*100:.2f}  (v09=23.99)',flush=True)
# extract real unseen-test
un=torch.load('outputs/emb_unseen_h_tta.pt',weights_only=False)
uf,uk=encode_tta(list(un['files']))
torch.save({'files':uk,'feats':uf,'scale':SCALE},'outputs/emb_unseen_ctft05.pt')
print('saved outputs/emb_unseen_ctft05.pt n=',len(uk),flush=True)
