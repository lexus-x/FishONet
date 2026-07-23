"""For the scaled ckpt outputs/ctft_big.pt: WiSE-FT scale sweep on pseudo (deployed H_taxon+0.5L,
hflip TTA), pick best scale, then extract unseen-test at that scale -> emb_unseen_ctftbig.pt."""
import os, json, pickle
from collections import defaultdict
import torch, torch.nn as nn, torch.nn.functional as F
import open_clip
from PIL import Image
from torch.utils.data import Dataset, DataLoader
MODEL='hf-hub:imageomics/bioclip-2.5-vith14'; DEV='cuda'; CKPT='outputs/ctft_big.pt'
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
ck=torch.load(CKPT,weights_only=False); print('ckpt',CKPT,'db',ck.get('db'),'step',ck.get('step'),'top_k',ck.get('top_k'),'rank',ck.get('rank'),flush=True)
model,_,pp=open_clip.create_model_and_transforms(MODEL); inject_lora(model,ck['rank'],ck['alpha'],ck['top_k'])
own=dict(model.named_parameters())
for n,v in ck['state'].items(): own[n].data.copy_(v)
model=model.to(DEV).eval()
TtH=F.normalize(torch.load('outputs/text_emb_h_taxon.pt',weights_only=False)['emb_taxon'].float(),dim=-1)
TnL=F.normalize(torch.load('outputs/text_emb.pt',weights_only=False)['emb_name'].float(),dim=-1)
tr=torch.load('outputs/emb_train_h.pt',weights_only=False)
LtrD=torch.load('outputs/emb_train.pt',weights_only=False); LI={fn:i for i,fn in enumerate(LtrD['files'])}; LF=F.normalize(LtrD['feats'].float(),dim=-1)
by=defaultdict(list)
for fn in tr['files']:
    if fn in lab and lab[fn] in ci and fn in imgidx and fn in LI: by[lab[fn]].append(fn)
seen=sorted(by.keys()); order=sorted(seen,key=lambda c:len(by[c]))
pseudo=set(order[:int(len(seen)*0.2)]); known=set(c for c in seen if c not in pseudo)
cand=[i for i,c in enumerate(classes) if c not in known]; cand_pos={x:j for j,x in enumerate(cand)}; cand_t=torch.tensor(cand)
qfn=[fn for c in pseudo for fn in by[c]]
class DS(Dataset):
    def __init__(s,it): s.it=it
    def __len__(s): return len(s.it)
    def __getitem__(s,i):
        fn=s.it[i]
        try: return pp(Image.open(imgidx[fn]).convert('RGB')),fn
        except: return torch.zeros(3,224,224),fn
def enc_tta(files):
    dl=DataLoader(DS(files),batch_size=256,num_workers=12,pin_memory=True); fs=[];kept=[]
    with torch.no_grad():
        for x,fns in dl:
            x=x.to(DEV)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                f=F.normalize(model.encode_image(x).float(),dim=-1); f2=F.normalize(model.encode_image(torch.flip(x,dims=[-1])).float(),dim=-1)
                f=F.normalize(f+f2,dim=-1)
            fs.append(f.cpu()); kept+=list(fns)
    return torch.cat(fs),kept
def db(M): return (M-M.mean(0,keepdim=True))/(M.std(0,keepdim=True)+1e-6)
qL=torch.stack([LF[LI[fn]] for fn in qfn]); gold=torch.tensor([cand_pos[ci[lab[fn]]] for fn in qfn])
sL=db(qL@TnL[cand_t].t())
print('scale  deployed(H_taxon+0.5L,hflip)   (v09/v11 ref: 23.99 / 26.32)')
best=(0,-1)
for sc in (0.0,0.4,0.5,0.6,0.7,1.0):
    set_scale(model,sc); qf,kept=enc_tta(qfn)
    gld=torch.tensor([cand_pos[ci[lab[fn]]] for fn in kept])
    S=db(qf@TtH[cand_t].t())+0.5*sL
    acc=(S.argmax(1)==gld).float().mean().item()*100
    print(f'{sc:<5}  {acc:.2f}',flush=True)
    if acc>best[1]: best=(sc,acc)
print(f'BEST scale={best[0]} deployed={best[1]:.2f}',flush=True)
set_scale(model,best[0])
un=torch.load('outputs/emb_unseen_h_tta.pt',weights_only=False)
uf,uk=enc_tta(list(un['files']))
torch.save({'files':uk,'feats':uf,'scale':best[0]},'outputs/emb_unseen_ctftbig.pt')
print('saved outputs/emb_unseen_ctftbig.pt n=',len(uk),'scale=',best[0],flush=True)
