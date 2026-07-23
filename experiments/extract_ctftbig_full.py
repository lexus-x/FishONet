"""Step 2: re-extract ctft_big (scale=0.4, hflip TTA) features for seen-eval (test) and train,
mirroring src/ctft_big_finalize.py's enc_tta exactly, so all three splits (train/test/unseen)
share one consistent feature extraction recipe for the compliant single pipeline."""
import os, sys, time
import torch, torch.nn as nn, torch.nn.functional as F
import open_clip
from PIL import Image
from torch.utils.data import Dataset, DataLoader

MODEL='hf-hub:imageomics/bioclip-2.5-vith14'; DEV='cuda'; CKPT='outputs/ctft_big.pt'
SCALE=0.4

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

ck=torch.load(CKPT,weights_only=False)
print('ckpt',CKPT,'db',ck.get('db'),'step',ck.get('step'),'top_k',ck.get('top_k'),'rank',ck.get('rank'),flush=True)
model,_,pp=open_clip.create_model_and_transforms(MODEL); inject_lora(model,ck['rank'],ck['alpha'],ck['top_k'])
own=dict(model.named_parameters())
for n,v in ck['state'].items(): own[n].data.copy_(v)
model=model.to(DEV).eval()
set_scale(model,SCALE)

imgidx=index_images('data/dl/images')

class DS(Dataset):
    def __init__(s,it): s.it=it
    def __len__(s): return len(s.it)
    def __getitem__(s,i):
        fn=s.it[i]
        try: return pp(Image.open(imgidx[fn]).convert('RGB')),fn
        except: return torch.zeros(3,224,224),fn

def enc_tta(files):
    dl=DataLoader(DS(files),batch_size=256,num_workers=12,pin_memory=True)
    fs=[];kept=[]
    with torch.no_grad():
        for x,fns in dl:
            x=x.to(DEV)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                f=F.normalize(model.encode_image(x).float(),dim=-1)
                f2=F.normalize(model.encode_image(torch.flip(x,dims=[-1])).float(),dim=-1)
                f=F.normalize(f+f2,dim=-1)
            fs.append(f.cpu()); kept+=list(fns)
    return torch.cat(fs),kept

jobs = [
    ('outputs/emb_test_h_tta.pt', 'outputs/emb_test_ctftbig.pt', 20097),
    ('outputs/emb_train_h_tta.pt', 'outputs/emb_train_ctftbig.pt', 64259),
]
for src, dst, expect_n in jobs:
    t0=time.time()
    files = list(torch.load(src, weights_only=False)['files'])
    print(f'=== {src} -> {dst}  n={len(files)} (expect {expect_n}) ===', flush=True)
    feats, kept = enc_tta(files)
    miss = len(files) - len(kept)
    print(f'  done in {time.time()-t0:.1f}s  n_kept={len(kept)}  miss={miss}  feats={tuple(feats.shape)}', flush=True)
    torch.save({'files':kept,'feats':feats,'scale':SCALE}, dst)
    print(f'  saved {dst}', flush=True)
print('ALL DONE', flush=True)
