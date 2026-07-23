import os, json, pickle, time
from collections import defaultdict
import torch, torch.nn as nn, torch.nn.functional as F
import open_clip
from PIL import Image
from torch.utils.data import Dataset, DataLoader
torch.set_num_threads(8)
MODEL='hf-hub:imageomics/bioclip-2.5-vith14'; DEV='cuda'; CKPT='outputs/ctft_big.pt'; SCALE=0.4
STATUS='outputs/job_status.txt'; t0=time.time()
def st(m):
    open(STATUS+'.tmp','w').write("JOB: validate DBNorm on ctft (v12 unseen)\n  elapsed %6.1fs\n  %s\n"%(time.time()-t0,m)); os.replace(STATUS+'.tmp',STATUS); print(m,flush=True)
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
D='data/dl'; st('loading model + ctft_big...')
classes=list(pickle.load(open(f'{D}/all_classes.pkl','rb'))); ci={c:i for i,c in enumerate(classes)}
lab=json.load(open(f'{D}/label_train.json')); imgidx=index_images(f'{D}/images')
ck=torch.load(CKPT,weights_only=False)
model,_,pp=open_clip.create_model_and_transforms(MODEL); inject_lora(model,ck['rank'],ck['alpha'],ck['top_k'])
own=dict(model.named_parameters())
for n,v in ck['state'].items(): own[n].data.copy_(v)
model=model.to(DEV).eval(); set_scale(model,SCALE)
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
    dl=DataLoader(DS(files),batch_size=256,num_workers=12,pin_memory=True); fs=[];kept=[];n=0
    with torch.no_grad():
        for x,fns in dl:
            x=x.to(DEV)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                f=F.normalize(model.encode_image(x).float(),dim=-1)
                f2=F.normalize(model.encode_image(torch.flip(x,dims=[-1])).float(),dim=-1)
                f=F.normalize(f+f2,dim=-1)
            fs.append(f.cpu()); kept+=list(fns); n+=len(fns); st('encoded %d/%d pseudo-unseen via ctft@%.1f+hflip'%(n,len(files),SCALE))
    return torch.cat(fs),kept
qf,kept=enc_tta(qfn)
qL=torch.stack([LF[LI[fn]] for fn in kept]); gold=torch.tensor([cand_pos[ci[lab[fn]]] for fn in kept])
MH=qf@TtH[cand_t].t(); ML=qL@TnL[cand_t].t()
def db(M): return (M-M.mean(0,keepdim=True))/(M.std(0,keepdim=True)+1e-6)
def dis(M,a,b): return F.log_softmax(M/a,dim=0)+F.log_softmax(M/b,dim=1)
def t1(S): return (S.argmax(1)==gold).float().mean().item()*100
base=t1(db(MH)+0.5*db(ML))
best=(-1.0,None)
for tc in [0.01,0.02,0.05,0.1]:
    for tr2 in [0.05,0.1,0.5,1e9]:
        a=t1(dis(MH,tc,tr2)+0.5*dis(ML,tc,tr2))
        if a>best[0]: best=(a,(tc,tr2))
        st('DBNorm tc=%s tr=%s: %.2f   (z-debias %.2f, best %.2f)'%(tc,tr2,a,base,best[0]))
out={'base':round(base,2),'dbnorm':round(best[0],2),'tc':best[1][0],'tr':best[1][1],'delta':round(best[0]-base,2),'n':len(kept)}
json.dump(out,open('outputs/v15_val.json','w'))
st('DONE: z-debias %.2f vs DBNorm %.2f @tc=%s tr=%s  delta %+.2f (n=%d)'%(base,best[0],best[1][0],best[1][1],best[0]-base,len(kept)))
