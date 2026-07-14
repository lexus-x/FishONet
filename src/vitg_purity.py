"""Decisive unseen-graph test with DINOv2 ViT-g (stronger fine-grained features).
Encode ONLY the 2318 pseudo-unseen imgs, measure 1-NN purity + propagation gain vs the
cached BioCLIP DBNorm seed. If purity/gain are high -> worth encoding the full test pool for v16."""
import json, time, os, torch, timm
import torch.nn.functional as F
from collections import defaultdict
from PIL import Image
from timm.data import resolve_data_config, create_transform
torch.set_num_threads(8); dev='cuda'
STATUS='outputs/job_status.txt'; t0=time.time()
def st(m):
    open(STATUS+'.tmp','w').write("JOB: DINOv2 ViT-g unseen purity probe\n  elapsed %6.1fs\n  %s\n"%(time.time()-t0,m)); os.replace(STATUS+'.tmp',STATUS); print(m,flush=True)
def index_images(root):
    idx={}
    for dp,_,fs in os.walk(root):
        for f in fs:
            if f.lower().endswith(('.jpg','.jpeg','.png')): idx[f]=os.path.join(dp,f)
    return idx
def loadfeat(p):
    d=torch.load(p,weights_only=False); return {fn:i for i,fn in enumerate(d['files'])}, F.normalize(d['feats'].float(),dim=-1)
st('loading BioCLIP cache + pseudo split...')
txtH=torch.load('outputs/text_emb_h.pt',weights_only=False); classes=txtH['classes']; ci={c:i for i,c in enumerate(classes)}
TtH=F.normalize(torch.load('outputs/text_emb_h_taxon.pt',weights_only=False)['emb_taxon'].float(),dim=-1)
TnL=F.normalize(torch.load('outputs/text_emb.pt',weights_only=False)['emb_name'].float(),dim=-1)
lab=json.load(open('data/dl/label_train.json'))
HI,HF=loadfeat('outputs/emb_train_h_tta.pt'); LI,LF=loadfeat('outputs/emb_train.pt')
by=defaultdict(list)
for fn in HI:
    if fn in lab and lab[fn] in ci and fn in LI: by[lab[fn]].append(fn)
seen=sorted(by.keys()); order=sorted(seen,key=lambda c:len(by[c]))
pseudo=set(order[:int(len(seen)*0.2)]); known=set(c for c in seen if c not in pseudo)
nonk=[i for i,c in enumerate(classes) if c not in known]; nt=torch.tensor(nonk); cpos={v:j for j,v in enumerate(nonk)}
qf=[f for c in pseudo for f in by[c]]
gold=torch.tensor([cpos[ci[lab[f]]] for f in qf]).to(dev)
QH=torch.stack([HF[HI[f]] for f in qf]).to(dev); QL=torch.stack([LF[LI[f]] for f in qf]).to(dev)
TtH_=TtH[nt].to(dev); TnL_=TnL[nt].to(dev)
def db(M): return (M-M.mean(0,keepdim=True))/(M.std(0,keepdim=True)+1e-6)
def dis(M,a,b): return F.log_softmax(M/a,dim=0)+F.log_softmax(M/b,dim=1)
def t1(M): return (M.argmax(1)==gold).float().mean().item()*100
MH=QH@TtH_.t(); ML=QL@TnL_.t()
seed=dis(MH,0.05,0.5)+0.5*dis(ML,0.05,0.5); base=t1(seed)
st('seed DBNorm %.2f; loading DINOv2 ViT-g (downloads ~4GB if absent)...'%base)
model=timm.create_model('vit_giant_patch14_dinov2',pretrained=True,num_classes=0).to(dev).eval()
cfg=resolve_data_config({},model=model); tf=create_transform(**cfg); H=cfg['input_size'][1]; Wd=cfg['input_size'][2]
idx=index_images('data/dl/images')
def encode(files):
    out=[]; buf=[]; n=0
    for fn in files:
        p=idx.get(fn)
        try: buf.append(tf(Image.open(p).convert('RGB')))
        except: buf.append(torch.zeros(3,H,Wd))
        if len(buf)>=64:
            x=torch.stack(buf).to(dev)
            with torch.no_grad(), torch.autocast('cuda',dtype=torch.bfloat16): f=F.normalize(model(x).float(),dim=-1)
            out.append(f.cpu()); n+=len(buf); buf=[]; st('ViT-g encoded %d/%d pseudo-unseen'%(n,len(files)))
    if buf:
        x=torch.stack(buf).to(dev)
        with torch.no_grad(), torch.autocast('cuda',dtype=torch.bfloat16): f=F.normalize(model(x).float(),dim=-1)
        out.append(f.cpu()); n+=len(buf); st('ViT-g encoded %d/%d pseudo-unseen'%(n,len(files)))
    return torch.cat(out).to(dev)
QD=encode(qf)
Sg=QD@QD.t(); s=Sg.clone(); s.fill_diagonal_(-1e9); nn1=s.argmax(1)
purity=(gold[nn1]==gold).float().mean().item()*100
st('ViT-g 1-NN purity = %.1f%%  (ViT-L was 29.3%%)'%purity)
def propagate(seed,k,alpha,iters):
    n=seed.shape[0]; ss=Sg.clone(); ss.fill_diagonal_(-1e9)
    tv,tiq=ss.topk(k,dim=1); W=torch.zeros(n,n,device=dev); W.scatter_(1,tiq,tv.clamp(min=0))
    W=0.5*(W+W.t()); d=W.sum(1).clamp(min=1e-6); Di=d.pow(-0.5); Wn=Di.unsqueeze(1)*W*Di.unsqueeze(0)
    P=F.softmax(seed,dim=1); Fp=P.clone()
    for _ in range(iters): Fp=alpha*(Wn@Fp)+(1-alpha)*P
    return Fp
best=(base,'seed-only')
for k in [10,20,50]:
    for alpha in [0.3,0.5,0.7]:
        a=t1(propagate(seed,k,alpha,20))
        if a>best[0]: best=(a,'k=%d a=%.1f'%(k,alpha))
        st('ViT-g prop k=%d alpha=%.1f: %.2f (seed %.2f best %.2f)'%(k,alpha,a,base,best[0]))
verdict='PURSUE (encode test pool + build v16)' if best[0]>base+0.5 else 'DROP unseen graph (purity/gain too low)'
json.dump({'base':round(base,2),'prop':round(best[0],2),'cfg':best[1],'purity':round(purity,1),'verdict':verdict},open('outputs/vitg_val.json','w'))
st('DONE: purity %.1f%% | seed %.2f -> prop %.2f [%s] delta %+.2f | %s'%(purity,base,best[0],best[1],best[0]-base,verdict))
