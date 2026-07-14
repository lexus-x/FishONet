"""Lever 2 probe: does a DINOv2 image-image graph label-propagation improve the unseen
assignment over the BioCLIP-score seed? Validate on hard-sim using ONLY cached features
(pseudo-unseen are train imgs -> BioCLIP H/L + DINOv2 ViT-L all cached). No encoding."""
import json, time, os, torch
import torch.nn.functional as F
from collections import defaultdict
torch.set_num_threads(8); dev='cuda'
STATUS='outputs/job_status.txt'; t0=time.time()
def st(m):
    open(STATUS+'.tmp','w').write("JOB: Lever2 DINOv2-graph propagation (hard-sim validate)\n  elapsed %6.1fs\n  %s\n"%(time.time()-t0,m)); os.replace(STATUS+'.tmp',STATUS); print(m,flush=True)
def loadfeat(p):
    d=torch.load(p,weights_only=False); return {fn:i for i,fn in enumerate(d['files'])}, F.normalize(d['feats'].float(),dim=-1)
st('loading cached features...')
txtH=torch.load('outputs/text_emb_h.pt',weights_only=False); classes=txtH['classes']; ci={c:i for i,c in enumerate(classes)}
TtH=F.normalize(torch.load('outputs/text_emb_h_taxon.pt',weights_only=False)['emb_taxon'].float(),dim=-1)
TnL=F.normalize(torch.load('outputs/text_emb.pt',weights_only=False)['emb_name'].float(),dim=-1)
lab=json.load(open('data/dl/label_train.json'))
HI,HF=loadfeat('outputs/emb_train_h_tta.pt')
LI,LF=loadfeat('outputs/emb_train.pt')
DI,DF=loadfeat('outputs/emb_train_dino.pt')
by=defaultdict(list)
for fn in HI:
    if fn in lab and lab[fn] in ci and fn in LI and fn in DI: by[lab[fn]].append(fn)
seen=sorted(by.keys()); order=sorted(seen,key=lambda c:len(by[c]))
pseudo=set(order[:int(len(seen)*0.2)]); known=set(c for c in seen if c not in pseudo)
nonk=[i for i,c in enumerate(classes) if c not in known]; nt=torch.tensor(nonk); cpos={v:j for j,v in enumerate(nonk)}
qf=[f for c in pseudo for f in by[c]]
gold=torch.tensor([cpos[ci[lab[f]]] for f in qf]).to(dev)
QH=torch.stack([HF[HI[f]] for f in qf]).to(dev); QL=torch.stack([LF[LI[f]] for f in qf]).to(dev); QD=torch.stack([DF[DI[f]] for f in qf]).to(dev)
TtH_=TtH[nt].to(dev); TnL_=TnL[nt].to(dev)
def db(M): return (M-M.mean(0,keepdim=True))/(M.std(0,keepdim=True)+1e-6)
def dis(M,a,b): return F.log_softmax(M/a,dim=0)+F.log_softmax(M/b,dim=1)
def t1(M): return (M.argmax(1)==gold).float().mean().item()*100
MH=QH@TtH_.t(); ML=QL@TnL_.t()
seed_z=db(MH)+0.5*db(ML); seed_d=dis(MH,0.05,0.5)+0.5*dis(ML,0.05,0.5)
base_z=t1(seed_z); base_d=t1(seed_d); base=max(base_z,base_d)
seed=seed_d if base_d>=base_z else seed_z
st('seed: z-debias %.2f  DBNorm %.2f  (using best as seed)'%(base_z,base_d))
Sg=QD@QD.t()   # DINOv2 cosine image-image
def propagate(seed,k,alpha,iters):
    n=seed.shape[0]; sim=Sg.clone(); sim.fill_diagonal_(-1e9)
    topv,topi=sim.topk(k,dim=1); W=torch.zeros(n,n,device=dev); W.scatter_(1,topi,topv.clamp(min=0))
    W=0.5*(W+W.t()); d=W.sum(1).clamp(min=1e-6); Di=d.pow(-0.5); Wn=Di.unsqueeze(1)*W*Di.unsqueeze(0)
    P=F.softmax(seed,dim=1); Fp=P.clone()
    for _ in range(iters): Fp=alpha*(Wn@Fp)+(1-alpha)*P
    return Fp
# DINOv2 neighbour purity (sanity: do nearest DINOv2 neighbours share the true class?)
with torch.no_grad():
    sim=Sg.clone(); sim.fill_diagonal_(-1e9); nn1=sim.argmax(1)
    purity=(gold[nn1]==gold).float().mean().item()*100
st('DINOv2 1-NN purity (same pseudo-class) = %.1f%%'%purity)
best=(base,'seed-only')
for k in [10,20,50]:
    for alpha in [0.5,0.8,0.9]:
        a=t1(propagate(seed,k,alpha,20))
        if a>best[0]: best=(a,'k=%d alpha=%.1f'%(k,alpha))
        st('prop k=%d alpha=%.1f: %.2f   (seed %.2f, best %.2f)'%(k,alpha,a,base,best[0]))
fin=['seed-best %.2f -> propagation-best %.2f [%s]  delta %+.2f'%(base,best[0],best[1],best[0]-base),
     'DINOv2 1-NN purity %.1f%%  (n=%d, candidates=%d)'%(purity,len(qf),len(nonk)),
     ('VERDICT: propagation HELPS -> encode unseen DINOv2 + build v16' if best[0]>base+0.3 else
      'VERDICT: ViT-L DINOv2 graph does NOT help (purity too low) -> try ViT-g or drop')]
out={'base':round(base,2),'prop':round(best[0],2),'cfg':best[1],'purity':round(purity,1)}
json.dump(out,open('outputs/lever2_val.json','w'))
st('DONE | '+' | '.join(fin))
