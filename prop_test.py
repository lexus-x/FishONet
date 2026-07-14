import json, pickle, torch, time
import torch.nn.functional as F
from collections import defaultdict
dev='cuda'; D='data/dl'; t0=time.time()
classes=list(pickle.load(open(f'{D}/all_classes.pkl','rb'))); ci={c:i for i,c in enumerate(classes)}
TtH=F.normalize(torch.load('outputs/text_emb_h_taxon.pt',weights_only=False)['emb_taxon'].float(),dim=-1)
TnL=F.normalize(torch.load('outputs/text_emb.pt',weights_only=False)['emb_name'].float(),dim=-1)
tr=torch.load('outputs/emb_train_h_tta.pt',weights_only=False)
HI={fn:i for i,fn in enumerate(tr['files'])}; HF=F.normalize(tr['feats'].float(),dim=-1)
trL=torch.load('outputs/emb_train.pt',weights_only=False)
LI={fn:i for i,fn in enumerate(trL['files'])}; LF=F.normalize(trL['feats'].float(),dim=-1)
lab=json.load(open(f'{D}/label_train.json'))
by=defaultdict(list)
for fn in tr['files']:
    if fn in lab and lab[fn] in ci and fn in LI: by[lab[fn]].append(fn)
seen=sorted(by.keys()); order=sorted(seen,key=lambda c:len(by[c]))
pseudo=[c for c in order[:int(len(seen)*0.2)]]; known=set(c for c in seen if c not in set(pseudo))
cand=[i for i,c in enumerate(classes) if c not in known]
cand_pos={cii:j for j,cii in enumerate(cand)}; cand_t=torch.tensor(cand)
qfn=[fn for c in pseudo for fn in by[c]]
gold=torch.tensor([cand_pos[ci[lab[fn]]] for fn in qfn]).to(dev)
Q=torch.stack([HF[HI[fn]] for fn in qfn]).to(dev)
QL=torch.stack([LF[LI[fn]] for fn in qfn]).to(dev)
Tt=TtH[cand_t].to(dev); TnLc=TnL[cand_t].to(dev)
print(f'qimgs={len(qfn)} candidates={len(cand)} [{time.time()-t0:.0f}s]',flush=True)
def db(M): return (M-M.mean(0,keepdim=True))/(M.std(0,keepdim=True)+1e-6)
def dis(M,a,b): return F.log_softmax(M/a,dim=0)+F.log_softmax(M/b,dim=1)
def t1(S): return (S.argmax(1)==gold).float().mean().item()*100
MH=Q@Tt.t(); ML=QL@TnLc.t()
s_db = db(MH)+0.5*db(ML)
s_dn = dis(MH,0.05,0.5)+0.5*dis(ML,0.05,0.5)
print(f'BASELINE ensemble  z-debias top1={t1(s_db):.2f}   DBNorm top1={t1(s_dn):.2f}',flush=True)
GH=Q@Q.t(); GL=QL@QL.t(); GHL=0.5*GH+0.5*GL
def propagate(seed, G, k, alpha, iters=20):
    sim=G.clone(); sim.fill_diagonal_(-2)
    tv,ti=sim.topk(k,1); W=torch.zeros_like(sim); W.scatter_(1,ti,tv.clamp(min=0))
    W=0.5*(W+W.t()); d=W.sum(1).clamp(min=1e-6); Di=d.pow(-0.5); Wn=Di[:,None]*W*Di[None,:]
    P=F.softmax(seed,dim=1); Fp=P.clone()
    for _ in range(iters): Fp=alpha*(Wn@Fp)+(1-alpha)*P
    return Fp
for seedname,seed in [('z-debias',s_db),('DBNorm',s_dn)]:
    base=t1(seed); best=(base,'none')
    for gname,G in [('H',GH),('L',GL),('H+L',GHL)]:
        for k in (5,10,20,50):
            for alpha in (0.3,0.5,0.7,0.9):
                a=t1(propagate(seed,G,k,alpha))
                if a>best[0]: best=(a,f'{gname} k={k} a={alpha}')
    print(f'[{seedname}] base={base:.2f}  BEST prop={best[0]:.2f} (+{best[0]-base:.2f}) [{best[1]}]',flush=True)
print('--- kNN feature-averaging (TransCLIP-lite) on H graph, re-score z-debias ensemble ---')
for k in (3,5,10,20):
    nbr=GH.topk(k+1,1).indices
    Qa=F.normalize(Q[nbr].mean(1),dim=-1)
    s=db(Qa@Tt.t())+0.5*db(ML)
    print(f'  kNN-avg(H) k={k:<2d} top1={t1(s):.2f}  (base z-debias {t1(s_db):.2f})',flush=True)
print('DONE_PROP',flush=True)
