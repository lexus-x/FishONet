import json, pickle, torch, time
import torch.nn.functional as F
from collections import defaultdict
dev='cuda'; D='data/dl'; t0=time.time()
classes=list(pickle.load(open(f'{D}/all_classes.pkl','rb'))); ci={c:i for i,c in enumerate(classes)}
TnH=F.normalize(torch.load('outputs/text_emb_h.pt',weights_only=False)['emb_name'].float(),dim=-1)
TtH=F.normalize(torch.load('outputs/text_emb_h_taxon.pt',weights_only=False)['emb_taxon'].float(),dim=-1)
tr=torch.load('outputs/emb_train_h_tta.pt',weights_only=False)
HI={fn:i for i,fn in enumerate(tr['files'])}; HF=F.normalize(tr['feats'].float(),dim=-1)
lab=json.load(open(f'{D}/label_train.json'))
by=defaultdict(list)
for fn in tr['files']:
    if fn in lab and lab[fn] in ci: by[lab[fn]].append(fn)
seen=sorted(by.keys()); order=sorted(seen,key=lambda c:len(by[c]))
pseudo=[c for c in order[:int(len(seen)*0.2)]]; known=set(c for c in seen if c not in set(pseudo))
cand=[i for i,c in enumerate(classes) if c not in known]
cand_pos={cii:j for j,cii in enumerate(cand)}; cand_t=torch.tensor(cand)
qfn=[fn for c in pseudo for fn in by[c]]
gold=torch.tensor([cand_pos[ci[lab[fn]]] for fn in qfn]).to(dev)
Q=torch.stack([HF[HI[fn]] for fn in qfn]).to(dev)
Tt=TtH[cand_t].to(dev); Tn=TnH[cand_t].to(dev)
print(f'pseudo={len(pseudo)} qimgs={len(qfn)} candidates={len(cand)} [{time.time()-t0:.0f}s]',flush=True)
def db(M): return (M-M.mean(0,keepdim=True))/(M.std(0,keepdim=True)+1e-6)
def stats(S,tag):
    o=S.argsort(1,descending=True); ranks=(o==gold.unsqueeze(1)).float().argmax(1)
    r={k:round((ranks<k).float().mean().item()*100,2) for k in (1,5,10,50,100)}
    print(f'{tag:34s} top1={r[1]:5.2f} r@5={r[5]:5.2f} r@10={r[10]:5.2f} r@50={r[50]:5.2f} r@100={r[100]:5.2f}',flush=True)
    return r
print('--- BASELINES (recall = candidate coverage) ---')
braw=stats(Q@Tt.t(),'taxon-H raw cosine')
bdb =stats(db(Q@Tt.t()),'taxon-H db (deployed unseen)')
stats(Q@Tn.t(),'name-H raw cosine')
print(f'>>> BASELINE r@100 (coverage wall) = raw {braw[100]:.2f} / db {bdb[100]:.2f}',flush=True)
print('--- ALL-BUT-THE-TOP (PCs from TEXT bank, applied to BOTH; raw-cosine recall) ---')
muT=Tt.mean(0,keepdim=True); _,_,Vt=torch.pca_lowrank((Tt-muT),q=24,niter=4)
for k in (1,2,4,8,16):
    Qa=Q-muT; Ta=Tt-muT
    Qa=Qa-(Qa@Vt[:,:k])@Vt[:,:k].t(); Ta=Ta-(Ta@Vt[:,:k])@Vt[:,:k].t()
    Qa=F.normalize(Qa,dim=-1); Ta=F.normalize(Ta,dim=-1)
    stats(Qa@Ta.t(),f'ABTT textPC k={k} (raw)')
    stats(db(Qa@Ta.t()),f'ABTT textPC k={k} (db)')
print('--- ALL-BUT-THE-TOP (PCs from QUERY images, applied to BOTH; raw-cosine recall) ---')
muQ=Q.mean(0,keepdim=True); _,_,Vq=torch.pca_lowrank((Q-muQ),q=24,niter=4)
for k in (1,2,4,8,16):
    Qa=Q-muQ; Ta=Tt-muQ
    Qa=Qa-(Qa@Vq[:,:k])@Vq[:,:k].t(); Ta=Ta-(Ta@Vq[:,:k])@Vq[:,:k].t()
    Qa=F.normalize(Qa,dim=-1); Ta=F.normalize(Ta,dim=-1)
    stats(Qa@Ta.t(),f'ABTT imgPC k={k} (raw)')
print('--- ECALP per-channel reweight 1-NN purity (GO if > ~36%, baseline ~29%) ---')
def purity(Z,tag):
    G=Z@Z.t(); G.fill_diagonal_(-2); nn=G.argmax(1)
    p=(gold[nn]==gold).float().mean().item()*100
    print(f'{tag:34s} 1-NN purity={p:5.2f}%',flush=True); return p
purity(Q,'raw H image')
stext=Tt.std(0); purity(F.normalize(Q*(1.0/(stext+1e-6)),dim=-1),'reweight 1/std_text')
sq=Q.std(0);    purity(F.normalize(Q*(1.0/(sq+1e-6)),dim=-1),'reweight 1/std_img')
print('DONE_CHEAP_UNSEEN',flush=True)
