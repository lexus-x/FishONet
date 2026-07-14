"""Probe the TRANSDUCTIVE ceiling for unseen.
(A) hard-sim: per-image top1 vs per-CLASS-CENTROID top1 (oracle clustering) -> upper bound of averaging.
(B) kNN-averaging top1 (realistic, label-free): average each query with its k nearest unseen neighbors.
(C) real unseen test cluster structure: nearest-neighbor cosine distribution -> do species repeat?"""
import json, pickle, torch
from collections import defaultdict
import torch.nn.functional as F

D = 'data/dl'
classes = list(pickle.load(open(f'{D}/all_classes.pkl', 'rb')))
ci = {c: i for i, c in enumerate(classes)}
TnH = F.normalize(torch.load('outputs/text_emb_h.pt', weights_only=False)['emb_name'].float(), dim=-1)
tr = torch.load('outputs/emb_train_h.pt', weights_only=False)
HI = {fn: i for i, fn in enumerate(tr['files'])}
HF = F.normalize(tr['feats'].float(), dim=-1)
lab = json.load(open(f'{D}/label_train.json'))
by = defaultdict(list)
for fn in tr['files']:
    if fn in lab and lab[fn] in ci: by[lab[fn]].append(fn)
seen = sorted(by.keys())
order = sorted(seen, key=lambda c: len(by[c]))
pseudo = [c for c in order[:int(len(seen)*0.2)]]; known = set(c for c in seen if c not in set(pseudo))
cand = [i for i, c in enumerate(classes) if c not in known]
cand_pos = {ci_: j for j, ci_ in enumerate(cand)}; cand_t = torch.tensor(cand)
Tc = TnH[cand_t]
def db(M): return (M - M.mean(0, keepdim=True)) / (M.std(0, keepdim=True) + 1e-6)

# (A) per-image vs per-class-centroid (oracle clustering)
qfn, qy, qc = [], [], []
for c in pseudo:
    for fn in by[c]: qfn.append(fn); qy.append(cand_pos[ci[c]]); qc.append(c)
Q = torch.stack([HF[HI[fn]] for fn in qfn]); gold = torch.tensor(qy)
Simg = db(Q @ Tc.t())
print(f'pseudo classes={len(pseudo)} imgs={len(qfn)}')
print(f'(A) per-IMAGE  top1 = {(Simg.argmax(1)==gold).float().mean()*100:.2f}')
# centroid per pseudo class
cent = {c: F.normalize(torch.stack([HF[HI[fn]] for fn in by[c]]).mean(0), dim=0) for c in pseudo}
Cc = torch.stack([cent[c] for c in pseudo]); Cy = torch.tensor([cand_pos[ci[c]] for c in pseudo])
Scen = db(Cc @ Tc.t())
print(f'(A) per-CENTROID top1 = {(Scen.argmax(1)==Cy).float().mean()*100:.2f}  <-- ORACLE clustering ceiling')
# weighted to images (each centroid counts for its #imgs, like real per-image scoring)
hit = (Scen.argmax(1)==Cy)
img_weighted = sum(hit[i].item()*len(by[c]) for i,c in enumerate(pseudo))/len(qfn)*100
print(f'(A) per-CENTROID (image-weighted) = {img_weighted:.2f}')

# (B) realistic kNN-averaging: average each query with its k nearest neighbors among ALL queries
G = Q @ Q.t()                                  # query-query sim
for k in (3, 5, 10, 20):
    nbr = G.topk(k+1, dim=1).indices           # includes self
    Qavg = F.normalize(Q[nbr].mean(1), dim=-1)
    S = db(Qavg @ Tc.t())
    # purity of the neighborhoods (fraction of neighbors sharing the true class)
    gold_nbr = gold[nbr]
    purity = (gold_nbr == gold.unsqueeze(1)).float().mean().item()*100
    print(f'(B) kNN-avg k={k:<2} top1 = {(S.argmax(1)==gold).float().mean()*100:.2f}   neighbor purity={purity:.1f}%')

# (C) real unseen test cluster structure
un = torch.load('outputs/emb_unseen_h_tta.pt', weights_only=False)
U = F.normalize(un['feats'].float(), dim=-1)
print(f'\n(C) real unseen test images = {U.shape[0]}')
# nearest-neighbor cosine (exclude self) in chunks
N = U.shape[0]; nn1 = torch.empty(N)
for s in range(0, N, 2000):
    g = U[s:s+2000] @ U.t();
    for r in range(g.shape[0]): g[r, s+r] = -1
    nn1[s:s+2000] = g.max(1).values
for thr in (0.95, 0.90, 0.85, 0.80, 0.75):
    print(f'    frac with NN cosine > {thr} = {(nn1>thr).float().mean()*100:.1f}%')
print(f'    median NN cosine = {nn1.median():.3f}   mean = {nn1.mean():.3f}')
