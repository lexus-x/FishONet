"""BREAKTHROUGH ATTEMPT for unseen: learn a text->image-centroid adapter on SEEN species,
transfer it zero-shot to UNSEEN text. Rules-legal (trained only on provided seen train data).
Validated on the hard-sim split: train adapter on 'known' 80%, test on pseudo-unseen rarest-20%.

Adapters tested: identity (baseline), ridge regression W, orthogonal Procrustes R, and a small MLP.
Metric: top-1 of pseudo-unseen images matched to adapted candidate text embeddings (+z-debias)."""
import json, pickle, torch
from collections import defaultdict
import torch.nn.functional as F

D = 'data/dl'
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
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
n_un = int(len(seen) * 0.2)
pseudo = [c for c in order[:n_un]]; known = [c for c in seen if c not in set(pseudo)]

def centroid(c):
    idx = torch.tensor([HI[fn] for fn in by[c]])
    return F.normalize(HF[idx].mean(0), dim=0)

# training pairs (known): text -> image centroid
Tk = torch.stack([TnH[ci[c]] for c in known])          # [Nk, d]
Pk = torch.stack([centroid(c) for c in known])         # [Nk, d]
print(f'known={len(known)} pseudo={len(pseudo)}  d={Tk.shape[1]}')

# pseudo-unseen eval: candidates = ALL non-known classes
known_set = set(known)
cand = [i for i, c in enumerate(classes) if c not in known_set]
cand_pos = {ci_: j for j, ci_ in enumerate(cand)}
cand_t = torch.tensor(cand)
Tcand = TnH[cand_t]                                     # [Ncand, d]
qfn, qy = [], []
for c in pseudo:
    for fn in by[c]: qfn.append(fn); qy.append(ci[c])
Q = torch.stack([HF[HI[fn]] for fn in qfn])            # [Nq, d]
gold = torch.tensor([cand_pos[y] for y in qy])
# true centroids of pseudo classes (to measure adapter transfer quality)
Pp = {ci[c]: centroid(c) for c in pseudo}

def db(M): return (M - M.mean(0, keepdim=True)) / (M.std(0, keepdim=True) + 1e-6)
def t1(S): return (S.argmax(1) == gold).float().mean().item() * 100
def evaltext(Tc, tag):
    Tc = F.normalize(Tc, dim=-1)
    S = Q @ Tc.t()
    print(f'  [{tag:22}] raw {t1(S):.2f}   debias {t1(db(S)):.2f}')
    # adapter transfer quality: cos(adapted text, true centroid) on pseudo
    cs = []
    for c in pseudo:
        cs.append(torch.dot(Tc[cand_pos[ci[c]]], Pp[ci[c]]).item())
    print(f'        mean cos(adapted_text, TRUE pseudo-centroid) = {sum(cs)/len(cs):.4f}')

print('\n=== BASELINE (raw BioCLIP text) ===')
evaltext(Tcand, 'identity')

print('\n=== RIDGE  W = (T^T T + lambda I)^-1 T^T P ===')
d = Tk.shape[1]; I = torch.eye(d)
TtT = Tk.t() @ Tk; TtP = Tk.t() @ Pk
for lam in (1.0, 10.0, 50.0, 200.0, 1000.0):
    W = torch.linalg.solve(TtT + lam * I, TtP)         # [d,d]
    # blend with identity (residual): adapted = (1-b)*T + b*(T@W)
    Pcand = Tcand @ W
    evaltext(Pcand, f'ridge lam={lam}')

print('\n=== PROCRUSTES (orthogonal R, T R ~ P) ===')
U, S_, Vt = torch.linalg.svd(Tk.t() @ Pk)
R = U @ Vt
evaltext(Tcand @ R, 'procrustes')

print('\n=== RESIDUAL BLEND ridge lam=50 with identity ===')
W = torch.linalg.solve(TtT + 50.0 * I, TtP)
for b in (0.25, 0.5, 0.75, 1.0):
    evaltext((1 - b) * Tcand + b * (Tcand @ W), f'(1-{b})I+{b}W')

print('\nREF: H-alone name+debias=22.35 ; deployed ensemble=23.99 ; real~0.5*sim')
