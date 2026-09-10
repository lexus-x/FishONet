#!/bin/bash
# FAST parallel evaluation of multi-encoder ensemble combinations
# Uses all existing cached embeddings (no GPU needed for scoring)
# Goal: find if any new checkpoint gives v20 ensemble boost
source ~/miniconda3/etc/profile.d/conda.sh
conda activate onet
cd ~/onet

echo "=== PARALLEL ENSEMBLE SWEEP $(date) ==="
echo "Testing: ft_lora_v2b, ft_robust, cap336, capA in v20-style multi-enc ensemble"

python -c "
import json, torch
from collections import defaultdict
import torch.nn.functional as F
torch.set_num_threads(8)

# Load all checkpoints
def load(p):
    d = torch.load(p, weights_only=False)
    return {fn: i for i, fn in enumerate(d['files'])}, F.normalize(d['feats'].float(), dim=-1), list(d['files'])

def zc(M): return (M - M.mean()) / (M.std() + 1e-6)

# Text embeddings
txtH = torch.load('outputs/text_emb_h.pt', weights_only=False)
txtHt = torch.load('outputs/text_emb_h_taxon.pt', weights_only=False)
classes = txtH['classes']; ci = {c: i for i, c in enumerate(classes)}
TtH = F.normalize(txtHt['emb_taxon'].float(), dim=-1)

lab = json.load(open('data/dl/label_train.json'))

# Available image embeddings
emb_configs = {
    'ft': ('outputs/emb_train_ft.pt', None),
    'ft_robust': ('outputs/emb_train_ftrobust.pt', None),
    'cap': ('outputs/emb_train_cap.pt', None),
    'L': ('outputs/emb_train.pt', None),  # lineage (frozen BioCLIP)
    'h': ('outputs/emb_train_h.pt', None),
    'h_tta': ('outputs/emb_train_h_tta.pt', None),
    'h336': ('outputs/emb_train_h336.pt', None),
    'dino': ('outputs/emb_train_dino.pt', None),
}

# Load all
loaded = {}
for name, (path, _) in emb_configs.items():
    try:
        itemidx, feats, files = load(path)
        loaded[name] = (itemidx, feats, files)
        print(f'Loaded {name}: {len(files)} images, dim={feats.shape[1]}')
    except Exception as e:
        print(f'Failed {name}: {e}')

# Build train set
trainfiles = [fn for fn in loaded['ft'][2] 
              if fn in loaded['L'][0] and fn in loaded['cap'][0] and fn in lab and lab[fn] in ci]
by = defaultdict(list)
for fn in trainfiles:
    by[lab[fn]].append(fn)
seen = sorted(by.keys()); s2i = {c: i for i, c in enumerate(seen)}; S = len(seen)
TseenTax = torch.stack([TtH[ci[c]] for c in seen])

def protos_and_train(TrI, TrF, fns_by_cls, order):
    P = torch.zeros(S, TrF.shape[1]); cnt = torch.zeros(S); TF = []; TL = []
    for c in order:
        for fn in fns_by_cls[c]:
            f = TrF[TrI[fn]]; P[s2i[c]] += f; cnt[s2i[c]] += 1; TF.append(f); TL.append(s2i[c])
    P = F.normalize(P / cnt.clamp(min=1).unsqueeze(1), dim=-1)
    return P, torch.stack(TF), torch.tensor(TL)

def seen_score(qF, P, TF, TL, Tx=None, lam=0.0):
    out = torch.empty(qF.shape[0], S)
    for i in range(0, qF.shape[0], 1000):
        e = qF[i:i + 1000]; ps = e @ P.t(); sim = e @ TF.t()
        cmax = torch.full((e.shape[0], S), -1e9)
        cmax.scatter_reduce_(1, TL.unsqueeze(0).expand(e.shape[0], -1), sim, reduce='amax')
        sc = ps + 2.0 * cmax
        if Tx is not None and lam > 0: sc = sc + lam * (e @ Tx.t())
        out[i:i + 1000] = sc
    return out

# Holdout split (same as v20)
trby = defaultdict(list); valrows = []
for c in seen:
    fns = sorted(by[c])
    if len(fns) >= 3:
        k = max(1, round(0.2 * len(fns)))
        for f in fns[:-k]: trby[c].append(f)
        for f in fns[-k:]: valrows.append((f, s2i[c]))
    else:
        for f in fns: trby[c].append(f)

print(f'Seen classes: {S}, Val samples: {len(valrows)}')

# Test each encoder pair & ensemble
configs = [
    # (name, enc1, enc2, enc3, w1, w2, w3)
    ('v20_exact', 'ft', 'cap', 'L', 1.0, 1.0, 0.5),
    ('ft_robust_repl', 'ft_robust', 'cap', 'L', 1.0, 1.0, 0.5),
    ('h336_repl', 'h336', 'cap', 'L', 1.0, 1.0, 0.5),
    ('ft_robust_only', 'ft_robust', None, None, 1.0, 0, 0),
    ('ft_only', 'ft', None, None, 1.0, 0, 0),
    ('cap_only', 'cap', None, None, 1.0, 0, 0),
    ('h336_only', 'h336', None, None, 1.0, 0, 0),
    ('ft_robust+h336', 'ft_robust', 'h336', None, 1.0, 1.0, 0),
    ('ft+ft_robust', 'ft', 'ft_robust', None, 1.0, 1.0, 0),
    ('ft+ft_robust+cap', 'ft', 'ft_robust', 'cap', 0.5, 0.5, 1.0),
]

# Build val queries
VY = torch.tensor([y for _, y in valrows])

for name, e1, e2, e3, w1, w2, w3 in configs:
    try:
        # Protos + val features for enc1
        PF1, TFF1, TLF1 = protos_and_train(*loaded[e1][:2], trby, seen)
        vF1 = torch.stack([loaded[e1][1][loaded[e1][0][f]] for f, _ in valrows])
        sF1 = seen_score(vF1, PF1, TFF1, TLF1, TseenTax, 4.0)
        sc = zc(sF1) * w1
        
        if e2 and w2 > 0:
            PF2, TFF2, TLF2 = protos_and_train(*loaded[e2][:2], trby, seen)
            vF2 = torch.stack([loaded[e2][1][loaded[e2][0][f]] for f, _ in valrows])
            sF2 = seen_score(vF2, PF2, TFF2, TLF2, TseenTax, 4.0)
            sc = sc + zc(sF2) * w2
        
        if e3 and w3 > 0:
            PF3, TFF3, TLF3 = protos_and_train(*loaded[e3][:2], trby, seen)
            vF3 = torch.stack([loaded[e3][1][loaded[e3][0][f]] for f, _ in valrows])
            sF3 = seen_score(vF3, PF3, TFF3, TLF3)
            sc = sc + zc(sF3) * w3
        
        acc = sc.argmax(1).eq(VY).float().mean().item() * 100
        print(f'{name:25s}: holdout seen = {acc:.2f}%')
    except Exception as e:
        print(f'{name:25s}: ERROR - {e}')

print('\\nDONE - v20 reference: 87.94%')
" 2>&1

echo "=== DONE $(date) ==="