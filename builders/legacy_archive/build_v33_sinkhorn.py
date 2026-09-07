"""v33: Sinkhorn balanced assignment on the unseen route + re-optimized routing fraction.

WHY (corrects an earlier wrong "dead" verdict): the unseen eval is 15,568 images over 11,598 classes
(~1.34/class, i.e. nearly every class present, roughly uniform). Per-image argmax predicts only
**38.6% of unseen classes** -- 61.4% are never proposed at all, while a few get proposed up to 17x.
That is a coverage failure, not a head-collapse (top-20 absorb just 1.9%), which is why the earlier
collapse check missed it.

Sinkhorn fixes it by balancing the assignment so each class receives ~n_routed/n_classes mass.
Holdout simulation of the real condition (pool restricted to the 1159 pseudo-unseen classes,
2318 imgs = 2.0/class uniform): per-image argmax 51.90 -> **Sinkhorn 56.47 (+4.57)**, coverage
84.6% -> 94% (outputs/unseen_sinkhorn_test.log).

Because Sinkhorn raises b, ejecting images to the unseen head becomes MORE valuable, so the optimal
routing fraction shifts back DOWN from 0.72. Hence the f sweep here.

*** COMPLIANCE NOTE ***
Sinkhorn is TRANSDUCTIVE: it couples predictions across the eval batch (a class "used up" by one
image is less available to another). This is strictly more transductive than v31, which was frozen
to be per-image independent. The pipeline is still SINGLE (one uniform rule, no folder-oracle
routing, one argmax over the full label space), but it is no longer per-image independent.
CONFIRM AGAINST COMPETITION RULES before submitting. If transduction is disallowed, use
builders/build_v31_strict_singlepipeline.py (47.69%) instead.
"""
import json, pickle, torch, zipfile
from collections import defaultdict
import torch.nn.functional as F

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_num_threads(8)
D = 'data/dl'
SEEN_FRACS = [0.55, 0.62, 0.72]     # Sinkhorn raises b -> optimum shifts down from 0.72
SINK_TAU, SINK_ITER = 2.0, 50       # best on holdout

def load(p):
    d = torch.load(p, weights_only=False)
    return {fn: i for i, fn in enumerate(d['files'])}, F.normalize(d['feats'].float(), dim=-1), list(d['files'])
def zc(M): return (M - M.mean()) / (M.std() + 1e-6)
def z1(v): return (v - v.mean()) / (v.std() + 1e-6)
def dbnorm(S, tc=0.05, tr=0.5): return F.log_softmax(S / tc, dim=0) + F.log_softmax(S / tr, dim=1)

txtHt = torch.load('outputs/text_emb_h_taxon.pt', weights_only=False)
txtL = torch.load('outputs/text_emb.pt', weights_only=False)
classes = list(pickle.load(open(f'{D}/all_classes.pkl', 'rb')))
ci = {c: i for i, c in enumerate(classes)}; NCLS = len(classes)
TtH = F.normalize(txtHt['emb_taxon'].float(), dim=-1).to(dev)
TnL = F.normalize(txtL['emb_name'].float(), dim=-1).to(dev)
_pe = torch.load('outputs/text_emb_h_promptens.pt', weights_only=False)
TTX = F.normalize(_pe['emb_taxctx'].float(), dim=-1).to(dev)
lab = json.load(open(f'{D}/label_train.json'))

MEMBERS = [('ctftbig', True, 1.0), ('ftshift', True, 2.5), ('fullft336shift', True, 2.5),
           ('L', False, 0.0), ('fullft336_v2', True, 0.0)]
TRAIN = {'ctftbig': 'emb_train_ctftbig', 'ftshift': 'emb_train_ftshift',
         'fullft336shift': 'emb_train_fullft336shift', 'L': 'emb_train',
         'fullft336_v2': 'emb_train_fullft336_v2'}
train = {t: load(f'outputs/{TRAIN[t]}.pt') for t, _, _ in MEMBERS}
common = None
for t, (idx, _, files) in train.items():
    s = set(files); common = s if common is None else (common & s)
common = [fn for fn in common if fn in lab and lab[fn] in ci]
by = defaultdict(list)
for fn in common: by[lab[fn]].append(fn)
seen = sorted(by.keys()); s2i = {c: i for i, c in enumerate(seen)}; S = len(seen)
kept_idx = torch.tensor([ci[c] for c in seen]).to(dev)
other_idx = torch.tensor([i for i in range(NCLS) if classes[i] not in set(seen)]).to(dev)
TseenTax = TtH[kept_idx]; LAM = 4.0
print(f'seen classes: {S}  unseen cols: {len(other_idx)}')

def protos(idx, feats):
    P = torch.zeros(S, feats.shape[1]); cnt = torch.zeros(S); TF, TL = [], []
    for c in seen:
        for fn in by[c]:
            f = feats[idx[fn]]; P[s2i[c]] += f; cnt[s2i[c]] += 1; TF.append(f); TL.append(s2i[c])
    P = F.normalize(P / cnt.clamp(min=1).unsqueeze(1), dim=-1)
    return P.to(dev), torch.stack(TF).to(dev), torch.tensor(TL).to(dev)
PR = {t: protos(*train[t][:2]) for t, _, _ in MEMBERS}

def seen_score(qF, P, TF, TL, hspace):
    qF = qF.to(dev); n = qF.shape[0]; out = torch.empty(n, S, device=dev)
    for i in range(0, n, 2000):
        e = qF[i:i + 2000]; ps = e @ P.t(); sim = e @ TF.t()
        cmax = torch.full((e.shape[0], S), -1e9, device=dev)
        cmax.scatter_reduce_(1, TL.unsqueeze(0).expand(e.shape[0], -1), sim, reduce='amax')
        sc = ps + 2.0 * cmax
        if hspace: sc = sc + LAM * (e @ TseenTax.t())
        out[i:i + 2000] = sc
    return out

test = {t: load(f'outputs/{TRAIN[t].replace("emb_train","emb_test")}.pt') for t, _, _ in MEMBERS}
unseen = {t: load(f'outputs/{TRAIN[t].replace("emb_train","emb_unseen")}.pt') for t, _, _ in MEMBERS}
tf = sorted(set.intersection(*[set(f) for (_, _, f) in test.values()]))
uf = sorted(set.intersection(*[set(f) for (_, _, f) in unseen.values()]))
all_files = tf + uf
print(f'combined eval batch: {len(all_files)}')
def qcat(t):
    ti, tfeat, _ = test[t]; ui, ufeat, _ = unseen[t]
    return torch.cat([torch.stack([tfeat[ti[fn]] for fn in tf]),
                      torch.stack([ufeat[ui[fn]] for fn in uf])])
Q = {t: qcat(t).to(dev) for t, _, _ in MEMBERS}

seen_block = torch.zeros(len(all_files), S, device=dev)
for t, hs, w in MEMBERS:
    if w == 0.0: continue
    seen_block = seen_block + w * zc(seen_score(Q[t], *PR[t], hs))
text_full = (dbnorm(Q['ctftbig'] @ TtH.t()) + 0.5 * dbnorm(Q['L'] @ TnL.t())
             + 0.75 * dbnorm(Q['fullft336_v2'] @ TtH.t()) + 1.0 * dbnorm(Q['ftshift'] @ TtH.t())
             + 1.0 * dbnorm(Q['ctftbig'] @ TTX.t()))
text_unseen_only = text_full[:, other_idx]

img_seenmax = seen_block.max(1).values
text_margin = (Q['ctftbig'] @ TtH[kept_idx].t()).max(1).values - (Q['ctftbig'] @ TtH[other_idx].t()).max(1).values
combined = z1(img_seenmax) + 2.0 * z1(text_margin)

def sinkhorn(logits, n_iter=SINK_ITER, tau=SINK_TAU):
    """Balance assignment: each image mass 1, each class ~n/C. Operates on the routed subset only."""
    P = torch.softmax(logits / tau, dim=1)
    n, C = P.shape
    target_col = n / C
    for _ in range(n_iter):
        P = P / P.sum(dim=1, keepdim=True).clamp(min=1e-9)
        P = P * (target_col / P.sum(dim=0, keepdim=True).clamp(min=1e-9))
    return P

seen_set = set(seen)
for SEEN_FRAC in SEEN_FRACS:
    k_seen = int(round(SEEN_FRAC * len(all_files)))
    thr = torch.topk(combined, k_seen).values.min()
    route_seen = combined >= thr
    idx_uns = (~route_seen).nonzero(as_tuple=True)[0]

    pred_idx = torch.empty(len(all_files), dtype=torch.long, device=dev)
    pred_idx[route_seen] = kept_idx[seen_block[route_seen].argmax(1)]
    # ---- Sinkhorn over the routed-unseen subset ----
    sub = text_unseen_only[idx_uns]
    P = sinkhorn(sub)
    pred_idx[idx_uns] = other_idx[P.argmax(1)]

    preds = {fn: classes[j] for fn, j in zip(all_files, pred_idx.tolist())}
    assert len(preds) == 35665 and all(c in ci for c in preds.values())
    cov = len(set(p for p in preds.values() if p not in seen_set))
    o_t = sum(1 for fn in tf if preds[fn] not in seen_set)
    o_u = sum(1 for fn in uf if preds[fn] not in seen_set)
    tag = f'v33_sink{int(SEEN_FRAC*100)}'
    json.dump(preds, open(f'outputs/prediction_{tag}.json', 'w'))
    z = zipfile.ZipFile(f'outputs/submission_{tag}.zip', 'w', zipfile.ZIP_DEFLATED)
    z.write(f'outputs/prediction_{tag}.json', arcname='prediction.json'); z.close()
    print(f'f={SEEN_FRAC}: routed_unseen={len(idx_uns)} | unseen-class coverage {cov}/{len(other_idx)} '
          f'({100*cov/len(other_idx):.1f}%) | test-eval {100*(1-o_t/len(tf)):.1f}% seen-cols | '
          f'unseen-eval {100*o_u/len(uf):.1f}% other-cols -> submission_{tag}.zip', flush=True)
