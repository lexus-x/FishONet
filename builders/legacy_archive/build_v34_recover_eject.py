"""v34: recover-on-eject — keep f=0.72 gate, but ejected images argmax over FULL text space.

WHY: hard unseen-ONLY pool zeros wrongly ejected seen images (1,654 tf @ f=0.72).
Allowing full-space text argmax on the eject route recovers ~25% of those to seen-cls
while still letting true novel images compete for unseen classes.

Holdout (zero-shot rarest-20%): hard 54.94% → eject_text_full 58.45% (+3.51pt).
Eval diagnostic: tf→seen 91.8→93.9%, uf→uns 53.5→45.4%.

Still single-pipeline: one gate, one rule, argmax over competition label space.
No folder-oracle. Sinkhorn dropped on eject route (plain argmax); optional sink variant
kept for ablation.

*** Holdout overstates real — submit only if budgeting a real probe. ***
"""
import json
import pickle
import zipfile
from collections import defaultdict

import torch
import torch.nn.functional as F

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_num_threads(8)
D = 'data/dl'
SEEN_FRAC = 0.72


def load(p):
    d = torch.load(p, weights_only=False)
    return {fn: i for i, fn in enumerate(d['files'])}, F.normalize(d['feats'].float(), dim=-1), list(d['files'])


def zc(M):
    return (M - M.mean()) / (M.std() + 1e-6)


def z1(v):
    return (v - v.mean()) / (v.std() + 1e-6)


def dbnorm(S, tc=0.05, tr=0.5):
    return F.log_softmax(S / tc, dim=0) + F.log_softmax(S / tr, dim=1)


def sinkhorn(logits, n_iter=50, tau=2.0):
    P = torch.softmax(logits / tau, dim=1)
    n, C = P.shape
    target_col = n / C
    for _ in range(n_iter):
        P = P / P.sum(dim=1, keepdim=True).clamp(min=1e-9)
        P = P * (target_col / P.sum(dim=0, keepdim=True).clamp(min=1e-9))
    return P


txtHt = torch.load('outputs/text_emb_h_taxon.pt', weights_only=False)
txtL = torch.load('outputs/text_emb.pt', weights_only=False)
classes = list(pickle.load(open(f'{D}/all_classes.pkl', 'rb')))
ci = {c: i for i, c in enumerate(classes)}
NCLS = len(classes)
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
    s = set(files)
    common = s if common is None else (common & s)
common = [fn for fn in common if fn in lab and lab[fn] in ci]
by = defaultdict(list)
for fn in common:
    by[lab[fn]].append(fn)
seen = sorted(by.keys())
s2i = {c: i for i, c in enumerate(seen)}
S = len(seen)
kept_idx = torch.tensor([ci[c] for c in seen]).to(dev)
other_idx = torch.tensor([i for i in range(NCLS) if classes[i] not in set(seen)]).to(dev)
TseenTax = TtH[kept_idx]
LAM = 4.0
print(f'seen classes: {S}  unseen cols: {len(other_idx)}')


def protos(idx, feats):
    P = torch.zeros(S, feats.shape[1])
    cnt = torch.zeros(S)
    TF, TL = [], []
    for c in seen:
        for fn in by[c]:
            f = feats[idx[fn]]
            P[s2i[c]] += f
            cnt[s2i[c]] += 1
            TF.append(f)
            TL.append(s2i[c])
    P = F.normalize(P / cnt.clamp(min=1).unsqueeze(1), dim=-1)
    return P.to(dev), torch.stack(TF).to(dev), torch.tensor(TL).to(dev)


PR = {t: protos(*train[t][:2]) for t, _, _ in MEMBERS}


def seen_score(qF, P, TF, TL, hspace):
    qF = qF.to(dev)
    n = qF.shape[0]
    out = torch.empty(n, S, device=dev)
    for i in range(0, n, 2000):
        e = qF[i:i + 2000]
        ps = e @ P.t()
        sim = e @ TF.t()
        cmax = torch.full((e.shape[0], S), -1e9, device=dev)
        cmax.scatter_reduce_(1, TL.unsqueeze(0).expand(e.shape[0], -1), sim, reduce='amax')
        sc = ps + 2.0 * cmax
        if hspace:
            sc = sc + LAM * (e @ TseenTax.t())
        out[i:i + 2000] = sc
    return out


test = {t: load(f'outputs/{TRAIN[t].replace("emb_train", "emb_test")}.pt') for t, _, _ in MEMBERS}
unseen = {t: load(f'outputs/{TRAIN[t].replace("emb_train", "emb_unseen")}.pt') for t, _, _ in MEMBERS}
tf = sorted(set.intersection(*[set(f) for (_, _, f) in test.values()]))
uf = sorted(set.intersection(*[set(f) for (_, _, f) in unseen.values()]))
all_files = tf + uf
print(f'combined eval batch: {len(all_files)}')


def qcat(t):
    ti, tfeat, _ = test[t]
    ui, ufeat, _ = unseen[t]
    return torch.cat([torch.stack([tfeat[ti[fn]] for fn in tf]),
                      torch.stack([ufeat[ui[fn]] for fn in uf])])


Q = {t: qcat(t).to(dev) for t, _, _ in MEMBERS}
seen_block = torch.zeros(len(all_files), S, device=dev)
for t, hs, w in MEMBERS:
    if w == 0.0:
        continue
    seen_block = seen_block + w * zc(seen_score(Q[t], *PR[t], hs))
text_full = (dbnorm(Q['ctftbig'] @ TtH.t()) + 0.5 * dbnorm(Q['L'] @ TnL.t())
             + 0.75 * dbnorm(Q['fullft336_v2'] @ TtH.t()) + 1.0 * dbnorm(Q['ftshift'] @ TtH.t())
             + 1.0 * dbnorm(Q['ctftbig'] @ TTX.t()))
text_unseen_only = text_full[:, other_idx]
img_seenmax = seen_block.max(1).values
text_margin = ((Q['ctftbig'] @ TtH[kept_idx].t()).max(1).values
               - (Q['ctftbig'] @ TtH[other_idx].t()).max(1).values)
combined = z1(img_seenmax) + 2.0 * z1(text_margin)

k_seen = int(round(SEEN_FRAC * len(all_files)))
thr = torch.topk(combined, k_seen).values.min()
route_seen = combined >= thr
idx_uns = (~route_seen).nonzero(as_tuple=True)[0]
print(f'f={SEEN_FRAC}: routed_unseen={len(idx_uns)}')

seen_set = set(seen)


def write_sub(tag, pred_idx):
    preds = {fn: classes[j] for fn, j in zip(all_files, pred_idx.tolist())}
    assert len(preds) == 35665 and all(c in ci for c in preds.values())
    cov = len(set(p for p in preds.values() if p not in seen_set))
    o_t = sum(1 for fn in tf if preds[fn] not in seen_set)
    o_u = sum(1 for fn in uf if preds[fn] not in seen_set)
    json.dump(preds, open(f'outputs/prediction_{tag}.json', 'w'))
    z = zipfile.ZipFile(f'outputs/submission_{tag}.zip', 'w', zipfile.ZIP_DEFLATED)
    z.write(f'outputs/prediction_{tag}.json', arcname='prediction.json')
    z.close()
    # also copy to submissions/
    import shutil
    shutil.copy(f'outputs/submission_{tag}.zip', f'submissions/submission_{tag}.zip')
    print(f'{tag}: unseen-class coverage {cov}/{len(other_idx)} ({100*cov/len(other_idx):.1f}%) | '
          f'test-eval {100*(1-o_t/len(tf)):.1f}% seen-cols | '
          f'unseen-eval {100*o_u/len(uf):.1f}% other-cols -> submission_{tag}.zip', flush=True)


# primary: eject → text_full argmax
pred = torch.empty(len(all_files), dtype=torch.long, device=dev)
pred[route_seen] = kept_idx[seen_block[route_seen].argmax(1)]
pred[idx_uns] = text_full[idx_uns].argmax(1)
write_sub('v34_recover_textfull', pred)

# ablation: eject → soft b=1.1 (best holdout overall among soft recover)
def row_z(M):
    return (M - M.mean(1, keepdim=True)) / (M.std(1, keepdim=True) + 1e-6)


sb = zc(seen_block)
tu = row_z(text_unseen_only)
full = torch.full((len(all_files), NCLS), -1e9, device=dev)
full[:, kept_idx] = sb
full[:, other_idx] = 1.1 * tu
pred = torch.empty(len(all_files), dtype=torch.long, device=dev)
pred[route_seen] = kept_idx[seen_block[route_seen].argmax(1)]
pred[idx_uns] = full[idx_uns].argmax(1)
write_sub('v34_recover_soft11', pred)

# ablation: recover seen picks from text_full, else Sinkhorn on unseen pool
pred = torch.empty(len(all_files), dtype=torch.long, device=dev)
pred[route_seen] = kept_idx[seen_block[route_seen].argmax(1)]
tf_pred = text_full[idx_uns].argmax(1)
seen_set_t = torch.zeros(NCLS, dtype=torch.bool, device=dev)
seen_set_t[kept_idx] = True
is_seen_pick = seen_set_t[tf_pred]
pred[idx_uns] = tf_pred
need_sink = idx_uns[~is_seen_pick]
if len(need_sink):
    pred[need_sink] = other_idx[sinkhorn(text_unseen_only[need_sink]).argmax(1)]
write_sub('v34_recover_text_sink', pred)
