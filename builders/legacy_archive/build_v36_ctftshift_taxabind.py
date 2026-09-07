"""v36: v33 but ctftbig → ctftshift (shift_aug LoRA) + TaxaBind w=1.0 on unseen text.

Proxy hard-sim deployed-ish: 29.38 → 33.35 (+3.97) with dep_shift+1.0*TaxaBind
(outputs/stack_shift_tb_results.json). Nearly additive vs shift alone +2.07 / TB alone +1.86.

Perfect-gate ceiling with proportional b≈18.05% → ~53.6% overall.
Current gate efficiency ~90.7% would only realize ~48.7% — still a real probe.

Single-pipeline; disclose TaxaBind + shift_aug retrain.
"""
import json
import pickle
import shutil
import zipfile
from collections import defaultdict

import torch
import torch.nn.functional as F

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_num_threads(8)
D = 'data/dl'
SEEN_FRAC = 0.72
SINK_TAU, SINK_ITER = 2.0, 50
TAXABIND_W = 1.0


def load(p):
    d = torch.load(p, weights_only=False)
    return {fn: i for i, fn in enumerate(d['files'])}, F.normalize(d['feats'].float(), dim=-1), list(d['files'])


def zc(M):
    return (M - M.mean()) / (M.std() + 1e-6)


def z1(v):
    return (v - v.mean()) / (v.std() + 1e-6)


def dbnorm(S, tc=0.05, tr=0.5):
    return F.log_softmax(S / tc, dim=0) + F.log_softmax(S / tr, dim=1)


def sinkhorn(logits, n_iter=SINK_ITER, tau=SINK_TAU):
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
Ttb = F.normalize(
    torch.load('outputs/text_emb_taxabind_taxctx.pt', weights_only=False)['emb_taxctx'].float(),
    dim=-1,
).to(dev)
lab = json.load(open(f'{D}/label_train.json'))

# ctftshift replaces ctftbig; keep L + fullft336_v2 for text legs
MEMBERS = [('ctftshift', True, 1.0), ('ftshift', True, 2.5), ('fullft336shift', True, 2.5),
           ('L', False, 0.0), ('fullft336_v2', True, 0.0)]
TRAIN = {'ctftshift': 'emb_train_ctftshift', 'ftshift': 'emb_train_ftshift',
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
# taxabind eval emb
tb_test = load('outputs/emb_test_taxabind.pt')
tb_uns = load('outputs/emb_unseen_taxabind.pt')
tf = sorted(set.intersection(*[set(f) for (_, _, f) in test.values()] + [set(tb_test[2]), set(tb_uns[2])]))
uf = sorted(set.intersection(*[set(f) for (_, _, f) in unseen.values()] + [set(tb_test[2]), set(tb_uns[2])]))
# fix: tb_test files are test only
tf = sorted(set.intersection(*[set(f) for (_, _, f) in test.values()], set(tb_test[2])))
uf = sorted(set.intersection(*[set(f) for (_, _, f) in unseen.values()], set(tb_uns[2])))
all_files = tf + uf
print(f'combined eval batch: {len(all_files)}')


def qcat(t):
    ti, tfeat, _ = test[t]
    ui, ufeat, _ = unseen[t]
    return torch.cat([torch.stack([tfeat[ti[fn]] for fn in tf]),
                      torch.stack([ufeat[ui[fn]] for fn in uf])])


Q = {t: qcat(t).to(dev) for t, _, _ in MEMBERS}
Qtb = torch.cat([
    torch.stack([tb_test[1][tb_test[0][fn]] for fn in tf]),
    torch.stack([tb_uns[1][tb_uns[0][fn]] for fn in uf]),
]).to(dev)

seen_block = torch.zeros(len(all_files), S, device=dev)
for t, hs, w in MEMBERS:
    if w == 0.0:
        continue
    seen_block = seen_block + w * zc(seen_score(Q[t], *PR[t], hs))

text_full = (dbnorm(Q['ctftshift'] @ TtH.t()) + 0.5 * dbnorm(Q['L'] @ TnL.t())
             + 0.75 * dbnorm(Q['fullft336_v2'] @ TtH.t()) + 1.0 * dbnorm(Q['ftshift'] @ TtH.t())
             + 1.0 * dbnorm(Q['ctftshift'] @ TTX.t())
             + TAXABIND_W * dbnorm(Qtb @ Ttb.t()))
text_unseen_only = text_full[:, other_idx]

img_seenmax = seen_block.max(1).values
text_margin = ((Q['ctftshift'] @ TtH[kept_idx].t()).max(1).values
               - (Q['ctftshift'] @ TtH[other_idx].t()).max(1).values)
combined = z1(img_seenmax) + 2.0 * z1(text_margin)

k_seen = int(round(SEEN_FRAC * len(all_files)))
thr = torch.topk(combined, k_seen).values.min()
route_seen = combined >= thr
idx_uns = (~route_seen).nonzero(as_tuple=True)[0]
print(f'f={SEEN_FRAC}: routed_unseen={len(idx_uns)}')

pred_idx = torch.empty(len(all_files), dtype=torch.long, device=dev)
pred_idx[route_seen] = kept_idx[seen_block[route_seen].argmax(1)]
sub = text_unseen_only[idx_uns]
pred_idx[idx_uns] = other_idx[sinkhorn(sub).argmax(1)]

seen_set = set(seen)
preds = {fn: classes[j] for fn, j in zip(all_files, pred_idx.tolist())}
assert len(preds) == 35665 and all(c in ci for c in preds.values())
cov = len(set(p for p in preds.values() if p not in seen_set))
o_t = sum(1 for fn in tf if preds[fn] not in seen_set)
o_u = sum(1 for fn in uf if preds[fn] not in seen_set)
tag = 'v36_ctftshift_tb_sink72'
json.dump(preds, open(f'outputs/prediction_{tag}.json', 'w'))
z = zipfile.ZipFile(f'outputs/submission_{tag}.zip', 'w', zipfile.ZIP_DEFLATED)
z.write(f'outputs/prediction_{tag}.json', arcname='prediction.json')
z.close()
shutil.copy(f'outputs/submission_{tag}.zip', f'submissions/submission_{tag}.zip')
print(f'{tag}: coverage {cov}/{len(other_idx)} ({100*cov/len(other_idx):.1f}%) | '
      f'test {100*(1-o_t/len(tf)):.1f}% seen-cols | unseen {100*o_u/len(uf):.1f}% other-cols',
      flush=True)
