"""v39: v37 + orthogonal TaxaBind iNat image-prototype leg on the UNSEEN route.

Keeps gate + seen route identical to v37. Unseen stack:
  v36 text (incl. TaxaBind @ taxctx) + IMG_W * dbnorm(ctftshift Q @ ctftshift iNat protos)
  + TB_IMG_W * dbnorm(TaxaBind Q @ TaxaBind iNat protos), unseen cols only.

Proxy (research/inat_taxabind_stack_proxy.py): v37 ref S0+3*ctft=43.53 -> +0.5 TB -> 44.05 (+0.52);
class-disjoint CV +0.31 on covered gold. Projected real ~50.5% at ~0.12 overall-pt per proxy-pt (v37 calib).
"""
import json
import os
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
IMG_W = float(os.environ.get('IMG_W', '3.0'))
TB_IMG_W = float(os.environ.get('TB_IMG_W', '0.5'))
PROTO_PATH = os.environ.get('PROTO_PATH', 'outputs/inat_protos_ctftshift_full.pt')
TB_PROTO_PATH = os.environ.get('TB_PROTO_PATH', 'outputs/inat_protos_taxabind.pt')


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


def proto_unseen_leg(Qmat, proto_path, classes, other_idx):
    pd = torch.load(proto_path, weights_only=False)
    proto_classes = pd['classes']
    P_src = F.normalize(pd['protos'].float(), dim=-1)
    name2row = {c: i for i, c in enumerate(proto_classes)}
    dimp = P_src.shape[1]
    NCLS = len(classes)
    protoC = torch.zeros(NCLS, dimp)
    hasp = torch.zeros(NCLS, dtype=torch.bool)
    for i, c in enumerate(classes):
        r = name2row.get(c)
        if r is not None and float(P_src[r].norm()) > 0.5:
            protoC[i] = P_src[r]
            hasp[i] = True
    protoC = protoC.to(dev)
    hasp = hasp.to(dev)
    Po = protoC[other_idx]
    haso = hasp[other_idx]
    S = Qmat @ Po.t()
    S = S.masked_fill(~haso.unsqueeze(0), -1e4)
    return dbnorm(S), int(hasp.sum()), int(haso.sum())


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
tb_test = load('outputs/emb_test_taxabind.pt')
tb_uns = load('outputs/emb_unseen_taxabind.pt')
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

img_leg, n_cov, n_uns_cov = proto_unseen_leg(Q['ctftshift'], PROTO_PATH, classes, other_idx)
text_unseen_only = text_unseen_only + IMG_W * img_leg
print(f'ctftshift iNat protos: {n_cov}/{NCLS} classes; unseen-route {n_uns_cov}/{len(other_idx)} '
      f'({100 * n_uns_cov / len(other_idx):.1f}%); IMG_W={IMG_W}')

tb_leg, n_tb_cov, n_tb_uns = proto_unseen_leg(Qtb, TB_PROTO_PATH, classes, other_idx)
text_unseen_only = text_unseen_only + TB_IMG_W * tb_leg
print(f'TaxaBind iNat protos: {n_tb_cov}/{NCLS} classes; unseen-route {n_tb_uns}/{len(other_idx)} '
      f'({100 * n_tb_uns / len(other_idx):.1f}%); TB_IMG_W={TB_IMG_W}')

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
tag = f'v39_inat_tbimg_w{IMG_W:g}_{TB_IMG_W:g}_sink72'
json.dump(preds, open(f'outputs/prediction_{tag}.json', 'w'))
z = zipfile.ZipFile(f'outputs/submission_{tag}.zip', 'w', zipfile.ZIP_DEFLATED)
z.write(f'outputs/prediction_{tag}.json', arcname='prediction.json')
z.close()
os.makedirs('submissions', exist_ok=True)
shutil.copy(f'outputs/submission_{tag}.zip', f'submissions/submission_{tag}.zip')
print(f'{tag}: coverage {cov}/{len(other_idx)} ({100*cov/len(other_idx):.1f}%) | '
      f'test {100*(1-o_t/len(tf)):.1f}% seen-cols | unseen {100*o_u/len(uf):.1f}% other-cols',
      flush=True)
print(f'wrote submissions/submission_{tag}.zip')
