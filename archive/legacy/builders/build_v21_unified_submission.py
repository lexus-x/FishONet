"""Build v21 unified (compliant single-pipeline) submission.
ONE score matrix over ALL 17393 classes, ONE argmax, for EVERY eval image
(test/seen-eval 20097 + unseen-eval 15568 = 35665), using the best_compliant
config found by unified_holdout.py (variant a, flat calibrated-stacking).

CAVEAT (documented, not a routing rule): unseen-eval images only have L/H
embeddings cached (no ft/cap extracted for that folder). The seen-block ft/cap
terms are simply absent for those images (L+H only) -- same formula, fewer
available backbones, not a decision based on which folder an image "is".
Test-eval images get the full ft+cap+L+H ensemble.
"""
import json, pickle, torch, zipfile
import torch.nn.functional as F
from collections import defaultdict

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_num_threads(8)
D = 'data/dl'

CFG = json.load(open('outputs/unified_frontier.json'))
best_cfg = CFG['best_compliant']['config']
print('using config:', best_cfg)
# parse "variant=a_wtext0.5 gamma=10"
variant_tag, gamma_tag = best_cfg.split(' ')
variant = variant_tag.split('=')[1].split('_wtext')[0]
w_text = float(variant_tag.split('_wtext')[1])
gamma = float(gamma_tag.split('=')[1])
print(f'variant={variant} w_text={w_text} gamma={gamma}')
assert variant == 'a', 'submission builder only implements flat calibrated-stacking (a); adjust if best_compliant picked b/c'


def load(p):
    d = torch.load(p, weights_only=False)
    return {fn: i for i, fn in enumerate(d['files'])}, F.normalize(d['feats'].float(), dim=-1), list(d['files'])


def zc(M):
    return (M - M.mean()) / (M.std() + 1e-6)


def dbnorm(S, tc=0.05, tr=0.5):
    return F.log_softmax(S / tc, dim=0) + F.log_softmax(S / tr, dim=1)


txtHt = torch.load('outputs/text_emb_h_taxon.pt', weights_only=False)
txtH = torch.load('outputs/text_emb_h.pt', weights_only=False)
classes = list(pickle.load(open(f'{D}/all_classes.pkl', 'rb')))
assert txtHt['classes'] == classes and txtH['classes'] == classes
ci = {c: i for i, c in enumerate(classes)}
NCLS = len(classes)
TtH = F.normalize(txtHt['emb_taxon'].float(), dim=-1).to(dev)
TnH = F.normalize(txtH['emb_name'].float(), dim=-1).to(dev)

lab = json.load(open(f'{D}/label_train.json'))

# ---- ALL seen classes get prototypes now (no pseudo-unseen holdout carve-out) ----
FtI, FtF, Ftf = load('outputs/emb_train_ft.pt')
CtI, CtF, Ctf = load('outputs/emb_train_cap.pt')
LtI, LtF, Ltf = load('outputs/emb_train.pt')

common = [fn for fn in Ftf if fn in CtI and fn in LtI and fn in lab and lab[fn] in ci]
by = defaultdict(list)
for fn in common:
    by[lab[fn]].append(fn)
seen = sorted(by.keys())
s2i = {c: i for i, c in enumerate(seen)}
S = len(seen)
print(f'seen classes: {S}  train imgs: {len(common)}')
kept_idx = torch.tensor([ci[c] for c in seen]).to(dev)
other_idx = torch.tensor([i for i in range(NCLS) if classes[i] not in set(seen)]).to(dev)
TseenTax = TtH[kept_idx]


def protos_and_train(FI, FF, order):
    P = torch.zeros(S, FF.shape[1])
    cnt = torch.zeros(S)
    TF, TL = [], []
    for c in order:
        for fn in by[c]:
            f = FF[FI[fn]]
            P[s2i[c]] += f
            cnt[s2i[c]] += 1
            TF.append(f)
            TL.append(s2i[c])
    P = F.normalize(P / cnt.clamp(min=1).unsqueeze(1), dim=-1)
    return P.to(dev), torch.stack(TF).to(dev), torch.tensor(TL).to(dev)


PF, TFF, TLF = protos_and_train(FtI, FtF, seen)
PC, TFC, TLC = protos_and_train(CtI, CtF, seen)
PL, TFL, TLL = protos_and_train(LtI, LtF, seen)
LAM = 4.0


def seen_score(qF, P, TF, TL, Tx=None, lam=0.0):
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
        if Tx is not None and lam > 0:
            sc = sc + lam * (e @ Tx.t())
        out[i:i + 2000] = sc
    return out


def unified_predict(qF, qC, qL, qH, files, has_ftcap):
    """qF/qC may be None if has_ftcap is False (unseen-eval images)."""
    qH_d = qH.to(dev)
    M_taxon = qH_d @ TtH.t()
    M_name = qH_d @ TnH.t()
    text_block = zc(dbnorm(M_taxon, 0.05, 0.5) + 0.5 * dbnorm(M_name, 0.05, 0.5))

    sc_l = seen_score(qL, PL, TFL, TLL)
    if has_ftcap:
        sc_ft = seen_score(qF, PF, TFF, TLF, TseenTax, LAM)
        sc_cap = seen_score(qC, PC, TFC, TLC, TseenTax, LAM)
        seen_block = zc(sc_ft) + 1.0 * zc(sc_cap) + 0.5 * zc(sc_l)
    else:
        seen_block = zc(sc_l)  # L-only proxy: ft/cap not extracted for this eval set

    U = text_block.clone()
    U[:, kept_idx] = seen_block + w_text * text_block[:, kept_idx] - gamma
    pred_idx = U.argmax(1)
    return {fn: classes[j] for fn, j in zip(files, pred_idx.tolist())}


preds = {}

# ---- TEST (seen-eval, full ft+cap+L+H) ----
FteI, FteF, Ftef = load('outputs/emb_test_ft.pt')
CteI, CteF, Ctef = load('outputs/emb_test_cap.pt')
LteI, LteF, Ltef = load('outputs/emb_test.pt')
HteI, HteF, Htef = load('outputs/emb_test_h.pt')
tf = [fn for fn in Htef if fn in FteI and fn in CteI and fn in LteI]
qF = torch.stack([FteF[FteI[fn]] for fn in tf])
qC = torch.stack([CteF[CteI[fn]] for fn in tf])
qL = torch.stack([LteF[LteI[fn]] for fn in tf])
qH = torch.stack([HteF[HteI[fn]] for fn in tf])
preds.update(unified_predict(qF, qC, qL, qH, tf, has_ftcap=True))
print(f'test (seen-eval) predicted: {len(tf)}')

# ---- UNSEEN-eval (L+H only, no ft/cap cached) ----
LunI, LunF, Lunf = load('outputs/emb_unseen.pt')
HunI, HunF, Hunf = load('outputs/emb_unseen_h.pt')
uf = sorted(set(Lunf) & set(Hunf))
qL2 = torch.stack([LunF[LunI[fn]] for fn in uf])
qH2 = torch.stack([HunF[HunI[fn]] for fn in uf])
preds.update(unified_predict(None, None, qL2, qH2, uf, has_ftcap=False))
print(f'unseen-eval predicted: {len(uf)}')

print(f'total predictions: {len(preds)}  (expect 35665)')

tag = 'v21_unified'
json.dump(preds, open(f'outputs/prediction_{tag}.json', 'w'))
z = zipfile.ZipFile(f'outputs/submission_{tag}.zip', 'w', zipfile.ZIP_DEFLATED)
z.write(f'outputs/prediction_{tag}.json', arcname='prediction.json')
z.close()
print(f'wrote outputs/submission_{tag}.zip -- NOT uploaded')
