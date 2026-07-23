"""Build v21 unified submission -- SYMMETRIC formula (H+L only), the genuinely
compliant version: identical scoring pipeline for every one of the 35665 real
eval images (test + unseen), because H and L are cached for every image with
no folder-dependent backbone availability gap (unlike the ft/cap ensemble,
which is only cached for the test/seen-eval folder -- see v21 first-pass
finding: that asymmetric version collapsed unseen-eval to 100% non-seen
predictions, indistinguishable in effect from a banned seen/unseen router).
"""
import json, pickle, torch, zipfile
import torch.nn.functional as F
from collections import defaultdict

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_num_threads(8)
D = 'data/dl'

CFG = json.load(open('outputs/unified_frontier_symmetric.json'))
best_cfg = CFG['best_compliant']['config']
print('using config:', best_cfg)
variant_tag, gamma_tag = best_cfg.split(' ')
w_text = float(variant_tag.split('_wtext')[1])
gamma = float(gamma_tag.split('=')[1])
print(f'w_text={w_text} gamma={gamma}')


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

HtI, HtF, Htf = load('outputs/emb_train_h.pt')
LtI, LtF, Ltf = load('outputs/emb_train.pt')
common = [fn for fn in Htf if fn in LtI and fn in lab and lab[fn] in ci]
by = defaultdict(list)
for fn in common:
    by[lab[fn]].append(fn)
seen = sorted(by.keys())
s2i = {c: i for i, c in enumerate(seen)}
S = len(seen)
print(f'seen classes: {S}  train imgs: {len(common)}')
kept_idx = torch.tensor([ci[c] for c in seen]).to(dev)
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


PH, TFH, TLH = protos_and_train(HtI, HtF, seen)
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


def unified_predict(qH, qL, files):
    qH_d = qH.to(dev)
    M_taxon = qH_d @ TtH.t()
    M_name = qH_d @ TnH.t()
    text_block = zc(dbnorm(M_taxon, 0.05, 0.5) + 0.5 * dbnorm(M_name, 0.05, 0.5))

    sc_h = seen_score(qH, PH, TFH, TLH, TseenTax, LAM)
    sc_l = seen_score(qL, PL, TFL, TLL)
    seen_block = zc(sc_h) + 0.5 * zc(sc_l)

    U = text_block.clone()
    U[:, kept_idx] = seen_block + w_text * text_block[:, kept_idx] - gamma
    pred_idx = U.argmax(1)
    return {fn: classes[j] for fn, j in zip(files, pred_idx.tolist())}


preds = {}
seen_set = set(seen)

# TEST (seen-eval)
HteI, HteF, Htef = load('outputs/emb_test_h.pt')
LteI, LteF, Ltef = load('outputs/emb_test.pt')
tf = [fn for fn in Htef if fn in LteI]
qH = torch.stack([HteF[HteI[fn]] for fn in tf])
qL = torch.stack([LteF[LteI[fn]] for fn in tf])
preds.update(unified_predict(qH, qL, tf))
print(f'test (seen-eval) predicted: {len(tf)}')

# UNSEEN-eval -- SAME formula, same code path
HunI, HunF, Hunf = load('outputs/emb_unseen_h.pt')
LunI, LunF, Lunf = load('outputs/emb_unseen.pt')
uf = sorted(set(Hunf) & set(Lunf))
qH2 = torch.stack([HunF[HunI[fn]] for fn in uf])
qL2 = torch.stack([LunF[LunI[fn]] for fn in uf])
preds.update(unified_predict(qH2, qL2, uf))
print(f'unseen-eval predicted: {len(uf)}')

print(f'total predictions: {len(preds)}  (expect 35665)')

# distribution check
def frac_other(fns):
    n = o = 0
    for fn in fns:
        if fn in preds:
            n += 1
            if preds[fn] not in seen_set:
                o += 1
    return o, n

o_t, n_t = frac_other(tf)
o_u, n_u = frac_other(uf)
print(f'test-eval: {100*(n_t-o_t)/n_t:.1f}% seen-cols, {100*o_t/n_t:.1f}% other')
print(f'unseen-eval: {100*(n_u-o_u)/n_u:.1f}% seen-cols, {100*o_u/n_u:.1f}% other')

tag = 'v21_unified_symmetric'
json.dump(preds, open(f'outputs/prediction_{tag}.json', 'w'))
z = zipfile.ZipFile(f'outputs/submission_{tag}.zip', 'w', zipfile.ZIP_DEFLATED)
z.write(f'outputs/prediction_{tag}.json', arcname='prediction.json')
z.close()
print(f'wrote outputs/submission_{tag}.zip -- NOT uploaded')
