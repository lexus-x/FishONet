"""Build v21 unified (compliant single-pipeline) submission -- ctftbig text-block variant.

Identical structure to build_v21_unified_submission_c.py (same seen-block: ft/cap/L NCM+cmax+
taxon; same variant-c smooth novelty gate; same single combined pass over all 35665 eval
images). The ONLY change is the text-block's source features, restoring the proven unseen
unlock (v20's deployed unseen route, verified in outputs/unified_holdout_ctftbig.py /
outputs/unified_frontier_ctftbig.json):
    was:  M_taxon = qH      @ TtH_taxon(H) ;  M_name = qH      @ TnH_name(H)
    now:  M_taxon = qB_ctftbig @ TtH_taxon(H) ;  M_name = qL_plain  @ TnL_name(L)
qB (ctftbig) uses the re-extracted outputs/emb_{test,unseen}_ctftbig.pt (scale=0.4 LoRA WiSE-FT
+ hflip TTA, src/ctft_big_finalize.py recipe, verified cosine>0.999 against the original cached
outputs/emb_unseen_ctftbig.pt on a 200-image subset).

Config from outputs/unified_frontier_ctftbig.json (variant c, ctftbig text-block):
  w_text=0.5, gamma=20  ->  holdout seen=84.28 unseen=20.58 overall_5644=56.51
  (vs plainH's own w_text=0.5,gamma=20: seen=83.94 unseen=15.83 overall_5644=54.24 --
  same operating point/gamma, ctftbig strictly better on unseen and overall.)
"""
import json, pickle, torch, zipfile
import torch.nn.functional as F
from collections import defaultdict

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_num_threads(8)
D = 'data/dl'

W_TEXT = 0.5
GAMMA = 20.0
print(f'variant=c(ctftbig text-block) w_text={W_TEXT} gamma={GAMMA} '
      f'(from unified_frontier_ctftbig.json ctftbig w_text0.5 gamma=20 row)')


def load(p):
    d = torch.load(p, weights_only=False)
    return {fn: i for i, fn in enumerate(d['files'])}, F.normalize(d['feats'].float(), dim=-1), list(d['files'])


def zc(M):
    return (M - M.mean()) / (M.std() + 1e-6)


def dbnorm(S, tc=0.05, tr=0.5):
    return F.log_softmax(S / tc, dim=0) + F.log_softmax(S / tr, dim=1)


txtHt = torch.load('outputs/text_emb_h_taxon.pt', weights_only=False)
txtH = torch.load('outputs/text_emb_h.pt', weights_only=False)
txtL = torch.load('outputs/text_emb.pt', weights_only=False)
classes = list(pickle.load(open(f'{D}/all_classes.pkl', 'rb')))
assert txtHt['classes'] == classes and txtH['classes'] == classes and txtL['classes'] == classes
ci = {c: i for i, c in enumerate(classes)}
NCLS = len(classes)
TtH = F.normalize(txtHt['emb_taxon'].float(), dim=-1).to(dev)       # [NCLS,1024] taxon (H-space)
TnL = F.normalize(txtL['emb_name'].float(), dim=-1).to(dev)         # [NCLS,768]  name (L-space)

lab = json.load(open(f'{D}/label_train.json'))

# ---- ALL seen classes get prototypes (deployment: no pseudo-unseen holdout carve-out) ----
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


def seen_score(qF, P, TF, TL, Tx=None, lam=0.0, ret_maxproto=False):
    qF = qF.to(dev)
    n = qF.shape[0]
    out = torch.empty(n, S, device=dev)
    ps_all = torch.empty(n, S, device=dev) if ret_maxproto else None
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
        if ret_maxproto:
            ps_all[i:i + 2000] = ps
    if ret_maxproto:
        return out, ps_all.max(1).values
    return out


# ================= load ALL 35665 eval images, build ONE combined query batch =================
# same population as build_v21_unified_submission_c.py (ft/cap/L/H-availability-gated), plus
# ctftbig (verified miss=0 for both splits in step 2, so this filter is a no-op on population).
FteI, FteF, Ftef = load('outputs/emb_test_ft.pt')
CteI, CteF, Ctef = load('outputs/emb_test_cap.pt')
LteI, LteF, Ltef = load('outputs/emb_test.pt')
HteI, HteF, Htef = load('outputs/emb_test_h.pt')
BteI, BteF, Btef = load('outputs/emb_test_ctftbig.pt')
tf = sorted(fn for fn in Htef if fn in FteI and fn in CteI and fn in LteI)
print(f'test (seen-eval) files: {len(tf)}')
assert all(fn in BteI for fn in tf), 'ctftbig test cache missing some files -- alignment risk'

FunI, FunF, Funf = load('outputs/emb_unseen_ft.pt')
CunI, CunF, Cunf = load('outputs/emb_unseen_cap.pt')
LunI, LunF, Lunf = load('outputs/emb_unseen.pt')
HunI, HunF, Hunf = load('outputs/emb_unseen_h.pt')
BunI, BunF, Bunf = load('outputs/emb_unseen_ctftbig.pt')
uf = sorted(fn for fn in Hunf if fn in FunI and fn in CunI and fn in LunI)
print(f'unseen-eval files: {len(uf)}')
assert all(fn in BunI for fn in uf), 'ctftbig unseen cache missing some files -- alignment risk'

assert set(tf).isdisjoint(uf), 'test/unseen filename overlap -- would silently collide in one dict'

all_files = tf + uf
n_t, n_u = len(tf), len(uf)
print(f'combined eval batch: {len(all_files)} (expect 35665)')

qF = torch.cat([torch.stack([FteF[FteI[fn]] for fn in tf]), torch.stack([FunF[FunI[fn]] for fn in uf])])
qC = torch.cat([torch.stack([CteF[CteI[fn]] for fn in tf]), torch.stack([CunF[CunI[fn]] for fn in uf])])
qL = torch.cat([torch.stack([LteF[LteI[fn]] for fn in tf]), torch.stack([LunF[LunI[fn]] for fn in uf])])
qB = torch.cat([torch.stack([BteF[BteI[fn]] for fn in tf]), torch.stack([BunF[BunI[fn]] for fn in uf])])

# ================= ONE combined pass: seen_block, text_block, novelty, argmax =================
qB_d = qB.to(dev)
qL_d = qL.to(dev)
M_taxon = qB_d @ TtH.t()     # ctftbig (H-backbone LoRA-FT) vs taxon text (H-space)
M_name = qL_d @ TnL.t()      # plain-L vs name text (L-space)
text_block = zc(dbnorm(M_taxon, 0.05, 0.5) + 0.5 * dbnorm(M_name, 0.05, 0.5))   # [35665, NCLS]

sc_ft, maxproto = seen_score(qF, PF, TFF, TLF, TseenTax, LAM, ret_maxproto=True)
sc_cap = seen_score(qC, PC, TFC, TLC, TseenTax, LAM)
sc_l = seen_score(qL, PL, TFL, TLL)
seen_block = zc(sc_ft) + 1.0 * zc(sc_cap) + 0.5 * zc(sc_l)   # [35665, S]

novelty = 1.0 - (maxproto - maxproto.min()) / (maxproto.max() - maxproto.min() + 1e-6)
gamma_row = (GAMMA * novelty).unsqueeze(1)   # [35665,1], per-query, reads only its own maxproto

U = text_block.clone()
U[:, kept_idx] = seen_block + W_TEXT * text_block[:, kept_idx] - gamma_row
pred_idx = U.argmax(1)

preds = {fn: classes[j] for fn, j in zip(all_files, pred_idx.tolist())}
print(f'total predictions: {len(preds)}  (expect 35665, all unique keys)')
assert len(preds) == 35665, f'expected 35665 unique predictions, got {len(preds)}'
assert all(c in ci for c in preds.values()), 'invalid class in predictions'

seen_set = set(seen)
def frac_other(fns):
    n = o = 0
    for fn in fns:
        n += 1
        if preds[fn] not in seen_set: o += 1
    return o, n
o_t, n_t2 = frac_other(tf)
o_u, n_u2 = frac_other(uf)
print(f'test-eval: {100*(n_t2-o_t)/n_t2:.1f}% seen-cols, {100*o_t/n_t2:.1f}% other-cols')
print(f'unseen-eval: {100*(n_u2-o_u)/n_u2:.1f}% seen-cols, {100*o_u/n_u2:.1f}% other-cols')
print(f'novelty stats: min={novelty.min().item():.4f} max={novelty.max().item():.4f} mean={novelty.mean().item():.4f}')
print(f'  test-eval novelty mean={novelty[:n_t].mean().item():.4f}  unseen-eval novelty mean={novelty[n_t:].mean().item():.4f}')

tag = 'v21_unified_ctftbig'
json.dump(preds, open(f'outputs/prediction_{tag}.json', 'w'))
z = zipfile.ZipFile(f'outputs/submission_{tag}.zip', 'w', zipfile.ZIP_DEFLATED)
z.write(f'outputs/prediction_{tag}.json', arcname='prediction.json')
z.close()
print(f'wrote outputs/submission_{tag}.zip -- NOT uploaded')

# ================= v20-agreement diagnostic on unseen-eval filenames =================
try:
    v20 = json.load(open('outputs/prediction_v20_real.json'))
    common_uf = [fn for fn in uf if fn in v20]
    print(f'v20 pred keys: {len(v20)}  intersection with uf: {len(common_uf)} / {len(uf)}')
    agree = sum(1 for fn in common_uf if preds[fn] == v20[fn])
    frac = 100 * agree / len(common_uf) if common_uf else float('nan')
    print(f'unseen-eval agreement with v20 (prediction_v20_real.json): {frac:.1f}%  '
          f'({agree}/{len(common_uf)})  [was 34.8% for the plain-H v21_unified_c version]')
except FileNotFoundError:
    print('outputs/prediction_v20_real.json not found -- skipping v20-agreement diagnostic')
