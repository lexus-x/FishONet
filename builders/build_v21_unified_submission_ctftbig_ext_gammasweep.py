"""Real-world gamma recalibration, in response to the first real Codabench score for the
ctftbig_ext single-pipeline recipe (gamma=30 -> real seen=71.60 unseen=11.10 overall=45.19,
vs holdout seen=85.10 unseen=20.10 overall=56.76).

Diagnosis: real seen mis-route rate into the unseen head is ~19.4%, vs ~4.9% implied by the
holdout pseudo-unseen simulation (~3.9x worse). The novelty gate is far more trigger-happy on
real photos than holdout predicts, and the real unseen route's conditional accuracy (~13.9%) is
only ~0.5x the holdout oracle's -- so routing to unseen pays off less than the gate assumes, while
misrouting a true-seen image (56% of real population) costs more. Net: gamma should probably be
LOWER for real deployment than the holdout sweep alone would pick.

This script is IDENTICAL to build_v21_unified_submission_ctftbig_ext.py's architecture/data, but
loads everything once and writes out predictions/submissions for several candidate gammas so they
can be tried on Codabench without repeating the (cheap but non-trivial) embedding-load step each
time. NOT validated against a fresh holdout number -- these are untested extrapolations from the
diagnosis above, ranked by holdout overall_5644 only, to be confirmed empirically.
"""
import json, pickle, torch, zipfile
import torch.nn.functional as F
from collections import defaultdict

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_num_threads(8)
D = 'data/dl'

W_TEXT = 0.5
GAMMAS = [5.0, 10.0, 15.0, 20.0]   # candidates, all well below the real-world-miscalibrated 30
W_B = 1.0
W_U = 0.75
print(f'variant=ctftbig_ext_gammasweep w_text={W_TEXT} gammas={GAMMAS} W_B={W_B} W_U={W_U}')


def load(p):
    d = torch.load(p, weights_only=False)
    return {fn: i for i, fn in enumerate(d['files'])}, F.normalize(d['feats'].float(), dim=-1), list(d['files'])


def zc(M):
    return (M - M.mean()) / (M.std() + 1e-6)


def dbnorm(S, tc=0.05, tr=0.5):
    return F.log_softmax(S / tc, dim=0) + F.log_softmax(S / tr, dim=1)


txtHt = torch.load('outputs/text_emb_h_taxon.pt', weights_only=False)
txtL = torch.load('outputs/text_emb.pt', weights_only=False)
classes = list(pickle.load(open(f'{D}/all_classes.pkl', 'rb')))
assert txtHt['classes'] == classes and txtL['classes'] == classes
ci = {c: i for i, c in enumerate(classes)}
NCLS = len(classes)
TtH = F.normalize(txtHt['emb_taxon'].float(), dim=-1).to(dev)
TnL = F.normalize(txtL['emb_name'].float(), dim=-1).to(dev)

lab = json.load(open(f'{D}/label_train.json'))

FtI, FtF, Ftf = load('outputs/emb_train_ft.pt')
CtI, CtF, Ctf = load('outputs/emb_train_cap.pt')
LtI, LtF, Ltf = load('outputs/emb_train.pt')
BtI, BtF, Btf = load('outputs/emb_train_ctftbig.pt')

common = [fn for fn in Ftf if fn in CtI and fn in LtI and fn in BtI and fn in lab and lab[fn] in ci]
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
PB, TFB, TLB = protos_and_train(BtI, BtF, seen)
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


FteI, FteF, Ftef = load('outputs/emb_test_ft.pt')
CteI, CteF, Ctef = load('outputs/emb_test_cap.pt')
LteI, LteF, Ltef = load('outputs/emb_test.pt')
HteI, HteF, Htef = load('outputs/emb_test_h.pt')
BteI, BteF, Btef = load('outputs/emb_test_ctftbig.pt')
FF2teI, FF2teF, FF2tef = load('outputs/emb_test_fullft336_v2.pt')
tf = sorted(fn for fn in Htef if fn in FteI and fn in CteI and fn in LteI)
print(f'test (seen-eval) files: {len(tf)}')

FunI, FunF, Funf = load('outputs/emb_unseen_ft.pt')
CunI, CunF, Cunf = load('outputs/emb_unseen_cap.pt')
LunI, LunF, Lunf = load('outputs/emb_unseen.pt')
HunI, HunF, Hunf = load('outputs/emb_unseen_h.pt')
BunI, BunF, Bunf = load('outputs/emb_unseen_ctftbig.pt')
FF2unI, FF2unF, FF2unf = load('outputs/emb_unseen_fullft336_v2.pt')
uf = sorted(fn for fn in Hunf if fn in FunI and fn in CunI and fn in LunI)
print(f'unseen-eval files: {len(uf)}')

all_files = tf + uf
n_t, n_u = len(tf), len(uf)
print(f'combined eval batch: {len(all_files)} (expect 35665)')

qF = torch.cat([torch.stack([FteF[FteI[fn]] for fn in tf]), torch.stack([FunF[FunI[fn]] for fn in uf])])
qC = torch.cat([torch.stack([CteF[CteI[fn]] for fn in tf]), torch.stack([CunF[CunI[fn]] for fn in uf])])
qL = torch.cat([torch.stack([LteF[LteI[fn]] for fn in tf]), torch.stack([LunF[LunI[fn]] for fn in uf])])
qB = torch.cat([torch.stack([BteF[BteI[fn]] for fn in tf]), torch.stack([BunF[BunI[fn]] for fn in uf])])
qFF2 = torch.cat([torch.stack([FF2teF[FF2teI[fn]] for fn in tf]), torch.stack([FF2unF[FF2unI[fn]] for fn in uf])])

qB_d = qB.to(dev)
qL_d = qL.to(dev)
qFF2_d = qFF2.to(dev)
M_taxon = qB_d @ TtH.t()
M_name = qL_d @ TnL.t()
M_taxon_FF2 = qFF2_d @ TtH.t()
text_block = zc(dbnorm(M_taxon, 0.05, 0.5) + 0.5 * dbnorm(M_name, 0.05, 0.5)
                + W_U * dbnorm(M_taxon_FF2, 0.05, 0.5))

sc_ft, maxproto = seen_score(qF, PF, TFF, TLF, TseenTax, LAM, ret_maxproto=True)
sc_cap = seen_score(qC, PC, TFC, TLC, TseenTax, LAM)
sc_l = seen_score(qL, PL, TFL, TLL)
sc_b = seen_score(qB, PB, TFB, TLB, TseenTax, LAM)
seen_block = zc(sc_ft) + 1.0 * zc(sc_cap) + 0.5 * zc(sc_l) + W_B * zc(sc_b)

novelty = 1.0 - (maxproto - maxproto.min()) / (maxproto.max() - maxproto.min() + 1e-6)
seen_set = set(seen)
kidx = kept_idx

for GAMMA in GAMMAS:
    gamma_row = (GAMMA * novelty).unsqueeze(1)
    U = text_block.clone()
    U[:, kidx] = seen_block + W_TEXT * text_block[:, kidx] - gamma_row
    pred_idx = U.argmax(1)
    preds = {fn: classes[j] for fn, j in zip(all_files, pred_idx.tolist())}
    assert len(preds) == 35665

    o_t = sum(1 for fn in tf if preds[fn] not in seen_set)
    o_u = sum(1 for fn in uf if preds[fn] not in seen_set)
    print(f'gamma={GAMMA:>5.1f}  test->other {100*o_t/n_t:.1f}%  unseen->other {100*o_u/n_u:.1f}%')

    tag = f'v21_unified_ctftbig_ext_g{int(GAMMA)}'
    json.dump(preds, open(f'outputs/prediction_{tag}.json', 'w'))
    z = zipfile.ZipFile(f'outputs/submission_{tag}.zip', 'w', zipfile.ZIP_DEFLATED)
    z.write(f'outputs/prediction_{tag}.json', arcname='prediction.json')
    z.close()
    print(f'  wrote outputs/submission_{tag}.zip')
