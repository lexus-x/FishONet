"""Real submission builder: current real-tested recipe (ctftbig, w_text=0.5, gamma=13 --
the ONE config with an actual Codabench score: seen 77.73/unseen 5.22/overall 46.08,
2026-07-16) + fullft336 as a 4th seen-block channel @ w=0.75 (holdout +0.33pt, untested
real). Everything else byte-identical to build_gamma_sweep.py's g13 row so this is a
clean single-variable test of whether the fullft336 holdout gain transfers to real.
"""
import json, pickle, torch, zipfile
import torch.nn.functional as F
from collections import defaultdict

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_num_threads(8)
D = 'data/dl'

def load(p):
    d = torch.load(p, weights_only=False)
    return {fn: i for i, fn in enumerate(d['files'])}, F.normalize(d['feats'].float(), dim=-1), list(d['files'])
def zc(M): return (M - M.mean()) / (M.std() + 1e-6)
def dbnorm(S, tc=0.05, tr=0.5): return F.log_softmax(S / tc, dim=0) + F.log_softmax(S / tr, dim=1)

txtHt = torch.load('outputs/text_emb_h_taxon.pt', weights_only=False)
txtL = torch.load('outputs/text_emb.pt', weights_only=False)
classes = list(pickle.load(open(f'{D}/all_classes.pkl', 'rb')))
ci = {c: i for i, c in enumerate(classes)}
TtH = F.normalize(txtHt['emb_taxon'].float(), dim=-1).to(dev)
TnL = F.normalize(txtL['emb_name'].float(), dim=-1).to(dev)
lab = json.load(open(f'{D}/label_train.json'))

FtI, FtF, Ftf = load('outputs/emb_train_ft.pt')
CtI, CtF, Ctf = load('outputs/emb_train_cap.pt')
LtI, LtF, Ltf = load('outputs/emb_train.pt')
QtI, QtF, Qtf = load('outputs/emb_train_fullft336.pt')
common = [fn for fn in Ftf if fn in CtI and fn in LtI and fn in QtI and fn in lab and lab[fn] in ci]
by = defaultdict(list)
for fn in common: by[lab[fn]].append(fn)
seen = sorted(by.keys()); s2i = {c: i for i, c in enumerate(seen)}; S = len(seen)
kept_idx = torch.tensor([ci[c] for c in seen]).to(dev)
TseenTax = TtH[kept_idx]

def protos_and_train(FI, FF, order):
    P = torch.zeros(S, FF.shape[1]); cnt = torch.zeros(S); TF, TL = [], []
    for c in order:
        for fn in by[c]:
            f = FF[FI[fn]]; P[s2i[c]] += f; cnt[s2i[c]] += 1; TF.append(f); TL.append(s2i[c])
    P = F.normalize(P / cnt.clamp(min=1).unsqueeze(1), dim=-1)
    return P.to(dev), torch.stack(TF).to(dev), torch.tensor(TL).to(dev)

PF, TFF, TLF = protos_and_train(FtI, FtF, seen)
PC, TFC, TLC = protos_and_train(CtI, CtF, seen)
PL, TFL, TLL = protos_and_train(LtI, LtF, seen)
PQ, TFQ, TLQ = protos_and_train(QtI, QtF, seen)
LAM = 4.0

def seen_score(qF, P, TF, TL, Tx=None, lam=0.0, ret_maxproto=False):
    qF = qF.to(dev); n = qF.shape[0]
    out = torch.empty(n, S, device=dev); ps_all = torch.empty(n, S, device=dev) if ret_maxproto else None
    for i in range(0, n, 2000):
        e = qF[i:i + 2000]; ps = e @ P.t(); sim = e @ TF.t()
        cmax = torch.full((e.shape[0], S), -1e9, device=dev)
        cmax.scatter_reduce_(1, TL.unsqueeze(0).expand(e.shape[0], -1), sim, reduce='amax')
        sc = ps + 2.0 * cmax
        if Tx is not None and lam > 0: sc = sc + lam * (e @ Tx.t())
        out[i:i + 2000] = sc
        if ret_maxproto: ps_all[i:i + 2000] = ps
    return (out, ps_all.max(1).values) if ret_maxproto else out

FteI, FteF, Ftef = load('outputs/emb_test_ft.pt')
CteI, CteF, _ = load('outputs/emb_test_cap.pt')
LteI, LteF, _ = load('outputs/emb_test.pt')
HteI, _, Htef = load('outputs/emb_test_h.pt')
BteI, BteF, _ = load('outputs/emb_test_ctftbig.pt')
QteI, QteF, _ = load('outputs/emb_test_fullft336.pt')
tf = sorted(fn for fn in Htef if fn in FteI and fn in CteI and fn in LteI and fn in QteI)
FunI, FunF, _ = load('outputs/emb_unseen_ft.pt')
CunI, CunF, _ = load('outputs/emb_unseen_cap.pt')
LunI, LunF, _ = load('outputs/emb_unseen.pt')
HunI, _, Hunf = load('outputs/emb_unseen_h.pt')
BunI, BunF, _ = load('outputs/emb_unseen_ctftbig.pt')
QunI, QunF, _ = load('outputs/emb_unseen_fullft336.pt')
uf = sorted(fn for fn in Hunf if fn in FunI and fn in CunI and fn in LunI and fn in QunI)
all_files = tf + uf; n_t = len(tf)
print(f'combined eval batch: {len(all_files)} (expect 35665, tf={len(tf)} uf={len(uf)})')

qF = torch.cat([torch.stack([FteF[FteI[fn]] for fn in tf]), torch.stack([FunF[FunI[fn]] for fn in uf])])
qC = torch.cat([torch.stack([CteF[CteI[fn]] for fn in tf]), torch.stack([CunF[CunI[fn]] for fn in uf])])
qL = torch.cat([torch.stack([LteF[LteI[fn]] for fn in tf]), torch.stack([LunF[LunI[fn]] for fn in uf])])
qB = torch.cat([torch.stack([BteF[BteI[fn]] for fn in tf]), torch.stack([BunF[BunI[fn]] for fn in uf])])
qQ = torch.cat([torch.stack([QteF[QteI[fn]] for fn in tf]), torch.stack([QunF[QunI[fn]] for fn in uf])])

M_taxon = qB.to(dev) @ TtH.t()
M_name = qL.to(dev) @ TnL.t()
text_block = zc(dbnorm(M_taxon, 0.05, 0.5) + 0.5 * dbnorm(M_name, 0.05, 0.5))
sc_ft, maxproto = seen_score(qF, PF, TFF, TLF, TseenTax, LAM, ret_maxproto=True)
sc_cap = seen_score(qC, PC, TFC, TLC, TseenTax, LAM)
sc_l = seen_score(qL, PL, TFL, TLL)
sc_q = seen_score(qQ, PQ, TFQ, TLQ)
seen_block = zc(sc_ft) + 1.0 * zc(sc_cap) + 0.5 * zc(sc_l) + 0.75 * zc(sc_q)   # +4th channel
novelty = 1.0 - (maxproto - maxproto.min()) / (maxproto.max() - maxproto.min() + 1e-6)

seen_set = set(seen)
W_TEXT, GAMMA = 0.5, 13.0
U = text_block.clone()
U[:, kept_idx] = seen_block + W_TEXT * text_block[:, kept_idx] - (GAMMA * novelty).unsqueeze(1)
pred_idx = U.argmax(1)
preds = {fn: classes[j] for fn, j in zip(all_files, pred_idx.tolist())}
o_t = sum(1 for fn in tf if preds[fn] not in seen_set)
o_u = sum(1 for fn in uf if preds[fn] not in seen_set)
tag = 'v21_g13_w0.5_4ch_fullft336'
json.dump(preds, open(f'outputs/prediction_{tag}.json', 'w'))
z = zipfile.ZipFile(f'outputs/submission_{tag}.zip', 'w', zipfile.ZIP_DEFLATED)
z.write(f'outputs/prediction_{tag}.json', arcname='prediction.json'); z.close()
print(f'{tag}: test->other {100*o_t/len(tf):.1f}%  unseen->other {100*o_u/len(uf):.1f}%  wrote submission_{tag}.zip')
print('DONE (not uploaded -- for you to verify)')
