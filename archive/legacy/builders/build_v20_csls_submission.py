"""Build v20 + CSLS blended submission. Seen route = exact v20.
Unseen route = 50/50 DBNorm + CSLS(k=5) blend."""
import json, torch, zipfile
import torch.nn.functional as F
from collections import defaultdict

print('Building v20+CSLS submission...')
torch.set_num_threads(8)

def load(p):
    d = torch.load(p, weights_only=False)
    return {fn: i for i, fn in enumerate(d['files'])}, F.normalize(d['feats'].float(), dim=-1), list(d['files'])

def zc(M):
    return (M - M.mean()) / (M.std() + 1e-6)

txtHt = torch.load('outputs/text_emb_h_taxon.pt', weights_only=False)
classes = txtHt['classes']
ci = {c: i for i, c in enumerate(classes)}
TtH = F.normalize(txtHt['emb_taxon'].float(), dim=-1)
TnL = F.normalize(torch.load('outputs/text_emb.pt', weights_only=False)['emb_name'].float(), dim=-1)
lab = json.load(open('data/dl/label_train.json'))

# ===== SEEN ROUTE (exact v20) =====
FtI, FtF, Ftf = load('outputs/emb_train_ft.pt')
CtI, CtF, Ctf = load('outputs/emb_train_cap.pt')
LtI, LtF, Ltf = load('outputs/emb_train.pt')

trainfiles = [fn for fn in Ftf if fn in LtI and fn in CtI and fn in lab and lab[fn] in ci]
by = defaultdict(list)
for fn in trainfiles:
    by[lab[fn]].append(fn)
seen = sorted(by.keys())
s2i = {c: i for i, c in enumerate(seen)}
S = len(seen)
TseenTax = torch.stack([TtH[ci[c]] for c in seen])

def protos_and_train(TrI, TrF, fns_by_cls, order):
    P = torch.zeros(S, TrF.shape[1])
    cnt = torch.zeros(S)
    TF = []
    TL = []
    for c in order:
        for fn in fns_by_cls[c]:
            f = TrF[TrI[fn]]
            P[s2i[c]] += f
            cnt[s2i[c]] += 1
            TF.append(f)
            TL.append(s2i[c])
    P = F.normalize(P / cnt.clamp(min=1).unsqueeze(1), dim=-1)
    return P, torch.stack(TF), torch.tensor(TL)

def seen_score(qF, P, TF, TL, Tx=None, lam=0.0):
    out = torch.empty(qF.shape[0], S)
    for i in range(0, qF.shape[0], 1000):
        e = qF[i:i + 1000]
        ps = e @ P.t()
        sim = e @ TF.t()
        cmax = torch.full((e.shape[0], S), -1e9)
        cmax.scatter_reduce_(1, TL.unsqueeze(0).expand(e.shape[0], -1), sim, reduce='amax')
        sc = ps + 2.0 * cmax
        if Tx is not None and lam > 0:
            sc = sc + lam * (e @ Tx.t())
        out[i:i + 1000] = sc
    return out

PF, TFF, TLF = protos_and_train(FtI, FtF, by, seen)
PC, TFC, TLC = protos_and_train(CtI, CtF, by, seen)
PL, TFL, TLL = protos_and_train(LtI, LtF, by, seen)

FteI, FteF, Ftef = load('outputs/emb_test_ft.pt')
CteI, CteF, Ctef = load('outputs/emb_test_cap.pt')
LteI, LteF, Ltef = load('outputs/emb_test.pt')

tf = [fn for fn in Ftef if fn in LteI and fn in CteI]
qF = torch.stack([FteF[FteI[fn]] for fn in tf])
qC = torch.stack([CteF[CteI[fn]] for fn in tf])
qL = torch.stack([LteF[LteI[fn]] for fn in tf])

LAM = 4.0
sc_seen = (
    zc(seen_score(qF, PF, TFF, TLF, TseenTax, LAM))
    + 1.0 * zc(seen_score(qC, PC, TFC, TLC, TseenTax, LAM))
    + 0.5 * zc(seen_score(qL, PL, TFL, TLL))
)
preds = {fn: seen[k] for fn, k in zip(tf, sc_seen.argmax(1).tolist())}
print(f'seen: {len(preds)}')

# ===== UNSEEN ROUTE (CSLS + DBNorm blend) =====
nonk = torch.tensor([i for i, c in enumerate(classes) if c not in set(seen)])

HunI, HunF, Hunf = load('outputs/emb_unseen_ctftbig.pt')
LunI, LunF, Lunf = load('outputs/emb_unseen.pt')

uf = sorted(set(Hunf) & set(Lunf))
uH = torch.stack([HunF[HunI[fn]] for fn in uf])
uL = torch.stack([LunF[LunI[fn]] for fn in uf])
TtHu = TtH[nonk]
TnLu = TnL[nonk]

def db(M):
    return (M - M.mean(0, keepdim=True)) / (M.std(0, keepdim=True) + 1e-6)

def dis(M, a, b):
    return F.log_softmax(M / a, dim=0) + F.log_softmax(M / b, dim=1)

MH = uH @ TtHu.t()
ML = uL @ TnLu.t()
dbnorm_score = dis(MH, 0.05, 0.5) + 0.5 * dis(ML, 0.05, 0.5)

def csls(S, k=5):
    r_T = S.topk(k, dim=0).values.mean(dim=0)
    r_I = S.topk(k, dim=1).values.mean(dim=1)
    return 2 * S - r_T.unsqueeze(0) - r_I.unsqueeze(1)

cs_taxon = csls(uH @ TtHu.t(), k=5)
cs_name = csls(uL @ TnLu.t(), k=5)
cs_score = cs_taxon + 0.5 * cs_name

alpha = 0.5
final_unseen = alpha * dbnorm_score + (1 - alpha) * cs_score

for fn, j in zip(uf, final_unseen.argmax(1).tolist()):
    preds[fn] = classes[nonk[j].item()]
print(f'unseen: {len(uf)}')
print(f'total preds: {len(preds)}')

tag = 'v20_csls05'
json.dump(preds, open(f'outputs/prediction_{tag}.json', 'w'))
z = zipfile.ZipFile(f'outputs/submission_{tag}.zip', 'w', zipfile.ZIP_DEFLATED)
z.write(f'outputs/prediction_{tag}.json', arcname='prediction.json')
z.close()
print(f'wrote outputs/submission_{tag}.zip')

try:
    vp = json.load(open('outputs/prediction_v20_real.json'))
    diff = sum(1 for k in preds if preds[k] != vp.get(k, ''))
    print(f'differs from v20: {diff}/{len(preds)} ({100*diff/len(preds):.1f}%)')
except:
    print('(no v20 reference to compare)')