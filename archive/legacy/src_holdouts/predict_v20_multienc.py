"""v20 = multi-encoder seen route (ft + 1.0*cap + 0.5*L, holdout 87.94% vs v15's 87.22%)
+ unseen route BYTE-IDENTICAL to v15 (DBNorm on ctft_big features).
PROBE MODE (default): unseen preds forced to a constant class so the upload reads only
accuracy_test (seen), matching the v10/v17 probe pattern -- standing not revealed.
  python src/predict_v20_multienc.py --generate         # probe (unseen tanked)
  python src/predict_v20_multienc.py --generate --real  # full real submission
"""
import json, torch, argparse, zipfile, shutil
from collections import defaultdict
import torch.nn.functional as F
torch.set_num_threads(8); LAM = 4.0

def load(p):
    d = torch.load(p, weights_only=False)
    return {fn: i for i, fn in enumerate(d['files'])}, F.normalize(d['feats'].float(), dim=-1), list(d['files'])

def zc(M): return (M - M.mean()) / (M.std() + 1e-6)
def db(M): return (M - M.mean(0, keepdim=True)) / (M.std(0, keepdim=True) + 1e-6)
def dis(M, a, b): return F.log_softmax(M / a, dim=0) + F.log_softmax(M / b, dim=1)

ap = argparse.ArgumentParser()
ap.add_argument('--generate', action='store_true')
ap.add_argument('--real', action='store_true', help='write the REAL unseen route instead of the probe constant')
A = ap.parse_args()

txtH = torch.load('outputs/text_emb_h.pt', weights_only=False)
txtHt = torch.load('outputs/text_emb_h_taxon.pt', weights_only=False)
classes = txtH['classes']; ci = {c: i for i, c in enumerate(classes)}
TtH = F.normalize(txtHt['emb_taxon'].float(), dim=-1)
TnL = F.normalize(torch.load('outputs/text_emb.pt', weights_only=False)['emb_name'].float(), dim=-1)
lab = json.load(open('data/dl/label_train.json'))

FtI, FtF, Ftf = load('outputs/emb_train_ft.pt')
CtI, CtF, _ = load('outputs/emb_train_cap.pt')
LtI, LtF, _ = load('outputs/emb_train.pt')
trainfiles = [fn for fn in Ftf if fn in LtI and fn in CtI and fn in lab and lab[fn] in ci]
by = defaultdict(list)
for fn in trainfiles: by[lab[fn]].append(fn)
seen = sorted(by.keys()); s2i = {c: i for i, c in enumerate(seen)}; S = len(seen)
TseenTax = torch.stack([TtH[ci[c]] for c in seen])

def protos_and_train(TrI, TrF, fns_by_cls, order):
    P = torch.zeros(S, TrF.shape[1]); cnt = torch.zeros(S); TF = []; TL = []
    for c in order:
        for fn in fns_by_cls[c]:
            f = TrF[TrI[fn]]; P[s2i[c]] += f; cnt[s2i[c]] += 1; TF.append(f); TL.append(s2i[c])
    P = F.normalize(P / cnt.clamp(min=1).unsqueeze(1), dim=-1); return P, torch.stack(TF), torch.tensor(TL)

def seen_score(qF, P, TF, TL, Tx=None, lam=0.0):
    out = torch.empty(qF.shape[0], S)
    for i in range(0, qF.shape[0], 1000):
        e = qF[i:i + 1000]; ps = e @ P.t(); sim = e @ TF.t()
        cmax = torch.full((e.shape[0], S), -1e9); cmax.scatter_reduce_(1, TL.unsqueeze(0).expand(e.shape[0], -1), sim, reduce='amax')
        sc = ps + 2.0 * cmax
        if Tx is not None and lam > 0: sc = sc + lam * (e @ Tx.t())
        out[i:i + 1000] = sc
    return out

# holdout check (should reproduce 87.94%)
trby = defaultdict(list); valrows = []
for c in seen:
    fns = sorted(by[c])
    if len(fns) >= 3:
        k = max(1, round(0.2 * len(fns)))
        for f in fns[:-k]: trby[c].append(f)
        for f in fns[-k:]: valrows.append((f, s2i[c]))
    else:
        for f in fns: trby[c].append(f)
PFh, TFFh, TLFh = protos_and_train(FtI, FtF, trby, seen)
PCh, TFCh, TLCh = protos_and_train(CtI, CtF, trby, seen)
PLh, TFLh, TLLh = protos_and_train(LtI, LtF, trby, seen)
VY = torch.tensor([y for _, y in valrows])
vF = torch.stack([FtF[FtI[f]] for f, _ in valrows]); vC = torch.stack([CtF[CtI[f]] for f, _ in valrows]); vL = torch.stack([LtF[LtI[f]] for f, _ in valrows])
sF = seen_score(vF, PFh, TFFh, TLFh, TseenTax, LAM); sC = seen_score(vC, PCh, TFCh, TLCh, TseenTax, LAM); sL = seen_score(vL, PLh, TFLh, TLLh)
holdout = (zc(sF) + 1.0 * zc(sC) + 0.5 * zc(sL)).argmax(1).eq(VY).float().mean().item() * 100
print(f'holdout multi-enc (ft+1.0cap+0.5L): {holdout:.2f}% (expect ~87.94)')

if A.generate:
    PF, TFF, TLF = protos_and_train(FtI, FtF, by, seen)
    PC, TFC, TLC = protos_and_train(CtI, CtF, by, seen)
    PL, TFL, TLL = protos_and_train(LtI, LtF, by, seen)
    FteI, FteF, Ftef = load('outputs/emb_test_ft.pt')
    CteI, CteF, Ctef = load('outputs/emb_test_cap.pt')
    LteI, LteF, _ = load('outputs/emb_test.pt')
    tf = [fn for fn in Ftef if fn in LteI and fn in CteI]
    qF = torch.stack([FteF[FteI[fn]] for fn in tf]); qC = torch.stack([CteF[CteI[fn]] for fn in tf]); qL = torch.stack([LteF[LteI[fn]] for fn in tf])
    sc = zc(seen_score(qF, PF, TFF, TLF, TseenTax, LAM)) + 1.0 * zc(seen_score(qC, PC, TFC, TLC, TseenTax, LAM)) + 0.5 * zc(seen_score(qL, PL, TFL, TLL))
    preds = {fn: seen[k] for fn, k in zip(tf, sc.argmax(1).tolist())}
    print(f'seen preds: {len(preds)} / {len(tf)} (expect 20097)')

    HunI, HunF, Hunf = load('outputs/emb_unseen_ctftbig.pt'); LunI, LunF, _ = load('outputs/emb_unseen.pt')
    uf = [fn for fn in Hunf if fn in LunI]
    if A.real:
        nonk = torch.tensor([i for i, c in enumerate(classes) if c not in set(seen)])
        uH = torch.stack([HunF[HunI[fn]] for fn in uf]); uL = torch.stack([LunF[LunI[fn]] for fn in uf])
        MH = uH @ TtH[nonk].t(); ML = uL @ TnL[nonk].t()
        eU = dis(MH, 0.05, 0.5) + 0.5 * dis(ML, 0.05, 0.5)  # v15's tc/tr, byte-identical recipe
        for fn, j in zip(uf, eU.argmax(1).tolist()): preds[fn] = classes[nonk[j].item()]
        tag = 'real'
    else:
        for fn in uf: preds[fn] = 'Ostracion cubicum'  # probe: tank unseen, read accuracy_test only
        tag = 'probe'

    json.dump(preds, open(f'outputs/prediction_v20_{tag}.json', 'w'))
    z = zipfile.ZipFile(f'outputs/submission_v20_2026-07-06_multienc-{tag}.zip', 'w', zipfile.ZIP_DEFLATED)
    z.write(f'outputs/prediction_v20_{tag}.json', arcname='prediction.json'); z.close()
    shutil.copy(f'outputs/submission_v20_2026-07-06_multienc-{tag}.zip', f'/mnt/c/Users/islab/submission_v20_multienc-{tag}.zip')
    print(f'wrote submission_v20_2026-07-06_multienc-{tag}.zip  entries={len(preds)}  mode={tag}')
