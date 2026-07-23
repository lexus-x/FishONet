"""Stage 3 (diagnostic only, no labels, no upload): does transductive prototype
refinement actually ENGAGE sensibly on the real 35,665-image eval batch -- the batch
holdout structurally can't represent (no camera/domain shift in the holdout split)?

No accuracy is measurable here (no real labels available locally). Reports: churn,
novelty-distribution shift, per-class eligible-count histogram, and how much churn
concentrates in the extreme-high-novelty band (likely-true-unseen territory) vs
mid-novelty (likely-actually-seen, borderline) -- extreme-band churn would mean the
gate is failing on the real, actually-shifted data even if it looked fine on holdout.

Reuses build_gamma_sweep.py's loading pattern verbatim (full train protos, real
test+unseen combined query batch). Picks its (alpha, nov_q, conf_q) from
handoff/unified_frontier_ctftbig_protorefine.json's stage1_best if present, else falls
back to a mid-grid default.
"""
import json, pickle, torch, os
import torch.nn.functional as F
from collections import defaultdict

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_num_threads(8)
D = 'data/dl'


def load(p):
    d = torch.load(p, weights_only=False)
    return {fn: i for i, fn in enumerate(d['files'])}, F.normalize(d['feats'].float(), dim=-1), list(d['files'])


def zc(M):
    return (M - M.mean()) / (M.std() + 1e-6)


txtHt = torch.load('outputs/text_emb_h_taxon.pt', weights_only=False)
classes = list(pickle.load(open(f'{D}/all_classes.pkl', 'rb')))
ci = {c: i for i, c in enumerate(classes)}
TtH = F.normalize(txtHt['emb_taxon'].float(), dim=-1).to(dev)
lab = json.load(open(f'{D}/label_train.json'))

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
    return (out, ps_all.max(1).values) if ret_maxproto else out


FteI, FteF, Ftef = load('outputs/emb_test_ft.pt')
CteI, CteF, _ = load('outputs/emb_test_cap.pt')
LteI, LteF, _ = load('outputs/emb_test.pt')
HteI, _, Htef = load('outputs/emb_test_h.pt')
tf = sorted(fn for fn in Htef if fn in FteI and fn in CteI and fn in LteI)
FunI, FunF, _ = load('outputs/emb_unseen_ft.pt')
CunI, CunF, _ = load('outputs/emb_unseen_cap.pt')
LunI, LunF, _ = load('outputs/emb_unseen.pt')
HunI, _, Hunf = load('outputs/emb_unseen_h.pt')
uf = sorted(fn for fn in Hunf if fn in FunI and fn in CunI and fn in LunI)
all_files = tf + uf
n_t = len(tf)
print(f'combined real eval batch: {len(all_files)} (expect 35665)')

qF = torch.cat([torch.stack([FteF[FteI[fn]] for fn in tf]), torch.stack([FunF[FunI[fn]] for fn in uf])])
qC = torch.cat([torch.stack([CteF[CteI[fn]] for fn in tf]), torch.stack([CunF[CunI[fn]] for fn in uf])])
qL = torch.cat([torch.stack([LteF[LteI[fn]] for fn in tf]), torch.stack([LunF[LunI[fn]] for fn in uf])])

sc_ft, maxproto = seen_score(qF, PF, TFF, TLF, TseenTax, LAM, ret_maxproto=True)
sc_cap = seen_score(qC, PC, TFC, TLC, TseenTax, LAM)
sc_l = seen_score(qL, PL, TFL, TLL)
seen_block = zc(sc_ft) + 1.0 * zc(sc_cap) + 0.5 * zc(sc_l)
novelty = 1.0 - (maxproto - maxproto.min()) / (maxproto.max() - maxproto.min() + 1e-6)
baseline_pred = kept_idx[seen_block.argmax(1)]
print(f'novelty: mean={novelty.mean().item():.3f} median={novelty.median().item():.3f} '
      f'p90={torch.quantile(novelty, 0.9).item():.3f}')


def refine_prototypes(P, qF_this, seen_block0, novelty0, alpha, nov_q, conf_q, min_count=2):
    conf, pred = F.softmax(seen_block0, dim=1).max(1)
    gate = (novelty0 <= torch.quantile(novelty0, nov_q)) & (conf >= torch.quantile(conf, conf_q))
    w = gate.float() * conf
    qFd = qF_this.to(P.device)
    sums = torch.zeros_like(P).index_add_(0, pred, qFd * w.unsqueeze(1))
    wsum = torch.zeros(P.shape[0], device=P.device).index_add_(0, pred, w)
    cnt = torch.zeros(P.shape[0], device=P.device).index_add_(0, pred, gate.float())
    centroid = F.normalize(sums / wsum.clamp(min=1e-8).unsqueeze(1), dim=-1)
    eligible = cnt >= min_count
    P2 = P.clone()
    P2[eligible] = F.normalize((1 - alpha) * P[eligible] + alpha * centroid[eligible], dim=-1)
    return P2, gate, pred, cnt


# pick config from stage-1 result if available
alpha, nov_q, conf_q = 0.3, 0.5, 0.7
src = 'default (stage-1 result not found)'
frontier_path = 'handoff/unified_frontier_ctftbig_protorefine.json'
if os.path.exists(frontier_path):
    fr = json.load(open(frontier_path))
    if fr.get('stage1_best'):
        b = fr['stage1_best']
        alpha, nov_q, conf_q = b['alpha'], b['nov_q'], b['conf_q']
        src = f'stage1_best from {frontier_path}'
print(f'using config alpha={alpha} nov_q={nov_q} conf_q={conf_q} (source: {src})')

P2F, gate, pred, cnt = refine_prototypes(PF, qF, seen_block, novelty, alpha, nov_q, conf_q)
P2C, _, _, _ = refine_prototypes(PC, qC, seen_block, novelty, alpha, nov_q, conf_q)
P2L, _, _, _ = refine_prototypes(PL, qL, seen_block, novelty, alpha, nov_q, conf_q)
sc_ft2 = seen_score(qF, P2F, TFF, TLF, TseenTax, LAM)
sc_cap2 = seen_score(qC, P2C, TFC, TLC, TseenTax, LAM)
sc_l2 = seen_score(qL, P2L, TFL, TLL)
seen_block2 = zc(sc_ft2) + 1.0 * zc(sc_cap2) + 0.5 * zc(sc_l2)
refined_pred = kept_idx[seen_block2.argmax(1)]

churn_mask = baseline_pred != refined_pred
churn_pct = churn_mask.float().mean().item() * 100
gated_n = int(gate.float().sum().item())
gated_pct = 100.0 * gated_n / len(all_files)
eligible_classes = int((cnt >= 2).sum().item())

# churn concentration by novelty band -- the key sanity check: real engagement should
# concentrate in the MID band (borderline-seen), not the extreme-high band (likely true unseen)
bands = [(0.0, 0.33, 'low(likely-seen)'), (0.33, 0.66, 'mid(borderline)'), (0.66, 1.01, 'high(likely-unseen)')]
band_stats = []
for lo, hi, name in bands:
    m = (novelty >= lo) & (novelty < hi)
    n_band = int(m.sum().item())
    churn_in_band = int((churn_mask & m).sum().item())
    band_stats.append(dict(band=name, lo=lo, hi=hi, n=n_band,
                            churn_n=churn_in_band,
                            churn_pct_of_band=100.0 * churn_in_band / max(n_band, 1)))
    print(f'novelty band {name}: n={n_band} churn={churn_in_band} ({100.0*churn_in_band/max(n_band,1):.2f}%)')

result = dict(
    n_total=len(all_files), config=dict(alpha=alpha, nov_q=nov_q, conf_q=conf_q),
    novelty_mean=novelty.mean().item(), novelty_median=novelty.median().item(),
    novelty_p90=torch.quantile(novelty, 0.9).item(),
    churn_pct=churn_pct, gated_n=gated_n, gated_pct=gated_pct,
    eligible_classes=eligible_classes, total_seen_classes=S,
    band_stats=band_stats,
)
json.dump(result, open('handoff/diag_protorefine_realbatch.json', 'w'), indent=2)
print(f'churn={churn_pct:.2f}%  gated={gated_pct:.2f}% of batch  eligible_classes={eligible_classes}/{S}')
print('wrote handoff/diag_protorefine_realbatch.json')
