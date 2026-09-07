"""v77: learned novelty gate, evaluated at a FIXED f=0.60 operating point.

Why this shape and not the prior learned pipelines (v60/v63/v72-v76):

    overall = 0.5635*a_cond*TPR + 0.4365*b_cond*TNR = 0.4917*TPR + 0.1205*TNR

with the v56 real calibration (a_cond=87.26%, b_cond=27.61%). v60 (TPR 91.1/TNR 52.5)
and v63 (90.6/51.1) only MOVED the operating point toward seen, which trades unseen away
at ~4:1 and loses. The one intervention that raises seen AND unseen together is a better
gate RANKING at the same f. So: learn the ranking, pin f=0.60, change nothing else.

Features are computed identically here and in builders/build_v77_learned_gate.py.
The gate's photo-bank feature uses the CENTER view only -- holdout crop views exist only
for the 2,318 pseudo-novel rows, so the 7-view crop-max stays in the unseen HEAD
(untouched from v56) and never enters the gate.

Validation is class-disjoint 5-fold CV over the train-derived holdout, scored both as
AUC and as the f=0.60 operating-point projection (directly comparable to 51.616).

  conda activate onet && python research/learned_gate_v77.py
"""
from __future__ import annotations

import json
import os
import pickle
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

OUT = 'outputs'
D = 'data/dl'
dev = 'cuda' if torch.cuda.is_available() else 'cpu'

SEEN_FRAC = 0.60
W_SEEN, W_UNS = 0.5635, 0.4365
A_COND, B_COND = 0.8726, 0.2761          # v56 real calibration (HANDOFF 2026-08-25)
COEF_TPR = W_SEEN * A_COND               # 0.4917
COEF_TNR = W_UNS * B_COND                # 0.1205
V56_REAL = 51.61643067433057
KILL = 0.30                              # repo-standard kill bar, in overall-pt
NFOLD = 5

# The deployed chain (v81-v109) loads outputs/learned_gate_v79.pkl, which is this same
# 12-feature gate refit at C=100. Both knobs are env-configurable so that artifact is
# reproducible from this script; the defaults reproduce v77 exactly as before.
#   v77:  python research/learned_gate_v77.py
#   v79:  GATE_C=100 GATE_TAG=v79 python research/learned_gate_v77.py
GATE_C = float(os.environ.get('GATE_C', '1.0'))
GATE_TAG = os.environ.get('GATE_TAG', 'v77')

MEMBERS = [('ctftshift', True, 1.0), ('ftshift', True, 2.5), ('fullft336shift', True, 2.5),
           ('L', False, 0.0), ('fullft336_v2', True, 0.0)]
TRAIN = {'ctftshift': 'emb_train_ctftshift', 'ftshift': 'emb_train_ftshift',
         'fullft336shift': 'emb_train_fullft336shift', 'L': 'emb_train',
         'fullft336_v2': 'emb_train_fullft336_v2'}
ACTIVE = [t for t, _, w in MEMBERS if w > 0]
BANK_CTFT = f'{OUT}/inat_photo_bank_ctftshift.pt'
B2_PROTO_FROZEN = f'{OUT}/inat_tol_merged_b2_a05.pt'
B2_PROTO_LORA = f'{OUT}/inat_tol_merged_b2lora_a0.5.pt'
LAM = 4.0
TOPM = 4

FEATS = ['sb_max', 'sb_margin', 'sb_lse_gap', 'm_ctftshift', 'm_ftshift', 'm_fullft336shift',
         'tm_ctft', 'tm_full', 'bank_max', 'bank_margin', 'b2f_max', 'b2l_max']


def load(p):
    d = torch.load(p, weights_only=False)
    return {fn: i for i, fn in enumerate(d['files'])}, F.normalize(d['feats'].float(), dim=-1), list(d['files'])


def zc(M):
    return (M - M.mean()) / (M.std() + 1e-6)


def dbnorm(S, tc=0.05, tr=0.5):
    return F.log_softmax(S / tc, dim=0) + F.log_softmax(S / tr, dim=1)


def wz1(v, w):
    """Weighted standardization. At eval the batch already carries the natural 56/44
    mix so w is uniform; on the holdout w re-weights the 84/16 count mix to match."""
    mu = (w * v).sum()
    sd = torch.sqrt((w * (v - mu) ** 2).sum()).clamp(min=1e-6)
    return (v - mu) / sd


def seen_score(qF, P, TF, TL, hspace, Tseen, S):
    qF = qF.to(dev)
    out = torch.empty(qF.shape[0], S, device=dev)
    for i in range(0, qF.shape[0], 2000):
        e = qF[i:i + 2000]
        sim = e @ TF.t()
        cmax = torch.full((e.shape[0], S), -1e9, device=dev)
        cmax.scatter_reduce_(1, TL.unsqueeze(0).expand(e.shape[0], -1), sim, reduce='amax')
        sc = e @ P.t() + 2.0 * cmax
        if hspace:
            sc = sc + LAM * (e @ Tseen.t())
        out[i:i + 2000] = sc
    return out


def bank_scores(Q, bank, other_list):
    """top-TOPM-mean cosine per candidate class, center view only."""
    Sraw = torch.full((Q.shape[0], len(other_list)), -1e4, device=dev)
    for j, gidx in enumerate(other_list):
        photos = bank.get(gidx)
        if photos is None or (hasattr(photos, 'numel') and photos.numel() == 0):
            continue
        if not isinstance(photos, torch.Tensor):
            photos = torch.stack(photos)
        photos = F.normalize(photos.float(), dim=-1).to(dev)
        sim = Q @ photos.t()
        Sraw[:, j] = sim.topk(min(TOPM, sim.shape[1]), dim=1).values.mean(dim=1)
    return Sraw


def proto_max(Q, proto_path, other_idx):
    P = F.normalize(torch.load(proto_path, weights_only=False)['protos'].float(), dim=-1).to(dev)[other_idx]
    has = P.norm(dim=-1) > 0.5
    return (Q @ P.t()).masked_fill(~has.unsqueeze(0), -1e4).max(1).values


def assemble(seen_block, member_max, tm_ctft, tm_full, bank_raw, b2f, b2l):
    """The 12 gate features. Identical definition on holdout and eval."""
    top2 = seen_block.topk(2, dim=1).values
    bank2 = bank_raw.topk(2, dim=1).values
    f = {
        'sb_max': top2[:, 0],
        'sb_margin': top2[:, 0] - top2[:, 1],
        'sb_lse_gap': torch.logsumexp(seen_block, dim=1) - top2[:, 0],
        'tm_ctft': tm_ctft,
        'tm_full': tm_full,
        'bank_max': bank2[:, 0],
        'bank_margin': bank2[:, 0] - bank2[:, 1],
        'b2f_max': b2f,
        'b2l_max': b2l,
    }
    for t in ACTIVE:
        f[f'm_{t}'] = member_max[t]
    return torch.stack([f[k] for k in FEATS], dim=1)


# ----------------------------------------------------------------------------- holdout

def load_train_embs():
    train = {t: load(f'{OUT}/{TRAIN[t]}.pt') for t, _, _ in MEMBERS}
    return train, load(f'{OUT}/emb_train_taxabind.pt'), \
        load(f'{OUT}/emb_train_bioclip2.pt'), load(f'{OUT}/emb_train_bioclip2_lora_v2.pt')


def holdout_split(train, tb_train, b2f_train, b2l_train):
    """The train-derived split, factored out so extractors and v78 get the SAME ordering.
    Identical logic to research/gate_holdout_check.py: rarest 20% classes -> pseudo-novel."""
    classes = list(pickle.load(open(f'{D}/all_classes.pkl', 'rb')))
    ci = {c: i for i, c in enumerate(classes)}
    lab = json.load(open(f'{D}/label_train.json'))
    common = None
    for t, (_, _, files) in train.items():
        s = set(files)
        common = s if common is None else (common & s)
    common = [fn for fn in common if fn in lab and lab[fn] in ci and fn in tb_train[0]
              and fn in b2f_train[0] and fn in b2l_train[0]]
    by = defaultdict(list)
    for fn in common:
        by[lab[fn]].append(fn)
    seen = sorted(by.keys())
    order = sorted(seen, key=lambda c: len(by[c]))
    pseudo = set(order[:int(len(seen) * 0.2)])
    kept = sorted(order[int(len(seen) * 0.2):])
    trby = defaultdict(list)
    val_seen, val_uns = [], []
    for c in seen:
        fns = sorted(by[c])
        if c in pseudo:
            val_uns += fns
            continue
        if len(fns) >= 3:
            k = max(1, round(0.2 * len(fns)))
            trby[c] += fns[:-k]
            val_seen += fns[-k:]
        else:
            trby[c] += fns
    val_files = val_seen + val_uns
    y = np.array([1] * len(val_seen) + [0] * len(val_uns))
    cls = np.array([lab[fn] for fn in val_files])
    return {'val_files': val_files, 'y': y, 'cls': cls, 'trby': trby, 'kept': kept,
            'pseudo': pseudo, 'classes': classes, 'ci': ci, 'n_seen': len(val_seen)}


def build_holdout():
    classes = list(pickle.load(open(f'{D}/all_classes.pkl', 'rb')))
    ci = {c: i for i, c in enumerate(classes)}
    NCLS = len(classes)
    lab = json.load(open(f'{D}/label_train.json'))
    TtH = F.normalize(torch.load(f'{OUT}/text_emb_h_taxon.pt', weights_only=False)['emb_taxon'].float(), dim=-1).to(dev)
    TnL = F.normalize(torch.load(f'{OUT}/text_emb.pt', weights_only=False)['emb_name'].float(), dim=-1).to(dev)
    TTX = F.normalize(torch.load(f'{OUT}/text_emb_h_promptens.pt', weights_only=False)['emb_taxctx'].float(), dim=-1).to(dev)
    Ttb = F.normalize(torch.load(f'{OUT}/text_emb_taxabind_taxctx.pt', weights_only=False)['emb_taxctx'].float(), dim=-1).to(dev)

    train, tb_train, b2f_train, b2l_train = load_train_embs()

    common = None
    for t, (_, _, files) in train.items():
        s = set(files)
        common = s if common is None else (common & s)
    common = [fn for fn in common if fn in lab and lab[fn] in ci and fn in tb_train[0]
              and fn in b2f_train[0] and fn in b2l_train[0]]
    by = defaultdict(list)
    for fn in common:
        by[lab[fn]].append(fn)
    seen = sorted(by.keys())

    # same split as research/gate_holdout_check.py: rarest 20% classes -> pseudo-novel
    order = sorted(seen, key=lambda c: len(by[c]))
    pseudo = set(order[:int(len(seen) * 0.2)])
    kept = sorted(order[int(len(seen) * 0.2):])
    k2i = {c: i for i, c in enumerate(kept)}
    Sk = len(kept)
    kept_idx = torch.tensor([ci[c] for c in kept], device=dev)
    other_idx = torch.tensor([i for i in range(NCLS) if classes[i] not in set(kept)], device=dev)

    trby = defaultdict(list)
    val_seen, val_uns = [], []
    for c in seen:
        fns = sorted(by[c])
        if c in pseudo:
            val_uns += fns
            continue
        if len(fns) >= 3:
            k = max(1, round(0.2 * len(fns)))
            trby[c] += fns[:-k]
            val_seen += fns[-k:]
        else:
            trby[c] += fns
    val_files = val_seen + val_uns
    y = np.array([1] * len(val_seen) + [0] * len(val_uns))
    cls = np.array([lab[fn] for fn in val_files])
    print(f'holdout: seen={len(val_seen)} pseudo-novel={len(val_uns)} '
          f'| kept classes={Sk} other={len(other_idx)}', flush=True)

    def protos(t):
        idx, feats, _ = train[t]
        P = torch.zeros(Sk, feats.shape[1])
        cnt = torch.zeros(Sk)
        TF, TL = [], []
        for c in kept:
            for fn in trby[c]:
                f = feats[idx[fn]]
                P[k2i[c]] += f
                cnt[k2i[c]] += 1
                TF.append(f)
                TL.append(k2i[c])
        P = F.normalize(P / cnt.clamp(min=1).unsqueeze(1), dim=-1)
        return P.to(dev), torch.stack(TF).to(dev), torch.tensor(TL, device=dev)

    PR = {t: protos(t) for t, _, _ in MEMBERS}
    Q = {t: torch.stack([train[t][1][train[t][0][fn]] for fn in val_files]).to(dev) for t, _, _ in MEMBERS}
    Qtb = torch.stack([tb_train[1][tb_train[0][fn]] for fn in val_files]).to(dev)
    Qb2f = torch.stack([b2f_train[1][b2f_train[0][fn]] for fn in val_files]).to(dev)
    Qb2l = torch.stack([b2l_train[1][b2l_train[0][fn]] for fn in val_files]).to(dev)
    Tseen = TtH[kept_idx]

    seen_block = torch.zeros(len(val_files), Sk, device=dev)
    member_max = {}
    for t, hs, w in MEMBERS:
        if w == 0:
            continue
        sc = seen_score(Q[t], *PR[t], hs, Tseen, Sk)
        member_max[t] = sc.max(1).values
        seen_block += w * zc(sc)

    text_full = (dbnorm(Q['ctftshift'] @ TtH.t()) + 0.5 * dbnorm(Q['L'] @ TnL.t())
                 + 0.75 * dbnorm(Q['fullft336_v2'] @ TtH.t()) + 1.0 * dbnorm(Q['ftshift'] @ TtH.t())
                 + 1.0 * dbnorm(Q['ctftshift'] @ TTX.t()) + 1.0 * dbnorm(Qtb @ Ttb.t()))
    tm_ctft = ((Q['ctftshift'] @ TtH[kept_idx].t()).max(1).values
               - (Q['ctftshift'] @ TtH[other_idx].t()).max(1).values)
    tm_full = text_full[:, kept_idx].max(1).values - text_full[:, other_idx].max(1).values
    del text_full

    print('scoring iNat photo bank (center view) ...', flush=True)
    bank_raw = bank_scores(Q['ctftshift'], torch.load(BANK_CTFT, weights_only=False)['bank'], other_idx.tolist())
    b2f = proto_max(Qb2f, B2_PROTO_FROZEN, other_idx)
    b2l = proto_max(Qb2l, B2_PROTO_LORA, other_idx)

    X = assemble(seen_block, member_max, tm_ctft, tm_full, bank_raw, b2f, b2l)
    return X.cpu(), y, cls


# --------------------------------------------------------------------------- scoring

def deploy_weights(y):
    w = np.where(y == 1, W_SEEN / max((y == 1).sum(), 1), W_UNS / max((y == 0).sum(), 1))
    return torch.tensor(w, dtype=torch.float32)


def operating_point(score, y, w):
    """Pin the routed-to-seen mass at SEEN_FRAC, then read TPR/TNR and project overall."""
    o = torch.argsort(score, descending=True)
    cw = torch.cumsum(w[o], 0)
    k = int(torch.searchsorted(cw, torch.tensor(SEEN_FRAC * cw[-1].item())).item())
    route_seen = torch.zeros_like(score, dtype=torch.bool)
    route_seen[o[:k + 1]] = True
    ys = torch.tensor(y)
    tpr = (w * route_seen * (ys == 1)).sum().item() / (w * (ys == 1)).sum().item()
    tnr = (w * ~route_seen * (ys == 0)).sum().item() / (w * (ys == 0)).sum().item()
    return tpr, tnr, 100 * (COEF_TPR * tpr + COEF_TNR * tnr)


def auc(score, y, w):
    o = torch.argsort(score)
    ys = torch.tensor(y)[o]
    ws = w[o]
    cum_neg = torch.cumsum(ws * (ys == 0), 0) - ws * (ys == 0) * 0.5
    num = (ws * (ys == 1) * cum_neg).sum().item()
    return num / ((ws * (ys == 1)).sum().item() * (ws * (ys == 0)).sum().item())


def standardize(X, w):
    return torch.stack([wz1(X[:, j], w) for j in range(X.shape[1])], dim=1)


def main():
    torch.set_num_threads(8)
    cache = f'{OUT}/gate_feats_holdout_v77.pt'
    if os.path.exists(cache):
        d = torch.load(cache, weights_only=False)
        X, y, cls = d['X'], d['y'], d['cls']
        print(f'loaded cached holdout features {tuple(X.shape)}', flush=True)
    else:
        X, y, cls = build_holdout()
        torch.save({'X': X, 'y': y, 'cls': cls, 'feats': FEATS}, cache)
        print(f'wrote {cache}', flush=True)

    w = deploy_weights(y)
    Z = standardize(X, w)
    base = Z[:, FEATS.index('sb_max')] + 2.0 * Z[:, FEATS.index('tm_ctft')]   # the v56 gate

    uniq = sorted(set(cls.tolist()))
    fold_of = {c: i % NFOLD for i, c in enumerate(uniq)}   # deterministic, class-disjoint
    folds = np.array([fold_of[c] for c in cls])

    models = {
        'logistic': lambda: LogisticRegression(max_iter=2000, C=GATE_C),
        'hgb': lambda: HistGradientBoostingClassifier(max_iter=200, max_depth=4,
                                                      learning_rate=0.06, min_samples_leaf=40,
                                                      l2_regularization=1.0, random_state=0),
    }
    oof = {k: torch.zeros(len(y)) for k in models}
    for f in range(NFOLD):
        tr, va = folds != f, folds == f
        Ztr = standardize(X[tr], deploy_weights(y[tr]))
        Zva = standardize(X[va], deploy_weights(y[va]))
        for name, mk in models.items():
            m = mk()
            m.fit(Ztr.numpy(), y[tr], sample_weight=deploy_weights(y[tr]).numpy())
            oof[name][torch.tensor(va)] = torch.tensor(
                m.predict_proba(Zva.numpy())[:, 1], dtype=torch.float32)
        print(f'  fold {f}: train={tr.sum()} val={va.sum()}', flush=True)

    rows = {}
    b_tpr, b_tnr, b_proj = operating_point(base, y, w)
    rows['v56_gate'] = {'auc': auc(base, y, w), 'tpr': b_tpr, 'tnr': b_tnr, 'proj': b_proj}
    for name in models:
        t, n, p = operating_point(oof[name], y, w)
        rows[name] = {'auc': auc(oof[name], y, w), 'tpr': t, 'tnr': n, 'proj': p}

    print('\n=== class-disjoint 5-fold CV, operating point pinned at f=0.60 ===', flush=True)
    print(f'{"gate":12s} {"AUC":>8s} {"TPR":>8s} {"TNR":>8s} {"proj%":>8s} {"vs v56":>8s}', flush=True)
    for name, r in rows.items():
        d = r['proj'] - b_proj
        print(f'{name:12s} {r["auc"]:8.4f} {100*r["tpr"]:8.2f} {100*r["tnr"]:8.2f} '
              f'{r["proj"]:8.3f} {d:+8.3f}', flush=True)

    best = max((k for k in models), key=lambda k: rows[k]['proj'])
    delta = rows[best]['proj'] - b_proj
    print(f'\nbest learned = {best}  delta = {delta:+.3f} proxy-pt  '
          f'(kill bar {KILL:+.2f})  -> {"CLEARS" if delta >= KILL else "FAILS"}', flush=True)

    # refit the winner on all folds for deployment
    m = models[best]()
    m.fit(Z.numpy(), y, sample_weight=w.numpy())
    payload = {'model': m, 'kind': best, 'feats': FEATS, 'cv': rows, 'delta': delta,
               'clears_kill': bool(delta >= KILL), 'seen_frac': SEEN_FRAC}
    with open(f'{OUT}/learned_gate_{GATE_TAG}.pkl', 'wb') as fh:
        pickle.dump(payload, fh)
    json.dump({'cv': rows, 'best': best, 'delta': delta, 'clears_kill': bool(delta >= KILL),
               'v56_real': V56_REAL, 'feats': FEATS},
              open(f'{OUT}/learned_gate_{GATE_TAG}.json', 'w'), indent=2)
    if best == 'logistic':
        print('\ncoefficients (standardized):', flush=True)
        for n, c in sorted(zip(FEATS, m.coef_[0]), key=lambda t: -abs(t[1])):
            print(f'  {n:20s} {c:+.4f}', flush=True)
    print(f'\nwrote {OUT}/learned_gate_{GATE_TAG}.pkl + .json  (C={GATE_C})', flush=True)


if __name__ == '__main__':
    main()
