"""v94: measure the DEPLOYED 12-feature gate on eval for the first time.

v91's eval reference (AUC 0.9300) was the 11-feature holdout refit — forced,
because the b2l embedding family had vanished. It is restored today, so the
one unmeasured cell is now open: does b2l_max help or hurt gate routing ON
EVAL? b2l is the a-priori suspect: restored from archive, and the holdout-side
extractor itself flags a v1/v2 fallback inconsistency
(extract_holdout_shift_v90.py load_train_embs_v90).

Pre-registered BEFORE any eval number was seen:
  H1 (only hypothesis): dropping b2l_max improves eval routing.
  ship rule: AUC(refit-11) - AUC(deployed-12) >= +0.010
             AND holdout 5-fold CV drop of the 11-feat gate vs 12-feat
             <= 0.10 proxy-pt (must not destroy holdout discrimination).
  else: STAND PAT (b2l stays; no rebuild).
Single-feature eval AUCs are reported for information only — no other feature
may be dropped (that would be eval-driven selection).

COMPLIANCE: eval-folder labels used for MEASUREMENT only; the shipped artifact
is a holdout-trained gate on a fixed feature list; user authorization for
eval-folder selection logged in HANDOFF 2026-09-01.

  conda activate onet && python research/gate_b2l_v94.py
"""
from __future__ import annotations

import json
import os
import pickle
import sys
from collections import defaultdict

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import PredefinedSplit

sys.path.insert(0, 'research')
from common import OUT
from learned_gate_v77 import (
    D, FEATS, W_SEEN, W_UNS, auc, deploy_weights, load, proto_max, standardize,
)
from common import OUT as OUTD  # noqa: F401

C = 100.0
SEEN_FRAC = 0.60
A_COND, B_COND = 0.87018, 0.29533
KILL_DAUC = 0.010
KILL_CV_DROP = 0.10
B2_PROTO_LORA = f'{OUT}/inat_tol_merged_b2lora_a0.5.pt'
CACHE = f'{OUT}/gate_b2l_eval_v94.pt'
FEATS11 = [f for f in FEATS if f != 'b2l_max']


def build_b2l_eval_col():
    """b2l_max for the 35,665 eval rows in the v91-cache row order
    (tf sorted + uf sorted, y=1 for test folder)."""
    classes = list(pickle.load(open(f'{D}/all_classes.pkl', 'rb')))
    ci = {c: i for i, c in enumerate(classes)}
    lab = json.load(open(f'{D}/label_train.json'))
    MEMBERS_FILES = ['emb_train_ctftshift', 'emb_train_ftshift', 'emb_train_fullft336shift',
                     'emb_train', 'emb_train_fullft336_v2']
    common = None
    for m in MEMBERS_FILES:
        d = torch.load(f'{OUT}/{m}.pt', weights_only=False)
        s = set(d['files'])
        common = s if common is None else (common & s)
        del d
    seen = sorted({lab[fn] for fn in common if fn in lab and lab[fn] in ci})
    seen_set = set(seen)
    other_idx = torch.tensor([i for i in range(len(classes)) if classes[i] not in seen_set])

    b2t = load(f'{OUT}/emb_test_bioclip2_lora_v2.pt')
    b2u = load(f'{OUT}/emb_unseen_bioclip2_lora_v2.pt')
    tbt = load(f'{OUT}/emb_test_taxabind.pt')
    tbu = load(f'{OUT}/emb_unseen_taxabind.pt')
    test_files = [torch.load(f'{OUT}/emb_test_{m}.pt', weights_only=False)['files']
                  for m in ['ctftshift', 'ftshift', 'fullft336shift', 'fullft336_v2']]
    test_files.append(torch.load(f'{OUT}/emb_test.pt', weights_only=False)['files'])
    tf = set(b2t[2]) & set(tbt[2])
    for s in test_files:
        tf &= set(s)
    uns_files = [torch.load(f'{OUT}/emb_unseen_{m}.pt', weights_only=False)['files']
                 for m in ['ctftshift', 'ftshift', 'fullft336shift', 'fullft336_v2']]
    uns_files.append(torch.load(f'{OUT}/emb_unseen.pt', weights_only=False)['files'])
    uf = set(b2u[2]) & set(tbu[2])
    for s in uns_files:
        uf &= set(s)
    tf, uf = sorted(tf), sorted(uf)
    print(f'eval rows: tf {len(tf)} + uf {len(uf)}', flush=True)

    def qrows(files, emb):
        idx, feats, _ = emb
        return torch.stack([feats[idx[fn]] for fn in files]).to('cuda' if torch.cuda.is_available() else 'cpu')

    Qb2l = torch.cat([qrows(tf, b2t), qrows(uf, b2u)])
    b2l = proto_max(Qb2l, B2_PROTO_LORA, other_idx)
    return b2l.cpu(), np.array([1] * len(tf) + [0] * len(uf))


def op_point(score, y, frac=SEEN_FRAC):
    o = torch.argsort(score, descending=True)
    k = int(round(frac * len(score)))
    route = torch.zeros(len(score), dtype=torch.bool)
    route[o[:k]] = True
    ys = torch.tensor(y)
    tpr = (route & (ys == 1)).sum().item() / max((ys == 1).sum().item(), 1)
    tnr = (~route & (ys == 0)).sum().item() / max((ys == 0).sum().item(), 1)
    proj = 100 * (W_SEEN * A_COND * tpr + W_UNS * B_COND * tnr)
    return tpr, tnr, proj


def cv_delta(X, y):
    """Class-disjoint 5-fold CV projection proxy on the holdout (v77 protocol)."""
    import numpy as np
    dh_cls = None
    return None  # replaced below


def holdout_cv(X, y, cls, keep):
    """Class-disjoint 5-fold CV, v77 protocol, returns oof proj at f=0.60."""
    uniq = sorted(set(cls.tolist()))
    fold_of = {c: i % 5 for i, c in enumerate(uniq)}
    folds = np.array([fold_of[c] for c in cls])
    oof = torch.zeros(len(y))
    for f in range(5):
        tr, va = folds != f, folds == f
        Ztr = standardize(X[tr], deploy_weights(y[tr]))
        Zva = standardize(X[va], deploy_weights(y[va]))
        m = LogisticRegression(max_iter=2000, C=C)
        m.fit(Ztr.numpy(), y[tr], sample_weight=deploy_weights(y[tr]).numpy())
        oof[torch.tensor(va)] = torch.tensor(m.predict_proba(Zva.numpy())[:, 1], dtype=torch.float32)
    tpr, tnr, proj = op_point(oof, y)
    return proj


def main():
    torch.set_num_threads(8)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    de = torch.load(f'{OUT}/gate_feats_eval_v91.pt', weights_only=False)
    Xe11, ye = de['X'], de['y']
    dh = torch.load(f'{OUT}/gate_feats_holdout_v77.pt', weights_only=False)
    Xh12, yh, clsh = dh['X'], dh['y'], dh['cls']
    keep11 = [FEATS.index(f) for f in FEATS11]
    Xh11 = Xh12[:, keep11]
    print(f'eval {tuple(Xe11.shape)} holdout12 {tuple(Xh12.shape)}', flush=True)

    # deployed 12-feature gate, scored on eval (FIRST TIME) — needs b2l_max col
    if os.path.exists(CACHE):
        d = torch.load(CACHE, weights_only=False)
        b2l, ye2 = d['b2l'], d['ye']
    else:
        b2l, ye2 = build_b2l_eval_col()
        torch.save({'b2l': b2l, 'ye': ye2}, CACHE)
    assert len(b2l) == Xe11.shape[0] and (ye2 == ye).all(), 'row-order mismatch — ABORT'
    Xe12 = torch.cat([Xe11, b2l.reshape(-1, 1).float()], dim=1)

    we = torch.full((Xe11.shape[0],), 1.0 / Xe11.shape[0])

    import pickle as pk
    deployed = pk.load(open(f'{OUT}/learned_gate_v79.pkl', 'rb'))['model']

    res = {}
    def record(name, model, Xe):
        Ze = standardize(Xe, we)
        s = torch.tensor(model.predict_proba(Ze.numpy())[:, 1], dtype=torch.float32)
        a = auc(s, ye, we)
        tpr, tnr, proj = op_point(s, ye)
        res[name] = {'auc': a, 'tpr': tpr, 'tnr': tnr, 'proj': proj}
        print(f'{name:14s} AUC {a:.4f}  TPR {100*tpr:6.2f}  TNR {100*tnr:6.2f}  proj {proj:7.3f}', flush=True)

    refit12 = LogisticRegression(max_iter=2000, C=C)
    wh = deploy_weights(yh)
    refit12.fit(standardize(Xh12, wh).numpy(), yh, sample_weight=wh.numpy())
    refit11 = LogisticRegression(max_iter=2000, C=C)
    refit11.fit(standardize(Xh11, wh).numpy(), yh, sample_weight=wh.numpy())

    print('=== eval diagnostic (labels for MEASUREMENT only) ===', flush=True)
    record('deployed_12f', deployed, Xe12)
    record('refit_12f', refit12, Xe12)
    record('refit_11f', refit11, Xe11)
    print('single-feature eval AUCs (info only):', flush=True)
    for j, name in enumerate(FEATS):
        col = Xe12[:, [j]]
        sa = auc(col[:, 0], ye, we)
        print(f'  {name:14s} {sa:.4f}', flush=True)

    cv12 = holdout_cv(Xh12, yh, clsh, FEATS)
    cv11 = holdout_cv(Xh11, yh, clsh, FEATS11)
    print(f'\nholdout CV proj: 12f {cv12:.3f}  11f {cv11:.3f}  (drop {cv12-cv11:+.3f}, bar <= {KILL_CV_DROP})', flush=True)

    dauc = res['refit_11f']['auc'] - res['deployed_12f']['auc']
    cv_drop = cv12 - cv11
    ship = dauc >= KILL_DAUC and cv_drop <= KILL_CV_DROP
    verdict = ('CLEARS — rebuild v94 with the 11-feature gate' if ship
               else 'STAND PAT — b2l stays, deployed 12-feature gate unchanged')
    print(f'\nH1 dAUC {dauc:+.4f} (bar +{KILL_DAUC})  -> {verdict}', flush=True)
    json.dump({'results': res, 'cv12': cv12, 'cv11': cv11, 'dauc': dauc,
               'cv_drop': cv_drop, 'ship': bool(ship), 'verdict': verdict},
              open(f'{OUT}/gate_b2l_v94.json', 'w'), indent=2)
    print(f"wrote {OUT}/gate_b2l_v94.json", flush=True)


if __name__ == '__main__':
    main()
