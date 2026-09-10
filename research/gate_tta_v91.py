"""v91: test-time adaptation of the learned gate on the eval batch — the last
untested lever class after v90 killed training-side shift adaptation.

The gate's eval-folder AUC (~0.911) sits 7pt under its holdout AUC (0.981) and v90
showed the gap is content-level domain shift, unfixable from the training side.
Transduction is established practice here (Sinkhorn scored fine, HANDOFF:261-267;
wz1 standardizes on the eval batch). This adapts the gate's decision function ON
the eval features with folder-blind self-training.

COMPLIANCE: pseudo-labels come only from the deployed-style gate's own scores
(no folder identity anywhere in training). Eval-folder labels are used ONLY to
MEASURE AUC/TPR/TNR — the same slot-free diagnostic as HANDOFF:399. No
hyperparameter is tuned on eval-folder AUC (folder-fit guard, HANDOFF:361): both
variants below are fixed a priori and both are reported.

11 features: the full v77 set minus b2l_max — the entire bioclip2_lora_v2
embedding family vanished from outputs/ by 2026-09-01 (train+test+unseen), so the
deployed 12-feature pickle cannot be scored on eval either; the reference gate is
retrained on holdout at 11 features instead (b2l coefficient was minor).

Pre-registered verdict: TTA is DEAD if best ΔAUC vs the reference < +0.010.

  conda activate onet && python research/gate_tta_v91.py
"""
from __future__ import annotations

import json
import os
import pickle
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT
from extract_holdout_shift_v90 import load_train_embs_v90
from learned_gate_v77 import (
    FEATS, MEMBERS, TRAIN, W_SEEN, W_UNS, auc, bank_scores, dbnorm, deploy_weights,
    dev, load, proto_max, standardize, wz1, zc,
)

D = 'data/dl'
FEATS11 = [f for f in FEATS if f != 'b2l_max']
C = 100.0
SEEN_FRAC = 0.60
A_COND, B_COND = 0.87018, 0.29533          # v83 real calibration (HANDOFF f-curve table)
TAXABIND_W = 1.0
B2_PROTO_FROZEN = f'{OUT}/inat_tol_merged_b2_a05.pt'
BANK_CTFT = f'{OUT}/inat_photo_bank_ctftshift.pt'
# pre-registered TTA pseudo-label bands (fractions of the eval batch by gate score)
PSEUDO_SEEN_TOP = 0.40
PSEUDO_UNS_BOT = 0.25
CACHE = f'{OUT}/gate_feats_eval_v91.pt'


def build_eval_feats():
    """The 11 gate features on the full 35,665-image eval batch — the builder's
    gate-feature block (build_v77_learned_gate.py:126-283) minus crop/ostrip head
    legs (not gate inputs) and minus the vanished b2l leg."""
    classes = list(pickle.load(open(f'{D}/all_classes.pkl', 'rb')))
    ci = {c: i for i, c in enumerate(classes)}
    NCLS = len(classes)
    lab = json.load(open(f'{D}/label_train.json'))
    TtH = F.normalize(torch.load(f'{OUT}/text_emb_h_taxon.pt', weights_only=False)['emb_taxon'].float(), dim=-1).to(dev)
    TnL = F.normalize(torch.load(f'{OUT}/text_emb.pt', weights_only=False)['emb_name'].float(), dim=-1).to(dev)
    TTX = F.normalize(torch.load(f'{OUT}/text_emb_h_promptens.pt', weights_only=False)['emb_taxctx'].float(), dim=-1).to(dev)
    Ttb = F.normalize(torch.load(f'{OUT}/text_emb_taxabind_taxctx.pt', weights_only=False)['emb_taxctx'].float(), dim=-1).to(dev)

    train = {t: load(f'{OUT}/{TRAIN[t]}.pt') for t, _, _ in MEMBERS}
    common = None
    for t, (idx, _, files) in train.items():
        s = set(files)
        common = s if common is None else (common & s)
    common = [fn for fn in common if fn in lab and lab[fn] in ci]
    by = defaultdict(list)
    for fn in common:
        by[lab[fn]].append(fn)
    seen = sorted(by.keys())
    s2i = {c: i for i, c in enumerate(seen)}
    S = len(seen)
    kept_idx = torch.tensor([ci[c] for c in seen]).to(dev)
    other_idx = torch.tensor([i for i in range(NCLS) if classes[i] not in set(seen)]).to(dev)
    TseenTax = TtH[kept_idx]
    LAM = 4.0
    print(f'seen classes {S} | unseen {len(other_idx)}', flush=True)

    def protos(idx, feats):
        P = torch.zeros(S, feats.shape[1])
        cnt = torch.zeros(S)
        TF, TL = [], []
        for c in seen:
            for fn in by[c]:
                f = feats[idx[fn]]
                P[s2i[c]] += f
                cnt[s2i[c]] += 1
                TF.append(f)
                TL.append(s2i[c])
        P = F.normalize(P / cnt.clamp(min=1).unsqueeze(1), dim=-1)
        return P.to(dev), torch.stack(TF).to(dev), torch.tensor(TL).to(dev)

    PR = {t: protos(*train[t][:2]) for t, _, _ in MEMBERS}

    def seen_score(qF, P, TF, TL, hspace):
        qF = qF.to(dev)
        out = torch.empty(qF.shape[0], S, device=dev)
        for i in range(0, qF.shape[0], 2000):
            e = qF[i:i + 2000]
            sim = e @ TF.t()
            cmax = torch.full((e.shape[0], S), -1e9, device=dev)
            cmax.scatter_reduce_(1, TL.unsqueeze(0).expand(e.shape[0], -1), sim, reduce='amax')
            sc = e @ P.t() + 2.0 * cmax
            if hspace:
                sc = sc + LAM * (e @ TseenTax.t())
            out[i:i + 2000] = sc
        return out

    test = {t: load(f'{OUT}/{TRAIN[t].replace("emb_train", "emb_test")}.pt') for t, _, _ in MEMBERS}
    unseen = {t: load(f'{OUT}/{TRAIN[t].replace("emb_train", "emb_unseen")}.pt') for t, _, _ in MEMBERS}
    tb_test = load(f'{OUT}/emb_test_taxabind.pt')
    tb_uns = load(f'{OUT}/emb_unseen_taxabind.pt')
    tf = sorted(set.intersection(*[set(f) for (_, _, f) in test.values()], set(tb_test[2])))
    uf = sorted(set.intersection(*[set(f) for (_, _, f) in unseen.values()], set(tb_uns[2])))
    all_files = tf + uf
    y_folder = np.array([1] * len(tf) + [0] * len(uf))   # MEASUREMENT ONLY
    print(f'eval batch {len(all_files)} (test={len(tf)} unseen={len(uf)})', flush=True)

    def qcat(t):
        ti, tfeat, _ = test[t]
        ui, ufeat, _ = unseen[t]
        return torch.cat([torch.stack([tfeat[ti[fn]] for fn in tf]),
                          torch.stack([ufeat[ui[fn]] for fn in uf])])

    Q = {t: qcat(t).to(dev) for t, _, _ in MEMBERS}
    Qtb = torch.cat([torch.stack([tb_test[1][tb_test[0][fn]] for fn in tf]),
                     torch.stack([tb_uns[1][tb_uns[0][fn]] for fn in uf])]).to(dev)

    seen_block = torch.zeros(len(all_files), S, device=dev)
    member_max = {}
    for t, hs, w in MEMBERS:
        if w == 0.0:
            continue
        sc = seen_score(Q[t], *PR[t], hs)
        member_max[t] = sc.max(1).values
        seen_block = seen_block + w * zc(sc)

    text_full = (dbnorm(Q['ctftshift'] @ TtH.t()) + 0.5 * dbnorm(Q['L'] @ TnL.t())
                 + 0.75 * dbnorm(Q['fullft336_v2'] @ TtH.t()) + 1.0 * dbnorm(Q['ftshift'] @ TtH.t())
                 + 1.0 * dbnorm(Q['ctftshift'] @ TTX.t())
                 + TAXABIND_W * dbnorm(Qtb @ Ttb.t()))
    tm_ctft = ((Q['ctftshift'] @ TtH[kept_idx].t()).max(1).values
               - (Q['ctftshift'] @ TtH[other_idx].t()).max(1).values)
    tm_full = text_full[:, kept_idx].max(1).values - text_full[:, other_idx].max(1).values
    del text_full

    print('scoring gate bank feature (center view) ...', flush=True)
    bank_raw = bank_scores(Q['ctftshift'], torch.load(BANK_CTFT, weights_only=False)['bank'], other_idx.tolist())

    def b2_leg_max():
        b2_test = load(f'{OUT}/emb_test_bioclip2.pt')
        b2_uns = load(f'{OUT}/emb_unseen_bioclip2.pt')
        Qb2 = torch.cat([torch.stack([b2_test[1][b2_test[0][fn]] for fn in tf]),
                         torch.stack([b2_uns[1][b2_uns[0][fn]] for fn in uf])]).to(dev)
        return proto_max(Qb2, B2_PROTO_FROZEN, other_idx)

    b2f = b2_leg_max()

    top2 = seen_block.topk(2, dim=1).values
    bank2 = bank_raw.topk(2, dim=1).values
    f = {'sb_max': top2[:, 0], 'sb_margin': top2[:, 0] - top2[:, 1],
         'sb_lse_gap': torch.logsumexp(seen_block, dim=1) - top2[:, 0],
         'tm_ctft': tm_ctft, 'tm_full': tm_full,
         'bank_max': bank2[:, 0], 'bank_margin': bank2[:, 0] - bank2[:, 1], 'b2f_max': b2f}
    for t in [m for m, _, w in MEMBERS if w > 0]:
        f[f'm_{t}'] = member_max[t]
    X = torch.stack([f[k] for k in FEATS11], dim=1).cpu()
    return X, y_folder


def op_point(score, y, frac=SEEN_FRAC):
    """Deployed routing rule: top `frac` of the batch by score -> seen route."""
    o = torch.argsort(score, descending=True)
    k = int(round(frac * len(score)))
    route = torch.zeros(len(score), dtype=torch.bool)
    route[o[:k]] = True
    ys = torch.tensor(y)
    tpr = (route & (ys == 1)).sum().item() / max((ys == 1).sum().item(), 1)
    tnr = (~route & (ys == 0)).sum().item() / max((ys == 0).sum().item(), 1)
    proj = 100 * (W_SEEN * A_COND * tpr + W_UNS * B_COND * tnr)
    return tpr, tnr, proj


def main():
    torch.set_num_threads(8)
    if os.path.exists(CACHE):
        d = torch.load(CACHE, weights_only=False)
        Xe, ye = d['X'], d['y']
        print(f'loaded cached eval features {tuple(Xe.shape)}', flush=True)
    else:
        Xe, ye = build_eval_feats()
        torch.save({'X': Xe, 'y': ye, 'feats': FEATS11}, CACHE)
        print(f'wrote {CACHE}', flush=True)

    # holdout 11-feature matrix from the v90 norm cache
    dh = torch.load(f'{OUT}/gate_feats_holdout_v90_norm.pt', weights_only=False)
    keep = [FEATS.index(f) for f in FEATS11]
    Xh, yh = dh['X'][:, keep], dh['y']
    wh = deploy_weights(yh)
    Zh = standardize(Xh, wh)
    we = torch.full((Xe.shape[0],), 1.0 / Xe.shape[0])
    Ze = standardize(Xe, we)   # builder standardizes the eval batch uniformly

    # reference: deployed-style gate (holdout-trained, C=100, all rows)
    ref = LogisticRegression(max_iter=2000, C=C)
    ref.fit(Zh.numpy(), yh, sample_weight=wh.numpy())
    s_ref = torch.tensor(ref.predict_proba(Ze.numpy())[:, 1], dtype=torch.float32)

    res = {}

    def record(name, s):
        a = auc(s, ye, we)
        tpr, tnr, proj = op_point(s, ye)
        res[name] = {'auc': a, 'tpr': tpr, 'tnr': tnr, 'proj': proj}
        print(f'{name:14s} AUC {a:.4f}  TPR {100*tpr:6.2f}  TNR {100*tnr:6.2f}  proj {proj:7.3f}', flush=True)

    print('\n=== eval-folder diagnostic (labels used for MEASUREMENT only) ===', flush=True)
    record('reference', s_ref)

    # folder-blind pseudo-labels from the reference gate's own scores
    o = torch.argsort(s_ref, descending=True)
    n = len(s_ref)
    top = o[:int(PSEUDO_SEEN_TOP * n)]
    bot = o[-int(PSEUDO_UNS_BOT * n):]
    idx = torch.cat([top, bot]).numpy()
    yp = np.array([1] * len(top) + [0] * len(bot))
    print(f'pseudo-labels: seen {len(top)} / unseen {len(bot)} / unlabeled {n - len(idx)}', flush=True)

    # V1: pure self-training on eval features
    m1 = LogisticRegression(max_iter=2000, C=C)
    m1.fit(Ze.numpy()[idx], yp)
    record('tta_pure', torch.tensor(m1.predict_proba(Ze.numpy())[:, 1], dtype=torch.float32))

    # V2: anchored — eval pseudo rows + holdout true rows
    Xtr = np.concatenate([Ze.numpy()[idx], Zh.numpy()])
    ytr = np.concatenate([yp, yh])
    wtr = np.concatenate([np.full(len(idx), 1.0 / len(idx)), wh.numpy() / wh.numpy().sum()])
    m2 = LogisticRegression(max_iter=2000, C=C)
    m2.fit(Xtr, ytr, sample_weight=wtr)
    record('tta_anchored', torch.tensor(m2.predict_proba(Ze.numpy())[:, 1], dtype=torch.float32))

    best = max(('tta_pure', 'tta_anchored'), key=lambda k: res[k]['auc'])
    dauc = res[best]['auc'] - res['reference']['auc']
    dproj = res[best]['proj'] - res['reference']['proj']
    print(f'\nbest = {best}  dAUC {dauc:+.4f}  dproj {dproj:+.3f}pt  (kill: dAUC < +0.010)', flush=True)
    verdict = 'CLEARS — worth a slot discussion' if dauc >= 0.010 else 'DEAD — TTA does not move eval routing'
    print(f'VERDICT: {verdict}', flush=True)
    json.dump({'results': res, 'best': best, 'dauc': dauc, 'dproj': dproj, 'verdict': verdict,
               'feats': FEATS11, 'pseudo_bands': [PSEUDO_SEEN_TOP, PSEUDO_UNS_BOT]},
              open(f'{OUT}/gate_tta_v91.json', 'w'), indent=2)
    print(f'wrote {OUT}/gate_tta_v91.json', flush=True)


if __name__ == '__main__':
    main()
