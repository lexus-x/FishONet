"""v85: give the gate BOTH heads' learned confidences on a common scale.

The last gate test (post-v83) added only the SEEN re-ranker's confidence: -0.065, dead. But the
quantity that should matter is the COMPARISON -- "how well does the seen head explain this image"
vs "how well does the unseen head explain it" -- on a common calibrated scale. HANDOFF 2026-08-08
killed unified/soft routing precisely because the two heads' raw scores are not comparable
(different legs, different normalisations). Two learned re-rankers each emit a calibrated
P(top candidate is correct), which is exactly the missing comparability.

Worth the compute because gate points transfer at ~3.4x (v77: +0.523 proj -> +1.767 real) while
per-candidate points transfer at ~0.03x.

Needs the 10 unseen legs for ALL 14,184 holdout rows over the full 12,757 candidate pool (previous
work only ever built them for the 2,318 pseudo-novel rows). Both re-ranker confidences are
OUT-OF-FOLD to avoid self-leakage.

  conda activate onet && python research/gate_dualconf_v85.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import FishData, dbnorm, load_emb  # noqa: E402
from crop_chase53_proxy import max_dbnorm  # noqa: E402
from inat_maxpool_proxy import score_maxpool  # noqa: E402
from learned_gate_v77 import (  # noqa: E402
    FEATS, LAM, MEMBERS, NFOLD, OUT, auc, deploy_weights, dev, holdout_split, load_train_embs,
    operating_point, standardize, zc,
)
from rerank_leakfree_v82 import pool_features  # noqa: E402
from rerank_seen_v83 import SIGNALS, cand_features, components  # noqa: E402
from rerank_seen_v83 import K as K_SEEN  # noqa: E402
from rerank_unseen_v81 import DEPLOYED_W, K as K_UNS, LEGS  # noqa: E402
from v50_shiftbank_proxy import (  # noqa: E402
    BANK_336, BANK_CTFT, FROZEN_PROTO, FROZEN_Q, LORA_PROTO, LORA_Q, proto_leg,
)

VIEW_KEYS = ['center', 'squash'] + [f'ostrip{i}' for i in range(5)]
HOLD_VIEWS7 = f'{OUT}/emb_holdout_all_ctft_views7.pt'
CACHE = f'{OUT}/gate_dualconf_v85.pt'


def unseen_legs_all(files, cand):
    """The 10 deployed unseen legs for ARBITRARY holdout rows over the full candidate pool."""
    D = FishData()
    TtH_c, TnL_c = D.TtH[cand], D.TnL[cand]
    TTX = F.normalize(torch.load(f'{OUT}/text_emb_h_promptens.pt',
                                 weights_only=False)['emb_taxctx'].float(), dim=-1)[cand]
    Ttb = F.normalize(torch.load(f'{OUT}/text_emb_taxabind_taxctx.pt',
                                 weights_only=False)['emb_taxctx'].float(), dim=-1)[cand]

    def q(path):
        idx, feats, _ = load_emb(path)
        return torch.stack([feats[idx[fn]] for fn in files])

    legs = {
        't_ctft_taxon': dbnorm(q(f'{OUT}/emb_train_ctftshift.pt') @ TtH_c.t()),
        't_L_name': dbnorm(q(f'{OUT}/emb_train.pt') @ TnL_c.t()),
        't_336v2_taxon': dbnorm(q(f'{OUT}/emb_train_fullft336_v2.pt') @ TtH_c.t()),
        't_ftshift_taxon': dbnorm(q(f'{OUT}/emb_train_ftshift.pt') @ TtH_c.t()),
        't_ctft_taxctx': dbnorm(q(f'{OUT}/emb_train_ctftshift.pt') @ TTX.t()),
        't_taxabind': dbnorm(q(f'{OUT}/emb_train_taxabind.pt') @ Ttb.t()),
    }
    print('  text legs done', flush=True)
    d = torch.load(HOLD_VIEWS7, weights_only=False)
    assert d['files'] == files
    tens = {k: F.normalize(d['views'][k].float(), dim=-1) for k in VIEW_KEYS}
    bank = torch.load(BANK_CTFT, weights_only=False)['bank']
    legs['bank_ct'] = max_dbnorm(tens, VIEW_KEYS, bank, cand)
    print('  ctft crop-max bank done', flush=True)
    legs['bank_336'] = dbnorm(score_maxpool(q(f'{OUT}/emb_train_fullft336shift.pt'),
                                            torch.load(BANK_336, weights_only=False)['bank'],
                                            cand, topm=4))
    legs['b2_frozen'] = proto_leg(q(FROZEN_Q), FROZEN_PROTO, cand)
    legs['b2_lora'] = proto_leg(q(LORA_Q), LORA_PROTO, cand)
    print('  image legs done', flush=True)
    n_photos = torch.zeros(len(cand))
    for j, g in enumerate(cand.tolist()):
        p = bank.get(g)
        if p is not None:
            n_photos[j] = p.shape[0] if hasattr(p, 'shape') else len(p)
    return legs, n_photos


def seen_signals(sp, files):
    train, *_ = load_train_embs(), None
    train = load_train_embs()[0]
    kept, ci, trby = sp['kept'], sp['ci'], sp['trby']
    k2i = {c: i for i, c in enumerate(kept)}
    S = len(kept)
    TtH = F.normalize(torch.load(f'{OUT}/text_emb_h_taxon.pt',
                                 weights_only=False)['emb_taxon'].float(), dim=-1).to(dev)
    Tseen = TtH[torch.tensor([ci[c] for c in kept], device=dev)]
    n_train = torch.tensor([len(trby[c]) for c in kept], dtype=torch.float32, device=dev)
    sig, block = {}, torch.zeros(len(files), S, device=dev)
    for t, hs, w in MEMBERS:
        if w == 0:
            continue
        idx, feats, _ = train[t]
        P = torch.zeros(S, feats.shape[1])
        cnt = torch.zeros(S)
        TF, TL = [], []
        for c in kept:
            for fn in trby[c]:
                f = feats[idx[fn]]
                P[k2i[c]] += f
                cnt[k2i[c]] += 1
                TF.append(f)
                TL.append(k2i[c])
        P = F.normalize(P / cnt.clamp(min=1).unsqueeze(1), dim=-1).to(dev)
        Q = torch.stack([feats[idx[fn]] for fn in files]).to(dev)
        ps, cm, tx = components(Q, P, torch.stack(TF).to(dev), torch.tensor(TL, device=dev), Tseen, S)
        sig[f'{t}_proto'], sig[f'{t}_cmax'], sig[f'{t}_taxon'] = ps, cm, tx
        block += w * zc(ps + 2.0 * cm + (LAM * tx if hs else 0.0))
        print(f'  seen {t} done', flush=True)
    sig['block'] = block
    return sig, block, n_train, k2i


def oof_conf(X, Y, folds, train_mask):
    """Out-of-fold calibrated confidence: max and top-2 gap of the re-ranker score."""
    nq, k, nf = X.shape
    Xf = X.reshape(nq * k, nf).cpu().numpy()
    Yf = Y.reshape(nq * k).numpy()
    qf = np.repeat(folds, k)
    tm = np.repeat(train_mask, k)
    oof = np.zeros(nq * k)
    for f in range(NFOLD):
        tr = (qf != f) & tm
        va = qf == f
        mu, sd = Xf[tr].mean(0), Xf[tr].std(0) + 1e-6
        m = LogisticRegression(max_iter=4000, C=1.0)
        m.fit((Xf[tr] - mu) / sd, Yf[tr])
        oof[va] = m.decision_function((Xf[va] - mu) / sd)
    sc = torch.tensor(oof, dtype=torch.float32).reshape(nq, k)
    t2 = sc.topk(2, dim=1).values
    return t2[:, 0], t2[:, 0] - t2[:, 1]


def main():
    torch.set_num_threads(8)
    train, tb, b2f, b2l = load_train_embs()
    sp = holdout_split(train, tb, b2f, b2l)
    files, y, cls, ns = sp['val_files'], sp['y'], sp['cls'], sp['n_seen']
    N = len(files)
    uniq = sorted(set(cls.tolist()))
    fold_of = {c: i % NFOLD for i, c in enumerate(uniq)}
    folds = np.array([fold_of[c] for c in cls])

    if os.path.exists(CACHE):
        d = torch.load(CACHE, weights_only=False)
        sconf, sgap, uconf, ugap = d['sconf'], d['sgap'], d['uconf'], d['ugap']
        print('loaded cached confidences', flush=True)
    else:
        D = FishData()
        cand = D.cand
        print(f'building seen signals for all {N} rows ...', flush=True)
        sig, block, n_train, k2i = seen_signals(sp, files)
        topk_s = block.topk(K_SEEN, dim=1).indices
        Xs, _ = cand_features(sig, topk_s, n_train, block)
        gold_s = torch.tensor([k2i.get(c, -1) for c in cls], device=dev)
        Ys = (topk_s == gold_s[:, None]).float().cpu()
        del sig, block
        torch.cuda.empty_cache()
        sconf, sgap = oof_conf(Xs.cpu(), Ys, folds, y == 1)   # trained on SEEN rows only
        del Xs
        print(f'building unseen legs for all {N} rows over {len(cand)} candidates ...', flush=True)
        legs, n_photos = unseen_legs_all(files, cand)
        fused = sum(w * legs[n] for n, w in zip(LEGS, DEPLOYED_W))
        topk_u = fused.topk(K_UNS, dim=1).indices
        Xu, _ = pool_features(legs, topk_u, n_photos, fused)
        cpos = {int(c): j for j, c in enumerate(cand.tolist())}
        gold_u = torch.tensor([cpos.get(D.ci[c], -1) for c in cls])
        Yu = (topk_u == gold_u[:, None]).float()
        uconf, ugap = oof_conf(Xu, Yu, folds, y == 0)         # trained on PSEUDO rows only
        torch.save({'sconf': sconf, 'sgap': sgap, 'uconf': uconf, 'ugap': ugap}, CACHE)
        print(f'wrote {CACHE}', flush=True)

    b = torch.load(f'{OUT}/gate_feats_holdout_v77.pt', weights_only=False)
    X12 = b['X']
    assert X12.shape[0] == N
    extra = torch.stack([sconf, sgap, uconf, ugap, sconf - uconf], dim=1)
    X = torch.cat([X12, extra], dim=1)
    ALL = FEATS + ['s_conf', 's_gap', 'u_conf', 'u_gap', 'conf_diff']
    w = deploy_weights(y)
    Z = standardize(X, w)
    base = Z[:, ALL.index('sb_max')] + 2.0 * Z[:, ALL.index('tm_ctft')]

    def run(cols, tag):
        o = torch.zeros(N)
        for f in range(NFOLD):
            tr, va = folds != f, folds == f
            Ztr = standardize(X[tr], deploy_weights(y[tr]))
            Zva = standardize(X[va], deploy_weights(y[va]))
            m = LogisticRegression(max_iter=20000, C=100.0)
            m.fit(Ztr[:, cols].numpy(), y[tr], sample_weight=deploy_weights(y[tr]).numpy())
            o[torch.tensor(va)] = torch.tensor(m.decision_function(Zva[:, cols].numpy()),
                                               dtype=torch.float32)
        t, n, p = operating_point(o, y, w)
        print(f'{tag:28s} AUC={auc(o, y, w):.4f} TPR={100*t:.2f} TNR={100*n:.2f} proj={p:.3f}',
              flush=True)
        return p, o

    print()
    t, n, p = operating_point(base, y, w)
    print(f'{"v56 gate":28s} AUC={auc(base, y, w):.4f} TPR={100*t:.2f} TNR={100*n:.2f} proj={p:.3f}')
    p79, _ = run(list(range(12)), 'v79 (12 feat)')
    p85, o85 = run(list(range(len(ALL))), 'v85 (+dual confidence)')
    run(list(range(12)) + [ALL.index('conf_diff')], 'v79 + conf_diff only')
    print(f'\nv85 - v79 = {p85 - p79:+.3f} proj-pt   (kill bar +0.30; gate transfer ~3.4x)', flush=True)
    json.dump({'p79': p79, 'p85': p85, 'delta': p85 - p79},
              open(f'{OUT}/gate_dualconf_v85.json', 'w'), indent=2)
    if p85 - p79 >= 0.30:
        import pickle
        m = LogisticRegression(max_iter=20000, C=100.0)
        m.fit(Z.numpy(), y, sample_weight=w.numpy())
        pickle.dump({'model': m, 'kind': 'logistic C=100 dualconf', 'feats': ALL,
                     'delta': p85 - p79, 'clears_kill': True, 'seen_frac': 0.60},
                    open(f'{OUT}/learned_gate_v85.pkl', 'wb'))
        print(f'wrote {OUT}/learned_gate_v85.pkl', flush=True)


if __name__ == '__main__':
    main()
