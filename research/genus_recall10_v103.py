"""v103: recall@10-level recheck of genus-level hierarchical backoff (seen route).

v100 found LAM_G=0 was argmax-optimal (genus bonus never helps top-1 on the
deployed 3-leg fusion). Task: recheck at recall@10 specifically -- a genus
signal might not move top-1 but could still pull the right species higher into
the top-10 for rows the base fusion currently ranks just outside it (the exact
quantity the seen re-ranker (research/rerank_seen_v83.py) depends on: it can
only re-rank within the base fusion's top-10, so recall@10 is the stack's true
ceiling, per research/probe2_retriever_recall.py's method).

Genus bonus construction verbatim from research/seen_genus_backoff_v100.py:
ctftshift-only proto+2*cmax pooled ACROSS sibling species of the same genus,
zc-normalized, added to the deployed 3-leg (1.0/2.5/2.5) base fusion score.

Gate: recall@10 at ANY nonzero LAM_G in the same pre-registered grid as v100
must beat the deployed baseline (98.854, per rerank_seen_v83.py header).
If none do -> DEAD, stop. If any do -> retrain the v83 re-ranker leak-free on
the MODIFIED (base+genus) fusion's top-10 and report the true stack accuracy
vs the 91.674 bar (need >=92.674).

  conda activate onet && python research/genus_recall10_v103.py
"""
from __future__ import annotations

import json
import pickle
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, 'research')
from common import OUT
from learned_gate_v77 import D, LAM, MEMBERS, NFOLD, dev, holdout_split, load_train_embs, zc

LAM_G_GRID = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
GENUS_LEG = 'ctftshift'
DEPLOYED = {'ctftshift': 1.0, 'ftshift': 2.5, 'fullft336shift': 2.5}
RECALL10_BASELINE = 98.854
K = 10
ACTIVE = [t for t, _, w in MEMBERS if w > 0]
SIGNALS = [f'{t}_{c}' for t in ACTIVE for c in ('proto', 'cmax', 'taxon')] + ['genus', 'block']


def member_scores(qF, P, TF, TL, Tseen, S):
    out = torch.empty(qF.shape[0], S, device=dev)
    for i in range(0, qF.shape[0], 2000):
        e = qF[i:i + 2000]
        sim = e @ TF.t()
        cmax = torch.full((e.shape[0], S), -1e9, device=dev)
        cmax.scatter_reduce_(1, TL.unsqueeze(0).expand(e.shape[0], -1), sim, reduce='amax')
        out[i:i + 2000] = e @ P.t() + 2.0 * cmax + LAM * (e @ Tseen.t())
    return out


def components(Q, P, TF, TL, Tseen, S):
    n = Q.shape[0]
    ps = torch.empty(n, S, device=dev)
    cm = torch.empty(n, S, device=dev)
    for i in range(0, n, 2000):
        e = Q[i:i + 2000]
        ps[i:i + 2000] = e @ P.t()
        sim = e @ TF.t()
        c = torch.full((e.shape[0], S), -1e9, device=dev)
        c.scatter_reduce_(1, TL.unsqueeze(0).expand(e.shape[0], -1), sim, reduce='amax')
        cm[i:i + 2000] = c
    return ps, cm, Q @ Tseen.t()


def recall_at(scores, yv, ks=(1, 5, 10)):
    topk = scores.topk(max(ks), dim=1).indices.cpu()
    return {k: 100 * (topk[:, :k] == yv.unsqueeze(1)).any(1).float().mean().item() for k in ks}


def cand_features(sig, topk, n_train, block):
    nq, k = topk.shape
    C = block.shape[1]
    cols, names = [], []
    for name in SIGNALS:
        M = sig[name]
        v = torch.gather(M, 1, topk)
        cols += [v - M.max(1, keepdim=True).values,
                 (v - M.mean(1, keepdim=True)) / (M.std(1, keepdim=True) + 1e-6)]
        names += [f'{name}_gapmax', f'{name}_z']
        r = torch.empty(nq, k, device=M.device)
        for s in range(0, nq, 256):
            e = min(s + 256, nq)
            r[s:e] = (M[s:e].unsqueeze(1) > v[s:e].unsqueeze(2)).sum(2).float()
        cols.append(r / C)
        names.append(f'{name}_pctrank')
    cols += [torch.arange(k, device=block.device).float().expand(nq, k) / k,
             torch.log1p(n_train[topk])]
    names += ['block_rank', 'log_ntrain']
    return torch.stack(cols, dim=2), names


def main():
    torch.set_num_threads(8)
    lab = json.load(open(f'{D}/label_train.json'))
    classes = list(pickle.load(open(f'{D}/all_classes.pkl', 'rb')))
    ci = {c: i for i, c in enumerate(classes)}

    train, tb_train, b2f_train, b2l_train = load_train_embs()
    sp = holdout_split(train, tb_train, b2f_train, b2l_train)
    trby, val_seen = sp['trby'], [f for f, y_ in zip(sp['val_files'], sp['y']) if y_ == 1]
    seen = sorted(sp['kept'])
    s2i = {c: i for i, c in enumerate(seen)}
    S = len(seen)
    kept_idx = torch.tensor([ci[c] for c in seen]).to(dev)
    yv = torch.tensor([s2i[lab[f]] for f in val_seen])
    n_train = torch.tensor([len(trby[c]) for c in seen], dtype=torch.float32, device=dev)
    print(f'val_seen rows {len(val_seen)} | seen classes {S}', flush=True)

    genus_of = {c: c.split()[0] for c in seen}
    genus_list = sorted(set(genus_of.values()))
    g2i = {g: i for i, g in enumerate(genus_list)}
    G = len(genus_list)
    species_genus_idx = torch.tensor([g2i[genus_of[c]] for c in seen]).to(dev)

    TtH = F.normalize(torch.load(f'{OUT}/text_emb_h_taxon.pt', weights_only=False)['emb_taxon'].float(), dim=-1).to(dev)
    Tseen = TtH[kept_idx]

    # --- deployed per-leg species-level base scores (identical to v100/v83) ---
    Zbase, sig, TF_by, TL_by, qF_by = {}, {}, {}, {}, {}
    block = torch.zeros(len(val_seen), S, device=dev)
    for t, hs, w in MEMBERS:
        if t not in DEPLOYED:
            continue
        idx, feats, _ = train[t]
        P = torch.zeros(S, feats.shape[1])
        cnt = torch.zeros(S)
        TFl, TLl = [], []
        for c in seen:
            for fn in trby[c]:
                if fn not in idx:
                    continue
                f = feats[idx[fn]]
                P[s2i[c]] += f
                cnt[s2i[c]] += 1
                TFl.append(f)
                TLl.append(s2i[c])
        P = F.normalize(P / cnt.clamp(min=1).unsqueeze(1), dim=-1).to(dev)
        TF = torch.stack(TFl).to(dev)
        TL = torch.tensor(TLl).to(dev)
        qF = torch.stack([feats[idx[fn]] for fn in val_seen]).to(dev)
        ps, cm, tx = components(qF, P, TF, TL, Tseen, S)
        sig[f'{t}_proto'], sig[f'{t}_cmax'], sig[f'{t}_taxon'] = ps, cm, tx
        base_t = ps + 2.0 * cm + LAM * tx
        Zbase[t] = zc(base_t)
        block += DEPLOYED[t] * Zbase[t]
        TF_by[t], TL_by[t], qF_by[t] = TF, TL, qF
        print(f'  {t} done', flush=True)

    base_rec = recall_at(block, yv)
    print(f"\ndeployed 3-leg: top1 {base_rec[1]:.3f}  recall@5 {base_rec[5]:.3f}  "
          f"recall@10 {base_rec[10]:.3f}  (given baseline recall@10={RECALL10_BASELINE})", flush=True)

    # --- genus-level prototype + cmax, pooled ACROSS sibling species (v100 recipe) ---
    TL_genus = species_genus_idx[TL_by[GENUS_LEG]]
    d = TF_by[GENUS_LEG].shape[1]
    Pg = torch.zeros(G, d, device=dev)
    cntg = torch.zeros(G, device=dev)
    Pg.scatter_add_(0, TL_genus.unsqueeze(1).expand(-1, d), TF_by[GENUS_LEG])
    cntg.scatter_add_(0, TL_genus, torch.ones_like(TL_genus, dtype=torch.float32))
    Pg = F.normalize(Pg / cntg.clamp(min=1).unsqueeze(1), dim=-1)

    q = qF_by[GENUS_LEG]
    sim = q @ TF_by[GENUS_LEG].t()
    cmaxg = torch.full((q.shape[0], G), -1e9, device=dev)
    cmaxg.scatter_reduce_(1, TL_genus.unsqueeze(0).expand(q.shape[0], -1), sim, reduce='amax')
    genus_raw_G = q @ Pg.t() + 2.0 * cmaxg
    genus_bonus = genus_raw_G[:, species_genus_idx]
    Zg = zc(genus_bonus)
    sig['genus'] = genus_bonus
    sig['block'] = block

    print('\n=== LAM_G grid: recall@10 with genus bonus added to deployed 3-leg fusion ===', flush=True)
    best_lam, best_r10 = None, base_rec[10]
    grid_results = {}
    for lam in LAM_G_GRID:
        f = block + lam * Zg
        rec = recall_at(f, yv)
        grid_results[lam] = rec
        tag = ' <-- BEATS BASELINE' if rec[10] > base_rec[10] else ''
        print(f'  LAM_G={lam:<5} top1 {rec[1]:6.3f}  recall@5 {rec[5]:6.3f}  '
              f'recall@10 {rec[10]:6.3f}  ({rec[10] - base_rec[10]:+.3f}){tag}', flush=True)
        if rec[10] > best_r10:
            best_lam, best_r10 = lam, rec[10]

    improved = best_lam is not None
    print(f'\nRECALL@10 GATE: {"IMPROVED" if improved else "NOT IMPROVED"} '
          f'(best {best_r10:.3f} @ LAM_G={best_lam}, baseline {base_rec[10]:.3f})', flush=True)
    result = {'recall10_baseline': base_rec[10], 'top1_baseline': base_rec[1],
              'recall5_baseline': base_rec[5], 'grid': grid_results,
              'best_lam': best_lam, 'best_recall10': best_r10, 'improved': improved}

    if not improved:
        print('\nGate did not clear -- STOPPING, no re-ranker retrain (per instructions).', flush=True)
        json.dump(result, open(f'{OUT}/genus_recall10_v103.json', 'w'), indent=2)
        print(f'wrote {OUT}/genus_recall10_v103.json', flush=True)
        return

    # ============== stage 2: leak-free re-ranker retrain on genus-augmented fusion ==============
    print(f'\n=== recall@10 improved -- retraining v83-style re-ranker at LAM_G={best_lam} ===', flush=True)
    fused = block + best_lam * Zg
    topk = fused.topk(K, dim=1).indices
    n_gold_in_k = 100 * (topk == yv.to(dev).unsqueeze(1)).any(1).float().mean().item()
    print(f'gold in top-{K}: {n_gold_in_k:.2f}%', flush=True)

    X, names = cand_features(sig, topk, n_train, fused)
    Y = (topk == yv.to(dev)[:, None]).float()
    X, Y = X.cpu(), Y.cpu()
    cls = np.array(sp['cls'][:len(val_seen)])

    nq, k, nf = X.shape
    stack_base = 100 * (Y[:, 0] > 0).float().mean().item()
    print(f'genus-fusion rank-1 (base of re-rank) = {stack_base:.3f}', flush=True)

    uniq = sorted(set(cls.tolist()))
    fold_of = {c: i % NFOLD for i, c in enumerate(uniq)}
    folds = np.array([fold_of[c] for c in cls])
    qfold = np.repeat(folds, k)
    Xf = X.reshape(nq * k, nf).numpy()
    Yf = Y.reshape(nq * k).numpy()

    oof = np.zeros(nq * k)
    for f in range(NFOLD):
        tr, va = qfold != f, qfold == f
        mu, sd = Xf[tr].mean(0), Xf[tr].std(0) + 1e-6
        m = LogisticRegression(max_iter=4000, C=1.0)
        m.fit((Xf[tr] - mu) / sd, Yf[tr])
        oof[va] = m.decision_function((Xf[va] - mu) / sd)
        print(f'  fold {f}', flush=True)
    sc = torch.tensor(oof).reshape(nq, k)
    stack_acc = 100 * Y[torch.arange(nq), sc.argmax(1)].mean().item()
    print(f'\ndeployed seen rank-1 (no genus)     {base_rec[1]:6.3f}')
    print(f'genus-fusion rank-1 (no rerank)     {stack_base:6.3f}')
    print(f'GENUS-FUSION + LEARNED RE-RANKER    {stack_acc:6.3f}')
    print(f'v83 (deployed, current best) real bar to beat: 91.674 -> need >=92.674', flush=True)
    clears = stack_acc >= 92.674
    print(f'VERDICT: {"CLEARS +1.0pt bar" if clears else "DOES NOT CLEAR +1.0pt bar"}', flush=True)

    result.update({'stack_base': stack_base, 'stack_acc': stack_acc, 'clears_bar': clears})
    json.dump(result, open(f'{OUT}/genus_recall10_v103.json', 'w'), indent=2)
    print(f'wrote {OUT}/genus_recall10_v103.json', flush=True)


if __name__ == '__main__':
    main()
