"""Persist the genus-augmented seen re-ranker (genus_recall10_v103.py found stack_acc=91.775
via 5-fold OOF but never saved a deployable model). Same recipe, LAM_G=3.0 (the winning grid
point), final fit on the FULL holdout-split feature matrix (mirrors rerank_seen_v83.py's own
fold-then-full-refit pattern) so builders/ can load a real pickle.

  conda activate onet && python research/genus_rerank_v103_train.py
"""
from __future__ import annotations

import json
import pickle
import sys

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, 'research')
from common import OUT
from learned_gate_v77 import D, LAM, MEMBERS, dev, holdout_split, load_train_embs, zc

LAM_G = 3.0
GENUS_LEG = 'ctftshift'
DEPLOYED = {'ctftshift': 1.0, 'ftshift': 2.5, 'fullft336shift': 2.5}
K = 10
ACTIVE = [t for t, _, w in MEMBERS if w > 0]
SIGNALS = [f'{t}_{c}' for t in ACTIVE for c in ('proto', 'cmax', 'taxon')] + ['genus', 'block']


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

    genus_of = {c: c.split()[0] for c in seen}
    genus_list = sorted(set(genus_of.values()))
    g2i = {g: i for i, g in enumerate(genus_list)}
    G = len(genus_list)
    species_genus_idx = torch.tensor([g2i[genus_of[c]] for c in seen]).to(dev)

    TtH = F.normalize(torch.load(f'{OUT}/text_emb_h_taxon.pt', weights_only=False)['emb_taxon'].float(), dim=-1).to(dev)
    Tseen = TtH[kept_idx]

    sig, TF_by, TL_by, qF_by = {}, {}, {}, {}
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
        block += DEPLOYED[t] * zc(base_t)
        TF_by[t], TL_by[t], qF_by[t] = TF, TL, qF
        print(f'  {t} done', flush=True)

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
    sig['genus'] = genus_bonus
    fused = block + LAM_G * zc(genus_bonus)
    sig['block'] = fused

    topk = fused.topk(K, dim=1).indices
    n_gold_in_k = 100 * (topk == yv.to(dev).unsqueeze(1)).any(1).float().mean().item()
    print(f'gold in top-{K}: {n_gold_in_k:.2f}%', flush=True)

    X, names = cand_features(sig, topk, n_train, fused)
    Y = (topk == yv.to(dev)[:, None]).float()
    X, Y = X.cpu(), Y.cpu()

    nq, k, nf = X.shape
    print(f'queries={nq} K={k} feats={nf}', flush=True)
    Xf = X.reshape(nq * k, nf).numpy()
    Yf = Y.reshape(nq * k).numpy()

    mu, sd = Xf.mean(0), Xf.std(0) + 1e-6
    m = LogisticRegression(max_iter=4000, C=1.0)
    m.fit((Xf - mu) / sd, Yf)
    acc_infit = 100 * Y[torch.arange(nq), torch.tensor(
        m.decision_function((Xf - mu) / sd)).reshape(nq, k).argmax(1)].mean().item()
    print(f'in-fit sanity acc (expect close to 91.775 OOF from v103): {acc_infit:.3f}', flush=True)

    pickle.dump({'model': m, 'mu': mu, 'sd': sd, 'names': names, 'K': K,
                 'signals': SIGNALS, 'lam_g': LAM_G, 'genus_leg': GENUS_LEG},
                open(f'{OUT}/rerank_seen_genus_v103.pkl', 'wb'))
    print(f'wrote {OUT}/rerank_seen_genus_v103.pkl', flush=True)


if __name__ == '__main__':
    main()
