"""v92: genus-pooled iNat photo prototypes on the unseen route — soft taxonomic
narrowing via congeneric bank photos.

Adjacent dead results this is NOT: hard family 2-stage (−20pt, §4), ToL/B2 "genus
backoff" (≤0, v43 era), genus consensus in the re-ranker (−0.432, 2026-08-31).
Unmeasured: pooling the H-leg iNat photo bank (`inat_photo_bank_ctftshift.pt`,
7,364 classes / 101k embedded photos) at GENUS level, so unseen candidates with no
own photos borrow congeners' visual evidence. Two variants, fixed a priori:

  A backoff : candidates with NO own photos get λ · genus top-M score (λ 0.5 / 1.0)
  B additive: a separate genus leg for all candidates (own-class photos excluded),
              fused at w_g · dbnorm (w_g 1.0 / 2.0)

Proxy: deployed-style unseen route for the 2,318 pseudo-novel holdout queries over
the full 12,757-candidate pool: text_full + 4·dbnorm(bank_ct) + 3·dbnorm(bank_336)
+ 2.5·dbnorm(b2f). (No crop-max views / no b2l here — recipe approximation; the
kill bar is the standard +0.5 proxy-pt, class-disjoint CV for selection honesty.
Anchor for any gain: proto legs transfer at ~0.08–0.12 overall-pt per proxy-pt.)

  conda activate onet && python research/genus_bank_v92.py
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT
from extract_holdout_shift_v90 import load_train_embs_v90
from learned_gate_v77 import dbnorm, holdout_split

D = 'data/dl'
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
TOPM = 4
W_CT, W_336, W_B2F = 4.0, 3.0, 2.5
LAMBDAS = [0.5, 1.0]
WGS = [1.0, 2.0]
KILL = 0.5
NFOLD = 5


def topm_scores(Q, bank, other_list):
    """[Nq, C] top-M-mean cosine per candidate class from its own photos."""
    S = torch.full((Q.shape[0], len(other_list)), -1e4, device=dev)
    for j, g in enumerate(other_list):
        p = bank.get(g)
        if p is None or p.numel() == 0:
            continue
        p = F.normalize(p.float(), dim=-1).to(dev)
        S[:, j] = (Q @ p.t()).topk(min(TOPM, p.shape[0]), dim=1).values.mean(1)
    return S


def main():
    torch.set_num_threads(8)
    classes = list(pickle.load(open(f'{D}/all_classes.pkl', 'rb')))
    ci = {c: i for i, c in enumerate(classes)}
    train, tb_train, b2f_train, _ = load_train_embs_v90()
    sp = holdout_split(train, tb_train, b2f_train, b2f_train)
    n_seen = sp['n_seen']
    pseudo_files = sp['val_files'][n_seen:]
    gold_names = sp['cls'][n_seen:]
    kept = set(sp['kept'])
    other_list = [i for i in range(len(classes)) if classes[i] not in kept]
    pos = {g: j for j, g in enumerate(other_list)}
    gold_col = np.array([pos[ci[n]] for n in gold_names])
    Nq = len(pseudo_files)
    print(f'pseudo queries {Nq} | candidate pool {len(other_list)}', flush=True)

    def q(t):
        idx, feats, _ = train[t]
        return torch.stack([feats[idx[fn]] for fn in pseudo_files]).to(dev)

    Qct, Q336 = q('ctftshift'), q('fullft336shift')
    Qb2 = torch.stack([b2f_train[1][b2f_train[0][fn]] for fn in pseudo_files]).to(dev)
    Qtb = torch.stack([tb_train[1][tb_train[0][fn]] for fn in pseudo_files]).to(dev)

    TtH = F.normalize(torch.load(f'{OUT}/text_emb_h_taxon.pt', weights_only=False)['emb_taxon'].float(), dim=-1).to(dev)
    TnL = F.normalize(torch.load(f'{OUT}/text_emb.pt', weights_only=False)['emb_name'].float(), dim=-1).to(dev)
    TTX = F.normalize(torch.load(f'{OUT}/text_emb_h_promptens.pt', weights_only=False)['emb_taxctx'].float(), dim=-1).to(dev)
    Ttb = F.normalize(torch.load(f'{OUT}/text_emb_taxabind_taxctx.pt', weights_only=False)['emb_taxctx'].float(), dim=-1).to(dev)
    text_full = (dbnorm(Qct @ TtH.t()) + 0.5 * dbnorm(q('L') @ TnL.t())
                 + 0.75 * dbnorm(q('fullft336_v2') @ TtH.t()) + 1.0 * dbnorm(q('ftshift') @ TtH.t())
                 + 1.0 * dbnorm(Qct @ TTX.t()) + 1.0 * dbnorm(Qtb @ Ttb.t()))
    text_o = text_full[:, torch.tensor(other_list, device=dev)]
    del text_full

    bank_ct = torch.load(f'{OUT}/inat_photo_bank_ctftshift.pt', weights_only=False)['bank']
    bank_336 = torch.load(f'{OUT}/inat_photo_bank_fullft336shift.pt', weights_only=False)['bank']
    print('scoring species banks ...', flush=True)
    S_ct = topm_scores(Qct, bank_ct, other_list)
    S_336 = topm_scores(Q336, bank_336, other_list)
    P_b2 = F.normalize(torch.load(f'{OUT}/inat_tol_merged_b2_a05.pt', weights_only=False)['protos'].float(), dim=-1).to(dev)
    S_b2 = Qb2 @ P_b2[torch.tensor(other_list, device=dev)].t()
    has_b2 = P_b2[torch.tensor(other_list, device=dev)].norm(dim=-1) > 0.5
    S_b2 = S_b2.masked_fill(~has_b2.unsqueeze(0), -1e4)

    base = text_o + W_CT * dbnorm(S_ct) + W_336 * dbnorm(S_336) + W_B2F * dbnorm(S_b2)

    # ---- genus structures on the ctft bank ----
    genus_of = {i: classes[i].split()[0] for i in range(len(classes))}
    gphotos = defaultdict(list)   # genus -> [(class_idx, normalized photos)]
    for g, p in bank_ct.items():
        if p is not None and p.numel() > 0:
            gphotos[genus_of[g]].append((g, F.normalize(p.float(), dim=-1)))
    own = {g for g in other_list if bank_ct.get(g) is not None and bank_ct[g].numel() > 0}
    sib = {c for c in other_list
           if any(src != c for src, _ in gphotos.get(genus_of[c], []))}
    print(f'candidates with own photos {len(own)} | with genus-sibling photos {len(sib)} | '
          f'sibling-only (backoff targets) {len(sib - own)}', flush=True)

    print('scoring genus bank ...', flush=True)
    S_gen = torch.full((Nq, len(other_list)), -1e4, device=dev)
    by_genus = defaultdict(list)
    for c in other_list:
        by_genus[genus_of[c]].append(c)
    for gname, members in gphotos.items():
        cands = [c for c in by_genus.get(gname, []) if c in sib]
        if not cands:
            continue
        Pg = torch.cat([p for _, p in members]).to(dev)
        srcs = np.concatenate([[src] * p.shape[0] for src, p in members])
        sim = Qct @ Pg.t()
        for c in cands:
            mask = torch.tensor(srcs != c, device=dev)
            s = sim[:, mask]
            S_gen[:, pos[c]] = s.topk(min(TOPM, s.shape[1]), dim=1).values.mean(1)

    def top1(S):
        return 100.0 * (S.argmax(1).cpu().numpy() == gold_col).mean()

    variants = {'base': base}
    for lam in LAMBDAS:
        S_aug = S_ct.clone()
        fill = torch.tensor([j for j, g in enumerate(other_list) if g not in own and g in sib], device=dev)
        S_aug[:, fill] = lam * S_gen[:, fill]
        variants[f'A_backoff_l{lam}'] = text_o + W_CT * dbnorm(S_aug) + W_336 * dbnorm(S_336) + W_B2F * dbnorm(S_b2)
    for wg in WGS:
        variants[f'B_additive_w{wg}'] = base + wg * dbnorm(S_gen)

    # class-disjoint CV: pick best non-base variant on 4 folds, score held fold
    uniq = sorted(set(gold_names.tolist()))
    fold_of = {c: i % NFOLD for i, c in enumerate(uniq)}
    folds = np.array([fold_of[c] for c in gold_names])
    hits = {k: (v.argmax(1).cpu().numpy() == gold_col) for k, v in variants.items()}
    cv_hit = np.zeros(Nq, dtype=bool)
    for f in range(NFOLD):
        tr, va = folds != f, folds == f
        best = max((k for k in variants if k != 'base'), key=lambda k: hits[k][tr].mean())
        cv_hit[va] = hits[best][va]

    print(f'\n=== pseudo-unseen proxy top-1 (full {len(other_list)}-candidate pool) ===', flush=True)
    res = {k: top1(v) for k, v in variants.items()}
    for k, v in sorted(res.items(), key=lambda t: -t[1]):
        print(f'{k:16s} {v:7.3f}  ({v - res["base"]:+.3f})', flush=True)
    cv = 100.0 * cv_hit.mean()
    delta = cv - res['base']
    print(f'\nCV-selected genus variant: {cv:.3f}  delta vs base {delta:+.3f}  (kill {KILL:+.1f})', flush=True)
    verdict = 'CLEARS — full-fidelity recheck warranted' if delta >= KILL else 'DEAD'
    print(f'VERDICT: {verdict}', flush=True)
    json.dump({'results': res, 'cv': cv, 'delta': delta, 'verdict': verdict,
               'coverage': {'own': len(own), 'sibling': len(sib), 'backoff_targets': len(sib - own)}},
              open(f'{OUT}/genus_bank_v92.json', 'w'), indent=2)
    print(f'wrote {OUT}/genus_bank_v92.json', flush=True)


if __name__ == '__main__':
    main()
