"""v100: genus-level hierarchical backoff for the SEEN closed-set classifier.

NEW MECHANISM (not a reweight/re-rank of the existing 3-leg fusion): every class
name in all_classes.pkl is a Linnaean binomial "Genus species" -- taxonomic
structure that is already on disk and unused by any score in the pipeline except
as a raw STRING fed into text embeddings. This builds a coarser, genus-level
prototype+cmax by POOLING TRAINING IMAGES ACROSS ALL SIBLING SPECIES OF THE SAME
GENUS (same proto+2*cmax formula the deployed species-level classifier already
uses, just applied one taxonomic rank up), then adds it as a 4th zc()-normalized
fusion term. It does not touch the existing species-level score at all.

Why this targets the diagnosed failure mode specifically (research/seen_inat_v97_bucket.py):
  n_train in [0,2]  (5.7% of val rows): 59.91% acc  <- prototype from 1-2 noisy images
  n_train in [3,5]  (8.0% of val rows): 74.39% acc
  n_train in [6,+]  (86.3% of val rows): 93.29% acc
A genus with k sibling species pools k times as many supporting images into ONE
coarse prototype, so a starved species (1-2 images) inherits a well-supported
"is this even the right neighborhood" signal from its genus-mates -- something no
existing leg computes today (proto/cmax are per-species-only; the taxon-text leg
reads the binomial NAME as text, it never pools sibling IMAGES). Architecturally
distinct from prototype shrinkage (no embedding-space blending/interpolation of
the species prototype -- the species-level score is untouched, this is a
SEPARATE additive term) and from soft k-NN (pooling happens ACROSS classes by
taxon, not within one class by distance). Additive, same spirit as v77's gate
features: new information, not a reweighting of old information.

Holdout methodology: verbatim research/learned_gate_v77.py::holdout_split()
(rarest-20%-classes pseudo-novel split, val_seen = the 20%-of-images-per-kept-
class val rows), matching research/seen_reweight_v98.py's harness exactly so
numbers are directly comparable to the repo's existing 89.87% (deployed 3-leg) /
91.20% (ctft solo) / 91.674% (seen re-ranker, current best) figures.

Pre-registered before any number was seen:
  base       = {deployed 3-leg fusion (1.0/2.5/2.5), ctftshift solo}
  LAM_G grid = [0, 0.5, 1, 1.5, 2, 3, 4, 6]
  genus bonus computed once, from ctftshift training features only (the
  strongest solo leg) -- kept minimal for a first test, not per-leg.
  PASS bar: best (base, LAM_G) combo reaches >= 92.674% val_seen argmax
  accuracy (+1.0pt over the 91.674% seen-re-ranker holdout number, the current
  best-in-repo on this exact split/metric).
  Also reports the v97-style n_train bucket breakdown to confirm the lift (if
  any) is concentrated in the [0,2]/[3,5] starved buckets and is not just
  redistributing the already-saturated [6,+] bucket.

  conda activate onet && python research/seen_genus_backoff_v100.py
"""
from __future__ import annotations

import json
import pickle
import sys
from collections import defaultdict

import torch
import torch.nn.functional as F

sys.path.insert(0, 'research')
from common import OUT
from learned_gate_v77 import D, MEMBERS, dev, holdout_split, load_train_embs, zc
from seen_inat_v95 import BASE_W, member_scores

LAM_G_GRID = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
KILL_LIFT = 1.0
RERANKER_BEST = 91.674
PASS_BAR = RERANKER_BEST + KILL_LIFT
GENUS_LEG = 'ctftshift'          # strongest solo leg (91.20% holdout) supplies the pooled genus evidence
BUCKETS = [(0, 2), (3, 5), (6, 1_000_000)]


def main():
    torch.set_num_threads(8)
    lab = json.load(open(f'{D}/label_train.json'))
    classes = list(pickle.load(open(f'{D}/all_classes.pkl', 'rb')))
    ci = {c: i for i, c in enumerate(classes)}

    train, tb_train, b2f_train, b2l_train = load_train_embs()
    sp = holdout_split(train, tb_train, b2f_train, b2l_train)
    trby, val_seen = sp['trby'], [f for f, yv in zip(sp['val_files'], sp['y']) if yv == 1]
    seen = sorted(sp['kept'])
    s2i = {c: i for i, c in enumerate(seen)}
    S = len(seen)
    kept_idx = torch.tensor([ci[c] for c in seen]).to(dev)
    yv = torch.tensor([s2i[lab[f]] for f in val_seen])
    n_train = torch.tensor([len(trby[c]) for c in seen], dtype=torch.float32)
    print(f'val_seen rows {len(val_seen)} | seen classes {S}', flush=True)

    # --- genus grouping: free metadata already on disk, "Genus species" strings ---
    genus_of = {c: c.split()[0] for c in seen}
    genus_list = sorted(set(genus_of.values()))
    g2i = {g: i for i, g in enumerate(genus_list)}
    G = len(genus_list)
    species_genus_idx = torch.tensor([g2i[genus_of[c]] for c in seen]).to(dev)
    sizes = defaultdict(int)
    for c in seen:
        sizes[genus_of[c]] += 1
    multi = sum(1 for g in genus_list if sizes[g] > 1)
    print(f'genera among {S} kept species: {G} total, {multi} have >=2 sibling species '
          f'({100 * multi / G:.1f}%) | covers '
          f'{sum(v for g, v in sizes.items() if v > 1)}/{S} species with a genus-mate', flush=True)

    TtH = F.normalize(torch.load(f'{OUT}/text_emb_h_taxon.pt', weights_only=False)['emb_taxon'].float(), dim=-1).to(dev)
    Tseen = TtH[kept_idx]

    # --- deployed per-leg species-level base scores (identical to v95/v98) ---
    Zbase, TF_by, TL_by, qF_by = {}, {}, {}, {}
    for t, _, _ in MEMBERS:
        if t not in BASE_W:
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
        base = member_scores(qF, P, TF, TL, Tseen, S)
        Zbase[t] = zc(base)
        acc = 100 * (base.argmax(1).cpu() == yv).float().mean().item()
        print(f'  {t:16s} solo val acc {acc:6.2f}', flush=True)
        TF_by[t], TL_by[t], qF_by[t] = TF, TL, qF

    def fused_acc(w, extra=None):
        f = sum(w[t] * Zbase[t] for t in BASE_W)
        if extra is not None:
            f = f + extra
        return 100 * (f.argmax(1).cpu() == yv).float().mean().item()

    DEPLOYED = {'ctftshift': 1.0, 'ftshift': 2.5, 'fullft336shift': 2.5}
    SOLO = {'ctftshift': 1.0, 'ftshift': 0.0, 'fullft336shift': 0.0}
    base_deployed = fused_acc(DEPLOYED)
    base_solo = fused_acc(SOLO)
    print(f'\nsanity: deployed 3-leg {base_deployed:.2f}  ctftshift solo {base_solo:.2f}', flush=True)

    # --- genus-level prototype + cmax, pooled ACROSS sibling species, one leg ---
    TL_genus = species_genus_idx[TL_by[GENUS_LEG]]           # per-training-image genus id
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
    genus_raw_G = q @ Pg.t() + 2.0 * cmaxg                    # (N, G)
    genus_bonus = genus_raw_G[:, species_genus_idx]           # (N, S) broadcast to each species' genus

    Zg = zc(genus_bonus)

    print('\n=== LAM_G grid, genus bonus added to deployed 3-leg fusion ===', flush=True)
    rows_dep = {lam: fused_acc(DEPLOYED, lam * Zg) for lam in LAM_G_GRID}
    for lam, acc in rows_dep.items():
        print(f'  LAM_G={lam:<5} -> {acc:6.2f}', flush=True)

    print('\n=== LAM_G grid, genus bonus added to ctftshift solo ===', flush=True)
    rows_solo = {lam: fused_acc(SOLO, lam * Zg) for lam in LAM_G_GRID}
    for lam, acc in rows_solo.items():
        print(f'  LAM_G={lam:<5} -> {acc:6.2f}', flush=True)

    best_dep = max(rows_dep, key=rows_dep.get)
    best_solo = max(rows_solo, key=rows_solo.get)
    best_overall = max(rows_dep[best_dep], rows_solo[best_solo])
    verdict = ('CLEARS' if best_overall >= PASS_BAR else 'FAILS') + \
        f' -- best {best_overall:.3f} vs pass bar {PASS_BAR:.3f} (91.674 + 1.0)'
    print(f'\nbest deployed+genus: LAM_G={best_dep} -> {rows_dep[best_dep]:.3f} '
          f'(lift {rows_dep[best_dep] - base_deployed:+.3f} over deployed)', flush=True)
    print(f'best solo+genus:     LAM_G={best_solo} -> {rows_solo[best_solo]:.3f} '
          f'(lift {rows_solo[best_solo] - base_solo:+.3f} over ctftshift solo)', flush=True)
    print(f'VERDICT: {verdict}', flush=True)

    # --- bucket breakdown: does the lift concentrate in the starved tail? ---
    winning_w, winning_lam = (DEPLOYED, best_dep) if rows_dep[best_dep] >= rows_solo[best_solo] else (SOLO, best_solo)
    n_train_gold = n_train[yv]

    def bucket_report(tag, w, lam):
        fbase = sum(w[t] * Zbase[t] for t in BASE_W)
        fwith = fbase + lam * Zg
        print(f'\n=== n_train-bucket breakdown, {tag} (LAM_G={lam}) ===', flush=True)
        for lo, hi in BUCKETS:
            mask = (n_train_gold >= lo) & (n_train_gold <= hi)
            yb = yv[mask]
            a0 = 100 * (fbase[mask].argmax(1).cpu() == yb).float().mean().item()
            a1 = 100 * (fwith[mask].argmax(1).cpu() == yb).float().mean().item()
            print(f'  n_train in [{lo},{hi}]: {int(mask.sum()):5d} rows | base {a0:6.2f} -> '
                  f'+genus {a1:6.2f}  ({a1 - a0:+.2f})', flush=True)

    bucket_report('winning config', winning_w, winning_lam)
    # also report at the best NON-TRIVIAL deployed+genus point, since the argmax-overall
    # winner above is the trivial LAM_G=0 (genus bonus never helps ctftshift solo)
    best_dep_nonzero = max((l for l in LAM_G_GRID if l > 0), key=lambda l: rows_dep[l])
    bucket_report('deployed+genus peak', DEPLOYED, best_dep_nonzero)

    json.dump({'rows_deployed': rows_dep, 'rows_solo': rows_solo, 'base_deployed': base_deployed,
               'base_solo': base_solo, 'pass_bar': PASS_BAR, 'verdict': verdict,
               'n_genera': G, 'genera_with_siblings': multi},
              open(f'{OUT}/seen_genus_backoff_v100.json', 'w'), indent=2)
    print(f'\nwrote {OUT}/seen_genus_backoff_v100.json', flush=True)


if __name__ == '__main__':
    main()
