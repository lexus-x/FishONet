"""v99: cross-leg training-set denoising for the SEEN prototype route.

Idea (never tried): a training image's label may be wrong, or the photo may be an
outlier crop, if ALL THREE independently-fine-tuned legs (ctftshift/ftshift/
fullft336shift) agree, under leave-one-out, that the image looks more like ANOTHER
class than its own assigned class. Flag + drop such images before building the seen
prototypes (mean term) and the per-exemplar max-pool term (cmax), then re-score
val_seen closed-set accuracy at the deployed 1.0/2.5/2.5 fusion (LAM=4 taxon term
unchanged -- identical scoring to v95/v98/v83's member_scores).

Per training image fn of class c (only classes with >= min_n training images --
leave-one-out needs real neighbors left to be meaningful):
  loo_sim_t   = f_i . normalize(sum_{j in c, j!=i} f_j)     (own class, leave-one-out)
  other_sim_t = max_{c' != c} f_i . P_full[c']               (best other class, full proto)
  margin_t    = loo_sim_t - other_sim_t
  votes       = #{t in 3 legs : margin_t < 0}   (legs where the image looks more like
                                                  some other class than its own)
Pre-registered grid over (min_n, min_votes) -- flagged images are HARD-DROPPED from
both the mean-prototype sum and the TF/TL exemplar pool, for ALL 3 members:
  (3,2) (3,3) (5,2) (5,3)
Kill bar: val closed-set accuracy lift >= +1.0pt over the no-denoise baseline (same
bar as v93/v95/v98).

  conda activate onet && python research/denoise_train_v99.py
"""
from __future__ import annotations

import json
import pickle
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, 'research')
from common import OUT
from learned_gate_v77 import D, MEMBERS, dev, holdout_split, load_train_embs, zc
from seen_inat_v95 import BASE_W, member_scores

GRID = [(3, 2), (3, 3), (5, 2), (5, 3)]
KILL_LIFT = 1.0


def build_pieces(feats, idx, seen, s2i, trby, drop=None):
    """P (mean prototype) + TF/TL (per-exemplar pool), optionally excluding `drop` files."""
    S = len(seen)
    P = torch.zeros(S, feats.shape[1])
    cnt = torch.zeros(S)
    TFl, TLl = [], []
    for c in seen:
        for fn in trby[c]:
            if fn not in idx or (drop is not None and fn in drop):
                continue
            f = feats[idx[fn]]
            P[s2i[c]] += f
            cnt[s2i[c]] += 1
            TFl.append(f)
            TLl.append(s2i[c])
    P = F.normalize(P / cnt.clamp(min=1).unsqueeze(1), dim=-1).to(dev)
    TF = torch.stack(TFl).to(dev)
    TL = torch.tensor(TLl).to(dev)
    return P, TF, TL


def flag_outliers(feats, idx, seen, s2i, trby, min_n):
    """Leave-one-out margin per training image (own-class LOO sim minus best-other-
    class full-proto sim). Only for classes with >= min_n training images."""
    S = len(seen)
    Ddim = feats.shape[1]
    raw_sum = torch.zeros(S, Ddim)
    cnt = torch.zeros(S)
    members = []
    for c in seen:
        for fn in trby[c]:
            if fn not in idx:
                continue
            f = feats[idx[fn]]
            raw_sum[s2i[c]] += f
            cnt[s2i[c]] += 1
            members.append((fn, s2i[c], f))
    P_full = F.normalize(raw_sum, dim=-1).to(dev)

    eligible = [(fn, ci_, f) for fn, ci_, f in members if cnt[ci_] >= min_n]
    margins = {}
    if not eligible:
        return margins
    fns = [e[0] for e in eligible]
    cidx = torch.tensor([e[1] for e in eligible]).to(dev)
    F_i = torch.stack([e[2] for e in eligible]).to(dev)
    own_raw = raw_sum[[e[1] for e in eligible]].to(dev) - F_i
    loo_proto = F.normalize(own_raw, dim=-1)
    sim_own = (F_i * loo_proto).sum(1)

    sim_other = torch.empty(len(fns), device=dev)
    for i in range(0, len(fns), 4000):
        e = F_i[i:i + 4000]
        sim = e @ P_full.t()
        sim[torch.arange(sim.shape[0], device=dev), cidx[i:i + 4000]] = -1e9
        sim_other[i:i + 4000] = sim.max(1).values

    margin = (sim_own - sim_other).cpu()
    for fn, m in zip(fns, margin.tolist()):
        margins[fn] = m
    return margins


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
    print(f'val_seen rows {len(val_seen)} | seen classes {S}', flush=True)

    TtH = F.normalize(torch.load(f'{OUT}/text_emb_h_taxon.pt', weights_only=False)['emb_taxon'].float(), dim=-1).to(dev)
    Tseen = TtH[kept_idx]

    feats_idx_by_t, qF_by_t = {}, {}
    for t, _, _ in MEMBERS:
        if t not in BASE_W:
            continue
        idx, feats, _ = train[t]
        feats_idx_by_t[t] = (feats, idx)
        qF_by_t[t] = torch.stack([feats[idx[fn]] for fn in val_seen]).to(dev)

    def fused_acc(Z):
        f = sum(BASE_W[t] * Z[t] for t in BASE_W)
        return 100 * (f.argmax(1).cpu() == yv).float().mean().item()

    Zbase = {}
    for t in BASE_W:
        feats, idx = feats_idx_by_t[t]
        P, TF, TL = build_pieces(feats, idx, seen, s2i, trby)
        sc = member_scores(qF_by_t[t], P, TF, TL, Tseen, S)
        Zbase[t] = zc(sc)
        acc = 100 * (sc.argmax(1).cpu() == yv).float().mean().item()
        print(f'  {t:16s} baseline (no denoise) val acc {acc:6.2f}', flush=True)
    base_acc = fused_acc(Zbase)
    print(f'\nbaseline (no denoise) fused val acc: {base_acc:.2f}', flush=True)

    print('\n=== pre-registered denoise grid (val closed-set accuracy) ===', flush=True)
    margin_cache = {}
    rows = {}
    for min_n, min_votes in GRID:
        if min_n not in margin_cache:
            margin_cache[min_n] = {t: flag_outliers(*feats_idx_by_t[t], seen, s2i, trby, min_n=min_n)
                                    for t in BASE_W}
        margins_this_n = margin_cache[min_n]
        common_fns = set.intersection(*[set(m.keys()) for m in margins_this_n.values()])
        drop = {fn for fn in common_fns
                if sum(1 for t in BASE_W if margins_this_n[t][fn] < 0) >= min_votes}

        Zd = {}
        for t in BASE_W:
            feats, idx = feats_idx_by_t[t]
            P, TF, TL = build_pieces(feats, idx, seen, s2i, trby, drop=drop)
            sc = member_scores(qF_by_t[t], P, TF, TL, Tseen, S)
            Zd[t] = zc(sc)
        acc = fused_acc(Zd)
        rows[f'min_n{min_n}_votes{min_votes}'] = {'acc': acc, 'n_dropped': len(drop),
                                                    'n_eligible': len(common_fns)}
        print(f'  min_n={min_n} votes>={min_votes}: dropped {len(drop):5d}/{len(common_fns):5d} '
              f'eligible -> val acc {acc:.2f}', flush=True)

    best_key = max(rows, key=lambda k: rows[k]['acc'])
    lift = rows[best_key]['acc'] - base_acc
    verdict = (f'CLEARS — {best_key}' if lift >= KILL_LIFT
               else 'DEAD — cross-leg denoising does not lift holdout closed-set acc >= +1.0')
    print(f'\nbase {base_acc:.2f}  best {best_key} {rows[best_key]["acc"]:.2f}  lift {lift:+.2f}pt '
          f'(kill >= +{KILL_LIFT})', flush=True)
    print(f'VERDICT: {verdict}', flush=True)
    json.dump({'rows': rows, 'base': base_acc, 'best': best_key, 'lift': lift, 'verdict': verdict},
              open(f'{OUT}/denoise_train_v99.json', 'w'), indent=2)
    print(f'wrote {OUT}/denoise_train_v99.json', flush=True)


if __name__ == '__main__':
    main()
