"""v99: post-hoc class-frequency logit adjustment (Menon-style long-tail bias correction)
on the SEEN closed-set fusion.

v97 bucketed accuracy by n_train and found a steep gradient (0-2 img: 59.91%, 3-5:
74.39%, 6+: 93.29%) -- still unexploited. This is distinct from prototype shrinkage
(which moves the prototype vector itself): here the fused score is untouched and we
only add a per-class ADDITIVE bias after scoring, sized by the class's train count:
    fused'_c = fused_c + tau * log(n_train_c)
fit purely from train-fold class counts (no val/gold peeking). tau > 0 favors
well-populated classes (compounds the existing skew -- included as a sanity check it
should hurt); tau < 0 boosts starved classes' logits to fight exactly the confusion
v97 exposed (common prototypes out-competing correct rare ones). Also tries the
population-normalized prior log(n_train_c / sum n_train) which differs from
log(n_train_c) only by a constant -- same argmax, included once to confirm that
algebraic identity rather than assume it.

Base fusion = deployed 1.0/2.5/2.5 weighted zc(proto + 2*cmax + 4*taxon) sum, i.e.
the same base v98/v95 reuse. Pre-registered before any number was seen: sweep
tau in {-3,-2,-1.5,-1,-0.5,-0.25,0,0.25,0.5,1,1.5,2,3}, kill bar val closed-set
accuracy lift >= +1.0pt over tau=0.

  conda activate onet && python research/logit_adjust_v99.py
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
from seen_inat_v95 import member_scores

BASE_W = {'ctftshift': 1.0, 'ftshift': 2.5, 'fullft336shift': 2.5}
TAU_GRID = [-3, -2, -1.5, -1, -0.5, -0.25, 0, 0.25, 0.5, 1, 1.5, 2, 3]
KILL_LIFT = 1.0


def main():
    torch.set_num_threads(8)
    lab = json.load(open(f'{D}/label_train.json'))
    classes = list(pickle.load(open(f'{D}/all_classes.pkl', 'rb')))
    train, tb_train, b2f_train, b2l_train = load_train_embs()
    sp = holdout_split(train, tb_train, b2f_train, b2l_train)
    trby, val_seen = sp['trby'], [f for f, yv in zip(sp['val_files'], sp['y']) if yv == 1]
    seen = sorted(sp['kept'])
    s2i = {c: i for i, c in enumerate(seen)}
    S = len(seen)
    ci = {c: i for i, c in enumerate(classes)}
    kept_idx = torch.tensor([ci[c] for c in seen]).to(dev)
    yv = torch.tensor([s2i[lab[f]] for f in val_seen])
    n_train = torch.tensor([len(trby[c]) for c in seen], dtype=torch.float32)
    log_n = torch.log(n_train.clamp(min=1)).to(dev)
    print(f'val_seen rows {len(val_seen)} | seen classes {S}', flush=True)
    print(f'n_train range [{int(n_train.min())},{int(n_train.max())}] median {n_train.median():.0f}', flush=True)

    TtH = F.normalize(torch.load(f'{OUT}/text_emb_h_taxon.pt', weights_only=False)['emb_taxon'].float(), dim=-1).to(dev)
    Tseen = TtH[kept_idx]

    Zbase = {}
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
        del TF, TL, qF, base
        torch.cuda.empty_cache()

    fused = sum(BASE_W[t] * Zbase[t] for t in BASE_W)  # [N, S]
    base_acc = 100 * (fused.argmax(1).cpu() == yv).float().mean().item()
    print(f'\nbase (tau=0) val closed-set acc: {base_acc:.2f}', flush=True)

    # sanity: log(n_train) vs log(n_train/sum) differ by a constant -> identical argmax
    log_n_prior = log_n - torch.log(n_train.sum().to(dev))
    prior_acc = 100 * ((fused - 1.0 * log_n_prior).argmax(1).cpu() == yv).float().mean().item()
    raw_acc = 100 * ((fused - 1.0 * log_n).argmax(1).cpu() == yv).float().mean().item()
    print(f'sanity: normalized-prior bias == raw-count bias under argmax? '
          f'{prior_acc:.4f} vs {raw_acc:.4f} (should match)', flush=True)

    print('\n=== tau sweep: fused + tau*log(n_train), val closed-set accuracy ===', flush=True)
    rows = {}
    for tau in TAU_GRID:
        adj = fused + tau * log_n
        acc = 100 * (adj.argmax(1).cpu() == yv).float().mean().item()
        rows[tau] = acc
        print(f'  tau={tau:+.2f} -> {acc:6.2f}', flush=True)

    best_tau = max(rows, key=rows.get)
    lift = rows[best_tau] - base_acc

    # bucketed breakdown at best tau vs tau=0
    BUCKETS = [(0, 2), (3, 5), (6, 1_000_000)]
    n_train_gold = n_train[yv]
    print(f'\n=== bucket breakdown: tau=0 vs best tau={best_tau:+.2f} ===', flush=True)
    bucket_lines = []
    for lo, hi in BUCKETS:
        mask = (n_train_gold >= lo) & (n_train_gold <= hi)
        n_rows = int(mask.sum())
        yv_b = yv[mask]
        a0 = 100 * (fused[mask].argmax(1).cpu() == yv_b).float().mean().item()
        ab = 100 * ((fused + best_tau * log_n)[mask].argmax(1).cpu() == yv_b).float().mean().item()
        line = f'  n_train[{lo},{hi}] ({n_rows} rows): tau=0 {a0:.2f} -> tau={best_tau:+.2f} {ab:.2f}'
        print(line, flush=True)
        bucket_lines.append(line)

    verdict = (f'CLEARS — tau={best_tau:+.2f}' if lift >= KILL_LIFT
               else 'DEAD — no tau lifts holdout closed-set acc >= +1.0')
    print(f'\nbase {base_acc:.2f}  best tau={best_tau:+.2f} {rows[best_tau]:.2f}  lift {lift:+.2f}pt '
          f'(kill >= +{KILL_LIFT})', flush=True)
    print(f'VERDICT: {verdict}', flush=True)
    json.dump({'rows': rows, 'base': base_acc, 'best_tau': best_tau, 'best_acc': rows[best_tau],
               'lift': lift, 'verdict': verdict, 'buckets': bucket_lines},
              open(f'{OUT}/logit_adjust_v99.json', 'w'), indent=2)
    print(f'wrote {OUT}/logit_adjust_v99.json', flush=True)


def _selftest():
    """No GPU/data dependency: log(n) additive bias and log(n/sum(n)) differ only
    by a constant shift -> identical argmax under any fixed score vector. Guards
    the algebraic identity the in-script sanity check above also checks live."""
    import torch as T
    n = T.tensor([1., 5., 20.])
    a = T.log(n)
    b = T.log(n / n.sum())
    assert T.allclose(a - a.mean(), b - b.mean(), atol=1e-5)
    scores = T.tensor([0.9, 0.5, 0.1])
    assert (scores + 1.0 * a).argmax() == (scores + 1.0 * b).argmax()
    print('selftest ok', flush=True)


if __name__ == '__main__':
    if '--selftest' in sys.argv:
        _selftest()
    else:
        main()
