"""Adversarial b-lift re-test on the v36 SHIFT base, class-disjoint CV.

Prior sweeps (fusion_sweep_final) only re-weighted the 6 legs already in the stack.
unseen_legs tested h/h_tta/fullft336 additions on the OLD deployed (ctftbig) base.
This adds every cheap cached leg onto the v36 shift base (ctftshift stack) and greedily
grows it under a CLASS-DISJOINT tune/val split of the pseudo-unseen classes -> any gain
must generalize to unseen classes, not just unseen images. Reports held-out val deltas.

No TaxaBind train emb on box -> base is v36-minus-TB (proxy 31.45). A leg that helps here
and is orthogonal to TB is the only thing worth extracting on the real eval.
"""
import json
import os
import sys
import time
import random

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import FishData, dbnorm, load_emb, OUT

t0 = time.time()
random.seed(0)
D = FishData()
cand = D.cand
print(f'[{time.time()-t0:.0f}s] cand={len(cand)} pseudo={len(D.pseudo)}', flush=True)

# class-disjoint split of pseudo classes -> tune / val
pseudo = list(D.pseudo)
random.shuffle(pseudo)
half = len(pseudo) // 2
tune_cls, val_cls = set(pseudo[:half]), set(pseudo[half:])


def queries(idx, feats, clsset):
    q, y = [], []
    for c in clsset:
        for fn in D.by[c]:
            if fn in idx:
                q.append(feats[idx[fn]])
                y.append(D.cand_pos[D.ci[c]])
    return torch.stack(q), torch.tensor(y)


TtH_c = D.TtH[cand]
TnL_c = D.TnL[cand]
TTX = F.normalize(torch.load(os.path.join(OUT, 'text_emb_h_promptens.pt'), weights_only=False)['emb_taxctx'].float(), dim=-1)
TTX_c = TTX[cand]
TPE = F.normalize(torch.load(os.path.join(OUT, 'text_emb_h_promptens.pt'), weights_only=False)['emb_promptens'].float(), dim=-1)
TPE_c = TPE[cand]

# load image legs (all align to 1024 H text space)
leg_tags = ['ctftshift', 'ftshift', 'fullft336_v2', 'fullft336shift', 'h', 'h_tta', 'cap', 'ctftbig']
feats_by = {}
for tag in leg_tags:
    p = os.path.join(OUT, f'emb_train_{tag}.pt')
    if not os.path.exists(p):
        print(f'  MISSING {tag}', flush=True)
        continue
    idx, feats, _ = load_emb(p)
    feats_by[tag] = (idx, feats)
idxL, featsL, _ = load_emb(os.path.join(OUT, 'emb_train.pt'))


def leg_scores(clsset):
    """Return dict leg-name -> [Nq, Ncand] score matrix, plus gold."""
    out, gold = {}, None
    for tag, (idx, feats) in feats_by.items():
        q, g = queries(idx, feats, clsset)
        gold = g
        out[f'{tag}@TtH'] = dbnorm(q @ TtH_c.t())
        if tag in ('ctftshift', 'ctftbig'):
            out[f'{tag}@TTX'] = dbnorm(q @ TTX_c.t())
            out[f'{tag}@TPE'] = dbnorm(q @ TPE_c.t())
    qL, gL = queries(idxL, featsL, clsset)
    out['L@TnL'] = dbnorm(qL @ TnL_c.t())
    return out, gold


St, gt = leg_scores(tune_cls)
Sv, gv = leg_scores(val_cls)
print(f'[{time.time()-t0:.0f}s] tune n={len(gt)} val n={len(gv)}', flush=True)


def acc(S, gold):
    return (S.argmax(1) == gold).float().mean().item() * 100


# v36 base (minus TB): ctftshift@TtH + 0.5 L + 0.75 fullft336_v2@TtH + 1.0 ftshift@TtH + 1.0 ctftshift@TTX
BASE = {'ctftshift@TtH': 1.0, 'L@TnL': 0.5, 'fullft336_v2@TtH': 0.75,
        'ftshift@TtH': 1.0, 'ctftshift@TTX': 1.0}


def build(S, weights):
    out = None
    for k, w in weights.items():
        if k not in S or w == 0:
            continue
        out = S[k] * w if out is None else out + S[k] * w
    return out


base_tune = acc(build(St, BASE), gt)
base_val = acc(build(Sv, BASE), gv)
print(f'BASE (v36-minus-TB): tune {base_tune:.2f}  val {base_val:.2f}', flush=True)

# greedy add: candidates NOT in base
candidates = [k for k in St.keys() if k not in BASE]
results = {'base': BASE, 'base_tune': base_tune, 'base_val': base_val, 'steps': []}
cur = dict(BASE)
for _ in range(6):
    best = None
    for k in candidates:
        if k in cur:
            continue
        for w in [0.25, 0.5, 1.0]:
            trial = dict(cur)
            trial[k] = w
            at = acc(build(St, trial), gt)
            if best is None or at > best[0]:
                best = (at, k, w)
    at, k, w = best
    trial = dict(cur)
    trial[k] = w
    av = acc(build(Sv, trial), gv)
    cur_tune = acc(build(St, cur), gt)
    cur_val = acc(build(Sv, cur), gv)
    step = {'add': k, 'w': w, 'tune_after': round(at, 3), 'val_after': round(av, 3),
            'val_delta_vs_prev': round(av - cur_val, 3)}
    results['steps'].append(step)
    print(f'  + {k} w={w}: tune {cur_tune:.2f}->{at:.2f}  val {cur_val:.2f}->{av:.2f} '
          f'(val d {av-cur_val:+.3f})', flush=True)
    cur[k] = w

results['final_weights'] = cur
results['final_val'] = acc(build(Sv, cur), gv)
results['val_gain_over_base'] = round(results['final_val'] - base_val, 3)
json.dump(results, open(os.path.join(OUT, 'b_lift_cv_results.json'), 'w'), indent=1)
print(f'\nFINAL val {results["final_val"]:.2f} vs base {base_val:.2f} '
      f'(gain {results["val_gain_over_base"]:+.3f})', flush=True)
print('wrote outputs/b_lift_cv_results.json', flush=True)
