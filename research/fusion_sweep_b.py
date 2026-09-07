"""Greedy coordinate-ascent fusion search for unseen conditional accuracy (b).

Proxy = pseudo-unseen (rarest-20% seen classes) top-1 over cand columns,
matching research/ctftshift_fuse_proxy (dbnorm applied over cand). Starts from
the v36 unseen text stack and searches leg/text-space term weights to beat it.

Legs with train embeddings are candidate terms; TaxaBind included iff
outputs/emb_train_taxabind.pt exists. Reports proxy b for the tuned stack and
the projected real overall using d(overall)=0.4365*u_rec*d(b_cond/b_proxy).
"""
from __future__ import annotations

import json
import os
import time
from collections import defaultdict

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'outputs')
DATA = os.path.join(ROOT, 'data', 'dl')
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
t0 = time.time()


def load(p):
    d = torch.load(p, weights_only=False)
    return {fn: i for i, fn in enumerate(d['files'])}, F.normalize(d['feats'].float(), dim=-1), list(d['files'])


def dbnorm(S, tc=0.05, tr=0.5):
    return F.log_softmax(S / tc, dim=0) + F.log_softmax(S / tr, dim=1)


def main():
    torch.set_num_threads(8)
    import pickle
    classes = list(pickle.load(open(f'{DATA}/all_classes.pkl', 'rb')))
    ci = {c: i for i, c in enumerate(classes)}
    lab = json.load(open(f'{DATA}/label_train.json'))

    TtH = F.normalize(torch.load(f'{OUT}/text_emb_h_taxon.pt', weights_only=False)['emb_taxon'].float(), dim=-1)
    TnL = F.normalize(torch.load(f'{OUT}/text_emb.pt', weights_only=False)['emb_name'].float(), dim=-1)
    TTX = F.normalize(torch.load(f'{OUT}/text_emb_h_promptens.pt', weights_only=False)['emb_taxctx'].float(), dim=-1)
    Ttb = F.normalize(torch.load(f'{OUT}/text_emb_taxabind_taxctx.pt', weights_only=False)['emb_taxctx'].float(), dim=-1)

    # image legs (train embeddings)
    leg_files = {
        'ctftshift': 'emb_train_ctftshift', 'ctftbig': 'emb_train_ctftbig',
        'ftshift': 'emb_train_ftshift', 'fullft336_v2': 'emb_train_fullft336_v2',
        'fullft336shift': 'emb_train_fullft336shift', 'L': 'emb_train',
    }
    has_tb = os.path.exists(f'{OUT}/emb_train_taxabind.pt')
    if has_tb:
        leg_files['taxabind'] = 'emb_train_taxabind'
    legs = {}
    for name, f in leg_files.items():
        p = f'{OUT}/{f}.pt'
        if os.path.exists(p):
            legs[name] = load(p)
    print(f'[{time.time()-t0:.0f}s] legs: {list(legs.keys())}  has_tb={has_tb}', flush=True)

    # common train files + seen classes (match v36 builder membership)
    core = ['ctftshift', 'ftshift', 'fullft336shift', 'L', 'fullft336_v2']
    common = None
    for t in core:
        s = set(legs[t][2])
        common = s if common is None else (common & s)
    common = [fn for fn in common if fn in lab and lab[fn] in ci]
    by = defaultdict(list)
    for fn in common:
        by[lab[fn]].append(fn)
    seen = sorted(by.keys())
    # hard-sim pseudo-unseen = rarest 20%
    order = sorted(seen, key=lambda c: len(by[c]))
    n_pseudo = int(len(seen) * 0.2)
    pseudo = set(order[:n_pseudo])
    kept_set = set(order[n_pseudo:])
    cand = torch.tensor([i for i, c in enumerate(classes) if c not in kept_set])
    cand_pos = {int(x): j for j, x in enumerate(cand.tolist())}
    print(f'[{time.time()-t0:.0f}s] seen={len(seen)} pseudo={len(pseudo)} cand={len(cand)}', flush=True)

    # queries = pseudo-unseen train images (present in all legs used).
    # class-disjoint tune/val split to guard against small-sample overfit.
    pseudo_sorted = sorted(pseudo)
    tune_cls = set(pseudo_sorted[::2])
    qfiles, is_tune = [], []
    for c in pseudo:
        for fn in by[c]:
            qfiles.append((fn, cand_pos[ci[c]]))
            is_tune.append(c in tune_cls)
    for name in legs:
        idxset = legs[name][0]
        keep = [i for i, (fn, g) in enumerate(qfiles) if fn in idxset]
        qfiles = [qfiles[i] for i in keep]
        is_tune = [is_tune[i] for i in keep]
    gold = torch.tensor([g for _, g in qfiles]).to(dev)
    qf = [fn for fn, _ in qfiles]
    tune_mask = torch.tensor(is_tune, device=dev)
    val_mask = ~tune_mask
    print(f'[{time.time()-t0:.0f}s] queries={len(qf)} tune={int(tune_mask.sum())} val={int(val_mask.sum())}', flush=True)

    # candidate terms: (leg, textspace)
    text_by = {'TtH': TtH, 'TTX': TTX, 'TnL': TnL, 'Ttb': Ttb}
    # which text spaces are valid per leg (dim must match)
    def dim_ok(leg, T):
        return legs[leg][1].shape[1] == T.shape[1]

    # precompute dbnorm score matrix over cand for each term
    terms = {}
    for leg in legs:
        idx, feats, _ = legs[leg]
        Q = torch.stack([feats[idx[fn]] for fn in qf]).to(dev)
        for tname, T in text_by.items():
            # taxabind text only pairs with taxabind image; TnL only with L-dim
            if not dim_ok(leg, T):
                continue
            if tname == 'Ttb' and leg != 'taxabind':
                continue
            if leg == 'taxabind' and tname != 'Ttb':
                continue
            Tc = T[cand].to(dev)
            terms[(leg, tname)] = dbnorm(Q @ Tc.t())
    print(f'[{time.time()-t0:.0f}s] terms: {list(terms.keys())}', flush=True)

    def acc(weights, mask=None):
        S = None
        for k, w in weights.items():
            if w == 0 or k not in terms:
                continue
            S = terms[k] * w if S is None else S + terms[k] * w
        pred = S.argmax(1)
        if mask is None:
            return (pred == gold).float().mean().item() * 100
        return (pred[mask] == gold[mask]).float().mean().item() * 100

    # v36 base stack
    base = {
        ('ctftshift', 'TtH'): 1.0, ('L', 'TnL'): 0.5, ('fullft336_v2', 'TtH'): 0.75,
        ('ftshift', 'TtH'): 1.0, ('ctftshift', 'TTX'): 1.0,
    }
    if has_tb:
        base[('taxabind', 'Ttb')] = 1.0
    base = {k: v for k, v in base.items() if k in terms}
    a_base_full = acc(base)
    a_base_tune = acc(base, tune_mask)
    a_base_val = acc(base, val_mask)
    print(f'[{time.time()-t0:.0f}s] v36 base proxy b: full={a_base_full:.3f} '
          f'tune={a_base_tune:.3f} val={a_base_val:.3f}', flush=True)

    # greedy coordinate ascent — tune weights on TUNE split only
    cur = dict(base)
    all_terms = list(terms.keys())
    grid = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    best = a_base_tune
    for it in range(4):
        improved = False
        for k in all_terms:
            local_best, local_w = best, cur.get(k, 0.0)
            for w in grid:
                cur[k] = w
                a = acc(cur, tune_mask)
                if a > local_best + 1e-6:
                    local_best, local_w = a, w
            cur[k] = local_w
            if local_best > best + 1e-6:
                best = local_best
                improved = True
        print(f'[{time.time()-t0:.0f}s] iter {it}: tune b = {best:.3f}', flush=True)
        if not improved:
            break

    cur = {k: v for k, v in cur.items() if v != 0}
    tuned_val = acc(cur, val_mask)
    tuned_full = acc(cur)
    print('\n=== tuned stack ===', flush=True)
    for k, v in sorted(cur.items(), key=lambda x: -x[1]):
        print(f'  {k[0]:16s} @ {k[1]:4s} : {v}', flush=True)
    print(f'TUNE   b: {a_base_tune:.3f} -> {best:.3f}  (+{best-a_base_tune:.3f})', flush=True)
    print(f'VAL    b: {a_base_val:.3f} -> {tuned_val:.3f}  (+{tuned_val-a_base_val:.3f})  '
          f'<-- honest generalization', flush=True)
    print(f'FULL   b: {a_base_full:.3f} -> {tuned_full:.3f}', flush=True)
    # project real using VAL delta only (honest)
    real_b0 = 19.48
    val_ratio = tuned_val / a_base_val if a_base_val > 0 else 1.0
    real_b = real_b0 * val_ratio
    d_overall = 0.4365 * 0.545 * (real_b - real_b0) / 100
    print(f'projected real b_cond (val-scaled): {real_b0:.2f} -> {real_b:.2f}  '
          f'=> d_overall ~ {100*d_overall:+.3f}pt', flush=True)
    best = tuned_full

    json.dump({'base': base_keys(base), 'base_full': a_base_full, 'base_val': a_base_val,
               'tuned': base_keys(cur), 'tuned_full': tuned_full, 'tuned_val': tuned_val,
               'proj_real_b': real_b, 'proj_d_overall_pt': 100 * d_overall},
              open(f'{OUT}/fusion_sweep_b_results.json', 'w'), indent=1)
    print('wrote outputs/fusion_sweep_b_results.json', flush=True)


def base_keys(d):
    return {f'{k[0]}@{k[1]}': v for k, v in d.items()}


if __name__ == '__main__':
    main()
