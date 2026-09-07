"""Compare ctftshift vs ctftbig on deployed unseen hard-sim; fuse with TaxaBind.

Requires: outputs/emb_train_ctftshift.pt, emb_train_ctftbig.pt, emb_*_taxabind, text embeds.
"""
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import FishData, dbnorm, load_emb, OUT

t0 = time.time()
D = FishData()
cand = D.cand
print(f'[{time.time()-t0:.0f}s] cand={len(cand)} pseudo={len(D.pseudo)}', flush=True)


def queries(idx, feats):
    q, y = [], []
    for c in D.pseudo:
        for fn in D.by[c]:
            if fn in idx:
                q.append(feats[idx[fn]])
                y.append(D.cand_pos[D.ci[c]])
    return torch.stack(q), torch.tensor(y)


def t1(S, gold):
    return (S.argmax(1) == gold).float().mean().item() * 100


TtH_c = D.TtH[cand]
TnL_c = D.TnL[cand]
TTX = F.normalize(torch.load(os.path.join(OUT, 'text_emb_h_promptens.pt'), weights_only=False)['emb_taxctx'].float(), dim=-1)
TTX_c = TTX[cand]
Ttb = F.normalize(torch.load(os.path.join(OUT, 'text_emb_taxabind_taxctx.pt'), weights_only=False)['emb_taxctx'].float(), dim=-1)
Ttb_c = Ttb[cand]

# legs
legs = {}
for tag in ['ctftbig', 'ctftshift', 'ftshift', 'fullft336_v2']:
    p = os.path.join(OUT, f'emb_train_{tag}.pt')
    if not os.path.exists(p):
        print(f'MISSING {p}', flush=True)
        continue
    idx, feats, _ = load_emb(p)
    q, gold = queries(idx, feats)
    legs[tag] = (q, gold)
    print(f'[{time.time()-t0:.0f}s] {tag}: n={len(q)}', flush=True)

qL, gold = queries(D.LtI, D.LtF)
# taxabind train emb may not exist — use frozen extract if we only have test/unseen
# For proxy use train images encoded... we don't have emb_train_taxabind.
# Fall back: score taxabind only if emb_train_taxabind exists; else skip alone and use
# the cached proxy approach from image files is too slow. Check.
ptb = os.path.join(OUT, 'emb_train_taxabind.pt')
has_tb_train = os.path.exists(ptb)
if has_tb_train:
    idx, feats, _ = load_emb(ptb)
    qtb, _ = queries(idx, feats)
else:
    qtb = None
    print('no emb_train_taxabind — TaxaBind fuse uses precomputed proxy numbers only', flush=True)

results = {}
# deployed-ish full recipe (approximate v33 text stack on hard-sim)
dep = (dbnorm(legs['ctftbig'][0] @ TtH_c.t()) + 0.5 * dbnorm(qL @ TnL_c.t())
       + 0.75 * dbnorm(legs['fullft336_v2'][0] @ TtH_c.t())
       + 1.0 * dbnorm(legs['ftshift'][0] @ TtH_c.t())
       + 1.0 * dbnorm(legs['ctftbig'][0] @ TTX_c.t()))
a_dep = t1(dep, gold)
print(f'DEPLOYED-ish: {a_dep:.2f}', flush=True)
results['deployed_ish'] = a_dep

# replace ctftbig with ctftshift in the stack
if 'ctftshift' in legs:
    dep_s = (dbnorm(legs['ctftshift'][0] @ TtH_c.t()) + 0.5 * dbnorm(qL @ TnL_c.t())
             + 0.75 * dbnorm(legs['fullft336_v2'][0] @ TtH_c.t())
             + 1.0 * dbnorm(legs['ftshift'][0] @ TtH_c.t())
             + 1.0 * dbnorm(legs['ctftshift'][0] @ TTX_c.t()))
    a_s = t1(dep_s, gold)
    print(f'DEPLOYED with ctftshift: {a_s:.2f} (delta {a_s-a_dep:+.2f})', flush=True)
    results['deployed_ctftshift'] = a_s
    results['delta_ctftshift'] = a_s - a_dep

    # blend both ctft legs
    for w in [0.5, 1.0]:
        blend = dep + w * dbnorm(legs['ctftshift'][0] @ TtH_c.t()) + w * dbnorm(legs['ctftshift'][0] @ TTX_c.t())
        a = t1(blend, gold)
        print(f'deployed + {w}*ctftshift(taxon+taxctx): {a:.2f} (d {a-a_dep:+.2f})', flush=True)
        results[f'deployed+{w}*ctftshift'] = a

# TaxaBind add-on (need train emb — extract on the fly from cached if missing)
# Use research extract is heavy; if no train, approximate with existing fuse result file
fuse_prev = os.path.join(OUT, 'unseen_altleg_fuse_deployed_results.json')
if os.path.exists(fuse_prev):
    prev = json.load(open(fuse_prev))
    results['prev_taxabind_fuse'] = prev
    print(f'prev TaxaBind fuse best from file: see JSON', flush=True)

if qtb is not None:
    for w in [0.5, 1.0, 1.5]:
        a = t1(dep + w * dbnorm(qtb @ Ttb_c.t()), gold)
        print(f'deployed + {w}*taxabind: {a:.2f} (d {a-a_dep:+.2f})', flush=True)
        results[f'deployed+{w}*tb'] = a
        if 'ctftshift' in legs:
            a2 = t1(dep_s + w * dbnorm(qtb @ Ttb_c.t()), gold)
            print(f'deployed_shift + {w}*taxabind: {a2:.2f} (d {a2-a_dep:+.2f})', flush=True)
            results[f'deployed_shift+{w}*tb'] = a2

json.dump(results, open(os.path.join(OUT, 'ctftshift_fuse_results.json'), 'w'), indent=1)
print('wrote outputs/ctftshift_fuse_results.json', flush=True)
