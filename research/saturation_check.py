"""Adversarial audit: is the v36 unseen recognizer SATURATED vs its own oracle-family ceiling?

Computes, on the SAME pseudo-unseen proxy (train-image hard-sim), the current best
image+text unseen stack top-1 both UNCONSTRAINED (argmax over full cand) and
ORACLE-FAMILY-CONSTRAINED (argmax restricted to gold's family). If the headroom from
PERFECT family knowledge is small, no TTA/reweight/backbone tweak on this encoder can
lift b materially -> b is model-limited and 53% is unreachable with this family.

No TaxaBind train emb available; uses the v36 image+text stack minus the frozen-TB leg
(deployed_ctftshift, proxy 31.45). TB adds ~+1.9 on top and does not change the ceiling
argument (a frozen ViT-B leg cannot exceed the family-oracle bound of the encoder family).
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

# family map per class
tax = json.load(open(os.path.join(OUT, 'taxonomy_full.json')))
fam_of = {}
for i, c in enumerate(D.classes):
    rec = tax.get(c)
    fam_of[i] = (rec or {}).get('family') if rec else None
cand_list = cand.tolist()
cand_fam = [fam_of[int(x)] for x in cand_list]


def queries(idx, feats):
    q, y = [], []
    for c in D.pseudo:
        for fn in D.by[c]:
            if fn in idx:
                q.append(feats[idx[fn]])
                y.append(D.cand_pos[D.ci[c]])
    return torch.stack(q), torch.tensor(y)


TtH_c = D.TtH[cand]
TnL_c = D.TnL[cand]
TTX = F.normalize(torch.load(os.path.join(OUT, 'text_emb_h_promptens.pt'), weights_only=False)['emb_taxctx'].float(), dim=-1)
TTX_c = TTX[cand]

legs = {}
for tag in ['ctftshift', 'ftshift', 'fullft336_v2']:
    idx, feats, _ = load_emb(os.path.join(OUT, f'emb_train_{tag}.pt'))
    q, gold = queries(idx, feats)
    legs[tag] = q
    print(f'[{time.time()-t0:.0f}s] {tag}: n={len(q)}', flush=True)
qL, gold = queries(D.LtI, D.LtF)

# v36 unseen text stack (minus TaxaBind)
S = (dbnorm(legs['ctftshift'] @ TtH_c.t()) + 0.5 * dbnorm(qL @ TnL_c.t())
     + 0.75 * dbnorm(legs['fullft336_v2'] @ TtH_c.t())
     + 1.0 * dbnorm(legs['ftshift'] @ TtH_c.t())
     + 1.0 * dbnorm(legs['ctftshift'] @ TTX_c.t()))

gold_np = gold.tolist()
# gold family per query
gold_fam = [cand_fam[g] for g in gold_np]

uncon = (S.argmax(1) == gold).float().mean().item() * 100
print(f'\nv36 stack (minus TB) UNCONSTRAINED top-1 = {uncon:.2f}', flush=True)

# oracle-family-constrained: mask candidates not in gold's family
famcand = {}  # family -> bool mask over cand
for f in set(cand_fam):
    if f is None:
        continue
    famcand[f] = torch.tensor([cf == f for cf in cand_fam])

Sc = S.clone()
n = S.shape[0]
correct = 0
n_valid = 0
famsize = []
for i in range(n):
    gf = gold_fam[i]
    if gf is None or gf not in famcand:
        # no family info -> leave unconstrained (counts as unconstrained)
        pred = int(S[i].argmax())
    else:
        mask = famcand[gf]
        famsize.append(int(mask.sum()))
        row = S[i].clone()
        row[~mask] = -1e9
        pred = int(row.argmax())
    if pred == gold_np[i]:
        correct += 1
    n_valid += 1
oracle = 100.0 * correct / n_valid
import statistics
print(f'v36 stack (minus TB) ORACLE-FAMILY top-1 = {oracle:.2f}  headroom {oracle-uncon:+.2f}', flush=True)
print(f'median family cand size = {statistics.median(famsize) if famsize else 0}, mean = {sum(famsize)/max(1,len(famsize)):.1f}', flush=True)

res = {'unconstrained': uncon, 'oracle_family': oracle, 'headroom': oracle - uncon,
       'n': n, 'note': 'minus TaxaBind leg; TB adds ~+1.9 base, not ceiling'}
json.dump(res, open(os.path.join(OUT, 'saturation_check_results.json'), 'w'), indent=1)
print('wrote outputs/saturation_check_results.json', flush=True)
