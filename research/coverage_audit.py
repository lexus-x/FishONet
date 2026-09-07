"""Enumerate covered vs uncovered unseen-route classes given a proto set."""
import json
import os
import pickle
import sys

import torch
import torch.nn.functional as F

D = 'data/dl'
PROTO = sys.argv[1] if len(sys.argv) > 1 else 'outputs/inat_protos_ctftshift_full.pt'
classes = list(pickle.load(open(f'{D}/all_classes.pkl', 'rb')))
ci = {c: i for i, c in enumerate(classes)}
lab = json.load(open(f'{D}/label_train.json'))
seen = sorted(set(v for v in lab.values() if v in ci))
seen_set = set(seen)
other = [c for c in classes if c not in seen_set]
print('total classes', len(classes), 'seen', len(seen), 'unseen-route', len(other))

pd = torch.load(PROTO, weights_only=False)
P = F.normalize(pd['protos'].float(), dim=-1)
name2row = {c: i for i, c in enumerate(pd['classes'])}
covered = set()
for c in other:
    r = name2row.get(c)
    if r is not None and float(P[r].norm()) > 0.5:
        covered.add(c)
uncov = [c for c in other if c not in covered]
print(f'unseen-route covered {len(covered)}/{len(other)} ({100*len(covered)/len(other):.1f}%)')
print(f'uncovered {len(uncov)}')
json.dump(uncov, open('outputs/uncovered_unseen_classes.json', 'w'))
json.dump(other, open('outputs/unseen_route_classes.json', 'w'))
print('sample uncovered:', uncov[:15])
