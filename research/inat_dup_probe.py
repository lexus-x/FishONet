"""Near-duplicate leakage probe: are iNat proto images duplicates of train queries?

For a sample of covered pseudo classes, embed each iNat image individually (frozen H)
and compare to the frozen-H train-query embeddings of the SAME class and of OTHER
classes. If same-class max-sim is ~0.98+ we have duplicate leakage; if it's well
below and comparable to genuine visual similarity, the proto lift is legitimate.
"""
import json
import os
import random
import sys
import time

import open_clip
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import FishData, load_emb, OUT

MODEL = 'hf-hub:imageomics/bioclip-2.5-vith14'
DEV = 'cuda'

D = FishData()
meta = json.load(open(os.path.join(OUT, 'inat_image_files.json')))
hIdx, hFeat, _ = load_emb(os.path.join(OUT, 'emb_train_h.pt'))

covered_pseudo = [c for c in D.pseudo if meta.get(c)]
random.Random(1).shuffle(covered_pseudo)
sample = covered_pseudo[:80]
print(f'sampling {len(sample)} covered pseudo classes', flush=True)

model, _, pp = open_clip.create_model_and_transforms(MODEL)
model = model.to(DEV).eval()

# embed iNat images per sampled class
inat_emb = {}  # class -> [k,1024]
for c in sample:
    fs = []
    for p in meta[c][:20]:
        if os.path.exists(p) and os.path.getsize(p) > 1000:
            try:
                im = pp(Image.open(p).convert('RGB')).unsqueeze(0).to(DEV)
                with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                    f = F.normalize(model.encode_image(im).float(), dim=-1)
                fs.append(f.cpu()[0])
            except Exception:
                pass
    if fs:
        inat_emb[c] = torch.stack(fs)

# train query embeddings per class (H)
tq = {}
for c in sample:
    v = [hFeat[hIdx[fn]] for fn in D.by[c] if fn in hIdx]
    if v:
        tq[c] = torch.stack(v)

allc = [c for c in sample if c in inat_emb and c in tq]
# same-class max sim per train query
same_max = []
for c in allc:
    S = tq[c] @ inat_emb[c].t()  # [nq, k]
    same_max.extend(S.max(1).values.tolist())
same_max = torch.tensor(same_max)

# cross-class max sim: each train query vs iNat of a random OTHER class
other_max = []
for c in allc:
    oc = random.choice([x for x in allc if x != c])
    S = tq[c] @ inat_emb[oc].t()
    other_max.extend(S.max(1).values.tolist())
other_max = torch.tensor(other_max)


def pct(t, q):
    return torch.quantile(t, q).item()


print(f'\nSAME-class train-query -> iNat max cosine (n={len(same_max)}):')
print(f'  mean={same_max.mean():.3f} median={pct(same_max,0.5):.3f} '
      f'p90={pct(same_max,0.9):.3f} p99={pct(same_max,0.99):.3f} max={same_max.max():.3f}')
print(f'  frac >0.99 (near-dup): {(same_max>0.99).float().mean()*100:.1f}%  '
      f'>0.95: {(same_max>0.95).float().mean()*100:.1f}%  >0.90: {(same_max>0.90).float().mean()*100:.1f}%')
print(f'\nCROSS-class (calibration) max cosine (n={len(other_max)}):')
print(f'  mean={other_max.mean():.3f} median={pct(other_max,0.5):.3f} p90={pct(other_max,0.9):.3f} max={other_max.max():.3f}')

res = {'n_same': len(same_max), 'same_mean': same_max.mean().item(),
       'same_median': pct(same_max, 0.5), 'same_p99': pct(same_max, 0.99),
       'same_max': same_max.max().item(),
       'frac_gt_099': (same_max > 0.99).float().mean().item(),
       'frac_gt_095': (same_max > 0.95).float().mean().item(),
       'cross_mean': other_max.mean().item(), 'cross_p90': pct(other_max, 0.9)}
json.dump(res, open(os.path.join(OUT, 'inat_dup_probe_results.json'), 'w'), indent=1)
print('\nwrote outputs/inat_dup_probe_results.json')
