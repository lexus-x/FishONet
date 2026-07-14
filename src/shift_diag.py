import json, torch, torch.nn.functional as F
from collections import defaultdict
torch.set_num_threads(8)

def load(p):
    d = torch.load(p, weights_only=False)
    return {fn: i for i, fn in enumerate(d["files"])}, F.normalize(d["feats"].float(), dim=-1), list(d["files"])

FtrI, FtrF, Ftrf = load("outputs/emb_train_ft.pt")
FteI, FteF, Ftef = load("outputs/emb_test_ft.pt")
LtrI, LtrF, _ = load("outputs/emb_train.pt")

lab = json.load(open("data/dl/label_train.json"))
trainfiles = [fn for fn in Ftrf if fn in LtrI and fn in lab]

# same-distribution control: held-out 20% tail of train (same split ft.py/predict_v09 use)
by = defaultdict(list)
for fn in trainfiles: by[lab[fn]].append(fn)
fit_files, val_files = [], []
for c, fns in by.items():
    fns = sorted(fns)
    if len(fns) >= 3:
        k = max(1, round(0.2 * len(fns)))
        fit_files += fns[:-k]; val_files += fns[-k:]
    else:
        fit_files += fns

mu_fit  = FtrF[[FtrI[f] for f in fit_files]].mean(0)
mu_val  = FtrF[[FtrI[f] for f in val_files]].mean(0)
mu_test = FteF[[FteI[f] for f in Ftef]].mean(0)

def cos(a, b): return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
def l2(a, b): return (a - b).norm().item()

# baseline: typical inter-prototype spread (how far apart two random class prototypes sit)
protos = []
for c, fns in by.items():
    if len(fns) >= 2:
        protos.append(FtrF[[FtrI[f] for f in fns]].mean(0))
protos = torch.stack(protos)
protos_n = F.normalize(protos, dim=-1)
import random
random.seed(0)
idx = list(range(len(protos)))
pair_cos = []
for _ in range(2000):
    i, j = random.sample(idx, 2)
    pair_cos.append(F.cosine_similarity(protos_n[i:i+1], protos_n[j:j+1]).item())
pair_cos = torch.tensor(pair_cos)

print(f"fit_files={len(fit_files)} val_files={len(val_files)} test_files={len(Ftef)}")
print(f"mean-shift cos(fit,val)  [same-dist control] = {cos(mu_fit, mu_val):.4f}   L2={l2(mu_fit, mu_val):.4f}")
print(f"mean-shift cos(fit,test) [real domain gap]   = {cos(mu_fit, mu_test):.4f}   L2={l2(mu_fit, mu_test):.4f}")
print(f"inter-class-prototype cos: mean={pair_cos.mean():.4f} std={pair_cos.std():.4f} (reference scale for 'far apart')")
