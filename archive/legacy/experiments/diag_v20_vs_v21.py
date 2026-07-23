import json, pickle, torch
import torch.nn.functional as F
D = "data/dl"
classes = list(pickle.load(open(f"{D}/all_classes.pkl", "rb")))
ci = {c: i for i, c in enumerate(classes)}
lab = json.load(open(f"{D}/label_train.json"))
seen_set = set(lab.values())
unseen_files = pickle.load(open(f"{D}/splits/unseen.pkl", "rb"))

p21 = json.load(open("outputs/prediction_v21_unified_c.json"))
p20 = json.load(open("outputs/prediction_v20_real.json"))

# 1) v20 vs v21 agreement on the 15568 unseen images
both = [f for f in unseen_files if f in p20 and f in p21]
agree = sum(1 for f in both if p20[f] == p21[f])
v20_unseen_col = sum(1 for f in both if p20[f] not in seen_set)
v21_unseen_col = sum(1 for f in both if p21[f] not in seen_set)
print(f"unseen imgs in both preds: {len(both)}")
print(f"  v20 predicts UNSEEN-class: {100*v20_unseen_col/len(both):.1f}%  (v20 routed unseen->unseen-only, expect ~100)")
print(f"  v21 predicts UNSEEN-class: {100*v21_unseen_col/len(both):.1f}%")
print(f"  v20==v21 agreement: {agree}/{len(both)} = {100*agree/len(both):.1f}%")

# 2) rebuild v20's ACTUAL unseen route here (ctftbig taxon + name, DBNorm over unseen-ONLY),
#    then rebuild it with UNIFIED combined-batch DBNorm, to isolate the normalization effect.
def loadn(p):
    d = torch.load(p, weights_only=False)
    return {fn: i for i, fn in enumerate(d["files"])}, F.normalize(d["feats"].float(), dim=-1), list(d["files"])

TtH = F.normalize(torch.load("outputs/text_emb_h_taxon.pt", weights_only=False)["emb_taxon"].float(), dim=-1)
TnL = F.normalize(torch.load("outputs/text_emb.pt", weights_only=False)["emb_name"].float(), dim=-1)
nonk = torch.tensor([i for i, c in enumerate(classes) if c not in seen_set])

def dbnorm(S, tc=0.05, tr=0.5):
    return F.log_softmax(S / tc, dim=0) + F.log_softmax(S / tr, dim=1)

# v20 features (ctftbig for taxon-H, plain L for name) over unseen images
HunI, HunF, Hunf = loadn("outputs/emb_unseen_ctftbig.pt")
LunI, LunF, Lunf = loadn("outputs/emb_unseen.pt")
uf = [f for f in unseen_files if f in HunI and f in LunI]
uH = torch.stack([HunF[HunI[f]] for f in uf])
uL = torch.stack([LunF[LunI[f]] for f in uf])
TtHu = TtH[nonk]; TnLu = TnL[nonk]
MH = uH @ TtHu.t(); ML = uL @ TnLu.t()

# (B) v20-style: DBNorm over unseen-only batch
scoreB = dbnorm(MH, 0.05, 0.5) + 0.5 * dbnorm(ML, 0.05, 0.5)
predB = {f: classes[nonk[j].item()] for f, j in zip(uf, scoreB.argmax(1).tolist())}
agreeB_v20 = sum(1 for f in uf if predB[f] == p20.get(f)) / len(uf)
print(f"\nrebuilt v20-route (ctftbig, unseen-only DBNorm) vs prediction_v20_real: {100*agreeB_v20:.1f}% agree (sanity, expect high)")

# 3) v21 features (plain H) same unseen-only DBNorm -> isolate FEATURE effect (h vs ctftbig)
HhI, HhF, Hhf = loadn("outputs/emb_unseen_h.pt")
uf2 = [f for f in unseen_files if f in HhI and f in LunI]
uHh = torch.stack([HhF[HhI[f]] for f in uf2])
uLh = torch.stack([LunF[LunI[f]] for f in uf2])
TnH = F.normalize(torch.load("outputs/text_emb_h.pt", weights_only=False)["emb_name"].float(), dim=-1)
MHh = uHh @ TtH[nonk].t(); MLh = uHh @ TnH[nonk].t()
scoreC = dbnorm(MHh, 0.05, 0.5) + 0.5 * dbnorm(MLh, 0.05, 0.5)
predC = {f: classes[nonk[j].item()] for f, j in zip(uf2, scoreC.argmax(1).tolist())}
agreeC_B = sum(1 for f in uf2 if f in predB and predC[f] == predB[f]) / len(uf2)
agreeC_v21 = sum(1 for f in uf2 if predC[f] == p21.get(f)) / len(uf2)
print(f"v21-feats(plain H) unseen-only DBNorm vs v20-route(ctftbig): {100*agreeC_B:.1f}% agree (feature effect)")
print(f"v21-feats unseen-only DBNorm vs v21-AS-SUBMITTED(combined-batch): {100*agreeC_v21:.1f}% agree (normalization effect)")
