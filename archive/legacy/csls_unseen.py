"""CSLS hubness correction for unseen classification. Post-hoc, no training.
Compares DBNorm (current v20), CSLS, and z-score debias."""
import json, torch, sys
import torch.nn.functional as F

print("=== CSLS UNSEEN EXPERIMENT ===")
# Load embeddings
txtHt = torch.load('outputs/text_emb_h_taxon.pt', weights_only=False)
classes = txtHt['classes']
ci = {c: i for i, c in enumerate(classes)}
TtH = F.normalize(txtHt['emb_taxon'].float(), dim=-1)
TnL = F.normalize(torch.load('outputs/text_emb.pt', weights_only=False)['emb_name'].float(), dim=-1)

lab = json.load(open('data/dl/label_train.json'))
seen = set(lab.values())
nonk = torch.tensor([i for i, c in enumerate(classes) if c not in seen])
print(f"Unseen: {len(nonk)}, Total: {len(classes)}")

# Load unseen features
Hun = torch.load('outputs/emb_unseen_ctftbig.pt', weights_only=False)
Lun = torch.load('outputs/emb_unseen.pt', weights_only=False)
Hfiles = Hun.get('files', [])
Lfiles = Lun.get('files', [])

uf = sorted(set(Hfiles) & set(Lfiles))
Hmap = {fn: i for i, fn in enumerate(Hfiles)}
Lmap = {fn: i for i, fn in enumerate(Lfiles)}

uH = F.normalize(Hun['feats'][torch.tensor([Hmap[fn] for fn in uf])].float(), dim=-1)
uL = F.normalize(Lun['feats'][torch.tensor([Lmap[fn] for fn in uf])].float(), dim=-1)
TtHu = TtH[nonk]
TnLu = TnL[nonk]

N, U = uH.shape[0], len(nonk)
print(f"Unseen queries: {N}, Unseen candidates: {U}")

# === DBNorm (current v20) ===
def db(M):
    return (M - M.mean(0, keepdim=True)) / (M.std(0, keepdim=True) + 1e-6)
def dis(M, a, b):
    return F.log_softmax(M / a, dim=0) + F.log_softmax(M / b, dim=1)

MH = uH @ TtHu.t()
ML = uL @ TnLu.t()
dbnorm_score = dis(MH, 0.05, 0.5) + 0.5 * dis(ML, 0.05, 0.5)
dbnorm_pred = nonk[dbnorm_score.argmax(1)]
print(f"DBNorm preds: {len(dbnorm_pred)}")

# === CSLS ===
def csls(S, k=5):
    r_T = S.topk(k, dim=0).values.mean(dim=0)
    r_I = S.topk(k, dim=1).values.mean(dim=1)
    return 2 * S - r_T.unsqueeze(0) - r_I.unsqueeze(1)

for k in [3, 5, 10, 20]:
    S_taxon = uH @ TtHu.t()
    S_name = uL @ TnLu.t()
    cs_taxon = csls(S_taxon, k=k)
    cs_name = csls(S_name, k=k)
    sc = cs_taxon + 0.5 * cs_name
    pred = nonk[sc.argmax(1)]
    diff = (pred != dbnorm_pred).sum().item()
    print(f"CSLS k={k:2d}: differs from DBNorm = {diff}/{N} ({100*diff/N:.1f}%)")

# === Z-score debias ===
S_taxon = uH @ TtHu.t()
S_name = uL @ TnLu.t()
z_taxon = db(S_taxon)
z_name = db(S_name)
z_score = z_taxon + 0.5 * z_name
z_pred = nonk[z_score.argmax(1)]
diff_z = (z_pred != dbnorm_pred).sum().item()
print(f"Z-score: differs from DBNorm = {diff_z}/{N} ({100*diff_z/N:.1f}%)")

# === Blend CSLS+DBnorm ===
cs_taxon = csls(uH @ TtHu.t(), k=5)
cs_name = csls(uL @ TnLu.t(), k=5)
cs_score = cs_taxon + 0.5 * cs_name

for alpha in [0.3, 0.5, 0.7]:
    blended = alpha * dbnorm_score + (1 - alpha) * cs_score
    bl_pred = nonk[blended.argmax(1)]
    diff_bl = (bl_pred != dbnorm_pred).sum().item()
    print(f"Blend a={alpha}: differs from DBNorm = {diff_bl}/{N} ({100*diff_bl/N:.1f}%)")

print("\nDONE. CSLS vs DBNorm requires Codabench submission to validate.")
print(f"CSLS k=5 affects {100*(pred != dbnorm_pred).sum().item()/N:.1f}% of unseen predictions.")