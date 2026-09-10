#!/bin/bash
# CSLS (Cross-domain Similarity Local Scaling) for unseen class improvement
# ~5 lines of code, O(N*k), typically +0.5-2% on zero-shot retrieval
# Runs on existing cached embeddings (no GPU needed)

source ~/miniconda3/etc/profile.d/conda.sh
conda activate onet
cd ~/onet

echo "=== CSLS UNSEEN EXPERIMENT $(date) ==="
echo "Hypothesis: CSLS hubness correction beats DBNorm for unseen classification"
echo ""

python -c "
import json, torch
import torch.nn.functional as F

# Load text embeddings and unseen features (same as v15/v20 unseen route)
txtHt = torch.load('outputs/text_emb_h_taxon.pt', weights_only=False)
classes = txtHt['classes']; ci = {c: i for i, c in enumerate(classes)}
TtH = F.normalize(txtHt['emb_taxon'].float(), dim=-1)
TnL = F.normalize(torch.load('outputs/text_emb.pt', weights_only=False)['emb_name'].float(), dim=-1)

lab = json.load(open('data/dl/label_train.json'))
seen = set(lab.values())

# Unseen class indices
nonk = torch.tensor([i for i, c in enumerate(classes) if c not in seen])
print(f'Unseen classes: {len(nonk)}, Total: {len(classes)}')

# Load unseen features (same as v20)
Hun = torch.load('outputs/emb_unseen_ctftbig.pt', weights_only=False)
Lun = torch.load('outputs/emb_unseen.pt', weights_only=False)
Hfiles = Hun.pop('files', []) or Hun.get('files', [])
Lfiles = Lun.pop('files', []) or Lun.get('files', [])

# Get common files
uf = [fn for fn in Hfiles if fn in Lfiles]
Hmap = {fn: i for i, fn in enumerate(Hfiles)}
Lmap = {fn: i for i, fn in enumerate(Lfiles)}

uH = F.normalize(Hun['feats'][torch.tensor([Hmap[fn] for fn in uf])].float(), dim=-1)
uL = F.normalize(Lun['feats'][torch.tensor([Lmap[fn] for fn in uf])].float(), dim=-1)

# Text embeddings for unseen
TtHu = TtH[nonk]  # [U, D]
TnLu = TnL[nonk]  # [U, D]

# === DBNorm (current v20 method) ===
def db(M): return (M - M.mean(0, keepdim=True)) / (M.std(0, keepdim=True) + 1e-6)
def dis(M, a, b): return F.log_softmax(M / a, dim=0) + F.log_softmax(M / b, dim=1)

MH = uH @ TtHu.t()
ML = uL @ TnLu.t()
dbnorm_score = dis(MH, 0.05, 0.5) + 0.5 * dis(ML, 0.05, 0.5)
dbnorm_pred = nonk[dbnorm_score.argmax(1)]
print(f'DBNorm predictions: {len(dbnorm_pred)}')

# === CSLS hubness correction ===
def csls(S_img_txt, k=5):
    \"\"\"CSLS: penalize hubs by subtracting mean similarity to nearest neighbors\"\"\"
    # r_T: mean text-similarity to k-nearest images per text column
    r_T = S_img_txt.topk(k, dim=0).values.mean(dim=0)  # [U]
    # r_I: mean image-similarity to k-nearest texts per image row  (usually 0 for retrieval)
    r_I = S_img_txt.topk(k, dim=1).values.mean(dim=1)  # [N]
    # CSLS(x,y) = 2*cos(x,y) - r_T(y) - r_I(x)  
    return 2 * S_img_txt - r_T.unsqueeze(0) - r_I.unsqueeze(1)

# Try CSLS with different k values
for k in [3, 5, 10, 20]:
    # On taxonomic text (ViT-H)
    S_taxon = uH @ TtHu.t()  # [N, U]
    csls_taxon = csls(S_taxon, k=k)
    
    # On name-text (ViT-L, different dim)
    S_name = uL @ TnLu.t()
    csls_name = csls(S_name, k=k)
    
    # Ensemble: same as v20 (taxon + 0.5*name)
    sc = csls_taxon + 0.5 * csls_name
    pred = nonk[sc.argmax(1)]
    
    tot = len(pred)
    match_csls = sum(1 for i in range(tot) if csls_taxon.argmax(1)[i] != csls_name.argmax(1)[i])
    print(f'CSLS k={k:2d}: taxon-name disagreement = {match_csls}/{tot} ({100*match_csls/tot:.1f}%)')
    
    # Save for comparison with v20'dbnorm
    if k == 5:
        # Check how many differ from DBNorm
        diff_from_dbnorm = sum(1 for i in range(tot) if pred[i].item() != dbnorm_pred[i].item())
        print(f'  Differ from DBNorm: {diff_from_dbnorm}/{tot} ({100*diff_from_dbnorm/tot:.1f}%)')

print('')
print('CSLS-different prediction locations found: these are test images where')
print('CSLS disagrees with DBNorm — could be wins or losses on real leaderboard.')
print('Needs Codabench submission to verify.')

# === Z-score debias (original cheap win) ===
S_taxon_z = (S_taxon - S_taxon.mean(0, keepdim=True)) / (S_taxon.std(0, keepdim=True) + 1e-6)
S_name_z = (S_name - S_name.mean(0, keepdim=True)) / (S_name.std(0, keepdim=True) + 1e-6)
z_score = S_taxon_z + 0.5 * S_name_z
z_pred = nonk[z_score.argmax(1)]
diff_z = sum(1 for i in range(tot) if z_pred[i].item() != dbnorm_pred[i].item())
print(f'Z-score debias differs from DBNorm: {diff_z}/{tot} ({100*diff_z/tot:.1f}%)')

# === Weighted blend (CSLS + DBNorm) ===
for alpha in [0.3, 0.5, 0.7]:
    blended = alpha * dbnorm_score + (1 - alpha) * (csls_taxon + 0.5 * csls_name)
    bl_pred = nonk[blended.argmax(1)]
    diff_bl = sum(1 for i in range(tot) if bl_pred[i].item() != dbnorm_pred[i].item())
    print(f'Blend alpha={alpha}: differs from DBNorm = {diff_bl}/{tot} ({100*diff_bl/tot:.1f}%)')

print('\\nNOTE: Cannot validate which is better without Codabench upload.')
print('CSLS typically gives +0.5-2% on zero-shot retrieval over z-score alone.')
" 2>&1

echo ""
echo "=== DONE $(date) ==="
echo "NEXT: Build a prediction.json that combines the best seen route (v20 ensemble)"
echo "with the candidate unseen route (CSLS or CSLS+DBNorm blend) and submit."