"""Test adding fullft336 cropviews, ftshift bank, bioclip2 bank, and additional prototypes
on the holdout proxy to evaluate accuracy lift on unseen species!
"""
import os
import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, 'research')
from common import FishData, dbnorm, load_emb, OUT
from crop_views_proxy import score_bank_raw
from inat_maxpool_proxy import score_maxpool, top1
from v41_taxctx_weight_cv import stack_base
from v50_shiftbank_proxy import BANK_336, B2_WF, B2_WL, F336_Q, IMG_W, proto_leg, queries

dev = 'cuda' if torch.cuda.is_available() else 'cpu'

def test_unseen_bank_expansion():
    print("=== Testing Unseen Bank Expansion on Holdout Proxy ===", flush=True)
    D = FishData()
    cand = D.cand
    S0, gold, qH = stack_base(D, cand, 1.0)
    
    # Baseline v56
    bank_ctft = torch.load('outputs/inat_photo_bank_ctftshift.pt', weights_only=False)['bank']
    Sct = score_maxpool(qH, bank_ctft, cand, topm=4)
    
    fi, ff, _ = load_emb('outputs/emb_train_bioclip2.pt')
    Qf, _ = queries(D, fi, ff)
    Sf = proto_leg(Qf, 'outputs/inat_tol_merged_b2_a05.pt', cand)
    
    li, lf, _ = load_emb('outputs/emb_train_bioclip2_lora_v2.pt')
    Ql, _ = queries(D, li, lf)
    Sl = proto_leg(Ql, 'outputs/inat_tol_merged_b2lora_a0.5.pt', cand)
    
    f336i, f336f, _ = load_emb('outputs/emb_train_fullft336shift.pt')
    Q336, _ = queries(D, f336i, f336f)
    bank_336 = torch.load(BANK_336, weights_only=False)['bank']
    S336 = score_maxpool(Q336, bank_336, cand, topm=4)
    
    base_v56 = S0 + 4.0 * Sct + 3.0 * S336 + B2_WF * Sf + B2_WL * Sl
    acc_v56 = top1(base_v56, gold)
    print(f"Baseline v56 Proxy Holdout Top-1: {acc_v56:.4f}", flush=True)
    
    # Test 1: Add ftshift photo bank
    if os.path.exists('outputs/inat_photo_bank_ftshift.pt'):
        print("Testing + ftshift photo bank...", flush=True)
        fti, ftf, _ = load_emb('outputs/emb_train_ftshift.pt')
        Qft, _ = queries(D, fti, ftf)
        bank_ft = torch.load('outputs/inat_photo_bank_ftshift.pt', weights_only=False)['bank']
        Sft = score_maxpool(Qft, bank_ft, cand, topm=4)
        for w_ft in [1.0, 2.0, 3.0, 4.0]:
            acc = top1(base_v56 + w_ft * Sft, gold)
            print(f"  + w_ft={w_ft}: {acc:.4f} (diff: {acc - acc_v56:+.4f})", flush=True)

    # Test 2: Add bioclip2 photo bank
    if os.path.exists('outputs/inat_photo_bank_bioclip2.pt'):
        print("Testing + bioclip2 photo bank...", flush=True)
        bank_b2 = torch.load('outputs/inat_photo_bank_bioclip2.pt', weights_only=False)['bank']
        Sb2 = score_maxpool(Qf, bank_b2, cand, topm=4)
        for w_b2 in [1.0, 2.0, 3.0, 4.0]:
            acc = top1(base_v56 + w_b2 * Sb2, gold)
            print(f"  + w_b2={w_b2}: {acc:.4f} (diff: {acc - acc_v56:+.4f})", flush=True)

    # Test 3: Add siglip2 protos
    if os.path.exists('outputs/inat_protos_siglip2.pt') and os.path.exists('outputs/emb_train_siglip2.pt'):
        print("Testing + siglip2 protos...", flush=True)
        sigi, sigf, _ = load_emb('outputs/emb_train_siglip2.pt')
        Qsig, _ = queries(D, sigi, sigf)
        Ssig = proto_leg(Qsig, 'outputs/inat_protos_siglip2.pt', cand)
        for w_sig in [0.5, 1.0, 1.5, 2.0]:
            acc = top1(base_v56 + w_sig * Ssig, gold)
            print(f"  + w_sig={w_sig}: {acc:.4f} (diff: {acc - acc_v56:+.4f})", flush=True)

    # Test 4: Add dinov2 protos
    if os.path.exists('outputs/inat_protos_dinov2_vitl14.pt') and os.path.exists('outputs/emb_train_dino.pt'):
        print("Testing + dinov2 protos...", flush=True)
        dinoi, dinof, _ = load_emb('outputs/emb_train_dino.pt')
        Qdino, _ = queries(D, dinoi, dinof)
        Sdino = proto_leg(Qdino, 'outputs/inat_protos_dinov2_vitl14.pt', cand)
        for w_dino in [0.5, 1.0, 1.5, 2.0]:
            acc = top1(base_v56 + w_dino * Sdino, gold)
            print(f"  + w_dino={w_dino}: {acc:.4f} (diff: {acc - acc_v56:+.4f})", flush=True)

if __name__ == '__main__':
    test_unseen_bank_expansion()
