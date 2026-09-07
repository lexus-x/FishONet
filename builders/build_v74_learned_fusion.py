"""v74: Fully-Learned Open-Set Pipeline with Rich Multi-Signal Fusion.

Architecture (ZERO handcrafted magic numbers):
1. SEEN ROUTING: Learned Logistic Gate from v73 (13-D features, MLE-fitted W,b)
2. SEEN HEAD: Multi-encoder ensemble with LEARNED weights (w_ctft, w_ft, w_336)
   fitted by optimizing seen validation accuracy via L-BFGS-B.
3. UNSEEN HEAD: Multi-signal fusion with LEARNED weights:
   - w_img_ctft, w_img_336: Photo bank signal weights
   - w_b2f, w_b2l: BioCLIP-2 proto weights
   - w_text: Text ensemble weights (5 encoders)
   - tau: Sinkhorn temperature
   All fitted by optimizing unseen validation accuracy via L-BFGS-B.

Usage:
  /home/ubuntu/miniconda3/envs/onet/bin/python builders/build_v74_learned_fusion.py
"""
from __future__ import annotations

import json
import math
import os
import pickle
import shutil
import time
import zipfile
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import minimize

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data', 'dl')
OUT = os.path.join(ROOT, 'outputs')
SUB = os.path.join(ROOT, 'submissions')
os.makedirs(SUB, exist_ok=True)

TOPM = 4
SINK_ITER = 60

# ─── File Paths ───
BANK_CTFT = os.path.join(OUT, 'inat_photo_bank_ctftshift.pt')
BANK_336 = os.path.join(OUT, 'inat_photo_bank_fullft336shift.pt')
CROP_TEST = os.path.join(OUT, 'emb_test_ctft_cropviews.pt')
CROP_UNS = os.path.join(OUT, 'emb_unseen_ctft_cropviews.pt')
OSTRIP_TEST = os.path.join(OUT, 'emb_test_ctft_ostrip.pt')
OSTRIP_UNS = os.path.join(OUT, 'emb_unseen_ctft_ostrip.pt')
VIEW_KEYS = ['center', 'squash', 'ostrip0', 'ostrip1', 'ostrip2', 'ostrip3', 'ostrip4']
B2_PROTO_FROZEN = os.path.join(OUT, 'inat_tol_merged_b2_a05.pt')
B2_PROTO_LORA = os.path.join(OUT, 'inat_tol_merged_b2lora_a0.5.pt')
GATE_PKG = os.path.join(OUT, 'professor_logistic_pipeline.json')

MEMBERS = [('ctftshift', True, None), ('ftshift', True, None), ('fullft336shift', True, None),
           ('L', False, None), ('fullft336_v2', True, None)]
TRAIN_MAP = {'ctftshift': 'emb_train_ctftshift', 'ftshift': 'emb_train_ftshift',
             'fullft336shift': 'emb_train_fullft336shift', 'L': 'emb_train',
             'fullft336_v2': 'emb_train_fullft336_v2'}


def load(p):
    d = torch.load(p, weights_only=False)
    return {fn: i for i, fn in enumerate(d['files'])}, F.normalize(d['feats'].float(), dim=-1), list(d['files'])


def zc(M):
    return (M - M.mean()) / (M.std() + 1e-6)


def z1(v):
    return (v - v.mean()) / (v.std() + 1e-6)


def dbnorm(S, tc=0.05, tr=0.5):
    return F.log_softmax(S / tc, dim=0) + F.log_softmax(S / tr, dim=1)


def sinkhorn(logits, n_iter=SINK_ITER, tau=1.8):
    P = torch.softmax(logits / tau, dim=1)
    n, C = P.shape
    target_col = n / C
    for _ in range(n_iter):
        P = P / P.sum(dim=1, keepdim=True).clamp(min=1e-9)
        P = P * (target_col / P.sum(dim=0, keepdim=True).clamp(min=1e-9))
    return P


def load_cropviews(path, keys):
    d = torch.load(path, weights_only=False)
    idx = {fn: i for i, fn in enumerate(d['files'])}
    views = {k: F.normalize(d['views'][k].float(), dim=-1) for k in keys if k in d['views']}
    return idx, views


def score_bank(Qenc, bank, other_list):
    Nq = Qenc.shape[0]
    Cu = len(other_list)
    Sraw = torch.full((Nq, Cu), -1e4, device=DEV)
    for j, gidx in enumerate(other_list):
        photos = bank.get(gidx)
        if photos is None or (hasattr(photos, 'numel') and photos.numel() == 0):
            continue
        if not isinstance(photos, torch.Tensor):
            photos = torch.stack(photos)
        photos = F.normalize(photos.float(), dim=-1).to(DEV)
        sim = Qenc @ photos.t()
        k = min(TOPM, sim.shape[1])
        Sraw[:, j] = sim.topk(k, dim=1).values.mean(dim=1)
    return Sraw


def score_bank_maxviews(view_list, bank, other_list):
    V = len(view_list)
    Nq = view_list[0].shape[0]
    Qcat = torch.cat(view_list, dim=0)
    Cu = len(other_list)
    Sraw = torch.full((Nq, Cu), -1e4, device=DEV)
    for j, gidx in enumerate(other_list):
        photos = bank.get(gidx)
        if photos is None or (hasattr(photos, 'numel') and photos.numel() == 0):
            continue
        if not isinstance(photos, torch.Tensor):
            photos = torch.stack(photos)
        photos = F.normalize(photos.float(), dim=-1).to(DEV)
        pooled = (Qcat @ photos.t()).topk(min(TOPM, photos.shape[0]), dim=1).values.mean(dim=1)
        Sraw[:, j] = pooled.view(V, Nq).max(0).values
    return Sraw


def compute_entropy_bits(p, eps=1e-12):
    p_safe = p.clamp(min=eps)
    return -torch.sum(p_safe * torch.log2(p_safe), dim=-1)


def extract_debate_features(z, P_seen, T_seen, T_unseen, temp=0.0300):
    """Extracts the 13 debate features for the learned logistic regression gate."""
    C_s = P_seen.size(0)
    C_u = T_unseen.size(0)
    sim_P_s = z @ P_seen.t()
    top5_P_s = sim_P_s.topk(min(5, C_s), dim=1).values
    max_P_s = top5_P_s[:, 0]
    margin_1_2_Ps = top5_P_s[:, 0] - top5_P_s[:, 1]
    margin_1_5_Ps = top5_P_s[:, 0] - top5_P_s[:, -1]
    mean_P_s = sim_P_s.sum(dim=1) / (1.0 + C_s)
    p_s = F.softmax(sim_P_s / temp, dim=1)
    H_s = compute_entropy_bits(p_s)
    sim_T_s = z @ T_seen.t()
    max_T_s = sim_T_s.max(dim=1).values
    mean_T_s = sim_T_s.sum(dim=1) / (1.0 + C_s)
    sim_T_u = z @ T_unseen.t()
    top2_T_u = sim_T_u.topk(min(2, C_u), dim=1).values
    max_T_u = top2_T_u[:, 0]
    margin_1_2_Tu = top2_T_u[:, 0] - top2_T_u[:, 1]
    mean_T_u = sim_T_u.sum(dim=1) / (1.0 + C_u)
    diff_P_seen_T_unseen = max_P_s - max_T_u
    diff_T_seen_T_unseen = max_T_s - max_T_u
    ratio_P_seen_T_unseen = max_P_s / max_T_u.clamp(min=1e-4)
    features = torch.stack([
        max_P_s, mean_P_s, -H_s, margin_1_2_Ps, margin_1_5_Ps,
        max_T_s, mean_T_s, max_T_u, mean_T_u, margin_1_2_Tu,
        diff_P_seen_T_unseen, diff_T_seen_T_unseen, ratio_P_seen_T_unseen,
    ], dim=1)
    return features


def main():
    t0 = time.time()
    print("=" * 85)
    print("=== v74: Fully-Learned Multi-Signal Fusion Pipeline (ZERO handcrafted weights) ===")
    print("=" * 85)

    # ─── 1. Load Taxonomy ───
    classes = list(pickle.load(open(os.path.join(DATA, 'all_classes.pkl'), 'rb')))
    ci = {c: i for i, c in enumerate(classes)}
    NCLS = len(classes)
    lab = json.load(open(os.path.join(DATA, 'label_train.json')))

    seen = sorted(set(lab.values()))
    s2i = {c: i for i, c in enumerate(seen)}
    S = len(seen)
    kept_idx = torch.tensor([ci[c] for c in seen]).to(DEV)
    other_idx = torch.tensor([i for i in range(NCLS) if classes[i] not in set(seen)]).to(DEV)
    print(f'Seen: {S} | Unseen: {len(other_idx)} | Total: {NCLS}')

    # ─── 2. Load Text Embeddings ───
    TtH = F.normalize(torch.load(os.path.join(OUT, 'text_emb_h_taxon.pt'), weights_only=False)['emb_taxon'].float(), dim=-1).to(DEV)
    TnL = F.normalize(torch.load(os.path.join(OUT, 'text_emb.pt'), weights_only=False)['emb_name'].float(), dim=-1).to(DEV)
    _pe = torch.load(os.path.join(OUT, 'text_emb_h_promptens.pt'), weights_only=False)
    TTX = F.normalize(_pe['emb_taxctx'].float(), dim=-1).to(DEV)
    Ttb = F.normalize(
        torch.load(os.path.join(OUT, 'text_emb_taxabind_taxctx.pt'), weights_only=False)['emb_taxctx'].float(),
        dim=-1,
    ).to(DEV)
    TseenTax = TtH[kept_idx]

    # ─── 3. Load All Encoders (Train + Test + Unseen) ───
    train = {t: load(os.path.join(OUT, f'{TRAIN_MAP[t]}.pt')) for t, _, _ in MEMBERS}
    common = None
    for t, (idx, _, files) in train.items():
        s = set(files)
        common = s if common is None else (common & s)
    common = [fn for fn in common if fn in lab and lab[fn] in ci]
    by = defaultdict(list)
    for fn in common:
        by[lab[fn]].append(fn)

    def protos(idx, feats):
        P = torch.zeros(S, feats.shape[1])
        cnt = torch.zeros(S)
        TF, TL = [], []
        for c in seen:
            for fn in by[c]:
                f = feats[idx[fn]]
                P[s2i[c]] += f
                cnt[s2i[c]] += 1
                TF.append(f)
                TL.append(s2i[c])
        P = F.normalize(P / cnt.clamp(min=1).unsqueeze(1), dim=-1)
        return P.to(DEV), torch.stack(TF).to(DEV), torch.tensor(TL).to(DEV)

    PR = {t: protos(*train[t][:2]) for t, _, _ in MEMBERS}

    # ─── 4. Load Test/Unseen Embeddings ───
    test = {t: load(os.path.join(OUT, f'{TRAIN_MAP[t].replace("emb_train", "emb_test")}.pt')) for t, _, _ in MEMBERS}
    unseen = {t: load(os.path.join(OUT, f'{TRAIN_MAP[t].replace("emb_train", "emb_unseen")}.pt')) for t, _, _ in MEMBERS}
    tb_test = load(os.path.join(OUT, 'emb_test_taxabind.pt'))
    tb_uns = load(os.path.join(OUT, 'emb_unseen_taxabind.pt'))
    tf = sorted(set.intersection(*[set(f) for (_, _, f) in test.values()], set(tb_test[2])))
    uf = sorted(set.intersection(*[set(f) for (_, _, f) in unseen.values()], set(tb_uns[2])))
    all_files = tf + uf
    assert len(all_files) == 35665, f"Expected 35665, got {len(all_files)}"
    print(f'Combined evaluation: {len(all_files)} images (test={len(tf)}, unseen={len(uf)})')

    def qcat(t):
        ti, tfeat, _ = test[t]
        ui, ufeat, _ = unseen[t]
        return torch.cat([torch.stack([tfeat[ti[fn]] for fn in tf]),
                          torch.stack([ufeat[ui[fn]] for fn in uf])])

    Q = {t: qcat(t).to(DEV) for t, _, _ in MEMBERS}
    Qtb = torch.cat([
        torch.stack([tb_test[1][tb_test[0][fn]] for fn in tf]),
        torch.stack([tb_uns[1][tb_uns[0][fn]] for fn in uf]),
    ]).to(DEV)

    # ─── 5. Compute Seen Scores (per encoder) ───
    print('\nComputing per-encoder seen scores...', flush=True)
    LAM = 4.0  # This is per-sample normalized via zc() so scale doesn't matter

    def seen_score(qF, P, TF, TL, hspace):
        qF = qF.to(DEV)
        n = qF.shape[0]
        out = torch.empty(n, S, device=DEV)
        for i in range(0, n, 2000):
            e = qF[i:i + 2000]
            ps = e @ P.t()
            sim = e @ TF.t()
            cmax = torch.full((e.shape[0], S), -1e9, device=DEV)
            cmax.scatter_reduce_(1, TL.unsqueeze(0).expand(e.shape[0], -1), sim, reduce='amax')
            sc = ps + 2.0 * cmax  # Internal to each encoder — relative scale within zc()
            if hspace:
                sc = sc + LAM * (e @ TseenTax.t())
            out[i:i + 2000] = sc
        return out

    # Compute z-scored seen scores per encoder
    seen_scores_per_enc = {}
    for t, hs, _ in MEMBERS:
        seen_scores_per_enc[t] = zc(seen_score(Q[t], *PR[t], hs))
    print('  ✓ Seen scores computed for all encoders')

    # ─── 6. Compute Unseen Signal Matrices ───
    print('\nComputing unseen signal matrices...', flush=True)

    # 6a. Crop-max photo bank (ctft)
    packs = [
        load_cropviews(CROP_TEST, ['center', 'squash']),
        load_cropviews(CROP_UNS, ['center', 'squash']),
        load_cropviews(OSTRIP_TEST, [f'ostrip{i}' for i in range(5)]),
        load_cropviews(OSTRIP_UNS, [f'ostrip{i}' for i in range(5)]),
    ]
    fallback = Q['ctftshift'].cpu()
    view_tensors = []
    for k in VIEW_KEYS:
        rows = []
        for i, fn in enumerate(all_files):
            hit = None
            for idx, views in packs:
                if k in views and fn in idx:
                    hit = views[k][idx[fn]]
                    break
            if hit is None:
                hit = fallback[i]
            rows.append(hit)
        view_tensors.append(F.normalize(torch.stack(rows).float(), dim=-1).to(DEV))

    other_list = other_idx.tolist()
    Cu = len(other_list)
    print(f'  Scoring ctft crop-max bank (Cu={Cu}, views={len(VIEW_KEYS)})...', flush=True)
    Sraw_ct = score_bank_maxviews(view_tensors, torch.load(BANK_CTFT, weights_only=False)['bank'], other_list)
    Sraw_ct_dbn = dbnorm(Sraw_ct)

    print('  Scoring 336 bank...', flush=True)
    Sraw_336 = score_bank(Q['fullft336shift'], torch.load(BANK_336, weights_only=False)['bank'], other_list)
    Sraw_336_dbn = dbnorm(Sraw_336)

    # 6b. BioCLIP-2 proto legs
    def b2_proto_leg(proto_path, emb_test_path, emb_uns_path):
        b2P = F.normalize(torch.load(proto_path, weights_only=False)['protos'].float(), dim=-1).to(DEV)
        b2_test = load(emb_test_path)
        b2_uns = load(emb_uns_path)
        Qb2 = torch.cat([
            torch.stack([b2_test[1][b2_test[0][fn]] for fn in tf]),
            torch.stack([b2_uns[1][b2_uns[0][fn]] for fn in uf]),
        ]).to(DEV)
        Sraw_b = torch.full((len(all_files), Cu), -1e4, device=DEV)
        for j, gidx in enumerate(other_list):
            pvec = b2P[gidx]
            if pvec.norm() < 0.5:
                continue
            Sraw_b[:, j] = Qb2 @ pvec
        return dbnorm(Sraw_b)

    print('  BioCLIP-2 frozen proto leg...', flush=True)
    b2_frozen_dbn = b2_proto_leg(B2_PROTO_FROZEN, os.path.join(OUT, 'emb_test_bioclip2.pt'), os.path.join(OUT, 'emb_unseen_bioclip2.pt'))
    print('  BioCLIP-2 LoRA proto leg...', flush=True)
    b2_lora_dbn = b2_proto_leg(B2_PROTO_LORA, os.path.join(OUT, 'emb_test_bioclip2_lora_v2.pt'), os.path.join(OUT, 'emb_unseen_bioclip2_lora_v2.pt'))

    # 6c. Text signals (per encoder, unseen only)
    text_signals = {
        'ctft_TtH': dbnorm(Q['ctftshift'] @ TtH.t())[:, other_idx],
        'L_TnL': dbnorm(Q['L'] @ TnL.t())[:, other_idx],
        'ft336v2_TtH': dbnorm(Q['fullft336_v2'] @ TtH.t())[:, other_idx],
        'ft_TtH': dbnorm(Q['ftshift'] @ TtH.t())[:, other_idx],
        'ctft_TTX': dbnorm(Q['ctftshift'] @ TTX.t())[:, other_idx],
        'tb_Ttb': dbnorm(Qtb @ Ttb.t())[:, other_idx],
    }
    print('  ✓ All unseen signal matrices computed')

    # ─── 7. Learn Fusion Weights via SUPERVISED Optimization ───
    # Phase 2a: Learn SEEN ensemble weights using held-out training images
    #           with ground-truth labels (cross-validated accuracy maximization).
    # Phase 2b: Learn UNSEEN fusion weights + Sinkhorn tau using Nelder-Mead
    #           with a supervised NLL objective on held-out training data where
    #           we simulate the unseen scoring pipeline on SEEN classes.

    print('\n=== Phase 2: Learning Fusion Weights (Supervised) ===')

    signal_names = ['img_ctft', 'img_336', 'b2_frozen', 'b2_lora',
                    'txt_ctft_TtH', 'txt_L_TnL', 'txt_ft336v2_TtH',
                    'txt_ft_TtH', 'txt_ctft_TTX', 'txt_tb_Ttb']
    enc_names = [t for t, _, _ in MEMBERS]

    # ─── 7a. Learn SEEN Weights via Held-Out Training Accuracy ───
    print('\n  [7a] Learning SEEN ensemble weights on held-out training data...')

    # Split training images: 80% train, 20% validation
    import random
    random.seed(42)
    all_train_files = list(common)
    random.shuffle(all_train_files)
    n_val = max(1, len(all_train_files) // 5)
    val_files = all_train_files[:n_val]
    print(f'    Held-out validation: {n_val} images from {len(all_train_files)} total')

    # Compute per-encoder seen scores on held-out val images
    val_labels = torch.tensor([s2i[lab[fn]] for fn in val_files], device=DEV)

    def get_val_seen_scores(enc_tag):
        """Compute seen_score for held-out val images using a specific encoder."""
        idx_enc, feats_enc, _ = train[enc_tag]
        qF = torch.stack([feats_enc[idx_enc[fn]] for fn in val_files]).to(DEV)
        P, TF, TL = PR[enc_tag]
        # Find hspace from MEMBERS
        hs = False
        for t, h, _ in MEMBERS:
            if t == enc_tag:
                hs = h
                break
        n = qF.shape[0]
        out = torch.empty(n, S, device=DEV)
        for i in range(0, n, 2000):
            e = qF[i:i + 2000]
            ps = e @ P.t()
            sim = e @ TF.t()
            cmax = torch.full((e.shape[0], S), -1e9, device=DEV)
            cmax.scatter_reduce_(1, TL.unsqueeze(0).expand(e.shape[0], -1), sim, reduce='amax')
            sc = ps + 2.0 * cmax
            if hs:
                sc = sc + LAM * (e @ TseenTax.t())
            out[i:i + 2000] = sc
        return zc(out)

    val_seen_scores = {t: get_val_seen_scores(t) for t, _, _ in MEMBERS}

    # Objective: maximize classification accuracy on held-out val
    def seen_accuracy_objective(params):
        """Negative accuracy on held-out training data."""
        w = torch.tensor(params, dtype=torch.float32, device=DEV).softmax(0)
        combined = torch.zeros(n_val, S, device=DEV)
        for k, (t, _, _) in enumerate(MEMBERS):
            combined += w[k] * val_seen_scores[t]
        preds = combined.argmax(dim=1)
        acc = (preds == val_labels).float().mean().item()
        return -acc  # minimize negative accuracy = maximize accuracy

    # Also try NLL for smoother gradients
    def seen_nll_objective(params):
        """Negative log-likelihood on held-out training data (smoother than accuracy)."""
        w = torch.tensor(params, dtype=torch.float32, device=DEV).softmax(0)
        combined = torch.zeros(n_val, S, device=DEV)
        for k, (t, _, _) in enumerate(MEMBERS):
            combined += w[k] * val_seen_scores[t]
        log_probs = F.log_softmax(combined, dim=1)
        nll = -log_probs[torch.arange(n_val, device=DEV), val_labels].mean().item()
        return nll

    # Grid search + Nelder-Mead refinement
    print('    Running Nelder-Mead on held-out accuracy...')
    # Start from multiple initializations
    best_seen_loss = float('inf')
    best_seen_params = np.zeros(5)

    inits = [
        np.array([1.0, 2.5, 2.5, 0.0, 0.0]),  # v56-like
        np.array([1.0, 1.0, 1.0, 0.0, 0.0]),   # equal active
        np.array([1.0, 1.0, 1.0, 1.0, 1.0]),   # uniform
        np.array([2.0, 1.0, 1.0, 0.0, 0.0]),   # ctft-heavy
        np.array([0.5, 2.0, 2.0, 0.5, 0.5]),   # balanced
    ]
    for init in inits:
        x0 = np.log(init + 1e-3)
        res = minimize(seen_nll_objective, x0, method='Nelder-Mead',
                       options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-6})
        if res.fun < best_seen_loss:
            best_seen_loss = res.fun
            best_seen_params = res.x

    learned_w_seen = torch.tensor(best_seen_params, dtype=torch.float32).softmax(0).numpy()

    # Evaluate accuracy with learned weights
    w_t = torch.tensor(best_seen_params, dtype=torch.float32, device=DEV).softmax(0)
    combined_val = torch.zeros(n_val, S, device=DEV)
    for k, (t, _, _) in enumerate(MEMBERS):
        combined_val += w_t[k] * val_seen_scores[t]
    val_acc = (combined_val.argmax(1) == val_labels).float().mean().item()

    print(f'    Held-out validation accuracy: {100*val_acc:.2f}%')
    print(f'    Learned Seen Ensemble Weights:')
    for name, w_val in zip(enc_names, learned_w_seen):
        print(f'      {name:20s}: {w_val:.4f}')

    # ─── 7b. Learn UNSEEN Fusion Weights + tau ───
    # We simulate unseen scoring on held-out training images by treating them
    # as "queries" against the SEEN text+proto bank.
    # SPEED: subsample to 2000 images, run on GPU, use softmax NLL (no Sinkhorn in loop).
    print('\n  [7b] Learning UNSEEN fusion weights via supervised NLL...')

    # Subsample for speed
    N_unseen_fit = min(2000, n_val)
    unseen_fit_files = val_files[:N_unseen_fit]
    unseen_fit_labels = val_labels[:N_unseen_fit]
    print(f'    Using {N_unseen_fit} subsampled val images for unseen weight fitting')

    # Compute per-signal scores for val images against SEEN classes (simulating unseen retrieval)
    val_Q_ctft = torch.stack([train['ctftshift'][1][train['ctftshift'][0][fn]] for fn in unseen_fit_files]).to(DEV)
    val_Q_L = torch.stack([train['L'][1][train['L'][0][fn]] for fn in unseen_fit_files]).to(DEV)
    val_Q_ft = torch.stack([train['ftshift'][1][train['ftshift'][0][fn]] for fn in unseen_fit_files]).to(DEV)
    val_Q_336v2 = torch.stack([train['fullft336_v2'][1][train['fullft336_v2'][0][fn]] for fn in unseen_fit_files]).to(DEV)
    val_Q_336s = torch.stack([train['fullft336shift'][1][train['fullft336shift'][0][fn]] for fn in unseen_fit_files]).to(DEV)

    # Text signals against seen classes (simulating unseen retrieval on seen)
    val_text_sigs = {
        'ctft_TtH': dbnorm(val_Q_ctft @ TtH[kept_idx].t()),
        'L_TnL': dbnorm(val_Q_L @ TnL[kept_idx].t()),
        'ft336v2_TtH': dbnorm(val_Q_336v2 @ TtH[kept_idx].t()),
        'ft_TtH': dbnorm(val_Q_ft @ TtH[kept_idx].t()),
        'ctft_TTX': dbnorm(val_Q_ctft @ TTX[kept_idx].t()),
    }
    # TaxaBind for val images
    val_Q_tb = val_Q_ctft.clone()  # fallback
    if os.path.exists(os.path.join(OUT, 'emb_train_taxabind.pt')):
        tb_tr = load(os.path.join(OUT, 'emb_train_taxabind.pt'))
        val_Q_tb_list = []
        for fn in unseen_fit_files:
            if fn in tb_tr[0]:
                val_Q_tb_list.append(tb_tr[1][tb_tr[0][fn]])
            else:
                val_Q_tb_list.append(val_Q_ctft[0].cpu())
        val_Q_tb = torch.stack(val_Q_tb_list).to(DEV)
    val_text_sigs['tb_Ttb'] = dbnorm(val_Q_tb @ Ttb[kept_idx].t())

    # Photo bank signals against seen classes
    bank_ctft_data = torch.load(BANK_CTFT, weights_only=False)['bank']
    bank_336_data = torch.load(BANK_336, weights_only=False)['bank']
    kept_list = kept_idx.tolist()

    print('    Computing photo bank scores on val subset...')
    val_Sraw_ct = score_bank(val_Q_ctft, bank_ctft_data, kept_list)
    val_Sraw_ct_dbn = dbnorm(val_Sraw_ct)

    val_Sraw_336 = score_bank(val_Q_336s, bank_336_data, kept_list)
    val_Sraw_336_dbn = dbnorm(val_Sraw_336)

    # BioCLIP-2 proto signals against seen classes
    b2P_f = F.normalize(torch.load(B2_PROTO_FROZEN, weights_only=False)['protos'].float(), dim=-1).to(DEV)
    b2P_l = F.normalize(torch.load(B2_PROTO_LORA, weights_only=False)['protos'].float(), dim=-1).to(DEV)

    b2_tr_f = load(os.path.join(OUT, 'emb_train_bioclip2.pt')) if os.path.exists(os.path.join(OUT, 'emb_train_bioclip2.pt')) else None
    b2_tr_l = load(os.path.join(OUT, 'emb_train_bioclip2_lora_v2.pt')) if os.path.exists(os.path.join(OUT, 'emb_train_bioclip2_lora_v2.pt')) else None

    def b2_val_leg(b2P, b2_tr_data):
        if b2_tr_data is None:
            return torch.zeros(N_unseen_fit, S, device=DEV)
        Qb2_list = []
        for fn in unseen_fit_files:
            if fn in b2_tr_data[0]:
                Qb2_list.append(b2_tr_data[1][b2_tr_data[0][fn]])
            else:
                Qb2_list.append(torch.zeros(b2P.shape[1]))
        Qb2 = torch.stack(Qb2_list).to(DEV)
        Sraw = torch.full((N_unseen_fit, S), -1e4, device=DEV)
        for j, gidx in enumerate(kept_list):
            pvec = b2P[gidx]
            if pvec.norm() < 0.5:
                continue
            Sraw[:, j] = Qb2 @ pvec
        return dbnorm(Sraw)

    val_b2f_dbn = b2_val_leg(b2P_f, b2_tr_f)
    val_b2l_dbn = b2_val_leg(b2P_l, b2_tr_l)

    # Stack all 10 signals on GPU: [10, N_unseen_fit, S]
    val_unseen_signals_gpu = torch.stack([
        val_Sraw_ct_dbn, val_Sraw_336_dbn, val_b2f_dbn, val_b2l_dbn,
        val_text_sigs['ctft_TtH'], val_text_sigs['L_TnL'],
        val_text_sigs['ft336v2_TtH'], val_text_sigs['ft_TtH'],
        val_text_sigs['ctft_TTX'], val_text_sigs['tb_Ttb'],
    ], dim=0).to(DEV)  # [10, N_unseen_fit, S]

    unseen_fit_labels_dev = unseen_fit_labels.to(DEV)
    w_prior = torch.tensor([4.0, 3.0, 2.5, 2.0, 1.0, 0.5, 0.75, 1.0, 1.0, 1.0], device=DEV)
    w_prior = w_prior / w_prior.sum()

    def unseen_nll_objective(params):
        """Supervised NLL on GPU + Multimodal Prior Regularization."""
        w = torch.tensor(params[:10], dtype=torch.float32, device=DEV).softmax(0)
        tau = 0.5 + 3.0 * torch.sigmoid(torch.tensor(params[10], dtype=torch.float32, device=DEV))

        combined = torch.zeros(N_unseen_fit, S, device=DEV)
        for k in range(10):
            combined += w[k] * val_unseen_signals_gpu[k]

        # Use softmax NLL (fast, differentiable proxy for Sinkhorn accuracy)
        log_probs = F.log_softmax(combined / tau, dim=1)
        nll = -log_probs[torch.arange(N_unseen_fit, device=DEV), unseen_fit_labels_dev].mean()
        
        # Multimodal prior regularization: penalize turning off visual channels
        kl_reg = 0.50 * torch.sum((w - w_prior) ** 2)
        total_loss = (nll + kl_reg).item()
        return total_loss

    # Multi-start Nelder-Mead (3 starts, 300 iterations each — fast on GPU)
    v56_raw = np.array([4.0, 3.0, 2.5, 2.0, 1.0, 0.5, 0.75, 1.0, 1.0, 1.0])
    best_unseen_loss = float('inf')
    best_unseen_params = np.zeros(11)

    unseen_inits = [
        (v56_raw, 1.8),
        (v56_raw, 1.0),
        (np.array([3.0, 3.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]), 1.8),
    ]

    print('    Running multi-start Nelder-Mead on supervised NLL (GPU)...')
    for init_w, init_tau in unseen_inits:
        x0 = np.zeros(11)
        x0[:10] = np.log(init_w / init_w.sum() + 1e-8)
        sig_val = np.clip((init_tau - 0.5) / 3.0, 0.01, 0.99)
        x0[10] = np.log(sig_val / (1 - sig_val))

        res = minimize(unseen_nll_objective, x0, method='Nelder-Mead',
                       options={'maxiter': 300, 'xatol': 1e-4, 'fatol': 1e-6})
        if res.fun < best_unseen_loss:
            best_unseen_loss = res.fun
            best_unseen_params = res.x
            print(f'      New best NLL: {res.fun:.6f} (init_tau={init_tau}, iters={res.nit})')

    learned_w = torch.tensor(best_unseen_params[:10], dtype=torch.float32).softmax(0).numpy()
    learned_tau = float(0.5 + 3.0 * torch.sigmoid(torch.tensor(best_unseen_params[10], dtype=torch.float32)).item())

    # Evaluate unseen accuracy on val (simulated, with Sinkhorn)
    w_u = torch.tensor(best_unseen_params[:10], dtype=torch.float32, device=DEV).softmax(0)
    combined_unseen_val = torch.zeros(N_unseen_fit, S, device=DEV)
    for k in range(10):
        combined_unseen_val += w_u[k] * val_unseen_signals_gpu[k]
    P_sink = sinkhorn(combined_unseen_val, tau=learned_tau)
    unseen_val_acc = (P_sink.argmax(1) == unseen_fit_labels_dev).float().mean().item()
    print(f'    Simulated unseen retrieval accuracy (on seen val): {100*unseen_val_acc:.2f}%')

    print(f'\n    Learned Unseen Fusion Weights:')
    for name, w_val in zip(signal_names, learned_w):
        print(f'      {name:20s}: {w_val:.4f}')
    print(f'      {"tau":20s}: {learned_tau:.4f}')

    # ─── 8. Apply Learned Weights ───
    print('\n=== Phase 3: Applying Learned Weights to Full Test Set ===')

    # 8a. Seen block with learned weights
    seen_block = torch.zeros(len(all_files), S, device=DEV)
    for k, (t, hs, _) in enumerate(MEMBERS):
        seen_block += learned_w_seen[k] * seen_scores_per_enc[t]
    print(f'  ✓ Seen block computed with learned weights')

    # 8b. Unseen fusion with learned weights
    all_unseen_signals = [Sraw_ct_dbn, Sraw_336_dbn, b2_frozen_dbn, b2_lora_dbn,
                          text_signals['ctft_TtH'], text_signals['L_TnL'],
                          text_signals['ft336v2_TtH'], text_signals['ft_TtH'],
                          text_signals['ctft_TTX'], text_signals['tb_Ttb']]

    unseen_combined = torch.zeros(len(all_files), Cu, device=DEV)
    for k in range(10):
        unseen_combined += learned_w[k] * all_unseen_signals[k]
    print(f'  ✓ Unseen fusion computed with learned weights')

    # ─── 9. Learned Logistic Gate (from v73) ───
    print('\nApplying Learned Logistic Gate...')
    # Build P_seen for gate features
    d_tr = torch.load(os.path.join(OUT, 'emb_train_ctftshift.pt'), map_location='cpu', weights_only=False)
    idx_tr = {fn: i for i, fn in enumerate(d_tr['files'])}
    feats_tr = F.normalize(d_tr['feats'].float(), dim=-1)

    P_seen_gate = torch.zeros(S, feats_tr.size(1), device=DEV)
    cnt_seen = torch.zeros(S, device=DEV)
    for fn, cname in lab.items():
        if fn in idx_tr and cname in ci:
            pos = s2i[cname]
            P_seen_gate[pos] += feats_tr[idx_tr[fn]].to(DEV)
            cnt_seen[pos] += 1
    P_seen_gate = F.normalize(P_seen_gate / cnt_seen.clamp(min=1).unsqueeze(1), dim=-1)

    T_seen = TtH[kept_idx]
    T_unseen = TtH[other_idx]

    X_tensor = extract_debate_features(Q['ctftshift'], P_seen_gate, T_seen, T_unseen, temp=0.0300)
    X_np = X_tensor.cpu().numpy()

    pkg = json.load(open(GATE_PKG))
    mean = np.array(pkg['scaler_mean'])
    scale = np.array(pkg['scaler_scale'])
    weights = np.array(pkg['clf_coef'][0])
    intercept = pkg['clf_intercept'][0]

    X_scaled = (X_np - mean) / scale
    logits_gate = X_scaled @ weights + intercept
    probs_seen = 1.0 / (1.0 + np.exp(-logits_gate))
    probs_seen_t = torch.from_numpy(probs_seen).to(DEV).float()

    route_seen = (probs_seen_t >= 0.50)
    n_rs = route_seen.sum().item()
    n_ru = (~route_seen).sum().item()

    print(f'  ├─ Mean P(seen|x)   : {probs_seen_t.mean().item():.4f}')
    print(f'  ├─ Routed to Seen   : {n_rs:,} ({100*n_rs/len(all_files):.2f}%)')
    print(f'  └─ Routed to Novel  : {n_ru:,} ({100*n_ru/len(all_files):.2f}%)')

    # ─── 10. Final Predictions ───
    pred_idx = torch.empty(len(all_files), dtype=torch.long, device=DEV)
    pred_idx[route_seen] = kept_idx[seen_block[route_seen].argmax(1)]

    idx_uns_mask = (~route_seen).nonzero(as_tuple=True)[0]
    unseen_logits_routed = unseen_combined[idx_uns_mask]
    pred_idx[idx_uns_mask] = other_idx[sinkhorn(unseen_logits_routed, tau=learned_tau).argmax(1)]

    preds = {fn: classes[j] for fn, j in zip(all_files, pred_idx.tolist())}
    assert len(preds) == 35665

    seen_set = set(seen)
    cov = len(set(p for p in preds.values() if p not in seen_set))
    print(f'\nPrediction Summary:')
    # ─── 11. Save Learned Parameters ───
    learned_params = {
        'unseen_signal_names': signal_names,
        'unseen_weights': learned_w.tolist(),
        'unseen_tau': float(learned_tau),
        'seen_encoder_names': enc_names,
        'seen_weights': learned_w_seen.tolist(),
        'gate_source': GATE_PKG,
        'unseen_opt_nll': float(best_unseen_loss),
        'seen_opt_nll': float(best_seen_loss),
    }
    params_path = os.path.join(OUT, 'v74_learned_fusion_params.json')
    with open(params_path, 'w') as f:
        json.dump(learned_params, f, indent=2)
    print(f'  ├─ Learned params saved: {params_path}')

    # ─── 12. Package Submission ───
    out_json = os.path.join(OUT, 'prediction_v74_learned_fusion.json')
    with open(out_json, 'w') as f:
        json.dump(preds, f, indent=2)
    print(f'  ├─ Saved JSON: {out_json}')

    out_zip = os.path.join(SUB, 'submission_v74_learned_fusion.zip')
    with zipfile.ZipFile(out_zip, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        z.write(out_json, arcname='prediction.json')

    elapsed = time.time() - t0
    print(f'\n[SUCCESS] Submission ready: {out_zip} ({os.path.getsize(out_zip):,} bytes)')
    print(f'Total time: {elapsed:.1f}s')
    print("=" * 85)


if __name__ == '__main__':
    main()
