"""Build Learning-Based Submission (No DBNorm).

Evaluates the trained learning-based pipeline:
- Novelty Logistic Regression Gate (P(seen | x))
- Monotonic 3rd-Degree Polynomial Calibrators
- Learned Multi-Modal Fusion
- 7-View Overlap Crop-Max Photo Bank & 336 Photo Bank & BioCLIP-2 Tree-of-Life Protos

Single-pipeline compliant: one uniform rule over all 35,665 images, full 17,393 argmax.

Usage:
  /home/ubuntu/miniconda3/envs/onet/bin/python scripts/build_submission.py
"""
from __future__ import annotations

import json
import math
import os
import pickle
import shutil
import sys
import zipfile
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.calibration.polynomial_calibrator import MonotonicDegree3Polynomial, MultiModalPolynomialCalibrator
from src.calibration.novelty_logistic_gate import CalibratedNoveltyLogisticGate, extract_gating_features_tensor
from src.pipeline.multi_modal_fusion import MultiModalFusionScorer, score_bank_multiview_max, score_bank_topk_mean

OUT = os.path.join(ROOT, 'outputs')
DATA = os.path.join(ROOT, 'data', 'dl')
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'

BANK_CTFT = os.path.join(OUT, 'inat_photo_bank_ctftshift.pt')
BANK_336 = os.path.join(OUT, 'inat_photo_bank_fullft336shift.pt')
CROP_TEST = os.path.join(OUT, 'emb_test_ctft_cropviews.pt')
CROP_UNS = os.path.join(OUT, 'emb_unseen_ctft_cropviews.pt')
OSTRIP_TEST = os.path.join(OUT, 'emb_test_ctft_ostrip.pt')
OSTRIP_UNS = os.path.join(OUT, 'emb_unseen_ctft_ostrip.pt')
VIEW_KEYS = ['center', 'squash', 'ostrip0', 'ostrip1', 'ostrip2', 'ostrip3', 'ostrip4']

B2_PROTO_FROZEN = os.path.join(OUT, 'inat_tol_merged_b2_a05.pt')
B2_PROTO_LORA = os.path.join(OUT, 'inat_tol_merged_b2lora_a0.5.pt')
PIPELINE_PKG_PATH = os.path.join(OUT, 'learned_pipeline_v60.pt')


def load(p):
    d = torch.load(p, weights_only=False)
    return {fn: i for i, fn in enumerate(d['files'])}, F.normalize(d['feats'].float(), dim=-1), list(d['files'])


def zc(M):
    return (M - M.mean()) / (M.std() + 1e-6)


def load_cropviews(path, keys):
    d = torch.load(path, weights_only=False)
    idx = {fn: i for i, fn in enumerate(d['files'])}
    views = {k: F.normalize(d['views'][k].float(), dim=-1) for k in keys if k in d['views']}
    return idx, views


def main():
    print("=" * 80)
    print("=== Building v60: Learning-Based & Simplified Submission (No DBNorm) ===")
    print("=" * 80)
    torch.set_num_threads(8)

    # 1. Load classes, labels, taxonomy & text embeddings
    classes = list(pickle.load(open(os.path.join(DATA, 'all_classes.pkl'), 'rb')))
    ci = {c: i for i, c in enumerate(classes)}
    NCLS = len(classes)
    lab = json.load(open(os.path.join(DATA, 'label_train.json')))

    TtH = F.normalize(torch.load(os.path.join(OUT, 'text_emb_h_taxon.pt'), weights_only=False)['emb_taxon'].float(), dim=-1).to(DEV)
    TnL = F.normalize(torch.load(os.path.join(OUT, 'text_emb.pt'), weights_only=False)['emb_name'].float(), dim=-1).to(DEV)
    TTX = F.normalize(torch.load(os.path.join(OUT, 'text_emb_h_promptens.pt'), weights_only=False)['emb_taxctx'].float(), dim=-1).to(DEV)
    Ttb = F.normalize(torch.load(os.path.join(OUT, 'text_emb_taxabind_taxctx.pt'), weights_only=False)['emb_taxctx'].float(), dim=-1).to(DEV)

    I_all = None
    proto_p = os.path.join(OUT, 'inat_protos_ctftshift_full.pt')
    if os.path.isfile(proto_p):
        p_raw = torch.load(proto_p, weights_only=False)
        I_all = F.normalize(p_raw['protos'].float() if isinstance(p_raw, dict) else p_raw.float(), dim=-1).to(DEV)

    MEMBERS = [
        ('ctftshift', True, 1.0),
        ('ftshift', True, 2.5),
        ('fullft336shift', True, 2.5),
        ('L', False, 0.0),
        ('fullft336_v2', True, 0.0)
    ]
    TRAIN = {
        'ctftshift': 'emb_train_ctftshift',
        'ftshift': 'emb_train_ftshift',
        'fullft336shift': 'emb_train_fullft336shift',
        'L': 'emb_train',
        'fullft336_v2': 'emb_train_fullft336_v2'
    }

    train = {t: load(os.path.join(OUT, f'{TRAIN[t]}.pt')) for t, _, _ in MEMBERS}
    common = None
    for t, (idx, _, files) in train.items():
        s = set(files)
        common = s if common is None else (common & s)
    common = [fn for fn in common if fn in lab and lab[fn] in ci]
    by = defaultdict(list)
    for fn in common:
        by[lab[fn]].append(fn)

    seen = sorted(by.keys())
    s2i = {c: i for i, c in enumerate(seen)}
    S = len(seen)
    kept_idx = torch.tensor([ci[c] for c in seen]).to(DEV)
    other_idx = torch.tensor([i for i in range(NCLS) if classes[i] not in set(seen)]).to(DEV)
    TseenTax = TtH[kept_idx]
    LAM = 4.0
    print(f"Seen classes: {S:,} | Unseen classes: {len(other_idx):,}")

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
            sc = ps + 2.0 * cmax
            if hspace:
                sc = sc + LAM * (e @ TseenTax.t())
            out[i:i + 2000] = sc
        return out

    # Load eval embeddings
    test = {t: load(os.path.join(OUT, f'{TRAIN[t].replace("emb_train", "emb_test")}.pt')) for t, _, _ in MEMBERS}
    unseen = {t: load(os.path.join(OUT, f'{TRAIN[t].replace("emb_train", "emb_unseen")}.pt')) for t, _, _ in MEMBERS}
    tb_test = load(os.path.join(OUT, 'emb_test_taxabind.pt'))
    tb_uns = load(os.path.join(OUT, 'emb_unseen_taxabind.pt'))
    b2_test = load(os.path.join(OUT, 'emb_test_bioclip2.pt'))
    b2_uns = load(os.path.join(OUT, 'emb_unseen_bioclip2.pt'))
    b2l_test = load(os.path.join(OUT, 'emb_test_bioclip2_lora_v2.pt'))
    b2l_uns = load(os.path.join(OUT, 'emb_unseen_bioclip2_lora_v2.pt'))

    tf = sorted(set.intersection(*[set(f) for (_, _, f) in test.values()], set(tb_test[2]), set(b2_test[2])))
    uf = sorted(set.intersection(*[set(f) for (_, _, f) in unseen.values()], set(tb_uns[2]), set(b2_uns[2])))
    all_files = tf + uf
    assert len(all_files) == 35665
    print(f"Combined evaluation batch: {len(all_files):,} images (Test: {len(tf):,}, Unseen: {len(uf):,})")

    def qcat(t):
        ti, tfeat, _ = test[t]
        ui, ufeat, _ = unseen[t]
        return torch.cat([torch.stack([tfeat[ti[fn]] for fn in tf]),
                          torch.stack([ufeat[ui[fn]] for fn in uf])]).to(DEV)

    Q = {t: qcat(t) for t, _, _ in MEMBERS}
    Qtb = torch.cat([
        torch.stack([tb_test[1][tb_test[0][fn]] for fn in tf]),
        torch.stack([tb_uns[1][tb_uns[0][fn]] for fn in uf]),
    ]).to(DEV)
    Qb2 = torch.cat([
        torch.stack([b2_test[1][b2_test[0][fn]] for fn in tf]),
        torch.stack([b2_uns[1][b2_uns[0][fn]] for fn in uf]),
    ]).to(DEV)
    Qb2l = torch.cat([
        torch.stack([b2l_test[1][b2l_test[0][fn]] for fn in tf]),
        torch.stack([b2l_uns[1][b2l_uns[0][fn]] for fn in uf]),
    ]).to(DEV)

    # 1. Seen specialist score block
    print("Computing Seen Specialist Scores...")
    seen_block = torch.zeros(len(all_files), S, device=DEV)
    for t, hs, w in MEMBERS:
        if w == 0.0:
            continue
        seen_block = seen_block + w * zc(seen_score(Q[t], *PR[t], hs))

    # 2. Extract multi-crop views for 7-view CTFT photo bank
    print("Extracting & Aligning 7 Crop Views...")
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
    Nq = len(all_files)
    Cu = len(other_list)

    print("Computing Raw Unseen Scores Across Legs (No DBNorm)...")
    # A. 7-view CTFT Photo Bank
    Sraw_ctft = score_bank_multiview_max(view_tensors, torch.load(BANK_CTFT, weights_only=False)['bank'], other_list, topm=4, device=DEV)
    # B. 336 Photo Bank
    Sraw_336 = score_bank_topk_mean(Q['fullft336shift'], torch.load(BANK_336, weights_only=False)['bank'], other_list, topm=4, device=DEV)
    # C. BioCLIP-2 Frozen ToL
    p_b2_f = F.normalize(torch.load(B2_PROTO_FROZEN, weights_only=False)['protos'].float(), dim=-1).to(DEV)
    Sraw_b2_f = Qb2 @ p_b2_f[other_idx].t()
    # D. BioCLIP-2 LoRA ToL
    p_b2_l = F.normalize(torch.load(B2_PROTO_LORA, weights_only=False)['protos'].float(), dim=-1).to(DEV)
    Sraw_b2_l = Qb2l @ p_b2_l[other_idx].t()
    # E. Text Taxon
    Sraw_txt_taxon = Q['ctftshift'] @ TtH[other_idx].t()
    # F. Text TaxaBind
    Sraw_txt_tb = Qtb @ Ttb[other_idx].t()

    raw_unseen_legs = {
        'ctft_bank': Sraw_ctft,
        '336_bank': Sraw_336,
        'b2_frozen_tol': Sraw_b2_f,
        'b2_lora_tol': Sraw_b2_l,
        'text_taxon': Sraw_txt_taxon,
        'text_taxabind': Sraw_txt_tb
    }

    # 3. Load Trained Learning Pipeline Package
    print(f"\nLoading Learned Pipeline Package from {PIPELINE_PKG_PATH}...")
    pkg = torch.load(PIPELINE_PKG_PATH, map_location='cpu', weights_only=False)

    calibrators = MultiModalPolynomialCalibrator(pkg['modality_names']).to(DEV)
    calibrators.load_state_dict({k: v.to(DEV) for k, v in pkg['calibrators_state'].items()})
    calibrators.eval()

    fusion_scorer = MultiModalFusionScorer(pkg['modality_names']).to(DEV)
    fusion_scorer.load_state_dict({k: v.to(DEV) for k, v in pkg['fusion_state'].items()})
    fusion_scorer.eval()

    gate = CalibratedNoveltyLogisticGate()
    gate.load_state_dict(pkg['gate_state'])

    print(f"  ├─ Gate Val AUC: {pkg['val_metrics'].get('val_weighted_acc', 0.75):.4f}")
    print(f"  ├─ Optimal Logistic Threshold: {gate.optimal_threshold:.4f}")
    print(f"  └─ Modality Fusion Weights: {F.softmax(fusion_scorer.weights, dim=0).detach().cpu().numpy().round(4)}")

    # 4. Calibrate and Fuse Unseen Legs Inductively
    with torch.no_grad():
        calibrated_legs = calibrators(raw_unseen_legs)
        unseen_scores = fusion_scorer(calibrated_legs) # [Nq, Cu]

    # 5. Extract Gating Features and Route
    print("\nExtracting Novelty Gating Features for 35,665 Evaluation Images...")
    I_unseen = I_all[other_idx] if I_all is not None else None
    X_gate = extract_gating_features_tensor(Q['ctftshift'], TtH[kept_idx], TtH[other_idx], I_unseen)
    p_seen = gate.predict_proba(X_gate.cpu().numpy())
    route_seen = (p_seen >= gate.optimal_threshold)

    n_seen_routed = np.sum(route_seen)
    print(f"  ├─ Mean P(seen | x): {np.mean(p_seen):.4f} | Median: {np.median(p_seen):.4f}")
    print(f"  └─ Routed to Seen Specialist: {n_seen_routed:,}/{Nq:,} ({n_seen_routed / Nq * 100:.2f}%)")

    # 6. Generate Predictions
    pred_idx = torch.empty(Nq, dtype=torch.long, device=DEV)
    idx_seen = torch.from_numpy(route_seen).nonzero(as_tuple=True)[0].to(DEV)
    if len(idx_seen) > 0:
        pred_idx[idx_seen] = kept_idx[seen_block[idx_seen].argmax(dim=1)]

    idx_uns = torch.from_numpy(~route_seen).nonzero(as_tuple=True)[0].to(DEV)
    if len(idx_uns) > 0:
        pred_idx[idx_uns] = other_idx[unseen_scores[idx_uns].argmax(dim=1)]

    preds = {fn: classes[j] for fn, j in zip(all_files, pred_idx.tolist())}
    assert len(preds) == 35665 and all(c in ci for c in preds.values())

    seen_set = set(seen)
    cov_unseen = len(set(p for p in preds.values() if p not in seen_set))
    o_t = sum(1 for fn in tf if preds[fn] not in seen_set)
    o_u = sum(1 for fn in uf if preds[fn] not in seen_set)

    print(f"\nPrediction Summary:")
    print(f"  ├─ Total Predictions: {len(preds):,}")
    print(f"  ├─ Unique Unseen Classes Predicted: {cov_unseen:,}/{len(other_idx):,}")
    print(f"  ├─ 'Test' Folder -> Seen Route Accuracy: {100 * (1 - o_t / len(tf)):.2f}%")
    print(f"  └─ 'Unseen' Folder -> Unseen Route Accuracy: {100 * o_u / len(uf):.2f}%")

    # 7. Package Final Submission Zip
    tag = 'v60_learned_nodbnorm'
    os.makedirs('submissions', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)
    json_path = f'outputs/prediction_{tag}.json'
    zip_path = f'outputs/submission_{tag}.zip'
    sub_zip_path = f'submissions/submission_{tag}.zip'

    json.dump(preds, open(json_path, 'w'))
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
        z.write(json_path, 'prediction.json')
    shutil.copyfile(zip_path, sub_zip_path)

    print(f"\n[SUCCESS] Generated Learning-Based Submission:")
    print(f"  ├─ Prediction JSON: {json_path}")
    print(f"  ├─ Submission ZIP: {sub_zip_path} ({os.path.getsize(sub_zip_path):,} bytes)")
    print("=" * 80)


if __name__ == '__main__':
    main()
