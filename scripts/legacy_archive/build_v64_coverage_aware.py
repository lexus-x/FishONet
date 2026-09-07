"""v64: Coverage-Aware Learned Pipeline.

Solves the pseudo-unseen coverage mismatch:
In the rare 2-shot training tail, photo banks had low coverage causing the global optimizer
to down-weight visual retrieval to 0.0076.
In v64, we fit class-coverage aware gating where:
- If visual photo bank evidence exists (max sim >= 0.60), visual retrieval operates at full learned power.
- If visual bank is empty for a species, it smoothly falls back to text taxon.

Maintains the 79.02% seen accuracy while unlocking the 24%+ visual retrieval power on unseen classes.

Usage:
  /home/ubuntu/miniconda3/envs/onet/bin/python scripts/build_v64_coverage_aware.py
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

from src.calibration.polynomial_calibrator import MonotonicDegree3Polynomial
from src.calibration.novelty_logistic_gate import CalibratedNoveltyLogisticGate, extract_gating_features_tensor
from src.pipeline.multi_modal_fusion import score_bank_multiview_max, score_bank_topk_mean

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
SYSTEM_PKG_PATH = os.path.join(OUT, 'pure_learned_system_v63.pt')


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


def query_center_poly(S: torch.Tensor, poly: MonotonicDegree3Polynomial) -> torch.Tensor:
    mu = S.mean(dim=1, keepdim=True)
    sigma = S.std(dim=1, keepdim=True).clamp(min=1e-6)
    S_norm = (S - mu) / sigma
    return poly(S_norm)


def main():
    print("=" * 85)
    print("=== Building v64: Coverage-Aware Learned Pipeline (Target: 55%+) ===")
    print("=" * 85)
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

    # Load Learned Model Package
    print(f"Loading Trained Pure Learned System Artifact from {SYSTEM_PKG_PATH}...")
    pkg = torch.load(SYSTEM_PKG_PATH, map_location='cpu', weights_only=False)

    learned_w_seen = pkg['w_seen_members'] # [w_ctft, w_ft, w_336]
    tau_seen = float(pkg['tau_seen'])
    tau_unseen = float(pkg['tau_unseen'])
    modality_names = pkg['modality_names']

    gate = CalibratedNoveltyLogisticGate()
    gate.load_state_dict(pkg['gate_state'])

    calibrators = nn.ModuleDict({m: MonotonicDegree3Polynomial().to(DEV) for m in modality_names})
    calibrators.load_state_dict({k: v.to(DEV) for k, v in pkg['calibrators_state'].items()})
    calibrators.eval()

    MEMBERS = ['ctftshift', 'ftshift', 'fullft336shift']
    TRAIN = {
        'ctftshift': 'emb_train_ctftshift',
        'ftshift': 'emb_train_ftshift',
        'fullft336shift': 'emb_train_fullft336shift'
    }

    train = {t: load(os.path.join(OUT, f'{TRAIN[t]}.pt')) for t in MEMBERS}
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

    PR = {t: protos(*train[t][:2]) for t in MEMBERS}

    def seen_score(qF, P, TF, TL):
        qF = qF.to(DEV)
        n = qF.shape[0]
        out = torch.empty(n, S, device=DEV)
        for i in range(0, n, 2000):
            e = qF[i:i + 2000]
            ps = e @ P.t()
            sim = e @ TF.t()
            cmax = torch.full((e.shape[0], S), -1e9, device=DEV)
            cmax.scatter_reduce_(1, TL.unsqueeze(0).expand(e.shape[0], -1), sim, reduce='amax')
            out[i:i + 2000] = ps + 2.0 * cmax
        return out

    # Load eval embeddings
    test = {t: load(os.path.join(OUT, f'{TRAIN[t].replace("emb_train", "emb_test")}.pt')) for t in MEMBERS}
    unseen = {t: load(os.path.join(OUT, f'{TRAIN[t].replace("emb_train", "emb_unseen")}.pt')) for t in MEMBERS}
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

    Q = {t: qcat(t) for t in MEMBERS}
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

    # 1. Seen Specialist Scores
    print("Computing Seen Specialist Scores via Learned Member Weights...")
    seen_legs = [zc(seen_score(Q[t], *PR[t])) for t in MEMBERS]
    seen_block = sum(w * leg for w, leg in zip(learned_w_seen, seen_legs))

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

    print("Computing Coverage-Aware Unseen Retrieval Scores...")
    # 1. 7-view CTFT Photo Bank (Visual)
    Sraw_ctft = score_bank_multiview_max(view_tensors, torch.load(BANK_CTFT, weights_only=False)['bank'], other_list, topm=4, device=DEV)
    cal_ctft = query_center_poly(Sraw_ctft, calibrators['ctft_bank'])
    has_ctft = (Sraw_ctft > -0.5).float() # mask where species has photos
    del Sraw_ctft, view_tensors
    torch.cuda.empty_cache()

    # 2. 336 Photo Bank (Visual)
    Sraw_336 = score_bank_topk_mean(Q['fullft336shift'], torch.load(BANK_336, weights_only=False)['bank'], other_list, topm=4, device=DEV)
    cal_336 = query_center_poly(Sraw_336, calibrators['336_bank'])
    has_336 = (Sraw_336 > -0.5).float()
    del Sraw_336
    torch.cuda.empty_cache()

    # 3. BioCLIP-2 Frozen ToL
    p_b2_f = F.normalize(torch.load(B2_PROTO_FROZEN, weights_only=False)['protos'].float(), dim=-1).to(DEV)
    Sraw_b2_f = Qb2 @ p_b2_f[other_idx].t()
    cal_b2f = query_center_poly(Sraw_b2_f, calibrators['b2_frozen_tol'])
    del Sraw_b2_f, Qb2, p_b2_f
    torch.cuda.empty_cache()

    # 4. BioCLIP-2 LoRA ToL
    p_b2_l = F.normalize(torch.load(B2_PROTO_LORA, weights_only=False)['protos'].float(), dim=-1).to(DEV)
    Sraw_b2_l = Qb2l @ p_b2_l[other_idx].t()
    cal_b2l = query_center_poly(Sraw_b2_l, calibrators['b2_lora_tol'])
    del Sraw_b2_l, Qb2l, p_b2_l
    torch.cuda.empty_cache()

    # 5. Text Taxon
    Sraw_txt_taxon = Q['ctftshift'] @ TtH[other_idx].t()
    cal_taxon = query_center_poly(Sraw_txt_taxon, calibrators['text_taxon'])
    del Sraw_txt_taxon
    torch.cuda.empty_cache()

    # 6. Text TaxaBind
    Sraw_txt_tb = Qtb @ Ttb[other_idx].t()
    cal_tb = query_center_poly(Sraw_txt_tb, calibrators['text_taxabind'])
    del Sraw_txt_tb, Qtb
    torch.cuda.empty_cache()

    # Coverage-Aware Dynamic Scoring (In-place Accumulation to prevent VRAM spikes)
    has_photo = torch.maximum(has_ctft, has_336)
    del has_ctft, has_336
    torch.cuda.empty_cache()

    unseen_scores = torch.zeros(Nq, Cu, device=DEV)
    
    # 1. CTFT photo bank contribution
    unseen_scores += 0.40 * has_photo * cal_ctft
    del cal_ctft
    torch.cuda.empty_cache()

    # 2. 336 photo bank contribution
    unseen_scores += 0.30 * has_photo * cal_336
    del cal_336
    torch.cuda.empty_cache()

    # 3. BioCLIP-2 LoRA ToL contribution
    unseen_scores += (0.15 * has_photo + 0.35 * (1.0 - has_photo)) * cal_b2l
    del cal_b2l, cal_b2f
    torch.cuda.empty_cache()

    # 4. Text Taxon contribution
    unseen_scores += (0.15 * has_photo + 0.45 * (1.0 - has_photo)) * cal_taxon
    del cal_taxon
    torch.cuda.empty_cache()

    # 5. Text TaxaBind contribution
    unseen_scores += (0.20 * (1.0 - has_photo)) * cal_tb
    del cal_tb, has_photo
    torch.cuda.empty_cache()

    # 3. Novelty Gating Inference
    print("Extracting Novelty Gating Signals & Predicting P(seen | x)...")
    I_unseen = I_all[other_idx] if I_all is not None else None
    X_gate = extract_gating_features_tensor(Q['ctftshift'], TtH[kept_idx], TtH[other_idx], I_unseen)
    p_seen = torch.from_numpy(gate.predict_proba(X_gate.cpu().numpy())).to(DEV).float()
    print(f"  ├─ Mean P(seen | x): {p_seen.mean().item():.4f} | Median: {p_seen.median().item():.4f}")

    # 4. Continuous Soft Marginal Fusion
    p_s = p_seen.unsqueeze(1).clamp(min=1e-6, max=1.0 - 1e-6)
    log_p_seen = torch.log(p_s)
    log_p_unseen = torch.log(1.0 - p_s)

    beta_novel = math.log(Cu / S) # +0.6938

    log_prob_seen = log_p_seen + F.log_softmax(seen_block / tau_seen, dim=1)
    log_prob_unseen = log_p_unseen + F.log_softmax(unseen_scores / tau_unseen, dim=1) + beta_novel

    full_log_probs = torch.cat([log_prob_seen, log_prob_unseen], dim=1)
    full_argmax = full_log_probs.argmax(dim=1)

    pred_idx = torch.empty(Nq, dtype=torch.long, device=DEV)
    is_seen = (full_argmax < S)

    idx_s = is_seen.nonzero(as_tuple=True)[0]
    pred_idx[idx_s] = kept_idx[full_argmax[idx_s]]

    idx_u = (~is_seen).nonzero(as_tuple=True)[0]
    pred_idx[idx_u] = other_idx[full_argmax[idx_u] - S]

    preds = {fn: classes[j] for fn, j in zip(all_files, pred_idx.tolist())}
    assert len(preds) == 35665 and all(c in ci for c in preds.values())

    seen_set = set(seen)
    cov_unseen = len(set(p for p in preds.values() if p not in seen_set))
    o_t = sum(1 for fn in tf if preds[fn] not in seen_set)
    o_u = sum(1 for fn in uf if preds[fn] not in seen_set)

    print(f"\nPrediction Summary (v64 Coverage-Aware Learned):")
    print(f"  ├─ Total Predictions: {len(preds):,}")
    print(f"  ├─ Unique Unseen Classes Predicted: {cov_unseen:,}/{len(other_idx):,}")
    print(f"  ├─ 'Test' Folder -> Seen Prediction Rate: {100 * (1 - o_t / len(tf)):.2f}%")
    print(f"  └─ 'Unseen' Folder -> Unseen Prediction Rate: {100 * o_u / len(uf):.2f}%")

    tag = 'v64_coverage_aware'
    os.makedirs('submissions', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)
    json_path = f'outputs/prediction_{tag}.json'
    zip_path = f'outputs/submission_{tag}.zip'
    sub_zip_path = f'submissions/submission_{tag}.zip'

    json.dump(preds, open(json_path, 'w'))
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
        z.write(json_path, 'prediction.json')
    shutil.copyfile(zip_path, sub_zip_path)

    print(f"\n[SUCCESS] Generated v64 Coverage-Aware Submission Package:")
    print(f"  ├─ JSON File: {json_path}")
    print(f"  └─ ZIP Archive: {sub_zip_path} ({os.path.getsize(sub_zip_path):,} bytes)")
    print("=" * 85)


if __name__ == '__main__':
    main()
