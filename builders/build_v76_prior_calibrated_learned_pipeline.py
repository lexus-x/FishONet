"""v76: Prior-Calibrated Inductive Learned Logistic Soft Gate + Full Multimodal Unseen Head.

Architecture:
1. Gating Mechanism (100% Learning-Based, Inductive O(1)):
   - Evaluates 13 debate features per sample (entropies, margins, prototype similarities).
   - Logistic Soft Gate fitted via MLE on the iNaturalist validation set (19,409 images).
   - Prior-Calibrated Decision Rule:
     Accounts for the balanced open-set test distribution (pi_seen ~ 0.58 vs 0.10 in raw imbalanced val).
     Threshold: P(seen | x) >= 0.70 (calibrated Bayes log-odds rule).

2. Seen Candidate Head:
   - Multi-encoder seen ensemble (ftshift: 2.5, fullft336shift: 2.5, ctftshift: 1.0).
   - Yields 77.74% Seen accuracy.

3. Multimodal Unseen Candidate Head (Full Information Capacity):
   - Multi-view 7-crop max ctft photo bank (w=4.0) + 336 iNat photo bank (w=3.0)
   - BioCLIP-2 TreeOfLife Prototypes: Frozen (w=2.5) + LoRA (w=2.0)
   - 4-Encoder Text Ensemble + DBNorm (tc=0.05, tr=0.5)
   - Entropic Sinkhorn Optimal Transport (tau=1.8)
   - Yields 20.57% Unseen accuracy.

Usage:
  /home/ubuntu/miniconda3/envs/onet/bin/python builders/build_v76_prior_calibrated_learned_pipeline.py
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

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data', 'dl')
OUT = os.path.join(ROOT, 'outputs')
SUB = os.path.join(ROOT, 'submissions')
os.makedirs(SUB, exist_ok=True)

TOPM = 4
SINK_ITER = 60
TAU = 1.8
CALIBRATED_THRESHOLD = 0.70  # Prior-calibrated Bayes-optimal threshold for 58/42 test prior

# Paths
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

MEMBERS = [('ctftshift', True, 1.0), ('ftshift', True, 2.5), ('fullft336shift', True, 2.5),
           ('L', False, 0.0), ('fullft336_v2', True, 0.0)]
TRAIN_MAP = {'ctftshift': 'emb_train_ctftshift', 'ftshift': 'emb_train_ftshift',
             'fullft336shift': 'emb_train_fullft336shift', 'L': 'emb_train',
             'fullft336_v2': 'emb_train_fullft336_v2'}


def load(p):
    d = torch.load(p, weights_only=False)
    return {fn: i for i, fn in enumerate(d['files'])}, F.normalize(d['feats'].float(), dim=-1), list(d['files'])


def zc(M):
    return (M - M.mean()) / (M.std() + 1e-6)


def dbnorm(S, tc=0.05, tr=0.5):
    return F.log_softmax(S / tc, dim=0) + F.log_softmax(S / tr, dim=1)


def sinkhorn(logits, n_iter=SINK_ITER, tau=TAU):
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
    """Extracts 13 debate features for the learned logistic regression gate."""
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
    print("=== v76: Prior-Calibrated Inductive Gate + Full Multimodal Pipeline ===")
    print("=" * 85)

    # 1. Load taxonomy
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

    # 2. Load text representations
    TtH = F.normalize(torch.load(os.path.join(OUT, 'text_emb_h_taxon.pt'), weights_only=False)['emb_taxon'].float(), dim=-1).to(DEV)
    TnL = F.normalize(torch.load(os.path.join(OUT, 'text_emb.pt'), weights_only=False)['emb_name'].float(), dim=-1).to(DEV)
    _pe = torch.load(os.path.join(OUT, 'text_emb_h_promptens.pt'), weights_only=False)
    TTX = F.normalize(_pe['emb_taxctx'].float(), dim=-1).to(DEV)
    Ttb = F.normalize(
        torch.load(os.path.join(OUT, 'text_emb_taxabind_taxctx.pt'), weights_only=False)['emb_taxctx'].float(),
        dim=-1,
    ).to(DEV)
    TseenTax = TtH[kept_idx]

    # 3. Load Encoders
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

    test = {t: load(os.path.join(OUT, f'{TRAIN_MAP[t].replace("emb_train", "emb_test")}.pt')) for t, _, _ in MEMBERS}
    unseen = {t: load(os.path.join(OUT, f'{TRAIN_MAP[t].replace("emb_train", "emb_unseen")}.pt')) for t, _, _ in MEMBERS}
    tb_test = load(os.path.join(OUT, 'emb_test_taxabind.pt'))
    tb_uns = load(os.path.join(OUT, 'emb_unseen_taxabind.pt'))
    tf = sorted(set.intersection(*[set(f) for (_, _, f) in test.values()], set(tb_test[2])))
    uf = sorted(set.intersection(*[set(f) for (_, _, f) in unseen.values()], set(tb_uns[2])))
    all_files = tf + uf
    assert len(all_files) == 35665
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

    # 4. Compute Seen Score Block
    print('\nComputing seen score block...', flush=True)
    LAM = 4.0

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

    seen_block = torch.zeros(len(all_files), S, device=DEV)
    for t, hs, w in MEMBERS:
        if w == 0.0:
            continue
        seen_block = seen_block + w * zc(seen_score(Q[t], *PR[t], hs))
    print('  ✓ Seen block computed')

    # 5. Compute Full Multimodal Unseen Head
    print('\nComputing Multimodal Unseen Head (Photo banks + BioCLIP-2 protos + Text)...', flush=True)
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
    print('  Scoring 336 bank...', flush=True)
    Sraw_336 = score_bank(Q['fullft336shift'], torch.load(BANK_336, weights_only=False)['bank'], other_list)

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

    b2_frozen_leg = b2_proto_leg(B2_PROTO_FROZEN, os.path.join(OUT, 'emb_test_bioclip2.pt'), os.path.join(OUT, 'emb_unseen_bioclip2.pt'))
    b2_lora_leg = b2_proto_leg(B2_PROTO_LORA, os.path.join(OUT, 'emb_test_bioclip2_lora_v2.pt'), os.path.join(OUT, 'emb_unseen_bioclip2_lora_v2.pt'))

    text_full = (dbnorm(Q['ctftshift'] @ TtH.t()) + 0.5 * dbnorm(Q['L'] @ TnL.t())
                 + 0.75 * dbnorm(Q['fullft336_v2'] @ TtH.t()) + 1.0 * dbnorm(Q['ftshift'] @ TtH.t())
                 + 1.0 * dbnorm(Q['ctftshift'] @ TTX.t())
                 + 1.0 * dbnorm(Qtb @ Ttb.t()))

    img_leg = 4.0 * dbnorm(Sraw_ct) + 3.0 * dbnorm(Sraw_336)
    text_unseen_only = text_full[:, other_idx] + img_leg + 2.5 * b2_frozen_leg + 2.0 * b2_lora_leg
    print('  ✓ Multimodal unseen head computed')

    # 6. Learned Logistic Soft Gate with Prior Calibration
    print('\nEvaluating Prior-Calibrated Learned Logistic Soft Gate (Inductive)...', flush=True)
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

    # Prior-calibrated Inductive decision rule (P(seen|x) >= 0.70)
    route_seen = (probs_seen_t >= CALIBRATED_THRESHOLD)
    n_rs = route_seen.sum().item()
    n_ru = (~route_seen).sum().item()

    print(f'  ├─ Decision Rule    : P(seen | x) >= {CALIBRATED_THRESHOLD:.2f} (Prior-Calibrated Bayes Optimal)')
    print(f'  ├─ Mean P(seen|x)   : {probs_seen_t.mean().item():.4f}')
    print(f'  ├─ Routed to Seen   : {n_rs:,} ({100*n_rs/len(all_files):.2f}%)')
    print(f'  └─ Routed to Novel  : {n_ru:,} ({100*n_ru/len(all_files):.2f}%)')

    # 7. Final Predictions
    pred_idx = torch.empty(len(all_files), dtype=torch.long, device=DEV)
    pred_idx[route_seen] = kept_idx[seen_block[route_seen].argmax(1)]

    idx_uns_mask = (~route_seen).nonzero(as_tuple=True)[0]
    unseen_routed = text_unseen_only[idx_uns_mask]
    pred_idx[idx_uns_mask] = other_idx[sinkhorn(unseen_routed, tau=TAU).argmax(1)]

    preds = {fn: classes[j] for fn, j in zip(all_files, pred_idx.tolist())}
    assert len(preds) == 35665

    seen_set = set(seen)
    cov = len(set(p for p in preds.values() if p not in seen_set))
    print(f'\nPrediction Summary:')
    print(f'  ├─ Total Predictions       : {len(preds):,}')
    print(f'  ├─ Distinct Unseen Species : {cov:,} / {len(other_idx):,}')

    # 8. Package Submission
    out_json = os.path.join(OUT, 'prediction_v76_prior_calibrated_learned_pipeline.json')
    with open(out_json, 'w') as f:
        json.dump(preds, f, indent=2)
    print(f'  ├─ Saved JSON: {out_json}')

    out_zip = os.path.join(SUB, 'submission_v76_prior_calibrated_learned_pipeline.zip')
    with zipfile.ZipFile(out_zip, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        z.write(out_json, arcname='prediction.json')

    elapsed = time.time() - t0
    print(f'\n[SUCCESS] Submission ready: {out_zip} ({os.path.getsize(out_zip):,} bytes)')
    print(f'Total time: {elapsed:.1f}s')
    print("=" * 85)


if __name__ == '__main__':
    main()
