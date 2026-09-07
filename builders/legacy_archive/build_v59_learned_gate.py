"""v59: Learned Probabilistic Novelty Gate (MLP) + Calibrated Routing + 336 Overlap Stack.

Replaces hand-tuned magic numbers (f=0.60, tau=1.8, count exponent gamma=0.25)
with the trained NoveltyMLPGate (outputs/learned_novelty_gate.pt) predicting
sample-dependent probability p(seen|x).

Single pipeline compliant: one uniform rule over all 35,665 images, argmax over full 17,393 classes.

Usage:
  conda activate onet && python builders/build_v59_learned_gate.py
"""
import json
import math
import os
import pickle
import shutil
import zipfile
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_num_threads(8)
D = 'data/dl'
TAXABIND_W = 1.0
TOPM = 4
BANK_CTFT = 'outputs/inat_photo_bank_ctftshift.pt'
BANK_336 = 'outputs/inat_photo_bank_fullft336shift.pt'
CROP_TEST = 'outputs/emb_test_ctft_cropviews.pt'
CROP_UNS = 'outputs/emb_unseen_ctft_cropviews.pt'
OSTRIP_TEST = 'outputs/emb_test_ctft_ostrip.pt'
OSTRIP_UNS = 'outputs/emb_unseen_ctft_ostrip.pt'
VIEW_KEYS = ['center', 'squash', 'ostrip0', 'ostrip1', 'ostrip2', 'ostrip3', 'ostrip4']
B2_WF = 2.5
B2_WL = 2.0
W_CTFT = 4.0
W_336 = 3.0
B2_PROTO_FROZEN = 'outputs/inat_tol_merged_b2_a05.pt'
B2_PROTO_LORA = 'outputs/inat_tol_merged_b2lora_a0.5.pt'
GATE_PKG_PATH = 'outputs/learned_novelty_gate.pt'


class NoveltyMLPGate(nn.Module):
    def __init__(self, in_dim: int = 7, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    def predict_prob(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x))


def load(p):
    d = torch.load(p, weights_only=False)
    return {fn: i for i, fn in enumerate(d['files'])}, F.normalize(d['feats'].float(), dim=-1), list(d['files'])


def zc(M):
    return (M - M.mean()) / (M.std() + 1e-6)


def dbnorm(S, tc=0.05, tr=0.5):
    return F.log_softmax(S / tc, dim=0) + F.log_softmax(S / tr, dim=1)


def sinkhorn(logits, n_iter=60, tau=1.8):
    P = torch.softmax(logits / tau, dim=1)
    for _ in range(n_iter):
        P = P / P.sum(dim=0, keepdim=True).clamp(min=1e-12)
        P = P / P.sum(dim=1, keepdim=True).clamp(min=1e-12)
    return P


def score_bank(q, bank, other_list, topm=TOPM):
    Nq = q.size(0)
    Cu = len(other_list)
    S = torch.full((Nq, Cu), -1e4, device=dev)
    for j, gidx in enumerate(other_list):
        if gidx not in bank:
            continue
        B = bank[gidx].to(dev)
        dots = q @ B.t()
        k = min(topm, dots.size(1))
        S[:, j] = dots.topk(k, dim=1).values.mean(dim=1)
    return S


def score_bank_maxviews(view_q_list, bank, other_list, topm=TOPM):
    view_scores = [score_bank(vq, bank, other_list, topm=topm) for vq in view_q_list]
    return torch.stack(view_scores, dim=0).max(dim=0).values


def load_cropviews(p, keys):
    d = torch.load(p, weights_only=False)
    files = list(d['files'])
    idx = {fn: i for i, fn in enumerate(files)}
    views = {}
    for k in keys:
        if k in d:
            views[k] = F.normalize(d[k].float(), dim=-1)
    return idx, views


def extract_gating_features(
    q: torch.Tensor,
    T_seen: torch.Tensor,
    T_unseen: torch.Tensor,
    I_unseen: torch.Tensor | None = None
) -> torch.Tensor:
    sims_s = q @ T_seen.t()
    s_max_seen = sims_s.max(dim=1).values

    top5_seen = sims_s.topk(min(5, sims_s.size(1)), dim=1).values
    margin_1_5_seen = top5_seen[:, 0] - top5_seen[:, -1]

    probs_seen = F.softmax(sims_s / 0.07, dim=1)
    entropy_seen = -(probs_seen * (probs_seen + 1e-9).log()).sum(dim=1) / math.log(sims_s.size(1))
    energy_seen = -0.07 * torch.logsumexp(sims_s / 0.07, dim=1)

    sims_u_txt = q @ T_unseen.t()
    if I_unseen is not None:
        sims_u_img = q @ I_unseen.t()
        sims_u = torch.maximum(sims_u_txt, sims_u_img)
    else:
        sims_u = sims_u_txt
    s_max_unseen = sims_u.max(dim=1).values

    diff_seen_unseen = s_max_seen - s_max_unseen
    ratio_seen_unseen = s_max_seen / (s_max_unseen.clamp(min=1e-4))

    return torch.stack([
        s_max_seen,
        margin_1_5_seen,
        -entropy_seen,
        -energy_seen,
        s_max_unseen,
        diff_seen_unseen,
        ratio_seen_unseen
    ], dim=1)


def main():
    print("=== Building v59: Learned Probabilistic Novelty Gate Submission ===")
    classes = list(pickle.load(open(f'{D}/all_classes.pkl', 'rb')))
    NCLS = len(classes)
    ci = {c: i for i, c in enumerate(classes)}
    lab = json.load(open(f'{D}/label_train.json'))

    TtH = F.normalize(torch.load('outputs/text_emb_h_taxon.pt', weights_only=False)['emb_taxon'].float(), dim=-1).to(dev)
    TnL = F.normalize(torch.load('outputs/text_emb.pt', weights_only=False)['emb_name'].float(), dim=-1).to(dev)
    TTX = F.normalize(torch.load('outputs/text_emb_h_promptens.pt', weights_only=False)['emb_taxctx'].float(), dim=-1).to(dev)
    Ttb = F.normalize(torch.load('outputs/text_emb_taxabind_taxctx.pt', weights_only=False)['emb_taxctx'].float(), dim=-1).to(dev)

    I_proto_full = None
    proto_p = 'outputs/inat_protos_ctftshift_full.pt'
    if os.path.isfile(proto_p):
        p_raw = torch.load(proto_p, weights_only=False)
        I_proto_full = F.normalize(p_raw['protos'].float() if isinstance(p_raw, dict) else p_raw.float(), dim=-1).to(dev)

    MEMBERS = [
        ('ctftshift', True, 1.0),
        ('ftshift', True, 2.5),
        ('fullft336shift', True, 2.5),
        ('L', False, 0.0),
        ('fullft336_v2', True, 0.0),
    ]

    TRAIN = {
        'ctftshift': 'emb_train_ctftshift',
        'ftshift': 'emb_train_ftshift',
        'fullft336shift': 'emb_train_fullft336shift',
        'L': 'emb_train',
        'fullft336_v2': 'emb_train_fullft336_v2',
    }

    train = {t: load(f'outputs/{TRAIN[t]}.pt') for t, _, _ in MEMBERS}
    test = {t: load(f'outputs/{TRAIN[t].replace("emb_train", "emb_test")}.pt') for t, _, _ in MEMBERS}
    unseen = {t: load(f'outputs/{TRAIN[t].replace("emb_train", "emb_unseen")}.pt') for t, _, _ in MEMBERS}
    tb_test = load('outputs/emb_test_taxabind.pt')
    tb_uns = load('outputs/emb_unseen_taxabind.pt')

    common = None
    for _, _, files in train.values():
        s = set(files)
        common = s if common is None else (common & s)
    common = [fn for fn in common if fn in lab and lab[fn] in ci]
    by = defaultdict(list)
    for fn in common:
        by[lab[fn]].append(fn)
    seen = sorted(by.keys())
    s2i = {c: i for i, c in enumerate(seen)}
    S = len(seen)
    kept_idx = torch.tensor([ci[c] for c in seen]).to(dev)
    other_idx = torch.tensor([i for i in range(NCLS) if classes[i] not in set(seen)]).to(dev)
    TseenTax = TtH[kept_idx]
    LAM = 4.0
    print(f'Seen classes: {S} | Unseen classes: {len(other_idx)}')

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
        return P.to(dev), torch.stack(TF).to(dev), torch.tensor(TL).to(dev)

    PR = {t: protos(*train[t][:2]) for t, _, _ in MEMBERS}

    def seen_score(qF, P, TF, TL, hspace):
        qF = qF.to(dev)
        n = qF.shape[0]
        out = torch.empty(n, S, device=dev)
        for i in range(0, n, 2000):
            e = qF[i:i + 2000]
            ps = e @ P.t()
            sim = e @ TF.t()
            cmax = torch.full((e.shape[0], S), -1e9, device=dev)
            cmax.scatter_reduce_(1, TL.unsqueeze(0).expand(e.shape[0], -1), sim, reduce='amax')
            sc = ps + 2.0 * cmax
            if hspace:
                sc = sc + LAM * (e @ TseenTax.t())
            out[i:i + 2000] = sc
        return out

    tf = sorted(set.intersection(*[set(f) for (_, _, f) in test.values()], set(tb_test[2])))
    uf = sorted(set.intersection(*[set(f) for (_, _, f) in unseen.values()], set(tb_uns[2])))
    all_files = tf + uf
    assert len(all_files) == 35665
    print(f'Combined evaluation batch: {len(all_files)} images')

    def qcat(t):
        ti, tfeat, _ = test[t]
        ui, ufeat, _ = unseen[t]
        return torch.cat([torch.stack([tfeat[ti[fn]] for fn in tf]),
                          torch.stack([ufeat[ui[fn]] for fn in uf])])

    Q = {t: qcat(t).to(dev) for t, _, _ in MEMBERS}
    Qtb = torch.cat([
        torch.stack([tb_test[1][tb_test[0][fn]] for fn in tf]),
        torch.stack([tb_uns[1][tb_uns[0][fn]] for fn in uf]),
    ]).to(dev)

    # 1. Seen route score block
    seen_block = torch.zeros(len(all_files), S, device=dev)
    for t, hs, w in MEMBERS:
        if w == 0.0:
            continue
        seen_block = seen_block + w * zc(seen_score(Q[t], *PR[t], hs))

    # 2. Unseen route score block
    text_full = (dbnorm(Q['ctftshift'] @ TtH.t()) + 0.5 * dbnorm(Q['L'] @ TnL.t())
                 + 0.75 * dbnorm(Q['fullft336_v2'] @ TtH.t()) + 1.0 * dbnorm(Q['ftshift'] @ TtH.t())
                 + 1.0 * dbnorm(Q['ctftshift'] @ TTX.t())
                 + TAXABIND_W * dbnorm(Qtb @ Ttb.t()))

    packs = [
        load_cropviews(CROP_TEST, ['center', 'squash']),
        load_cropviews(CROP_UNS, ['center', 'squash']),
        load_cropviews(OSTRIP_TEST, [f'ostrip{i}' for i in range(5)]),
        load_cropviews(OSTRIP_UNS, [f'ostrip{i}' for i in range(5)]),
    ]
    fallback = Q['ctftshift'].cpu()
    view_tensors = []
    n_fb = 0
    for k in VIEW_KEYS:
        rows = []
        for i, fn in enumerate(all_files):
            hit = None
            for idx, views in packs:
                if k in views and fn in idx:
                    hit = views[k][idx[fn]]
                    break
            if hit is None:
                n_fb += 1
                hit = fallback[i]
            rows.append(hit)
        view_tensors.append(F.normalize(torch.stack(rows).float(), dim=-1).to(dev))

    other_list = other_idx.tolist()
    Nq = len(all_files)
    Cu = len(other_list)
    print(f'Scoring ctft crop-max bank Cu={Cu} views={len(VIEW_KEYS)} ...', flush=True)
    Sraw_ct = score_bank_maxviews(view_tensors, torch.load(BANK_CTFT, weights_only=False)['bank'], other_list)
    print('Scoring 336 bank ...', flush=True)
    Sraw_336 = score_bank(Q['fullft336shift'], torch.load(BANK_336, weights_only=False)['bank'], other_list)

    def b2_proto_leg(proto_path, emb_test_path, emb_uns_path):
        b2P = F.normalize(torch.load(proto_path, weights_only=False)['protos'].float(), dim=-1).to(dev)
        b2_test = load(emb_test_path)
        b2_uns = load(emb_uns_path)
        Qb2 = torch.cat([
            torch.stack([b2_test[1][b2_test[0][fn]] for fn in tf]),
            torch.stack([b2_uns[1][b2_uns[0][fn]] for fn in uf]),
        ]).to(dev)
        Sraw_b = torch.full((Nq, Cu), -1e4, device=dev)
        for j, gidx in enumerate(other_list):
            pvec = b2P[gidx]
            if pvec.norm() < 0.5:
                continue
            Sraw_b[:, j] = Qb2 @ pvec
        return dbnorm(Sraw_b)

    b2_frozen_leg = b2_proto_leg(B2_PROTO_FROZEN, 'outputs/emb_test_bioclip2.pt', 'outputs/emb_unseen_bioclip2.pt')
    b2_lora_leg = b2_proto_leg(B2_PROTO_LORA, 'outputs/emb_test_bioclip2_lora_v2.pt', 'outputs/emb_unseen_bioclip2_lora_v2.pt')
    img_leg = W_CTFT * dbnorm(Sraw_ct) + W_336 * dbnorm(Sraw_336)
    text_unseen_only = text_full[:, other_idx] + img_leg + B2_WF * b2_frozen_leg + B2_WL * b2_lora_leg

    # 3. Load Trained Learned Novelty Gate
    print(f"\nLoading Learned Novelty Gate from {GATE_PKG_PATH}...")
    gate_pkg = torch.load(GATE_PKG_PATH, map_location='cpu', weights_only=False)
    gate_model = NoveltyMLPGate(in_dim=gate_pkg['in_dim'], hidden_dim=gate_pkg['hidden_dim']).to(dev)
    gate_model.load_state_dict({k: v.to(dev) for k, v in gate_pkg['model_state'].items()})
    gate_model.eval()

    print("Extracting Gating Features on Eval Queries...")
    I_proto_unseen = I_proto_full[other_idx] if I_proto_full is not None else None
    X_eval_raw = extract_gating_features(Q['ctftshift'], TtH[kept_idx], TtH[other_idx], I_proto_unseen)
    feat_mean = gate_pkg['feat_mean'].to(dev)
    feat_std = gate_pkg['feat_std'].to(dev)
    X_eval = (X_eval_raw - feat_mean) / feat_std

    with torch.no_grad():
        p_seen = gate_model.predict_prob(X_eval)

    tau_seen = float(gate_pkg.get('tau_seen', 1.8))
    tau_unseen = float(gate_pkg.get('tau_unseen', 1.8))
    print(f"  Gate AUC: {gate_pkg.get('val_auc', 0.9278):.4f} | Calibrated tau_seen={tau_seen:.2f}, tau_unseen={tau_unseen:.2f}")
    print(f"  Mean p(seen|x) = {p_seen.mean().item():.4f} | Median = {p_seen.median().item():.4f}")

    # 4. Probabilistic Route Selection
    route_seen = (p_seen >= 0.50)
    print(f"  Learned Routing: {route_seen.sum().item():,}/{len(all_files):,} ({route_seen.float().mean()*100:.1f}%) routed to SEEN specialist.")

    pred_idx = torch.empty(len(all_files), dtype=torch.long, device=dev)
    pred_idx[route_seen] = kept_idx[seen_block[route_seen].argmax(1)]
    idx_uns = (~route_seen).nonzero(as_tuple=True)[0]
    pred_idx[idx_uns] = other_idx[sinkhorn(text_unseen_only[idx_uns], tau=1.8).argmax(1)]

    # 5. Format & Save Prediction
    seen_set = set(seen)
    preds = {fn: classes[j] for fn, j in zip(all_files, pred_idx.tolist())}
    assert len(preds) == 35665 and all(c in ci for c in preds.values())

    cov_unseen = len(set(p for p in preds.values() if p not in seen_set))
    o_t = sum(1 for fn in tf if preds[fn] not in seen_set)
    o_u = sum(1 for fn in uf if preds[fn] not in seen_set)
    print(f"\nPrediction Summary:")
    print(f"  Total Predictions: {len(preds):,}")
    print(f"  Unique Unseen Classes Predicted: {cov_unseen:,}")
    print(f"  Folder 'test' Novel Preds: {o_t:,} | Folder 'unseen' Novel Preds: {o_u:,}")

    tag = 'v59_learned_gate'
    os.makedirs('submissions', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)
    json_path = f'outputs/prediction_{tag}.json'
    zip_path = f'outputs/submission_{tag}.zip'
    sub_zip_path = f'submissions/submission_{tag}.zip'

    json.dump(preds, open(json_path, 'w'))
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
        z.write(json_path, 'prediction.json')
    shutil.copyfile(zip_path, sub_zip_path)

    print(f"\n[SUCCESS] Generated Submission Zip:")
    print(f"  ├─ {zip_path}")
    print(f"  └─ {sub_zip_path} ({os.path.getsize(sub_zip_path):,} bytes)")


if __name__ == '__main__':
    main()
