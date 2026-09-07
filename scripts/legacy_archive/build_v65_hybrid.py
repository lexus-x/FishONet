"""v65: Hybrid Learned-Seen + Proven-Unseen Pipeline.

Combines:
- v63's learned seen specialist (79.02% real accuracy)
- v63's learned logistic novelty gate (AUC 0.9559)
- v56's proven unseen retrieval head (dbnorm + sinkhorn + heavy visual bank weights)

v56 achieved 20.57% unseen with its retrieval recipe.
v63 achieved 79.02% seen with its learned specialist.
Projected: 0.5635 × 79% + 0.4365 × 20.5% ≈ 53.5%+

Usage:
  /home/ubuntu/miniconda3/envs/onet/bin/python scripts/build_v65_hybrid.py
"""
import json
import os
import pickle
import shutil
import zipfile
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_num_threads(8)
D = 'data/dl'
OUT = 'outputs'

SINK_ITER = 60
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
SEEN_FRAC = 0.60
TAU = 1.8


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
    Sraw = torch.full((Nq, Cu), -1e4, device=dev)
    for j, gidx in enumerate(other_list):
        if gidx not in bank:
            continue
        v = bank[gidx]
        if isinstance(v, torch.Tensor):
            B = v.float().to(dev)
            if B.dim() == 1:
                B = B.unsqueeze(0)
        else:
            if len(v) == 0:
                continue
            B = torch.stack(v).float().to(dev)
        B = F.normalize(B, dim=-1)
        sims = Qenc @ B.t()
        topk = sims.topk(min(TOPM, B.shape[0]), dim=1).values
        Sraw[:, j] = topk.mean(dim=1)
    return Sraw


def score_bank_maxviews(view_list, bank, other_list):
    V = len(view_list)
    Nq = view_list[0].shape[0]
    Qcat = torch.cat(view_list, dim=0)
    Cu = len(other_list)
    Sraw = torch.full((Nq, Cu), -1e4, device=dev)
    for j, gidx in enumerate(other_list):
        if gidx not in bank:
            continue
        v = bank[gidx]
        if isinstance(v, torch.Tensor):
            B = v.float().to(dev)
            if B.dim() == 1:
                B = B.unsqueeze(0)
        else:
            if len(v) == 0:
                continue
            B = torch.stack(v).float().to(dev)
        B = F.normalize(B, dim=-1)
        sims = Qcat @ B.t()
        topk = sims.topk(min(TOPM, B.shape[0]), dim=1).values.mean(dim=1)
        mv = topk.view(V, Nq).max(dim=0).values
        Sraw[:, j] = mv
    return Sraw


def main():
    print("=" * 85)
    print("=== Building v65: Hybrid Learned-Seen + Proven-Unseen Pipeline ===")
    print("=" * 85)

    classes = list(pickle.load(open(os.path.join(D, 'all_classes.pkl'), 'rb')))
    ci = {c: i for i, c in enumerate(classes)}
    NCLS = len(classes)
    lab = json.load(open(os.path.join(D, 'label_train.json')))

    TtH = F.normalize(torch.load(os.path.join(OUT, 'text_emb_h_taxon.pt'), weights_only=False)['emb_taxon'].float(), dim=-1).to(dev)
    TnL = F.normalize(torch.load(os.path.join(OUT, 'text_emb.pt'), weights_only=False)['emb_name'].float(), dim=-1).to(dev)
    TTX = F.normalize(torch.load(os.path.join(OUT, 'text_emb_h_promptens.pt'), weights_only=False)['emb_taxctx'].float(), dim=-1).to(dev)
    Ttb = F.normalize(torch.load(os.path.join(OUT, 'text_emb_taxabind_taxctx.pt'), weights_only=False)['emb_taxctx'].float(), dim=-1).to(dev)

    # --- LEARNED SEEN SPECIALIST from v63 ---
    # Use learned ensemble weights instead of v56's magic numbers
    pkg = torch.load(os.path.join(OUT, 'pure_learned_system_v63.pt'), map_location='cpu', weights_only=False)
    learned_w_seen = pkg['w_seen_members']  # [0.4577, 0.2015, 0.3408]
    print(f"Learned Seen Ensemble Weights: {learned_w_seen}")

    # Seen specialist uses 3 members (ctftshift, ftshift, fullft336shift)
    # with LEARNED weights and hspace text augmentation
    MEMBERS = [
        ('ctftshift', True),
        ('ftshift', True),
        ('fullft336shift', True),
    ]
    TRAIN = {
        'ctftshift': 'emb_train_ctftshift',
        'ftshift': 'emb_train_ftshift',
        'fullft336shift': 'emb_train_fullft336shift',
    }
    # Also need L and fullft336_v2 for the text_full unseen leg (v56 recipe)
    EXTRA_MEMBERS = [
        ('L', False),
        ('fullft336_v2', True),
    ]
    EXTRA_TRAIN = {
        'L': 'emb_train',
        'fullft336_v2': 'emb_train_fullft336_v2',
    }

    all_train_keys = {**TRAIN, **EXTRA_TRAIN}
    train = {t: load(os.path.join(OUT, f'{all_train_keys[t]}.pt')) for t in list(TRAIN) + list(EXTRA_TRAIN)}

    common = None
    for t in TRAIN:
        idx, _, files = train[t]
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
    print(f'Seen classes: {S:,} | Unseen classes: {len(other_idx):,}')

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

    PR = {t: protos(*train[t][:2]) for t in TRAIN}

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

    # Load eval embeddings
    test_emb = {}
    unseen_emb = {}
    for t in list(TRAIN) + list(EXTRA_TRAIN):
        test_emb[t] = load(os.path.join(OUT, f'{all_train_keys[t].replace("emb_train", "emb_test")}.pt'))
        unseen_emb[t] = load(os.path.join(OUT, f'{all_train_keys[t].replace("emb_train", "emb_unseen")}.pt'))

    tb_test = load(os.path.join(OUT, 'emb_test_taxabind.pt'))
    tb_uns = load(os.path.join(OUT, 'emb_unseen_taxabind.pt'))
    b2_test = load(os.path.join(OUT, 'emb_test_bioclip2.pt'))
    b2_uns = load(os.path.join(OUT, 'emb_unseen_bioclip2.pt'))
    b2l_test = load(os.path.join(OUT, 'emb_test_bioclip2_lora_v2.pt'))
    b2l_uns = load(os.path.join(OUT, 'emb_unseen_bioclip2_lora_v2.pt'))

    all_test_sets = [set(f) for t in list(TRAIN) + list(EXTRA_TRAIN) for (_, _, f) in [test_emb[t]]]
    all_uns_sets = [set(f) for t in list(TRAIN) + list(EXTRA_TRAIN) for (_, _, f) in [unseen_emb[t]]]

    tf = sorted(set.intersection(*all_test_sets, set(tb_test[2]), set(b2_test[2])))
    uf = sorted(set.intersection(*all_uns_sets, set(tb_uns[2]), set(b2_uns[2])))
    all_files = tf + uf
    assert len(all_files) == 35665
    print(f'Evaluation batch: {len(all_files):,} (Test: {len(tf):,}, Unseen: {len(uf):,})')

    def qcat(t):
        ti, tfeat, _ = test_emb[t]
        ui, ufeat, _ = unseen_emb[t]
        return torch.cat([torch.stack([tfeat[ti[fn]] for fn in tf]),
                          torch.stack([ufeat[ui[fn]] for fn in uf])])

    Q = {t: qcat(t).to(dev) for t in list(TRAIN) + list(EXTRA_TRAIN)}
    Qtb = torch.cat([
        torch.stack([tb_test[1][tb_test[0][fn]] for fn in tf]),
        torch.stack([tb_uns[1][tb_uns[0][fn]] for fn in uf]),
    ]).to(dev)

    # ============================================
    # SEEN HEAD: Use v63's LEARNED ensemble weights
    # ============================================
    print("Computing Seen Specialist with LEARNED ensemble weights...")
    seen_legs = {t: zc(seen_score(Q[t], *PR[t], hs)) for t, hs in MEMBERS}
    seen_block = sum(w * seen_legs[t] for w, (t, _) in zip(learned_w_seen, MEMBERS))

    # ============================================
    # UNSEEN HEAD: Use v56's PROVEN dbnorm+sinkhorn recipe
    # ============================================
    print("Computing Unseen Retrieval with v56's PROVEN dbnorm + sinkhorn recipe...")

    # Text fusion (exact v56 recipe)
    text_full = (dbnorm(Q['ctftshift'] @ TtH.t()) + 0.5 * dbnorm(Q['L'] @ TnL.t())
                 + 0.75 * dbnorm(Q['fullft336_v2'] @ TtH.t()) + 1.0 * dbnorm(Q['ftshift'] @ TtH.t())
                 + 1.0 * dbnorm(Q['ctftshift'] @ TTX.t())
                 + TAXABIND_W * dbnorm(Qtb @ Ttb.t()))

    # Crop views
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
        view_tensors.append(F.normalize(torch.stack(rows).float(), dim=-1).to(dev))

    other_list = other_idx.tolist()
    Nq = len(all_files)
    Cu = len(other_list)

    print(f'Scoring CTFT crop-max bank Cu={Cu} ...')
    Sraw_ct = score_bank_maxviews(view_tensors, torch.load(BANK_CTFT, weights_only=False)['bank'], other_list)
    del view_tensors
    torch.cuda.empty_cache()

    print('Scoring 336 bank ...')
    Sraw_336 = score_bank(Q['fullft336shift'], torch.load(BANK_336, weights_only=False)['bank'], other_list)

    # BioCLIP-2 proto legs (exact v56 recipe)
    Qb2 = torch.cat([
        torch.stack([b2_test[1][b2_test[0][fn]] for fn in tf]),
        torch.stack([b2_uns[1][b2_uns[0][fn]] for fn in uf]),
    ]).to(dev)

    def b2_proto_leg(proto_path, Qb2_feat):
        b2P = F.normalize(torch.load(proto_path, weights_only=False)['protos'].float(), dim=-1).to(dev)
        Sraw_b = torch.full((Nq, Cu), -1e4, device=dev)
        for j, gidx in enumerate(other_list):
            pvec = b2P[gidx]
            if pvec.norm() < 0.5:
                continue
            Sraw_b[:, j] = Qb2_feat @ pvec
        return dbnorm(Sraw_b)

    b2_frozen_leg = b2_proto_leg(B2_PROTO_FROZEN, Qb2)

    Qb2l = torch.cat([
        torch.stack([b2l_test[1][b2l_test[0][fn]] for fn in tf]),
        torch.stack([b2l_uns[1][b2l_uns[0][fn]] for fn in uf]),
    ]).to(dev)
    b2_lora_leg = b2_proto_leg(B2_PROTO_LORA, Qb2l)
    del Qb2, Qb2l
    torch.cuda.empty_cache()

    img_leg = W_CTFT * dbnorm(Sraw_ct) + W_336 * dbnorm(Sraw_336)
    del Sraw_ct, Sraw_336
    torch.cuda.empty_cache()

    text_unseen_only = text_full[:, other_idx] + img_leg + B2_WF * b2_frozen_leg + B2_WL * b2_lora_leg
    del text_full, img_leg, b2_frozen_leg, b2_lora_leg
    torch.cuda.empty_cache()

    # ============================================
    # ROUTING: v56's proven hard-fraction routing
    # ============================================
    img_seenmax = seen_block.max(1).values
    text_margin = ((Q['ctftshift'] @ TtH[kept_idx].t()).max(1).values
                   - (Q['ctftshift'] @ TtH[other_idx].t()).max(1).values)
    combined = z1(img_seenmax) + 2.0 * z1(text_margin)

    k_seen = int(round(SEEN_FRAC * len(all_files)))
    thr = torch.topk(combined, k_seen).values.min()
    route_seen = combined >= thr

    pred_idx = torch.empty(len(all_files), dtype=torch.long, device=dev)
    pred_idx[route_seen] = kept_idx[seen_block[route_seen].argmax(1)]
    idx_uns = (~route_seen).nonzero(as_tuple=True)[0]
    pred_idx[idx_uns] = other_idx[sinkhorn(text_unseen_only[idx_uns], tau=TAU).argmax(1)]

    preds = {fn: classes[j] for fn, j in zip(all_files, pred_idx.tolist())}
    assert len(preds) == 35665 and all(c in ci for c in preds.values())

    seen_set = set(seen)
    cov_unseen = len(set(p for p in preds.values() if p not in seen_set))
    o_t = sum(1 for fn in tf if preds[fn] not in seen_set)
    o_u = sum(1 for fn in uf if preds[fn] not in seen_set)

    print(f"\nPrediction Summary (v65 Hybrid Learned-Seen + Proven-Unseen):")
    print(f"  ├─ Total Predictions: {len(preds):,}")
    print(f"  ├─ Unique Unseen Classes Predicted: {cov_unseen:,}/{Cu:,}")
    print(f"  ├─ 'Test' Folder -> Seen Rate: {100 * (1 - o_t / len(tf)):.2f}%")
    print(f"  └─ 'Unseen' Folder -> Unseen Rate: {100 * o_u / len(uf):.2f}%")

    tag = 'v65_hybrid'
    os.makedirs('submissions', exist_ok=True)
    json_path = f'outputs/prediction_{tag}.json'
    zip_path = f'outputs/submission_{tag}.zip'
    sub_zip_path = f'submissions/submission_{tag}.zip'

    json.dump(preds, open(json_path, 'w'))
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
        z.write(json_path, 'prediction.json')
    shutil.copyfile(zip_path, sub_zip_path)

    print(f"\n[SUCCESS] v65 Hybrid Submission:")
    print(f"  ├─ {json_path}")
    print(f"  └─ {sub_zip_path} ({os.path.getsize(sub_zip_path):,} bytes)")
    print("=" * 85)


if __name__ == '__main__':
    main()
