"""v82: v79 learned gate + LEAK-FREE learned re-ranker on the unseen head.

The unseen head was 10 legs summed with hand-picked constants (text 1/0.5/0.75/1/1/1,
bank 4, 336-bank 3, B2 2.5/2.0) applied uniformly to every candidate. Measured holdout
ceiling: recall@1 33.22 but recall@20 67.30 -- two thirds of the gold novel species are
already retrieved and then mis-ordered by a fixed weight vector.

v81 keeps that fusion only as a RETRIEVER (top-20) and learns the ordering with a
per-(query,candidate) model over 34 transfer-safe features: within-query score gaps, ranks
and z-scores per leg, plus bank coverage. v81 used a LEAKY holdout pool (+13.37 holdout -> +0.006 real). v82 retrains on a
pool of only gold-eligible classes, where the subpopulation shortcut carries zero
information: 59.28 -> 67.43 (+8.15), features made pool-size invariant (percentile ranks).
No raw dbnorm levels are used -- those are batch-coupled and do not transfer.

Seen head, gate (v79) and f=0.60 are unchanged from v79.

  conda activate onet && python builders/build_v81_rerank.py
"""
import json
import os
import pickle
import shutil
import sys
import zipfile
from collections import defaultdict

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'research'))
from learned_gate_v77 import FEATS, assemble, bank_scores, proto_max, standardize  # noqa: E402
from rerank_unseen_v81 import DEPLOYED_W, LEGS  # noqa: E402
from rerank_leakfree_v82 import pool_features  # noqa: E402

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_num_threads(8)
D = 'data/dl'
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
GATE_PKL = os.environ.get('GATE_PKL', 'outputs/learned_gate_v79.pkl')
RERANK_PKL = os.environ.get('RERANK_PKL', 'outputs/rerank_unseen_v82_leakfree.pkl')
TAG = os.environ.get('TAG', 'v82_rerank_leakfree_f60')
SEEN_FRAC = 0.60
TAU = 1.8


def load(p):
    d = torch.load(p, weights_only=False)
    return {fn: i for i, fn in enumerate(d['files'])}, F.normalize(d['feats'].float(), dim=-1), list(d['files'])


def zc(M):
    return (M - M.mean()) / (M.std() + 1e-6)


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
    Sraw = torch.full((Nq, len(other_list)), -1e4, device=dev)
    for j, gidx in enumerate(other_list):
        photos = bank.get(gidx)
        if photos is None or (hasattr(photos, 'numel') and photos.numel() == 0):
            continue
        if not isinstance(photos, torch.Tensor):
            photos = torch.stack(photos)
        photos = F.normalize(photos.float(), dim=-1).to(dev)
        sim = Qenc @ photos.t()
        Sraw[:, j] = sim.topk(min(TOPM, sim.shape[1]), dim=1).values.mean(dim=1)
    return Sraw


def score_bank_maxviews(view_list, bank, other_list):
    V = len(view_list)
    Nq = view_list[0].shape[0]
    Qcat = torch.cat(view_list, dim=0)
    Sraw = torch.full((Nq, len(other_list)), -1e4, device=dev)
    for j, gidx in enumerate(other_list):
        photos = bank.get(gidx)
        if photos is None or (hasattr(photos, 'numel') and photos.numel() == 0):
            continue
        if not isinstance(photos, torch.Tensor):
            photos = torch.stack(photos)
        photos = F.normalize(photos.float(), dim=-1).to(dev)
        pooled = (Qcat @ photos.t()).topk(min(TOPM, photos.shape[0]), dim=1).values.mean(dim=1)
        Sraw[:, j] = pooled.view(V, Nq).max(0).values
    return Sraw


def main():
    print('=== Building v77 learned gate (v56 heads, f=0.60 fixed) ===', flush=True)
    need = (CROP_TEST, CROP_UNS, OSTRIP_TEST, OSTRIP_UNS, BANK_CTFT, BANK_336, GATE_PKL, RERANK_PKL)
    for p in need:
        if not os.path.exists(p):
            raise SystemExit(f'missing {p}')
    gate = pickle.load(open(GATE_PKL, 'rb'))
    rr = pickle.load(open(RERANK_PKL, 'rb'))
    print(f'reranker: K={rr["K"]} feats={len(rr["names"])} leak-free holdout gain '
          f'+{rr["gain"]:.3f} (trained on {rr["pool_size"]}-class gold-eligible pool)', flush=True)
    print(f'gate: {gate["kind"]} | CV delta {gate["delta"]:+.3f} proxy-pt | '
          f'clears_kill={gate["clears_kill"]}', flush=True)

    txtHt = torch.load('outputs/text_emb_h_taxon.pt', weights_only=False)
    txtL = torch.load('outputs/text_emb.pt', weights_only=False)
    classes = list(pickle.load(open(f'{D}/all_classes.pkl', 'rb')))
    ci = {c: i for i, c in enumerate(classes)}
    NCLS = len(classes)
    TtH = F.normalize(txtHt['emb_taxon'].float(), dim=-1).to(dev)
    TnL = F.normalize(txtL['emb_name'].float(), dim=-1).to(dev)
    TTX = F.normalize(torch.load('outputs/text_emb_h_promptens.pt', weights_only=False)['emb_taxctx'].float(), dim=-1).to(dev)
    Ttb = F.normalize(torch.load('outputs/text_emb_taxabind_taxctx.pt', weights_only=False)['emb_taxctx'].float(), dim=-1).to(dev)
    lab = json.load(open(f'{D}/label_train.json'))

    MEMBERS = [('ctftshift', True, 1.0), ('ftshift', True, 2.5), ('fullft336shift', True, 2.5),
               ('L', False, 0.0), ('fullft336_v2', True, 0.0)]
    TRAIN = {'ctftshift': 'emb_train_ctftshift', 'ftshift': 'emb_train_ftshift',
             'fullft336shift': 'emb_train_fullft336shift', 'L': 'emb_train',
             'fullft336_v2': 'emb_train_fullft336_v2'}
    train = {t: load(f'outputs/{TRAIN[t]}.pt') for t, _, _ in MEMBERS}
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
    kept_idx = torch.tensor([ci[c] for c in seen]).to(dev)
    other_idx = torch.tensor([i for i in range(NCLS) if classes[i] not in set(seen)]).to(dev)
    TseenTax = TtH[kept_idx]
    LAM = 4.0
    print(f'seen classes: {S} | unseen classes: {len(other_idx)}')

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
        out = torch.empty(qF.shape[0], S, device=dev)
        for i in range(0, qF.shape[0], 2000):
            e = qF[i:i + 2000]
            sim = e @ TF.t()
            cmax = torch.full((e.shape[0], S), -1e9, device=dev)
            cmax.scatter_reduce_(1, TL.unsqueeze(0).expand(e.shape[0], -1), sim, reduce='amax')
            sc = e @ P.t() + 2.0 * cmax
            if hspace:
                sc = sc + LAM * (e @ TseenTax.t())
            out[i:i + 2000] = sc
        return out

    test = {t: load(f'outputs/{TRAIN[t].replace("emb_train", "emb_test")}.pt') for t, _, _ in MEMBERS}
    unseen = {t: load(f'outputs/{TRAIN[t].replace("emb_train", "emb_unseen")}.pt') for t, _, _ in MEMBERS}
    tb_test = load('outputs/emb_test_taxabind.pt')
    tb_uns = load('outputs/emb_unseen_taxabind.pt')
    tf = sorted(set.intersection(*[set(f) for (_, _, f) in test.values()], set(tb_test[2])))
    uf = sorted(set.intersection(*[set(f) for (_, _, f) in unseen.values()], set(tb_uns[2])))
    all_files = tf + uf
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

    seen_block = torch.zeros(len(all_files), S, device=dev)
    member_max = {}
    for t, hs, w in MEMBERS:
        if w == 0.0:
            continue
        sc = seen_score(Q[t], *PR[t], hs)
        member_max[t] = sc.max(1).values
        seen_block = seen_block + w * zc(sc)

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
    print(f'crop views aligned n={len(all_files)} fallback_slots={n_fb}', flush=True)

    other_list = other_idx.tolist()
    Nq = len(all_files)
    Cu = len(other_list)
    bank_ct = torch.load(BANK_CTFT, weights_only=False)['bank']
    print(f'scoring ctft crop-max bank Cu={Cu} views={len(VIEW_KEYS)} ...', flush=True)
    Sraw_ct = score_bank_maxviews(view_tensors, bank_ct, other_list)
    print('scoring 336 bank ...', flush=True)
    Sraw_336 = score_bank(Q['fullft336shift'], torch.load(BANK_336, weights_only=False)['bank'], other_list)

    def b2_leg(proto_path, emb_test_path, emb_uns_path):
        b2_test = load(emb_test_path)
        b2_uns = load(emb_uns_path)
        Qb2 = torch.cat([
            torch.stack([b2_test[1][b2_test[0][fn]] for fn in tf]),
            torch.stack([b2_uns[1][b2_uns[0][fn]] for fn in uf]),
        ]).to(dev)
        b2P = F.normalize(torch.load(proto_path, weights_only=False)['protos'].float(), dim=-1).to(dev)
        Sraw_b = torch.full((Nq, Cu), -1e4, device=dev)
        for j, gidx in enumerate(other_list):
            pvec = b2P[gidx]
            if pvec.norm() < 0.5:
                continue
            Sraw_b[:, j] = Qb2 @ pvec
        return dbnorm(Sraw_b), Qb2

    b2_frozen_leg, Qb2f = b2_leg(B2_PROTO_FROZEN, 'outputs/emb_test_bioclip2.pt', 'outputs/emb_unseen_bioclip2.pt')
    b2_lora_leg, Qb2l = b2_leg(B2_PROTO_LORA, 'outputs/emb_test_bioclip2_lora_v2.pt', 'outputs/emb_unseen_bioclip2_lora_v2.pt')
    legs = {
        't_ctft_taxon': dbnorm(Q['ctftshift'] @ TtH[other_idx].t()),
        't_L_name': dbnorm(Q['L'] @ TnL[other_idx].t()),
        't_336v2_taxon': dbnorm(Q['fullft336_v2'] @ TtH[other_idx].t()),
        't_ftshift_taxon': dbnorm(Q['ftshift'] @ TtH[other_idx].t()),
        't_ctft_taxctx': dbnorm(Q['ctftshift'] @ TTX[other_idx].t()),
        't_taxabind': dbnorm(Qtb @ Ttb[other_idx].t()),
        'bank_ct': dbnorm(Sraw_ct),
        'bank_336': dbnorm(Sraw_336),
        'b2_frozen': b2_frozen_leg,
        'b2_lora': b2_lora_leg,
    }
    fused = sum(w * legs[n] for n, w in zip(LEGS, DEPLOYED_W))
    n_photos = torch.zeros(len(other_list))
    for j, g in enumerate(other_list):
        ph = bank_ct.get(g)
        if ph is not None:
            n_photos[j] = ph.shape[0] if hasattr(ph, 'shape') else len(ph)

    # ---- gate features (identical definition to research/learned_gate_v77.py) ----
    print('scoring gate bank feature (center view) ...', flush=True)
    bank_raw = bank_scores(Q['ctftshift'], bank_ct, other_list)
    tm_ctft = ((Q['ctftshift'] @ TtH[kept_idx].t()).max(1).values
               - (Q['ctftshift'] @ TtH[other_idx].t()).max(1).values)
    tm_full = text_full[:, kept_idx].max(1).values - text_full[:, other_idx].max(1).values
    b2f = proto_max(Qb2f, B2_PROTO_FROZEN, other_idx)
    b2l = proto_max(Qb2l, B2_PROTO_LORA, other_idx)
    X = assemble(seen_block, member_max, tm_ctft, tm_full, bank_raw, b2f, b2l).cpu()
    Z = standardize(X, torch.full((X.shape[0],), 1.0 / X.shape[0]))
    combined = torch.tensor(gate['model'].predict_proba(Z.numpy())[:, 1], dtype=torch.float32).to(dev)

    os.makedirs('submissions', exist_ok=True)
    seen_set = set(seen)
    k_seen = int(round(SEEN_FRAC * len(all_files)))
    thr = torch.topk(combined, k_seen).values.min()
    route_seen = combined >= thr

    pred_idx = torch.empty(len(all_files), dtype=torch.long, device=dev)
    pred_idx[route_seen] = kept_idx[seen_block[route_seen].argmax(1)]
    idx_uns = (~route_seen).nonzero(as_tuple=True)[0]

    print(f'reranking {len(idx_uns)} ejected rows (top-{rr["K"]}) ...', flush=True)
    sub_legs = {n: legs[n][idx_uns] for n in LEGS}
    sub_fused = fused[idx_uns]
    topk = sub_fused.topk(rr['K'], dim=1).indices
    Xr, rnames = pool_features(sub_legs, topk, n_photos.to(dev), sub_fused)
    assert rnames == rr['names'], 'reranker feature definition drifted'
    Xs = Xr.reshape(-1, len(rr['names'])).cpu().numpy()
    # decision_function, NOT predict_proba: on eval the log-odds sit far enough negative that
    # every probability underflows to 0.0 in float32 and argmax degenerates to index 0.
    p = rr['model'].decision_function((Xs - rr['mu']) / rr['sd'])
    sc = torch.tensor(p, dtype=torch.float64).reshape(topk.shape)
    sel = sc.argmax(1).to(topk.device)
    pick = topk[torch.arange(topk.shape[0], device=topk.device), sel]
    pred_idx[idx_uns] = other_idx[pick]
    print(f'  reranker moved {100.0 * (sel != 0).float().mean().item():.1f}% of rows off fusion rank-1',
          flush=True)

    preds = {fn: classes[j] for fn, j in zip(all_files, pred_idx.tolist())}
    assert len(preds) == 35665 and all(c in ci for c in preds.values())

    cov = len(set(p for p in preds.values() if p not in seen_set))
    o_t = sum(1 for fn in tf if preds[fn] not in seen_set)
    o_u = sum(1 for fn in uf if preds[fn] not in seen_set)

    tag = TAG
    json_path = f'outputs/prediction_{tag}.json'
    zip_path = f'outputs/submission_{tag}.zip'
    sub_zip_path = f'submissions/submission_{tag}.zip'
    json.dump(preds, open(json_path, 'w'))
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
        z.write(json_path, arcname='prediction.json')
    shutil.copy(zip_path, sub_zip_path)

    v56 = 'outputs/prediction_v56_overlap_336_f60_tau18.json'
    n_diff = None
    if os.path.exists(v56):
        p56 = json.load(open(v56))
        n_diff = sum(1 for fn in all_files if preds[fn] != p56.get(fn))

    print(f'Tag: {tag}')
    print(f'  Total predictions: {len(preds)} (Coverage: {cov}/{len(other_idx)} unseen classes)')
    print(f'  Test folder -> seen: {100 * (1 - o_t / len(tf)):.2f}% | '
          f'Unseen folder -> unseen: {100 * o_u / len(uf):.2f}%   [v56: 86.71 / 74.49]')
    if n_diff is not None:
        print(f'  Diff vs v56: {n_diff}/{len(all_files)} ({100 * n_diff / len(all_files):.2f}%)')
    print(f'  Zip created: {sub_zip_path}')
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
