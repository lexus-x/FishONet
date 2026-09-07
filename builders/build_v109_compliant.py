"""v109-compliant: identical to build_v109_genus_gamble.py EXCEPT the gate is made strictly
per-image (each test image scored independent of every other image in the batch).

Two batch-coupled spots fixed (flagged 2026-09-01, see HANDOFF):
  1. `Z = standardize(X, uniform_weight)` recomputed wz1 mean/std LIVE on the eval batch's own
     X -- also a train/test mismatch, since the gate model was TRAINED on features standardized
     against the HOLDOUT's stats (learned_gate_v77.py main(), not the eval batch's).
     Fixed: reuse the frozen (mu, sd) from research/compute_frozen_gate_constants.py
     (outputs/frozen_gate_v109.json), fit once on the holdout, never touching eval data.
  2. `thr = topk(combined, 0.60*N).values.min()` picked a RANK cutoff on the live eval batch to
     force exactly 60% routed to "seen" -- requires seeing the whole batch first.
     Fixed: reuse the ABSOLUTE probability threshold calibrated once on the holdout
     (same json), applied per image as `combined[i] >= FROZEN_THR`.

Run research/compute_frozen_gate_constants.py first if outputs/frozen_gate_v109.json is stale.
Compliance check only -- writes a zip locally, does NOT submit to Codabench.

  conda activate onet && python builders/build_v109_compliant.py
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
from rerank_seen_v83 import cand_features as cand_features_v83  # noqa: E402
from rerank_seen_v83 import K as K_SEEN  # noqa: E402

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
GATE_PKL = os.environ.get('GATE_PKL', 'outputs/learned_gate_v79.pkl')
RERANK_PKL = os.environ.get('RERANK_PKL', 'outputs/rerank_unseen_v82_leakfree.pkl')
RERANK_SEEN_GENUS_PKL = os.environ.get('RERANK_SEEN_GENUS_PKL', 'outputs/rerank_seen_genus_v103.pkl')
TAG = os.environ.get('TAG', 'v109_compliant')
SEEN_FRAC = float(os.environ.get('SEEN_FRAC', '0.60'))
FROZEN_GATE_JSON = os.environ.get('FROZEN_GATE_JSON', 'outputs/frozen_gate_v109.json')


def standardize_frozen(X, mu_sd):
    return torch.stack([(X[:, j] - mu) / sd for j, (mu, sd) in enumerate(mu_sd)], dim=1)
GENUS_LEG = 'ctftshift'

MEMBERS = [('ctftshift', True, 1.0), ('ftshift', True, 2.5), ('fullft336shift', True, 2.5),
           ('L', False, 0.0), ('fullft336_v2', True, 0.0)]
TRAIN = {'ctftshift': 'emb_train_ctftshift', 'ftshift': 'emb_train_ftshift',
         'fullft336shift': 'emb_train_fullft336shift', 'L': 'emb_train',
         'fullft336_v2': 'emb_train_fullft336_v2'}
DEPLOYED = {'ctftshift': 1.0, 'ftshift': 2.5, 'fullft336shift': 2.5}
LAM = 4.0


def load(p):
    d = torch.load(p, weights_only=False)
    return {fn: i for i, fn in enumerate(d['files'])}, F.normalize(d['feats'].float(), dim=-1), list(d['files'])


def zc(M):
    return (M - M.mean()) / (M.std() + 1e-6)


def dbnorm(S, tc=0.05, tr=0.5):
    return F.log_softmax(S / tc, dim=0) + F.log_softmax(S / tr, dim=1)


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


def components(Q, P, TF, TL, Tseen, S):
    n = Q.shape[0]
    ps = torch.empty(n, S, device=dev)
    cm = torch.empty(n, S, device=dev)
    for i in range(0, n, 2000):
        e = Q[i:i + 2000]
        ps[i:i + 2000] = e @ P.t()
        sim = e @ TF.t()
        c = torch.full((e.shape[0], S), -1e9, device=dev)
        c.scatter_reduce_(1, TL.unsqueeze(0).expand(e.shape[0], -1), sim, reduce='amax')
        cm[i:i + 2000] = c
    return ps, cm, Q @ Tseen.t()


def cand_features_genus(sig, topk, n_train, block, signal_names):
    nq, k = topk.shape
    C = block.shape[1]
    cols, names = [], []
    for name in signal_names:
        M = sig[name]
        v = torch.gather(M, 1, topk)
        cols += [v - M.max(1, keepdim=True).values,
                 (v - M.mean(1, keepdim=True)) / (M.std(1, keepdim=True) + 1e-6)]
        names += [f'{name}_gapmax', f'{name}_z']
        r = torch.empty(nq, k, device=M.device)
        for s in range(0, nq, 256):
            e = min(s + 256, nq)
            r[s:e] = (M[s:e].unsqueeze(1) > v[s:e].unsqueeze(2)).sum(2).float()
        cols.append(r / C)
        names.append(f'{name}_pctrank')
    cols += [torch.arange(k, device=block.device).float().expand(nq, k) / k,
             torch.log1p(n_train[topk])]
    names += ['block_rank', 'log_ntrain']
    return torch.stack(cols, dim=2), names


def main():
    print(f'=== Building v109: v83 + genus backoff (LAM_G=3.0) on seen route (f={SEEN_FRAC} fixed) ===', flush=True)
    rgz = pickle.load(open(RERANK_SEEN_GENUS_PKL, 'rb'))
    lam_g = rgz['lam_g']
    signal_names = rgz['signals']
    print(f'genus reranker: K={rgz["K"]} feats={len(rgz["names"])} lam_g={lam_g} '
          f'signals={signal_names}', flush=True)
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
    classes = list(pickle.load(open('data/dl/all_classes.pkl', 'rb')))
    ci = {c: i for i, c in enumerate(classes)}
    NCLS = len(classes)
    TtH = F.normalize(txtHt['emb_taxon'].float(), dim=-1).to(dev)
    TnL = F.normalize(txtL['emb_name'].float(), dim=-1).to(dev)
    TTX = F.normalize(torch.load('outputs/text_emb_h_promptens.pt', weights_only=False)['emb_taxctx'].float(), dim=-1).to(dev)
    Ttb = F.normalize(torch.load('outputs/text_emb_taxabind_taxctx.pt', weights_only=False)['emb_taxctx'].float(), dim=-1).to(dev)
    lab = json.load(open(f'{D}/label_train.json'))

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

    # genus prototypes, built from ALL training images (by[c]) of the GENUS_LEG encoder
    genus_of = {c: c.split()[0] for c in seen}
    genus_list = sorted(set(genus_of.values()))
    g2i = {g: i for i, g in enumerate(genus_list)}
    G = len(genus_list)
    species_genus_idx = torch.tensor([g2i[genus_of[c]] for c in seen]).to(dev)
    _, TF_g, TL_g = PR[GENUS_LEG]
    TL_genus = species_genus_idx[TL_g]
    d = TF_g.shape[1]
    Pg = torch.zeros(G, d, device=dev)
    cntg = torch.zeros(G, device=dev)
    Pg.scatter_add_(0, TL_genus.unsqueeze(1).expand(-1, d), TF_g)
    cntg.scatter_add_(0, TL_genus, torch.ones_like(TL_genus, dtype=torch.float32))
    Pg = F.normalize(Pg / cntg.clamp(min=1).unsqueeze(1), dim=-1)
    print(f'genus groups: {G} (avg {S / G:.2f} species/genus)', flush=True)

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
    sig = {}
    n_train = torch.tensor([len(by[c]) for c in seen], dtype=torch.float32, device=dev)
    for t, hs, w in MEMBERS:
        if w == 0.0:
            continue
        P_, TF_, TL_ = PR[t]
        ps, cm, tx = components(Q[t].to(dev), P_, TF_, TL_, TseenTax, S)
        sig[f'{t}_proto'], sig[f'{t}_cmax'], sig[f'{t}_taxon'] = ps, cm, tx
        sc = ps + 2.0 * cm + (LAM * tx if hs else 0.0)
        member_max[t] = sc.max(1).values
        zsc = zc(sc)
        seen_block = seen_block + DEPLOYED.get(t, w) * zsc

    # genus bonus for the real 35,665-image eval batch
    q_g = Q[GENUS_LEG]
    sim_g = q_g @ TF_g.t()
    cmaxg = torch.full((q_g.shape[0], G), -1e9, device=dev)
    cmaxg.scatter_reduce_(1, TL_genus.unsqueeze(0).expand(q_g.shape[0], -1), sim_g, reduce='amax')
    genus_raw_G = q_g @ Pg.t() + 2.0 * cmaxg
    genus_bonus = genus_raw_G[:, species_genus_idx]
    sig['genus'] = genus_bonus
    fused_genus = seen_block + lam_g * zc(genus_bonus)
    sig['block'] = fused_genus

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

    print('scoring gate bank feature (center view) ...', flush=True)
    bank_raw = bank_scores(Q['ctftshift'], bank_ct, other_list)
    tm_ctft = ((Q['ctftshift'] @ TtH[kept_idx].t()).max(1).values
               - (Q['ctftshift'] @ TtH[other_idx].t()).max(1).values)
    tm_full = text_full[:, kept_idx].max(1).values - text_full[:, other_idx].max(1).values
    b2f = proto_max(Qb2f, B2_PROTO_FROZEN, other_idx)
    b2l = proto_max(Qb2l, B2_PROTO_LORA, other_idx)
    # Gate features use the ORIGINAL seen_block/member_max -- routing (TPR/TNR) is
    # bit-identical to v83; only the seen-route candidate SET below (genus-augmented) changes.
    frozen = json.load(open(FROZEN_GATE_JSON))
    assert frozen['feats'] == FEATS, 'frozen gate constants feature order drifted'
    assert frozen['gate_pkl'] == GATE_PKL, 'frozen gate constants fit for a different gate pickle'
    X = assemble(seen_block, member_max, tm_ctft, tm_full, bank_raw, b2f, b2l).cpu()
    Z = standardize_frozen(X, frozen['mu_sd'])   # per-image: fixed (mu,sd), no eval-batch stats
    combined = torch.tensor(gate['model'].predict_proba(Z.numpy())[:, 1], dtype=torch.float32).to(dev)

    os.makedirs('submissions', exist_ok=True)
    seen_set = set(seen)
    FROZEN_THR = float(os.environ.get('FROZEN_THR_OVERRIDE', frozen['thr']))  # per-image: fixed cutoff
    route_seen = combined >= FROZEN_THR
    print(f'frozen gate: thr={FROZEN_THR:.6f} (fixed constant, not rank-forced) | '
          f'routed-seen this batch = {100 * route_seen.float().mean().item():.2f}%', flush=True)

    pred_idx = torch.empty(len(all_files), dtype=torch.long, device=dev)
    idx_seen = route_seen.nonzero(as_tuple=True)[0]
    print(f'reranking {len(idx_seen)} seen-route rows (top-{rgz["K"]}, genus-augmented candidate set) ...',
          flush=True)
    topk_s = fused_genus[idx_seen].topk(rgz['K'], dim=1).indices
    Xsn, snames = cand_features_genus({n: sig[n][idx_seen] for n in signal_names}, topk_s, n_train,
                                       seen_block[idx_seen], signal_names)
    assert snames == rgz['names'], 'genus seen-reranker feature definition drifted'
    ps_ = rgz['model'].decision_function(
        (Xsn.reshape(-1, len(rgz['names'])).cpu().numpy() - rgz['mu']) / rgz['sd'])
    sel_s = torch.tensor(ps_, dtype=torch.float64).reshape(topk_s.shape).argmax(1).to(dev)
    pred_idx[idx_seen] = kept_idx[topk_s[torch.arange(topk_s.shape[0], device=dev), sel_s]]
    print(f'  genus reranker moved {100.0 * (sel_s != 0).float().mean().item():.1f}% off rank-1',
          flush=True)
    n_top1_diff = (fused_genus[idx_seen].argmax(1) != seen_block[idx_seen].argmax(1)).float().mean().item()
    print(f'  genus bonus moved {100.0 * n_top1_diff:.2f}% of seen-route top-1 vs un-augmented', flush=True)
    idx_uns = (~route_seen).nonzero(as_tuple=True)[0]
    del sig, fused_genus, seen_block, member_max, Xsn, topk_s, sel_s, ps_
    del genus_bonus, genus_raw_G, cmaxg, sim_g, q_g, Pg, cntg, TF_g, TL_g, TL_genus
    del PR, train
    torch.cuda.empty_cache()

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
    del Sraw_ct, Sraw_336
    n_photos = torch.zeros(len(other_list))
    for j, g in enumerate(other_list):
        ph = bank_ct.get(g)
        if ph is not None:
            n_photos[j] = ph.shape[0] if hasattr(ph, 'shape') else len(ph)

    print(f'reranking {len(idx_uns)} ejected rows (top-{rr["K"]}) ...', flush=True)
    sub_legs = {n: legs[n][idx_uns] for n in LEGS}
    sub_fused = fused[idx_uns]
    del legs, fused
    topk = sub_fused.topk(rr['K'], dim=1).indices
    Xr, rnames = pool_features(sub_legs, topk, n_photos.to(dev), sub_fused)
    assert rnames == rr['names'], 'reranker feature definition drifted'
    Xs = Xr.reshape(-1, len(rr['names'])).cpu().numpy()
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

    v83 = 'outputs/prediction_v83_rerank_both_f60.json'
    n_diff = None
    if os.path.exists(v83):
        p83 = json.load(open(v83))
        n_diff = sum(1 for fn in all_files if preds[fn] != p83.get(fn))

    v109 = 'outputs/prediction_v109_genus_gamble.json'
    n_diff_v109 = None
    if os.path.exists(v109):
        p109 = json.load(open(v109))
        n_diff_v109 = sum(1 for fn in all_files if preds[fn] != p109.get(fn))

    print(f'Tag: {tag}')
    print(f'  Total predictions: {len(preds)} (Coverage: {cov}/{len(other_idx)} unseen classes)')
    print(f'  Test folder -> seen: {100 * (1 - o_t / len(tf)):.2f}% | '
          f'Unseen folder -> unseen: {100 * o_u / len(uf):.2f}%')
    if n_diff is not None:
        print(f'  Diff vs v83: {n_diff}/{len(all_files)} ({100 * n_diff / len(all_files):.2f}%)')
    if n_diff_v109 is not None:
        print(f'  Diff vs v109 (scored 53.736% real): {n_diff_v109}/{len(all_files)} '
              f'({100 * n_diff_v109 / len(all_files):.2f}%) -- isolates the gate-freeze effect')
    print(f'  Zip created: {sub_zip_path}')
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
