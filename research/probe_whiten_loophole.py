"""probe_whiten_loophole.py -- why did v102 whitening look like a no-op?

H1: the embeddings ARE anisotropic, but the kill-test was run in a regime where
    whitening cannot act: fused score = 1*proto + 2*cmax + 4*taxon(raw), then
    per-leg z-scoring in the fusion. The whitened term carries ~1/7 of the score
    mass, and zc absorbs global shifts/scales, so a real geometry change gets
    diluted to ~1 flipped image out of 11,866.
H2: the whitening modes tested were mis-specified for this pipeline
    (PCA dim truncation losing dims; full-rank ZCA never tried).

Probe plan (same split/harness as seen_whiten_v102.py):
  1. measure anisotropy of the trby pool per leg: mean-vector norm, participation
     ratio of covariance eigenvalues, top-k variance share.
  2. SimpleShot NATIVE regime: proto-only argmax accuracy, raw vs center vs std
     vs full-rank ZCA whitening (no dim truncation).
  3. deployed 3-term fused score, raw vs center: accuracy + #flipped argmax rows.
"""
from __future__ import annotations

import json
import pickle
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, 'research')
from common import OUT
from learned_gate_v77 import D, MEMBERS, dev, holdout_split, load_train_embs

torch.set_num_threads(8)
BASE_W = {'ctftshift': 1.0, 'ftshift': 2.5, 'fullft336shift': 2.5}
LAM = 4.0
EPS = 1e-3


def zc(M):
    return (M - M.mean()) / (M.std() + 1e-6)


def anisotropy(X):
    """X: (N,D) train-fold pool. Returns dict of geometry stats."""
    mu = X.mean(0, keepdim=True)
    Xc = X - mu
    idx = torch.randperm(Xc.shape[0], device=Xc.device)[:20000] if Xc.shape[0] > 20000 \
        else torch.arange(Xc.shape[0], device=Xc.device)
    Xs = Xc[idx]
    C = (Xs.t() @ Xs) / (Xs.shape[0] - 1)
    evals = torch.linalg.eigvalsh(C).clamp(min=0).flip(0)
    tot = evals.sum()
    pr = (tot ** 2) / (evals ** 2).sum()          # participation ratio (D if isotropic)
    mu_n = F.normalize(mu.squeeze(0), dim=0)
    return {'mu_norm': mu.norm().item(),
            'mean_cos_to_mu': (F.normalize(X, dim=-1) @ mu_n).mean().item(),
            'participation_ratio': pr.item(), 'dim': X.shape[1],
            'top10_var_share': (evals[:10].sum() / tot).item(),
            'top50_var_share': (evals[:50].sum() / tot).item()}


def fit_whiten_full(Xraw, mode):
    mu = Xraw.mean(0, keepdim=True)
    Xc = Xraw - mu
    if mode == 'center':
        return lambda X: F.normalize(X - mu, dim=-1)
    if mode == 'std':
        sd = Xc.std(0, keepdim=True).clamp(min=EPS)
        return lambda X: F.normalize((X - mu) / sd, dim=-1)
    if mode == 'zca':  # full-rank whitening, no dim truncation
        C = (Xc.t() @ Xc) / (Xc.shape[0] - 1)
        evals, evecs = torch.linalg.eigh(C)
        evals = evals.clamp(min=EPS)
        W = evecs @ torch.diag(evals.rsqrt())
        return lambda X: F.normalize((X - mu) @ W, dim=-1)
    raise ValueError(mode)


def build_leg(t, train, seen, s2i, trby, val_seen):
    idx, feats, _ = train[t]
    TFl, TLl = [], []
    for c in seen:
        for fn in trby[c]:
            if fn not in idx:
                continue
            TFl.append(feats[idx[fn]])
            TLl.append(s2i[c])
    TF_raw = torch.stack(TFl).to(dev)
    TL = torch.tensor(TLl).to(dev)
    qF_raw = torch.stack([feats[idx[fn]] for fn in val_seen]).to(dev)
    return TF_raw, TL, qF_raw

def proto_scores(qF_w, TF_w, TL, S):
    P = torch.zeros(S, TF_w.shape[1], device=dev)
    cnt = torch.zeros(S, device=dev)
    P.index_add_(0, TL, TF_w)
    cnt.index_add_(0, TL, torch.ones_like(TL, dtype=torch.float32))
    P = F.normalize(P / cnt.clamp(min=1).unsqueeze(1), dim=-1)
    return qF_w @ P.t(), P


def member_scores(qF_raw, qF_w, P_w, TF_w, TL, Tseen, S):
    out = torch.empty(qF_raw.shape[0], S, device=dev)
    for i in range(0, qF_raw.shape[0], 2000):
        e_raw = qF_raw[i:i + 2000]
        e_w = qF_w[i:i + 2000]
        sim = e_w @ TF_w.t()
        cmax = torch.full((e_w.shape[0], S), -1e9, device=dev)
        cmax.scatter_reduce_(1, TL.unsqueeze(0).expand(e_w.shape[0], -1), sim, reduce='amax')
        out[i:i + 2000] = e_w @ P_w.t() + 2.0 * cmax + LAM * (e_raw @ Tseen.t())
    return out


def main():
    global yv
    lab = json.load(open(f'{D}/label_train.json'))
    classes = list(pickle.load(open(f'{D}/all_classes.pkl', 'rb')))
    ci = {c: i for i, c in enumerate(classes)}
    train, tb_train, b2f_train, b2l_train = load_train_embs()
    sp = holdout_split(train, tb_train, b2f_train, b2l_train)
    trby = sp['trby']
    val_seen = [f for f, y_ in zip(sp['val_files'], sp['y']) if y_ == 1]
    seen = sorted(sp['kept'])
    s2i = {c: i for i, c in enumerate(seen)}
    S = len(seen)
    kept_idx = torch.tensor([ci[c] for c in seen]).to(dev)
    yv = torch.tensor([s2i[lab[f]] for f in val_seen])
    TtH = F.normalize(torch.load(f'{OUT}/text_emb_h_taxon.pt', weights_only=False)['emb_taxon'].float(), dim=-1).to(dev)
    Tseen = TtH[kept_idx]
    print(f'val_seen {len(val_seen)} rows | {S} classes', flush=True)

    results = {}
    per_leg_fused = {}
    for t in BASE_W:
        TF_raw, TL, qF_raw = build_leg(t, train, seen, s2i, trby, val_seen)
        res = {'anisotropy': anisotropy(TF_raw)}
        print(f'\n=== leg {t} ===\n  geometry: {res["anisotropy"]}', flush=True)

        # ---- proto-only (SimpleShot native regime)
        for mode in ['raw', 'center', 'std', 'zca']:
            if mode == 'raw':
                qF_w, TF_w = qF_raw, TF_raw
            else:
                tr = fit_whiten_full(TF_raw, mode)
                qF_w, TF_w = tr(qF_raw), tr(TF_raw)
            acc = 100 * (proto_scores(qF_w, TF_w, TL, S)[0].argmax(1).cpu() == yv).float().mean().item()
            res[f'proto_only_{mode}'] = acc
            print(f'  proto-only {mode:6s}: {acc:.3f}', flush=True)
        res['proto_lift_center'] = res['proto_only_center'] - res['proto_only_raw']
        res['proto_lift_zca'] = res['proto_only_zca'] - res['proto_only_raw']

        # ---- deployed 3-term fused, raw vs center: how many argmax rows flip?
        _, P_raw = proto_scores(qF_raw, TF_raw, TL, S)
        s_raw = member_scores(qF_raw, qF_raw, P_raw, TF_raw, TL, Tseen, S)
        tr = fit_whiten_full(TF_raw, 'center')
        qF_w, TF_w = tr(qF_raw), tr(TF_raw)
        _, P_w = proto_scores(qF_w, TF_w, TL, S)
        s_ctr = member_scores(qF_raw, qF_w, P_w, TF_w, TL, Tseen, S)
        per_leg_fused[t] = {'raw': s_raw, 'center': s_ctr}
        res['fused_leg_raw'] = 100 * (s_raw.argmax(1).cpu() == yv).float().mean().item()
        res['fused_leg_center'] = 100 * (s_ctr.argmax(1).cpu() == yv).float().mean().item()
        res['flipped_rows'] = int((s_raw.argmax(1).cpu() != s_ctr.argmax(1).cpu()).sum().item())
        print(f'  deployed-formula solo leg: raw {res["fused_leg_raw"]:.3f} vs center '
              f'{res["fused_leg_center"]:.3f} | flipped argmax rows {res["flipped_rows"]}/{len(val_seen)}',
              flush=True)
        results[t] = res
        del TF_raw, TL, qF_raw, s_raw, s_ctr
        torch.cuda.empty_cache()

    # ---- 3-leg fusion raw vs center + flip count
    accs, argm = {}, {}
    for mode in ['raw', 'center']:
        f = sum(BASE_W[t] * zc(per_leg_fused[t][mode]) for t in BASE_W)
        argm[mode] = f.argmax(1).cpu()
        accs[mode] = 100 * (argm[mode] == yv).float().mean().item()
    flips = int((argm['raw'] != argm['center']).sum().item())
    print(f'\n=== 3-leg fused: raw {accs["raw"]:.3f} vs center {accs["center"]:.3f} | '
          f'flipped rows {flips}/{len(yv)} ===', flush=True)
    results['_fusion'] = {'raw': accs['raw'], 'center': accs['center'], 'flips': flips}
    json.dump(results, open(f'{OUT}/probe_whiten_loophole.json', 'w'), indent=2)
    print(f'wrote {OUT}/probe_whiten_loophole.json', flush=True)


if __name__ == '__main__':
    main()
