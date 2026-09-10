"""v102: embedding whitening / SimpleShot-style feature-space standardization before
prototype matching, on the SEEN closed-set classifier.

Every prior seen-accuracy lever (v95 iNat evidence, v98 reweighting) changed how the
THREE members' SCORES combine. None has touched the raw embedding GEOMETRY that proto/
cmax matching runs cosine similarity in. SimpleShot (Wang et al. 2019) found few-shot
nearest-prototype classifiers on raw (even L2-normalized) embeddings are measurably
worse than on CENTERED + (optionally variance-normalized / PCA-whitened) embeddings,
because a handful of high-variance / high-mean dimensions dominate the raw cosine dot
product and swamp the class-discriminative directions.

This fits a transform per leg (ctftshift/ftshift/fullft336shift) ON THE TRAIN-FOLD POOL
ONLY (trby, identical leak-free split as v95/v97/v98's holdout_split()) and re-runs the
proto+cmax matching in the transformed space. The taxon (image-text CLIP) term is left
on RAW embeddings deliberately -- centering by an IMAGE-only mean would misalign the
image and text halves of a joint CLIP space that the taxon term depends on, and that
cross-modal term has never been implicated in any accuracy complaint.

Modes (fit on pooled trby embeddings per leg):
  center      : x' = normalize(x - mu)                         (SimpleShot CL2N)
  std         : x' = normalize((x - mu) / std)                 (CL2N + per-dim variance norm)
  pca{K}      : x' = normalize((x - mu) @ V_K / sqrt(eig_K))   (top-K PCA whitening)

Pass/fail bar: >=+1.0pt over the current best-in-repo (v83 seen re-ranker, 91.674%
holdout) on the SAME val_seen pool / SAME metric (top-1 argmax accuracy against the
gold class), as measured by learned_gate_v77.holdout_split -- i.e. >=92.674% holdout
for the best (mode, k) combination on the full 3-leg fused argmax. Also reports vs the
two v98 reference points (deployed 3-leg 89.87%, ctftshift solo 91.20%) so a partial
win is visible even if the harder re-ranker bar isn't cleared.

Why this is more likely to TRANSFER to real than v98's reweight:
  - v98 (linear reweight of the SAME fixed prototype geometry) shipped +0.0pt holdout
    lift and NEGATIVE real (53.60% vs v83's 53.70%) -- reweighting a saturated ordering
    doesn't change which images are hard, so a_cond/b_cond don't move (repo's
    "re-weighting doesn't compound" law, HANDOFF 2026-08-29).
  - Whitening is not a reweight: it changes the SIMILARITY GEOMETRY itself inside each
    leg's native embedding space, before scores even form -- something no linear
    combination of the existing 3 leg-scores can reach. If it lifts closed-set argmax
    accuracy, that is NEW per-image evidence (which queries the classifier gets right)
    rather than a re-weighted view of old evidence -- the repo's own "new information
    compounds" finding says that is the class of change likeliest to also move
    a_cond/b_cond on real eval, not just slide a holdout number.
  - The fit uses only the train-fold pool already used identically to build proto/cmax
    -- no new leakage surface vs the existing harness.

Known risk: CLIP-style contrastively-trained embeddings are already closer to isotropic
than the ResNet features SimpleShot was demonstrated on, so the gain may be small or
zero here -- this is a real possibility, not just a formality, and is exactly why this
is a cheap holdout-only check before any real submission slot is spent.

  conda activate onet && python research/seen_embedding_whitening_v102.py
"""
from __future__ import annotations

import json
import pickle
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, 'research')
from common import OUT
from learned_gate_v77 import D, MEMBERS, dev, holdout_split, load_train_embs, zc

LAM = 4.0
BASE_W = {'ctftshift': 1.0, 'ftshift': 2.5, 'fullft336shift': 2.5}
DEPLOYED_3LEG = 89.87          # v98 reference point, same split/harness
CTFT_SOLO = 91.20              # v98 reference point, same split/harness
RERANK_BEST = 91.674           # v83 seen re-ranker, current best-in-repo
KILL_LIFT = 1.0                # bar is +1.0pt over RERANK_BEST
EPS = 1e-3                     # variance-floor knob for std/pca modes -- raise if a
                                # near-zero-variance dim blows up a whitened score
PCA_KS = [64, 128, 256]        # skipped automatically if >= embedding dim


def fit_whiten(Xraw, mode, k=None):
    """Fit a whitening transform on a pooled TRAIN-FOLD embedding matrix. Returns a
    callable applied identically to any other embedding matrix (queries, prototype
    inputs) in that leg's raw space. mode in {'center', 'std', 'pca'}."""
    mu = Xraw.mean(0, keepdim=True)
    Xc = Xraw - mu
    if mode == 'center':
        return lambda X: F.normalize(X - mu, dim=-1)
    if mode == 'std':
        sd = Xc.std(0, keepdim=True).clamp(min=EPS)
        return lambda X: F.normalize((X - mu) / sd, dim=-1)
    if mode == 'pca':
        k = min(k, Xc.shape[1] - 1, Xc.shape[0] - 1)
        _, S, V = torch.pca_lowrank(Xc, q=k, niter=4)
        var = (S ** 2) / max(Xc.shape[0] - 1, 1)
        Wt = V / torch.sqrt(var + EPS).unsqueeze(0)          # (D, k)
        return lambda X: F.normalize((X - mu) @ Wt, dim=-1)
    raise ValueError(mode)


def member_scores_whiten(qF_raw, qF_w, P_w, TF_w, TL, Tseen, S):
    """Deployed base with proto+cmax computed in the WHITENED space; the taxon term is
    left on the RAW embedding (cross-modal image-text term, must stay in the native
    joint CLIP space -- an image-only mean/whitener would misalign it)."""
    out = torch.empty(qF_raw.shape[0], S, device=dev)
    for i in range(0, qF_raw.shape[0], 2000):
        e_raw = qF_raw[i:i + 2000]
        e_w = qF_w[i:i + 2000]
        sim = e_w @ TF_w.t()
        cmax = torch.full((e_w.shape[0], S), -1e9, device=dev)
        cmax.scatter_reduce_(1, TL.unsqueeze(0).expand(e_w.shape[0], -1), sim, reduce='amax')
        out[i:i + 2000] = e_w @ P_w.t() + 2.0 * cmax + LAM * (e_raw @ Tseen.t())
    return out


def build_leg(t, train, seen, s2i, trby, val_seen, kept_idx, TtH):
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
    Tseen = TtH[kept_idx]
    return TF_raw, TL, qF_raw, Tseen


def score_mode(TF_raw, TL, qF_raw, Tseen, S, transform):
    if transform is None:
        qF_w, TF_w = qF_raw, TF_raw
    else:
        qF_w, TF_w = transform(qF_raw), transform(TF_raw)
    P_w = torch.zeros(S, TF_w.shape[1], device=dev)
    cnt = torch.zeros(S, device=dev)
    P_w.index_add_(0, TL, TF_w)
    cnt.index_add_(0, TL, torch.ones_like(TL, dtype=torch.float32))
    P_w = F.normalize(P_w / cnt.clamp(min=1).unsqueeze(1), dim=-1)
    return member_scores_whiten(qF_raw, qF_w, P_w, TF_w, TL, Tseen, S)


def main():
    torch.set_num_threads(8)
    lab = json.load(open(f'{D}/label_train.json'))
    classes = list(pickle.load(open(f'{D}/all_classes.pkl', 'rb')))
    ci = {c: i for i, c in enumerate(classes)}
    train, tb_train, b2f_train, b2l_train = load_train_embs()
    sp = holdout_split(train, tb_train, b2f_train, b2l_train)
    trby, val_seen = sp['trby'], [f for f, yv in zip(sp['val_files'], sp['y']) if yv == 1]
    seen = sorted(sp['kept'])
    s2i = {c: i for i, c in enumerate(seen)}
    S = len(seen)
    kept_idx = torch.tensor([ci[c] for c in seen]).to(dev)
    yv = torch.tensor([s2i[lab[f]] for f in val_seen])
    print(f'val_seen rows {len(val_seen)} | seen classes {S}', flush=True)

    TtH = F.normalize(torch.load(f'{OUT}/text_emb_h_taxon.pt', weights_only=False)['emb_taxon'].float(), dim=-1).to(dev)

    legs = {}
    for t, _, _ in MEMBERS:
        if t not in BASE_W:
            continue
        legs[t] = build_leg(t, train, seen, s2i, trby, val_seen, kept_idx, TtH)
        print(f'  {t}: pooled train-fold {legs[t][0].shape[0]} imgs, dim={legs[t][0].shape[1]}', flush=True)

    modes = [('raw', None, None), ('center', 'center', None), ('std', 'std', None)]
    for k in PCA_KS:
        modes.append((f'pca{k}', 'pca', k))

    per_leg_scores = {name: {} for name, _, _ in modes}
    for name, kind, k in modes:
        for t, (TF_raw, TL, qF_raw, Tseen) in legs.items():
            transform = None if kind is None else fit_whiten(TF_raw, kind, k)
            per_leg_scores[name][t] = score_mode(TF_raw, TL, qF_raw, Tseen, S, transform)
            del transform
        torch.cuda.empty_cache()

    print('\n=== solo-leg accuracy by whitening mode ===', flush=True)
    print(f'{"mode":10s}' + ''.join(f'{t:>16s}' for t in BASE_W), flush=True)
    for name, _, _ in modes:
        accs = [100 * (per_leg_scores[name][t].argmax(1).cpu() == yv).float().mean().item()
                for t in BASE_W]
        print(f'{name:10s}' + ''.join(f'{a:16.2f}' for a in accs), flush=True)

    def fused_acc(name):
        f = sum(BASE_W[t] * zc(per_leg_scores[name][t]) for t in BASE_W)
        return 100 * (f.argmax(1).cpu() == yv).float().mean().item()

    print('\n=== 3-leg fused accuracy (fixed 1.0/2.5/2.5 weights, unchanged) ===', flush=True)
    rows = {}
    for name, _, _ in modes:
        rows[name] = fused_acc(name)
        print(f'  {name:10s} -> {rows[name]:6.3f}', flush=True)

    base = rows['raw']
    best = max(rows, key=rows.get)
    lift_vs_raw = rows[best] - base
    lift_vs_rerank = rows[best] - RERANK_BEST
    print(f'\nraw parity check: {base:.3f} (expect ~{DEPLOYED_3LEG:.2f} from v98)', flush=True)
    print(f'best mode = {best}  fused acc = {rows[best]:.3f}  '
          f'lift vs raw = {lift_vs_raw:+.3f}  lift vs rerank-best(91.674) = {lift_vs_rerank:+.3f}', flush=True)
    verdict = ('CLEARS -- whitening beats the re-ranker ceiling by >=1.0pt'
               if rows[best] >= RERANK_BEST + KILL_LIFT
               else 'DEAD -- whitening does not clear +1.0pt over 91.674%')
    print(f'VERDICT: {verdict}', flush=True)
    json.dump({'rows': rows, 'raw_parity': base, 'best': best, 'lift_vs_raw': lift_vs_raw,
               'lift_vs_rerank': lift_vs_rerank, 'verdict': verdict,
               'rerank_best': RERANK_BEST, 'kill_lift': KILL_LIFT,
               'deployed_3leg_ref': DEPLOYED_3LEG, 'ctft_solo_ref': CTFT_SOLO},
              open(f'{OUT}/seen_whiten_v102.json', 'w'), indent=2)
    print(f'wrote {OUT}/seen_whiten_v102.json', flush=True)


if __name__ == '__main__':
    main()
