"""Unified single-head 2-stage pipeline vs the production hard-gate routing.

Requested shape (drop the separate seen/unseen heads):
  Stage 1  vision<->prototype comparison over train-seen classes -> top-k shortlist
  Stage 2  ONE score function over (top-k seen + all non-train classes) -> argmax

Arms, all on the same pseudo-unseen holdout and the same legs:
  A   production hard gate (v50 shape): seen head | unseen head, f-routed, Sinkhorn
  B   requested spec exactly: stage-2 score = vision<->text only
  C   unified: stage-2 = vision<->text + BioCLIP-2 / ToL image-proto legs on the FULL
      class space (same legs on seen and unseen columns -- the symmetry arm B lacks)
  C+bank  C plus the iNat photo-bank leg. Flagged: that bank covers 0/4636 holdout-seen
      classes, so it can only ever fire on one side of the partition.

  python research/eval_unified_2stage.py            # holdout accuracy (arms A/B/C)
  python research/eval_unified_2stage.py --real     # 35,665-image routing diagnostic

Holdout overstates real (HANDOFF sec 7) and this change class -- soft/unified routing --
has produced two holdout mirages already (v34 recover +3.51 holdout -> -0.07 real;
soft_b1.05 +1.91 holdout -> killed). Read the routing diagnostic, not just the accuracy.
"""
import argparse
import json
import os
import pickle
import sys
from collections import defaultdict

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT, DATA, DEV  # noqa: E402

dev = DEV
D = os.path.join(DATA, 'dl')

# production constants (builders/build_v50_chase53.py)
MEMBERS = [('ctftshift', True, 1.0), ('ftshift', True, 2.5), ('fullft336shift', True, 2.5),
           ('L', False, 0.0), ('fullft336_v2', True, 0.0)]
EMB = {'ctftshift': 'ctftshift', 'ftshift': 'ftshift', 'fullft336shift': 'fullft336shift',
       'L': '', 'fullft336_v2': 'fullft336_v2'}
LAM, TAXABIND_W, TOPM = 4.0, 1.0, 4
IMG_W, B2_WF, B2_WL = 4.0, 2.5, 2.0
B2_PROTO_FROZEN = os.path.join(OUT, 'inat_tol_merged_b2_a05.pt')
B2_PROTO_LORA = os.path.join(OUT, 'inat_tol_merged_b2lora_a0.5.pt')
BANK_PATH = os.path.join(OUT, 'inat_photo_bank_ctftshift.pt')
W_SEEN, W_UNS = 0.5635, 0.4365
KS = [5, 10, 25, 50, 100]
# measured Codabench anchor: submissions/submission_v50_f60_tau18.zip (HANDOFF sec 1)
REAL_ANCHOR = {'arm': 'A hard gate f=0.60 +sinkhorn', 'seen': 76.30989699955217,
               'unseen': 19.340955806783144, 'overall': 51.44259077526987}


def load(p):
    d = torch.load(p, weights_only=False)
    return {fn: i for i, fn in enumerate(d['files'])}, F.normalize(d['feats'].float(), dim=-1), list(d['files'])


def emb_path(split, tag):
    return os.path.join(OUT, f'emb_{split}_{tag}.pt' if tag else f'emb_{split}.pt')


def zc(M):
    return (M - M.mean()) / (M.std() + 1e-6)


def z1(v):
    return (v - v.mean()) / (v.std() + 1e-6)


def dbnorm(S, tc=0.05, tr=0.5):
    return F.log_softmax(S / tc, dim=0) + F.log_softmax(S / tr, dim=1)


def sinkhorn(logits, n_iter=60, tau=1.8):
    P = torch.softmax(logits / tau, dim=1)
    target_col = P.shape[0] / P.shape[1]
    for _ in range(n_iter):
        P = P / P.sum(dim=1, keepdim=True).clamp(min=1e-9)
        P = P * (target_col / P.sum(dim=0, keepdim=True).clamp(min=1e-9))
    return P


def proto_leg_raw(Q, proto_path, cols):
    """[N, len(cols)] cosine to per-class prototypes, -1e4 where the class has no proto."""
    P = F.normalize(torch.load(proto_path, weights_only=False)['protos'].float(), dim=-1).to(dev)
    Pc = P[cols]
    has = Pc.norm(dim=-1) > 0.5
    return (Q @ Pc.t()).masked_fill(~has.unsqueeze(0), -1e4), has


def bank_leg_raw(Q, cols):
    """[N, len(cols)] top-TOPM mean cosine against the iNat photo bank."""
    bank = torch.load(BANK_PATH, weights_only=False)['bank']
    S = torch.full((Q.shape[0], len(cols)), -1e4, device=dev)
    has = torch.zeros(len(cols), dtype=torch.bool)
    for j, gidx in enumerate(cols.tolist()):
        photos = bank.get(gidx)
        if photos is None or (hasattr(photos, 'numel') and photos.numel() == 0):
            continue
        if not isinstance(photos, torch.Tensor):
            photos = torch.stack(photos)
        photos = F.normalize(photos.float(), dim=-1).to(dev)
        sim = Q @ photos.t()
        S[:, j] = sim.topk(min(TOPM, sim.shape[1]), dim=1).values.mean(dim=1)
        has[j] = True
    return S, has


def train_photo_leg_raw(Q, TF, TL, S_cols):
    """Same statistic as bank_leg_raw, but against a class's own TRAINING images.

    Lets one uniform rule -- 'top-TOPM mean cosine to this class's reference photos' --
    cover seen classes (train images) and novel classes (iNat photos) alike.
    """
    out = torch.full((Q.shape[0], S_cols), -1e4, device=dev)
    sim = Q @ TF.t()
    order = TL.argsort()
    bounds = torch.searchsorted(TL[order].contiguous(), torch.arange(S_cols + 1, device=dev))
    for s in range(S_cols):
        idx = order[bounds[s]:bounds[s + 1]]
        if idx.numel() == 0:
            continue
        sub = sim[:, idx]
        out[:, s] = sub.topk(min(TOPM, sub.shape[1]), dim=1).values.mean(dim=1)
    del sim
    return out


def neutral(leg, has):
    """Uncovered classes carry NO evidence (per-row mean of covered columns).

    Production masks them to -1e4, which is fine inside a dedicated unseen head -- the
    class is simply unreachable there anyway. In a unified score it would delete the
    class from the whole candidate space, so a text-only hit could never win.
    """
    out = leg.clone()
    out[:, ~has] = leg[:, has].mean(1, keepdim=True)
    return out


def two_stage(seen_block, score, seen_gidx, other_gidx, k, NCLS):
    """Stage 1 shortlist on vision protos; stage 2 argmax of ONE score over shortlist + all unseen."""
    logits = torch.full((seen_block.shape[0], NCLS), -1e9, device=dev)
    logits[:, other_gidx] = score[:, other_gidx]
    short = seen_gidx[seen_block.topk(min(k, seen_block.shape[1]), dim=1).indices]  # [N,k] global
    logits.scatter_(1, short, score.gather(1, short))
    return logits.argmax(1)


def route_stats(pred_gidx, seen_mask_global, n_seen_rows):
    """(% of seen-origin rows landing on a seen class, % of unseen-origin rows landing on unseen)."""
    on_seen = seen_mask_global[pred_gidx]
    return (100 * on_seen[:n_seen_rows].float().mean().item(),
            100 * (~on_seen[n_seen_rows:]).float().mean().item())


# --------------------------------------------------------------------------- holdout

def run_holdout():
    classes = list(pickle.load(open(os.path.join(D, 'all_classes.pkl'), 'rb')))
    ci = {c: i for i, c in enumerate(classes)}
    NCLS = len(classes)
    lab = json.load(open(os.path.join(D, 'label_train.json')))

    TtH = F.normalize(torch.load(os.path.join(OUT, 'text_emb_h_taxon.pt'), weights_only=False)['emb_taxon'].float(), dim=-1).to(dev)
    TnL = F.normalize(torch.load(os.path.join(OUT, 'text_emb.pt'), weights_only=False)['emb_name'].float(), dim=-1).to(dev)
    TTX = F.normalize(torch.load(os.path.join(OUT, 'text_emb_h_promptens.pt'), weights_only=False)['emb_taxctx'].float(), dim=-1).to(dev)
    Ttb = F.normalize(torch.load(os.path.join(OUT, 'text_emb_taxabind_taxctx.pt'), weights_only=False)['emb_taxctx'].float(), dim=-1).to(dev)

    train = {t: load(emb_path('train', EMB[t])) for t, _, _ in MEMBERS}
    train['taxabind'] = load(emb_path('train', 'taxabind'))
    train['b2'] = load(emb_path('train', 'bioclip2'))
    train['b2l'] = load(emb_path('train', 'bioclip2_lora_v2'))
    common = None
    for _, _, files in train.values():
        common = set(files) if common is None else (common & set(files))
    common = [fn for fn in common if fn in lab and lab[fn] in ci]
    by = defaultdict(list)
    for fn in common:
        by[lab[fn]].append(fn)
    seen = sorted(by.keys())

    # hard-sim split: rarest 20% of train-seen classes become pseudo-unseen
    order = sorted(seen, key=lambda c: len(by[c]))
    pseudo = set(order[:int(len(seen) * 0.2)])
    kept = sorted(order[int(len(seen) * 0.2):])
    k2i = {c: i for i, c in enumerate(kept)}
    Sk = len(kept)
    kept_gidx = torch.tensor([ci[c] for c in kept], device=dev)
    kept_set = set(kept)
    other_gidx = torch.tensor([i for i, c in enumerate(classes) if c not in kept_set], device=dev)
    seen_mask_global = torch.zeros(NCLS, dtype=torch.bool, device=dev)
    seen_mask_global[kept_gidx] = True

    trby, val_seen, val_uns = defaultdict(list), [], []
    for c in seen:
        fns = sorted(by[c])
        if c in pseudo:
            val_uns += [(f, c) for f in fns]
        elif len(fns) >= 3:
            k = max(1, round(0.2 * len(fns)))
            trby[c] += fns[:-k]
            val_seen += [(f, c) for f in fns[-k:]]
        else:
            trby[c] += fns
    val = val_seen + val_uns
    val_files = [f for f, _ in val]
    gold = torch.tensor([ci[c] for _, c in val], device=dev)
    N = len(val_files)
    print(f'holdout: {N} queries ({len(val_seen)} seen + {len(val_uns)} pseudo-unseen) | '
          f'kept {Sk} | candidate-complement {len(other_gidx)} | classes {NCLS}', flush=True)

    def Qof(t):
        idx, feats, _ = train[t]
        return torch.stack([feats[idx[fn]] for fn in val_files]).to(dev)

    Q = {t: Qof(t) for t in train}

    # ---- stage-1 signal: vision prototypes over kept (train-seen) classes
    TkeptTax = TtH[kept_gidx]
    seen_block = torch.zeros(N, Sk, device=dev)
    ctft_TF = ctft_TL = None
    for t, hspace, w in MEMBERS:
        if w == 0.0:
            continue
        idx, feats, _ = train[t]
        P = torch.zeros(Sk, feats.shape[1])
        cnt = torch.zeros(Sk)
        TF, TL = [], []
        for c in kept:
            for fn in trby[c]:
                f = feats[idx[fn]]
                P[k2i[c]] += f
                cnt[k2i[c]] += 1
                TF.append(f)
                TL.append(k2i[c])
        P = F.normalize(P / cnt.clamp(min=1).unsqueeze(1), dim=-1).to(dev)
        TF, TL = torch.stack(TF).to(dev), torch.tensor(TL, device=dev)
        out = torch.empty(N, Sk, device=dev)
        for i in range(0, N, 2000):
            e = Q[t][i:i + 2000]
            cmax = torch.full((e.shape[0], Sk), -1e9, device=dev)
            cmax.scatter_reduce_(1, TL.unsqueeze(0).expand(e.shape[0], -1), e @ TF.t(), reduce='amax')
            sc = (e @ P.t()) + 2.0 * cmax
            if hspace:
                sc = sc + LAM * (e @ TkeptTax.t())
            out[i:i + 2000] = sc
        seen_block += w * zc(out)
        if t == 'ctftshift':
            ctft_TF, ctft_TL = TF, TL
        else:
            del TF, TL
        del out, P

    # ---- stage-2 signals, full class space
    text_full = (dbnorm(Q['ctftshift'] @ TtH.t()) + 0.5 * dbnorm(Q['L'] @ TnL.t())
                 + 0.75 * dbnorm(Q['fullft336_v2'] @ TtH.t()) + 1.0 * dbnorm(Q['ftshift'] @ TtH.t())
                 + 1.0 * dbnorm(Q['ctftshift'] @ TTX.t())
                 + TAXABIND_W * dbnorm(Q['taxabind'] @ Ttb.t()))
    all_gidx = torch.arange(NCLS, device=dev)
    b2f_raw, hf = proto_leg_raw(Q['b2'], B2_PROTO_FROZEN, all_gidx)
    b2l_raw, hl = proto_leg_raw(Q['b2l'], B2_PROTO_LORA, all_gidx)
    bank_raw, hb = bank_leg_raw(Q['ctftshift'], all_gidx)
    kept_cpu, hb_kept = kept_gidx.cpu(), hb[kept_gidx.cpu()]
    print(f'leg coverage  b2-frozen: kept {int(hf[kept_cpu].sum())}/{Sk} vs complement '
          f'{int(hf.sum()) - int(hf[kept_cpu].sum())}/{len(other_gidx)} | '
          f'b2-lora: kept {int(hl[kept_cpu].sum())}/{Sk} | '
          f'iNat photo-bank: kept {int(hb_kept.sum())}/{Sk} complement '
          f'{int(hb.sum()) - int(hb_kept.sum())}/{len(other_gidx)}', flush=True)

    # arm A uses the production legs verbatim: dbnorm over the complement columns only
    unseen_route = (text_full[:, other_gidx] + IMG_W * dbnorm(bank_raw[:, other_gidx])
                    + B2_WF * dbnorm(b2f_raw[:, other_gidx]) + B2_WL * dbnorm(b2l_raw[:, other_gidx]))
    # arms C/D: same legs dbnorm'd over the FULL class space, uncovered classes neutral
    b2f_leg, b2l_leg = neutral(dbnorm(b2f_raw), hf), neutral(dbnorm(b2l_raw), hl)
    score_C = text_full + B2_WF * b2f_leg + B2_WL * b2l_leg
    score_Cb = score_C + IMG_W * neutral(dbnorm(bank_raw), hb)
    del b2f_raw, b2l_raw

    # arm D: one uniform image->reference-photo leg. Same encoder (ctftshift), same
    # statistic (top-TOPM mean cosine) on both sides; a class's reference photos are its
    # training images when it has them, its iNat photos when it does not.
    ref_raw = bank_raw
    ref_raw[:, kept_gidx] = train_photo_leg_raw(Q['ctftshift'], ctft_TF, ctft_TL, Sk)
    has_ref = hb.clone()
    has_ref[kept_cpu] = True
    print(f'unified reference-photo leg covers {int(has_ref.sum())}/{NCLS} classes '
          f'({Sk} from train images, {int(hb.sum()) - int(hb_kept.sum())} from iNat)', flush=True)
    score_D = score_C + IMG_W * neutral(dbnorm(ref_raw), has_ref)
    del bank_raw, ref_raw

    rows = []

    def report(name, pred):
        ok = (pred == gold)
        s = ok[:len(val_seen)].float().mean().item()
        u = ok[len(val_seen):].float().mean().item()
        o = W_SEEN * s + W_UNS * u
        r_s, r_u = route_stats(pred, seen_mask_global, len(val_seen))
        rows.append({'arm': name, 'seen': 100 * s, 'unseen': 100 * u, 'overall': 100 * o,
                     'seen_rows_to_seen_cls': r_s, 'unseen_rows_to_unseen_cls': r_u})
        print(f'{name:<34} seen {100*s:6.2f}  unseen {100*u:6.2f}  overall {100*o:6.2f}   '
              f'[route  seen->seen {r_s:5.1f}%  unseen->unseen {r_u:5.1f}%]', flush=True)

    # ---- arm A: production hard gate
    text_margin = ((Q['ctftshift'] @ TtH[kept_gidx].t()).max(1).values
                   - (Q['ctftshift'] @ TtH[other_gidx].t()).max(1).values)
    gate = z1(seen_block.max(1).values) + 2.0 * z1(text_margin)
    print('\n--- A: production hard gate (two heads, v50 shape) ---', flush=True)
    for f in [0.60, 0.72]:
        thr = torch.topk(gate, int(round(f * N))).values.min()
        route_seen = gate >= thr
        pred = torch.empty(N, dtype=torch.long, device=dev)
        pred[route_seen] = kept_gidx[seen_block[route_seen].argmax(1)]
        iu = (~route_seen).nonzero(as_tuple=True)[0]
        pred[iu] = other_gidx[sinkhorn(unseen_route[iu]).argmax(1)]
        report(f'A hard gate f={f:.2f} +sinkhorn', pred)

    # ---- arm B: requested spec (stage 2 = vision<->text only)
    print('\n--- B: unified 2-stage, stage-2 = vision<->text only (as specified) ---', flush=True)
    full_text_argmax = text_full.argmax(1)
    for k in KS + [Sk]:
        pred = two_stage(seen_block, text_full, kept_gidx, other_gidx, k, NCLS)
        if k == Sk:  # no pruning => must collapse to a plain full-space argmax
            assert (pred == full_text_argmax).all(), 'shortlist scatter is wrong'
        report(f'B k={k}' + (' (no stage-1 pruning)' if k == Sk else ''), pred)

    # ---- arm C: unified 2-stage, image-proto legs on the full class space
    print('\n--- C: unified 2-stage, stage-2 = vision<->text + B2/ToL protos (full space) ---', flush=True)
    for k in KS + [Sk]:
        report(f'C k={k}' + (' (no stage-1 pruning)' if k == Sk else ''),
               two_stage(seen_block, score_C, kept_gidx, other_gidx, k, NCLS))
    print('\n--- C+bank: C plus the iNat photo bank (covers 0 holdout-seen classes) ---', flush=True)
    for k in KS:
        report(f'C+bank k={k}', two_stage(seen_block, score_Cb, kept_gidx, other_gidx, k, NCLS))
    print('\n--- D: C + one uniform reference-photo leg (train images | iNat photos) ---', flush=True)
    for k in KS + [Sk]:
        report(f'D k={k}' + (' (no stage-1 pruning)' if k == Sk else ''),
               two_stage(seen_block, score_D, kept_gidx, other_gidx, k, NCLS))

    best = max(rows, key=lambda r: r['overall'])
    print(f"\nbest holdout arm: {best['arm']} overall {best['overall']:.2f}", flush=True)
    out = os.path.join(OUT, 'unified_2stage_holdout.json')
    json.dump({'rows': rows, 'n_seen': len(val_seen), 'n_unseen': len(val_uns)}, open(out, 'w'), indent=1)
    print(f'wrote {out}', flush=True)


# ------------------------------------------------------------------ real eval batch

def run_real():
    """Same arms on the 35,665 eval images. No labels here -- the readable signal is the
    routing diagnostic (HANDOFF sec 6: tf/uf are enumeration only, never used to route)."""
    classes = list(pickle.load(open(os.path.join(D, 'all_classes.pkl'), 'rb')))
    ci = {c: i for i, c in enumerate(classes)}
    NCLS = len(classes)
    lab = json.load(open(os.path.join(D, 'label_train.json')))
    TtH = F.normalize(torch.load(os.path.join(OUT, 'text_emb_h_taxon.pt'), weights_only=False)['emb_taxon'].float(), dim=-1).to(dev)
    TnL = F.normalize(torch.load(os.path.join(OUT, 'text_emb.pt'), weights_only=False)['emb_name'].float(), dim=-1).to(dev)
    TTX = F.normalize(torch.load(os.path.join(OUT, 'text_emb_h_promptens.pt'), weights_only=False)['emb_taxctx'].float(), dim=-1).to(dev)
    Ttb = F.normalize(torch.load(os.path.join(OUT, 'text_emb_taxabind_taxctx.pt'), weights_only=False)['emb_taxctx'].float(), dim=-1).to(dev)

    train = {t: load(emb_path('train', EMB[t])) for t, _, _ in MEMBERS}
    common = None
    for _, _, files in train.values():
        common = set(files) if common is None else (common & set(files))
    common = [fn for fn in common if fn in lab and lab[fn] in ci]
    by = defaultdict(list)
    for fn in common:
        by[lab[fn]].append(fn)
    seen = sorted(by.keys())
    s2i = {c: i for i, c in enumerate(seen)}
    S = len(seen)
    seen_gidx = torch.tensor([ci[c] for c in seen], device=dev)
    seen_set = set(seen)
    other_gidx = torch.tensor([i for i, c in enumerate(classes) if c not in seen_set], device=dev)
    seen_mask_global = torch.zeros(NCLS, dtype=torch.bool, device=dev)
    seen_mask_global[seen_gidx] = True

    tags = {t: EMB[t] for t, _, _ in MEMBERS}
    tags.update({'taxabind': 'taxabind', 'b2': 'bioclip2', 'b2l': 'bioclip2_lora_v2'})
    test = {t: load(emb_path('test', g)) for t, g in tags.items()}
    uns = {t: load(emb_path('unseen', g)) for t, g in tags.items()}
    tf = sorted(set.intersection(*[set(f) for (_, _, f) in test.values()]))
    uf = sorted(set.intersection(*[set(f) for (_, _, f) in uns.values()]))
    N = len(tf) + len(uf)
    print(f'eval batch: {N} images ({len(tf)} test-folder + {len(uf)} unseen-folder) | seen {S}', flush=True)
    Q = {t: torch.cat([torch.stack([test[t][1][test[t][0][fn]] for fn in tf]),
                       torch.stack([uns[t][1][uns[t][0][fn]] for fn in uf])]).to(dev) for t in tags}

    TseenTax = TtH[seen_gidx]
    seen_block = torch.zeros(N, S, device=dev)
    ctft_TF = ctft_TL = None
    for t, hspace, w in MEMBERS:
        if w == 0.0:
            continue
        idx, feats, _ = train[t]
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
        P = F.normalize(P / cnt.clamp(min=1).unsqueeze(1), dim=-1).to(dev)
        TF, TL = torch.stack(TF).to(dev), torch.tensor(TL, device=dev)
        out = torch.empty(N, S, device=dev)
        for i in range(0, N, 2000):
            e = Q[t][i:i + 2000]
            cmax = torch.full((e.shape[0], S), -1e9, device=dev)
            cmax.scatter_reduce_(1, TL.unsqueeze(0).expand(e.shape[0], -1), e @ TF.t(), reduce='amax')
            sc = (e @ P.t()) + 2.0 * cmax
            if hspace:
                sc = sc + LAM * (e @ TseenTax.t())
            out[i:i + 2000] = sc
        seen_block += w * zc(out)
        if t == 'ctftshift':
            ctft_TF, ctft_TL = TF, TL
        else:
            del TF, TL
        del out, P

    text_full = (dbnorm(Q['ctftshift'] @ TtH.t()) + 0.5 * dbnorm(Q['L'] @ TnL.t())
                 + 0.75 * dbnorm(Q['fullft336_v2'] @ TtH.t()) + 1.0 * dbnorm(Q['ftshift'] @ TtH.t())
                 + 1.0 * dbnorm(Q['ctftshift'] @ TTX.t())
                 + TAXABIND_W * dbnorm(Q['taxabind'] @ Ttb.t()))
    all_gidx = torch.arange(NCLS, device=dev)
    b2f_raw, hf = proto_leg_raw(Q['b2'], B2_PROTO_FROZEN, all_gidx)
    b2l_raw, hl = proto_leg_raw(Q['b2l'], B2_PROTO_LORA, all_gidx)
    bank_raw, hb = bank_leg_raw(Q['ctftshift'], all_gidx)
    seen_cpu, hb_seen = seen_gidx.cpu(), hb[seen_gidx.cpu()]
    print(f'leg coverage  b2-frozen: seen {int(hf[seen_cpu].sum())}/{S} complement '
          f'{int(hf.sum()) - int(hf[seen_cpu].sum())}/{len(other_gidx)} | '
          f'iNat photo-bank: seen {int(hb_seen.sum())}/{S} complement '
          f'{int(hb.sum()) - int(hb_seen.sum())}/{len(other_gidx)}', flush=True)

    unseen_route = (text_full[:, other_gidx] + IMG_W * dbnorm(bank_raw[:, other_gidx])
                    + B2_WF * dbnorm(b2f_raw[:, other_gidx]) + B2_WL * dbnorm(b2l_raw[:, other_gidx]))
    b2f_leg, b2l_leg = neutral(dbnorm(b2f_raw), hf), neutral(dbnorm(b2l_raw), hl)
    score_C = text_full + B2_WF * b2f_leg + B2_WL * b2l_leg
    score_Cb = score_C + IMG_W * neutral(dbnorm(bank_raw), hb)
    del b2f_raw, b2l_raw

    ref_raw = bank_raw
    ref_raw[:, seen_gidx] = train_photo_leg_raw(Q['ctftshift'], ctft_TF, ctft_TL, S)
    has_ref = hb.clone()
    has_ref[seen_cpu] = True
    print(f'unified reference-photo leg covers {int(has_ref.sum())}/{NCLS} classes '
          f'({S} from train images, {int(hb.sum()) - int(hb_seen.sum())} from iNat)', flush=True)
    score_D = score_C + IMG_W * neutral(dbnorm(ref_raw), has_ref)
    del bank_raw, ref_raw

    rows = []

    def report(name, pred):
        r_s, r_u = route_stats(pred, seen_mask_global, len(tf))
        cov = len(set(pred[len(tf):].tolist()) - set(seen_gidx.tolist()))
        rows.append({'arm': name, 'tf_to_seen': r_s, 'uf_to_unseen': r_u, 'unseen_cls_used': cov})
        print(f'{name:<34} tf->seen-cls {r_s:5.1f}%   uf->unseen-cls {r_u:5.1f}%   '
              f'unseen classes used {cov}', flush=True)

    print('\n--- A: production hard gate (reference: v50 f=0.60 is the 51.44% submission) ---', flush=True)
    text_margin = ((Q['ctftshift'] @ TtH[seen_gidx].t()).max(1).values
                   - (Q['ctftshift'] @ TtH[other_gidx].t()).max(1).values)
    gate = z1(seen_block.max(1).values) + 2.0 * z1(text_margin)
    for f in [0.60, 0.72]:
        thr = torch.topk(gate, int(round(f * N))).values.min()
        route_seen = gate >= thr
        pred = torch.empty(N, dtype=torch.long, device=dev)
        pred[route_seen] = seen_gidx[seen_block[route_seen].argmax(1)]
        iu = (~route_seen).nonzero(as_tuple=True)[0]
        pred[iu] = other_gidx[sinkhorn(unseen_route[iu]).argmax(1)]
        report(f'A hard gate f={f:.2f} +sinkhorn', pred)
        # the projection is calibrated on this arm -- prove it is the scored submission
        ref = os.path.join(OUT, f'prediction_v50_f{int(f*100)}_tau18.json')
        if os.path.exists(ref):
            r = json.load(open(ref))
            mine = {fn: classes[j] for fn, j in zip(tf + uf, pred.tolist())}
            agree = sum(1 for k in r if mine.get(k) == r[k])
            print(f'    vs {os.path.basename(ref)}: {agree}/{len(r)} identical '
                  f'({100*agree/len(r):.2f}%)', flush=True)

    print('\n--- B / C: unified 2-stage ---', flush=True)
    for k in KS:
        report(f'B k={k}', two_stage(seen_block, text_full, seen_gidx, other_gidx, k, NCLS))
    for k in KS:
        report(f'C k={k}', two_stage(seen_block, score_C, seen_gidx, other_gidx, k, NCLS))
    for k in KS:
        report(f'C+bank k={k}', two_stage(seen_block, score_Cb, seen_gidx, other_gidx, k, NCLS))
    for k in KS:
        report(f'D k={k}', two_stage(seen_block, score_D, seen_gidx, other_gidx, k, NCLS))

    # Calibrate conditional accuracies off the one arm with a measured Codabench score,
    # then project every other arm from its routing. UPPER BOUND: it credits each arm with
    # the production head's conditional accuracy, which the unified arms do not have (they
    # decide seen classes without the 3-encoder prototype ensemble).
    anchor = next(r for r in rows if r['arm'] == REAL_ANCHOR['arm'])
    a_cond = REAL_ANCHOR['seen'] / anchor['tf_to_seen']
    b_cond = REAL_ANCHOR['unseen'] / anchor['uf_to_unseen']
    print(f'\nconditional accuracy calibrated on the {REAL_ANCHOR["overall"]:.2f}% submission: '
          f'seen-kept {100*a_cond:.1f}%  unseen-caught {100*b_cond:.1f}%', flush=True)
    print('projected real overall (UPPER BOUND -- assumes production conditional accuracy):', flush=True)
    for r in rows:
        r['proj_overall'] = W_SEEN * r['tf_to_seen'] * a_cond + W_UNS * r['uf_to_unseen'] * b_cond
        print(f"  {r['arm']:<32} <= {r['proj_overall']:6.2f}%  "
              f"({r['proj_overall'] - REAL_ANCHOR['overall']:+.2f} vs measured best)", flush=True)

    out = os.path.join(OUT, 'unified_2stage_real_routing.json')
    json.dump({'rows': rows, 'n_tf': len(tf), 'n_uf': len(uf), 'anchor': REAL_ANCHOR,
               'a_cond': a_cond, 'b_cond': b_cond}, open(out, 'w'), indent=1)
    print(f'\nwrote {out}', flush=True)
    print('Reference routing (HANDOFF sec 8): hard gate tf->seen ~92.5% / uf->unseen ~54.5%; '
          'soft_b1.05 was 84.2% / 37.0% and was killed on exactly this diagnostic.', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--real', action='store_true', help='run the 35,665-image routing diagnostic')
    a = ap.parse_args()
    (run_real if a.real else run_holdout)()
