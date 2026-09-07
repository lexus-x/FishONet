"""Soft full-space scoring vs hard unseen-pool routing.

Hypothesis (absolute-limit push):
  Hard gate → unseen-ONLY class pool zeros wrongly ejected seen images.
  Soft argmax over all 17,393 classes can keep seen≈78% while letting
  uf images reach ~b≈15–16% (leader pattern: seen~78, unseen~15).

Holdout: hard-sim rarest-20% as pseudo-unseen (research/common.py).
Eval: diagnostic folder agreement only (no labels) — never bake tf/uf into preds.

Does NOT write a Codabench zip unless --build is passed and holdout clears bar.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from collections import defaultdict

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from common import DEV, OUT, DATA, dbnorm, load_emb, zc  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(DATA, 'dl')


def z1(v):
    return (v - v.mean()) / (v.std() + 1e-6)


def sinkhorn(logits, n_iter=50, tau=2.0):
    P = torch.softmax(logits / tau, dim=1)
    n, C = P.shape
    target_col = n / C
    for _ in range(n_iter):
        P = P / P.sum(dim=1, keepdim=True).clamp(min=1e-9)
        P = P * (target_col / P.sum(dim=0, keepdim=True).clamp(min=1e-9))
    return P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--build', action='store_true', help='write submission zips')
    ap.add_argument('--kill-bar', type=float, default=0.5,
                    help='holdout overall pts above hard baseline to clear')
    args = ap.parse_args()
    dev = DEV
    torch.set_num_threads(8)

    classes = list(pickle.load(open(os.path.join(D, 'all_classes.pkl'), 'rb')))
    ci = {c: i for i, c in enumerate(classes)}
    NCLS = len(classes)
    lab = json.load(open(os.path.join(D, 'label_train.json')))

    txtHt = torch.load(os.path.join(OUT, 'text_emb_h_taxon.pt'), weights_only=False)
    txtL = torch.load(os.path.join(OUT, 'text_emb.pt'), weights_only=False)
    _pe = torch.load(os.path.join(OUT, 'text_emb_h_promptens.pt'), weights_only=False)
    TtH = F.normalize(txtHt['emb_taxon'].float(), dim=-1).to(dev)
    TnL = F.normalize(txtL['emb_name'].float(), dim=-1).to(dev)
    TTX = F.normalize(_pe['emb_taxctx'].float(), dim=-1).to(dev)

    MEMBERS = [('ctftbig', True, 1.0), ('ftshift', True, 2.5), ('fullft336shift', True, 2.5),
               ('L', False, 0.0), ('fullft336_v2', True, 0.0)]
    TRAIN = {'ctftbig': 'emb_train_ctftbig', 'ftshift': 'emb_train_ftshift',
             'fullft336shift': 'emb_train_fullft336shift', 'L': 'emb_train',
             'fullft336_v2': 'emb_train_fullft336_v2'}
    train = {t: load_emb(os.path.join(OUT, f'{TRAIN[t]}.pt')) for t, _, _ in MEMBERS}
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
    kept_idx = torch.tensor([ci[c] for c in seen], device=dev)
    other_idx = torch.tensor([i for i in range(NCLS) if classes[i] not in set(seen)], device=dev)
    TseenTax = TtH[kept_idx]
    LAM = 4.0

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
        return P.to(dev), torch.stack(TF).to(dev), torch.tensor(TL, device=dev)

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

    # ---- eval batch (same as v33) ----
    test = {t: load_emb(os.path.join(OUT, f'{TRAIN[t].replace("emb_train", "emb_test")}.pt'))
            for t, _, _ in MEMBERS}
    unseen = {t: load_emb(os.path.join(OUT, f'{TRAIN[t].replace("emb_train", "emb_unseen")}.pt'))
              for t, _, _ in MEMBERS}
    tf = sorted(set.intersection(*[set(f) for (_, _, f) in test.values()]))
    uf = sorted(set.intersection(*[set(f) for (_, _, f) in unseen.values()]))
    all_files = tf + uf
    print(f'eval batch {len(all_files)} tf={len(tf)} uf={len(uf)}', flush=True)

    def qcat(t):
        ti, tfeat, _ = test[t]
        ui, ufeat, _ = unseen[t]
        return torch.cat([torch.stack([tfeat[ti[fn]] for fn in tf]),
                          torch.stack([ufeat[ui[fn]] for fn in uf])])

    Q = {t: qcat(t).to(dev) for t, _, _ in MEMBERS}

    seen_block = torch.zeros(len(all_files), S, device=dev)
    for t, hs, w in MEMBERS:
        if w == 0.0:
            continue
        seen_block = seen_block + w * zc(seen_score(Q[t], *PR[t], hs))
    text_full = (dbnorm(Q['ctftbig'] @ TtH.t()) + 0.5 * dbnorm(Q['L'] @ TnL.t())
                 + 0.75 * dbnorm(Q['fullft336_v2'] @ TtH.t()) + 1.0 * dbnorm(Q['ftshift'] @ TtH.t())
                 + 1.0 * dbnorm(Q['ctftbig'] @ TTX.t()))
    text_unseen_only = text_full[:, other_idx]
    text_seen_only = text_full[:, kept_idx]

    img_seenmax = seen_block.max(1).values
    text_margin = ((Q['ctftbig'] @ TtH[kept_idx].t()).max(1).values
                   - (Q['ctftbig'] @ TtH[other_idx].t()).max(1).values)
    combined = z1(img_seenmax) + 2.0 * z1(text_margin)

    y_folder = torch.tensor([0] * len(tf) + [1] * len(uf))  # diagnostic only

    def hard_route(seen_frac, use_sink=True):
        k_seen = int(round(seen_frac * len(all_files)))
        thr = torch.topk(combined, k_seen).values.min()
        route_seen = combined >= thr
        pred = torch.empty(len(all_files), dtype=torch.long, device=dev)
        pred[route_seen] = kept_idx[seen_block[route_seen].argmax(1)]
        idx_uns = (~route_seen).nonzero(as_tuple=True)[0]
        sub = text_unseen_only[idx_uns]
        if use_sink and len(idx_uns):
            pred[idx_uns] = other_idx[sinkhorn(sub).argmax(1)]
        else:
            pred[idx_uns] = other_idx[sub.argmax(1)]
        return pred, route_seen

    def soft_fullspace(alpha=1.0, beta=1.0, gamma=0.0, use_sink_unseen=False):
        """Full [N, NCLS] score: seen cols from seen_block (+ optional text), unseen from text.

        alpha: weight on zc(seen_block) for seen cols
        beta:  weight on z1-scaled text for unseen cols
        gamma: weight on text for seen cols (blend)
        """
        full = torch.full((len(all_files), NCLS), -1e9, device=dev)
        sb = zc(seen_block)
        # calibrate text blocks to comparable scale via per-row z-score of candidates
        tu = text_unseen_only
        ts = text_seen_only
        # row-wise standardize the two pools then place
        def row_z(M):
            return (M - M.mean(1, keepdim=True)) / (M.std(1, keepdim=True) + 1e-6)

        full[:, kept_idx] = alpha * sb + gamma * row_z(ts)
        full[:, other_idx] = beta * row_z(tu)
        if use_sink_unseen:
            # soft: still argmax full, but optionally rebalance only among images
            # whose soft pick is an unseen class
            pred = full.argmax(1)
            return pred
        return full.argmax(1)

    def soft_margin_gate(temp=1.0):
        """Continuous mix: score = seen_max + margin-weighted unseen bump — still one argmax."""
        # Build full: start from -inf, fill seen from seen_block, unseen from text
        full = torch.full((len(all_files), NCLS), -1e9, device=dev)
        sb = zc(seen_block)
        tu = (text_unseen_only - text_unseen_only.mean(1, keepdim=True)) / (
            text_unseen_only.std(1, keepdim=True) + 1e-6)
        # novelty prior: lower combined => boost unseen cols
        nov = torch.sigmoid(-combined / temp)  # high when novel
        full[:, kept_idx] = sb
        full[:, other_idx] = tu + 3.0 * nov.unsqueeze(1)  # novelty bonus on unseen cols
        return full.argmax(1)

    def folder_stats(pred, name):
        pred_cpu = pred.detach().cpu()
        seen_set_t = torch.zeros(NCLS, dtype=torch.bool)
        seen_set_t[kept_idx.cpu()] = True
        pred_seen_cls = seen_set_t[pred_cpu]
        # among folders
        tf_ok_type = pred_seen_cls[:len(tf)].float().mean().item()
        uf_uns_type = (~pred_seen_cls[len(tf):]).float().mean().item()
        print(
            f'  {name:28s} pred_seen_cls={100*pred_seen_cls.float().mean():.1f}% | '
            f'tf→seen-cls {100*tf_ok_type:.1f}% | uf→uns-cls {100*uf_uns_type:.1f}%',
            flush=True,
        )
        return {
            'pred_seen_cls_frac': float(pred_seen_cls.float().mean()),
            'tf_to_seen_cls': tf_ok_type,
            'uf_to_uns_cls': uf_uns_type,
        }

    print('\n=== EVAL diagnostic routing (folder type agreement; NOT accuracy) ===', flush=True)
    results = {'variants': {}}
    for f in [0.55, 0.62, 0.72]:
        pred, _ = hard_route(f, True)
        results['variants'][f'hard_sink_f{int(f*100)}'] = folder_stats(pred, f'hard_sink_f{f}')

    # Fine beta sweep — a1/b1 vs a1/b1.5 was a cliff on first run
    for beta in [0.8, 0.9, 1.0, 1.05, 1.1, 1.15, 1.2, 1.3]:
        pred = soft_fullspace(1.0, beta, 0.0)
        key = f'soft_a1_b{beta}_g0'
        results['variants'][key] = folder_stats(pred, key)

    # Pure text_full soft (no image prototypes) — diagnostic from gate_absolute: ~69% uf
    pred_text = text_full.argmax(1)
    results['variants']['soft_text_full'] = folder_stats(pred_text, 'soft_text_full')

    # Calibrated: match per-image max(seen) and max(unseen) scales, then argmax
    def soft_calibrated(bias=0.0):
        sb = zc(seen_block)
        tu = (text_unseen_only - text_unseen_only.mean(1, keepdim=True)) / (
            text_unseen_only.std(1, keepdim=True) + 1e-6)
        ms = sb.max(1).values
        mu = tu.max(1).values
        # shift unseen so batch-mean max matches seen, plus bias (positive => more unseen)
        shift = (ms.mean() - mu.mean()) + bias
        full = torch.full((len(all_files), NCLS), -1e9, device=dev)
        full[:, kept_idx] = sb
        full[:, other_idx] = tu + shift
        return full.argmax(1)

    for bias in [-0.5, -0.25, 0.0, 0.25, 0.5, 1.0]:
        pred = soft_calibrated(bias)
        key = f'soft_cal_bias{bias}'
        results['variants'][key] = folder_stats(pred, key)

    for temp in [0.5, 1.0, 2.0]:
        pred = soft_margin_gate(temp)
        key = f'soft_nov_t{temp}'
        results['variants'][key] = folder_stats(pred, key)

    # ---- Holdout accuracy (pseudo-unseen = rarest 20% classes) ----
    print('\n=== HOLDOUT accuracy (pseudo-unseen = rarest 20% train classes) ===', flush=True)
    order = sorted(seen, key=lambda c: len(by[c]))
    n_pseudo = int(len(seen) * 0.2)
    pseudo = set(order[:n_pseudo])
    kept_h = sorted(order[n_pseudo:])
    kept_h_set = set(kept_h)
    k2i = {c: i for i, c in enumerate(kept_h)}
    Sk = len(kept_h)
    kept_h_idx = torch.tensor([ci[c] for c in kept_h], device=dev)
    other_h_idx = torch.tensor([i for i in range(NCLS) if classes[i] not in kept_h_set], device=dev)

    # Zero-shot holdout: rarest 20% classes have NO train images (all go to val_uns).
    # Kept classes: last 20% of files -> val_seen.
    val_seen, val_uns = [], []  # (fn, class_name)
    trby = defaultdict(list)
    for c in seen:
        fns = sorted(by[c])
        if c in pseudo:
            for f in fns:
                val_uns.append((f, c))
            continue
        if len(fns) >= 3:
            k = max(1, round(0.2 * len(fns)))
            for f in fns[:-k]:
                trby[c].append(f)
            for f in fns[-k:]:
                val_seen.append((f, c))
        else:
            for f in fns:
                trby[c].append(f)
    print(f'holdout val: seen-cls imgs={len(val_seen)} pseudo-uns imgs={len(val_uns)} '
          f'kept_cls={Sk} cand_cls={len(other_h_idx)}', flush=True)

    def holdout_protos(t):
        idx, feats, _ = train[t]
        P = torch.zeros(Sk, feats.shape[1])
        cnt = torch.zeros(Sk)
        TF, TL = [], []
        for c in kept_h:
            for fn in trby[c]:
                f = feats[idx[fn]]
                P[k2i[c]] += f
                cnt[k2i[c]] += 1
                TF.append(f)
                TL.append(k2i[c])
        P = F.normalize(P / cnt.clamp(min=1).unsqueeze(1), dim=-1)
        return P.to(dev), torch.stack(TF).to(dev), torch.tensor(TL, device=dev)

    PR_h = {t: holdout_protos(t) for t, _, _ in MEMBERS}
    TkeptTax = TtH[kept_h_idx]

    def seen_score_h(qF, P, TF, TL, hspace):
        qF = qF.to(dev)
        n = qF.shape[0]
        out = torch.empty(n, Sk, device=dev)
        for i in range(0, n, 2000):
            e = qF[i:i + 2000]
            ps = e @ P.t()
            sim = e @ TF.t()
            cmax = torch.full((e.shape[0], Sk), -1e9, device=dev)
            cmax.scatter_reduce_(1, TL.unsqueeze(0).expand(e.shape[0], -1), sim, reduce='amax')
            sc = ps + 2.0 * cmax
            if hspace:
                sc = sc + LAM * (e @ TkeptTax.t())
            out[i:i + 2000] = sc
        return out

    val_all = val_seen + val_uns
    val_files = [f for f, _ in val_all]
    gold = [c for _, c in val_all]
    is_uns = torch.tensor([0] * len(val_seen) + [1] * len(val_uns))

    Qh = {}
    for t, _, _ in MEMBERS:
        idx, feats, _ = train[t]
        Qh[t] = torch.stack([feats[idx[fn]] for fn in val_files]).to(dev)

    seen_block_h = torch.zeros(len(val_files), Sk, device=dev)
    for t, hs, w in MEMBERS:
        if w == 0.0:
            continue
        seen_block_h = seen_block_h + w * zc(seen_score_h(Qh[t], *PR_h[t], hs))
    text_full_h = (dbnorm(Qh['ctftbig'] @ TtH.t()) + 0.5 * dbnorm(Qh['L'] @ TnL.t())
                   + 0.75 * dbnorm(Qh['fullft336_v2'] @ TtH.t()) + 1.0 * dbnorm(Qh['ftshift'] @ TtH.t())
                   + 1.0 * dbnorm(Qh['ctftbig'] @ TTX.t()))
    text_uns_h = text_full_h[:, other_h_idx]
    text_seen_h = text_full_h[:, kept_h_idx]

    img_max_h = seen_block_h.max(1).values
    text_margin_h = ((Qh['ctftbig'] @ TtH[kept_h_idx].t()).max(1).values
                     - (Qh['ctftbig'] @ TtH[other_h_idx].t()).max(1).values)
    combined_h = z1(img_max_h) + 2.0 * z1(text_margin_h)

    def acc(pred_names, mask=None):
        ok = [p == g for p, g in zip(pred_names, gold)]
        if mask is None:
            return sum(ok) / len(ok)
        m = mask.tolist() if torch.is_tensor(mask) else mask
        sel = [o for o, mm in zip(ok, m) if mm]
        return sum(sel) / max(1, len(sel))

    def overall(s_acc, u_acc):
        return 0.5635 * s_acc + 0.4365 * u_acc

    holdout_rows = []

    def report(name, pred_idx):
        names = [classes[i] for i in pred_idx.tolist()]
        s_acc = acc(names, is_uns == 0)
        u_acc = acc(names, is_uns == 1)
        o = overall(s_acc, u_acc)
        print(f'  {name:28s} seen={100*s_acc:.2f}% uns={100*u_acc:.2f}% overall={100*o:.2f}%',
              flush=True)
        holdout_rows.append({'name': name, 'seen': s_acc, 'unseen': u_acc, 'overall': o})
        return o

    # hard baselines on holdout
    for f in [0.55, 0.72]:
        k_seen = int(round(f * len(val_files)))
        thr = torch.topk(combined_h, k_seen).values.min()
        route_seen = combined_h >= thr
        pred = torch.empty(len(val_files), dtype=torch.long, device=dev)
        pred[route_seen] = kept_h_idx[seen_block_h[route_seen].argmax(1)]
        idx_uns = (~route_seen).nonzero(as_tuple=True)[0]
        if len(idx_uns):
            pred[idx_uns] = other_h_idx[sinkhorn(text_uns_h[idx_uns]).argmax(1)]
        report(f'hard_sink_f{int(f*100)}', pred)

    hard_base = holdout_rows[-1]['overall']  # f72

    def row_z(M):
        return (M - M.mean(1, keepdim=True)) / (M.std(1, keepdim=True) + 1e-6)

    for beta in [0.8, 0.9, 1.0, 1.05, 1.1, 1.15, 1.2, 1.3]:
        full = torch.full((len(val_files), NCLS), -1e9, device=dev)
        full[:, kept_h_idx] = zc(seen_block_h)
        full[:, other_h_idx] = beta * row_z(text_uns_h)
        report(f'soft_a1_b{beta}_g0', full.argmax(1))

    # pure text
    report('soft_text_full', text_full_h.argmax(1))

    for bias in [-0.5, -0.25, 0.0, 0.25, 0.5, 1.0]:
        sb = zc(seen_block_h)
        tu = row_z(text_uns_h)
        shift = (sb.max(1).values.mean() - tu.max(1).values.mean()) + bias
        full = torch.full((len(val_files), NCLS), -1e9, device=dev)
        full[:, kept_h_idx] = sb
        full[:, other_h_idx] = tu + shift
        report(f'soft_cal_bias{bias}', full.argmax(1))

    for temp in [0.5, 1.0, 2.0, 4.0]:
        full = torch.full((len(val_files), NCLS), -1e9, device=dev)
        tu = row_z(text_uns_h)
        nov = torch.sigmoid(-combined_h / temp)
        full[:, kept_h_idx] = zc(seen_block_h)
        full[:, other_h_idx] = tu + 3.0 * nov.unsqueeze(1)
        report(f'soft_nov_t{temp}', full.argmax(1))

    # oracle pool (perfect gate upper bound on this holdout)
    pred = torch.empty(len(val_files), dtype=torch.long, device=dev)
    pred[is_uns.to(dev) == 0] = kept_h_idx[seen_block_h[is_uns.to(dev) == 0].argmax(1)]
    iu = (is_uns.to(dev) == 1).nonzero(as_tuple=True)[0]
    pred[iu] = other_h_idx[sinkhorn(text_uns_h[iu]).argmax(1)]
    report('oracle_pool_sink', pred)

    best = max(holdout_rows, key=lambda r: r['overall'])
    delta = 100 * (best['overall'] - hard_base)
    print(f'\nbest={best["name"]} overall={100*best["overall"]:.2f}% '
          f'(Δ vs hard_f72 = {delta:+.2f}pt)', flush=True)
    results['holdout'] = holdout_rows
    results['best'] = best
    results['hard_f72'] = hard_base
    results['delta_pt'] = delta
    results['note'] = (
        'Holdout overstates; real gate AUC only 0.884. Soft clears submit bar only if '
        f'Δ≥{args.kill_bar}pt on holdout AND routing diagnostic looks leader-like '
        '(uf→uns-cls≳70%, tf→seen-cls≳80%).'
    )

    out_p = os.path.join(OUT, 'soft_fullspace_results.json')
    json.dump(results, open(out_p, 'w'), indent=1)
    print('wrote', out_p, flush=True)

    if args.build and delta >= args.kill_bar:
        print('\n--build: clearing bar; writing soft eval submissions', flush=True)
        # pick best soft hyperparams from name
        # default to soft_a1_b1_g0 if best is soft
        tag = 'v34_soft'
        pred = soft_fullspace(1.0, 1.0, 0.0)
        if best['name'].startswith('soft_a'):
            # parse soft_a1.0_b1.5_g0.0
            parts = best['name'].replace('soft_a', '').split('_')
            a = float(parts[0])
            b = float(parts[1][1:])
            g = float(parts[2][1:])
            pred = soft_fullspace(a, b, g)
            tag = f'v34_soft_a{a}_b{b}_g{g}'.replace('.', '')
        elif best['name'].startswith('soft_nov'):
            temp = float(best['name'].split('t')[-1])
            pred = soft_margin_gate(temp)
            tag = f'v34_soft_nov_t{temp}'.replace('.', '')
        preds = {fn: classes[j] for fn, j in zip(all_files, pred.tolist())}
        jp = os.path.join(OUT, f'prediction_{tag}.json')
        json.dump(preds, open(jp, 'w'))
        import zipfile
        zp = os.path.join(OUT, f'submission_{tag}.zip')
        with zipfile.ZipFile(zp, 'w', zipfile.ZIP_DEFLATED) as z:
            z.write(jp, arcname='prediction.json')
        print('wrote', zp, flush=True)
    elif args.build:
        print(f'\n--build skipped: Δ={delta:.2f}pt < kill-bar {args.kill_bar}', flush=True)


if __name__ == '__main__':
    main()
