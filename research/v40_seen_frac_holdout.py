"""Train-derived holdout sweep of seen_frac (routing f) on the v40 maxpool stack.

f=0.72 was tuned at v36 b (~10.4% unseen pop). v40 real unseen is 13.76% — recheck
whether a lower f (more eject) lifts holdout overall before any Codabench slot.

Compliance: uses only train labels + train-derived pseudo-unseen holdout (same split as
gate_holdout_check.py). Does NOT read eval splits for routing decisions.

  conda activate onet && python research/v40_seen_frac_holdout.py
"""
from __future__ import annotations

import json
import os
import pickle
from collections import defaultdict

import torch
import torch.nn.functional as F

OUT = 'outputs'
D = 'data/dl'
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
W_S, W_U = 0.5635, 0.4365
IMG_W = 4.0
TOPM = 4
BANK_PATH = os.environ.get('BANK_PATH', f'{OUT}/inat_photo_bank_ctftshift.pt')
FRACS = [0.55, 0.62, 0.65, 0.68, 0.72, 0.76, 0.80]


def load(p):
    d = torch.load(p, weights_only=False)
    return {fn: i for i, fn in enumerate(d['files'])}, F.normalize(d['feats'].float(), dim=-1), list(d['files'])


def zc(M):
    return (M - M.mean()) / (M.std() + 1e-6)


def z1(v):
    return (v - v.mean()) / (v.std() + 1e-6)


def dbnorm(S, tc=0.05, tr=0.5):
    return F.log_softmax(S / tc, dim=0) + F.log_softmax(S / tr, dim=1)


def sinkhorn(logits, n_iter=50, tau=2.0):
    P = torch.softmax(logits / tau, dim=1)
    n, C = P.shape
    target_col = n / C
    for _ in range(n_iter):
        P = P / P.sum(dim=1, keepdim=True).clamp(min=1e-9)
        P = P * (target_col / P.sum(dim=0, keepdim=True).clamp(min=1e-9))
    return P


def main():
    torch.set_num_threads(8)
    classes = list(pickle.load(open(f'{D}/all_classes.pkl', 'rb')))
    ci = {c: i for i, c in enumerate(classes)}
    NCLS = len(classes)
    lab = json.load(open(f'{D}/label_train.json'))
    TtH = F.normalize(torch.load(f'{OUT}/text_emb_h_taxon.pt', weights_only=False)['emb_taxon'].float(), dim=-1).to(dev)
    TnL = F.normalize(torch.load(f'{OUT}/text_emb.pt', weights_only=False)['emb_name'].float(), dim=-1).to(dev)
    TTX = F.normalize(torch.load(f'{OUT}/text_emb_h_promptens.pt', weights_only=False)['emb_taxctx'].float(), dim=-1).to(dev)
    Ttb = F.normalize(torch.load(f'{OUT}/text_emb_taxabind_taxctx.pt', weights_only=False)['emb_taxctx'].float(), dim=-1).to(dev)

    MEMBERS = [('ctftshift', True, 1.0), ('ftshift', True, 2.5), ('fullft336shift', True, 2.5),
               ('L', False, 0.0), ('fullft336_v2', True, 0.0)]
    TRAIN = {'ctftshift': 'emb_train_ctftshift', 'ftshift': 'emb_train_ftshift',
             'fullft336shift': 'emb_train_fullft336shift', 'L': 'emb_train',
             'fullft336_v2': 'emb_train_fullft336_v2'}
    train = {t: load(f'{OUT}/{TRAIN[t]}.pt') for t, _, _ in MEMBERS}
    tb_train = load(f'{OUT}/emb_train_taxabind.pt')

    common = None
    for t, (idx, _, files) in train.items():
        s = set(files)
        common = s if common is None else (common & s)
    common = [fn for fn in common if fn in lab and lab[fn] in ci and fn in tb_train[0]]
    by = defaultdict(list)
    for fn in common:
        by[lab[fn]].append(fn)
    seen = sorted(by.keys())

    order = sorted(seen, key=lambda c: len(by[c]))
    n_pseudo = int(len(seen) * 0.2)
    pseudo = set(order[:n_pseudo])
    kept_h = sorted(order[n_pseudo:])
    k2i = {c: i for i, c in enumerate(kept_h)}
    Sk = len(kept_h)
    kept_h_idx = torch.tensor([ci[c] for c in kept_h], device=dev)
    other_h_idx = torch.tensor([i for i in range(NCLS) if classes[i] not in set(kept_h)], device=dev)
    LAM = 4.0
    Tkept = TtH[kept_h_idx]

    trby = defaultdict(list)
    val_seen, val_uns = [], []
    gold_cls = {}
    for c in seen:
        fns = sorted(by[c])
        if c in pseudo:
            for f in fns:
                val_uns.append(f)
                gold_cls[f] = ci[c]
            continue
        if len(fns) >= 3:
            k = max(1, round(0.2 * len(fns)))
            for f in fns[:-k]:
                trby[c].append(f)
            for f in fns[-k:]:
                val_seen.append(f)
                gold_cls[f] = ci[c]
        else:
            for f in fns:
                trby[c].append(f)
    val_files = val_seen + val_uns
    n_seen, n_uns = len(val_seen), len(val_uns)
    print(f'holdout: seen={n_seen} pseudo-unseen={n_uns}  Sk={Sk} other={len(other_h_idx)}', flush=True)

    def protos(t):
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

    PR = {t: protos(t) for t, _, _ in MEMBERS}

    def seen_score(qF, P, TF, TL, hspace):
        qF = qF.to(dev)
        out = torch.empty(qF.shape[0], Sk, device=dev)
        for i in range(0, qF.shape[0], 2000):
            e = qF[i:i + 2000]
            ps = e @ P.t()
            sim = e @ TF.t()
            cmax = torch.full((e.shape[0], Sk), -1e9, device=dev)
            cmax.scatter_reduce_(1, TL.unsqueeze(0).expand(e.shape[0], -1), sim, reduce='amax')
            sc = ps + 2.0 * cmax
            if hspace:
                sc = sc + LAM * (e @ Tkept.t())
            out[i:i + 2000] = sc
        return out

    Qh = {t: torch.stack([train[t][1][train[t][0][fn]] for fn in val_files]).to(dev) for t, _, _ in MEMBERS}
    sb = torch.zeros(len(val_files), Sk, device=dev)
    for t, hs, w in MEMBERS:
        if w == 0:
            continue
        sb = sb + w * zc(seen_score(Qh[t], *PR[t], hs))

    Qtb = torch.stack([tb_train[1][tb_train[0][fn]] for fn in val_files]).to(dev)
    text_full = (dbnorm(Qh['ctftshift'] @ TtH.t()) + 0.5 * dbnorm(Qh['L'] @ TnL.t())
                 + 0.75 * dbnorm(Qh['fullft336_v2'] @ TtH.t()) + 1.0 * dbnorm(Qh['ftshift'] @ TtH.t())
                 + 1.0 * dbnorm(Qh['ctftshift'] @ TTX.t()) + 1.0 * dbnorm(Qtb @ Ttb.t()))
    text_unseen_base = text_full[:, other_h_idx]

    bd = torch.load(BANK_PATH, weights_only=False)
    bank = bd['bank']
    Qenc = Qh['ctftshift']
    Cu = len(other_h_idx)
    Sraw = torch.full((len(val_files), Cu), -1e4, device=dev)
    other_list = other_h_idx.tolist()
    for j, gidx in enumerate(other_list):
        photos = bank.get(gidx)
        if photos is None or (hasattr(photos, 'numel') and photos.numel() == 0):
            continue
        if not isinstance(photos, torch.Tensor):
            photos = torch.stack(photos)
        photos = F.normalize(photos.float(), dim=-1).to(dev)
        sim = Qenc @ photos.t()
        k = min(TOPM, sim.shape[1])
        Sraw[:, j] = sim.topk(k, dim=1).values.mean(dim=1)
    img_leg = dbnorm(Sraw)
    text_unseen = text_unseen_base + IMG_W * img_leg

    combined = z1(sb.max(1).values) + 2.0 * z1(
        (Qh['ctftshift'] @ TtH[kept_h_idx].t()).max(1).values
        - (Qh['ctftshift'] @ TtH[other_h_idx].t()).max(1).values
    )

    gold_t = torch.tensor([gold_cls[fn] for fn in val_files], device=dev)
    is_val_seen = torch.tensor([1 if fn in val_seen else 0 for fn in val_files], device=dev)

    rows = []
    best = None
    for f in FRACS:
        k_seen = max(1, int(round(f * len(val_files))))
        thr = torch.topk(combined, k_seen).values.min()
        route_seen = combined >= thr
        pred_g = torch.empty(len(val_files), dtype=torch.long, device=dev)
        pred_g[route_seen] = kept_h_idx[sb[route_seen].argmax(1)]
        idx_u = (~route_seen).nonzero(as_tuple=True)[0]
        sub = text_unseen[idx_u]
        pred_g[idx_u] = other_h_idx[sinkhorn(sub).argmax(1)]
        ok = (pred_g == gold_t).float()
        acc_s = ok[is_val_seen == 1].mean().item() if n_seen else 0.0
        acc_u = ok[is_val_seen == 0].mean().item() if n_uns else 0.0
        # population-weighted proxy (match competition w_seen/w_unseen on holdout counts)
        wpop = (n_seen / (n_seen + n_uns), n_uns / (n_seen + n_uns))
        hold_overall = wpop[0] * acc_s + wpop[1] * acc_u
        row = {
            'f': f, 'routed_unseen': int((~route_seen).sum().item()),
            'acc_seen_holdout': round(100 * acc_s, 3),
            'acc_unseen_holdout': round(100 * acc_u, 3),
            'holdout_overall': round(100 * hold_overall, 3),
        }
        rows.append(row)
        print(f"f={f:.2f}  route_uns={row['routed_unseen']:5d}  "
              f"seen={row['acc_seen_holdout']:.2f}%  uns={row['acc_unseen_holdout']:.2f}%  "
              f"hold={row['holdout_overall']:.2f}%", flush=True)
        if best is None or hold_overall > best[0]:
            best = (hold_overall, f)

    base = next(r for r in rows if r['f'] == 0.72)
    out = {
        'stack': 'v40_maxpool_w4_sink72',
        'img_w': IMG_W,
        'fracs': FRACS,
        'rows': rows,
        'best_holdout': {'f': best[1], 'overall_pct': round(100 * best[0], 3)},
        'delta_vs_f072_holdout_pt': round(rows[[r['f'] for r in rows].index(best[1])]['holdout_overall']
                                          - base['holdout_overall'], 3),
        'note': 'Holdout overstates real (HANDOFF §7). Real EV for f retune needs submission if delta material.',
    }
    path = f'{OUT}/v40_seen_frac_holdout_results.json'
    json.dump(out, open(path, 'w'), indent=1)
    print(f'\nbest holdout f={best[1]}  wrote {path}', flush=True)


if __name__ == '__main__':
    main()
