"""Compliance-safe gate selection: compare v36_current vs ctft+fullm gate on the
TRAIN-DERIVED holdout (pseudo-unseen vs kept-seen val), NOT on eval folders.

Baking a gate weight chosen on eval-folder AUC = folder-fit (borderline non-compliant
+ overfit). This validates the gate on train-derived data. Eval-folder AUC printed only
as a sanity cross-check.

Uses TaxaBind in the full text stack iff outputs/emb_train_taxabind.pt exists.
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


def load(p):
    d = torch.load(p, weights_only=False)
    return {fn: i for i, fn in enumerate(d['files'])}, F.normalize(d['feats'].float(), dim=-1), list(d['files'])


def zc(M):
    return (M - M.mean()) / (M.std() + 1e-6)


def z1(v):
    return (v - v.mean()) / (v.std() + 1e-6)


def dbnorm(S, tc=0.05, tr=0.5):
    return F.log_softmax(S / tc, dim=0) + F.log_softmax(S / tr, dim=1)


def auc_binary(scores, y_pos):
    s = scores.detach().float().cpu()
    y = y_pos.detach().float().cpu()
    pos, neg = s[y == 1], s[y == 0]
    g = torch.Generator().manual_seed(0)
    if len(pos) > 20000:
        pos = pos[torch.randperm(len(pos), generator=g)[:20000]]
    if len(neg) > 20000:
        neg = neg[torch.randperm(len(neg), generator=g)[:20000]]
    tot = n = 0.0
    for i in range(0, len(pos), 2000):
        p = pos[i:i + 2000]
        tot += ((p[:, None] > neg[None, :]).float() + 0.5 * (p[:, None] == neg[None, :]).float()).sum().item()
        n += p.numel() * neg.numel()
    return tot / max(n, 1)


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
    has_tb = os.path.exists(f'{OUT}/emb_train_taxabind.pt')
    tb_train = load(f'{OUT}/emb_train_taxabind.pt') if has_tb else None
    print(f'has_tb_train={has_tb}', flush=True)

    common = None
    for t, (idx, _, files) in train.items():
        s = set(files)
        common = s if common is None else (common & s)
    common = [fn for fn in common if fn in lab and lab[fn] in ci]
    if has_tb:
        common = [fn for fn in common if fn in tb_train[0]]
    by = defaultdict(list)
    for fn in common:
        by[lab[fn]].append(fn)
    seen = sorted(by.keys())

    # train-derived holdout: pseudo=rarest 20% classes; kept classes val = last 20% imgs
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
    for c in seen:
        fns = sorted(by[c])
        if c in pseudo:
            for f in fns:
                val_uns.append(f)
            continue
        if len(fns) >= 3:
            k = max(1, round(0.2 * len(fns)))
            for f in fns[:-k]:
                trby[c].append(f)
            for f in fns[-k:]:
                val_seen.append(f)
        else:
            for f in fns:
                trby[c].append(f)
    val_files = val_seen + val_uns
    y = torch.tensor([1] * len(val_seen) + [0] * len(val_uns))  # 1 = seen folder
    print(f'holdout seen={len(val_seen)} uns={len(val_uns)}', flush=True)

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

    text_full = (dbnorm(Qh['ctftshift'] @ TtH.t()) + 0.5 * dbnorm(Qh['L'] @ TnL.t())
                 + 0.75 * dbnorm(Qh['fullft336_v2'] @ TtH.t()) + 1.0 * dbnorm(Qh['ftshift'] @ TtH.t())
                 + 1.0 * dbnorm(Qh['ctftshift'] @ TTX.t()))
    if has_tb:
        Qtb = torch.stack([tb_train[1][tb_train[0][fn]] for fn in val_files]).to(dev)
        text_full = text_full + 1.0 * dbnorm(Qtb @ Ttb.t())

    img = sb.max(1).values
    tm_ctft = ((Qh['ctftshift'] @ TtH[kept_h_idx].t()).max(1).values
               - (Qh['ctftshift'] @ TtH[other_h_idx].t()).max(1).values)
    full_margin = text_full[:, kept_h_idx].max(1).values - text_full[:, other_h_idx].max(1).values

    gates = {
        'v36_current': z1(img) + 2.0 * z1(tm_ctft),
        'ctft+fullm': z1(img) + 1.0 * z1(tm_ctft) + 1.0 * z1(full_margin),
        'ctft+0.5fullm': z1(img) + 2.0 * z1(tm_ctft) + 0.5 * z1(full_margin),
        'fullm': z1(img) + 2.0 * z1(full_margin),
    }
    print('\n=== HOLDOUT gate AUC (train-derived; compliance-safe) ===', flush=True)
    res = {}
    for name, sig in gates.items():
        a = auc_binary(sig, y)
        res[name] = a
        print(f'  {name:16s} AUC={a:.4f}', flush=True)
    base = res['v36_current']
    for name in res:
        if name != 'v36_current':
            print(f'  {name:16s} dAUC={res[name]-base:+.4f}', flush=True)
    json.dump(res, open(f'{OUT}/gate_holdout_check_results.json', 'w'), indent=1)
    print('wrote outputs/gate_holdout_check_results.json', flush=True)


if __name__ == '__main__':
    main()
