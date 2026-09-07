"""Gate diagnostic on the EXACT v36 ensemble: does a full-stack text margin
beat the current single-leg (ctftshift) text_margin in the combined gate?

path53 (ctftbig-based) hinted combined_fullm > combined_v33 by +0.0034 AUC.
Here we use the real v36 seen_block + full text stack (incl TaxaBind) and
measure real-folder AUC + honest projected overall using the CALIBRATED
constants that reproduce v36 real 49.04 (A_kept=0.8603, b_cond=0.1948).

tf/uf (splits) are used ONLY for diagnostic AUC/recall — never for routing.
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
# calibrated to reproduce v36 real (seen 78.95 / unseen 10.42 @ f=0.72)
A_KEPT, B_COND = 0.8603, 0.1948
W_S, W_U = 0.5635, 0.4365


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

    test = {t: load(f'{OUT}/{TRAIN[t].replace("emb_train", "emb_test")}.pt') for t, _, _ in MEMBERS}
    unseen = {t: load(f'{OUT}/{TRAIN[t].replace("emb_train", "emb_unseen")}.pt') for t, _, _ in MEMBERS}
    tb_test = load(f'{OUT}/emb_test_taxabind.pt')
    tb_uns = load(f'{OUT}/emb_unseen_taxabind.pt')
    tf = sorted(set.intersection(*[set(f) for (_, _, f) in test.values()], set(tb_test[2])))
    uf = sorted(set.intersection(*[set(f) for (_, _, f) in unseen.values()], set(tb_uns[2])))
    all_files = tf + uf
    print(f'eval batch: {len(all_files)}  tf={len(tf)} uf={len(uf)}', flush=True)

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
    for t, hs, w in MEMBERS:
        if w == 0.0:
            continue
        seen_block = seen_block + w * zc(seen_score(Q[t], *PR[t], hs))

    text_full = (dbnorm(Q['ctftshift'] @ TtH.t()) + 0.5 * dbnorm(Q['L'] @ TnL.t())
                 + 0.75 * dbnorm(Q['fullft336_v2'] @ TtH.t()) + 1.0 * dbnorm(Q['ftshift'] @ TtH.t())
                 + 1.0 * dbnorm(Q['ctftshift'] @ TTX.t()) + 1.0 * dbnorm(Qtb @ Ttb.t()))

    img_seenmax = seen_block.max(1).values
    # current gate margin (single leg ctftshift @ taxon)
    tm_ctft = ((Q['ctftshift'] @ TtH[kept_idx].t()).max(1).values
               - (Q['ctftshift'] @ TtH[other_idx].t()).max(1).values)
    # full-stack margin
    full_seen = text_full[:, kept_idx].max(1).values
    full_uns = text_full[:, other_idx].max(1).values
    full_margin = full_seen - full_uns
    # TaxaBind-only margin
    tb_margin = ((Qtb @ Ttb[kept_idx].t()).max(1).values - (Qtb @ Ttb[other_idx].t()).max(1).values)

    y = torch.tensor([1] * len(tf) + [0] * len(uf))  # 1 = seen folder

    gates = {
        'v36_current': z1(img_seenmax) + 2.0 * z1(tm_ctft),
        'fullm': z1(img_seenmax) + 2.0 * z1(full_margin),
        'fullm_1.5': z1(img_seenmax) + 1.5 * z1(full_margin),
        'fullm_2.5': z1(img_seenmax) + 2.5 * z1(full_margin),
        'ctft+fullm': z1(img_seenmax) + 1.0 * z1(tm_ctft) + 1.0 * z1(full_margin),
        'ctft+0.5fullm': z1(img_seenmax) + 2.0 * z1(tm_ctft) + 0.5 * z1(full_margin),
        'ctft+tb': z1(img_seenmax) + 2.0 * z1(tm_ctft) + 0.5 * z1(tb_margin),
        'fullm+tb': z1(img_seenmax) + 2.0 * z1(full_margin) + 0.5 * z1(tb_margin),
    }

    n_all = len(all_files)
    n_u = int((y == 0).sum())
    n_s = int((y == 1).sum())
    print('\n=== gate AUC + projected overall (calibrated A_kept=%.4f b_cond=%.4f) ===' % (A_KEPT, B_COND), flush=True)
    results = {}
    for name, sig in gates.items():
        a = auc_binary(sig, y)
        order = sig.detach().cpu().argsort()  # low = eject first
        row = {'auc': a, 'ops': {}}
        best_o = 0.0
        for eject in [0.24, 0.26, 0.28, 0.30, 0.32]:
            k = int(round(eject * n_all))
            ej = order[:k]
            kept = order[k:]
            u_rec = (y[ej] == 0).sum().item() / n_u
            s_kept = (y[kept] == 1).sum().item() / n_s
            proj_o = W_S * s_kept * A_KEPT + W_U * u_rec * B_COND
            row['ops'][f'{eject:.2f}'] = {'u_rec': u_rec, 's_kept': s_kept, 'proj_o': proj_o}
            best_o = max(best_o, proj_o)
        row['best_proj_o'] = best_o
        results[name] = row
        o28 = row['ops']['0.28']
        print(f'  {name:16s} AUC={a:.4f}  @0.28 u_rec={100*o28["u_rec"]:.1f}% '
              f's_kept={100*o28["s_kept"]:.1f}% proj_o={100*o28["proj_o"]:.2f}%  '
              f'best_proj={100*best_o:.2f}%', flush=True)

    base = results['v36_current']
    print(f'\nv36_current AUC={base["auc"]:.4f} proj@0.28={100*base["ops"]["0.28"]["proj_o"]:.2f}%', flush=True)
    for name in results:
        if name == 'v36_current':
            continue
        d_auc = results[name]['auc'] - base['auc']
        d_o = results[name]['ops']['0.28']['proj_o'] - base['ops']['0.28']['proj_o']
        print(f'  {name:16s} dAUC={d_auc:+.4f}  dproj@0.28={100*d_o:+.3f}pt', flush=True)

    json.dump(results, open(f'{OUT}/gate_v36_margin_results.json', 'w'), indent=1)
    print('\nwrote outputs/gate_v36_margin_results.json', flush=True)


if __name__ == '__main__':
    main()
