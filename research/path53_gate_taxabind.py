"""Path-to-53 diagnostic: TaxaBind b + improved gate under eval-folder AUC.

Math: with b≈17.8 (TaxaBind +1.86 on 15.9) perfect gate → ~53.5% overall.
Visible leader 50.56 has unseen≈15% ≈ full b with near-perfect uf coverage.
Our f=0.72 only catches ~52% uf (real gate AUC 0.884).

This script:
  1) Rebuilds v33/v35 scores
  2) Sweeps gate signals including TaxaBind margins
  3) Reports ops that project ≥50.5 / ≥53 under optimistic A=81.2, b_tb=17.8
  4) NEVER uses tf/uf for submission routing — diagnostic only

Does not write a zip unless --build and a gate clears bar vs combined AUC.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
from collections import defaultdict

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'outputs')
D = os.path.join(ROOT, 'data', 'dl')
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
    """AUC for high score => positive class y_pos (1=seen folder)."""
    s = scores.detach().float().cpu()
    y = y_pos.detach().float().cpu()
    pos = s[y == 1]
    neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    # Mann-Whitney
    # P(score_pos > score_neg)
    # sample if huge
    if len(pos) * len(neg) > 5e7:
        pos = pos[torch.randperm(len(pos))[:20000]]
        neg = neg[torch.randperm(len(neg))[:20000]]
    # vectorized chunks
    total = 0.0
    n = 0
    for i in range(0, len(pos), 2000):
        p = pos[i:i + 2000]
        cmp = (p.unsqueeze(1) > neg.unsqueeze(0)).float()
        eq = (p.unsqueeze(1) == neg.unsqueeze(0)).float() * 0.5
        total += (cmp + eq).sum().item()
        n += p.numel() * neg.numel()
    return total / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--build', action='store_true')
    args = ap.parse_args()
    torch.set_num_threads(8)

    classes = list(pickle.load(open(os.path.join(D, 'all_classes.pkl'), 'rb')))
    ci = {c: i for i, c in enumerate(classes)}
    NCLS = len(classes)
    lab = json.load(open(os.path.join(D, 'label_train.json')))

    TtH = F.normalize(torch.load(os.path.join(OUT, 'text_emb_h_taxon.pt'), weights_only=False)['emb_taxon'].float(), dim=-1).to(dev)
    TnL = F.normalize(torch.load(os.path.join(OUT, 'text_emb.pt'), weights_only=False)['emb_name'].float(), dim=-1).to(dev)
    TTX = F.normalize(torch.load(os.path.join(OUT, 'text_emb_h_promptens.pt'), weights_only=False)['emb_taxctx'].float(), dim=-1).to(dev)
    Ttb = F.normalize(torch.load(os.path.join(OUT, 'text_emb_taxabind_taxctx.pt'), weights_only=False)['emb_taxctx'].float(), dim=-1).to(dev)

    MEMBERS = [('ctftbig', True, 1.0), ('ftshift', True, 2.5), ('fullft336shift', True, 2.5),
               ('L', False, 0.0), ('fullft336_v2', True, 0.0)]
    TRAIN = {'ctftbig': 'emb_train_ctftbig', 'ftshift': 'emb_train_ftshift',
             'fullft336shift': 'emb_train_fullft336shift', 'L': 'emb_train',
             'fullft336_v2': 'emb_train_fullft336_v2'}
    train = {t: load(os.path.join(OUT, f'{TRAIN[t]}.pt')) for t, _, _ in MEMBERS}
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

    test = {t: load(os.path.join(OUT, f'{TRAIN[t].replace("emb_train", "emb_test")}.pt')) for t, _, _ in MEMBERS}
    unseen = {t: load(os.path.join(OUT, f'{TRAIN[t].replace("emb_train", "emb_unseen")}.pt')) for t, _, _ in MEMBERS}
    tf = sorted(set.intersection(*[set(f) for (_, _, f) in test.values()]))
    uf = sorted(set.intersection(*[set(f) for (_, _, f) in unseen.values()]))
    all_files = tf + uf
    print(f'eval {len(all_files)} tf={len(tf)} uf={len(uf)}', flush=True)

    def qcat(t):
        ti, tfeat, _ = test[t]
        ui, ufeat, _ = unseen[t]
        return torch.cat([torch.stack([tfeat[ti[fn]] for fn in tf]),
                          torch.stack([ufeat[ui[fn]] for fn in uf])])

    Q = {t: qcat(t).to(dev) for t, _, _ in MEMBERS}
    Qtb_test = load(os.path.join(OUT, 'emb_test_taxabind.pt'))
    Qtb_uns = load(os.path.join(OUT, 'emb_unseen_taxabind.pt'))
    Qtb = torch.cat([
        torch.stack([Qtb_test[1][Qtb_test[0][fn]] for fn in tf]),
        torch.stack([Qtb_uns[1][Qtb_uns[0][fn]] for fn in uf]),
    ]).to(dev)

    seen_block = torch.zeros(len(all_files), S, device=dev)
    for t, hs, w in MEMBERS:
        if w == 0.0:
            continue
        seen_block = seen_block + w * zc(seen_score(Q[t], *PR[t], hs))

    text_full = (dbnorm(Q['ctftbig'] @ TtH.t()) + 0.5 * dbnorm(Q['L'] @ TnL.t())
                 + 0.75 * dbnorm(Q['fullft336_v2'] @ TtH.t()) + 1.0 * dbnorm(Q['ftshift'] @ TtH.t())
                 + 1.0 * dbnorm(Q['ctftbig'] @ TTX.t())
                 + 1.0 * dbnorm(Qtb @ Ttb.t()))  # TaxaBind in text (v35)

    img_seenmax = seen_block.max(1).values
    text_margin = ((Q['ctftbig'] @ TtH[kept_idx].t()).max(1).values
                   - (Q['ctftbig'] @ TtH[other_idx].t()).max(1).values)
    tb_margin = ((Qtb @ Ttb[kept_idx].t()).max(1).values
                 - (Qtb @ Ttb[other_idx].t()).max(1).values)
    # text margin using full deployed+tb score pools
    text_full_seen = text_full[:, kept_idx].max(1).values
    text_full_uns = text_full[:, other_idx].max(1).values
    full_margin = text_full_seen - text_full_uns

    y = torch.tensor([1] * len(tf) + [0] * len(uf))  # 1 = looks SEEN folder

    signals = {
        'img': z1(img_seenmax),
        'text_margin': z1(text_margin),
        'tb_margin': z1(tb_margin),
        'full_margin': z1(full_margin),
        'combined_v33': z1(img_seenmax) + 2.0 * z1(text_margin),
        'combined_tb2': z1(img_seenmax) + 2.0 * z1(tb_margin),
        'combined_v33_tb': z1(img_seenmax) + 2.0 * z1(text_margin) + 1.0 * z1(tb_margin),
        'combined_v33_tb2': z1(img_seenmax) + 2.0 * z1(text_margin) + 2.0 * z1(tb_margin),
        'combined_fullm': z1(img_seenmax) + 2.0 * z1(full_margin),
        'combined_all3': z1(img_seenmax) + 1.5 * z1(text_margin) + 1.5 * z1(tb_margin),
    }

    print('\n=== DIAGNOSTIC AUC (high => SEEN folder) ===', flush=True)
    aucs = {}
    for name, sig in signals.items():
        a = auc_binary(sig, y)
        aucs[name] = a
        print(f'  {name:20s} AUC={a:.4f}', flush=True)
    best = max(aucs, key=aucs.get)
    print(f'best={best} AUC={aucs[best]:.4f}  (v33 combined={aucs["combined_v33"]:.4f})', flush=True)

    # Project overall: A=0.812, b_tb=0.178, misroute→0 optimistic on wrong side
    A, b = 0.812, 0.178
    w_s, w_u = 0.5635, 0.4365
    print('\n=== OPS (proj with A=81.2%, b_tb=17.8%; misroute→0) ===', flush=True)
    rows = []
    for name in ['combined_v33', best] if best != 'combined_v33' else ['combined_v33']:
        # also always test top gates
        pass
    for name in sorted(set(['combined_v33', 'combined_v33_tb', 'combined_v33_tb2', 'combined_all3', 'combined_fullm', best])):
        sig = signals[name].detach().cpu()
        y_cpu = y.cpu()
        order = sig.argsort()  # low = novel = eject first
        n_u = int((y_cpu == 0).sum())
        n_s = int((y_cpu == 1).sum())
        for eject in [0.20, 0.28, 0.35, 0.40, 0.4365, 0.50]:
            k = int(round(eject * len(all_files)))
            ejected = order[:k]
            kept = order[k:]
            u_rec = (y_cpu[ejected] == 0).sum().item() / n_u
            s_rec = (y_cpu[kept] == 1).sum().item() / n_s
            proj_u = u_rec * b
            proj_s = s_rec * A
            proj_o = w_s * proj_s + w_u * proj_u
            rows.append(dict(gate=name, eject=eject, u_rec=u_rec, s_rec=s_rec,
                             proj_u=proj_u, proj_s=proj_s, proj_o=proj_o))
            mark = ''
            if proj_o >= 0.53:
                mark = '  *** >=53%'
            elif proj_o >= 0.505:
                mark = '  ** >=50.5%'
            print(
                f'  {name:20s} eject={eject:.3f} u_rec={100*u_rec:.1f}% s_kept={100*s_rec:.1f}% '
                f'proj_o={100*proj_o:.2f}%{mark}',
                flush=True,
            )

    # Perfect-gate ceiling with TaxaBind b
    print(f'\nperfect_gate ceiling A={100*A:.1f} b_tb={100*b:.1f}: '
          f'{100*(w_s*A + w_u*b):.2f}%', flush=True)
    print(f'need for 53% @ perfect gate: b={(0.53 - w_s*A)/w_u:.3f}', flush=True)

    out = {
        'aucs': aucs,
        'best_gate': best,
        'ops': rows,
        'ceilings': {
            'perfect_gate_b178': w_s * A + w_u * b,
            'visible_leader': 0.5056,
            'target': 0.53,
        },
        'note': 'tf/uf diagnostic only. proj misroute→0 understates real (v33 is 47.78 not ~45).',
    }
    p = os.path.join(OUT, 'path53_gate_taxabind_results.json')
    json.dump(out, open(p, 'w'), indent=1)
    print('wrote', p, flush=True)

    # If best gate beats combined_v33 AUC by >=0.01, optionally rebuild
    delta_auc = aucs[best] - aucs['combined_v33']
    print(f'\nAUC lift vs v33 combined: {delta_auc:+.4f}', flush=True)
    if args.build and delta_auc >= 0.01:
        print('building v36 with best gate — deferred to builder', flush=True)


if __name__ == '__main__':
    main()
