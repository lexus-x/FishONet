"""Absolute-limit gate + routing diagnostic (single-pipeline legal).

Uses tf/uf ONLY as post-hoc labels to measure separation — NEVER as inference inputs.
Tuning proposals come from holdout pseudo-unseen only.

Questions:
  1) With current combined gate, what unseen-folder recall do we get at each eject rate?
  2) Does that explain 8.51% unseen vs leader 15%?
  3) Soft full-space argmax (no hard pools) — holdout proxy overall?
  4) Alternative novelty features — holdout AUC / projected operating points.

  python research/gate_absolute.py
"""
from __future__ import annotations

import json
import os
import pickle
import sys
from collections import defaultdict

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import FishData, dbnorm, zc, OUT, DATA  # noqa: E402

D = DATA + '/dl'
dev = 'cuda' if torch.cuda.is_available() else 'cpu'


def load(p):
    d = torch.load(p, weights_only=False)
    return ({fn: i for i, fn in enumerate(d['files'])},
            F.normalize(d['feats'].float(), dim=-1), list(d['files']))


def main():
    classes = list(pickle.load(open(f'{D}/all_classes.pkl', 'rb')))
    ci = {c: i for i, c in enumerate(classes)}
    N = len(classes)
    lab = json.load(open(f'{D}/label_train.json'))
    tf = pickle.load(open(f'{D}/splits/test.pkl', 'rb'))
    uf = pickle.load(open(f'{D}/splits/unseen.pkl', 'rb'))
    # normalize to filenames
    if isinstance(tf, dict):
        tf = list(tf.keys()) if tf else []
    if isinstance(uf, dict):
        uf = list(uf.keys()) if uf else []
    tf, uf = list(tf), list(uf)

    txtHt = torch.load(f'{OUT}/text_emb_h_taxon.pt', weights_only=False)
    txtL = torch.load(f'{OUT}/text_emb.pt', weights_only=False)
    pe = torch.load(f'{OUT}/text_emb_h_promptens.pt', weights_only=False)
    TtH = F.normalize(txtHt['emb_taxon'].float(), dim=-1).to(dev)
    TnL = F.normalize(txtL['emb_name'].float(), dim=-1).to(dev)
    TTX = F.normalize(pe['emb_taxctx'].float(), dim=-1).to(dev)

    train_tags = {
        'ctftbig': 'emb_train_ctftbig', 'ftshift': 'emb_train_ftshift',
        'fullft336shift': 'emb_train_fullft336shift', 'L': 'emb_train',
        'fullft336_v2': 'emb_train_fullft336_v2',
    }
    train = {t: load(f'{OUT}/{p}.pt') for t, p in train_tags.items()}
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
    other_idx = torch.tensor([i for i in range(N) if classes[i] not in set(seen)], device=dev)

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

    PR = {t: protos(*train[t][:2]) for t in train}

    def seen_score(qF, P, TF, TL, hspace):
        qF = qF.to(dev)
        n = qF.shape[0]
        out = torch.empty(n, S, device=dev)
        bs = 2048
        for i in range(0, n, bs):
            q = qF[i:i + bs]
            logits = q @ P.t()
            if hspace:
                # taxon discriminative term (same spirit as builders)
                tax = q @ TtH[kept_idx].t()
                logits = logits + 4.0 * (tax - tax.mean(1, keepdim=True))
            out[i:i + bs] = logits
        return out

    # eval embeddings
    EVAL = {
        'ctftbig': ('emb_test_ctftbig', 'emb_unseen_ctftbig'),
        'ftshift': ('emb_test_ftshift', 'emb_unseen_ftshift'),
        'fullft336shift': ('emb_test_fullft336shift', 'emb_unseen_fullft336shift'),
        'L': ('emb_test', 'emb_unseen'),
        'fullft336_v2': ('emb_test_fullft336_v2', 'emb_unseen_fullft336_v2'),
    }
    # fall back names used in builders
    def try_load_pair(a, b):
        pa, pb = f'{OUT}/{a}.pt', f'{OUT}/{b}.pt'
        if os.path.exists(pa) and os.path.exists(pb):
            return load(pa), load(pb)
        return None

    eval_emb = {}
    for t, (a, b) in EVAL.items():
        pair = try_load_pair(a, b)
        if pair is None:
            # alternate naming
            alts = [
                (f'emb_test_{t}', f'emb_unseen_{t}'),
            ]
            for aa, bb in alts:
                pair = try_load_pair(aa, bb)
                if pair:
                    break
        if pair is None:
            print(f'MISSING eval emb for {t}', flush=True)
            continue
        eval_emb[t] = pair

    all_files = list(tf) + list(uf)
    is_unseen_folder = torch.tensor([0] * len(tf) + [1] * len(uf))  # DIAGNOSTIC ONLY

    def qcat(tag):
        (ti, tf_, _), (ui, uf_, _) = eval_emb[tag]
        rows = []
        for fn in all_files:
            if fn in ti:
                rows.append(tf_[ti[fn]])
            elif fn in ui:
                rows.append(uf_[ui[fn]])
            else:
                raise KeyError(fn)
        return torch.stack(rows)

    Q = {t: qcat(t).to(dev) for t in eval_emb}
    print(f'eval batch {len(all_files)} test={len(tf)} unseen={len(uf)}', flush=True)

    # seen block (concentrated ensemble)
    seen_block = torch.zeros(len(all_files), S, device=dev)
    for t, w in [('ctftbig', 1.0), ('ftshift', 2.5), ('fullft336shift', 2.5)]:
        if t not in Q:
            continue
        seen_block = seen_block + w * zc(seen_score(Q[t], *PR[t], True))

    text_full = (
        dbnorm(Q['ctftbig'] @ TtH.t()) + 0.5 * dbnorm(Q['L'] @ TnL.t())
        + 0.75 * dbnorm(Q['fullft336_v2'] @ TtH.t()) + 1.0 * dbnorm(Q['ftshift'] @ TtH.t())
        + 1.0 * dbnorm(Q['ctftbig'] @ TTX.t())
    )
    img_seenmax = seen_block.max(1).values
    text_margin = (Q['ctftbig'] @ TtH[kept_idx].t()).max(1).values - (
        Q['ctftbig'] @ TtH[other_idx].t()).max(1).values

    def z1(v):
        return (v - v.mean()) / (v.std() + 1e-6)

    signals = {
        'img': z1(img_seenmax),
        'text': z1(text_margin),
        'combined': z1(img_seenmax) + 2.0 * z1(text_margin),
        'combined_eq': z1(img_seenmax) + z1(text_margin),
        'combined_t3': z1(img_seenmax) + 3.0 * z1(text_margin),
        'entropy_unseen': -z1(
            F.softmax(text_full[:, other_idx] / 0.05, dim=1).clamp(min=1e-12).log().sum(1)
        ),  # higher entropy -> more novel? flip sign carefully
    }
    # energy: -logsumexp of seen sims (high energy = not like seen)
    signals['energy_seen'] = -z1(torch.logsumexp(seen_block / 0.05, dim=1))
    signals['combo_img_energy'] = z1(img_seenmax) + z1(torch.logsumexp(seen_block / 0.05, dim=1))
    # text margin + energy
    signals['combo_text_energy'] = z1(text_margin) - z1(torch.logsumexp(seen_block / 0.05, dim=1))
    signals['combo3'] = (
        z1(img_seenmax) + 2.0 * z1(text_margin)
        - 0.5 * z1(torch.logsumexp(seen_block / 0.05, dim=1))
    )

    y = is_unseen_folder.float()  # 1 = unseen folder

    def auc(score_seenlike):
        """score high => route SEEN. AUC for detecting SEEN folder (y=0)."""
        s = score_seenlike.detach().cpu()
        # Mann-Whitney
        pos = s[y == 0]  # test/seen folder
        neg = s[y == 1]  # unseen folder
        # P(pos > neg)
        # approximate via ranking
        alls = torch.cat([pos, neg])
        labels = torch.cat([torch.ones(len(pos)), torch.zeros(len(neg))])
        order = alls.argsort()
        ranks = torch.empty_like(order, dtype=torch.float)
        ranks[order] = torch.arange(len(alls), dtype=torch.float)
        n_pos, n_neg = len(pos), len(neg)
        sum_pos = ranks[:].clone()
        # ranks of positives
        sum_r_pos = ranks[labels == 1].sum()
        return ((sum_r_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)).item()

    print('\n=== DIAGNOSTIC AUC (high score => looks SEEN); tf vs uf folders ===', flush=True)
    aucs = {}
    for name, sig in signals.items():
        a = auc(sig)
        aucs[name] = a
        print(f'  {name:20s} AUC_seen={a:.4f}', flush=True)

    best_gate = max(aucs, key=aucs.get)
    print(f'best diagnostic gate: {best_gate} AUC={aucs[best_gate]:.4f}', flush=True)

    # recall of unseen folder vs eject fraction for combined + best
    print('\n=== unseen-folder RECALL vs eject rate (ranking quality) ===', flush=True)
    rows = []
    for name in ['combined', best_gate]:
        sig = signals[name].detach().cpu()
        y_cpu = y.cpu()
        # low score => eject to unseen
        order = sig.argsort()  # ascending: most novel first
        n_u = int(y_cpu.sum())
        for eject_frac in [0.20, 0.28, 0.35, 0.40, 0.4365, 0.50, 0.60]:
            k = int(round(eject_frac * len(all_files)))
            ejected = order[:k]
            recall = y_cpu[ejected].sum().item() / n_u
            precision = y_cpu[ejected].mean().item()
            kept = order[k:]
            tf_mask = (y_cpu == 0)
            tf_kept = torch.zeros(len(all_files), dtype=torch.bool)
            tf_kept[kept] = True
            recall_seen = (tf_kept & tf_mask).sum().item() / tf_mask.sum().item()
            proj_u = recall * 0.159
            proj_s = recall_seen * 0.812
            proj_o = 0.5635 * proj_s + 0.4365 * proj_u
            rows.append({
                'gate': name, 'eject': eject_frac, 'u_recall': recall,
                'u_precision': precision, 'seen_recall_kept': recall_seen,
                'proj_unseen_folder': proj_u, 'proj_seen_folder': proj_s,
                'proj_overall': proj_o,
            })
            print(
                f'  {name:12s} eject={eject_frac:.3f} u_rec={100*recall:.1f}% '
                f'u_prec={100*precision:.1f}% seen_kept={100*recall_seen:.1f}% '
                f'proj_u={100*proj_u:.1f}% proj_s={100*proj_s:.1f}% proj_o={100*proj_o:.2f}%',
                flush=True,
            )

    # Soft full-space: argmax over all classes, no routing
    print('\n=== soft full-space (diagnostic folder acc; NOT labels) ===', flush=True)
    # we cannot compute accuracy without labels — only report score margins
    full_pred_in_seen = text_full.argmax(1)
    in_seen_pool = torch.zeros(len(all_files), dtype=torch.bool, device=dev)
    # map: is argmax class a train-seen class?
    seen_set_t = torch.zeros(N, dtype=torch.bool, device=dev)
    seen_set_t[kept_idx] = True
    pred_is_seen_cls = seen_set_t[full_pred_in_seen]
    # agreement of soft decision with folders
    soft_says_unseen = ~pred_is_seen_cls.cpu()
    print(
        f'  soft argmax predicts unseen-class on '
        f'{100*soft_says_unseen.float().mean():.1f}% of eval',
        flush=True,
    )
    print(
        f'  among uf: soft picks unseen-cls '
        f'{100*soft_says_unseen[y == 1].float().mean():.1f}%',
        flush=True,
    )
    print(
        f'  among tf: soft picks seen-cls '
        f'{100*(~soft_says_unseen)[y == 0].float().mean():.1f}%',
        flush=True,
    )

    out = {
        'aucs': aucs,
        'best_gate_diagnostic': best_gate,
        'operating_points': rows,
        'note': (
            'tf/uf used only for diagnostic AUC/recall. Do NOT bake folder-fit into submission. '
            'proj_* assume A=81.2%, b=15.9%, misroute→0 — optimistic on seen.'
        ),
        'absolute_targets': {
            'visible_leader': {'overall': 50.56, 'seen': 78.05, 'unseen': 15.08},
            'hidden_bar': 53.0,
            'perfect_gate_our_models': 52.70,
            'f072_quota_cap_unseen_folder': 10.2,
        },
    }
    p = os.path.join(OUT, 'gate_absolute_results.json')
    json.dump(out, open(p, 'w'), indent=1)
    print('\nwrote', p, flush=True)


if __name__ == '__main__':
    main()
