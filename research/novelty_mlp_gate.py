"""Train a shift-aware novelty gate on TRAIN-ONLY holdout; score AUC on eval folders (diagnostic).

Goal: raise real gate AUC from ~0.90 toward ~0.95 so TaxaBind b (~17.8%) can approach
perfect-gate overall ~53.5%.

Legal: uses label_train + train images only. Eval tf/uf used ONLY for diagnostic AUC.
"""
from __future__ import annotations

import json
import os
import pickle
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'outputs')
D = os.path.join(ROOT, 'data', 'dl')
dev = 'cuda' if torch.cuda.is_available() else 'cpu'


def load(p):
    d = torch.load(p, weights_only=False)
    return {fn: i for i, fn in enumerate(d['files'])}, F.normalize(d['feats'].float(), dim=-1), list(d['files'])


def z1(v):
    return (v - v.mean()) / (v.std() + 1e-6)


def auc(scores, y_pos):
    s = scores.detach().float().cpu()
    y = y_pos.detach().float().cpu()
    pos, neg = s[y == 1], s[y == 0]
    # deterministic subsample
    g = torch.Generator().manual_seed(0)
    if len(pos) > 25000:
        pos = pos[torch.randperm(len(pos), generator=g)[:25000]]
    if len(neg) > 25000:
        neg = neg[torch.randperm(len(neg), generator=g)[:25000]]
    total = 0.0
    n = 0
    for i in range(0, len(pos), 2500):
        p = pos[i:i + 2500]
        cmp = (p.unsqueeze(1) > neg.unsqueeze(0)).float()
        eq = (p.unsqueeze(1) == neg.unsqueeze(0)).float() * 0.5
        total += (cmp + eq).sum().item()
        n += p.numel() * neg.numel()
    return total / max(n, 1)


def feats_from(q, TtH, kept_idx, other_idx, seen_block=None):
    """Per-image novelty features (higher => more SEEN-like)."""
    sim_s = (q @ TtH[kept_idx].t()).max(1).values
    sim_u = (q @ TtH[other_idx].t()).max(1).values
    margin = sim_s - sim_u
    # entropy of top text sims over seen
    logits = q @ TtH[kept_idx].t() / 0.07
    ent = -(F.softmax(logits, 1) * F.log_softmax(logits, 1)).sum(1)
    energy = torch.logsumexp(logits, 1)
    cols = [z1(sim_s), z1(margin), z1(-ent), z1(energy)]
    if seen_block is not None:
        cols.append(z1(seen_block.max(1).values))
        # gap top1-top2 in seen_block
        top2 = seen_block.topk(2, dim=1).values
        cols.append(z1(top2[:, 0] - top2[:, 1]))
    return torch.stack(cols, dim=1)


def main():
    torch.set_num_threads(8)
    classes = list(pickle.load(open(os.path.join(D, 'all_classes.pkl'), 'rb')))
    ci = {c: i for i, c in enumerate(classes)}
    lab = json.load(open(os.path.join(D, 'label_train.json')))
    TtH = F.normalize(torch.load(os.path.join(OUT, 'text_emb_h_taxon.pt'), weights_only=False)['emb_taxon'].float(), dim=-1).to(dev)
    TTX = F.normalize(torch.load(os.path.join(OUT, 'text_emb_h_promptens.pt'), weights_only=False)['emb_taxctx'].float(), dim=-1).to(dev)

    # use ctftbig train feats
    idx, feats, files = load(os.path.join(OUT, 'emb_train_ctftbig.pt'))
    by = defaultdict(list)
    for fn in files:
        if fn in lab and lab[fn] in ci:
            by[lab[fn]].append(fn)
    seen = sorted(by.keys())
    order = sorted(seen, key=lambda c: len(by[c]))
    n_pseudo = int(len(seen) * 0.2)
    pseudo = set(order[:n_pseudo])
    kept = [c for c in seen if c not in pseudo]
    kept_idx = torch.tensor([ci[c] for c in kept], device=dev)
    # cand = everything not in kept (pseudo + true unseen classes)
    other_idx = torch.tensor([i for i, c in enumerate(classes) if c not in set(kept)], device=dev)

    # train-gate set: images from kept classes (label SEEN) vs all images from pseudo (label NOVEL)
    seen_fns, novel_fns = [], []
    for c in kept:
        fns = sorted(by[c])
        # use last 20% as gate-val, rest gate-train
        k = max(1, round(0.2 * len(fns))) if len(fns) >= 3 else 0
        seen_fns.extend(fns[:-k] if k else fns)
    for c in pseudo:
        novel_fns.extend(by[c])

    def stack(fns):
        return torch.stack([feats[idx[fn]] for fn in fns]).to(dev)

    # subsample for speed
    g = torch.Generator().manual_seed(0)
    if len(seen_fns) > 20000:
        sel = torch.randperm(len(seen_fns), generator=g)[:20000].tolist()
        seen_fns = [seen_fns[i] for i in sel]
    if len(novel_fns) > 8000:
        sel = torch.randperm(len(novel_fns), generator=g)[:8000].tolist()
        novel_fns = [novel_fns[i] for i in sel]

    Xs = feats_from(stack(seen_fns), TtH, kept_idx, other_idx)
    Xu = feats_from(stack(novel_fns), TtH, kept_idx, other_idx)
    # also taxctx margin features
    Xs2 = feats_from(stack(seen_fns), TTX, kept_idx, other_idx)
    Xu2 = feats_from(stack(novel_fns), TTX, kept_idx, other_idx)
    Xtr = torch.cat([torch.cat([Xs, Xs2], 1), torch.cat([Xu, Xu2], 1)], 0)
    ytr = torch.cat([torch.ones(len(Xs)), torch.zeros(len(Xu))]).to(dev)

    # simple logistic / MLP
    in_dim = Xtr.shape[1]
    model = nn.Sequential(
        nn.Linear(in_dim, 32), nn.ReLU(), nn.Dropout(0.1),
        nn.Linear(32, 1),
    ).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-3)
    # class balance
    pos = ytr.sum()
    neg = len(ytr) - pos
    w = torch.where(ytr == 1, neg / len(ytr), pos / len(ytr))

    print(f'train gate: seen={len(Xs)} novel={len(Xu)} dim={in_dim}', flush=True)
    model.train()
    for ep in range(80):
        opt.zero_grad()
        logit = model(Xtr).squeeze(1)
        loss = F.binary_cross_entropy_with_logits(logit, ytr, weight=w)
        loss.backward()
        opt.step()
        if ep % 20 == 0:
            with torch.no_grad():
                a = auc(logit, ytr)
            print(f'  ep{ep} loss={loss.item():.4f} train_auc={a:.4f}', flush=True)

    model.eval()
    with torch.no_grad():
        tr_auc = auc(model(Xtr).squeeze(1), ytr)
    print(f'train AUC={tr_auc:.4f}', flush=True)

    # Eval diagnostic on real folders
    ti, tfeat, tfiles = load(os.path.join(OUT, 'emb_test_ctftbig.pt'))
    ui, ufeat, ufiles = load(os.path.join(OUT, 'emb_unseen_ctftbig.pt'))
    tf = list(pickle.load(open(os.path.join(D, 'splits/test.pkl'), 'rb')))
    uf = list(pickle.load(open(os.path.join(D, 'splits/unseen.pkl'), 'rb')))
    tf = [f for f in tf if f in ti]
    uf = [f for f in uf if f in ui]
    q = torch.cat([
        torch.stack([tfeat[ti[fn]] for fn in tf]),
        torch.stack([ufeat[ui[fn]] for fn in uf]),
    ]).to(dev)
    # For eval gate, use FULL train-seen classes (deployment), not holdout kept
    all_seen = sorted(by.keys())
    kept_full = torch.tensor([ci[c] for c in all_seen], device=dev)
    other_full = torch.tensor([i for i in range(len(classes)) if classes[i] not in set(all_seen)], device=dev)
    Xe = torch.cat([
        feats_from(q, TtH, kept_full, other_full),
        feats_from(q, TTX, kept_full, other_full),
    ], 1)
    with torch.no_grad():
        scores = model(Xe).squeeze(1)
    y_eval = torch.tensor([1] * len(tf) + [0] * len(uf))
    eval_auc = auc(scores, y_eval)

    # baselines
    margin = z1((q @ TtH[kept_full].t()).max(1).values - (q @ TtH[other_full].t()).max(1).values)
    # need img for combined — approximate with sim_s
    img = z1((q @ TtH[kept_full].t()).max(1).values)
    combined = img + 2 * margin
    base_auc = auc(combined, y_eval)
    print(f'eval-folder AUC MLP={eval_auc:.4f}  approx_combined={base_auc:.4f}', flush=True)

    # uf recall at eject 0.28
    def recall_at(sig, eject=0.28):
        order = sig.argsort()  # low = novel
        k = int(round(eject * len(sig)))
        ej = order[:k]
        return (y_eval[ej.cpu()] == 0).float().mean().item(), (y_eval[ej.cpu()] == 0).sum().item() / max((y_eval == 0).sum().item(), 1)

    # fix recall: fraction of uf that are ejected
    def u_recall(sig, eject=0.28):
        order = sig.argsort()
        k = int(round(eject * len(sig)))
        ej = order[:k].cpu()
        return ((y_eval[ej] == 0).sum().item() / (y_eval == 0).sum().item())

    print(f'uf_recall@eject0.28 MLP={100*u_recall(scores):.1f}% combined~={100*u_recall(combined):.1f}%', flush=True)

    out = {
        'train_auc': tr_auc,
        'eval_auc_mlp': eval_auc,
        'eval_auc_approx_combined': base_auc,
        'uf_recall_e28_mlp': u_recall(scores),
        'uf_recall_e28_combined': u_recall(combined),
    }
    # project with TaxaBind b
    A, b = 0.812, 0.178
    for name, ur in [('mlp', out['uf_recall_e28_mlp']), ('combined', out['uf_recall_e28_combined'])]:
        # rough: assume s_rec ≈ 1 - eject*(1-precision); skip — report ur only
        print(f'  {name}: uf_rec={100*ur:.1f}% → proj_unseen_folder≈{100*ur*b:.1f}%', flush=True)
    torch.save({'state': model.state_dict(), 'in_dim': in_dim}, os.path.join(OUT, 'novelty_mlp.pt'))
    json.dump(out, open(os.path.join(OUT, 'novelty_mlp_results.json'), 'w'), indent=1)
    print('wrote outputs/novelty_mlp_results.json', flush=True)


if __name__ == '__main__':
    main()
