"""Aggressive gate ranking toward 53%.

At f=0.72, uf_recall≈53.5% → unseen-folder 10.42 with b_cond≈19.5%.
Need uf_recall≈100% at same eject to hit ~53% with seen≈79%.

Trains a RankNet-style novelty scorer on TRAIN holdout (pseudo-unseen vs kept),
using multi-view shift_aug features so the gate matches test framing.
Eval tf/uf used ONLY for diagnostic AUC/recall — never baked into weights for submit
unless --build and eval AUC clears combined by ≥0.01 AND uf_rec@0.28 ≥60%.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset

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


def u_recall(sig, y, eject=0.28):
    order = sig.argsort()
    k = int(round(eject * len(sig)))
    ej = order[:k].cpu()
    return (y[ej] == 0).sum().item() / max((y == 0).sum().item(), 1)


def feat_row(q, Tt, kept, other, sb_max=None, sb_gap=None):
    sim_s = (q @ Tt[kept].t()).max(1).values
    sim_u = (q @ Tt[other].t()).max(1).values
    margin = sim_s - sim_u
    logits = q @ Tt[kept].t() / 0.07
    ent = -(F.softmax(logits, 1) * F.log_softmax(logits, 1)).sum(1)
    energy = torch.logsumexp(logits, 1)
    cols = [sim_s, margin, -ent, energy]
    if sb_max is not None:
        cols.append(sb_max)
    if sb_gap is not None:
        cols.append(sb_gap)
    return torch.stack(cols, dim=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--build', action='store_true')
    args = ap.parse_args()
    torch.set_num_threads(8)

    classes = list(pickle.load(open(os.path.join(D, 'all_classes.pkl'), 'rb')))
    ci = {c: i for i, c in enumerate(classes)}
    lab = json.load(open(os.path.join(D, 'label_train.json')))
    TtH = F.normalize(torch.load(os.path.join(OUT, 'text_emb_h_taxon.pt'), weights_only=False)['emb_taxon'].float(), dim=-1).to(dev)
    TTX = F.normalize(torch.load(os.path.join(OUT, 'text_emb_h_promptens.pt'), weights_only=False)['emb_taxctx'].float(), dim=-1).to(dev)
    Ttb = F.normalize(torch.load(os.path.join(OUT, 'text_emb_taxabind_taxctx.pt'), weights_only=False)['emb_taxctx'].float(), dim=-1).to(dev)

    # Use ctftshift + ftshift + taxabind embeddings
    tags = {
        'ctftshift': 'emb_train_ctftshift',
        'ftshift': 'emb_train_ftshift',
        'fullft336shift': 'emb_train_fullft336shift',
    }
    trains = {t: load(os.path.join(OUT, f'{p}.pt')) for t, p in tags.items()}
    tb_test = load(os.path.join(OUT, 'emb_test_taxabind.pt'))
    tb_uns = load(os.path.join(OUT, 'emb_unseen_taxabind.pt'))

    # common train files
    common = set(trains['ctftshift'][2])
    for t in trains:
        common &= set(trains[t][2])
    common = [fn for fn in common if fn in lab and lab[fn] in ci]
    by = defaultdict(list)
    for fn in common:
        by[lab[fn]].append(fn)
    seen = sorted(by.keys())
    order = sorted(seen, key=lambda c: len(by[c]))
    n_pseudo = int(len(seen) * 0.2)
    pseudo = set(order[:n_pseudo])
    kept = [c for c in seen if c not in pseudo]
    kept_idx = torch.tensor([ci[c] for c in kept], device=dev)
    other_idx = torch.tensor([i for i in range(len(classes)) if classes[i] not in set(kept)], device=dev)

    # Build train gate set
    seen_fns, novel_fns = [], []
    for c in kept:
        fns = sorted(by[c])
        k = max(1, round(0.15 * len(fns))) if len(fns) >= 4 else 0
        seen_fns.extend(fns[k:] if k else fns)  # more train
    for c in pseudo:
        novel_fns.extend(by[c])
    g = torch.Generator().manual_seed(0)
    if len(seen_fns) > 25000:
        sel = torch.randperm(len(seen_fns), generator=g)[:25000].tolist()
        seen_fns = [seen_fns[i] for i in sel]
    if len(novel_fns) > 4000:
        sel = torch.randperm(len(novel_fns), generator=g)[:4000].tolist()
        novel_fns = [novel_fns[i] for i in sel]

    def stack_tag(tag, fns):
        idx, feats, _ = trains[tag]
        return torch.stack([feats[idx[fn]] for fn in fns]).to(dev)

    def make_X(fns):
        parts = []
        for tag in ['ctftshift', 'ftshift', 'fullft336shift']:
            q = stack_tag(tag, fns)
            parts.append(feat_row(q, TtH, kept_idx, other_idx))
            parts.append(feat_row(q, TTX, kept_idx, other_idx))
        # no taxabind train emb — skip
        return torch.cat(parts, dim=1)

    Xs = make_X(seen_fns)
    Xu = make_X(novel_fns)
    Xtr = torch.cat([Xs, Xu], 0)
    ytr = torch.cat([torch.ones(len(Xs)), torch.zeros(len(Xu))]).to(dev)
    # standardize
    mu, std = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr_n = (Xtr - mu) / std

    in_dim = Xtr.shape[1]
    model = nn.Sequential(
        nn.Linear(in_dim, 64), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.1),
        nn.Linear(32, 1),
    ).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-3)
    pos = ytr.sum()
    neg = len(ytr) - pos
    w = torch.where(ytr == 1, neg / len(ytr), pos / len(ytr))

    print(f'train gate seen={len(Xs)} novel={len(Xu)} dim={in_dim}', flush=True)
    # pairwise RankNet batches
    for ep in range(120):
        model.train()
        opt.zero_grad()
        logit = model(Xtr_n).squeeze(1)
        # BCE
        loss_bce = F.binary_cross_entropy_with_logits(logit, ytr, weight=w)
        # RankNet: sample pairs
        ip = (ytr == 1).nonzero(as_tuple=True)[0]
        iu = (ytr == 0).nonzero(as_tuple=True)[0]
        n_pair = min(4096, len(ip), len(iu))
        pi = ip[torch.randint(0, len(ip), (n_pair,), device=dev)]
        ui = iu[torch.randint(0, len(iu), (n_pair,), device=dev)]
        # want score_seen > score_novel
        loss_rank = F.softplus(-(logit[pi] - logit[ui])).mean()
        loss = loss_bce + 0.5 * loss_rank
        loss.backward()
        opt.step()
        if ep % 30 == 0:
            with torch.no_grad():
                a = auc(logit, ytr)
            print(f'  ep{ep} loss={loss.item():.4f} train_auc={a:.4f}', flush=True)

    model.eval()
    with torch.no_grad():
        tr_auc = auc(model(Xtr_n).squeeze(1), ytr)

    # ---- Eval diagnostic ----
    # full train-seen classes for deployment features
    all_seen = sorted(by.keys())
    kept_full = torch.tensor([ci[c] for c in all_seen], device=dev)
    other_full = torch.tensor([i for i in range(len(classes)) if classes[i] not in set(all_seen)], device=dev)

    def load_eval(tag_train):
        te = tag_train.replace('emb_train', 'emb_test')
        ue = tag_train.replace('emb_train', 'emb_unseen')
        return load(os.path.join(OUT, f'{te}.pt')), load(os.path.join(OUT, f'{ue}.pt'))

    tests, uns = {}, {}
    for t, p in tags.items():
        tests[t], uns[t] = load_eval(p)
    tf = list(pickle.load(open(os.path.join(D, 'splits/test.pkl'), 'rb')))
    uf = list(pickle.load(open(os.path.join(D, 'splits/unseen.pkl'), 'rb')))
    # intersection
    for t in tags:
        tf = [f for f in tf if f in tests[t][0] and f in tb_test[0]]
        uf = [f for f in uf if f in uns[t][0] and f in tb_uns[0]]
    print(f'eval tf={len(tf)} uf={len(uf)}', flush=True)

    def qcat(t):
        ti, tfeat, _ = tests[t]
        ui, ufeat, _ = uns[t]
        return torch.cat([torch.stack([tfeat[ti[fn]] for fn in tf]),
                          torch.stack([ufeat[ui[fn]] for fn in uf])]).to(dev)

    def make_Xe():
        parts = []
        for tag in ['ctftshift', 'ftshift', 'fullft336shift']:
            q = qcat(tag)
            parts.append(feat_row(q, TtH, kept_full, other_full))
            parts.append(feat_row(q, TTX, kept_full, other_full))
        # taxabind
        qtb = torch.cat([
            torch.stack([tb_test[1][tb_test[0][fn]] for fn in tf]),
            torch.stack([tb_uns[1][tb_uns[0][fn]] for fn in uf]),
        ]).to(dev)
        parts.append(feat_row(qtb, Ttb, kept_full, other_full))
        return torch.cat(parts, dim=1)

    # Retrain briefly with matching dim including taxabind on eval-only? 
    # Train dim lacked TB — rebuild train features WITHOUT requiring TB, eval without TB for fair compare
    # Simpler: eval with same dim as train (no TB)
    def make_Xe_notb():
        parts = []
        for tag in ['ctftshift', 'ftshift', 'fullft336shift']:
            q = qcat(tag)
            parts.append(feat_row(q, TtH, kept_full, other_full))
            parts.append(feat_row(q, TTX, kept_full, other_full))
        return torch.cat(parts, dim=1)

    Xe = make_Xe_notb()
    Xe_n = (Xe - mu) / std
    with torch.no_grad():
        scores = model(Xe_n).squeeze(1)
    y_eval = torch.tensor([1] * len(tf) + [0] * len(uf))

    # baseline combined on ctftshift
    q = qcat('ctftshift')
    img = (q @ TtH[kept_full].t()).max(1).values
    tm = img - (q @ TtH[other_full].t()).max(1).values
    combined = z1(img) + 2 * z1(tm)

    mlp_auc = auc(scores, y_eval)
    base_auc = auc(combined, y_eval)
    # blend
    blend = z1(scores) + z1(combined)
    blend_auc = auc(blend, y_eval)

    print(f'train_auc={tr_auc:.4f}', flush=True)
    print(f'eval AUC MLP={mlp_auc:.4f} combined={base_auc:.4f} blend={blend_auc:.4f}', flush=True)
    for name, sig in [('mlp', scores), ('combined', combined), ('blend', blend)]:
        print(f'  {name} uf_rec@0.28={100*u_recall(sig,y_eval,0.28):.1f}% '
              f'@0.40={100*u_recall(sig,y_eval,0.40):.1f}%', flush=True)

    # project overall
    b, A = 0.195, 0.860
    best_sig = max([('mlp', scores), ('combined', combined), ('blend', blend)],
                   key=lambda x: auc(x[1], y_eval))
    print(f'best={best_sig[0]}', flush=True)
    for ej in [0.28, 0.35, 0.40]:
        ur = u_recall(best_sig[1], y_eval, ej)
        order = best_sig[1].argsort()
        k = int(round(ej * len(best_sig[1])))
        kept = order[k:].cpu()
        sk = (y_eval[kept] == 1).sum().item() / max((y_eval == 1).sum().item(), 1)
        po = 0.5635 * sk * A + 0.4365 * ur * b
        mark = ' ***' if po >= 0.53 else (' **' if po >= 0.505 else '')
        print(f'  eject={ej:.2f} u_rec={100*ur:.1f}% s_kept={100*sk:.1f}% proj_o={100*po:.2f}%{mark}', flush=True)

    out = {
        'train_auc': tr_auc, 'mlp_auc': mlp_auc, 'combined_auc': base_auc, 'blend_auc': blend_auc,
        'best': best_sig[0],
        'uf_rec_028': {n: u_recall(s, y_eval, 0.28) for n, s in [('mlp', scores), ('combined', combined), ('blend', blend)]},
    }
    json.dump(out, open(os.path.join(OUT, 'gate_ranknet_results.json'), 'w'), indent=1)
    torch.save({'state': model.state_dict(), 'mu': mu.cpu(), 'std': std.cpu(), 'in_dim': in_dim},
               os.path.join(OUT, 'gate_ranknet.pt'))
    print('wrote outputs/gate_ranknet_results.json', flush=True)


if __name__ == '__main__':
    main()
