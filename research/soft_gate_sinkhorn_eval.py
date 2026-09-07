"""Benchmark 2-stage soft pipeline with Sinkhorn optimal transport on unseen route.
"""
import os
import sys
import json
import pickle
import torch
import torch.nn.functional as F
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import FishData, dbnorm, zc, load_emb, OUT, DATA, DEV

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
    dev = DEV
    classes = list(pickle.load(open(os.path.join(DATA, 'dl', 'all_classes.pkl'), 'rb')))
    ci = {c: i for i, c in enumerate(classes)}
    NCLS = len(classes)
    lab = json.load(open(os.path.join(DATA, 'dl', 'label_train.json')))
    
    txtHt = torch.load(os.path.join(OUT, 'text_emb_h_taxon.pt'), weights_only=False)
    txtL = torch.load(os.path.join(OUT, 'text_emb.pt'), weights_only=False)
    _pe = torch.load(os.path.join(OUT, 'text_emb_h_promptens.pt'), weights_only=False)
    TtH = F.normalize(txtHt['emb_taxon'].float(), dim=-1).to(dev)
    TnL = F.normalize(txtL['emb_name'].float(), dim=-1).to(dev)
    TTX = F.normalize(_pe['emb_taxctx'].float(), dim=-1).to(dev)
    
    MEMBERS = [('ctftshift', True, 1.0), ('ftshift', True, 2.5), ('fullft336shift', True, 2.5),
               ('L', False, 0.0), ('fullft336_v2', True, 0.0)]
    TRAIN = {'ctftshift': 'emb_train_ctftshift', 'ftshift': 'emb_train_ftshift',
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
    
    order = sorted(seen, key=lambda c: len(by[c]))
    n_pseudo = int(len(seen) * 0.2)
    pseudo = set(order[:n_pseudo])
    kept_h = sorted(order[n_pseudo:])
    kept_h_set = set(kept_h)
    k2i = {c: i for i, c in enumerate(kept_h)}
    Sk = len(kept_h)
    kept_h_idx = torch.tensor([ci[c] for c in kept_h], device=dev)
    other_h_idx = torch.tensor([i for i in range(NCLS) if classes[i] not in kept_h_set], device=dev)
    
    val_seen, val_uns = [], []
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
    LAM = 4.0

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
    
    Qh = {}
    for t, _, _ in MEMBERS:
        idx, feats, _ = train[t]
        Qh[t] = torch.stack([feats[idx[fn]] for fn in val_files]).to(dev)

    seen_block_h = torch.zeros(len(val_files), Sk, device=dev)
    for t, hs, w in MEMBERS:
        if w == 0.0:
            continue
        seen_block_h = seen_block_h + w * zc(seen_score_h(Qh[t], *PR_h[t], hs))

    text_full_h = (dbnorm(Qh['ctftshift'] @ TtH.t()) + 0.5 * dbnorm(Qh['L'] @ TnL.t())
                   + 0.75 * dbnorm(Qh['fullft336_v2'] @ TtH.t()) + 1.0 * dbnorm(Qh['ftshift'] @ TtH.t())
                   + 1.0 * dbnorm(Qh['ctftshift'] @ TTX.t()))
    text_uns_h = text_full_h[:, other_h_idx]

    img_max_h = seen_block_h.max(1).values
    text_margin_h = ((Qh['ctftshift'] @ TtH[kept_h_idx].t()).max(1).values
                     - (Qh['ctftshift'] @ TtH[other_h_idx].t()).max(1).values)
    combined_h = z1(img_max_h) + 2.0 * z1(text_margin_h)

    def eval_accuracy(pred_names):
        ok = [p == g for p, g in zip(pred_names, gold)]
        s_acc = sum(ok[:len(val_seen)]) / len(val_seen)
        u_acc = sum(ok[len(val_seen):]) / len(val_uns)
        o_acc = 0.5635 * s_acc + 0.4365 * u_acc
        return s_acc, u_acc, o_acc

    print("=" * 75)
    print(" HOLDOUT EVALUATION: 2-STAGE SOFT PIPELINE WITH SINKHORN UNSEEN OPTIMAL TRANSPORT")
    print("=" * 75)
    print(f"{'f (Seen Frac)':<12} | {'Seen Acc':<12} | {'Unseen Acc':<12} | {'Overall Acc':<12}")
    print("-" * 65)

    for f in [0.5635, 0.65, 0.70, 0.72, 0.75]:
        k_seen = int(round(f * len(val_files)))
        thr = torch.topk(combined_h, k_seen).values.min()
        route_seen = combined_h >= thr
        
        pred = torch.empty(len(val_files), dtype=torch.long, device=dev)
        pred[route_seen] = kept_h_idx[seen_block_h[route_seen].argmax(dim=1)]
        
        idx_uns = (~route_seen).nonzero(as_tuple=True)[0]
        if len(idx_uns):
            sub = text_uns_h[idx_uns]
            pred[idx_uns] = other_h_idx[sinkhorn(sub).argmax(dim=1)]
            
        names_p = [classes[i] for i in pred.tolist()]
        s_a, u_a, o_a = eval_accuracy(names_p)
        print(f"{f:<12.4f} | {s_a*100:.2f}%       | {u_a*100:.2f}%       | {o_a*100:.2f}%")

if __name__ == '__main__':
    main()
