"""Re-test recover/soft routing with v36 scores (higher b may flip the trade).

v34 failed with weaker b. Now b_cond≈19.5% — soft/recover may be +EV.
Holdout proxy + eval diagnostic uf routing. Build zip only if holdout clears +0.5 vs hard f72.
"""
from __future__ import annotations

import json
import os
import pickle
import shutil
import zipfile
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
    print(f'eval {len(all_files)}', flush=True)

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
    text_uns = text_full[:, other_idx]
    img_max = seen_block.max(1).values
    text_margin = ((Q['ctftshift'] @ TtH[kept_idx].t()).max(1).values
                   - (Q['ctftshift'] @ TtH[other_idx].t()).max(1).values)
    combined = z1(img_max) + 2.0 * z1(text_margin)

    y = torch.tensor([0] * len(tf) + [1] * len(uf))  # 1=uf
    SEEN_FRAC = 0.72
    k_seen = int(round(SEEN_FRAC * len(all_files)))
    thr = torch.topk(combined, k_seen).values.min()
    route_seen = combined >= thr
    idx_uns = (~route_seen).nonzero(as_tuple=True)[0]

    def stats(pred, name):
        seen_set_t = torch.zeros(NCLS, dtype=torch.bool)
        seen_set_t[kept_idx.cpu()] = True
        ps = seen_set_t[pred.cpu()]
        tf_s = ps[:len(tf)].float().mean().item()
        uf_u = (~ps[len(tf):]).float().mean().item()
        print(f'{name:28s} tf→seen {100*tf_s:.1f}% uf→uns {100*uf_u:.1f}%', flush=True)
        return tf_s, uf_u

    variants = {}
    # hard v36
    pred = torch.empty(len(all_files), dtype=torch.long, device=dev)
    pred[route_seen] = kept_idx[seen_block[route_seen].argmax(1)]
    pred[idx_uns] = other_idx[sinkhorn(text_uns[idx_uns]).argmax(1)]
    variants['hard_v36'] = pred.clone()
    stats(pred, 'hard_v36')

    # recover text_full on eject
    pred = torch.empty(len(all_files), dtype=torch.long, device=dev)
    pred[route_seen] = kept_idx[seen_block[route_seen].argmax(1)]
    pred[idx_uns] = text_full[idx_uns].argmax(1)
    variants['recover_textfull'] = pred.clone()
    stats(pred, 'recover_textfull')

    # soft calibrated full
    def row_z(M):
        return (M - M.mean(1, keepdim=True)) / (M.std(1, keepdim=True) + 1e-6)

    sb = zc(seen_block)
    tu = row_z(text_uns)
    for beta in [1.0, 1.05, 1.1, 1.15]:
        full = torch.full((len(all_files), NCLS), -1e9, device=dev)
        full[:, kept_idx] = sb
        full[:, other_idx] = beta * tu
        pred = full.argmax(1)
        variants[f'soft_b{beta}'] = pred
        stats(pred, f'soft_b{beta}')

    # soft text_full only
    pred = text_full.argmax(1)
    variants['soft_text_full'] = pred
    stats(pred, 'soft_text_full')

    # ---- Holdout zero-shot ----
    print('\n=== HOLDOUT ===', flush=True)
    order = sorted(seen, key=lambda c: len(by[c]))
    n_pseudo = int(len(seen) * 0.2)
    pseudo = set(order[:n_pseudo])
    kept_h = sorted(order[n_pseudo:])
    k2i = {c: i for i, c in enumerate(kept_h)}
    Sk = len(kept_h)
    kept_h_idx = torch.tensor([ci[c] for c in kept_h], device=dev)
    other_h_idx = torch.tensor([i for i in range(NCLS) if classes[i] not in set(kept_h)], device=dev)
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
    val_all = val_seen + val_uns
    val_files = [f for f, _ in val_all]
    gold = [c for _, c in val_all]
    is_uns = torch.tensor([0] * len(val_seen) + [1] * len(val_uns))
    print(f'holdout seen={len(val_seen)} uns={len(val_uns)}', flush=True)

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
    Tkept = TtH[kept_h_idx]

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
                sc = sc + LAM * (e @ Tkept.t())
            out[i:i + 2000] = sc
        return out

    Qh = {t: torch.stack([train[t][1][train[t][0][fn]] for fn in val_files]).to(dev) for t, _, _ in MEMBERS}
    # taxabind: encode on the fly is slow — skip TB on holdout or use approx without TB
    # For fair compare use same stack WITHOUT taxabind if no train emb — use ctftshift stack only
    sb_h = torch.zeros(len(val_files), Sk, device=dev)
    for t, hs, w in MEMBERS:
        if w == 0:
            continue
        sb_h = sb_h + w * zc(seen_score_h(Qh[t], *PR_h[t], hs))
    text_h = (dbnorm(Qh['ctftshift'] @ TtH.t()) + 0.5 * dbnorm(Qh['L'] @ TnL.t())
              + 0.75 * dbnorm(Qh['fullft336_v2'] @ TtH.t()) + 1.0 * dbnorm(Qh['ftshift'] @ TtH.t())
              + 1.0 * dbnorm(Qh['ctftshift'] @ TTX.t()))
    text_uns_h = text_h[:, other_h_idx]
    img_h = sb_h.max(1).values
    tm_h = ((Qh['ctftshift'] @ TtH[kept_h_idx].t()).max(1).values
            - (Qh['ctftshift'] @ TtH[other_h_idx].t()).max(1).values)
    comb_h = z1(img_h) + 2 * z1(tm_h)
    k_seen = int(round(0.72 * len(val_files)))
    thr = torch.topk(comb_h, k_seen).values.min()
    rs = comb_h >= thr
    iu = (~rs).nonzero(as_tuple=True)[0]

    def report(name, pred):
        names = [classes[i] for i in pred.tolist()]
        ok = [p == g for p, g in zip(names, gold)]
        s = sum(o for o, m in zip(ok, (is_uns == 0).tolist()) if m) / max(1, (is_uns == 0).sum().item())
        u = sum(o for o, m in zip(ok, (is_uns == 1).tolist()) if m) / max(1, (is_uns == 1).sum().item())
        o = 0.5635 * s + 0.4365 * u
        print(f'{name:28s} s={100*s:.2f}% u={100*u:.2f}% o={100*o:.2f}%', flush=True)
        return o

    pred = torch.empty(len(val_files), dtype=torch.long, device=dev)
    pred[rs] = kept_h_idx[sb_h[rs].argmax(1)]
    pred[iu] = other_h_idx[sinkhorn(text_uns_h[iu]).argmax(1)]
    base = report('hard_f72', pred)

    pred = torch.empty(len(val_files), dtype=torch.long, device=dev)
    pred[rs] = kept_h_idx[sb_h[rs].argmax(1)]
    pred[iu] = text_h[iu].argmax(1)
    o_rec = report('recover_textfull', pred)

    sb = zc(sb_h)
    tu = row_z(text_uns_h)
    best_soft = base
    best_name = 'hard_f72'
    for beta in [1.0, 1.05, 1.1]:
        full = torch.full((len(val_files), NCLS), -1e9, device=dev)
        full[:, kept_h_idx] = sb
        full[:, other_h_idx] = beta * tu
        o = report(f'soft_b{beta}', full.argmax(1))
        if o > best_soft:
            best_soft, best_name = o, f'soft_b{beta}'

    o_txt = report('soft_text_full', text_h.argmax(1))
    if o_txt > best_soft:
        best_soft, best_name = o_txt, 'soft_text_full'

    delta = 100 * (best_soft - base)
    print(f'\nbest={best_name} Δ={delta:+.2f}pt vs hard', flush=True)
    json.dump({'base': base, 'best': best_name, 'best_o': best_soft, 'delta_pt': delta},
              open(f'{OUT}/v36_recover_retest_results.json', 'w'), indent=1)

    if delta >= 0.5 and best_name == 'recover_textfull':
        print('building v37 recover zip', flush=True)
        pred = variants['recover_textfull']
        preds = {fn: classes[j] for fn, j in zip(all_files, pred.tolist())}
        tag = 'v37_recover_v36base'
        json.dump(preds, open(f'{OUT}/prediction_{tag}.json', 'w'))
        z = zipfile.ZipFile(f'{OUT}/submission_{tag}.zip', 'w', zipfile.ZIP_DEFLATED)
        z.write(f'{OUT}/prediction_{tag}.json', arcname='prediction.json')
        z.close()
        shutil.copy(f'{OUT}/submission_{tag}.zip', f'submissions/submission_{tag}.zip')
        print('wrote', tag, flush=True)


if __name__ == '__main__':
    main()
