"""Replace dbnorm's across-image (test-batch) term with frozen per-class offsets.

dbnorm(S)[i,c] = log_softmax(S/tc, dim=images)[i,c] + log_softmax(S/tr, dim=classes)[i,c]
               = S[i,c]/tc - logsumexp_j(S[j,c]/tc) + log_softmax_c(S[i,:]/tr)
                             ^^^^^^^^^^^^^^^^^^^^^ the only test-batch coupling here

We estimate that per-class hub offset h_c on a REFERENCE pool of images that is
label-disjoint from the 11,598 novel classes, freeze it, and rebuild v56 with
everything else byte-identical. Then diff the predictions against the transductive
build. No labels needed.

  conda activate onet && python research/dbnorm_frozen_hub.py pseudo
  conda activate onet && python research/dbnorm_frozen_hub.py kept
"""
import json
import os
import pickle
import sys
from collections import defaultdict

import torch
import torch.nn.functional as F

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_num_threads(8)
D = 'data/dl'
SINK_ITER = 60
TAXABIND_W = 1.0
TOPM = 4
BANK_CTFT = 'outputs/inat_photo_bank_ctftshift.pt'
BANK_336 = 'outputs/inat_photo_bank_fullft336shift.pt'
CROP_TEST = 'outputs/emb_test_ctft_cropviews.pt'
CROP_UNS = 'outputs/emb_unseen_ctft_cropviews.pt'
OSTRIP_TEST = 'outputs/emb_test_ctft_ostrip.pt'
OSTRIP_UNS = 'outputs/emb_unseen_ctft_ostrip.pt'
HOLD_VIEWS = 'outputs/emb_holdout_ctft_cropviews_v2.pt'
VIEW_KEYS = ['center', 'squash', 'ostrip0', 'ostrip1', 'ostrip2', 'ostrip3', 'ostrip4']
B2_WF = 2.5
B2_WL = 2.0
W_CTFT = 4.0
W_336 = 3.0
B2_PROTO_FROZEN = 'outputs/inat_tol_merged_b2_a05.pt'
B2_PROTO_LORA = 'outputs/inat_tol_merged_b2lora_a0.5.pt'
SEEN_FRAC = 0.60
TAU = 1.8
REF_N_KEPT = 4000
TC, TR = 0.05, 0.5


def load(p):
    d = torch.load(p, weights_only=False)
    return {fn: i for i, fn in enumerate(d['files'])}, F.normalize(d['feats'].float(), dim=-1), list(d['files'])


def zc(M):
    return (M - M.mean()) / (M.std() + 1e-6)


def z1(v):
    return (v - v.mean()) / (v.std() + 1e-6)


def hub_offset(Sref):
    """Per-class across-image log-partition. Left uncentred: any class-uniform part of
    it (e.g. log of the pool size) cancels in both the argmax and Sinkhorn."""
    return torch.logsumexp(Sref / TC, dim=0)


def dbnorm_frozen(S, h):
    return S / TC - h.unsqueeze(0) + F.log_softmax(S / TR, dim=1)


def dbnorm(S):
    return F.log_softmax(S / TC, dim=0) + F.log_softmax(S / TR, dim=1)


def sinkhorn(logits, n_iter=SINK_ITER, tau=TAU):
    P = torch.softmax(logits / tau, dim=1)
    n, C = P.shape
    target_col = n / C
    for _ in range(n_iter):
        P = P / P.sum(dim=1, keepdim=True).clamp(min=1e-9)
        P = P * (target_col / P.sum(dim=0, keepdim=True).clamp(min=1e-9))
    return P


def load_cropviews(path, keys):
    d = torch.load(path, weights_only=False)
    idx = {fn: i for i, fn in enumerate(d['files'])}
    views = {k: F.normalize(d['views'][k].float(), dim=-1) for k in keys if k in d['views']}
    return idx, views


def score_bank(Qenc, bank, other_list):
    Sraw = torch.full((Qenc.shape[0], len(other_list)), -1e4, device=dev)
    for j, gidx in enumerate(other_list):
        photos = bank.get(gidx)
        if photos is None or (hasattr(photos, 'numel') and photos.numel() == 0):
            continue
        if not isinstance(photos, torch.Tensor):
            photos = torch.stack(photos)
        photos = F.normalize(photos.float(), dim=-1).to(dev)
        sim = Qenc @ photos.t()
        Sraw[:, j] = sim.topk(min(TOPM, sim.shape[1]), dim=1).values.mean(dim=1)
    return Sraw


def score_bank_maxviews(view_list, bank, other_list):
    V = len(view_list)
    Nq = view_list[0].shape[0]
    Qcat = torch.cat(view_list, dim=0)
    Sraw = torch.full((Nq, len(other_list)), -1e4, device=dev)
    for j, gidx in enumerate(other_list):
        photos = bank.get(gidx)
        if photos is None or (hasattr(photos, 'numel') and photos.numel() == 0):
            continue
        if not isinstance(photos, torch.Tensor):
            photos = torch.stack(photos)
        photos = F.normalize(photos.float(), dim=-1).to(dev)
        pooled = (Qcat @ photos.t()).topk(min(TOPM, photos.shape[0]), dim=1).values.mean(dim=1)
        Sraw[:, j] = pooled.view(V, Nq).max(0).values
    return Sraw


def main():
    pool_name = sys.argv[1] if len(sys.argv) > 1 else 'pseudo'
    assert pool_name in ('pseudo', 'pseudo_a', 'pseudo_b', 'kept')
    print(f'=== frozen hub offsets | reference pool = {pool_name} ===', flush=True)

    txtHt = torch.load('outputs/text_emb_h_taxon.pt', weights_only=False)
    txtL = torch.load('outputs/text_emb.pt', weights_only=False)
    classes = list(pickle.load(open(f'{D}/all_classes.pkl', 'rb')))
    ci = {c: i for i, c in enumerate(classes)}
    NCLS = len(classes)
    TtH = F.normalize(txtHt['emb_taxon'].float(), dim=-1).to(dev)
    TnL = F.normalize(txtL['emb_name'].float(), dim=-1).to(dev)
    TTX = F.normalize(torch.load('outputs/text_emb_h_promptens.pt', weights_only=False)['emb_taxctx'].float(), dim=-1).to(dev)
    Ttb = F.normalize(torch.load('outputs/text_emb_taxabind_taxctx.pt', weights_only=False)['emb_taxctx'].float(), dim=-1).to(dev)
    lab = json.load(open(f'{D}/label_train.json'))

    MEMBERS = [('ctftshift', True, 1.0), ('ftshift', True, 2.5), ('fullft336shift', True, 2.5),
               ('L', False, 0.0), ('fullft336_v2', True, 0.0)]
    TRAIN = {'ctftshift': 'emb_train_ctftshift', 'ftshift': 'emb_train_ftshift',
             'fullft336shift': 'emb_train_fullft336shift', 'L': 'emb_train',
             'fullft336_v2': 'emb_train_fullft336_v2'}
    train = {t: load(f'outputs/{TRAIN[t]}.pt') for t, _, _ in MEMBERS}
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
    kept_idx = torch.tensor([ci[c] for c in seen]).to(dev)
    other_idx = torch.tensor([i for i in range(NCLS) if classes[i] not in set(seen)]).to(dev)
    TseenTax = TtH[kept_idx]
    LAM = 4.0
    print(f'seen classes: {S} | unseen classes: {len(other_idx)}', flush=True)

    # ---- reference pool -------------------------------------------------
    hold = torch.load(HOLD_VIEWS, weights_only=False)
    pseudo_files = [fn for fn in hold['files'] if fn in set(common)]
    pseudo_cls = set(lab[fn] for fn in pseudo_files)
    if pool_name.startswith('pseudo'):
        if pool_name == 'pseudo_a':
            ref_files = pseudo_files[0::2]
        elif pool_name == 'pseudo_b':
            ref_files = pseudo_files[1::2]
        else:
            ref_files = pseudo_files
        hidx = {fn: i for i, fn in enumerate(hold['files'])}
        ref_views = [F.normalize(hold['views'][k].float(), dim=-1)[[hidx[fn] for fn in ref_files]].to(dev)
                     for k in VIEW_KEYS]
    else:
        kept_pool = sorted(fn for fn in common if lab[fn] not in pseudo_cls)
        stride = max(1, len(kept_pool) // REF_N_KEPT)
        ref_files = kept_pool[::stride][:REF_N_KEPT]
        ctr = torch.stack([train['ctftshift'][1][train['ctftshift'][0][fn]] for fn in ref_files]).to(dev)
        ref_views = [ctr]  # only the centre view exists for arbitrary training images
    print(f'reference images: {len(ref_files)} (classes disjoint from the {len(other_idx)} novel)', flush=True)

    def ref_emb(tag, path=None):
        if path is None:
            idx, feats, _ = train[tag]
        else:
            idx, feats, _ = load(path)
        return torch.stack([feats[idx[fn]] for fn in ref_files]).to(dev)

    R = {t: ref_emb(t) for t, _, _ in MEMBERS}
    Rtb = ref_emb(None, 'outputs/emb_train_taxabind.pt')

    # ---- frozen offsets, one per dbnorm leg ------------------------------
    other_list = other_idx.tolist()
    H = {}
    H['t_ctft_TtH'] = hub_offset(R['ctftshift'] @ TtH.t())
    H['t_L_TnL'] = hub_offset(R['L'] @ TnL.t())
    H['t_336v2_TtH'] = hub_offset(R['fullft336_v2'] @ TtH.t())
    H['t_ftshift_TtH'] = hub_offset(R['ftshift'] @ TtH.t())
    H['t_ctft_TTX'] = hub_offset(R['ctftshift'] @ TTX.t())
    H['t_taxabind'] = hub_offset(Rtb @ Ttb.t())
    print('text-leg offsets done', flush=True)

    bank_ct = torch.load(BANK_CTFT, weights_only=False)['bank']
    H['img_ct'] = hub_offset(score_bank_maxviews(ref_views, bank_ct, other_list))
    print('ctft bank offsets done', flush=True)
    bank_336 = torch.load(BANK_336, weights_only=False)['bank']
    H['img_336'] = hub_offset(score_bank(R['fullft336shift'], bank_336, other_list))
    print('336 bank offsets done', flush=True)

    def b2_ref(proto_path, train_path):
        b2P = F.normalize(torch.load(proto_path, weights_only=False)['protos'].float(), dim=-1).to(dev)
        Qr = ref_emb(None, train_path)
        Sr = torch.full((Qr.shape[0], len(other_list)), -1e4, device=dev)
        for j, gidx in enumerate(other_list):
            pvec = b2P[gidx]
            if pvec.norm() >= 0.5:
                Sr[:, j] = Qr @ pvec
        return hub_offset(Sr), b2P

    H['b2_frozen'], B2P_F = b2_ref(B2_PROTO_FROZEN, 'outputs/emb_train_bioclip2.pt')
    H['b2_lora'], B2P_L = b2_ref(B2_PROTO_LORA, 'outputs/emb_train_bioclip2_lora_v2.pt')
    print('b2 proto offsets done', flush=True)
    del R, Rtb

    # ---- rebuild v56, identical except for the frozen offsets -------------
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
        return P.to(dev), torch.stack(TF).to(dev), torch.tensor(TL).to(dev)

    PR = {t: protos(*train[t][:2]) for t, _, _ in MEMBERS}

    def seen_score(qF, P, TF, TL, hspace):
        qF = qF.to(dev)
        out = torch.empty(qF.shape[0], S, device=dev)
        for i in range(0, qF.shape[0], 2000):
            e = qF[i:i + 2000]
            sim = e @ TF.t()
            cmax = torch.full((e.shape[0], S), -1e9, device=dev)
            cmax.scatter_reduce_(1, TL.unsqueeze(0).expand(e.shape[0], -1), sim, reduce='amax')
            sc = e @ P.t() + 2.0 * cmax
            if hspace:
                sc = sc + LAM * (e @ TseenTax.t())
            out[i:i + 2000] = sc
        return out

    test = {t: load(f'outputs/{TRAIN[t].replace("emb_train", "emb_test")}.pt') for t, _, _ in MEMBERS}
    unseen = {t: load(f'outputs/{TRAIN[t].replace("emb_train", "emb_unseen")}.pt') for t, _, _ in MEMBERS}
    tb_test = load('outputs/emb_test_taxabind.pt')
    tb_uns = load('outputs/emb_unseen_taxabind.pt')
    tf = sorted(set.intersection(*[set(f) for (_, _, f) in test.values()], set(tb_test[2])))
    uf = sorted(set.intersection(*[set(f) for (_, _, f) in unseen.values()], set(tb_uns[2])))
    all_files = tf + uf
    print(f'evaluation batch: {len(all_files)}', flush=True)

    def qcat(t):
        ti, tfeat, _ = test[t]
        ui, ufeat, _ = unseen[t]
        return torch.cat([torch.stack([tfeat[ti[fn]] for fn in tf]),
                          torch.stack([ufeat[ui[fn]] for fn in uf])])

    Q = {t: qcat(t).to(dev) for t, _, _ in MEMBERS}
    Qtb = torch.cat([torch.stack([tb_test[1][tb_test[0][fn]] for fn in tf]),
                     torch.stack([tb_uns[1][tb_uns[0][fn]] for fn in uf])]).to(dev)

    seen_block = torch.zeros(len(all_files), S, device=dev)
    for t, hs, w in MEMBERS:
        if w:
            seen_block = seen_block + w * zc(seen_score(Q[t], *PR[t], hs))

    packs = [load_cropviews(CROP_TEST, ['center', 'squash']),
             load_cropviews(CROP_UNS, ['center', 'squash']),
             load_cropviews(OSTRIP_TEST, [f'ostrip{i}' for i in range(5)]),
             load_cropviews(OSTRIP_UNS, [f'ostrip{i}' for i in range(5)])]
    fallback = Q['ctftshift'].cpu()
    view_tensors = []
    for k in VIEW_KEYS:
        rows = []
        for i, fn in enumerate(all_files):
            hit = None
            for idx, views in packs:
                if k in views and fn in idx:
                    hit = views[k][idx[fn]]
                    break
            rows.append(fallback[i] if hit is None else hit)
        view_tensors.append(F.normalize(torch.stack(rows).float(), dim=-1).to(dev))

    Sraw_ct = score_bank_maxviews(view_tensors, bank_ct, other_list)
    Sraw_336 = score_bank(Q['fullft336shift'], bank_336, other_list)

    def b2_leg(B2P, emb_test_path, emb_uns_path):
        b2_test, b2_uns = load(emb_test_path), load(emb_uns_path)
        Qb2 = torch.cat([torch.stack([b2_test[1][b2_test[0][fn]] for fn in tf]),
                         torch.stack([b2_uns[1][b2_uns[0][fn]] for fn in uf])]).to(dev)
        Sr = torch.full((len(all_files), len(other_list)), -1e4, device=dev)
        for j, gidx in enumerate(other_list):
            pvec = B2P[gidx]
            if pvec.norm() >= 0.5:
                Sr[:, j] = Qb2 @ pvec
        return Sr

    Sb2f = b2_leg(B2P_F, 'outputs/emb_test_bioclip2.pt', 'outputs/emb_unseen_bioclip2.pt')
    Sb2l = b2_leg(B2P_L, 'outputs/emb_test_bioclip2_lora_v2.pt', 'outputs/emb_unseen_bioclip2_lora_v2.pt')

    # recomputed per mode rather than cached: each is [35665, 17393] float32 = 2.5 GB
    TEXT_LEGS = [('t_ctft_TtH', 1.0, lambda: Q['ctftshift'] @ TtH.t()),
                 ('t_L_TnL', 0.5, lambda: Q['L'] @ TnL.t()),
                 ('t_336v2_TtH', 0.75, lambda: Q['fullft336_v2'] @ TtH.t()),
                 ('t_ftshift_TtH', 1.0, lambda: Q['ftshift'] @ TtH.t()),
                 ('t_ctft_TTX', 1.0, lambda: Q['ctftshift'] @ TTX.t()),
                 ('t_taxabind', TAXABIND_W, lambda: Qtb @ Ttb.t())]

    # do the offsets themselves agree, or only the predictions they produce?
    Hb = {name: hub_offset(fn_raw()) for name, _, fn_raw in TEXT_LEGS}
    Hb.update({'img_ct': hub_offset(Sraw_ct), 'img_336': hub_offset(Sraw_336),
               'b2_frozen': hub_offset(Sb2f), 'b2_lora': hub_offset(Sb2l)})
    torch.save({'frozen': {k: v.cpu() for k, v in H.items()},
                'batch': {k: v.cpu() for k, v in Hb.items()}},
               f'outputs/hub_offsets_{pool_name}.pt')
    agree = {}
    for k in H:
        a, b = H[k].double(), Hb[k].double()
        m = (a > -1e4) & (b > -1e4)          # drop novel classes with no bank photographs
        a, b = a[m], b[m]
        ra = a.argsort().argsort().double()
        rb = b.argsort().argsort().double()
        pear = torch.corrcoef(torch.stack([a - a.mean(), b - b.mean()]))[0, 1].item()
        spear = torch.corrcoef(torch.stack([ra, rb]))[0, 1].item()
        agree[k] = {'n_classes': int(m.sum()), 'pearson': round(pear, 4), 'spearman': round(spear, 4)}
    print('offset agreement (reference pool vs eval batch):', flush=True)
    for k, v in agree.items():
        print(f"  {k:14s} n={v['n_classes']:6d}  r={v['pearson']:.4f}  rho={v['spearman']:.4f}", flush=True)

    img_seenmax = seen_block.max(1).values
    text_margin = ((Q['ctftshift'] @ TtH[kept_idx].t()).max(1).values
                   - (Q['ctftshift'] @ TtH[other_idx].t()).max(1).values)
    combined = z1(img_seenmax) + 2.0 * z1(text_margin)
    k_seen = int(round(SEEN_FRAC * len(all_files)))
    route_seen = combined >= torch.topk(combined, k_seen).values.min()
    idx_uns = (~route_seen).nonzero(as_tuple=True)[0]

    def build(mode):
        def nrm(name, Sm):
            return dbnorm(Sm) if mode == 'batch' else dbnorm_frozen(Sm, H[name])

        tot = None
        for name, w, fn_raw in TEXT_LEGS:
            part = (w * nrm(name, fn_raw()))[:, other_idx]
            tot = part if tot is None else tot + part
            del part
            torch.cuda.empty_cache()
        for name, w, Sm in [('img_ct', W_CTFT, Sraw_ct), ('img_336', W_336, Sraw_336),
                            ('b2_frozen', B2_WF, Sb2f), ('b2_lora', B2_WL, Sb2l)]:
            tot = tot + w * nrm(name, Sm)
        pred = torch.empty(len(all_files), dtype=torch.long, device=dev)
        pred[route_seen] = kept_idx[seen_block[route_seen].argmax(1)]
        pred[idx_uns] = other_idx[sinkhorn(tot[idx_uns]).argmax(1)]
        return {fn: classes[j] for fn, j in zip(all_files, pred.tolist())}

    p_batch, p_frozen = build('batch'), build('frozen')
    ref_pred = json.load(open('outputs/prediction_v56_overlap_336_f60_tau18.json'))
    n_eject = len(idx_uns)
    d_repro = sum(1 for fn in all_files if p_batch[fn] != ref_pred[fn])
    d_frozen = sum(1 for fn in all_files if p_frozen[fn] != ref_pred[fn])
    cov_b = len(set(p_batch.values()) - set(seen))
    cov_f = len(set(p_frozen.values()) - set(seen))
    # the harness must reproduce the shipped build before the swap, or nothing below means anything
    assert d_repro == 0, f'harness does not reproduce v56: {d_repro} predictions differ'

    out = {'pool': pool_name, 'n_ref': len(ref_files), 'n_eval': len(all_files),
           'n_ejected': int(n_eject), 'diff_reproduction_vs_v56': d_repro,
           'diff_frozen_vs_v56': d_frozen,
           'changed_share_of_ejected_pct': round(100 * d_frozen / n_eject, 3),
           'changed_share_of_all_pct': round(100 * d_frozen / len(all_files), 3),
           'novel_coverage_batch': cov_b, 'novel_coverage_frozen': cov_f,
           'offset_agreement_vs_batch': agree}
    print(json.dumps(out, indent=2), flush=True)
    path = f'outputs/dbnorm_frozen_hub_{pool_name}.json'
    json.dump(out, open(path, 'w'), indent=2)
    json.dump(p_frozen, open(f'outputs/prediction_v56_frozenhub_{pool_name}.json', 'w'))
    print('wrote', path, flush=True)


if __name__ == '__main__':
    main()
