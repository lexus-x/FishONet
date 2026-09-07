"""v55: v50 stack + max-over no-flip crop views on the ctft iNat bank + 336 iNat bank @ w=2.

Holdout (measured separately):
  crop-max all_no_flip  +0.561
  336 bank w=2          +0.777  (ftshift bank hurts — not added)
Eval crop views: outputs/emb_{test,unseen}_ctft_cropviews.pt
Single pipeline, f=0.60, tau=1.8, argmax over full 17393 classes.

  conda activate onet && python builders/build_v55_cropmax_336.py
"""
import json
import os
import pickle
import shutil
import zipfile
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
VIEW_KEYS = ['center', 'squash', 'letterbox', 'strip0', 'strip1', 'strip2']
B2_WF = 2.5
B2_WL = 2.0
W_CTFT = 4.0
W_336 = 2.0
B2_PROTO_FROZEN = 'outputs/inat_tol_merged_b2_a05.pt'
B2_PROTO_LORA = 'outputs/inat_tol_merged_b2lora_a0.5.pt'
SEEN_FRAC = 0.60
TAU = 1.8


def load(p):
    d = torch.load(p, weights_only=False)
    return {fn: i for i, fn in enumerate(d['files'])}, F.normalize(d['feats'].float(), dim=-1), list(d['files'])


def zc(M):
    return (M - M.mean()) / (M.std() + 1e-6)


def z1(v):
    return (v - v.mean()) / (v.std() + 1e-6)


def dbnorm(S, tc=0.05, tr=0.5):
    return F.log_softmax(S / tc, dim=0) + F.log_softmax(S / tr, dim=1)


def sinkhorn(logits, n_iter=SINK_ITER, tau=1.8):
    P = torch.softmax(logits / tau, dim=1)
    n, C = P.shape
    target_col = n / C
    for _ in range(n_iter):
        P = P / P.sum(dim=1, keepdim=True).clamp(min=1e-9)
        P = P * (target_col / P.sum(dim=0, keepdim=True).clamp(min=1e-9))
    return P


def load_cropviews(path):
    d = torch.load(path, weights_only=False)
    idx = {fn: i for i, fn in enumerate(d['files'])}
    views = {k: F.normalize(d['views'][k].float(), dim=-1) for k in VIEW_KEYS}
    return idx, views


def score_bank(Qenc, bank, other_list):
    Nq = Qenc.shape[0]
    Cu = len(other_list)
    Sraw = torch.full((Nq, Cu), -1e4, device=dev)
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
    return Sraw


def score_bank_maxviews(view_list, bank, other_list):
    V = len(view_list)
    Nq = view_list[0].shape[0]
    Qcat = torch.cat(view_list, dim=0)
    Cu = len(other_list)
    Sraw = torch.full((Nq, Cu), -1e4, device=dev)
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
    print('=== Building v55 crop-max + 336 iNat bank ===', flush=True)
    for p in (CROP_TEST, CROP_UNS, BANK_CTFT, BANK_336):
        if not os.path.exists(p):
            raise SystemExit(f'missing {p}')

    txtHt = torch.load('outputs/text_emb_h_taxon.pt', weights_only=False)
    txtL = torch.load('outputs/text_emb.pt', weights_only=False)
    classes = list(pickle.load(open(f'{D}/all_classes.pkl', 'rb')))
    ci = {c: i for i, c in enumerate(classes)}
    NCLS = len(classes)
    TtH = F.normalize(txtHt['emb_taxon'].float(), dim=-1).to(dev)
    TnL = F.normalize(txtL['emb_name'].float(), dim=-1).to(dev)
    _pe = torch.load('outputs/text_emb_h_promptens.pt', weights_only=False)
    TTX = F.normalize(_pe['emb_taxctx'].float(), dim=-1).to(dev)
    Ttb = F.normalize(
        torch.load('outputs/text_emb_taxabind_taxctx.pt', weights_only=False)['emb_taxctx'].float(),
        dim=-1,
    ).to(dev)
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
    print(f'seen classes: {S} | unseen classes: {len(other_idx)}')

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

    test = {t: load(f'outputs/{TRAIN[t].replace("emb_train", "emb_test")}.pt') for t, _, _ in MEMBERS}
    unseen = {t: load(f'outputs/{TRAIN[t].replace("emb_train", "emb_unseen")}.pt') for t, _, _ in MEMBERS}
    tb_test = load('outputs/emb_test_taxabind.pt')
    tb_uns = load('outputs/emb_unseen_taxabind.pt')
    tf = sorted(set.intersection(*[set(f) for (_, _, f) in test.values()], set(tb_test[2])))
    uf = sorted(set.intersection(*[set(f) for (_, _, f) in unseen.values()], set(tb_uns[2])))
    all_files = tf + uf
    print(f'Combined evaluation batch: {len(all_files)} images')

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
                 + 1.0 * dbnorm(Q['ctftshift'] @ TTX.t())
                 + TAXABIND_W * dbnorm(Qtb @ Ttb.t()))

    ct_test = load_cropviews(CROP_TEST)
    ct_uns = load_cropviews(CROP_UNS)
    fallback = Q['ctftshift'].cpu()
    view_tensors = []
    n_fb = 0
    for k in VIEW_KEYS:
        rows = []
        for i, fn in enumerate(all_files):
            if fn in ct_test[0]:
                rows.append(ct_test[1][k][ct_test[0][fn]])
            elif fn in ct_uns[0]:
                rows.append(ct_uns[1][k][ct_uns[0][fn]])
            else:
                n_fb += 1
                rows.append(fallback[i])
        view_tensors.append(F.normalize(torch.stack(rows).float(), dim=-1).to(dev))
    print(f'crop views aligned n={len(all_files)} fallback_slots={n_fb}', flush=True)

    other_list = other_idx.tolist()
    Nq = len(all_files)
    Cu = len(other_list)
    print(f'scoring ctft crop-max bank Cu={Cu} views={len(VIEW_KEYS)} ...', flush=True)
    Sraw_ct = score_bank_maxviews(view_tensors, torch.load(BANK_CTFT, weights_only=False)['bank'], other_list)
    print('scoring 336 bank ...', flush=True)
    Sraw_336 = score_bank(Q['fullft336shift'], torch.load(BANK_336, weights_only=False)['bank'], other_list)

    def b2_proto_leg(proto_path, emb_test_path, emb_uns_path):
        b2P = F.normalize(torch.load(proto_path, weights_only=False)['protos'].float(), dim=-1).to(dev)
        b2_test = load(emb_test_path)
        b2_uns = load(emb_uns_path)
        Qb2 = torch.cat([
            torch.stack([b2_test[1][b2_test[0][fn]] for fn in tf]),
            torch.stack([b2_uns[1][b2_uns[0][fn]] for fn in uf]),
        ]).to(dev)
        Sraw_b = torch.full((Nq, Cu), -1e4, device=dev)
        for j, gidx in enumerate(other_list):
            pvec = b2P[gidx]
            if pvec.norm() < 0.5:
                continue
            Sraw_b[:, j] = Qb2 @ pvec
        return dbnorm(Sraw_b)

    b2_frozen_leg = b2_proto_leg(B2_PROTO_FROZEN, 'outputs/emb_test_bioclip2.pt', 'outputs/emb_unseen_bioclip2.pt')
    b2_lora_leg = b2_proto_leg(B2_PROTO_LORA, 'outputs/emb_test_bioclip2_lora_v2.pt', 'outputs/emb_unseen_bioclip2_lora_v2.pt')
    img_leg = W_CTFT * dbnorm(Sraw_ct) + W_336 * dbnorm(Sraw_336)
    text_unseen_only = text_full[:, other_idx] + img_leg + B2_WF * b2_frozen_leg + B2_WL * b2_lora_leg

    img_seenmax = seen_block.max(1).values
    text_margin = ((Q['ctftshift'] @ TtH[kept_idx].t()).max(1).values
                   - (Q['ctftshift'] @ TtH[other_idx].t()).max(1).values)
    combined = z1(img_seenmax) + 2.0 * z1(text_margin)

    os.makedirs('submissions', exist_ok=True)
    seen_set = set(seen)
    k_seen = int(round(SEEN_FRAC * len(all_files)))
    thr = torch.topk(combined, k_seen).values.min()
    route_seen = combined >= thr

    pred_idx = torch.empty(len(all_files), dtype=torch.long, device=dev)
    pred_idx[route_seen] = kept_idx[seen_block[route_seen].argmax(1)]
    idx_uns = (~route_seen).nonzero(as_tuple=True)[0]
    pred_idx[idx_uns] = other_idx[sinkhorn(text_unseen_only[idx_uns], tau=TAU).argmax(1)]

    preds = {fn: classes[j] for fn, j in zip(all_files, pred_idx.tolist())}
    assert len(preds) == 35665 and all(c in ci for c in preds.values())

    cov = len(set(p for p in preds.values() if p not in seen_set))
    o_t = sum(1 for fn in tf if preds[fn] not in seen_set)
    o_u = sum(1 for fn in uf if preds[fn] not in seen_set)

    tag = 'v55_cropmax_336_f60_tau18'
    json_path = f'outputs/prediction_{tag}.json'
    zip_path = f'outputs/submission_{tag}.zip'
    sub_zip_path = f'submissions/submission_{tag}.zip'
    json.dump(preds, open(json_path, 'w'))
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
        z.write(json_path, arcname='prediction.json')
    shutil.copy(zip_path, sub_zip_path)

    n_diff = None
    v50 = 'outputs/prediction_v50_f60_tau18.json'
    if os.path.exists(v50):
        p50 = json.load(open(v50))
        n_diff = sum(1 for fn in all_files if preds[fn] != p50.get(fn))

    print(f'Tag: {tag}')
    print(f'  Total predictions: {len(preds)} (Coverage: {cov}/{len(other_idx)} unseen classes)')
    print(f'  Test folder -> seen: {100 * (1 - o_t / len(tf)):.1f}% | Unseen folder -> unseen: {100 * o_u / len(uf):.1f}%')
    if n_diff is not None:
        print(f'  Diff vs v50_f60: {n_diff}/{len(all_files)} ({100 * n_diff / len(all_files):.2f}%)')
    print(f'  Zip created: {sub_zip_path}')
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
