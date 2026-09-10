"""probe2: does 'center' on fullft336shift improve the RETRIEVER quality that the
v83 seen re-ranker feeds on? Stack logic: re-ranker re-ranks fusion top-10
(build_v83_rerank_both.py L332), so stack ceiling = fusion recall@10.
If a modified fusion has better top-1 AND recall@10, the fair stack-vs-stack
test (never run in Round 3) would likely beat the deployed 91.674.
"""
import json
import pickle
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, 'research')
from common import OUT
from learned_gate_v77 import D, MEMBERS, dev, holdout_split, load_train_embs

torch.set_num_threads(8)
BASE_W = {'ctftshift': 1.0, 'ftshift': 2.5, 'fullft336shift': 2.5}
LAM = 4.0
EPS = 1e-3


def zc(M):
    return (M - M.mean()) / (M.std() + 1e-6)


def center_tr(Xraw):
    mu = Xraw.mean(0, keepdim=True)
    return lambda X: F.normalize(X - mu, dim=-1)


def build_leg(t, train, seen, s2i, trby, val_seen):
    idx, feats, _ = train[t]
    TFl, TLl = [], []
    for c in seen:
        for fn in trby[c]:
            if fn not in idx:
                continue
            TFl.append(feats[idx[fn]])
            TLl.append(s2i[c])
    return (torch.stack(TFl).to(dev), torch.tensor(TLl).to(dev),
            torch.stack([feats[idx[fn]] for fn in val_seen]).to(dev))


def member_scores(qF_raw, qF_w, P_w, TF_w, TL, Tseen, S):
    out = torch.empty(qF_raw.shape[0], S, device=dev)
    for i in range(0, qF_raw.shape[0], 2000):
        e_raw = qF_raw[i:i + 2000]
        e_w = qF_w[i:i + 2000]
        sim = e_w @ TF_w.t()
        cmax = torch.full((e_w.shape[0], S), -1e9, device=dev)
        cmax.scatter_reduce_(1, TL.unsqueeze(0).expand(e_w.shape[0], -1), sim, reduce='amax')
        out[i:i + 2000] = e_w @ P_w.t() + 2.0 * cmax + LAM * (e_raw @ Tseen.t())
    return out


def pmat(TF_w, TL, S):
    P = torch.zeros(S, TF_w.shape[1], device=dev)
    cnt = torch.zeros(S, device=dev)
    P.index_add_(0, TL, TF_w)
    cnt.index_add_(0, TL, torch.ones_like(TL, dtype=torch.float32))
    return F.normalize(P / cnt.clamp(min=1).unsqueeze(1), dim=-1)


def main():
    lab = json.load(open(f'{D}/label_train.json'))
    classes = list(pickle.load(open(f'{D}/all_classes.pkl', 'rb')))
    ci = {c: i for i, c in enumerate(classes)}
    train, tb_train, b2f_train, b2l_train = load_train_embs()
    sp = holdout_split(train, tb_train, b2f_train, b2l_train)
    trby = sp['trby']
    val_seen = [f for f, y_ in zip(sp['val_files'], sp['y']) if y_ == 1]
    seen = sorted(sp['kept'])
    s2i = {c: i for i, c in enumerate(seen)}
    S = len(seen)
    kept_idx = torch.tensor([ci[c] for c in seen]).to(dev)
    yv = torch.tensor([s2i[lab[f]] for f in val_seen])
    TtH = F.normalize(torch.load(f'{OUT}/text_emb_h_taxon.pt', weights_only=False)['emb_taxon'].float(), dim=-1).to(dev)
    Tseen = TtH[kept_idx]

    legs = {}
    for t in BASE_W:
        TF, TL, qF = build_leg(t, train, seen, s2i, trby, val_seen)
        legs[t] = {'TF': TF, 'TL': TL, 'qF': qF}

    # variant A: deployed (raw). variant B: center ONLY fullft336shift (the leg
    # where centering helped +0.46 solo). variant C: center all legs.
    variants = {'raw': set(), 'center_fullft': {'fullft336shift'}, 'center_all': set(BASE_W)}
    out = {}
    fused_scores = {}
    for vname, ctr_set in variants.items():
        per_leg = {}
        for t in BASE_W:
            TF, TL, qF = legs[t]['TF'], legs[t]['TL'], legs[t]['qF']
            if t in ctr_set:
                tr = center_tr(TF)
                qF_w, TF_w = tr(qF), tr(TF)
            else:
                qF_w, TF_w = qF, TF
            per_leg[t] = member_scores(qF, qF_w, pmat(TF_w, TL, S), TF_w, TL, Tseen, S)
        f = sum(BASE_W[t] * zc(per_leg[t]) for t in BASE_W)
        fused_scores[vname] = f
        am = f.argmax(1).cpu()
        top10 = f.topk(10, dim=1).indices.cpu()
        rec = {k: 100 * (top10[:, :k] == yv.unsqueeze(1)).any(1).float().mean().item()
               for k in (1, 5, 10)}
        out[vname] = {'top1': rec[1], 'recall5': rec[5], 'recall10': rec[10]}
        print(f'{vname:14s} top1 {rec[1]:.3f}  recall@5 {rec[5]:.3f}  recall@10 {rec[10]:.3f}', flush=True)

    # how many rows does the re-ranker's candidate set change between raw and center_fullft?
    a = fused_scores['raw'].topk(10, 1).indices.cpu()
    b = fused_scores['center_fullft'].topk(10, 1).indices.cpu()
    changed = (a != b).any(1).float().mean().item() * 100
    print(f'\ncenter_fullft vs raw: {changed:.1f}% of rows have a different top-10 candidate set '
          f'(the re-ranker re-picks within exactly this set)', flush=True)
    json.dump(out, open(f'{OUT}/probe2_retriever_recall.json', 'w'), indent=2)
    print('wrote outputs/probe2_retriever_recall.json', flush=True)


if __name__ == '__main__':
    main()
