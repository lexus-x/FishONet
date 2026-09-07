"""Add a 3rd orthogonal iNat proto leg on top of v39 (S0 + wc*ctft + wt*tb + wn*new)."""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import FishData, dbnorm, load_emb, OUT  # noqa: E402


def top1(S, gold):
    return (S.argmax(1) == gold).float().mean().item() * 100


def proto_leg(Q, proto_path, cand):
    P = F.normalize(torch.load(proto_path, weights_only=False)['protos'].float(), dim=-1)[cand]
    has = P.norm(dim=-1) > 0.5
    S = Q @ P.t()
    S = S.masked_fill(~has.unsqueeze(0), -1e4)
    return dbnorm(S), has


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--proto_new', required=True)
    ap.add_argument('--qenc_new', required=True, help='emb_train_*.pt path')
    ap.add_argument('--tag', default='new')
    ap.add_argument('--wc', type=float, default=3.0)
    ap.add_argument('--wt', type=float, default=0.5)
    a = ap.parse_args()

    D = FishData()
    cand = D.cand

    def queries(idx, feats):
        q, y = [], []
        for c in D.pseudo:
            for fn in D.by[c]:
                if fn in idx:
                    q.append(feats[idx[fn]])
                    y.append(D.cand_pos[D.ci[c]])
        return torch.stack(q), torch.tensor(y)

    TtH_c = D.TtH[cand]
    TnL_c = D.TnL[cand]
    TTX = F.normalize(
        torch.load(os.path.join(OUT, 'text_emb_h_promptens.pt'), weights_only=False)['emb_taxctx'].float(),
        dim=-1,
    )
    TTX_c = TTX[cand]
    legs = {}
    for tag in ['ctftshift', 'ftshift', 'fullft336_v2']:
        idx, feats, _ = load_emb(os.path.join(OUT, f'emb_train_{tag}.pt'))
        q, gold = queries(idx, feats)
        legs[tag] = q
    qL, gold = queries(D.LtI, D.LtF)
    S0 = (
        dbnorm(legs['ctftshift'] @ TtH_c.t())
        + 0.5 * dbnorm(qL @ TnL_c.t())
        + 0.75 * dbnorm(legs['fullft336_v2'] @ TtH_c.t())
        + 1.0 * dbnorm(legs['ftshift'] @ TtH_c.t())
        + 1.0 * dbnorm(legs['ctftshift'] @ TTX_c.t())
    )
    S_ctft, has_c = proto_leg(legs['ctftshift'], os.path.join(OUT, 'inat_protos_ctftshift_full.pt'), cand)
    tb_idx, tb_feat, _ = load_emb(os.path.join(OUT, 'emb_train_taxabind.pt'))
    Qtb, _ = queries(tb_idx, tb_feat)
    S_tb, _ = proto_leg(Qtb, os.path.join(OUT, 'inat_protos_taxabind.pt'), cand)
    v39 = S0 + a.wc * S_ctft + a.wt * S_tb
    ref = top1(v39, gold)
    print(f'v39 ref wc={a.wc} wt={a.wt}: {ref:.2f}', flush=True)

    n_idx, n_feat, _ = load_emb(a.qenc_new)
    Qn, _ = queries(n_idx, n_feat)
    S_n, has_n = proto_leg(Qn, a.proto_new, cand)

    best, best_w = ref, 0.0
    for w in [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0]:
        acc = top1(v39 + w * S_n, gold)
        print(f'  +{w:g}*{a.tag}: {acc:.2f} ({acc-ref:+.2f})', flush=True)
        if acc > best:
            best, best_w = acc, w

    rng = random.Random(0)
    cov_classes = [c for c in D.pseudo if has_c[D.cand_pos[D.ci[c]]]]
    rng.shuffle(cov_classes)
    half = len(cov_classes) // 2
    tune_cls, val_cls = set(cov_classes[:half]), set(cov_classes[half:])
    cand_list = cand.tolist()
    q_cls = [D.classes[cand_list[g]] for g in gold.tolist()]
    tune_m = torch.tensor([c in tune_cls for c in q_cls])
    val_m = torch.tensor([c in val_cls for c in q_cls])
    best_w_tune, best_tune = 0.0, top1(v39[tune_m], gold[tune_m])
    for w in [0.0, 0.5, 1.0, 1.5, 2.0]:
        acc = top1((v39 + w * S_n)[tune_m], gold[tune_m])
        if acc > best_tune:
            best_tune, best_w_tune = acc, w
    val_base = top1(v39[val_m], gold[val_m])
    val_tuned = top1((v39 + best_w_tune * S_n)[val_m], gold[val_m])

    out = {
        'v39_ref': ref, 'tag': a.tag, 'best': best, 'best_w_new': best_w,
        'delta_vs_v39': best - ref,
        'cv': {'w_new': best_w_tune, 'val_base': val_base, 'val_tuned': val_tuned},
    }
    op = os.path.join(OUT, f'inat_ortho_stack_{a.tag}.json')
    json.dump(out, open(op, 'w'), indent=1)
    print(f'best {best:.2f} @ w={best_w} delta {best-ref:+.2f}; wrote {op}', flush=True)


if __name__ == '__main__':
    main()
