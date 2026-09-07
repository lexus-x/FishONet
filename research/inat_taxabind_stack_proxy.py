"""Stack test: ctftshift iNat protos + TaxaBind iNat protos (orthogonal encoder)."""
from __future__ import annotations

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


def main():
    tb_proto = os.path.join(OUT, 'inat_protos_taxabind.pt')
    if not os.path.exists(tb_proto):
        print(f'missing {tb_proto}', flush=True)
        return

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
    base = top1(S0, gold)

    Qc = legs['ctftshift']
    Pin = F.normalize(torch.load(os.path.join(OUT, 'inat_protos_ctftshift_full.pt'), weights_only=False)['protos'].float(), dim=-1)[cand]
    has_c = Pin.norm(dim=-1) > 0.5
    Sic = Qc @ Pin.t()
    Sic = Sic.masked_fill(~has_c.unsqueeze(0), -1e4)
    S_ctft = dbnorm(Sic)

    tb_idx, tb_feat, _ = load_emb(os.path.join(OUT, 'emb_train_taxabind.pt'))
    Qtb, _ = queries(tb_idx, tb_feat)
    Ptb = F.normalize(torch.load(tb_proto, weights_only=False)['protos'].float(), dim=-1)[cand]
    has_t = Ptb.norm(dim=-1) > 0.5
    Stb = Qtb @ Ptb.t()
    Stb = Stb.masked_fill(~has_t.unsqueeze(0), -1e4)
    S_tb = dbnorm(Stb)

    ref = top1(S0 + 3.0 * S_ctft, gold)
    print(f'S0={base:.2f} v37 ref S0+3*ctft={ref:.2f}', flush=True)

    best = ref
    best_cfg = (3.0, 0.0)
    for w_c in [2.5, 3.0, 3.5]:
        for w_t in [0.0, 0.25, 0.5, 1.0, 1.5, 2.0]:
            acc = top1(S0 + w_c * S_ctft + w_t * S_tb, gold)
            if acc > best:
                best, best_cfg = acc, (w_c, w_t)
            if w_t > 0:
                print(f'  w_ctft={w_c} w_tb={w_t}: {acc:.2f} ({acc-ref:+.2f})', flush=True)

    rng = random.Random(0)
    cov_classes = [c for c in D.pseudo if has_c[D.cand_pos[D.ci[c]]]]
    rng.shuffle(cov_classes)
    half = len(cov_classes) // 2
    tune_cls, val_cls = set(cov_classes[:half]), set(cov_classes[half:])
    cand_list = cand.tolist()
    q_cls = [D.classes[cand_list[g]] for g in gold.tolist()]
    tune_m = torch.tensor([c in tune_cls for c in q_cls])
    val_m = torch.tensor([c in val_cls for c in q_cls])
    best_wc, best_wt = 3.0, 0.0
    best_tune = top1(S0[tune_m], gold[tune_m])
    for w_c in [2.5, 3.0, 3.5]:
        for w_t in [0.0, 0.5, 1.0, 1.5, 2.0]:
            acc = top1((S0 + w_c * S_ctft + w_t * S_tb)[tune_m], gold[tune_m])
            if acc > best_tune:
                best_tune, best_wc, best_wt = acc, w_c, w_t
    val_base = top1(S0[val_m], gold[val_m])
    val_tuned = top1((S0 + best_wc * S_ctft + best_wt * S_tb)[val_m], gold[val_m])

    out = {
        'v37_ref': ref,
        'best': best,
        'best_w_ctft': best_cfg[0],
        'best_w_tb': best_cfg[1],
        'delta_vs_v37': best - ref,
        'cv': {'w_ctft': best_wc, 'w_tb': best_wt, 'val_base': val_base, 'val_tuned': val_tuned},
    }
    json.dump(out, open(os.path.join(OUT, 'inat_taxabind_stack_proxy.json'), 'w'), indent=1)
    print(f'best {best:.2f} @ wc={best_cfg[0]} wt={best_cfg[1]} (delta {best-ref:+.2f})', flush=True)
    print(f'CV val {val_base:.2f}->{val_tuned:.2f} wc={best_wc} wt={best_wt}', flush=True)


if __name__ == '__main__':
    main()
