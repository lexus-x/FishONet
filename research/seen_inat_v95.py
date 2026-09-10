"""v95 Tier A: external iNat photo evidence on the SEEN route.

Every real unseen gain came from external iNat photos; the seen head still
scores against only the training jpgs. This adds, per deployed member
(ctftshift 1.0 / ftshift 2.5 / fullft336shift 2.5), an iNat evidence leg
    leg_t = zc(q @ inat_proto_t + 2.0 * inat_cmax_t)
fused with a SINGLE pre-registered weight wi:
    fused = sum_t w_t * zc(base_t)  +  wi * sum_t w_t * leg_t
base_t = deployed proto + 2*cmax + 4*taxon (unchanged). Uncovered classes get
the leg's per-row mean (neutral fill) so coverage gaps do not bias the argmax.

Selection HOLDOUT-ONLY (v93 protocol: rarest-20% pseudo split, val_seen rows).
iNat photos are external -> no leak vs val_seen. Pre-registered before any
number was seen:
  grid wi in {0, 0.25, 0.5, 1.0, 1.5, 2.5}
  kill bar: best wi lifts val closed-set accuracy >= +1.0pt over wi=0

  conda activate onet && python research/seen_inat_v95.py
"""
from __future__ import annotations

import json
import pickle
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, 'research')
from common import OUT
from learned_gate_v77 import D, MEMBERS, dev, holdout_split, load_train_embs, zc

LAM = 4.0
BASE_W = {'ctftshift': 1.0, 'ftshift': 2.5, 'fullft336shift': 2.5}
GRID_WI = [0.0, 0.25, 0.5, 1.0, 1.5, 2.5]
KILL_LIFT = 1.0
BANKS = {
    'ctftshift': f'{OUT}/inat_seen_bank_ctftshift.pt',
    'ftshift': f'{OUT}/inat_seen_bank_ftshift.pt',
    'fullft336shift': f'{OUT}/inat_seen_bank_fullft336shift.pt',
}


def member_scores(qF, P, TF, TL, Tseen, S):
    """Deployed base: proto + 2*cmax + LAM*taxon."""
    out = torch.empty(qF.shape[0], S, device=dev)
    for i in range(0, qF.shape[0], 2000):
        e = qF[i:i + 2000]
        sim = e @ TF.t()
        cmax = torch.full((e.shape[0], S), -1e9, device=dev)
        cmax.scatter_reduce_(1, TL.unsqueeze(0).expand(e.shape[0], -1), sim, reduce='amax')
        out[i:i + 2000] = e @ P.t() + 2.0 * cmax + LAM * (e @ Tseen.t())
    return out


def inat_leg(qF, IP, IF, IL, covered_mask, S):
    """q @ inat_proto + 2*inat_cmax; uncovered columns = per-row mean (neutral)."""
    out = torch.empty(qF.shape[0], S, device=dev)
    for i in range(0, qF.shape[0], 2000):
        e = qF[i:i + 2000]
        sim = e @ IF.t()
        imax = torch.full((e.shape[0], S), -1e9, device=dev)
        imax.scatter_reduce_(1, IL.unsqueeze(0).expand(e.shape[0], -1), sim, reduce='amax')
        raw = e @ IP.t() + 2.0 * imax
        mu = raw[:, covered_mask].mean(1, keepdim=True)
        raw[:, ~covered_mask] = mu
        out[i:i + 2000] = raw
    return out


def main():
    torch.set_num_threads(8)
    lab = json.load(open(f'{D}/label_train.json'))
    classes = list(pickle.load(open(f'{D}/all_classes.pkl', 'rb')))
    ci = {c: i for i, c in enumerate(classes)}

    train, tb_train, b2f_train, b2l_train = load_train_embs()
    sp = holdout_split(train, tb_train, b2f_train, b2l_train)
    trby, val_seen = sp['trby'], [f for f, yv in zip(sp['val_files'], sp['y']) if yv == 1]
    seen = sorted(sp['kept'])
    s2i = {c: i for i, c in enumerate(seen)}
    S = len(seen)
    kept_idx = torch.tensor([ci[c] for c in seen]).to(dev)
    yv = torch.tensor([s2i[lab[f]] for f in val_seen])
    print(f'val_seen rows {len(val_seen)} | seen classes {S}', flush=True)

    TtH = F.normalize(torch.load(f'{OUT}/text_emb_h_taxon.pt', weights_only=False)['emb_taxon'].float(), dim=-1).to(dev)
    Tseen = TtH[kept_idx]

    Zbase, Zleg = {}, {}
    for t, _, _ in MEMBERS:
        if t not in BASE_W:
            continue
        idx, feats, _ = train[t]
        # deployed protos + train-fold bank (val rows excluded via trby)
        P = torch.zeros(S, feats.shape[1])
        cnt = torch.zeros(S)
        TFl, TLl = [], []
        for c in seen:
            for fn in trby[c]:
                if fn not in idx:
                    continue
                f = feats[idx[fn]]
                P[s2i[c]] += f
                cnt[s2i[c]] += 1
                TFl.append(f)
                TLl.append(s2i[c])
        P = F.normalize(P / cnt.clamp(min=1).unsqueeze(1), dim=-1).to(dev)
        TF = torch.stack(TFl).to(dev)
        TL = torch.tensor(TLl).to(dev)
        qF = torch.stack([feats[idx[fn]] for fn in val_seen]).to(dev)

        base = member_scores(qF, P, TF, TL, Tseen, S)
        Zbase[t] = zc(base)
        acc1 = 100 * (base.argmax(1).cpu() == yv).float().mean().item()

        bank = torch.load(BANKS[t], weights_only=False)['bank']
        IFl, ILl = [], []
        IP = torch.zeros(S, feats.shape[1])
        covered = torch.zeros(S, dtype=torch.bool)
        for gci, mat in bank.items():
            c = classes[gci]
            if c not in s2i:
                continue
            si = s2i[c]
            m = F.normalize(mat.float(), dim=-1)
            IP[si] = F.normalize(m.mean(0), dim=0)
            covered[si] = True
            IFl.append(m)
            ILl.append(torch.full((m.shape[0],), si, dtype=torch.long))
        IF = torch.cat(IFl).to(dev)
        IL = torch.cat(ILl).to(dev)
        IP = IP.to(dev)
        leg = inat_leg(qF, IP, IF, IL, covered.to(dev), S)
        Zleg[t] = zc(leg)
        lacc = 100 * (leg.argmax(1).cpu() == yv).float().mean().item()
        print(f'  {t:16s} base val acc {acc1:6.2f} | inat-leg solo {lacc:6.2f} '
              f'| bank photos {IF.shape[0]} covered {int(covered.sum())}/{S}', flush=True)
        del TF, TL, qF, base, leg, IF, IL
        torch.cuda.empty_cache()

    def fused_acc(wi):
        f = sum(BASE_W[t] * (Zbase[t] + wi * Zleg[t]) for t in BASE_W)
        return 100 * (f.argmax(1).cpu() == yv).float().mean().item()

    print('\n=== pre-registered wi grid (val closed-set accuracy, holdout-only) ===', flush=True)
    rows = {}
    for wi in GRID_WI:
        rows[f'wi{wi}'] = fused_acc(wi)
        print(f'  wi={wi:<5} ->  {rows[f"wi{wi}"]:6.2f}', flush=True)
    base_acc = rows['wi0.0']
    best = max(rows, key=rows.get)
    lift = rows[best] - base_acc
    verdict = (f'CLEARS — build v95 seen argmax with {best}' if lift >= KILL_LIFT
               else 'DEAD — iNat seen evidence does not lift holdout closed-set acc >= +1.0')
    print(f'\nbase {base_acc:.2f}  best {best} {rows[best]:.2f}  lift {lift:+.2f}pt '
          f'(kill >= +{KILL_LIFT})', flush=True)
    print(f'VERDICT: {verdict}', flush=True)
    json.dump({'rows': rows, 'base': base_acc, 'best': best, 'lift': lift,
               'verdict': verdict, 'kill_lift': KILL_LIFT},
              open(f'{OUT}/seen_inat_v95.json', 'w'), indent=2)
    print(f'wrote {OUT}/seen_inat_v95.json', flush=True)


if __name__ == '__main__':
    main()
