"""v105: DINOv2 self-embedding as a 4th SEEN-fusion member (orthogonal encoder,
never used in any MEMBERS/DEPLOYED list at all -- distinct from the DEAD DINOv2
iNat-proto "2nd leg" lines in HANDOFF, which used DINOv2 only to score external
iNat photos for the UNSEEN open-set leg. Here DINOv2 replaces nothing and adds
no external data: it is scored exactly like ctftshift/ftshift/fullft336shift --
train-fold class prototype + per-image bank cmax, over the SAME 64,259 training
files (`outputs/emb_train_dino.pt` has an IDENTICAL file set to ctftshift, so no
holdout-split shrinkage) -- and stacked onto the deployed 1.0/2.5/2.5 fusion with
one extra weight. DINOv2 has no text tower, so (unlike the 3 deployed members) it
gets NO taxon-text term (hs=False, matching the existing 'L' member's precedent
in learned_gate_v77.MEMBERS) -- the taxon text embedding lives in the BioCLIP-H
text space, which is meaningless to dot against a self-supervised ViT feature.

Kill bar: fused val_seen closed-set accuracy >= base(1.0/2.5/2.5) + 1.0pt.

  conda activate onet && python research/seen_dino_orthomember_v105.py
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
from seen_inat_v95 import member_scores  # deployed base_t = proto + 2*cmax + 4*taxon

DEPLOYED = {'ctftshift': 1.0, 'ftshift': 2.5, 'fullft336shift': 2.5}
DINO_GRID = [0.0, 0.25, 0.5, 1.0, 1.5, 2.5, 4.0]
KILL_LIFT = 1.0
BUCKETS = [(0, 2), (3, 5), (6, 1_000_000)]


def dino_scores(qF, P, TF, TL, S):
    """Same proto+2*cmax shape as member_scores, but NO taxon term (no text tower)."""
    out = torch.empty(qF.shape[0], S, device=dev)
    for i in range(0, qF.shape[0], 2000):
        e = qF[i:i + 2000]
        sim = e @ TF.t()
        cmax = torch.full((e.shape[0], S), -1e9, device=dev)
        cmax.scatter_reduce_(1, TL.unsqueeze(0).expand(e.shape[0], -1), sim, reduce='amax')
        out[i:i + 2000] = e @ P.t() + 2.0 * cmax
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
    n_train = torch.tensor([len(trby[c]) for c in seen], dtype=torch.float32)
    n_train_gold = n_train[yv]
    print(f'val_seen rows {len(val_seen)} | seen classes {S}', flush=True)

    TtH = F.normalize(torch.load(f'{OUT}/text_emb_h_taxon.pt', weights_only=False)['emb_taxon'].float(), dim=-1).to(dev)
    Tseen = TtH[kept_idx]

    # --- deployed 3-member base (reproduces the repo's current-best 91.674) ---
    Zbase = {}
    for t, _, _ in MEMBERS:
        if t not in DEPLOYED:
            continue
        idx, feats, _ = train[t]
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
        acc = 100 * (base.argmax(1).cpu() == yv).float().mean().item()
        print(f'  {t:16s} solo val acc {acc:6.2f}', flush=True)
        del TF, TL, qF, base
        torch.cuda.empty_cache()

    base_fused = sum(DEPLOYED[t] * Zbase[t] for t in DEPLOYED)
    base_acc = 100 * (base_fused.argmax(1).cpu() == yv).float().mean().item()
    print(f'\ndeployed 1.0/2.5/2.5 base val acc: {base_acc:.3f}', flush=True)

    # --- DINOv2 4th member: same recipe, own (never-before-used) embedding file ---
    didx, dfeats, dfiles = load_pt(f'{OUT}/emb_train_dino.pt')
    P = torch.zeros(S, dfeats.shape[1])
    cnt = torch.zeros(S)
    TFl, TLl = [], []
    missing = 0
    for c in seen:
        for fn in trby[c]:
            if fn not in didx:
                missing += 1
                continue
            f = dfeats[didx[fn]]
            P[s2i[c]] += f
            cnt[s2i[c]] += 1
            TFl.append(f)
            TLl.append(s2i[c])
    print(f'dino train-fold coverage: missing {missing} files (of {sum(len(trby[c]) for c in seen)})', flush=True)
    P = F.normalize(P / cnt.clamp(min=1).unsqueeze(1), dim=-1).to(dev)
    TF = torch.stack(TFl).to(dev)
    TL = torch.tensor(TLl).to(dev)
    qF = torch.stack([dfeats[didx[fn]] for fn in val_seen]).to(dev)
    dino_raw = dino_scores(qF, P, TF, TL, S)
    dino_acc = 100 * (dino_raw.argmax(1).cpu() == yv).float().mean().item()
    print(f'  {"dino":16s} solo val acc {dino_acc:6.2f}  (dim={dfeats.shape[1]})', flush=True)
    Zdino = zc(dino_raw)

    def fused_acc(w):
        f = base_fused + w * Zdino
        return 100 * (f.argmax(1).cpu() == yv).float().mean().item()

    print('\n=== pre-registered DINOv2-weight grid (val closed-set accuracy) ===', flush=True)
    rows = {}
    for w in DINO_GRID:
        rows[f'w{w}'] = fused_acc(w)
        print(f'  w_dino={w:<5} -> {rows[f"w{w}"]:.3f}', flush=True)

    best_key = max(rows, key=rows.get)
    best_w = float(best_key[1:])
    best_acc = rows[best_key]
    lift = best_acc - base_acc
    verdict = (f'CLEARS -- w_dino={best_w}' if lift >= KILL_LIFT
               else 'DEAD -- DINOv2 4th member does not lift holdout closed-set acc >= +1.0')
    print(f'\nbase {base_acc:.3f}  best {best_key} {best_acc:.3f}  lift {lift:+.3f}pt '
          f'(kill >= +{KILL_LIFT})', flush=True)
    print(f'VERDICT: {verdict}', flush=True)

    # bucket breakdown at the best weight vs base
    print('\n=== accuracy by n_train bucket: base vs base+w_dino*dino ===', flush=True)
    best_fused = base_fused + best_w * Zdino
    bucket_lines = []
    for lo, hi in BUCKETS:
        mask = (n_train_gold >= lo) & (n_train_gold <= hi)
        n_rows = int(mask.sum())
        yv_b = yv[mask]
        a_base = 100 * (base_fused[mask].argmax(1).cpu() == yv_b).float().mean().item()
        a_best = 100 * (best_fused[mask].argmax(1).cpu() == yv_b).float().mean().item()
        line = (f'  n_train in [{lo},{hi}]: {n_rows:5d} rows | base {a_base:6.2f} | '
                f'+dino(w={best_w}) {a_best:6.2f} | delta {a_best - a_base:+.2f}')
        print(line, flush=True)
        bucket_lines.append(line)

    json.dump({'rows': rows, 'base': base_acc, 'best': best_key, 'lift': lift,
               'verdict': verdict, 'dino_solo_acc': dino_acc, 'kill_lift': KILL_LIFT},
              open(f'{OUT}/seen_dino_orthomember_v105.json', 'w'), indent=2)
    print(f'\nwrote {OUT}/seen_dino_orthomember_v105.json', flush=True)
    return base_acc, best_acc, lift, bucket_lines


def load_pt(p):
    d = torch.load(p, weights_only=False)
    return {fn: i for i, fn in enumerate(d['files'])}, F.normalize(d['feats'].float(), dim=-1), list(d['files'])


if __name__ == '__main__':
    main()
