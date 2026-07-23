"""Post-FT SEEN-route eval: compare any feature set on the SAME per-class holdout used by ft.py,
with BOTH plain NCM (prototype) and the deployment blend (proto_sim + 2*class_max_sim).
Lets us read the TRUE deployment delta of fine-tuning (frozen vs FT) before touching a submission.

  frozen baseline: python src/ft_eval.py --feats outputs/emb_train_h_tta.pt
  FT features:     python src/ft_eval.py --feats outputs/emb_train_ft.pt
  (emb_train_ft.pt = FT features for ALL train imgs from `ft.py --extract train`)
"""
import json, argparse, torch
import torch.nn.functional as F
from collections import defaultdict

def build_split(lab, files_set):
    by = defaultdict(list)
    for fn, sp in lab.items():
        if fn in files_set: by[sp].append(fn)
    species = sorted(by.keys()); s2i = {c: i for i, c in enumerate(species)}
    tr_items, va_items = [], []
    for sp, fns in by.items():
        fns = sorted(fns); si = s2i[sp]
        if len(fns) >= 3:
            k = max(1, round(0.2 * len(fns))); tr, va = fns[:-k], fns[-k:]
        else:
            tr, va = fns, []
        tr_items += [(fn, si) for fn in tr]; va_items += [(fn, si) for fn in va]
    return species, s2i, tr_items, va_items

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--feats', required=True)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--chunk', type=int, default=256)
    a = ap.parse_args()
    dev = a.device
    d = torch.load(a.feats, weights_only=False)
    files, feats = d['files'], F.normalize(d['feats'].float(), dim=1)
    fmap = {fn: i for i, fn in enumerate(files)}
    lab = json.load(open('data/dl/label_train.json'))
    species, s2i, tr_items, va_items = build_split(lab, set(files))
    C = len(species); D = feats.shape[1]

    # train matrix + prototypes
    tr_idx = torch.tensor([fmap[fn] for fn, si in tr_items])
    tr_lab = torch.tensor([si for fn, si in tr_items])
    trF = feats[tr_idx].to(dev)                       # [Ntr, D]
    protos = torch.zeros(C, D); cnt = torch.zeros(C)
    protos.index_add_(0, tr_lab, feats[tr_idx]); cnt.index_add_(0, tr_lab, torch.ones(len(tr_lab)))
    protos = F.normalize(protos / cnt.clamp(min=1).unsqueeze(1), dim=1).to(dev)

    va_idx = torch.tensor([fmap[fn] for fn, si in va_items])
    vaF = feats[va_idx].to(dev); vaY = torch.tensor([si for fn, si in va_items]).to(dev)
    tr_lab = tr_lab.to(dev)

    ncm_ok = blend_ok = 0; N = len(vaY)
    for s in range(0, N, a.chunk):
        vf = vaF[s:s+a.chunk]; y = vaY[s:s+a.chunk]
        psim = vf @ protos.t()                                       # [c, C]
        ncm_ok += (psim.argmax(1) == y).sum().item()
        S = vf @ trF.t()                                             # [c, Ntr]
        cmax = torch.full((vf.shape[0], C), -1.0, device=dev)
        cmax.scatter_reduce_(1, tr_lab.unsqueeze(0).expand(vf.shape[0], -1), S, reduce='amax', include_self=True)
        blend = psim + 2.0 * cmax
        blend_ok += (blend.argmax(1) == y).sum().item()
    print(f'feats={a.feats}  val={N}  classes={C}')
    print(f'  SEEN NCM (proto)            : {100*ncm_ok/N:.2f}%')
    print(f'  SEEN blend (proto+2*cmax)   : {100*blend_ok/N:.2f}%')

if __name__ == '__main__':
    main()
