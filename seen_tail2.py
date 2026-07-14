"""Optimize the SEEN text-blend: sweep lam (name vs taxon text), re-tune cmax coeff jointly.
Same ft.py holdout. Also report the FROZEN-feature version for sanity (deploy uses FT)."""
import json, torch
import torch.nn.functional as F
from collections import defaultdict

D = 'data/dl'
th = torch.load('outputs/text_emb_h.pt', weights_only=False)
classesAll = th['classes']; ciAll = {c: i for i, c in enumerate(classesAll)}
TnH = F.normalize(th['emb_name'].float(), dim=-1)
TtH = F.normalize(torch.load('outputs/text_emb_h_taxon.pt', weights_only=False)['emb_taxon'].float(), dim=-1)

def run(featpath, tag):
    d = torch.load(featpath, weights_only=False)
    files, feats = d['files'], F.normalize(d['feats'].float(), dim=1)
    fmap = {fn: i for i, fn in enumerate(files)}
    lab = json.load(open(f'{D}/label_train.json'))
    by = defaultdict(list)
    for fn, sp in lab.items():
        if fn in fmap: by[sp].append(fn)
    species = sorted(by.keys()); s2i = {c: i for i, c in enumerate(species)}; C = len(species)
    Tn = torch.stack([TnH[ciAll[c]] for c in species]); Tt = torch.stack([TtH[ciAll[c]] for c in species])
    tr_items, va_items = [], []
    for sp, fns in by.items():
        fns = sorted(fns); si = s2i[sp]
        if len(fns) >= 3:
            k = max(1, round(0.2*len(fns))); tr, va = fns[:-k], fns[-k:]
        else: tr, va = fns, []
        tr_items += [(fn, si) for fn in tr]; va_items += [(fn, si) for fn in va]
    tr_idx = torch.tensor([fmap[fn] for fn,si in tr_items]); tr_lab = torch.tensor([si for fn,si in tr_items])
    trF = feats[tr_idx]
    protos = torch.zeros(C, feats.shape[1]); c2 = torch.zeros(C)
    protos.index_add_(0, tr_lab, trF); c2.index_add_(0, tr_lab, torch.ones(len(tr_lab)))
    protos = F.normalize(protos/c2.clamp(min=1).unsqueeze(1), dim=1)
    va_idx = torch.tensor([fmap[fn] for fn,si in va_items]); vaF = feats[va_idx]
    vaY = torch.tensor([si for fn,si in va_items])
    ps, cm, tn, tt = [], [], [], []
    for s in range(0, vaF.shape[0], 512):
        vf = vaF[s:s+512]
        psim = vf @ protos.t(); S = vf @ trF.t()
        cmax = torch.full((vf.shape[0], C), -1.0)
        cmax.scatter_reduce_(1, tr_lab.unsqueeze(0).expand(vf.shape[0],-1), S, reduce='amax', include_self=True)
        ps.append(psim); cm.append(cmax); tn.append(vf @ Tn.t()); tt.append(vf @ Tt.t())
    ps=torch.cat(ps); cm=torch.cat(cm); tn=torch.cat(tn); tt=torch.cat(tt); N=len(vaY)
    def acc(M): return 100*(M.argmax(1)==vaY).float().mean().item()
    print(f'\n===== {tag} ({featpath}) N={N} =====')
    print(f'blend (proto+2cmax)          : {acc(ps+2*cm):.2f}')
    for lam in (2,3,4,5,6,8):
        print(f'blend + {lam}*NAME-text        : {acc(ps+2*cm+lam*tn):.2f}    + {lam}*TAXON-text : {acc(ps+2*cm+lam*tt):.2f}')
    # joint retune cmax coeff with name-text lam
    print('  joint (cmax_w, name_lam):')
    best=(0,0,0)
    for cw in (1.5,2.0,2.5,3.0):
        row=[]
        for lam in (3,4,5):
            a=acc(ps+cw*cm+lam*tn); row.append(f'cw{cw}/lam{lam}:{a:.2f}')
            if a>best[0]: best=(a,cw,lam)
        print('   ',' '.join(row))
    print(f'  BEST: acc={best[0]:.2f} cmax_w={best[1]} name_lam={best[2]}')

run('outputs/emb_train_ft.pt', 'FT (deployed)')
run('outputs/emb_train_h_tta.pt', 'FROZEN-H-TTA (sanity)')
