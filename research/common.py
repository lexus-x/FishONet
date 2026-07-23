"""Shared data/scoring utilities for FishOnet embedding-level research.

Reproduces the exact v20 data wiring (predict_v20_multienc.py) and the
hard-sim unseen split (rarest-20% classes) used across the project.

Run scripts from the repo root:
  C:\\Users\\lalit\\Documents\\projects\\fishonet> .venv\\Scripts\\python research\\gate_reproduce.py
"""
import json
import os
from collections import defaultdict

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'outputs')
DATA = os.path.join(ROOT, 'data')

torch.set_num_threads(max(1, (os.cpu_count() or 8) - 2))
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_emb(path):
    """Returns (fn->idx dict, L2-normalized float32 feats, files list)."""
    d = torch.load(path, weights_only=False)
    return ({fn: i for i, fn in enumerate(d['files'])},
            F.normalize(d['feats'].float(), dim=-1),
            list(d['files']))


def zc(M):
    return (M - M.mean()) / (M.std() + 1e-6)


def dbnorm(S, tc=0.05, tr=0.5):
    return F.log_softmax(S / tc, dim=0) + F.log_softmax(S / tr, dim=1)


def db(M):
    return (M - M.mean(0, keepdim=True)) / (M.std(0, keepdim=True) + 1e-6)



class FishData:
    """Loads all cached assets once (CPU)."""

    def __init__(self):
        txtH = torch.load(os.path.join(OUT, 'text_emb_h.pt'), weights_only=False)
        txtHt = torch.load(os.path.join(OUT, 'text_emb_h_taxon.pt'), weights_only=False)
        self.classes = txtH['classes']
        self.ci = {c: i for i, c in enumerate(self.classes)}
        self.text_h_keys = list(txtH.keys())
        self.TtH = F.normalize(txtHt['emb_taxon'].float(), dim=-1)      # [C,1024] taxon (H)
        txtL = torch.load(os.path.join(OUT, 'text_emb.pt'), weights_only=False)
        self.text_l_keys = list(txtL.keys())
        self.TnL = F.normalize(txtL['emb_name'].float(), dim=-1)        # [C,768] name (L)
        self.TdL = (F.normalize(txtL['emb_desc'].float(), dim=-1)
                    if 'emb_desc' in txtL else None)                    # [C,768] desc (L)
        self.TnH = (F.normalize(txtH['emb_name'].float(), dim=-1)
                    if 'emb_name' in txtH else None)                    # [C,1024] name (H)
        self.TdH = (F.normalize(txtH['emb_desc'].float(), dim=-1)
                    if 'emb_desc' in txtH else None)                    # [C,1024] desc (H)

        self.lab = json.load(open(os.path.join(DATA, 'dl', 'label_train.json')))

        self.FtI, self.FtF, self.Ftf = load_emb(os.path.join(OUT, 'emb_train_ft.pt'))
        self.CtI, self.CtF, _ = load_emb(os.path.join(OUT, 'emb_train_cap.pt'))
        self.LtI, self.LtF, _ = load_emb(os.path.join(OUT, 'emb_train.pt'))

        lab, ci = self.lab, self.ci
        self.trainfiles = [fn for fn in self.Ftf if fn in self.LtI and fn in self.CtI
                           and fn in lab and lab[fn] in ci]
        by = defaultdict(list)
        for fn in self.trainfiles:
            by[lab[fn]].append(fn)
        self.by = by
        self.seen = sorted(by.keys())
        self.s2i = {c: i for i, c in enumerate(self.seen)}
        self.S = len(self.seen)
        self.seen_set = set(self.seen)
        self.TseenTax = torch.stack([self.TtH[ci[c]] for c in self.seen])
        self.nonk = torch.tensor([i for i, c in enumerate(self.classes)
                                  if c not in self.seen_set])

        # hard-sim split: rarest 20% of seen classes -> pseudo-unseen
        order = sorted(self.seen, key=lambda c: len(by[c]))
        n_pseudo = int(len(self.seen) * 0.2)
        self.pseudo = order[:n_pseudo]
        self.pseudo_set = set(self.pseudo)
        self.kept = sorted(order[n_pseudo:])
        self.kept_set = set(self.kept)
        self.k2i = {c: i for i, c in enumerate(self.kept)}
        self.cand = torch.tensor([i for i, c in enumerate(self.classes)
                                  if c not in self.kept_set])
        self.cand_pos = {int(x): j for j, x in enumerate(self.cand.tolist())}

    def v20_holdout_split(self):
        """Exact v20 split: per class (sorted fns), last k=max(1,round(20%)) -> val."""
        trby = defaultdict(list)
        valrows = []
        for c in self.seen:
            fns = sorted(self.by[c])
            if len(fns) >= 3:
                k = max(1, round(0.2 * len(fns)))
                for f in fns[:-k]:
                    trby[c].append(f)
                for f in fns[-k:]:
                    valrows.append((f, self.s2i[c]))
            else:
                for f in fns:
                    trby[c].append(f)
        return trby, valrows


def protos_and_train(TrI, TrF, fns_by_cls, order, S, s2i):
    P = torch.zeros(S, TrF.shape[1])
    cnt = torch.zeros(S)
    TF, TL = [], []
    for c in order:
        for fn in fns_by_cls[c]:
            f = TrF[TrI[fn]]
            P[s2i[c]] += f
            cnt[s2i[c]] += 1
            TF.append(f)
            TL.append(s2i[c])
    P = F.normalize(P / cnt.clamp(min=1).unsqueeze(1), dim=-1)
    return P, torch.stack(TF), torch.tensor(TL)


def seen_score(qF, P, TF, TL, Tx=None, lam=0.0, cmax_w=2.0, S=None, chunk=1000):
    S = S or P.shape[0]
    out = torch.empty(qF.shape[0], S)
    for i in range(0, qF.shape[0], chunk):
        e = qF[i:i + chunk]
        ps = e @ P.t()
        sim = e @ TF.t()
        cmax = torch.full((e.shape[0], S), -1e9)
        cmax.scatter_reduce_(1, TL.unsqueeze(0).expand(e.shape[0], -1), sim, reduce='amax')
        sc = ps + cmax_w * cmax
        if Tx is not None and lam > 0:
            sc = sc + lam * (e @ Tx.t())
        out[i:i + chunk] = sc
    return out


def class_topk_mean(qF, TF, TL, S, k=2, chunk=1000):
    """Mean of top-k within-class cosine sims (k=1 reduces to cmax). Exact per-class."""
    cls_idx = [ (TL == s).nonzero(as_tuple=True)[0] for s in range(S) ]
    out = torch.empty(qF.shape[0], S)
    for i in range(0, qF.shape[0], chunk):
        e = qF[i:i + chunk]
        sim = e @ TF.t()
        for s in range(S):
            idx = cls_idx[s]
            kk = min(k, idx.numel())
            out[i:i + chunk, s] = sim[:, idx].topk(kk, dim=1).values.mean(dim=1)
    return out

