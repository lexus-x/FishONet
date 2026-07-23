"""Test CSLS hubness correction (vs z-debias) + template-ensemble on the ViT-H unseen route.
All on cached embeddings, CPU. Transductive CSLS uses query-set stats (no labels) -> rules-legal."""
import json, pickle, torch
from collections import defaultdict
import torch.nn.functional as F

D = 'data/dl'
classes = list(pickle.load(open(f'{D}/all_classes.pkl', 'rb')))
cls_index = {c: i for i, c in enumerate(classes)}
th = torch.load('outputs/text_emb_h.pt', weights_only=False)
emb_name = F.normalize(th['emb_name'].float(), dim=-1)
tt = torch.load('outputs/text_emb_h_taxon.pt', weights_only=False)
emb_taxon = F.normalize(tt['emb_taxon'].float(), dim=-1)
# template-ensemble file if present
emb_ens = None
try:
    te = torch.load('outputs/text_emb_h_ens.pt', weights_only=False)
    print('text_emb_h_ens keys:', list(te.keys()))
    for k in ('emb_name', 'emb_ens', 'emb_name_ens'):
        if k in te:
            emb_ens = F.normalize(te[k].float(), dim=-1); print('using ens key', k); break
except Exception as e:
    print('no ens file', e)

tr = torch.load('outputs/emb_train_h.pt', weights_only=False)
lab = json.load(open(f'{D}/label_train.json'))
by_cls = defaultdict(list)
for fn, f in zip(tr['files'], tr['feats']):
    if fn in lab and lab[fn] in cls_index:
        by_cls[lab[fn]].append(f)
seen = sorted(by_cls.keys())
order = sorted(seen, key=lambda c: len(by_cls[c]))
n_un = int(len(seen) * 0.2)
pseudo = set(order[:n_un]); known = [c for c in seen if c not in pseudo]
known_idx = set(cls_index[c] for c in known)
cand = [i for i in range(len(classes)) if i not in known_idx]
cand_pos = {ci: j for j, ci in enumerate(cand)}
cand_t = torch.tensor(cand)
qf, qy = [], []
for c in pseudo:
    for f in by_cls[c]:
        qf.append(f); qy.append(cls_index[c])
Fq = F.normalize(torch.stack(qf).float(), dim=-1)
gold = torch.tensor([cand_pos[y] for y in qy])
print(f'queries={len(qy)} candidates={len(cand)}')

def t1(S): return (S.argmax(1) == gold).float().mean().item() * 100
def zdb(S): return (S - S.mean(0, keepdim=True)) / (S.std(0, keepdim=True) + 1e-6)
def csls(S, k=10):
    # S: queries x cands cosine. r_c = mean top-k over queries (per cand); r_q = mean top-k over cands (per query)
    r_c = S.topk(k, dim=0).values.mean(0, keepdim=True)      # 1 x cands
    r_q = S.topk(k, dim=1).values.mean(1, keepdim=True)      # queries x 1
    return 2 * S - r_c - r_q

def report(tag, T):
    S = Fq @ T.t()
    print(f'\n--- {tag} ---')
    print(f'  raw        {t1(S):.2f}')
    print(f'  z-debias   {t1(zdb(S)):.2f}')
    for k in (5, 10, 20):
        print(f'  CSLS k={k:<2}   {t1(csls(S,k)):.2f}')
    # CSLS then z-debias
    print(f'  CSLS10+zdb {t1(zdb(csls(S,10))):.2f}')

report('NAME', emb_name[cand_t])
report('TAXON', emb_taxon[cand_t])
if emb_ens is not None:
    report('TEMPLATE-ENSEMBLE', emb_ens[cand_t])

# combined: best debias on name+taxon ensemble similarity
Sn = Fq @ emb_name[cand_t].t(); St = Fq @ emb_taxon[cand_t].t()
print('\n=== combined name+taxon ===')
for w in (0.5, 0.7, 1.0):
    Sc = St + w * Sn
    print(f'  taxon+{w}*name : zdb {t1(zdb(Sc)):.2f}  CSLS10 {t1(csls(Sc,10)):.2f}  CSLS10+zdb {t1(zdb(csls(Sc,10))):.2f}')
print('\nREF: name+zdb 22.35 (real ~12.2)  taxon+zdb 22.99')
