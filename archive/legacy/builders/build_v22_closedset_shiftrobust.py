"""v22: maximal shift-robust CLOSED-SET single pipeline.

Two real submissions proved the novelty gate is worthless (gamma=10 and gamma=30 give byte-identical
45.19% -- every unseen caught ejects an equal hard-seen under the train->test shift). So this drops
the gate entirely: every image gets argmax over SEEN classes only. Single-pipeline compliant (one
uniform rule for all 35665 images; it simply never predicts an unseen class). This maximizes the
8:1-dominant seen term A_full, the only lever with headroom on existing assets.

The deployed recipe's seen_block used only ft+cap+L+ctftbig. This adds every ensemble member that
has train+test+unseen embeddings cached AND is shift-oriented: fullft336 + fullft336_v2 (full-network
FT, built to resist the shift) + frozen h + h_tta. Ensembling diverse members reduces the
shift-induced error and should lift REAL A_full above the 80.5% single-model closed-set floor.

Reports a holdout seen-accuracy check first (relative ensemble gains transfer even though absolute
holdout overstates real). Then builds the real 35665-image submission.

Ceiling honesty: pure closed-set caps at 0.5635 * A_full (unseen always scored 0). Even A_full=84%
-> only 47.3% overall. This is step 1 (does shift-robust ensembling move real A_full?), NOT the 53%
solution -- 53% is above the current models' perfect-gate ceiling of 51.4% and needs genuinely
better classifiers. See memory onet-53pct-challenge-analysis.
"""
import json, pickle, torch, zipfile
from collections import defaultdict
import torch.nn.functional as F

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_num_threads(8)
D = 'data/dl'


def load(p):
    d = torch.load(p, weights_only=False)
    return {fn: i for i, fn in enumerate(d['files'])}, F.normalize(d['feats'].float(), dim=-1), list(d['files'])


def zc(M):
    return (M - M.mean()) / (M.std() + 1e-6)


txtHt = torch.load('outputs/text_emb_h_taxon.pt', weights_only=False)
classes = list(pickle.load(open(f'{D}/all_classes.pkl', 'rb')))
assert txtHt['classes'] == classes
ci = {c: i for i, c in enumerate(classes)}
NCLS = len(classes)
TtH = F.normalize(txtHt['emb_taxon'].float(), dim=-1).to(dev)
lab = json.load(open(f'{D}/label_train.json'))

# members: (tag, train_path, hspace?, weight). L is 768-dim so taxon-text term only applies to H-space members.
MEMBERS = [
    ('ft',           'outputs/emb_train_ft.pt',           True,  1.0),
    ('cap',          'outputs/emb_train_cap.pt',          True,  1.0),
    ('L',            'outputs/emb_train.pt',              False, 0.5),
    ('ctftbig',      'outputs/emb_train_ctftbig.pt',      True,  1.0),
    ('fullft336',    'outputs/emb_train_fullft336.pt',    True,  0.75),
    ('fullft336_v2', 'outputs/emb_train_fullft336_v2.pt', True,  0.75),
    ('h',            'outputs/emb_train_h.pt',            True,  0.5),
    ('h_tta',        'outputs/emb_train_h_tta.pt',        True,  0.5),
]
TEST = {t: p.replace('emb_train', 'emb_test') for t, p, _, _ in MEMBERS}
UNSEEN = {t: p.replace('emb_train', 'emb_unseen') for t, p, _, _ in MEMBERS}

train = {t: load(p) for t, p, _, _ in MEMBERS}

# common train images across all members
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
TseenTax = TtH[kept_idx]
print(f'seen classes: {S}  common train imgs: {len(common)}')
LAM = 4.0


def protos_and_train(idx, feats):
    dim = feats.shape[1]
    P = torch.zeros(S, dim)
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


protos = {}
for t, (idx, feats, _) in train.items():
    protos[t] = protos_and_train(idx, feats)


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


# ---------- HOLDOUT CHECK (v20 image split): does the bigger ensemble beat ft+cap+L+ctftbig? ----------
trby = defaultdict(list)
valrows = []
for c in seen:
    fns = sorted(by[c])
    if len(fns) >= 3:
        k = max(1, round(0.2 * len(fns)))
        for f in fns[:-k]:
            trby[c].append(f)
        for f in fns[-k:]:
            valrows.append((f, s2i[c]))
    else:
        for f in fns:
            trby[c].append(f)
VY = torch.tensor([y for _, y in valrows]).to(dev)
valfns = [f for f, _ in valrows]


def protos_holdout(idx, feats):
    dim = feats.shape[1]
    P = torch.zeros(S, dim)
    cnt = torch.zeros(S)
    TF, TL = [], []
    for c in seen:
        for fn in trby[c]:
            f = feats[idx[fn]]
            P[s2i[c]] += f
            cnt[s2i[c]] += 1
            TF.append(f)
            TL.append(s2i[c])
    P = F.normalize(P / cnt.clamp(min=1).unsqueeze(1), dim=-1)
    return P.to(dev), torch.stack(TF).to(dev), torch.tensor(TL).to(dev)


def holdout_seen_block(tags):
    acc_block = torch.zeros(len(valfns), S, device=dev)
    for t, _, hs, w in MEMBERS:
        if t not in tags:
            continue
        idx, feats, _ = train[t]
        Ph, TFh, TLh = protos_holdout(idx, feats)
        q = torch.stack([feats[idx[fn]] for fn in valfns])
        acc_block = acc_block + w * zc(seen_score(q, Ph, TFh, TLh, hs))
    pred = kept_idx[acc_block.argmax(1)]
    return (pred == kept_idx[VY]).float().mean().item() * 100


base_tags = {'ft', 'cap', 'L', 'ctftbig'}
all_tags = {t for t, _, _, _ in MEMBERS}
print(f'[holdout] deployed seen_block (ft+cap+L+ctftbig): {holdout_seen_block(base_tags):.2f}')
print(f'[holdout] v22 full shift-robust ensemble:         {holdout_seen_block(all_tags):.2f}')

# ---------- REAL: build pure closed-set over all 35665 ----------
test = {t: load(TEST[t]) for t, _, _, _ in MEMBERS}
unseen = {t: load(UNSEEN[t]) for t, _, _, _ in MEMBERS}
tf = sorted(set.intersection(*[set(files) for (_, _, files) in test.values()]))
uf = sorted(set.intersection(*[set(files) for (_, _, files) in unseen.values()]))
assert set(tf).isdisjoint(uf)
all_files = tf + uf
print(f'combined eval batch: {len(all_files)} (test={len(tf)} unseen={len(uf)}, expect 35665)')

seen_block = torch.zeros(len(all_files), S, device=dev)
for t, _, hs, w in MEMBERS:
    ti, tfeat, _ = test[t]
    ui, ufeat, _ = unseen[t]
    q = torch.cat([torch.stack([tfeat[ti[fn]] for fn in tf]),
                   torch.stack([ufeat[ui[fn]] for fn in uf])])
    P, TF, TL = protos[t]
    seen_block = seen_block + w * zc(seen_score(q, P, TF, TL, hs))

pred_idx = kept_idx[seen_block.argmax(1)]
preds = {fn: classes[j] for fn, j in zip(all_files, pred_idx.tolist())}
assert len(preds) == 35665 and all(c in ci for c in preds.values())

tag = 'v22_closedset_shiftrobust'
json.dump(preds, open(f'outputs/prediction_{tag}.json', 'w'))
z = zipfile.ZipFile(f'outputs/submission_{tag}.zip', 'w', zipfile.ZIP_DEFLATED)
z.write(f'outputs/prediction_{tag}.json', arcname='prediction.json')
z.close()
print(f'wrote outputs/submission_{tag}.zip (pure closed-set; unseen scored 0 by design)')
print(f'expected overall = 0.5635 * real_A_full  (beats 45.19 iff real ensemble A_full > 80.2%)')
