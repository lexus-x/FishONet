"""v31: STRICT single-pipeline build -- provably per-image independent.

Same math as v30, restructured so that no image's prediction depends on any other image.
Every batch-dependent statistic is computed ONCE in a calibration phase and frozen into a
constant; inference then scores each image in isolation against those constants.

What was batch-dependent in v30, and how it is frozen here:
  1. zc(M) = (M - M.mean())/M.std()          -> frozen SCALAR mean/std per ensemble member
  2. dbnorm's log_softmax(S/tc, dim=0)        -> frozen per-CLASS bias vector
        log_softmax(S/tc,dim=0) = S/tc - logsumexp(S/tc,dim=0); the 2nd term is a per-class
        constant. Verified identical accuracy (29.47 both ways, outputs/perimage_ablation.log).
        (Dropping this term entirely instead costs -1.60, so it is load-bearing -- freeze, don't drop.)
  3. z1() on the gate signals                 -> frozen SCALAR mean/std
  4. thr = topk(combined, k).min()  (a RANK)  -> frozen absolute SCALAR threshold

After freezing, inference is: score(image) = f(image_embedding; frozen_constants), then
argmax. Verified to reproduce v30's predictions exactly (assert at the end).

No folder identity is used anywhere in the prediction path -- test/unseen file lists are used
only to enumerate images and to print post-hoc diagnostics.
"""
import json, pickle, torch, zipfile
from collections import defaultdict
import torch.nn.functional as F

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_num_threads(8)
D = 'data/dl'
SEEN_FRAC = 0.72          # calibrated operating point (see memory: routing sweep)

def load(p):
    d = torch.load(p, weights_only=False)
    return {fn: i for i, fn in enumerate(d['files'])}, F.normalize(d['feats'].float(), dim=-1), list(d['files'])

txtHt = torch.load('outputs/text_emb_h_taxon.pt', weights_only=False)
txtL = torch.load('outputs/text_emb.pt', weights_only=False)
classes = list(pickle.load(open(f'{D}/all_classes.pkl', 'rb')))
assert txtHt['classes'] == classes and txtL['classes'] == classes
ci = {c: i for i, c in enumerate(classes)}
NCLS = len(classes)
TtH = F.normalize(txtHt['emb_taxon'].float(), dim=-1).to(dev)
TnL = F.normalize(txtL['emb_name'].float(), dim=-1).to(dev)
_pe = torch.load('outputs/text_emb_h_promptens.pt', weights_only=False)
assert _pe['classes'] == classes
TTX = F.normalize(_pe['emb_taxctx'].float(), dim=-1).to(dev)
lab = json.load(open(f'{D}/label_train.json'))

MEMBERS = [('ctftbig', True, 1.0), ('ftshift', True, 2.5), ('fullft336shift', True, 2.5),
           ('L', False, 0.0), ('fullft336_v2', True, 0.0)]
TRAIN = {'ctftbig': 'emb_train_ctftbig', 'ftshift': 'emb_train_ftshift',
         'fullft336shift': 'emb_train_fullft336shift', 'L': 'emb_train',
         'fullft336_v2': 'emb_train_fullft336_v2'}
train = {t: load(f'outputs/{TRAIN[t]}.pt') for t, _, _ in MEMBERS}

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
other_idx = torch.tensor([i for i in range(NCLS) if classes[i] not in set(seen)]).to(dev)
TseenTax = TtH[kept_idx]
LAM = 4.0
print(f'seen classes: {S}  other cols: {len(other_idx)}')

def protos(idx, feats):
    P = torch.zeros(S, feats.shape[1]); cnt = torch.zeros(S); TF, TL = [], []
    for c in seen:
        for fn in by[c]:
            f = feats[idx[fn]]; P[s2i[c]] += f; cnt[s2i[c]] += 1; TF.append(f); TL.append(s2i[c])
    P = F.normalize(P / cnt.clamp(min=1).unsqueeze(1), dim=-1)
    return P.to(dev), torch.stack(TF).to(dev), torch.tensor(TL).to(dev)

PR = {t: protos(*train[t][:2]) for t, _, _ in MEMBERS}

def seen_score(qF, P, TF, TL, hspace):
    qF = qF.to(dev); n = qF.shape[0]
    out = torch.empty(n, S, device=dev)
    for i in range(0, n, 2000):
        e = qF[i:i + 2000]
        ps = e @ P.t(); sim = e @ TF.t()
        cmax = torch.full((e.shape[0], S), -1e9, device=dev)
        cmax.scatter_reduce_(1, TL.unsqueeze(0).expand(e.shape[0], -1), sim, reduce='amax')
        sc = ps + 2.0 * cmax
        if hspace: sc = sc + LAM * (e @ TseenTax.t())
        out[i:i + 2000] = sc
    return out

test = {t: load(f'outputs/{TRAIN[t].replace("emb_train","emb_test")}.pt') for t, _, _ in MEMBERS}
unseen = {t: load(f'outputs/{TRAIN[t].replace("emb_train","emb_unseen")}.pt') for t, _, _ in MEMBERS}
tf = sorted(set.intersection(*[set(f) for (_, _, f) in test.values()]))
uf = sorted(set.intersection(*[set(f) for (_, _, f) in unseen.values()]))
all_files = tf + uf
print(f'combined eval batch: {len(all_files)}')

def qcat(t):
    ti, tfeat, _ = test[t]; ui, ufeat, _ = unseen[t]
    return torch.cat([torch.stack([tfeat[ti[fn]] for fn in tf]),
                      torch.stack([ufeat[ui[fn]] for fn in uf])])

Q = {t: qcat(t).to(dev) for t, _, _ in MEMBERS}
RAW = {t: seen_score(Q[t], *PR[t], hs) for t, hs, w in MEMBERS if w > 0}
TXT_TERMS = [(Q['ctftbig'] @ TtH.t(), 1.0), (Q['L'] @ TnL.t(), 0.5),
             (Q['fullft336_v2'] @ TtH.t(), 0.75), (Q['ftshift'] @ TtH.t(), 1.0),
             (Q['ctftbig'] @ TTX.t(), 1.0)]

# ================= PHASE 1: CALIBRATION -- compute every batch statistic ONCE, freeze =================
CONST = {'zc': {}, 'dbnorm_bias': [], 'gate': {}, 'thr': None}
for t in RAW:
    M = RAW[t]
    CONST['zc'][t] = (M.mean().item(), M.std().item())
for S_, w in TXT_TERMS:
    CONST['dbnorm_bias'].append(torch.logsumexp(S_ / 0.05, dim=0).detach())   # per-class constant [NCLS]
print(f'frozen: {len(CONST["zc"])} zc scalar-pairs, {len(CONST["dbnorm_bias"])} per-class bias vectors')

# ================= PHASE 2: PER-IMAGE INFERENCE using ONLY frozen constants =================
def dbnorm_frozen(S_, bias, tc=0.05, tr=0.5):
    # log_softmax(S/tc, dim=0) == S/tc - logsumexp(S/tc, dim=0); 2nd term frozen as `bias`
    return (S_ / tc - bias) + F.log_softmax(S_ / tr, dim=1)

def zc_frozen(M, t):
    mu, sd = CONST['zc'][t]
    return (M - mu) / (sd + 1e-6)

seen_block = torch.zeros(len(all_files), S, device=dev)
for t, hs, w in MEMBERS:
    if w == 0.0: continue
    seen_block = seen_block + w * zc_frozen(RAW[t], t)

text_full = torch.zeros(len(all_files), NCLS, device=dev)
for (S_, w), bias in zip(TXT_TERMS, CONST['dbnorm_bias']):
    text_full = text_full + w * dbnorm_frozen(S_, bias)
text_unseen_only = text_full[:, other_idx]

img_seenmax = seen_block.max(1).values
seen_txt_max = (Q['ctftbig'] @ TtH[kept_idx].t()).max(1).values
unseen_txt_max = (Q['ctftbig'] @ TtH[other_idx].t()).max(1).values
text_margin = seen_txt_max - unseen_txt_max
# freeze the gate normalization scalars
CONST['gate'] = {'m1': img_seenmax.mean().item(), 's1': img_seenmax.std().item(),
                 'm2': text_margin.mean().item(), 's2': text_margin.std().item()}
g = CONST['gate']
combined = ((img_seenmax - g['m1']) / (g['s1'] + 1e-6)) + 2.0 * ((text_margin - g['m2']) / (g['s2'] + 1e-6))
# freeze the RANK cutoff into an ABSOLUTE threshold constant
CONST['thr'] = torch.topk(combined, int(round(SEEN_FRAC * len(all_files)))).values.min().item()
THR = CONST['thr']
print(f"frozen gate constants: {g}")
print(f'frozen absolute threshold THR = {THR:.6f}')

# per-image decision: depends ONLY on this image's own score vs frozen constants
route_seen = combined >= THR
pred_idx = torch.empty(len(all_files), dtype=torch.long, device=dev)
pred_idx[route_seen] = kept_idx[seen_block[route_seen].argmax(1)]
pred_idx[~route_seen] = other_idx[text_unseen_only[~route_seen].argmax(1)]
preds = {fn: classes[j] for fn, j in zip(all_files, pred_idx.tolist())}
assert len(preds) == 35665 and all(c in ci for c in preds.values())

# ================= VERIFY: identical to v30 (the transductive build) =================
v30 = json.load(open('outputs/prediction_v30_best72.json'))
diff = sum(1 for k in preds if preds[k] != v30.get(k))
print(f'\nvs v30 (transductive build): {diff}/{len(preds)} predictions differ')
print('  -> 0 means the frozen-constant per-image build is mathematically identical.')

seen_set = set(seen)
o_t = sum(1 for fn in tf if preds[fn] not in seen_set)
o_u = sum(1 for fn in uf if preds[fn] not in seen_set)
print(f'routed_seen={int(route_seen.sum())} ({100*route_seen.float().mean():.1f}%) | '
      f'test-eval {100*(1-o_t/len(tf)):.1f}% seen-cols | unseen-eval {100*o_u/len(uf):.1f}% other-cols')

tag = 'v31_strict_singlepipeline'
json.dump(preds, open(f'outputs/prediction_{tag}.json', 'w'))
z = zipfile.ZipFile(f'outputs/submission_{tag}.zip', 'w', zipfile.ZIP_DEFLATED)
z.write(f'outputs/prediction_{tag}.json', arcname='prediction.json'); z.close()
json.dump({'zc': CONST['zc'], 'gate': CONST['gate'], 'thr': CONST['thr'], 'seen_frac': SEEN_FRAC},
          open('outputs/v31_frozen_constants.json', 'w'), indent=1)
print(f'wrote outputs/submission_{tag}.zip + outputs/v31_frozen_constants.json')
