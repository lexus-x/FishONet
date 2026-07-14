"""Test whether adding a general-purpose SigLIP-2 backbone to the deployed unseen
ensemble beats the current confirmed hard-sim best (DBNorm on ctft_big + frozen-L,
recorded in outputs/v15_val.json). SigLIP-2 is general web-scale (not fish-specific),
explicitly allowed under the competition rules -- unlike DINOv2 (measured dead, no
text encoder), SigLIP-2 is CLIP-style so it can zero-shot text-match on its own.

Reuses the exact pseudo-unseen hard-sim split and MH/ML computation from
val_dbnorm_ctft.py (same rarest-20%-of-seen-classes split) so the comparison is
apples-to-apples, then adds a third term uG@TnG and sweeps its ensemble weight.
Gate: only worth adopting if the 3-way ensemble clearly beats the 2-way baseline
(>0.3, matching the threshold gen_v15.py used for DBNorm itself).

  python src/val_dbnorm_siglip2.py
"""
import os, json, pickle, time
from collections import defaultdict
import torch, torch.nn as nn, torch.nn.functional as F
import open_clip
from PIL import Image
from torch.utils.data import Dataset, DataLoader

torch.set_num_threads(8)
MODEL = 'hf-hub:imageomics/bioclip-2.5-vith14'; DEV = 'cuda'; CKPT = 'outputs/ctft_big.pt'; SCALE = 0.4
SIGLIP_MODEL = 'ViT-SO400M-16-SigLIP2-384'; SIGLIP_PRETRAINED = 'webli'
STATUS = 'outputs/job_status.txt'; t0 = time.time()


def st(m):
    open(STATUS + '.tmp', 'w').write(
        'JOB: SigLIP-2 third-backbone unseen ensemble test\n  elapsed %6.1fs\n  %s\n' % (time.time() - t0, m))
    os.replace(STATUS + '.tmp', STATUS); print(m, flush=True)


class LoRALinear(nn.Module):
    def __init__(s, base, r=16, alpha=32):
        super().__init__(); s.base = base
        for p in s.base.parameters(): p.requires_grad_(False)
        s.r = r; s.scaling = alpha / max(r, 1)
        s.A = nn.Parameter(torch.zeros(r, base.in_features)); s.B = nn.Parameter(torch.zeros(base.out_features, r))
        s.drop = nn.Dropout(0.0); s.lora_scale = 1.0
    def forward(s, x): return s.base(x) + (s.drop(x) @ s.A.t() @ s.B.t()) * s.scaling * s.lora_scale


def inject_lora(model, r, alpha, top_k=0):
    blocks = model.visual.transformer.resblocks
    for i in (range(len(blocks)) if top_k <= 0 else range(len(blocks) - top_k, len(blocks))):
        blk = blocks[i]; blk.mlp.c_fc = LoRALinear(blk.mlp.c_fc, r, alpha); blk.mlp.c_proj = LoRALinear(blk.mlp.c_proj, r, alpha)


def set_scale(model, s):
    for blk in model.visual.transformer.resblocks:
        for m in (blk.mlp.c_fc, blk.mlp.c_proj):
            if isinstance(m, LoRALinear): m.lora_scale = s


def index_images(root):
    idx = {}
    for dp, _, fs in os.walk(root):
        for f in fs:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')): idx[f] = os.path.join(dp, f)
    return idx


D = 'data/dl'
st('loading ctft_big (H) + building pseudo-unseen split...')
classes = list(pickle.load(open(f'{D}/all_classes.pkl', 'rb'))); ci = {c: i for i, c in enumerate(classes)}
lab = json.load(open(f'{D}/label_train.json')); imgidx = index_images(f'{D}/images')
ck = torch.load(CKPT, weights_only=False)
model, _, pp = open_clip.create_model_and_transforms(MODEL); inject_lora(model, ck['rank'], ck['alpha'], ck['top_k'])
own = dict(model.named_parameters())
for n, v in ck['state'].items(): own[n].data.copy_(v)
model = model.to(DEV).eval(); set_scale(model, SCALE)

TtH = F.normalize(torch.load('outputs/text_emb_h_taxon.pt', weights_only=False)['emb_taxon'].float(), dim=-1)
TnL = F.normalize(torch.load('outputs/text_emb.pt', weights_only=False)['emb_name'].float(), dim=-1)
tr = torch.load('outputs/emb_train_h.pt', weights_only=False)
LtrD = torch.load('outputs/emb_train.pt', weights_only=False); LI = {fn: i for i, fn in enumerate(LtrD['files'])}; LF = F.normalize(LtrD['feats'].float(), dim=-1)

by = defaultdict(list)
for fn in tr['files']:
    if fn in lab and lab[fn] in ci and fn in imgidx and fn in LI: by[lab[fn]].append(fn)
seen = sorted(by.keys()); order = sorted(seen, key=lambda c: len(by[c]))
pseudo = set(order[:int(len(seen) * 0.2)]); known = set(c for c in seen if c not in pseudo)
cand = [i for i, c in enumerate(classes) if c not in known]; cand_pos = {x: j for j, x in enumerate(cand)}; cand_t = torch.tensor(cand)
qfn = [fn for c in pseudo for fn in by[c]]


class DS(Dataset):
    def __init__(s, it, preprocess): s.it = it; s.pp = preprocess
    def __len__(s): return len(s.it)
    def __getitem__(s, i):
        fn = s.it[i]
        try: return s.pp(Image.open(imgidx[fn]).convert('RGB')), fn
        except Exception: return torch.zeros(3, 224, 224), fn


def enc_tta(files, model_, preprocess, tag):
    dl = DataLoader(DS(files, preprocess), batch_size=128, num_workers=12, pin_memory=True)
    fs, kept, n = [], [], 0
    with torch.no_grad():
        for x, fns in dl:
            x = x.to(DEV)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                f = F.normalize(model_.encode_image(x).float(), dim=-1)
                f2 = F.normalize(model_.encode_image(torch.flip(x, dims=[-1])).float(), dim=-1)
                f = F.normalize(f + f2, dim=-1)
            fs.append(f.cpu()); kept += list(fns); n += len(fns)
            st(f'{tag}: encoded {n}/{len(files)} pseudo-unseen +hflip')
    return torch.cat(fs), kept


qf, kept = enc_tta(qfn, model, pp, 'ctft_big(H)')
qL = torch.stack([LF[LI[fn]] for fn in kept]); gold = torch.tensor([cand_pos[ci[lab[fn]]] for fn in kept])
MH = qf @ TtH[cand_t].t(); ML = qL @ TnL[cand_t].t()

# free the BioCLIP model before loading SigLIP-2 (avoid holding two ViT-H-scale models on GPU at once)
del model; torch.cuda.empty_cache()

st(f'loading {SIGLIP_MODEL}...')
gmodel, _, gpp = open_clip.create_model_and_transforms(SIGLIP_MODEL, pretrained=SIGLIP_PRETRAINED)
gtok = open_clip.get_tokenizer(SIGLIP_MODEL)
gmodel = gmodel.to(DEV).eval()

qG, keptG = enc_tta(qfn, gmodel, gpp, 'SigLIP-2(G)')
assert keptG == kept, 'query order mismatch between H and G encode passes'

st('encoding SigLIP-2 text bank for candidate classes...')
cand_classes = [classes[i] for i in cand]
prompts = [f'a photo of {c}' for c in cand_classes]
with torch.no_grad():
    TnG_list = []
    for i in range(0, len(prompts), 512):
        toks = gtok(prompts[i:i + 512]).to(DEV)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            e = F.normalize(gmodel.encode_text(toks).float(), dim=-1)
        TnG_list.append(e.cpu())
        st(f'text encoded {min(i + 512, len(prompts))}/{len(prompts)}')
TnG = torch.cat(TnG_list)
MG = qG @ TnG.t()


def db(M): return (M - M.mean(0, keepdim=True)) / (M.std(0, keepdim=True) + 1e-6)
def dis(M, a, b): return F.log_softmax(M / a, dim=0) + F.log_softmax(M / b, dim=1)
def t1(S): return (S.argmax(1) == gold).float().mean().item() * 100


v15 = json.load(open('outputs/v15_val.json')) if os.path.exists('outputs/v15_val.json') else None
tc, tr2 = (v15['tc'], v15['tr']) if v15 else (0.05, 0.5)
baseline = t1(dis(MH, tc, tr2) + 0.5 * dis(ML, tc, tr2))
st(f'baseline (H+0.5L, DBNorm tc={tc} tr={tr2}): {baseline:.2f}%  (v15_val.json base/dbnorm: {v15})')

best = (-1.0, None)
for w in [0.1, 0.2, 0.3, 0.5, 0.75, 1.0]:
    a = t1(dis(MH, tc, tr2) + 0.5 * dis(ML, tc, tr2) + w * dis(MG, tc, tr2))
    st(f'H+0.5L+{w}*G: {a:.2f}%  (baseline {baseline:.2f}, best {max(a, best[0]):.2f})')
    if a > best[0]: best = (a, w)

out = {'baseline': round(baseline, 2), 'best_with_siglip2': round(best[0], 2), 'best_w': best[1],
       'delta': round(best[0] - baseline, 2), 'n': len(kept), 'siglip_model': SIGLIP_MODEL}
json.dump(out, open('outputs/siglip2_val.json', 'w'))
verdict = ('ADOPT: SigLIP-2 clearly beats baseline' if best[0] > baseline + 0.3 else
           'DROP: no clear win, keep 2-way ensemble')
st(f'DONE: baseline {baseline:.2f} -> best {best[0]:.2f} @ w={best[1]} (delta {best[0]-baseline:+.2f}, n={len(kept)})  {verdict}')
print(verdict, flush=True)

