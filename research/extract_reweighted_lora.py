"""Extract embeddings from a Colab 'reweighted_lora_distill' checkpoint
(parametrization-format LoRA: A_q/B_q/A_v/B_v on attn.in_proj_weight).
Merges the LoRA delta into the base weights, then hflip-TTA extraction
identical to extract_ctft_any.py. Saves {files, feats}.

Run: python research/extract_reweighted_lora.py \
  --ckpt ~/reweighted_lora_distill_lambda_0p01/final_lora_state.pt \
  --split train --out outputs/emb_train_reweighted_lora.pt
"""
import os, time, argparse, pickle, re
import torch, torch.nn.functional as F
import open_clip
from PIL import Image
from torch.utils.data import Dataset, DataLoader

MODEL = 'hf-hub:imageomics/bioclip-2.5-vith14'; DEV = 'cuda'

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', required=True)
ap.add_argument('--scale', type=float, default=1.0, help='WiSE-FT style scale on the LoRA delta')
ap.add_argument('--split', default='train')
ap.add_argument('--out', required=True)
ap.add_argument('--bs', type=int, default=128)
ap.add_argument('--workers', type=int, default=8)
a = ap.parse_args()

ck = torch.load(os.path.expanduser(a.ckpt), map_location='cpu', weights_only=False)
cfg = ck['config']
print('ckpt', a.ckpt, '| method', ck['method'], '| epoch', ck['epoch'],
      '| rank', cfg['lora_rank'], 'alpha', cfg['lora_alpha'], flush=True)
assert cfg['model_name'] == MODEL, cfg['model_name']
lora_scaling = cfg['lora_alpha'] / cfg['lora_rank']

model, _, pp = open_clip.create_model_and_transforms(MODEL)
blocks = model.visual.transformer.resblocks
d = blocks[0].attn.in_proj_weight.shape[1]
merged = 0
for k, A in ck['lora_state'].items():
    m = re.match(r'visual\.transformer\.resblocks\.(\d+)\.attn\.parametrizations\.in_proj_weight\.0\.A_([qv])$', k)
    if not m:
        continue
    i, which = int(m.group(1)), m.group(2)
    B = ck['lora_state'][k.replace('.A_', '.B_')]
    delta = (B.float() @ A.float()) * lora_scaling * a.scale
    W = blocks[i].attn.in_proj_weight.data
    row0 = 0 if which == 'q' else 2 * d
    W[row0:row0 + d] += delta
    merged += 1
assert merged == len(ck['lora_state']) // 2, (merged, len(ck['lora_state']))
print(f'merged {merged} LoRA deltas (scaling {lora_scaling} x scale {a.scale})', flush=True)
model = model.to(DEV).eval()

imgidx = {}
for dp, _, fs in os.walk('data/dl/images'):
    for f in fs:
        if f.lower().endswith(('.jpg', '.jpeg', '.png')): imgidx[f] = os.path.join(dp, f)
files = list(pickle.load(open(f'data/dl/splits/{a.split}.pkl', 'rb')))
print(f'split={a.split} files={len(files)}', flush=True)


class DS(Dataset):
    def __init__(s, it): s.it = it
    def __len__(s): return len(s.it)
    def __getitem__(s, i):
        fn = s.it[i]
        try:
            return pp(Image.open(imgidx[fn]).convert('RGB')), fn
        except Exception:
            return torch.zeros(3, 224, 224), fn


dl = DataLoader(DS(files), batch_size=a.bs, num_workers=a.workers, pin_memory=True)
fs_, kept = [], []
t0 = time.time()
with torch.no_grad():
    for bi, (x, fns) in enumerate(dl):
        x = x.to(DEV)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            f = F.normalize(model.encode_image(x).float(), dim=-1)
            f2 = F.normalize(model.encode_image(torch.flip(x, dims=[-1])).float(), dim=-1)
        f = F.normalize(f + f2, dim=-1)
        fs_.append(f.cpu()); kept += list(fns)
        if bi % 50 == 0: print(f'{bi * a.bs}/{len(files)} {(time.time() - t0):.0f}s', flush=True)
torch.save({'files': kept, 'feats': torch.cat(fs_)}, a.out)
print('saved', a.out, 'n=', len(kept), flush=True)
