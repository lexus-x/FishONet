"""LoRA fine-tune BioCLIP-2.5 ViT-H image tower for closed-set SEEN-species feature learning.
Adapts MLP layers (c_fc/c_proj) of the TOP-K blocks via low-rank adapters; warm-starts a cosine
classifier head from the frozen prototypes (75% baseline) so FT can only sharpen from a strong start.
Long-tail: balanced-softmax + ArcFace margin + genus-auxiliary multitask loss. Deploy via NCM.

SPEED: with LoRA only on the top-K blocks and a non-grad image input, the frozen bottom blocks run
GRAPH-FREE (forward-only, no stored activations, no backward) -> faster + low memory, no grad-checkpoint needed.

  train:   python src/ft.py --epochs 6 --top_k_blocks 12 --bs 128 --out outputs/ft_lora.pt
  extract: python src/ft.py --extract test --ckpt outputs/ft_lora.pt --out outputs/emb_test_ft.pt --hflip 1
  smoke:   python src/ft.py --epochs 1 --max_steps 30 --out /tmp/ft_smoke.pt
"""
import argparse, os, json, math, random
from collections import defaultdict
import torch, torch.nn as nn, torch.nn.functional as F
import open_clip
from PIL import Image
import torchvision.transforms as T

MODEL = 'hf-hub:imageomics/bioclip-2.5-vith14'
DIM = 1024
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'

# ----------------------------- LoRA -----------------------------
class LoRALinear(nn.Module):
    def __init__(self, base, r=16, alpha=32, dropout=0.05):
        super().__init__()
        self.base = base
        for p in self.base.parameters(): p.requires_grad_(False)
        self.r = r; self.scaling = alpha / max(r, 1)
        self.A = nn.Parameter(torch.zeros(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.drop = nn.Dropout(dropout)
        self.lora_scale = 1.0   # WiSE-FT alpha: 0=frozen, 1=full FT
    def forward(self, x):
        return self.base(x) + (self.drop(x) @ self.A.t() @ self.B.t()) * self.scaling * self.lora_scale

def inject_lora(model, r, alpha, top_k=0):
    blocks = model.visual.transformer.resblocks
    sel = range(len(blocks)) if top_k <= 0 else range(len(blocks) - top_k, len(blocks))
    n = 0
    for i in sel:
        blk = blocks[i]
        blk.mlp.c_fc = LoRALinear(blk.mlp.c_fc, r, alpha)
        blk.mlp.c_proj = LoRALinear(blk.mlp.c_proj, r, alpha)
        n += 2
    return n

def set_lora_scale(model, s):
    for blk in model.visual.transformer.resblocks:
        for mod in (blk.mlp.c_fc, blk.mlp.c_proj):
            if isinstance(mod, LoRALinear): mod.lora_scale = s

def lora_state(model):
    sd = {}
    for i, blk in enumerate(model.visual.transformer.resblocks):
        for tag, mod in (('c_fc', blk.mlp.c_fc), ('c_proj', blk.mlp.c_proj)):
            if isinstance(mod, LoRALinear):
                sd[f'b{i}.{tag}.A'] = mod.A.detach().cpu()
                sd[f'b{i}.{tag}.B'] = mod.B.detach().cpu()
    sd['ln_post.weight'] = model.visual.ln_post.weight.detach().cpu()
    sd['ln_post.bias'] = model.visual.ln_post.bias.detach().cpu()
    if getattr(model.visual, 'proj', None) is not None:
        sd['proj'] = model.visual.proj.detach().cpu()
    return sd

def load_lora_state(model, sd):
    for i, blk in enumerate(model.visual.transformer.resblocks):
        for tag, mod in (('c_fc', blk.mlp.c_fc), ('c_proj', blk.mlp.c_proj)):
            if isinstance(mod, LoRALinear) and f'b{i}.{tag}.A' in sd:
                mod.A.data.copy_(sd[f'b{i}.{tag}.A'].to(mod.A.device))
                mod.B.data.copy_(sd[f'b{i}.{tag}.B'].to(mod.B.device))
    model.visual.ln_post.weight.data.copy_(sd['ln_post.weight'].to(DEV))
    model.visual.ln_post.bias.data.copy_(sd['ln_post.bias'].to(DEV))
    if 'proj' in sd and getattr(model.visual, 'proj', None) is not None:
        model.visual.proj.data.copy_(sd['proj'].to(DEV))

# ----------------------------- data -----------------------------
def index_images(root):
    idx = {}
    for dp, _, fs in os.walk(root):
        for f in fs:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')): idx[f] = os.path.join(dp, f)
    return idx

class FishDS(torch.utils.data.Dataset):
    def __init__(self, items, imgidx, tf):
        self.items = items; self.imgidx = imgidx; self.tf = tf
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        fn, sy, gy = self.items[i]
        try:
            return self.tf(Image.open(self.imgidx[fn]).convert('RGB')), sy, gy
        except Exception:
            return torch.zeros(3, 224, 224), sy, gy

def build_splits(genus_path='outputs/genus_train.json'):
    lab = json.load(open('data/dl/label_train.json'))
    genus = json.load(open(genus_path)) if os.path.exists(genus_path) else {}
    species = sorted(set(lab.values())); s2i = {c: i for i, c in enumerate(species)}
    gset = sorted(set(genus.values())) if genus else []; g2i = {c: i for i, c in enumerate(gset)}
    imgidx = index_images('data/dl/images')
    by = defaultdict(list)
    for fn, sp in lab.items():
        if fn in imgidx: by[sp].append(fn)
    train_items, val_items = [], []
    for sp, fns in by.items():
        fns = sorted(fns); si = s2i[sp]
        if len(fns) >= 3:
            k = max(1, round(0.2 * len(fns))); tr, va = fns[:-k], fns[-k:]
        else:
            tr, va = fns, []
        for fn in tr: train_items.append((fn, si, g2i.get(genus.get(fn, ''), -1)))
        for fn in va: val_items.append((fn, si, g2i.get(genus.get(fn, ''), -1)))
    return dict(lab=lab, species=species, s2i=s2i, gset=gset, g2i=g2i,
                imgidx=imgidx, train=train_items, val=val_items)

def frozen_baseline(D):
    """Exact control: NCM val-acc using cached frozen ViT-H features on the SAME val split."""
    fr = torch.load('outputs/emb_train_h_tta.pt', weights_only=False)
    ff = F.normalize(fr['feats'].float(), dim=1)
    fmap = {fn: ff[i] for i, fn in enumerate(fr['files'])}
    C = len(D['species']); protos = torch.zeros(C, DIM); cnt = torch.zeros(C)
    for fn, si, gy in D['train']:
        if fn in fmap: protos[si] += fmap[fn]; cnt[si] += 1
    protos = F.normalize(protos / cnt.clamp(min=1).unsqueeze(1), dim=1)
    vf, vy = [], []
    for fn, si, gy in D['val']:
        if fn in fmap: vf.append(fmap[fn]); vy.append(si)
    if not vf: return protos, 0.0
    acc = ((torch.stack(vf) @ protos.t()).argmax(1) == torch.tensor(vy)).float().mean().item() * 100
    return protos, acc

@torch.no_grad()
def eval_val(model, Wc, val_loader):
    model.eval(); Wn = F.normalize(Wc, dim=1); correct = tot = 0
    for x, sy, gy in val_loader:
        x = x.to(DEV, non_blocking=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            f = model.encode_image(x)
        pred = (F.normalize(f.float(), dim=-1) @ Wn.t()).argmax(1).cpu()
        correct += (pred == sy).sum().item(); tot += len(sy)
    return 100.0 * correct / max(tot, 1)

# --------------------------- extract ----------------------------
@torch.no_grad()
def extract(args):
    import pickle
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL)
    ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    a = ck.get('args', {})
    inject_lora(model, a.get('rank', 16), a.get('alpha', 32), a.get('top_k_blocks', 0))
    # snapshot frozen ln_post/proj for WiSE-FT weight interpolation
    w0 = model.visual.ln_post.weight.detach().clone(); b0 = model.visual.ln_post.bias.detach().clone()
    proj0 = model.visual.proj.detach().clone() if getattr(model.visual, 'proj', None) is not None else None
    model = model.to(DEV).eval(); load_lora_state(model, ck['lora'])
    if abs(args.lora_scale - 1.0) > 1e-6:   # WiSE-FT: theta = (1-a)*frozen + a*FT
        al = args.lora_scale; set_lora_scale(model, al)
        model.visual.ln_post.weight.data.copy_(((1-al)*w0 + al*model.visual.ln_post.weight.data.cpu()).to(DEV))
        model.visual.ln_post.bias.data.copy_(((1-al)*b0 + al*model.visual.ln_post.bias.data.cpu()).to(DEV))
        if proj0 is not None:
            model.visual.proj.data.copy_(((1-al)*proj0 + al*model.visual.proj.data.cpu()).to(DEV))
        print(f'WiSE-FT lora_scale (alpha) = {al}')
    files = list(pickle.load(open(f'data/dl/splits/{args.extract}.pkl', 'rb')))
    imgidx = index_images('data/dl/images')
    feats, kept, buf, bufn, miss = [], [], [], [], 0
    def flush():
        if not buf: return
        x = torch.stack(buf).to(DEV)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            f = model.encode_image(x).float(); f = f / f.norm(dim=-1, keepdim=True)
            if args.hflip:
                f2 = model.encode_image(torch.flip(x, dims=[-1])).float(); f2 = f2 / f2.norm(dim=-1, keepdim=True)
                f = f + f2; f = f / f.norm(dim=-1, keepdim=True)
        feats.append(f.cpu().float()); kept.extend(bufn); buf.clear(); bufn.clear()
    for i, fn in enumerate(files):
        p = imgidx.get(fn)
        if p is None: miss += 1; continue
        try: buf.append(preprocess(Image.open(p).convert('RGB'))); bufn.append(fn)
        except Exception: miss += 1; continue
        if len(buf) >= args.bs: flush()
        if i % 5000 == 0: print(f'{args.extract} {i}/{len(files)} miss={miss}', flush=True)
    flush()
    torch.save({'files': kept, 'feats': torch.cat(feats)}, args.out)
    print('saved', args.out, 'n=', len(kept), 'dim=', feats[0].shape[1], 'hflip=', args.hflip)

# ----------------------------- train ----------------------------
def train(args):
    print('building splits ...', flush=True)
    D = build_splits()
    C = len(D['species']); G = len(D['gset'])
    print(f'species={C} genera={G} train={len(D["train"])} val={len(D["val"])}', flush=True)
    protos, base_acc = frozen_baseline(D)
    print(f'FROZEN baseline val NCM acc = {base_acc:.2f}%  (number to beat)', flush=True)

    model, _, preprocess = open_clip.create_model_and_transforms(MODEL)
    for p in model.parameters(): p.requires_grad_(False)
    nlora = inject_lora(model, args.rank, args.alpha, args.top_k_blocks)
    for p in model.visual.ln_post.parameters(): p.requires_grad_(True)
    if getattr(model.visual, 'proj', None) is not None and args.train_proj:
        model.visual.proj.requires_grad_(True)
    model = model.to(DEV)
    if args.grad_ckpt:
        try: model.visual.set_grad_checkpointing(True)
        except Exception as e: print('grad ckpt not set:', e)

    Wc = nn.Parameter(protos.clone().to(DEV))
    Wg = nn.Linear(DIM, G).to(DEV) if G > 0 else None

    mean = (0.48145466, 0.4578275, 0.40821073); std = (0.26862954, 0.26130258, 0.27577711)
    train_tf = T.Compose([
        T.RandomResizedCrop(224, scale=(0.5, 1.0), interpolation=T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(), T.ColorJitter(0.2, 0.2, 0.2),
        T.ToTensor(), T.Normalize(mean, std)])

    freq = torch.zeros(C)
    for fn, si, gy in D['train']: freq[si] += 1
    logprior = torch.log((freq / freq.sum()).clamp(min=1e-12)).to(DEV)
    sw = (1.0 / freq.clamp(min=1))[torch.tensor([it[1] for it in D['train']])] ** args.sample_pow
    sampler = torch.utils.data.WeightedRandomSampler(sw.double(), num_samples=len(D['train']), replacement=True)

    val_items = D['val']
    if args.val_subset and len(val_items) > args.val_subset:
        random.seed(0); val_items = random.sample(val_items, args.val_subset)
    train_loader = torch.utils.data.DataLoader(FishDS(D['train'], D['imgidx'], train_tf), batch_size=args.bs,
                                               sampler=sampler, num_workers=args.workers, pin_memory=True,
                                               drop_last=True, persistent_workers=True, prefetch_factor=4)
    val_loader = torch.utils.data.DataLoader(FishDS(val_items, D['imgidx'], preprocess), batch_size=args.bs,
                                             shuffle=False, num_workers=args.workers, pin_memory=True)

    trainable = [p for p in model.parameters() if p.requires_grad] + [Wc] + (list(Wg.parameters()) if Wg else [])
    print(f'LoRA pairs={nlora} (top_k={args.top_k_blocks}) trainable={sum(p.numel() for p in trainable)/1e6:.2f}M '
          f'grad_ckpt={args.grad_ckpt} val={len(val_items)}', flush=True)
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.wd)
    total_steps = max(1, len(train_loader)) * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=total_steps, pct_start=0.1)

    best = base_acc; step = 0; import time
    for ep in range(args.epochs):
        model.train(); t0 = time.time()
        for x, sy, gy in train_loader:
            x = x.to(DEV, non_blocking=True); sy = sy.to(DEV); gy = gy.to(DEV)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                f = F.normalize(model.encode_image(x).float(), dim=-1)
                cos = f @ F.normalize(Wc, dim=1).t()
                if args.margin > 0:
                    th = torch.acos(cos.clamp(-1 + 1e-6, 1 - 1e-6))
                    cos = torch.where(F.one_hot(sy, C).bool(), torch.cos(th + args.margin), cos)
                loss = F.cross_entropy(args.scale * cos + logprior, sy)
                if Wg is not None:
                    m = gy >= 0
                    if m.any(): loss = loss + args.gw * F.cross_entropy(Wg(f[m]), gy[m])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step(); step += 1
            if step % 50 == 0:
                print(f'ep{ep} step{step}/{total_steps} loss={loss.item():.3f} '
                      f'lr={sched.get_last_lr()[0]:.2e} {(time.time()-t0)/ (step-ep*len(train_loader)):.2f}s/it', flush=True)
            if args.max_steps and step >= args.max_steps: print('max_steps (smoke)', flush=True); break
        acc = eval_val(model, Wc, val_loader)
        print(f'== epoch {ep}: val NCM acc = {acc:.2f}%  (frozen={base_acc:.2f}, best={best:.2f}, epoch_time={(time.time()-t0)/60:.1f}m) ==', flush=True)
        if acc >= best:
            best = acc
            torch.save({'lora': lora_state(model), 'Wc': Wc.detach().cpu(),
                        'species': D['species'], 'args': vars(args), 'val_acc': acc}, args.out)
            print(f'  saved {args.out} (val {acc:.2f}%)', flush=True)
        if args.max_steps and step >= args.max_steps: break
    print(f'DONE best val NCM acc = {best:.2f}% (frozen {base_acc:.2f}%, delta {best-base_acc:+.2f})', flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=6)
    ap.add_argument('--top_k_blocks', type=int, default=12)
    ap.add_argument('--rank', type=int, default=16); ap.add_argument('--alpha', type=int, default=32)
    ap.add_argument('--lr', type=float, default=5e-4); ap.add_argument('--wd', type=float, default=0.05)
    ap.add_argument('--bs', type=int, default=128); ap.add_argument('--workers', type=int, default=10)
    ap.add_argument('--scale', type=float, default=30.0); ap.add_argument('--margin', type=float, default=0.2)
    ap.add_argument('--gw', type=float, default=0.3); ap.add_argument('--sample_pow', type=float, default=0.0)
    ap.add_argument('--train_proj', type=int, default=1); ap.add_argument('--grad_ckpt', type=int, default=0)
    ap.add_argument('--val_subset', type=int, default=3000); ap.add_argument('--max_steps', type=int, default=0)
    ap.add_argument('--out', default='outputs/ft_lora.pt'); ap.add_argument('--ckpt', default='outputs/ft_lora.pt')
    ap.add_argument('--extract', default=''); ap.add_argument('--hflip', type=int, default=1)
    ap.add_argument('--lora_scale', type=float, default=1.0)  # WiSE-FT alpha for extract
    args = ap.parse_args()
    os.makedirs('outputs', exist_ok=True)
    if args.extract: extract(args)
    else: train(args)

if __name__ == '__main__':
    main()
