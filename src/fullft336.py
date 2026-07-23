"""Full-network fine-tune BioCLIP-2.5 ViT-H at 336px for SEEN-species feature learning.
No LoRA — all parameters trainable. Uses gradient checkpointing + bf16 to fit L40S 46GB.
Long-tail: balanced-softmax + ArcFace margin + genus-auxiliary multitask loss.
Deploy via NCM (same as LoRA variant).

  train:   python src/fullft336.py --epochs 4 --bs 16 --lr 1e-5 --out outputs/fullft336.pt
  extract: python src/fullft336.py --extract test --ckpt outputs/fullft336.pt --out outputs/emb_test_fullft336.pt --hflip 1
  smoke:   python src/fullft336.py --epochs 1 --max_steps 30 --out /tmp/fullft336_smoke.pt
"""
import argparse, os, json, math, random, time
from collections import defaultdict
import torch, torch.nn as nn, torch.nn.functional as F
import open_clip
from PIL import Image
import torchvision.transforms as T

MODEL = 'hf-hub:imageomics/bioclip-2.5-vith14'
DIM = 1024
RES = 336
DEV = 'cuda'

# ----------------------------- data -----------------------------
def index_images(root):
    idx = {}
    for dp, _, fs in os.walk(root):
        for f in fs:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                idx[f] = os.path.join(dp, f)
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
            return torch.zeros(3, RES, RES), sy, gy

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
    """NCM val-acc using cached FROZEN ViT-H 336px features on the SAME val split."""
    fr = torch.load('outputs/emb_train_h336.pt', weights_only=False)
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
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL, force_image_size=RES)
    ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    # Full model weights are stored under 'model'
    if 'model' in ck:
        model.load_state_dict(ck['model'])
    model = model.to(DEV).eval()
    files = list(pickle.load(open(f'data/dl/splits/{args.extract}.pkl', 'rb')))
    imgidx = index_images('data/dl/images')
    squash_tf = T.Compose([T.Resize((RES, RES), interpolation=T.InterpolationMode.BICUBIC),
                           T.ToTensor(),
                           T.Normalize((0.48145466, 0.4578275, 0.40821073),
                                       (0.26862954, 0.26130258, 0.27577711))])
    feats, kept, buf, buf2, bufn, miss = [], [], [], [], [], 0
    def enc(x):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            f = model.encode_image(x).float()
        return f / f.norm(dim=-1, keepdim=True)
    def flush():
        if not buf: return
        x = torch.stack(buf).to(DEV)
        f = enc(x)
        if args.hflip: f = f + enc(torch.flip(x, dims=[-1]))
        if args.squash_tta:
            x2 = torch.stack(buf2).to(DEV)
            f = f + enc(x2)
            if args.hflip: f = f + enc(torch.flip(x2, dims=[-1]))
        f = f / f.norm(dim=-1, keepdim=True)
        feats.append(f.cpu().float()); kept.extend(bufn); buf.clear(); buf2.clear(); bufn.clear()
    for i, fn in enumerate(files):
        p = imgidx.get(fn)
        if p is None: miss += 1; continue
        try:
            img = Image.open(p).convert('RGB')
            buf.append(preprocess(img))
            if args.squash_tta: buf2.append(squash_tf(img))
            bufn.append(fn)
        except Exception: miss += 1; continue
        if len(buf) >= args.bs: flush()
        if i % 5000 == 0: print(f'{args.extract} {i}/{len(files)} miss={miss}', flush=True)
    flush()
    torch.save({'files': kept, 'feats': torch.cat(feats)}, args.out)
    print('saved', args.out, 'n=', len(kept), 'dim=', feats[0].shape[1], 'hflip=', args.hflip, 'squash_tta=', args.squash_tta)

# ----------------------------- train ----------------------------
def train(args):
    print('building splits ...', flush=True)
    D = build_splits()
    C = len(D['species']); G = len(D['gset'])
    print(f'species={C} genera={G} train={len(D["train"])} val={len(D["val"])}', flush=True)
    protos, base_acc = frozen_baseline(D)
    print(f'FROZEN 336px baseline val NCM acc = {base_acc:.2f}%  (number to beat)', flush=True)

    model, _, preprocess = open_clip.create_model_and_transforms(MODEL, force_image_size=RES)
    # Full network trainable
    for p in model.parameters():
        p.requires_grad_(True)
    # Enable gradient checkpointing
    model.visual.set_grad_checkpointing(True)
    model = model.to(DEV)

    # Warm-start classifier from frozen prototypes
    Wc = nn.Parameter(protos.clone().to(DEV))
    Wg = nn.Linear(DIM, G).to(DEV) if G > 0 and args.gw > 0 else None

    mean = (0.48145466, 0.4578275, 0.40821073); std = (0.26862954, 0.26130258, 0.27577711)
    if args.shift_aug:
        # test-shift-matched aug: widen aspect ratio (test fish elongated to 2.12; default RRC ratio
        # 0.75-1.33 never simulates it). Extract with --squash_tta 1 to match inference framing.
        train_tf = T.Compose([
            T.RandomResizedCrop(RES, scale=(0.35, 1.0), ratio=(0.5, 2.0),
                                interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(), T.ColorJitter(0.2, 0.2, 0.2),
            T.ToTensor(), T.Normalize(mean, std)])
        print('SHIFT-MATCHED aug: RandomResizedCrop scale(0.35,1.0) ratio(0.5,2.0)', flush=True)
    else:
        train_tf = T.Compose([
            T.RandomResizedCrop(RES, scale=(0.5, 1.0), interpolation=T.InterpolationMode.BICUBIC),
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

    train_loader = torch.utils.data.DataLoader(
        FishDS(D['train'], D['imgidx'], train_tf), batch_size=args.bs,
        sampler=sampler, num_workers=args.workers, pin_memory=True,
        drop_last=True, persistent_workers=True, prefetch_factor=4)
    val_loader = torch.utils.data.DataLoader(
        FishDS(val_items, D['imgidx'], preprocess), batch_size=args.bs,
        shuffle=False, num_workers=args.workers, pin_memory=True)

    trainable = list(model.parameters()) + [Wc] + (list(Wg.parameters()) if Wg else [])
    print(f'Full-network FT: trainable={sum(p.numel() for p in trainable)/1e6:.2f}M '
          f'grad_ckpt=True bs={args.bs} val={len(val_items)}', flush=True)

    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.wd)
    total_steps = max(1, len(train_loader)) * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=total_steps, pct_start=0.1)

    # Gradient accumulation for effective larger batch size
    grad_accum = max(1, args.accum)

    best = base_acc; step = 0; t0 = time.time()
    for ep in range(args.epochs):
        model.train(); ep_start = time.time()
        for batch_idx, (x, sy, gy) in enumerate(train_loader):
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
                    if m.any():
                        loss = loss + args.gw * F.cross_entropy(Wg(f[m]), gy[m])
                loss = loss / grad_accum

            loss.backward()
            if (batch_idx + 1) % grad_accum == 0 or (batch_idx + 1) == len(train_loader):
                opt.step(); opt.zero_grad(set_to_none=True)
                sched.step()
            step += 1

            if step % 50 == 0:
                elapsed = time.time() - t0
                print(f'ep{ep} step{step}/{total_steps} loss={loss.item()*grad_accum:.3f} '
                      f'lr={sched.get_last_lr()[0]:.2e} {elapsed/step:.2f}s/it {elapsed/60:.1f}m_elapsed', flush=True)
            if args.max_steps and step >= args.max_steps:
                print('max_steps (smoke)', flush=True); break

        acc = eval_val(model, Wc, val_loader)
        ep_time = (time.time() - ep_start) / 60
        print(f'== epoch {ep}: val NCM acc = {acc:.2f}%  (frozen336={base_acc:.2f}, best={best:.2f}, epoch_time={ep_time:.1f}m) ==', flush=True)
        if acc >= best:
            best = acc
            torch.save({'model': model.state_dict(), 'Wc': Wc.detach().cpu(),
                        'species': D['species'], 'args': vars(args), 'val_acc': acc,
                        'res': RES}, args.out)
            print(f'  saved {args.out} (val {acc:.2f}%)', flush=True)
        if args.max_steps and step >= args.max_steps: break

    print(f'DONE best val NCM acc = {best:.2f}% (frozen336 {base_acc:.2f}%, delta {best-base_acc:+.2f})', flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=4)
    ap.add_argument('--lr', type=float, default=1e-5)
    ap.add_argument('--wd', type=float, default=0.05)
    ap.add_argument('--bs', type=int, default=16)
    ap.add_argument('--accum', type=int, default=2, help='gradient accumulation steps')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--scale', type=float, default=30.0)
    ap.add_argument('--margin', type=float, default=0.3)
    ap.add_argument('--gw', type=float, default=0.3)
    ap.add_argument('--sample_pow', type=float, default=0.0)
    ap.add_argument('--val_subset', type=int, default=3000)
    ap.add_argument('--max_steps', type=int, default=0)
    ap.add_argument('--out', default='outputs/fullft336.pt')
    ap.add_argument('--ckpt', default='outputs/fullft336.pt')
    ap.add_argument('--extract', default='')
    ap.add_argument('--hflip', type=int, default=1)
    ap.add_argument('--shift_aug', type=int, default=0)   # 1 = test-shift-matched wide-aspect aug
    ap.add_argument('--squash_tta', type=int, default=0)  # 1 = add squash-view TTA (match shift_aug)
    args = ap.parse_args()
    os.makedirs('outputs', exist_ok=True)
    if args.extract:
        extract(args)
    else:
        train(args)

if __name__ == '__main__':
    main()