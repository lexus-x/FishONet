"""Native high-res (336px) LoRA FT for the SEEN route.

Unlike the frozen-336 pos-emb-interpolation gate that FAILED (nothing adapted to the distorted grid),
here we TRAIN at 336 so LoRA + ln_post + proj adapt to the interpolated positional structure.
Same recipe as ft_cap.py (ArcFace m + balanced-softmax + genus-aux, warm-start protos); resumable;
wall-clock safe. Ships a matching --extract at the SAME resolution (ft.py's extract is 224-only and
would mis-extract a 336-trained ckpt).

  train:   python src/ft_cap336.py --total_steps 2000 --res 336 --top_k_blocks 16 --rank 48 --alpha 96 \
             --bs 64 --state outputs/cap336_state.pt --out outputs/ft_cap336.pt --save_every 100 --val_every 200
  smoke:   python src/ft_cap336.py --total_steps 40 --bs 64 --val_every 999 --save_every 999 --state outputs/cap336_smoke.pt --out /tmp/x
  extract: python src/ft_cap336.py --extract train --ckpt outputs/ft_cap336.pt --out outputs/emb_train_cap336.pt --hflip 1
"""
import argparse, os, time, random, pickle
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import open_clip
from ft import (inject_lora, lora_state, load_lora_state, set_lora_scale, build_splits,
                frozen_baseline, eval_val, index_images, DIM, DEV, MODEL)


def cycle(loader):
    while True:
        for b in loader:
            yield b


class FishDS336(torch.utils.data.Dataset):
    def __init__(self, items, imgidx, tf, res):
        self.items = items; self.imgidx = imgidx; self.tf = tf; self.res = res
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        fn, sy, gy = self.items[i]
        try:
            return self.tf(Image.open(self.imgidx[fn]).convert('RGB')), sy, gy
        except Exception:
            return torch.zeros(3, self.res, self.res), sy, gy


def save_state(path, model, Wc, Wg, opt, sched, step, best, species, argsd):
    torch.save({'lora': lora_state(model), 'Wc': Wc.detach().cpu(),
                'Wg': (Wg.state_dict() if Wg is not None else None),
                'opt': opt.state_dict(), 'sched': sched.state_dict(),
                'step': step, 'best': best, 'species': species, 'args': argsd}, path + '.tmp')
    os.replace(path + '.tmp', path)


def save_deploy(path, model, Wc, species, argsd, acc):
    torch.save({'lora': lora_state(model), 'Wc': Wc.detach().cpu(),
                'species': species, 'args': argsd, 'val_acc': acc}, path + '.tmp')
    os.replace(path + '.tmp', path)


@torch.no_grad()
def extract(args):
    ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    a = ck.get('args', {}); res = a.get('res', 336)
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL, force_image_size=res)
    inject_lora(model, a.get('rank', 48), a.get('alpha', 96), a.get('top_k_blocks', 16))
    model = model.to(DEV).eval(); load_lora_state(model, ck['lora'])
    if abs(args.lora_scale - 1.0) > 1e-6:
        set_lora_scale(model, args.lora_scale); print(f'WiSE-FT lora_scale={args.lora_scale}')
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
    print('saved', args.out, 'n=', len(kept), 'dim=', feats[0].shape[1], 'res=', res, 'hflip=', args.hflip)


def train(args):
    os.makedirs('outputs', exist_ok=True); t_wall = time.time()
    D = build_splits(); C = len(D['species']); G = len(D['gset'])
    protos, base_acc = frozen_baseline(D)
    print(f'species={C} genera={G} train={len(D["train"])} val={len(D["val"])} FROZEN224 NCM={base_acc:.2f}%', flush=True)

    model, _, preprocess = open_clip.create_model_and_transforms(MODEL, force_image_size=args.res)
    for p in model.parameters(): p.requires_grad_(False)
    nlora = inject_lora(model, args.rank, args.alpha, args.top_k_blocks)
    for p in model.visual.ln_post.parameters(): p.requires_grad_(True)
    if getattr(model.visual, 'proj', None) is not None:
        model.visual.proj.requires_grad_(True)
    model = model.to(DEV)

    Wc = nn.Parameter(protos.clone().to(DEV))
    Wg = nn.Linear(DIM, G).to(DEV) if G > 0 else None

    mean = (0.48145466, 0.4578275, 0.40821073); std = (0.26862954, 0.26130258, 0.27577711)
    train_tf = T.Compose([
        T.RandomResizedCrop(args.res, scale=(0.5, 1.0), interpolation=T.InterpolationMode.BICUBIC),
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
    train_loader = torch.utils.data.DataLoader(FishDS336(D['train'], D['imgidx'], train_tf, args.res),
        batch_size=args.bs, sampler=sampler, num_workers=args.workers, pin_memory=True,
        drop_last=True, persistent_workers=True, prefetch_factor=4)
    val_loader = torch.utils.data.DataLoader(FishDS336(val_items, D['imgidx'], preprocess, args.res),
        batch_size=args.bs, shuffle=False, num_workers=args.workers, pin_memory=True)

    trainable = [p for p in model.parameters() if p.requires_grad] + [Wc] + (list(Wg.parameters()) if Wg else [])
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.total_steps, pct_start=0.1)

    step = 0; best = base_acc
    if os.path.exists(args.state):
        st = torch.load(args.state, map_location='cpu', weights_only=False)
        load_lora_state(model, st['lora']); Wc.data.copy_(st['Wc'].to(DEV))
        if Wg is not None and st.get('Wg'): Wg.load_state_dict(st['Wg'])
        opt.load_state_dict(st['opt']); sched.load_state_dict(st['sched'])
        step = st['step']; best = st['best']
        print(f'RESUMED step={step}/{args.total_steps} best={best:.2f}', flush=True)
    else:
        print(f'FRESH res={args.res}; LoRA pairs={nlora} top_k={args.top_k_blocks} rank={args.rank} '
              f'trainable={sum(p.numel() for p in trainable)/1e6:.2f}M', flush=True)

    it = cycle(train_loader); model.train(); t0 = time.time(); start_step = step; done = False
    while step < args.total_steps:
        x, sy, gy = next(it)
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
        if step % 10 == 0:
            r = (time.time() - t0) / max(step - start_step, 1)
            mem = torch.cuda.max_memory_allocated() / 1e9
            print(f'step {step}/{args.total_steps} loss={loss.item():.3f} lr={sched.get_last_lr()[0]:.2e} '
                  f'{r:.2f}s/it peakmem={mem:.1f}GB wall={time.time()-t_wall:.0f}s', flush=True)
        if step % args.val_every == 0:
            acc = eval_val(model, Wc, val_loader); model.train(); tag = ''
            if acc >= best:
                best = acc; save_deploy(args.out, model, Wc, D['species'], vars(args), acc); tag = ' *saved-best*'
            print(f'== step {step}: val NCM={acc:.2f}% (frozen224 {base_acc:.2f}, best {best:.2f}){tag} ==', flush=True)
        if step % args.save_every == 0:
            save_state(args.state, model, Wc, Wg, opt, sched, step, best, D['species'], vars(args))
        if time.time() - t_wall > args.wall_seconds:
            save_state(args.state, model, Wc, Wg, opt, sched, step, best, D['species'], vars(args))
            print(f'WALL_LIMIT step {step}/{args.total_steps} -> state saved, resume next window', flush=True)
            done = False; break
    else:
        done = True
    if done:
        acc = eval_val(model, Wc, val_loader)
        if acc >= best:
            best = acc; save_deploy(args.out, model, Wc, D['species'], vars(args), acc)
        save_state(args.state, model, Wc, Wg, opt, sched, step, best, D['species'], vars(args))
        print(f'TRAINING_COMPLETE steps={step} best_val_NCM={best:.2f}% (frozen224 {base_acc:.2f}, delta {best-base_acc:+.2f})', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--res', type=int, default=336)
    ap.add_argument('--total_steps', type=int, default=2000)
    ap.add_argument('--top_k_blocks', type=int, default=16)
    ap.add_argument('--rank', type=int, default=48); ap.add_argument('--alpha', type=int, default=96)
    ap.add_argument('--lr', type=float, default=5e-4); ap.add_argument('--wd', type=float, default=0.05)
    ap.add_argument('--bs', type=int, default=64); ap.add_argument('--workers', type=int, default=10)
    ap.add_argument('--scale', type=float, default=30.0); ap.add_argument('--margin', type=float, default=0.2)
    ap.add_argument('--gw', type=float, default=0.3); ap.add_argument('--sample_pow', type=float, default=0.0)
    ap.add_argument('--val_subset', type=int, default=1500)
    ap.add_argument('--wall_seconds', type=int, default=999999)
    ap.add_argument('--save_every', type=int, default=100)
    ap.add_argument('--val_every', type=int, default=200)
    ap.add_argument('--state', default='outputs/cap336_state.pt')
    ap.add_argument('--out', default='outputs/ft_cap336.pt')
    ap.add_argument('--extract', default=''); ap.add_argument('--ckpt', default='outputs/ft_cap336.pt')
    ap.add_argument('--hflip', type=int, default=1); ap.add_argument('--lora_scale', type=float, default=1.0)
    args = ap.parse_args()
    if args.extract: extract(args)
    else: train(args)


if __name__ == '__main__':
    main()
