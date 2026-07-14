"""Shift-robust heavy-aug LoRA retrain of the SEEN route.

Same recipe as ft.py (LoRA on top-K blocks, ArcFace margin, balanced-softmax,
genus-aux multitask loss) reused via import -- only the augmentation pipeline
and the checkpoint save-gate differ:
  - heavy aug (RandAugment + jitter + grayscale + erasing + RRC scale 0.25)
    to close the holdout(87%)/real(79%) distribution-shift gap.
  - always-save: ft.py only saves a checkpoint once an epoch BEATS the frozen
    zero-shot baseline (best=base_acc); under heavy aug early epochs may never
    clear that bar, so nothing would ever be saved. Here best starts at -1.0,
    so every improving epoch is saved.

Checkpoints are extract-compatible with ft.py's --extract (same lora/Wc format):
  python src/ft.py --extract test --ckpt outputs/ft_robust_lora.pt --out outputs/emb_test_ftrobust.pt --hflip 1

  train:   python src/ft_robust.py --epochs 6 --top_k_blocks 12 --bs 128 --out outputs/ft_robust_lora.pt
  resume:  python src/ft_robust.py --resume outputs/ft_robust_lora.pt --out outputs/ft_robust_lora.pt
  smoke:   python src/ft_robust.py --epochs 1 --max_steps 30 --out /tmp/ft_robust_smoke.pt
"""
import argparse, os, random, time
import torch, torch.nn as nn, torch.nn.functional as F
import open_clip
import torchvision.transforms as T

from ft import DIM, DEV, MODEL, inject_lora, lora_state, load_lora_state, build_splits, frozen_baseline, eval_val, FishDS


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

    if args.resume:   # warm-start LoRA weights from a prior checkpoint; optimizer/scheduler stay fresh
        rck = torch.load(args.resume, map_location='cpu', weights_only=False)
        load_lora_state(model, rck['lora']); Wc.data.copy_(rck['Wc'].to(DEV))
        print(f'resumed LoRA weights from {args.resume} (val_acc was {rck.get("val_acc", "?")})', flush=True)

    mean = (0.48145466, 0.4578275, 0.40821073); std = (0.26862954, 0.26130258, 0.27577711)
    train_tf = T.Compose([
        T.RandomResizedCrop(224, scale=(0.25, 1.0), interpolation=T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(), T.RandAugment(num_ops=2, magnitude=9),
        T.ColorJitter(0.4, 0.4, 0.4), T.RandomGrayscale(p=0.2),
        T.ToTensor(), T.Normalize(mean, std), T.RandomErasing(p=0.25)])

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
          f'grad_ckpt={args.grad_ckpt} val={len(val_items)} aug=heavy', flush=True)
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.wd)
    total_steps = max(1, len(train_loader)) * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=total_steps, pct_start=0.1)

    best = -1.0; step = 0   # always-save: an epoch under heavy aug may never clear the frozen baseline
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
    ap.add_argument('--out', default='outputs/ft_robust_lora.pt')
    ap.add_argument('--resume', default='')  # warm-start LoRA weights from a prior checkpoint
    args = ap.parse_args()
    os.makedirs('outputs', exist_ok=True)
    train(args)


if __name__ == '__main__':
    main()

