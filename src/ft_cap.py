"""Resumable higher-capacity LoRA FT for the SEEN route (drives ft.py's recipe over flaky SSH).

Reuses ft.py's model/LoRA/data/eval verbatim (same ArcFace + balanced-softmax + genus-aux recipe);
adds: (1) higher capacity via --top_k_blocks/--rank, (2) full-state resume (--state), (3) wall-clock
limit (--wall_seconds) so each autonomous window trains a chunk, saves state, and EXITS CLEAN before
the SSH/WSL teardown, (4) step-based val + best-save so progress survives sub-epoch windows.

  chunk (autonomous, call repeatedly):
    python src/ft_cap.py --total_steps 2800 --top_k_blocks 16 --rank 48 --alpha 96 --bs 128 \
        --state outputs/cap_state.pt --out outputs/ft_cap.pt --wall_seconds 480 --save_every 100 --val_every 400
  extract (unchanged ft.py path):
    python src/ft.py --extract train --ckpt outputs/ft_cap.pt --out outputs/emb_train_cap.pt --hflip 1
"""
import argparse, os, time, random
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision.transforms as T
from ft import (inject_lora, lora_state, load_lora_state, build_splits, frozen_baseline,
                eval_val, FishDS, DIM, DEV, MODEL)
import open_clip


def cycle(loader):
    while True:
        for b in loader:
            yield b


def save_state(path, model, Wc, Wg, opt, sched, step, best, species, argsd):
    torch.save({'lora': lora_state(model), 'Wc': Wc.detach().cpu(),
                'Wg': (Wg.state_dict() if Wg is not None else None),
                'opt': opt.state_dict(), 'sched': sched.state_dict(),
                'step': step, 'best': best, 'species': species, 'args': argsd}, path + '.tmp')
    os.replace(path + '.tmp', path)  # atomic: never leave a half-written resume file


def save_deploy(path, model, Wc, species, argsd, acc):
    torch.save({'lora': lora_state(model), 'Wc': Wc.detach().cpu(),
                'species': species, 'args': argsd, 'val_acc': acc}, path + '.tmp')
    os.replace(path + '.tmp', path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--total_steps', type=int, default=2800)
    ap.add_argument('--top_k_blocks', type=int, default=16)
    ap.add_argument('--rank', type=int, default=48); ap.add_argument('--alpha', type=int, default=96)
    ap.add_argument('--lr', type=float, default=5e-4); ap.add_argument('--wd', type=float, default=0.05)
    ap.add_argument('--bs', type=int, default=128); ap.add_argument('--workers', type=int, default=10)
    ap.add_argument('--scale', type=float, default=30.0); ap.add_argument('--margin', type=float, default=0.2)
    ap.add_argument('--gw', type=float, default=0.3); ap.add_argument('--sample_pow', type=float, default=0.0)
    ap.add_argument('--val_subset', type=int, default=3000)
    ap.add_argument('--wall_seconds', type=int, default=480)   # exit clean before the ~10min SSH cap
    ap.add_argument('--save_every', type=int, default=100)     # resume granularity (~few min)
    ap.add_argument('--val_every', type=int, default=400)
    ap.add_argument('--state', default='outputs/cap_state.pt')
    ap.add_argument('--out', default='outputs/ft_cap.pt')
    args = ap.parse_args()
    os.makedirs('outputs', exist_ok=True)
    t_wall = time.time()

    D = build_splits(); C = len(D['species']); G = len(D['gset'])
    protos, base_acc = frozen_baseline(D)
    print(f'species={C} genera={G} train={len(D["train"])} val={len(D["val"])} FROZEN NCM={base_acc:.2f}%', flush=True)

    model, _, preprocess = open_clip.create_model_and_transforms(MODEL)
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
        print(f'FRESH start; LoRA pairs={nlora} top_k={args.top_k_blocks} rank={args.rank} '
              f'trainable={sum(p.numel() for p in trainable)/1e6:.2f}M', flush=True)

    it = cycle(train_loader)
    model.train(); t0 = time.time(); done = False
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
        if step % 25 == 0:
            print(f'step {step}/{args.total_steps} loss={loss.item():.3f} lr={sched.get_last_lr()[0]:.2e} '
                  f'{(time.time()-t0)/step:.2f}s/it wall={time.time()-t_wall:.0f}s', flush=True)
        if step % args.val_every == 0:
            acc = eval_val(model, Wc, val_loader); model.train()
            tag = ''
            if acc >= best:
                best = acc; save_deploy(args.out, model, Wc, D['species'], vars(args), acc); tag = ' *saved-best*'
            print(f'== step {step}: val NCM={acc:.2f}% (frozen {base_acc:.2f}, best {best:.2f}){tag} ==', flush=True)
        if step % args.save_every == 0:
            save_state(args.state, model, Wc, Wg, opt, sched, step, best, D['species'], vars(args))
        if time.time() - t_wall > args.wall_seconds:
            save_state(args.state, model, Wc, Wg, opt, sched, step, best, D['species'], vars(args))
            print(f'WALL_LIMIT hit at step {step}/{args.total_steps} -> state saved, resume next window', flush=True)
            done = False; break
    else:
        done = True
    if done:
        acc = eval_val(model, Wc, val_loader)
        if acc >= best:
            best = acc; save_deploy(args.out, model, Wc, D['species'], vars(args), acc)
        save_state(args.state, model, Wc, Wg, opt, sched, step, best, D['species'], vars(args))
        print(f'TRAINING_COMPLETE steps={step} best_val_NCM={best:.2f}% (frozen {base_acc:.2f}, delta {best-base_acc:+.2f})', flush=True)


if __name__ == '__main__':
    main()
