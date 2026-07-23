"""Seen-route validation: BioCLIP vs DINOv2 vs blended ensemble, via per-class train holdout.
Measures whether adding DINOv2 image features to BioCLIP prototypes raises seen accuracy.
Baseline to beat: BioCLIP prototype NCM ~= 65.2% on this harness (matches real 66.5% seen)."""
import torch, json, argparse
import torch.nn.functional as F
from collections import defaultdict

def load_emb(path):
    d = torch.load(path, map_location='cpu')
    feats = F.normalize(d['feats'].float(), dim=-1)
    return {f: i for i, f in enumerate(d['files'])}, feats

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bioclip', default='outputs/emb_train.pt')
    ap.add_argument('--dino', default='outputs/emb_train_dino.pt')
    ap.add_argument('--holdout', type=float, default=0.2)
    a = ap.parse_args()

    labels = json.load(open('data/dl/label_train.json'))
    fb_idx, fb = load_emb(a.bioclip)
    fd_idx, fd = load_emb(a.dino)

    common = [f for f in fb_idx if f in fd_idx and f in labels]
    by_cls = defaultdict(list)
    for f in common:
        by_cls[labels[f]].append(f)

    train_files, val_files = [], []
    for cls, fs in by_cls.items():
        if len(fs) < 2:
            train_files += fs; continue
        fs = sorted(fs)
        k = min(max(1, round(a.holdout * len(fs))), len(fs) - 1)
        val_files += fs[-k:]; train_files += fs[:-k]

    classes = sorted(by_cls.keys())
    cls2i = {c: i for i, c in enumerate(classes)}
    C = len(classes)

    def build_protos(feats, fidx):
        proto = torch.zeros(C, feats.shape[1]); cnt = torch.zeros(C)
        for f in train_files:
            i = cls2i[labels[f]]; proto[i] += feats[fidx[f]]; cnt[i] += 1
        return F.normalize(proto / cnt.clamp(min=1).unsqueeze(1), dim=-1)

    pb = build_protos(fb, fb_idx); pd = build_protos(fd, fd_idx)
    vy = torch.tensor([cls2i[labels[f]] for f in val_files])
    vb = torch.stack([fb[fb_idx[f]] for f in val_files])
    vd = torch.stack([fd[fd_idx[f]] for f in val_files])
    sb = vb @ pb.t(); sd = vd @ pd.t()

    def acc(s): return (s.argmax(1) == vy).float().mean().item() * 100
    print(f"classes={C} train={len(train_files)} val={len(val_files)}")
    print(f"BioCLIP-only = {acc(sb):.2f}%")
    print(f"DINOv2-only  = {acc(sd):.2f}%")
    best = (0.0, 0.0)
    for t in range(0, 11):
        al = t / 10
        ac = acc(al * sb + (1 - al) * sd)
        print(f"  blend bioclip={al:.1f}: {ac:.2f}%")
        if ac > best[1]: best = (al, ac)
    print(f"BEST blend bioclip={best[0]:.1f} -> {best[1]:.2f}%  (baseline BioCLIP-only above)")

if __name__ == '__main__':
    main()
