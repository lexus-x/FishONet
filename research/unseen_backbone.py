"""Backbone ranking test on the pseudo-unseen proxy (HANDOFF §8 item 1).

Question: does a stronger general image-text backbone beat BioCLIP-2.5 on ZERO-SHOT
species matching? `b` (unseen ~15.9% real) is the binding constraint on the score.

Apples-to-apples: both models FROZEN, same taxonomy-context prompts, same pseudo-unseen
queries (rarest-20% seen classes), same candidate set, same DBNorm. This ranks backbones;
absolute numbers overstate real (HANDOFF §7) — a winner still needs a real submission.

  python research/unseen_backbone.py [--model hf-hub:timm/ViT-SO400M-16-SigLIP2-384]
"""
import argparse
import json
import os
import pickle
import sys
import time

import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import FishData, dbnorm, OUT, DATA  # noqa: E402

IMG = os.path.join(DATA, 'dl', 'images')


def prompts(classes, mode):
    """taxctx = src/build_text_promptens.py template. common = common name first —
    general (non-bio) backbones learned popular names, not binomial nomenclature."""
    tax = json.load(open(os.path.join(OUT, 'taxonomy_full.json')))
    cn = json.load(open(os.path.join(OUT, 'common_names.json')))
    out = []
    for c in classes:
        t = tax.get(c) or {}
        if mode == 'common':
            s = f'a photo of a {cn[c]}' if cn.get(c) else f'a photo of {c}'
            s += f", a fish of the family {t['family']}" if t.get('family') else ', a fish'
        else:
            s = f'a photo of {c}'
            if cn.get(c):
                s += f", commonly known as {cn[c]}"
            if t.get('family'):
                s += f", a fish of the family {t['family']}"
        out.append(s + '.')
    return out


def top1(S, gold):
    return (S.argmax(1) == gold).float().mean().item() * 100


def encode(model_name, texts, files, tcache, icache, bs_t=256, bs_i=64):
    """Frozen zero-shot encode. Text and image caches are separate so swapping the
    prompt template does not force a re-encode of the images."""
    T = Q = None
    if os.path.exists(tcache):
        d = torch.load(tcache, weights_only=False)
        if d['texts'] == texts:
            T, = (d['T'],)
            print(f'cache hit: {tcache}', flush=True)
    if os.path.exists(icache):
        d = torch.load(icache, weights_only=False)
        if d['files'] == files:
            Q = d['Q']
            print(f'cache hit: {icache}', flush=True)
    if T is not None and Q is not None:
        return T, Q
    import open_clip
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, _, pre = open_clip.create_model_and_transforms(model_name)
    tok = open_clip.get_tokenizer(model_name)
    model = model.to(dev).eval()
    t0 = time.time()

    @torch.no_grad()
    def run(items, bs, fn):
        out = []
        for i in range(0, len(items), bs):
            with torch.autocast('cuda', dtype=torch.bfloat16):
                f = fn(items[i:i + bs]).float()
            out.append(F.normalize(f, dim=-1).cpu())
            if i % (bs * 20) == 0:
                print(f'  {i}/{len(items)} [{time.time()-t0:.0f}s]', flush=True)
        return torch.cat(out)

    if T is None:
        print(f'encoding {len(texts)} prompts ...', flush=True)
        T = run(texts, bs_t, lambda b: model.encode_text(tok(b).to(dev)))
        torch.save({'T': T, 'texts': texts}, tcache)
    if Q is None:
        print(f'encoding {len(files)} images ...', flush=True)
        Q = run(files, bs_i, lambda b: model.encode_image(
            torch.stack([pre(Image.open(os.path.join(IMG, f)).convert('RGB')) for f in b]).to(dev)))
        torch.save({'Q': Q, 'files': files}, icache)
    print(f'wrote {tcache} / {icache}', flush=True)
    return T, Q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='hf-hub:timm/ViT-SO400M-16-SigLIP2-384')
    ap.add_argument('--tag', default='siglip2')
    ap.add_argument('--prompts', default='taxctx', choices=['taxctx', 'common'])
    a = ap.parse_args()

    D = FishData()
    classes = list(pickle.load(open(os.path.join(DATA, 'dl', 'all_classes.pkl'), 'rb')))
    assert classes == D.classes, 'class order drift between all_classes.pkl and text_emb_h.pt'
    cand_cls = [D.classes[i] for i in D.cand.tolist()]

    files, gold = [], []
    for c in D.pseudo:
        for fn in D.by[c]:
            files.append(fn)
            gold.append(D.cand_pos[D.ci[c]])
    gold = torch.tensor(gold)
    print(f'pseudo-unseen: {len(D.pseudo)} classes, {len(files)} images, '
          f'{len(cand_cls)} candidates', flush=True)

    # --- baseline: frozen BioCLIP-2.5 ViT-H, same prompts (emb_taxctx) ---
    txt = torch.load(os.path.join(OUT, 'text_emb_h_promptens.pt'), weights_only=False)
    Tb = F.normalize(txt['emb_taxctx'].float(), dim=-1)[D.cand]
    d = torch.load(os.path.join(OUT, 'emb_train_h.pt'), weights_only=False)
    idx = {fn: i for i, fn in enumerate(d['files'])}
    feats = F.normalize(d['feats'].float(), dim=-1)
    keep = [i for i, fn in enumerate(files) if fn in idx]
    Qb = torch.stack([feats[idx[files[i]]] for i in keep])
    gold_b = gold[keep]
    Sb = dbnorm(Qb @ Tb.t())
    a_base = top1(Sb, gold_b)
    # sanity ref: unseen_legs.py `alone_h` = 23.38 (same frozen leg, bare taxon prompts).
    # The 28.9 in HANDOFF is the DEPLOYED route (ctftbig + L + fullft336_v2), not this comparator.
    print(f'\nBioCLIP-2.5-H frozen @ taxctx : {a_base:.2f}  (n={len(keep)}, ref ~23.4)', flush=True)

    # --- candidate backbone ---
    Tn, Qn = encode(a.model, prompts(cand_cls, a.prompts), files,
                    os.path.join(OUT, f'unseen_backbone_{a.tag}_txt_{a.prompts}.pt'),
                    os.path.join(OUT, f'unseen_backbone_{a.tag}_img.pt'))
    Sn = dbnorm(Qn @ Tn.t())
    a_new = top1(Sn, gold)
    print(f'{a.model} frozen @ {a.prompts} : {a_new:.2f}  (delta {a_new-a_base:+.2f})', flush=True)

    # --- fusion (both legs, equal weight on the shared subset) ---
    a_fuse = top1(Sb + dbnorm(Qn[keep] @ Tn.t()), gold_b)
    print(f'fused (BioCLIP + {a.tag})    : {a_fuse:.2f}  (delta {a_fuse-a_base:+.2f})', flush=True)

    res = {'baseline_bioclip_h_taxctx': a_base, a.tag: a_new, 'fused': a_fuse,
           'model': a.model, 'prompts': a.prompts,
           'n_images': len(files), 'n_cand': len(cand_cls)}
    p = os.path.join(OUT, f'unseen_backbone_{a.tag}_{a.prompts}_results.json')
    json.dump(res, open(p, 'w'), indent=1)
    print('wrote', p, flush=True)


if __name__ == '__main__':
    main()
