"""BioCLIP-native taxonomic-prompt A/B for the UNSEEN (zero-shot) route on ViT-H.
Encodes Kingdom->species flattened lineage prompts with BioCLIP-2.5 ViT-H, then evaluates
name vs taxon vs ensembles on the HARD sim unseen split (rarest-20% seen = pseudo-unseen).
Rules-legal: GBIF general taxonomy only, no fish-specific images."""
import json, pickle, torch, open_clip
from collections import defaultdict
import torch.nn.functional as F

MODEL = 'hf-hub:imageomics/bioclip-2.5-vith14'
D = 'data/dl'
dev = 'cuda'

classes = list(pickle.load(open(f'{D}/all_classes.pkl', 'rb')))
tax = json.load(open('outputs/taxonomy_full.json'))
cls_index = {c: i for i, c in enumerate(classes)}

RANKS = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus']
def lineage_str(c):
    t = tax.get(c, {})
    parts = [t.get(r) for r in RANKS]
    parts = [p for p in parts if p]  # drop missing ranks (class is often None)
    epithet = c.split()[1] if len(c.split()) > 1 else c
    # genus already in parts via 'genus'; append species epithet -> "... Genus epithet"
    return ' '.join(parts + [epithet])

name_prompts = [f'a photo of {c}.' for c in classes]
taxon_prompts = [f'a photo of {lineage_str(c)}.' for c in classes]
# sample print
for c in classes[:3]:
    print('NAME :', f'a photo of {c}.')
    print('TAXON:', f'a photo of {lineage_str(c)}.')

print('encoding taxon prompts on ViT-H...', flush=True)
model, _, _ = open_clip.create_model_and_transforms(MODEL)
tok = open_clip.get_tokenizer(MODEL)
model = model.to(dev).eval()
def encode(texts, bs=256):
    out = []
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        for i in range(0, len(texts), bs):
            t = tok(texts[i:i+bs]).to(dev)
            f = model.encode_text(t).float(); f = f / f.norm(dim=-1, keepdim=True)
            out.append(f.cpu().float())
    return torch.cat(out)
emb_taxon = encode(taxon_prompts)
torch.save({'classes': classes, 'emb_taxon': emb_taxon, 'model': MODEL},
           'outputs/text_emb_h_taxon.pt')
print('saved outputs/text_emb_h_taxon.pt', tuple(emb_taxon.shape), flush=True)

# load ViT-H name text emb (already built)
th = torch.load('outputs/text_emb_h.pt', weights_only=False)
emb_name = F.normalize(th['emb_name'].float(), dim=-1)
emb_taxon = F.normalize(emb_taxon.float(), dim=-1)
assert th['classes'] == classes, 'class order mismatch'

# ---- HARD sim unseen split on ViT-H train features ----
tr = torch.load('outputs/emb_train_h.pt', weights_only=False)
lab = json.load(open(f'{D}/label_train.json'))
by_cls = defaultdict(list)
for fn, f in zip(tr['files'], tr['feats']):
    if fn in lab and lab[fn] in cls_index:
        by_cls[lab[fn]].append(f)
seen = sorted(by_cls.keys())
order = sorted(seen, key=lambda c: len(by_cls[c]))
n_un = int(len(seen) * 0.2)
pseudo = set(order[:n_un]); known = [c for c in seen if c not in pseudo]
known_idx = set(cls_index[c] for c in known)
cand = [i for i in range(len(classes)) if i not in known_idx]
cand_pos = {ci: j for j, ci in enumerate(cand)}
cand_t = torch.tensor(cand)
qf, qy = [], []
for c in pseudo:
    for f in by_cls[c]:
        qf.append(f); qy.append(cls_index[c])
Fq = F.normalize(torch.stack(qf).float(), dim=-1)
gold = torch.tensor([cand_pos[y] for y in qy])
print(f'\npseudo_unseen={len(pseudo)} queries={len(qy)} candidates={len(cand)}', flush=True)

Tn = emb_name[cand_t]; Tt = emb_taxon[cand_t]
def t1(S): return (S.argmax(1) == gold).float().mean().item() * 100
def db(S): return (S - S.mean(0, keepdim=True)) / (S.std(0, keepdim=True) + 1e-6)
def recallK(S, Ks=(1,5,20,100)):
    top = S.topk(max(Ks),1).indices
    return {K: round((top[:,:K]==gold.unsqueeze(1)).any(1).float().mean().item()*100,1) for K in Ks}

Sn = Fq @ Tn.t(); St = Fq @ Tt.t()
print('\n=== RAW cosine top1 ===')
print(f'[name ]  {t1(Sn):.2f}   recall {recallK(Sn)}')
print(f'[taxon]  {t1(St):.2f}   recall {recallK(St)}')
print(f'[n+t  ]  {t1(0.5*(Sn+St)):.2f}   recall {recallK(0.5*(Sn+St))}')
print('\n=== DEBIASED (z-score col) top1 ===')
dn, dt = db(Sn), db(St)
print(f'[name ]  {t1(dn):.2f}')
print(f'[taxon]  {t1(dt):.2f}')
for w in (0.3, 0.5, 0.7, 1.0):
    print(f'[name+{w}*taxon] {t1(dn + w*dt):.2f}')
for w in (0.3, 0.5, 0.7):
    print(f'[taxon+{w}*name] {t1(dt + w*dn):.2f}')
print('\nBASELINE REF (ViT-H name+debias prior anchor) ~ 22.4 sim -> ~12.2 real')
