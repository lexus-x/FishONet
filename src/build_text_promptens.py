"""Prompt-ensembled + taxonomy-context text embeddings for the UNSEEN route.

Current text_emb_h*.pt used a SINGLE template: "a photo of {name}." Standard CLIP practice is to
average embeddings over many templates (+2-5% absolute zero-shot). Also tries taxonomy CONTEXT in
the prompt ("a photo of {sp}, a fish of the family {fam}.") -- a different use of taxonomy than the
family-score prior (which failed: family top-1 23.8% < species 28.9%).
"""
import json, pickle, torch, argparse
import torch.nn.functional as F
import open_clip

MODEL='hf-hub:imageomics/bioclip-2.5-vith14'; DEV='cuda'
D='data/dl'
classes=list(pickle.load(open(f'{D}/all_classes.pkl','rb')))
tax=json.load(open('outputs/taxonomy_full.json'))
cn=json.load(open('outputs/common_names.json'))

TEMPLATES=[
    "a photo of {}.",
    "a photo of a {}, a type of fish.",
    "a close-up photo of a {}.",
    "an underwater photo of a {}.",
    "a photo of the fish {}.",
    "a specimen photo of a {}.",
    "a museum specimen of {}.",
    "a photograph of {}, a fish species.",
]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--mode',default='ens',choices=['ens','taxctx','both'])
    ap.add_argument('--out',default='outputs/text_emb_h_promptens.pt'); a=ap.parse_args()
    model,_,_=open_clip.create_model_and_transforms(MODEL)
    tok=open_clip.get_tokenizer(MODEL)
    model=model.to(DEV).eval()

    @torch.no_grad()
    def enc(texts,bs=256):
        out=[]
        for i in range(0,len(texts),bs):
            t=tok(texts[i:i+bs]).to(DEV)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                f=model.encode_text(t).float()
            out.append(F.normalize(f,dim=-1).cpu())
            if i%5120==0: print(f'  {i}/{len(texts)}',flush=True)
        return torch.cat(out)

    res={'classes':classes}
    if a.mode in ('ens','both'):
        acc=torch.zeros(len(classes),1024)
        for ti,tpl in enumerate(TEMPLATES):
            print(f'template {ti+1}/{len(TEMPLATES)}: {tpl}',flush=True)
            acc+=enc([tpl.format(c) for c in classes])
        res['emb_promptens']=F.normalize(acc,dim=-1)
    if a.mode in ('taxctx','both'):
        txts=[]
        for c in classes:
            t=tax.get(c) or {}
            fam=t.get('family'); gen=t.get('genus'); com=cn.get(c)
            s=f"a photo of {c}"
            if com: s+=f", commonly known as {com}"
            if fam: s+=f", a fish of the family {fam}"
            s+="."
            txts.append(s)
        print('encoding taxonomy-context prompts ...',flush=True)
        res['emb_taxctx']=enc(txts)
    torch.save(res,a.out)
    print('saved',a.out,list(res.keys()))

if __name__=='__main__': main()
