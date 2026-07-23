"""Embed a split with DINOv2 ViT-L/14 (general FM, allowed). Cache outputs/emb_<split>_dino.pt."""
import argparse, os, pickle, torch, timm
from PIL import Image
from timm.data import resolve_data_config, create_transform

def index_images(root):
    idx={}
    for dp,_,fs in os.walk(root):
        for f in fs:
            if f.lower().endswith((".jpg",".jpeg",".png")): idx[f]=os.path.join(dp,f)
    return idx

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--split",required=True)
    ap.add_argument("--imgroot",default="data/dl/images"); ap.add_argument("--batch",type=int,default=128)
    a=ap.parse_args()
    dev="cuda"
    model=timm.create_model("vit_large_patch14_dinov2.lvd142m",pretrained=True,num_classes=0).to(dev).eval()
    cfg=resolve_data_config({},model=model); tf=create_transform(**cfg)
    files=list(pickle.load(open(f"data/dl/splits/{a.split}.pkl","rb")))
    idx=index_images(a.imgroot)
    feats,kept,miss=[],[],0; buf,bufn=[],[]
    def flush():
        if not buf: return
        x=torch.stack(buf).to(dev)
        with torch.no_grad():
            f=model(x); f=torch.nn.functional.normalize(f,dim=-1)
        feats.append(f.cpu()); kept.extend(bufn); buf.clear(); bufn.clear()
    for i,fn in enumerate(files):
        p=idx.get(fn)
        if p is None: miss+=1; continue
        try: buf.append(tf(Image.open(p).convert("RGB"))); bufn.append(fn)
        except Exception: miss+=1; continue
        if len(buf)>=a.batch: flush()
        if i%5000==0: print(f"{a.split} {i}/{len(files)} miss={miss}",flush=True)
    flush()
    torch.save({"files":kept,"feats":torch.cat(feats)},f"outputs/emb_{a.split}_dino.pt")
    print(f"saved emb_{a.split}_dino.pt n={len(kept)} dim={feats[0].shape[1]}",flush=True)

if __name__=="__main__": main()
