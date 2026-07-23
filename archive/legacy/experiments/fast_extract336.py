import os, sys, pickle, time
import torch, torch.nn.functional as F
import open_clip
from PIL import Image
from torch.utils.data import Dataset, DataLoader

MODEL='hf-hub:imageomics/bioclip-2.5-vith14'; RES=336; DEV='cuda'
split=sys.argv[1]; out=sys.argv[2]; ckpt='outputs/fullft336.pt'
BS=int(os.environ.get('BS','128')); HFLIP=True
WORKERS=int(os.environ.get('EXWORKERS','6'))
torch.backends.cuda.matmul.allow_tf32=True
torch.backends.cudnn.allow_tf32=True
torch.backends.cudnn.benchmark=True

print(f'[{split}] loading model BS={BS} workers={WORKERS}', flush=True)
model,_,preprocess = open_clip.create_model_and_transforms(MODEL, force_image_size=RES)
ck=torch.load(ckpt, map_location='cpu', weights_only=False)
model.load_state_dict(ck['model']); model=model.to(DEV).eval()
del ck

def index_images(root):
    idx={}
    for dp,_,fs in os.walk(root):
        for f in fs:
            if f.lower().endswith(('.jpg','.jpeg','.png')): idx[f]=os.path.join(dp,f)
    return idx
imgidx=index_images('data/dl/images')
files=list(pickle.load(open(f'data/dl/splits/{split}.pkl','rb')))
present=[fn for fn in files if fn in imgidx]
print(f'[{split}] {len(present)}/{len(files)} images present', flush=True)

class DS(Dataset):
    def __init__(self, fns): self.fns=fns
    def __len__(self): return len(self.fns)
    def __getitem__(self, i):
        fn=self.fns[i]
        try: return preprocess(Image.open(imgidx[fn]).convert('RGB')), i
        except Exception: return torch.zeros(3,RES,RES), i

dl=DataLoader(DS(present), batch_size=BS, num_workers=WORKERS, pin_memory=True,
              shuffle=False, prefetch_factor=2, persistent_workers=False)

feats=torch.empty(len(present),1024,dtype=torch.float32); t0=time.time(); done=0
with torch.no_grad():
    for x, idxs in dl:
        x=x.to(DEV, non_blocking=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            f=F.normalize(model.encode_image(x).float(), dim=-1)
            if HFLIP:
                f2=F.normalize(model.encode_image(torch.flip(x, dims=[-1])).float(), dim=-1)
                f=F.normalize(f+f2, dim=-1)
        feats[idxs]=f.cpu().float()
        done+=len(idxs)
        print(f'[{split}] {done}/{len(present)} {done/(time.time()-t0):.0f} img/s', flush=True)
torch.save({'files':present, 'feats':feats}, out)
print(f'[{split}] SAVED {out} n={len(present)} dim={feats.shape[1]} in {(time.time()-t0)/60:.1f}m', flush=True)
