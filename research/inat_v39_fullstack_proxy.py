"""v39 proxy with deployed TaxaBind *text* in S0 + ctft/TB image protos; joint weight CV."""
import json, os, random, sys
import torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import FishData, dbnorm, load_emb, OUT

def top1(S,g): return (S.argmax(1)==g).float().mean().item()*100

D=FishData(); cand=D.cand
def queries(idx,feats):
    q,y=[],[]
    for c in D.pseudo:
        for fn in D.by[c]:
            if fn in idx: q.append(feats[idx[fn]]); y.append(D.cand_pos[D.ci[c]])
    return torch.stack(q), torch.tensor(y)

TtH_c=D.TtH[cand]; TnL_c=D.TnL[cand]
TTX=F.normalize(torch.load(os.path.join(OUT,'text_emb_h_promptens.pt'),weights_only=False)['emb_taxctx'].float(),dim=-1)[cand]
Ttb=F.normalize(torch.load(os.path.join(OUT,'text_emb_taxabind_taxctx.pt'),weights_only=False)['emb_taxctx'].float(),dim=-1)[cand]
legs={}
for tag in ['ctftshift','ftshift','fullft336_v2']:
    idx,feats,_=load_emb(os.path.join(OUT,f'emb_train_{tag}.pt'))
    q,gold=queries(idx,feats); legs[tag]=q
qL,gold=queries(D.LtI,D.LtF)
tb_idx,tb_feat,_=load_emb(os.path.join(OUT,'emb_train_taxabind.pt'))
Qtb,_=queries(tb_idx,tb_feat)
S0=(dbnorm(legs['ctftshift']@TtH_c.t())+0.5*dbnorm(qL@TnL_c.t())+0.75*dbnorm(legs['fullft336_v2']@TtH_c.t())
    +1.0*dbnorm(legs['ftshift']@TtH_c.t())+1.0*dbnorm(legs['ctftshift']@TTX.t())+1.0*dbnorm(Qtb@Ttb.t()))
Pin=F.normalize(torch.load(os.path.join(OUT,'inat_protos_ctftshift_full.pt'),weights_only=False)['protos'].float(),dim=-1)[cand]
has=Pin.norm(dim=-1)>0.5
Sic=dbnorm((legs['ctftshift']@Pin.t()).masked_fill(~has.unsqueeze(0),-1e4))
Ptb=F.normalize(torch.load(os.path.join(OUT,'inat_protos_taxabind.pt'),weights_only=False)['protos'].float(),dim=-1)[cand]
Stb=dbnorm((Qtb@Ptb.t()).masked_fill(~(Ptb.norm(dim=-1)>0.5).unsqueeze(0),-1e4))
ref=top1(S0+3*Sic+0.5*Stb,gold)
print(f'full-stack v39 ref {ref:.2f}',flush=True)
best=ref; bcfg=(3,.5)
for wc in [2.5,3,3.5]:
  for wt in [0,0.25,0.5,0.75,1.0]:
    a=top1(S0+wc*Sic+wt*Stb,gold)
    if a>best: best,bcfg=a,(wc,wt)
    print(f' wc={wc} wt={wt}: {a:.2f} ({a-ref:+.2f})',flush=True)
rng=random.Random(0)
cov=[c for c in D.pseudo if has[D.cand_pos[D.ci[c]]]]; rng.shuffle(cov)
half=len(cov)//2; tune,val=set(cov[:half]),set(cov[half:])
qcls=[D.classes[cand.tolist()[g]] for g in gold.tolist()]
tm=torch.tensor([c in tune for c in qcls]); vm=torch.tensor([c in val for c in qcls])
btw,btt=3,.5; bt=top1((S0+3*Sic)[tm],gold[tm])
for wc in [2.5,3,3.5]:
  for wt in [0,0.5,1.0]:
    a=top1((S0+wc*Sic+wt*Stb)[tm],gold[tm])
    if a>bt: bt,btw,btt=a,wc,wt
vb=top1((S0+3*Sic+0.5*Stb)[vm],gold[vm]); vt=top1((S0+btw*Sic+btt*Stb)[vm],gold[vm])
out={'ref':ref,'best':best,'cfg':bcfg,'delta':best-ref,'cv':{'wc':btw,'wt':btt,'val_base':vb,'val_tuned':vt}}
json.dump(out,open(os.path.join(OUT,'inat_v39_fullstack_proxy.json'),'w'),indent=1)
print('best',best,'cv',vt,'delta',vt-vb,flush=True)
