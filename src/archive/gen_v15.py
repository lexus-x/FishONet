import json, torch, torch.nn.functional as F, zipfile, shutil
def load(p):
    d=torch.load(p,weights_only=False); return {fn:i for i,fn in enumerate(d['files'])}, F.normalize(d['feats'].float(),dim=-1), list(d['files'])
preds=json.load(open('outputs/prediction_v12.json'))
val=json.load(open('outputs/v15_val.json')); tc,tr=val['tc'],val['tr']; USE=val['dbnorm']>val['base']+0.2
txtHt=torch.load('outputs/text_emb_h_taxon.pt',weights_only=False); classes=txtHt['classes']; ci={c:i for i,c in enumerate(classes)}
TtH=F.normalize(txtHt['emb_taxon'].float(),dim=-1)
TnL=F.normalize(torch.load('outputs/text_emb.pt',weights_only=False)['emb_name'].float(),dim=-1)
lab=json.load(open('data/dl/label_train.json'))
FtrI,FtrF,Ftrf=load('outputs/emb_train_ft.pt'); LtrI,_,_=load('outputs/emb_train.pt')
trainfiles=[fn for fn in Ftrf if fn in LtrI and fn in lab and lab[fn] in ci]
seen=set(sorted(set(lab[fn] for fn in trainfiles)))
nonk=torch.tensor([i for i,c in enumerate(classes) if c not in seen])
HunI,HunF,Hunf=load('outputs/emb_unseen_ctftbig.pt'); LunI,LunF,_=load('outputs/emb_unseen.pt')
uf=[fn for fn in Hunf if fn in LunI]
uH=torch.stack([HunF[HunI[fn]] for fn in uf]); uL=torch.stack([LunF[LunI[fn]] for fn in uf])
MH=uH@TtH[nonk].t(); ML=uL@TnL[nonk].t()
def db(M): return (M-M.mean(0,keepdim=True))/(M.std(0,keepdim=True)+1e-6)
def dis(M,a,b): return F.log_softmax(M/a,dim=0)+F.log_softmax(M/b,dim=1)
if USE: eU=dis(MH,tc,tr)+0.5*dis(ML,tc,tr); method='DBNorm tc=%s tr=%s'%(tc,tr)
else:   eU=db(MH)+0.5*db(ML); method='z-debias (v12; DBNorm did not beat baseline)'
changed=0
for fn,j in zip(uf,eU.argmax(1).tolist()):
    c=classes[nonk[j].item()]
    if preds.get(fn)!=c: changed+=1
    preds[fn]=c
json.dump(preds,open('outputs/prediction_v15.json','w'))
z=zipfile.ZipFile('outputs/submission_v15.zip','w',zipfile.ZIP_DEFLATED); z.write('outputs/prediction_v15.json',arcname='prediction.json'); z.close()
shutil.copy('outputs/submission_v15.zip','/mnt/c/Users/islab/submission_v15.zip')
ru=0.51*(val['dbnorm'] if USE else val['base'])
print('UNSEEN method   :',method)
print('val hard-sim    : z-debias %.2f / DBNorm %.2f  (delta %+.2f, n=%d)'%(val['base'],val['dbnorm'],val['delta'],val['n']))
print('unseen changed  : %d / %d'%(changed,len(uf)))
print('PROJECTED real unseen ~%.1f%%  ->  overall ~%.1f%%  (seen 79.1)'%(ru,0.563*79.1+0.437*ru))
print('wrote submission_v15.zip  entries=%d'%len(preds))
