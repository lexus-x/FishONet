import json, time, os, torch
import torch.nn.functional as F
from collections import defaultdict
torch.set_num_threads(8)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
STATUS = "outputs/job_status.txt"
t0 = time.time(); results=[]; best={"name":None,"acc":-1.0,"section":None}; done=0

def write_status(current="", final=None):
    el=time.time()-t0; L=[]
    L.append("="*60)
    L.append("  JOB: DBNorm / Dual-Inverted-Softmax  (unseen hard-sim)")
    L.append("="*60)
    L.append(f"  elapsed {el:6.1f}s    configs done: {done}")
    L.append("  "+("DONE [✓]" if final is not None else "RUNNING: "+current))
    if best["name"]: L.append(f"  best so far: {best['acc']:.2f}%  [{best['section']} | {best['name']}]")
    L.append("-"*60); sec=None
    for s,n,a in results:
        if s!=sec: L.append(f"  [{s}]"); sec=s
        tag="  <== BEST" if (n==best['name'] and s==best['section']) else ""
        L.append(f"     {n:28s} {a:6.2f}%{tag}")
    if final:
        L.append("-"*60)
        for ln in final: L.append("  "+ln)
    L.append("="*60)
    tmp=STATUS+".tmp"; open(tmp,"w").write("\n".join(L)+"\n"); os.replace(tmp,STATUS)

def rec(sec,name,acc):
    global done; acc=float(acc); results.append((sec,name,round(acc,2)))
    if acc>best["acc"]: best.update(name=name,acc=acc,section=sec)
    done+=1; write_status(current=f"{sec}/{name}"); time.sleep(0.35)
    print(f"[{sec}] {name}: {acc:.2f}%",flush=True)

def loadfeat(p):
    d=torch.load(p,weights_only=False); return {fn:i for i,fn in enumerate(d["files"])}, F.normalize(d["feats"].float(),dim=-1)

write_status(current="loading embeddings...")
txtH=torch.load("outputs/text_emb_h.pt",weights_only=False)
classes=txtH["classes"]; ci={c:i for i,c in enumerate(classes)}
TnH=F.normalize(txtH["emb_name"].float(),dim=-1)
TtaxH=F.normalize(torch.load("outputs/text_emb_h_taxon.pt",weights_only=False)["emb_taxon"].float(),dim=-1)
TnL=F.normalize(torch.load("outputs/text_emb.pt",weights_only=False)["emb_name"].float(),dim=-1)
lab=json.load(open("data/dl/label_train.json"))
HtrI,HtrF=loadfeat("outputs/emb_train_h_tta.pt"); LtrI,LtrF=loadfeat("outputs/emb_train.pt")
by=defaultdict(list)
for fn in HtrI:
    if fn in lab and lab[fn] in ci: by[lab[fn]].append(fn)
seen=sorted(by.keys()); order=sorted(seen,key=lambda c:len(by[c]))
pseudo=set(order[:int(len(seen)*0.2)]); known=set(c for c in seen if c not in pseudo)
nonk=[i for i,c in enumerate(classes) if c not in known]; nt=torch.tensor(nonk); cpos={v:j for j,v in enumerate(nonk)}
qfiles=[f for c in pseudo for f in by[c] if f in HtrI and f in LtrI]
gold=torch.tensor([cpos[ci[lab[f]]] for f in qfiles]).to(dev)
QH=torch.stack([HtrF[HtrI[f]] for f in qfiles]).to(dev); QL=torch.stack([LtrF[LtrI[f]] for f in qfiles]).to(dev)
TnH_=TnH[nt].to(dev); TtaxH_=TtaxH[nt].to(dev); TnL_=TnL[nt].to(dev)
print(f"pseudo={len(pseudo)} queries={len(qfiles)} candidates={len(nonk)}",flush=True)

def t1(M): return (M.argmax(1)==gold).float().mean().item()*100
def zc(S): return (S-S.mean(0,keepdim=True))/(S.std(0,keepdim=True)+1e-6)
def colis(S,tc): return F.log_softmax(S/tc,dim=0)
def dis(S,tc,tr): return F.log_softmax(S/tc,dim=0)+F.log_softmax(S/tr,dim=1)
def sink(S,temp,it):
    Lm=S/temp
    for _ in range(it): Lm=Lm-torch.logsumexp(Lm,1,keepdim=True); Lm=Lm-torch.logsumexp(Lm,0,keepdim=True)
    return Lm

SHn=(QH@TnH_.t()).double(); SHt=(QH@TtaxH_.t()).double(); SLn=(QL@TnL_.t()).double()
TCs=[0.01,0.02,0.05,0.1]; TRs=[0.05,0.1,0.5]
def best_dis(S,sec):
    b=(-1.0,None)
    for tc in TCs:
        a=t1(colis(S,tc)); write_status(current=f"{sec} col-IS tc={tc} -> {a:.2f}%"); time.sleep(0.04)
        if a>b[0]: b=(a,f"col-IS tc={tc}")
        for tr in TRs:
            a=t1(dis(S,tc,tr)); write_status(current=f"{sec} DIS tc={tc} tr={tr} -> {a:.2f}%"); time.sleep(0.04)
            if a>b[0]: b=(a,f"DIS tc={tc} tr={tr}")
    return b

rec("A H-name","raw cosine",t1(SHn))
A_z=t1(zc(SHn)); rec("A H-name","z-debias (current)",A_z)
sb=max((t1(sink(SHn,tp,10)),f"sinkhorn t={tp}") for tp in [0.02,0.04]); rec("A H-name",sb[1],sb[0])
ab=best_dis(SHn,"A H-name"); rec("A H-name",ab[1],ab[0])
B_z=t1(zc(SHt)); rec("B H-taxon","z-debias (current)",B_z)
bb=best_dis(SHt,"B H-taxon"); rec("B H-taxon",bb[1],bb[0])
C_cur=t1(zc(SHt)+0.5*zc(SLn)); rec("C ensemble","z-debias x2 (DEPLOYED)",C_cur)
cb=(-1.0,None)
for tc in TCs:
    for tr in TRs:
        a=t1(dis(SHt,tc,tr)+0.5*dis(SLn,tc,tr)); write_status(current=f"C ens DIS tc={tc} tr={tr} -> {a:.2f}%"); time.sleep(0.04)
        if a>cb[0]: cb=(a,f"DBNorm tc={tc} tr={tr}")
rec("C ensemble",cb[1],cb[0])
k=0.51; ov=lambda u:0.563*79.1+0.437*u
fin=[
 f"DEPLOYED ensemble (C): z-debias {C_cur:.2f}%  ->  DBNorm {cb[0]:.2f}%   (delta {cb[0]-C_cur:+.2f})",
 f"single-backbone DBNorm vs z-debias:  A name {ab[0]-A_z:+.2f}    B taxon {bb[0]-B_z:+.2f}",
 f"PROJECTED real unseen (x{k}, UNMEASURED): {k*C_cur:.1f}% -> {k*cb[0]:.1f}%",
 f"PROJECTED overall: {ov(k*C_cur):.1f}% -> {ov(k*cb[0]):.1f}%",
 ("VERDICT: DBNorm BEATS z-debias -> build into a submission, validate on board" if cb[0]>C_cur+0.3
  else "VERDICT: DBNorm ~= z-debias here -> no clear win; keep z-debias"),
]
for ln in fin: print(ln,flush=True)
write_status(final=fin); print("DONE",flush=True)
