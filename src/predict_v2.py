"""v2 submission: route by the provided seen/unseen test split.
  test.pkl  (seen)   -> prototype argmax over SEEN classes
  unseen.pkl(unseen) -> text argmax over NON-SEEN (unseen-candidate) classes
"""
import json, torch
from collections import defaultdict

text = torch.load("outputs/text_emb.pt", weights_only=False)
classes = text["classes"]; cls_index = {c:i for i,c in enumerate(classes)}
Tname = text["emb_name"]
lab = json.load(open("data/dl/label_train.json"))

tr = torch.load("outputs/emb_train.pt", weights_only=False)
by_cls = defaultdict(list)
for fn,f in zip(tr["files"], tr["feats"]):
    if fn in lab and lab[fn] in cls_index: by_cls[lab[fn]].append(f)
seen = sorted(by_cls.keys())
seen_set = set(seen)

# prototypes from ALL train images of each seen class
P = torch.stack([ (torch.stack(by_cls[c]).mean(0)) for c in seen ])
P = P / P.norm(dim=-1, keepdim=True)
P_cls = seen   # row -> class name

# unseen-candidate classes = all classes NOT in seen
unseen_cand_idx = torch.tensor([i for i,c in enumerate(classes) if c not in seen_set])
Tu = Tname[unseen_cand_idx]

preds = {}
# seen test -> prototypes
et = torch.load("outputs/emb_test.pt", weights_only=False)
ti = (et["feats"] @ P.t()).argmax(1).tolist()
for fn,k in zip(et["files"], ti): preds[fn] = P_cls[k]
# unseen test -> text over non-seen classes
eu = torch.load("outputs/emb_unseen.pt", weights_only=False)
ui = (eu["feats"] @ Tu.t()).argmax(1).tolist()
for fn,j in zip(eu["files"], ui): preds[fn] = classes[unseen_cand_idx[j]]

json.dump(preds, open("outputs/prediction_v2_routed.json","w"))
print("wrote outputs/prediction_v2_routed.json n=", len(preds))
print("  seen(test)=%d via prototypes over %d seen classes" % (len(et["files"]), len(seen)))
print("  unseen=%d via text over %d non-seen classes" % (len(eu["files"]), len(unseen_cand_idx)))
