import json, pickle, torch
import torch.nn.functional as F
D = "data/dl"
classes = list(pickle.load(open(f"{D}/all_classes.pkl", "rb")))
lab = json.load(open(f"{D}/label_train.json"))
seen_set = set(lab.values())
preds = json.load(open("outputs/prediction_v21_unified_c.json"))
unseen_files = pickle.load(open(f"{D}/splits/unseen.pkl", "rb"))
test_files = pickle.load(open(f"{D}/splits/test.pkl", "rb"))

missing = [f for f in unseen_files if f not in preds]
usub = {f: preds[f] for f in unseen_files if f in preds}
n_seen_pred = sum(1 for c in usub.values() if c in seen_set)
print(f"unseen-split: {len(unseen_files)} files, {len(missing)} MISSING from preds")
print(f"  predicted SEEN class:   {n_seen_pred} ({100*n_seen_pred/len(usub):.1f}%)")
print(f"  predicted UNSEEN class: {len(usub)-n_seen_pred} ({100*(len(usub)-n_seen_pred)/len(usub):.1f}%)")

tsub = {f: preds[f] for f in test_files if f in preds}
ts = sum(1 for c in tsub.values() if c in seen_set)
print(f"test-split: predicted SEEN {100*ts/len(tsub):.1f}% / UNSEEN {100*(len(tsub)-ts)/len(tsub):.1f}%")

# alignment: does emb_unseen_h score sensibly vs taxon text vs emb_train_h?
TtH = F.normalize(torch.load("outputs/text_emb_h_taxon.pt", weights_only=False)["emb_taxon"].float(), dim=-1)
uh = torch.load("outputs/emb_unseen_h.pt", weights_only=False)
trh = torch.load("outputs/emb_train_h.pt", weights_only=False)
UH = F.normalize(uh["feats"][:500].float(), dim=-1)
TR = F.normalize(trh["feats"][:500].float(), dim=-1)
print(f"emb_unseen_h vs TtH max-cos mean = {(UH@TtH.t()).max(1).values.mean():.3f}")
print(f"emb_train_h  vs TtH max-cos mean = {(TR@TtH.t()).max(1).values.mean():.3f}")
uh_files = uh["files"]
uset = set(unseen_files)
print(f"emb_unseen_h files sample: {uh_files[:3]}")
print(f"  in unseen.pkl? {[uh_files[i] in uset for i in range(3)]}")
print(f"  emb_unseen_h files present in unseen.pkl: {sum(1 for f in uh_files if f in uset)}/{len(uh_files)}")

# ground truth for unseen present anywhere on box?
import os
for cand in ["data/dl/label_unseen.json", "data/dl/label_test.json", "data/dl/labels_unseen.json", "outputs/gt_unseen.json"]:
    print("GT candidate", cand, "exists:", os.path.exists(cand))
