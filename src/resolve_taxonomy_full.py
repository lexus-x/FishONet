import pickle, json, os, requests
from concurrent.futures import ThreadPoolExecutor, as_completed
classes = list(pickle.load(open("data/dl/all_classes.pkl","rb")))
out = {}
if os.path.exists("outputs/taxonomy_full.json"):
    out = json.load(open("outputs/taxonomy_full.json"))
todo = [c for c in classes if c not in out]
print("total", len(classes), "to resolve", len(todo), flush=True)
sess = requests.Session()
def fetch(name):
    try:
        r = sess.get("https://api.gbif.org/v1/species/match",
                     params={"name": name, "kingdom": "Animalia"}, timeout=25).json()
        return name, {"kingdom": r.get("kingdom"), "phylum": r.get("phylum"),
                      "class": r.get("class"), "order": r.get("order"),
                      "family": r.get("family"), "genus": r.get("genus") or name.split()[0],
                      "match": r.get("matchType"), "confidence": r.get("confidence"),
                      "rank": r.get("rank"), "status": r.get("status")}
    except Exception:
        return name, {"kingdom": None,"phylum": None,"class": None,"order": None,"family": None,
                      "genus": name.split()[0],"match": "ERR","confidence": 0,"rank": None,"status": None}
done = 0
with ThreadPoolExecutor(max_workers=16) as ex:
    futs = [ex.submit(fetch, c) for c in todo]
    for f in as_completed(futs):
        name, info = f.result(); out[name] = info; done += 1
        if done % 2000 == 0:
            json.dump(out, open("outputs/taxonomy_full.json","w")); print(f"{done}/{len(todo)}", flush=True)
json.dump(out, open("outputs/taxonomy_full.json","w"))
def cov(k): return sum(1 for v in out.values() if v.get(k))
print(f"DONE n={len(out)} class={cov('class')} order={cov('order')} family={cov('family')}", flush=True)
# quick confidence + coverage audit vs seen
lab = json.load(open("data/dl/label_train.json")); seen = set(lab.values())
seen_fams = set(out[c]["family"] for c in seen if c in out and out[c].get("family"))
unseen = [c for c in classes if c not in seen]
cov_fam = sum(1 for c in unseen if out.get(c,{}).get("family") in seen_fams)
lowconf = sum(1 for c in unseen if (out.get(c,{}).get("confidence") or 0) < 90)
print(f"AUDIT unseen={len(unseen)} family-in-seen-families={cov_fam} ({100*cov_fam/len(unseen):.0f}%) low_conf(<90)={lowconf}", flush=True)
