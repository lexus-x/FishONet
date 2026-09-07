import pickle, json, os, requests
from concurrent.futures import ThreadPoolExecutor, as_completed
classes = list(pickle.load(open("data/dl/all_classes.pkl","rb")))
out = {}
if os.path.exists("outputs/taxonomy.json"):
    out = json.load(open("outputs/taxonomy.json"))
todo = [c for c in classes if c not in out]
print("total", len(classes), "to resolve", len(todo), flush=True)
sess = requests.Session()
def fetch(name):
    try:
        r = sess.get("https://api.gbif.org/v1/species/match",
                     params={"name": name, "kingdom": "Animalia"}, timeout=25).json()
        return name, {"order": r.get("order"), "family": r.get("family"),
                      "genus": r.get("genus") or name.split()[0], "match": r.get("matchType")}
    except Exception:
        return name, {"order": None, "family": None, "genus": name.split()[0], "match": "ERR"}
done = 0
with ThreadPoolExecutor(max_workers=16) as ex:
    futs = [ex.submit(fetch, c) for c in todo]
    for f in as_completed(futs):
        name, info = f.result(); out[name] = info; done += 1
        if done % 1000 == 0:
            json.dump(out, open("outputs/taxonomy.json","w"))
            print(f"{done}/{len(todo)}", flush=True)
json.dump(out, open("outputs/taxonomy.json","w"))
fam = sum(1 for v in out.values() if v.get("family"))
order = sum(1 for v in out.values() if v.get("order"))
print(f"DONE resolved={len(out)} with_family={fam} with_order={order}", flush=True)
