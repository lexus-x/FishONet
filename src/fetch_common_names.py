"""Fetch GBIF vernacular (common) names for all classes. Resumable, writes progress."""
import json, os, requests
from concurrent.futures import ThreadPoolExecutor, as_completed

classes = json.load(open("outputs/text_emb_h.pt.classes.json")) if os.path.exists("outputs/text_emb_h.pt.classes.json") else None
if classes is None:
    import torch
    classes = torch.load("outputs/text_emb_h.pt", weights_only=False)["classes"]

outp = "outputs/common_names.json"
cache = json.load(open(outp)) if os.path.exists(outp) else {}
todo = [c for c in classes if c not in cache]
print(f"{len(classes)} species total, {len(todo)} still to look up", flush=True)

sess = requests.Session()
def fetch(name):
    try:
        m = sess.get("https://api.gbif.org/v1/species/match",
                     params={"name": name, "kingdom": "Animalia"}, timeout=25).json()
        key = m.get("usageKey")
        if not key: return name, None
        v = sess.get(f"https://api.gbif.org/v1/species/{key}/vernacularNames", timeout=25).json()
        recs = v.get("results", [])
        eng = [r["vernacularName"] for r in recs if r.get("language") == "eng" and r.get("vernacularName")]
        any_ = [r["vernacularName"] for r in recs if r.get("vernacularName")]
        return name, (eng[0] if eng else (any_[0] if any_ else None))
    except Exception:
        return name, None

done = 0
with ThreadPoolExecutor(max_workers=16) as ex:
    futs = [ex.submit(fetch, c) for c in todo]
    for f in as_completed(futs):
        name, common = f.result(); cache[name] = common; done += 1
        if done % 500 == 0:
            json.dump(cache, open(outp, "w")); print(f"  {done}/{len(todo)} done", flush=True)

json.dump(cache, open(outp, "w"))
n_ok = sum(1 for c in classes if cache.get(c))
print(f"DONE: common name found for {n_ok}/{len(classes)} species ({100*n_ok/len(classes):.0f}%)", flush=True)
