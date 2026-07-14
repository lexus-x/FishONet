"""Generate the biological FAMILY for every fish species in the dataset.

Just run:   bash run.sh
Outputs (written next to this file):
    family_by_species.csv   <- open in Excel: species, family
    family_map.json         <- same thing for code (species -> family)

Both are ALREADY included (pre-computed). Running this just regenerates/updates them.
It asks GBIF (the public global species database) for each species' family.
Resumable: if it stops, run again and it picks up where it left off.
Legal: uses only public taxonomy *names*, not any fish images/data.
"""
import json, os, csv, requests
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
classes = [l.strip() for l in open(os.path.join(HERE, "all_classes.txt")) if l.strip()]
outp = os.path.join(HERE, "family_map.json")
fam = json.load(open(outp)) if os.path.exists(outp) else {}
todo = [c for c in classes if not fam.get(c)]
print(f"{len(classes)} species total, {len(todo)} still to look up", flush=True)

sess = requests.Session()
def fetch(name):
    try:
        r = sess.get("https://api.gbif.org/v1/species/match",
                     params={"name": name, "kingdom": "Animalia"}, timeout=25).json()
        return name, r.get("family")
    except Exception:
        return name, None

done = 0
with ThreadPoolExecutor(max_workers=16) as ex:
    for f in as_completed([ex.submit(fetch, c) for c in todo]):
        name, family = f.result(); fam[name] = family; done += 1
        if done % 2000 == 0:
            json.dump(fam, open(outp, "w")); print(f"  {done}/{len(todo)} done", flush=True)

json.dump(fam, open(outp, "w"), indent=0)
with open(os.path.join(HERE, "family_by_species.csv"), "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["species", "family"])
    for c in classes: w.writerow([c, fam.get(c) or ""])

n_ok = sum(1 for c in classes if fam.get(c))
print(f"DONE: family found for {n_ok}/{len(classes)} species ({100*n_ok/len(classes):.0f}%)")
print("wrote family_map.json and family_by_species.csv in this folder")
