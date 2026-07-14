import json
v15 = json.load(open("outputs/prediction_v15.json"))
v18 = json.load(open("outputs/prediction_v18_meanshift_probe.json"))
v19 = json.load(open("outputs/prediction_v19_coral_probe.json"))
# only look at seen/test keys (non-const in v18/v19 probes); use v19's tf keys via diff-from-v15
seen_keys = [k for k in v15 if v18.get(k) != "Ostracion cubicum" or v15[k] == "Ostracion cubicum"]
# simpler: seen/test keys are those where v18 differs from a "tanked" marker OR just use keys present in both changed sets
flip18 = {k for k in v15 if v18[k] != v15[k]}
flip19 = {k for k in v15 if v19[k] != v15[k]}
agree_flip = flip18 & flip19
agree_same_target = sum(1 for k in agree_flip if v18[k] == v19[k])
print(f"v18(meanshift) flips={len(flip18)}  v19(CORAL) flips={len(flip19)}")
print(f"both-flip-same-image={len(agree_flip)}  of-those-same-new-label={agree_same_target} ({100*agree_same_target/max(1,len(agree_flip)):.1f}%)")
print(f"v18-only flips={len(flip18-flip19)}  v19-only flips={len(flip19-flip18)}")
