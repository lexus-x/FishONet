"""Qwen2.5-VL rerank v2: give the model COMMON names + provided species DESCRIPTIONS
(not bare Latin), over a tighter top-K shortlist. Same hard-sim validation harness as
rerank_qwen.py so the numbers are directly comparable (text-only top1 / in-shortlist ceiling).
  python src/rerank_qwen2.py --k 10 --n 150            # names+desc
  python src/rerank_qwen2.py --k 10 --n 150 --nodesc   # common names only
"""
import json, torch, argparse, os, random, re
from collections import defaultdict
import torch.nn.functional as F

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--nodesc", action="store_true", help="common names only, no descriptions")
    ap.add_argument("--imgroot", default="data/dl/images")
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    a = ap.parse_args()

    text = torch.load("outputs/text_emb.pt", weights_only=False)
    classes = text["classes"]; cls_index = {c: i for i, c in enumerate(classes)}
    Tname = F.normalize(text["emb_name"].float(), dim=-1)
    common = json.load(open("outputs/common_names.json"))
    desc = json.load(open("data/dl/descriptions.json"))
    def label(sp):                         # what Qwen sees for a candidate
        cn = common.get(sp, sp)
        if a.nodesc: return f"{cn} ({sp})"
        d = desc.get(sp, "")
        d = (d[:220] + "…") if len(d) > 220 else d
        return f"{cn} ({sp}): {d}"

    tr = torch.load("outputs/emb_train.pt", weights_only=False)
    lab = json.load(open("data/dl/label_train.json"))
    by_cls = defaultdict(list)
    for fn, f in zip(tr["files"], tr["feats"]):
        if fn in lab and lab[fn] in cls_index:
            by_cls[lab[fn]].append((f, fn))
    seen = sorted(by_cls.keys())
    order = sorted(seen, key=lambda c: len(by_cls[c]))
    pseudo = set(order[:int(len(seen) * 0.2)])
    known_idx = set(cls_index[c] for c in seen if c not in pseudo)
    cand = [i for i in range(len(classes)) if i not in known_idx]
    cand_t = torch.tensor(cand)
    cand_pos = {ci: j for j, ci in enumerate(cand)}

    q = [(f, fn, cls_index[c]) for c in pseudo for f, fn in by_cls[c]]
    Fq = F.normalize(torch.stack([x[0] for x in q]).float(), dim=-1)
    S = Fq @ Tname[cand_t].t()
    S = (S - S.mean(0, keepdim=True)) / (S.std(0, keepdim=True) + 1e-6)   # z-score debias
    topk = S.topk(a.k, 1).indices

    random.seed(a.seed)
    idxs = list(range(len(q))); random.shuffle(idxs); idxs = idxs[:a.n]
    text_correct = sum(1 for i in idxs if topk[i, 0].item() == cand_pos[q[i][2]])
    in_topk = sum(1 for i in idxs if cand_pos[q[i][2]] in topk[i].tolist())

    idx_img = {}
    for dp, _, fs in os.walk(a.imgroot):
        for fn in fs: idx_img[fn] = os.path.join(dp, fn)

    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(a.model, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    proc = AutoProcessor.from_pretrained(a.model)

    qwen_correct = n = parsed = 0
    for i in idxs:
        f, fn, gold_cls = q[i]; gold = cand_pos[gold_cls]
        cands = topk[i].tolist(); names = [label(classes[cand[c]]) for c in cands]
        path = idx_img.get(fn)
        if path is None: continue
        listing = "\n".join(f"{j+1}. {nm}" for j, nm in enumerate(names))
        prompt = (f"You are an expert ichthyologist identifying a fish to species from a photo. "
                  f"Exactly one of the {len(names)} candidates below is correct. Match the visible "
                  f"body shape, fins, coloration and patterns against each candidate's description. "
                  f"Reply with ONLY the number (1-{len(names)}).\n\n{listing}")
        msgs = [{"role": "user", "content": [{"type": "image", "image": path}, {"type": "text", "text": prompt}]}]
        t = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        imgs, vids = process_vision_info(msgs)
        inp = proc(text=[t], images=imgs, videos=vids, padding=True, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=8, do_sample=False)
        ans = proc.batch_decode(out[:, inp.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()
        n += 1
        m = re.search(r"\d+", ans)
        if m:
            parsed += 1; pick = int(m.group()) - 1
            if 0 <= pick < len(cands) and cands[pick] == gold: qwen_correct += 1
        if n % 25 == 0: print(f"  ...{n}/{len(idxs)} qwen={qwen_correct}", flush=True)

    N = len(idxs)
    mode = "names-only" if a.nodesc else "names+desc"
    print(f"\n=== RERANK v2 RESULT ({mode}, K={a.k}, n={N}) ===")
    print(f"in-shortlist ceiling : {100*in_topk/N:.1f}%  ({in_topk}/{N})")
    print(f"text-only  top1      : {100*text_correct/N:.1f}%  ({text_correct}/{N})")
    print(f"qwen-rerank top1     : {100*qwen_correct/max(n,1):.1f}%  ({qwen_correct}/{n}, parsed {parsed})")

if __name__ == "__main__":
    main()
