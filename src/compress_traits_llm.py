"""Compress provided descriptions.json into short BioCLIP-native visual traits.

Uses a general-purpose LLM (Qwen2.5-Instruct) offline on competition text only —
ALLOWED under Codabench terms (LLM + provided descriptions). Disclose in DISCLOSURE.md.

  conda activate onet
  python src/compress_traits_llm.py [--model Qwen/Qwen2.5-3B-Instruct] [--limit N]

Writes outputs/traits_from_desc.json : {species: "trait phrase", ...}
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, 'data', 'dl')
OUT = os.path.join(ROOT, 'outputs')

SYS = (
    "You extract short visual identification traits for fish species photos. "
    "Reply with ONLY a comma-separated trait list. No species name. No fluff. "
    "Keep body shape, fins, colors, marks, counts. Drop hedges like 'normal', "
    "'unspecified', 'notably', 'captivating'. Max 25 words."
)

USER_TMPL = (
    "Scientific name (do not repeat in answer): {name}\n"
    "Description:\n{desc}\n\n"
    "Traits:"
)


def clean_traits(text: str, name: str) -> str:
    t = text.strip().split('\n')[0].strip()
    t = re.sub(r'^(traits?\s*:?\s*)', '', t, flags=re.I)
    t = t.strip(' "\'`')
    # drop if model echoed the binomial
    for part in name.split():
        t = re.sub(rf'\b{re.escape(part)}\b', '', t, flags=re.I)
    t = re.sub(r'\s{2,}', ' ', t).strip(' ,.;')
    # kill empty / hedge-only
    if len(t.split()) < 3:
        return ''
    words = t.split()
    if len(words) > 28:
        t = ' '.join(words[:28])
    return t


def heuristic_traits(desc: str) -> str:
    """Fallback if LLM returns empty: keep morph-heavy clauses, drop fluff."""
    fluff = re.compile(
        r'\b(captivating|striking|notably|unspecified|unnoted|not detailed|'
        r'normal in appearance|appear normal|without distinctive|collectively|'
        r'aiding in differentiation|these characteristics)\b',
        re.I,
    )
    keep_kw = re.compile(
        r'\b(fusiform|elongated|compressed|body|fin|caudal|dorsal|anal|pectoral|'
        r'pelvic|snout|mouth|eye|scale|bar|stripe|spot|blotch|color|colour|'
        r'silver|black|red|yellow|blue|green|olive|ray|spine|gill|forked|'
        r'truncate|concave|ctenoid|cycloid|mm|cm)\b',
        re.I,
    )
    parts = re.split(r'(?<=[.!;])\s+', desc.strip())
    kept = []
    for p in parts:
        p = fluff.sub('', p)
        p = re.sub(r'\s{2,}', ' ', p).strip(' ,.')
        if keep_kw.search(p) and len(p.split()) >= 4:
            kept.append(p.rstrip('.'))
        if sum(len(x.split()) for x in kept) >= 24:
            break
    t = ', '.join(kept)
    words = t.split()
    return ' '.join(words[:28]) if words else ''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='Qwen/Qwen2.5-3B-Instruct')
    ap.add_argument('--out', default=os.path.join(OUT, 'traits_from_desc.json'))
    ap.add_argument('--limit', type=int, default=0, help='debug: only first N classes')
    ap.add_argument('--bs', type=int, default=16)
    ap.add_argument('--resume', action='store_true')
    a = ap.parse_args()

    classes = list(pickle.load(open(os.path.join(D, 'all_classes.pkl'), 'rb')))
    desc = json.load(open(os.path.join(D, 'descriptions.json')))
    if a.limit:
        classes = classes[: a.limit]

    out = {}
    if a.resume and os.path.exists(a.out):
        out = json.load(open(a.out))
        print(f'resume: {len(out)} already done', flush=True)

    todo = [c for c in classes if c not in out or not out[c]]
    print(f'to compress: {len(todo)} / {len(classes)}  model={a.model}', flush=True)

    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = 'left'
    model = AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype=torch.bfloat16, device_map='auto', trust_remote_code=True
    )
    model.eval()

    t0 = time.time()
    for i in range(0, len(todo), a.bs):
        batch = todo[i: i + a.bs]
        messages = []
        for c in batch:
            d = desc.get(c) or c
            messages.append([
                {'role': 'system', 'content': SYS},
                {'role': 'user', 'content': USER_TMPL.format(name=c, desc=d[:1200])},
            ])
        prompts = [
            tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in messages
        ]
        inputs = tok(prompts, return_tensors='pt', padding=True, truncation=True,
                     max_length=1536).to(model.device)
        with torch.no_grad():
            gen = model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
            )
        # strip prompt tokens
        new_tokens = gen[:, inputs['input_ids'].shape[1]:]
        texts = tok.batch_decode(new_tokens, skip_special_tokens=True)
        for c, raw in zip(batch, texts):
            traits = clean_traits(raw, c)
            if not traits:
                traits = heuristic_traits(desc.get(c, ''))
            out[c] = traits
        if (i // a.bs) % 20 == 0 or i + a.bs >= len(todo):
            json.dump(out, open(a.out + '.tmp', 'w'), indent=1)
            os.replace(a.out + '.tmp', a.out)
            dt = time.time() - t0
            done = len(out)
            rate = (i + len(batch)) / max(dt, 1e-6)
            print(f'  {done}/{len(classes)}  {rate:.1f} cls/s  '
                  f'eg={todo[0] if i == 0 else batch[0][:40]!r} -> {out[batch[0]][:80]!r}',
                  flush=True)

    # ensure every class present
    for c in classes:
        if c not in out:
            out[c] = heuristic_traits(desc.get(c, ''))
    json.dump(out, open(a.out, 'w'), indent=1)
    n_empty = sum(1 for c in classes if not out.get(c))
    print(f'wrote {a.out}  n={len(out)}  empty={n_empty}  [{time.time()-t0:.0f}s]', flush=True)


if __name__ == '__main__':
    main()
