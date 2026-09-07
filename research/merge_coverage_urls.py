"""Merge coverage_urls_extra.json into coverage_urls.json (union URLs per class)."""
from __future__ import annotations

import argparse
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'outputs')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', default=os.path.join(OUT, 'coverage_urls.json'))
    ap.add_argument('--extra', default=os.path.join(OUT, 'coverage_urls_extra.json'))
    ap.add_argument('--classes', default=os.path.join(OUT, 'uncovered_unseen_classes.json'))
    ap.add_argument('--prune', action='store_true', help='Keep only keys in --classes list')
    a = ap.parse_args()

    classes = json.load(open(a.classes))
    base = json.load(open(a.base)) if os.path.exists(a.base) else {}
    extra = json.load(open(a.extra)) if os.path.exists(a.extra) else {}

    merged = dict(base)
    added_classes = 0
    added_urls = 0
    for c, urls in extra.items():
        prev = list(merged.get(c) or [])
        seen = set(prev)
        for u in urls or []:
            if u and u not in seen:
                prev.append(u)
                seen.add(u)
                added_urls += 1
        if urls and not base.get(c):
            added_classes += 1
        merged[c] = prev

    if a.prune:
        cls_set = set(classes)
        merged = {c: merged.get(c, []) for c in classes}

    json.dump(merged, open(a.base, 'w'))
    hit = sum(1 for c in classes if merged.get(c))
    print(
        f'merged -> {a.base}: +{added_classes} new classes, +{added_urls} urls; '
        f'hit {hit}/{len(classes)} ({100 * hit / len(classes):.1f}%)',
        flush=True,
    )


if __name__ == '__main__':
    main()
