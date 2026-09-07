"""Build a standardized, reproducible iNaturalist Validation Split Manifest.

Partitions the downloaded iNaturalist research-grade photos (outputs/inat_image_files.json)
into:
  1. Unseen / Zero-Shot Partition: Candidate species with 0 training images in train set.
  2. Seen / Domain Transfer Partition: Candidate species present in train set.

Usage:
  conda activate onet
  python scripts/build_inat_val_split.py --k_unseen 3 --k_seen 2 --save outputs/inat_val_manifest.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'outputs')
DATA = os.path.join(ROOT, 'data')


def main():
    parser = argparse.ArgumentParser(description="Build iNaturalist Validation Manifest")
    parser.add_argument('--inat_files', default=os.path.join(OUT, 'inat_image_files.json'),
                        help="Path to inat_image_files.json")
    parser.add_argument('--train_labels', default=os.path.join(DATA, 'dl', 'label_train.json'),
                        help="Path to label_train.json")
    parser.add_argument('--k_unseen', type=int, default=3,
                        help="Max photos per unseen class for zero-shot val")
    parser.add_argument('--k_seen', type=int, default=2,
                        help="Max photos per seen class for domain-transfer val")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    parser.add_argument('--save', default=os.path.join(OUT, 'inat_val_manifest.json'),
                        help="Path to save output manifest")
    args = parser.parse_args()

    random.seed(args.seed)

    print(f"Loading iNat image file index from {args.inat_files}...")
    with open(args.inat_files, 'r') as f:
        inat_map: dict[str, list[str]] = json.load(f)

    print(f"Loading training labels from {args.train_labels}...")
    with open(args.train_labels, 'r') as f:
        train_labels: dict[str, str] = json.load(f)

    train_classes = set(train_labels.values())

    all_inat_classes = sorted(inat_map.keys())
    seen_classes = sorted([c for c in all_inat_classes if c in train_classes])
    unseen_classes = sorted([c for c in all_inat_classes if c not in train_classes])

    print(f"\n--- iNaturalist Class Coverage ---")
    print(f"Total iNat classes available: {len(all_inat_classes):,}")
    print(f"  ├─ Seen classes (in train)    : {len(seen_classes):,}")
    print(f"  └─ Unseen classes (zero-shot) : {len(unseen_classes):,}")

    items = []
    unseen_img_count = 0
    seen_img_count = 0

    # 1. Unseen species split (zero-shot evaluation)
    for c in unseen_classes:
        files = inat_map[c]
        # Verify file existence
        valid_files = [p for p in files if os.path.isfile(p)]
        if not valid_files:
            continue
        # Deterministic sample
        sampled = random.sample(valid_files, min(len(valid_files), args.k_unseen))
        for p in sampled:
            items.append({
                "image_path": p,
                "class_name": c,
                "split": "unseen",
                "is_unseen": True
            })
            unseen_img_count += 1

    # 2. Seen species split (domain transfer evaluation)
    for c in seen_classes:
        files = inat_map[c]
        valid_files = [p for p in files if os.path.isfile(p)]
        if not valid_files:
            continue
        sampled = random.sample(valid_files, min(len(valid_files), args.k_seen))
        for p in sampled:
            items.append({
                "image_path": p,
                "class_name": c,
                "split": "seen",
                "is_unseen": False
            })
            seen_img_count += 1

    manifest = {
        "metadata": {
            "seed": args.seed,
            "k_unseen": args.k_unseen,
            "k_seen": args.k_seen,
            "total_images": len(items),
            "unseen_images": unseen_img_count,
            "seen_images": seen_img_count,
            "unseen_classes": len(unseen_classes),
            "seen_classes": len(seen_classes),
            "total_classes": len(all_inat_classes)
        },
        "items": items
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.save)), exist_ok=True)
    with open(args.save, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"\n--- Validation Manifest Generated ---")
    print(f"Output saved to: {args.save}")
    print(f"Total validation images: {len(items):,}")
    print(f"  ├─ Zero-shot unseen images : {unseen_img_count:,} (across {len(unseen_classes):,} species)")
    print(f"  └─ Seen domain-transfer imgs: {seen_img_count:,} (across {len(seen_classes):,} species)")


if __name__ == '__main__':
    main()
