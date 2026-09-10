#!/usr/bin/env python3
"""Build the compact FishONet v109 source package for direct email attachment."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

from build_repro_bundle import (
    CODE_SEEDS,
    HISTORY_SOURCES,
    ROOT,
    history_bytes,
    local_code_closure,
    sha256_file,
)


ARCHIVE = ROOT / "FishONet_v109_Email_Code_Package.zip"
PREFIX = "FishONet_v109_Email_Code_Package"
MAX_BYTES = 20 * 1024 * 1024

FILES = [
    "environment.yml",
    "scripts/setup_env.sh",
    "COMPETITION_RULES.md",
    "DISCLOSURE.md",
    "LICENSE",
    "data/dl/all_classes.pkl",
    "data/dl/label_train.json",
    "outputs/learned_gate_v79.pkl",
    "outputs/rerank_unseen_v82_leakfree.pkl",
    "outputs/rerank_seen_genus_v103.pkl",
    "outputs/frozen_gate_v109.json",
]

REFERENCE_FILES = {
    "outputs/prediction_v109_genus_gamble.json": "reference/prediction_v109_genus_gamble.json",
    "submissions/submission_v109_genus_gamble.zip": "reference/submission_v109_genus_gamble.zip",
}

README = """# FishONet v109 — Compact Email Code Package

This compact package accompanies the FishONet technical report. It contains the
important source code, environment definition, class metadata, learned quota-gate and
re-ranker checkpoints, and the exact official v109 reference prediction.

Official result: **53.76139071919248% overall**
(77.00% photographed / 23.77% unseen), rank #3, CWNU AIX.

## Important

This email-sized package is intended for immediate method and code review. The large
feature tensors and support-bank checkpoints required to execute the full 35,665-image
replay are in `FishONet_v109_Reproducibility_Package.zip`, shared separately through
Google Drive. Extract the full package to run:

```bash
conda env create -f environment.yml
conda activate onet
python verify_package.py
python builders/build_v109_genus_gamble.py
python verify_prediction.py
```

The compact package includes the submitted `prediction.json` and Codabench submission
ZIP under `reference/`, allowing the organizer to inspect the exact delivered output
without downloading the larger checkpoint bundle first.

## Included method sources

- Official v109 and strict per-image compliance builders
- Proposed quota-gate training and frozen calibration
- Leak-free seen and unseen candidate re-rankers
- Hierarchical genus backoff training/evaluation
- AM-Softmax/CosFace encoder adaptation sources
- Crop-view, TaxaBind, BioCLIP-2, text, and support-bank extractors
- Environment, rules, disclosure, class index, and training labels

See the separately attached PDF for the full methodology and ablations.
"""


def main() -> None:
    code_files = local_code_closure(CODE_SEEDS)
    disk_files = sorted(set(FILES + code_files))
    missing = [item for item in disk_files if not (ROOT / item).is_file()]
    missing += [item for item in REFERENCE_FILES if not (ROOT / item).is_file()]
    if missing:
        raise SystemExit("Missing required files:\n" + "\n".join(missing))

    generated = {"README.md": README.encode()}
    for source, archive_name in HISTORY_SOURCES.items():
        generated[archive_name] = history_bytes(source)

    hashes: dict[str, str] = {}
    for item in disk_files:
        hashes[item] = sha256_file(ROOT / item)
    for source, archive_name in REFERENCE_FILES.items():
        hashes[archive_name] = sha256_file(ROOT / source)
    for archive_name, content in generated.items():
        hashes[archive_name] = hashlib.sha256(content).hexdigest()
    manifest = "".join(f"{hashes[name]}  {name}\n" for name in sorted(hashes)).encode()
    summary = json.dumps(
        {
            "package": PREFIX,
            "purpose": "direct email attachment; code and lightweight checkpoints",
            "official_accuracy": 0.5376139071919248,
            "files": len(hashes),
            "full_artifacts": "FishONet_v109_Reproducibility_Package.zip (Google Drive)",
        },
        indent=2,
    ).encode()

    if ARCHIVE.exists():
        ARCHIVE.unlink()
    with zipfile.ZipFile(ARCHIVE, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for item in disk_files:
            archive.write(ROOT / item, f"{PREFIX}/{item}")
        for source, archive_name in REFERENCE_FILES.items():
            archive.write(ROOT / source, f"{PREFIX}/{archive_name}")
        for archive_name, content in generated.items():
            archive.writestr(f"{PREFIX}/{archive_name}", content)
        archive.writestr(f"{PREFIX}/MANIFEST.sha256", manifest)
        archive.writestr(f"{PREFIX}/PACKAGE_CONTENTS.json", summary)

    size = ARCHIVE.stat().st_size
    if size > MAX_BYTES:
        raise SystemExit(f"Email archive exceeds 20 MiB safety target: {size}")
    print(f"archive={ARCHIVE}")
    print(f"archive_bytes={size}")
    print(f"files={len(hashes)} code_files={len(code_files) + len(HISTORY_SOURCES)}")


if __name__ == "__main__":
    main()
