#!/usr/bin/env python3
"""Build the minimal FishONet v109 organizer reproducibility archive."""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import zipfile
from collections import deque
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "FishONet_v109_Reproducibility_Package.zip"
PREFIX = "FishONet_v109_Reproducibility"

RUNTIME_ARTIFACTS = [
    "data/dl/all_classes.pkl",
    "data/dl/label_train.json",
    "outputs/learned_gate_v79.pkl",
    "outputs/rerank_unseen_v82_leakfree.pkl",
    "outputs/rerank_seen_genus_v103.pkl",
    "outputs/frozen_gate_v109.json",
    "outputs/text_emb_h_taxon.pt",
    "outputs/text_emb.pt",
    "outputs/text_emb_h_promptens.pt",
    "outputs/text_emb_taxabind_taxctx.pt",
    "outputs/emb_train_ctftshift.pt",
    "outputs/emb_test_ctftshift.pt",
    "outputs/emb_unseen_ctftshift.pt",
    "outputs/emb_train_ftshift.pt",
    "outputs/emb_test_ftshift.pt",
    "outputs/emb_unseen_ftshift.pt",
    "outputs/emb_train_fullft336shift.pt",
    "outputs/emb_test_fullft336shift.pt",
    "outputs/emb_unseen_fullft336shift.pt",
    "outputs/emb_train.pt",
    "outputs/emb_test.pt",
    "outputs/emb_unseen.pt",
    "outputs/emb_train_fullft336_v2.pt",
    "outputs/emb_test_fullft336_v2.pt",
    "outputs/emb_unseen_fullft336_v2.pt",
    "outputs/emb_test_taxabind.pt",
    "outputs/emb_unseen_taxabind.pt",
    "outputs/emb_test_bioclip2.pt",
    "outputs/emb_unseen_bioclip2.pt",
    "outputs/emb_test_bioclip2_lora_v2.pt",
    "outputs/emb_unseen_bioclip2_lora_v2.pt",
    "outputs/emb_test_ctft_cropviews.pt",
    "outputs/emb_unseen_ctft_cropviews.pt",
    "outputs/emb_test_ctft_ostrip.pt",
    "outputs/emb_unseen_ctft_ostrip.pt",
    "outputs/inat_photo_bank_ctftshift.pt",
    "outputs/inat_photo_bank_fullft336shift.pt",
    "outputs/inat_tol_merged_b2_a05.pt",
    "outputs/inat_tol_merged_b2lora_a0.5.pt",
]

SMALL_MODEL_CHECKPOINTS = [
    "outputs/ctft_shift.pt",
    "outputs/ft_lora_shift.pt",
    "outputs/b2_shift_lora.pt",
]

CODE_SEEDS = [
    "builders/build_v109_genus_gamble.py",
    "builders/build_v109_compliant.py",
    "research/learned_gate_v77.py",
    "research/rerank_unseen_v81.py",
    "research/rerank_leakfree_v82.py",
    "research/rerank_seen_v83.py",
    "research/genus_recall10_v103.py",
    "research/genus_rerank_v103_train.py",
    "research/compute_frozen_gate_constants.py",
    "research/verify_compliance.py",
    "research/contrastive_ft_bioclip2.py",
    "research/extract_b2_lora.py",
    "research/extract_ctft_any.py",
    "research/extract_taxabind.py",
    "research/extract_eval_336_crops.py",
    "research/extract_eval_crop_views.py",
    "research/embed_inat.py",
    "research/embed_inat_shift_banks.py",
    "src/contrastive_ft.py",
    "src/ft.py",
    "src/fullft336.py",
    "src/build_text_embeddings.py",
    "src/build_text_b2_taxctx.py",
    "src/build_text_promptens.py",
    "fish_preprocessing.py",
]

DOCUMENTS = [
    "environment.yml",
    "scripts/setup_env.sh",
    "COMPETITION_RULES.md",
    "DISCLOSURE.md",
    "LICENSE",
    "reports/FishONet_Technical_Report_Email.pdf",
]

HISTORY_SOURCES = {
    "research/contrastive_ft_bioclip2_v2.py": "research/contrastive_ft_bioclip2_v2.py",
}

REFERENCE_FILES = {
    "outputs/prediction_v109_genus_gamble.json": "reference/prediction_v109_genus_gamble.json",
    "submissions/submission_v109_genus_gamble.zip": "reference/submission_v109_genus_gamble.zip",
}

SEARCH_DIRS = [ROOT, ROOT / "research", ROOT / "src", ROOT / "builders"]


def resolve_local_module(name: str) -> Path | None:
    parts = name.split(".")
    for base in SEARCH_DIRS:
        candidate = base.joinpath(*parts).with_suffix(".py")
        if candidate.is_file():
            return candidate
        package = base.joinpath(*parts, "__init__.py")
        if package.is_file():
            return package
    return None


def local_code_closure(seeds: list[str]) -> list[str]:
    queue = deque((ROOT / item).resolve() for item in seeds)
    found: set[Path] = set()
    while queue:
        path = queue.popleft()
        if path in found:
            continue
        if not path.is_file() or ROOT not in path.parents:
            raise FileNotFoundError(path)
        found.add(path)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names.append(node.module)
            for name in names:
                local = resolve_local_module(name)
                if local is not None and local not in found:
                    queue.append(local.resolve())
    return sorted(str(path.relative_to(ROOT)) for path in found)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def history_bytes(git_path: str) -> bytes:
    return subprocess.check_output(["git", "show", f"HEAD:{git_path}"], cwd=ROOT)


README = """# FishONet v109 — Organizer Reproducibility Package

This package contains the exact code and frozen artifacts needed to reproduce the
official `v109_genus_gamble` prediction file submitted by **CWNU AIX**. The official
score was **53.76139071919248%** overall (77.00% photographed / 23.77% unseen).

## Fast exact replay

Requirements: Linux, Python 3.10, CUDA-capable PyTorch, and an NVIDIA GPU with about
40 GB VRAM. CPU fallback exists but is not practical for the 35,665-image matrix job.

```bash
conda env create -f environment.yml
conda activate onet
python verify_package.py
python builders/build_v109_genus_gamble.py
python verify_prediction.py
```

The builder writes:

- `outputs/prediction_v109_genus_gamble.json`
- `outputs/submission_v109_genus_gamble.zip`
- `submissions/submission_v109_genus_gamble.zip`

`verify_prediction.py` requires an exact dictionary match against the submitted
reference prediction. JSON key ordering is ignored; every filename and species label
must match.

## Strict per-image compliance replay

The official v109 builder uses the disclosed global 60% quota. A second builder is
included for audit purposes; it freezes the gate normalization and probability
threshold so each image is processed independently:

```bash
python builders/build_v109_compliant.py
```

## What is included

- Final v109 official and compliance builders.
- All transitive local Python dependencies needed by those builders.
- Gate and leak-free seen/unseen re-ranker checkpoints.
- Training, evaluation, crop-view, text, TaxaBind, and BioCLIP-2 feature checkpoints.
- iNaturalist / TreeOfLife support-bank tensors used by the unseen route.
- Small LoRA checkpoints and the key training/extraction scripts for method audit.
- Exact submitted prediction and submission archive under `reference/`.
- Technical report, environment definition, competition rules, and disclosure.
- `MANIFEST.sha256` for file-level integrity verification.

## Deliberate exclusions

- Raw competition and external images: not needed for exact replay because the exact
  frozen feature tensors and support banks are included; redistribution may also be
  governed by their original dataset terms.
- Redundant full-model snapshots (`fullft336*.pt`): the exact extracted features used
  by v109 are included, while duplicating two approximately 4 GB encoder snapshots
  would not change the reproduced prediction.
- Failed experiments, caches, logs, old submissions, website assets, previews, and
  temporary files.

## Training provenance

The included scripts document and implement the AM-Softmax/CosFace adaptation,
long-tail-aware calibration experiments, quota gate, leak-free shortlist re-ranking,
support-bank construction, and genus backoff. The historical
`contrastive_ft_bioclip2_v2.py` source is restored inside this package from the Git
revision that produced the v2 feature family; its final extracted v2 features are
included even though the large intermediate v2 checkpoint was not retained locally.

## Data layout

Do not rename files or move `outputs/`, `data/dl/`, `research/`, or `builders/`.
The final builder uses these repository-relative paths.

## Result verification

The package reproduces the prediction artifact, not the hidden leaderboard scoring
labels. The organizer can upload the regenerated submission to Codabench or compare
it directly with `reference/prediction_v109_genus_gamble.json`.
"""

VERIFY_PACKAGE = r'''#!/usr/bin/env python3
import hashlib
from pathlib import Path

root = Path(__file__).resolve().parent
manifest = root / "MANIFEST.sha256"
failures = []
count = 0
for line in manifest.read_text(encoding="utf-8").splitlines():
    expected, rel = line.split("  ", 1)
    path = root / rel
    digest = hashlib.sha256()
    if not path.is_file():
        failures.append(f"missing: {rel}")
        continue
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected:
        failures.append(f"hash mismatch: {rel}")
    count += 1
if failures:
    raise SystemExit("PACKAGE CHECK FAILED\n" + "\n".join(failures))
print(f"PACKAGE CHECK PASSED: {count} files")
'''

VERIFY_PREDICTION = r'''#!/usr/bin/env python3
import json
from pathlib import Path

root = Path(__file__).resolve().parent
expected_path = root / "reference/prediction_v109_genus_gamble.json"
actual_path = root / "outputs/prediction_v109_genus_gamble.json"
if not actual_path.is_file():
    raise SystemExit("Run: python builders/build_v109_genus_gamble.py")
expected = json.loads(expected_path.read_text(encoding="utf-8"))
actual = json.loads(actual_path.read_text(encoding="utf-8"))
missing = sorted(set(expected) - set(actual))
extra = sorted(set(actual) - set(expected))
changed = sorted(key for key in expected.keys() & actual.keys() if expected[key] != actual[key])
if missing or extra or changed:
    raise SystemExit(
        f"PREDICTION CHECK FAILED: missing={len(missing)} extra={len(extra)} changed={len(changed)}"
    )
print(f"PREDICTION CHECK PASSED: exact match for {len(actual)} images")
'''


def main() -> None:
    code_files = local_code_closure(CODE_SEEDS)
    disk_files = sorted(set(RUNTIME_ARTIFACTS + SMALL_MODEL_CHECKPOINTS + DOCUMENTS + code_files))
    missing = [item for item in disk_files if not (ROOT / item).is_file()]
    missing += [item for item in REFERENCE_FILES if not (ROOT / item).is_file()]
    if missing:
        raise SystemExit("Missing required files:\n" + "\n".join(missing))

    generated: dict[str, bytes] = {
        "README_REPRODUCE.md": README.encode(),
        "verify_package.py": VERIFY_PACKAGE.encode(),
        "verify_prediction.py": VERIFY_PREDICTION.encode(),
    }
    for source, archive_name in HISTORY_SOURCES.items():
        generated[archive_name] = history_bytes(source)

    secret_markers = (b"BEGIN PRIVATE KEY", b"AKIA", b"ghp_", b"CLOUDFLARE_API_TOKEN")
    text_candidates = [item for item in disk_files if Path(item).suffix in {".py", ".sh", ".md", ".yml", ".json"}]
    for item in text_candidates:
        content = (ROOT / item).read_bytes()
        if any(marker in content for marker in secret_markers):
            raise SystemExit(f"Potential secret marker in {item}")

    hashes: dict[str, str] = {}
    sizes: dict[str, int] = {}
    for item in disk_files:
        path = ROOT / item
        hashes[item] = sha256_file(path)
        sizes[item] = path.stat().st_size
    for source, archive_name in REFERENCE_FILES.items():
        path = ROOT / source
        hashes[archive_name] = sha256_file(path)
        sizes[archive_name] = path.stat().st_size
    for archive_name, content in generated.items():
        hashes[archive_name] = hashlib.sha256(content).hexdigest()
        sizes[archive_name] = len(content)

    manifest = "".join(f"{hashes[name]}  {name}\n" for name in sorted(hashes)).encode()
    contents = json.dumps(
        {
            "package": PREFIX,
            "official_build": "v109_genus_gamble",
            "official_accuracy": 0.5376139071919248,
            "files": len(hashes),
            "payload_bytes": sum(sizes.values()),
            "runtime_artifacts": len(RUNTIME_ARTIFACTS),
            "model_checkpoints": len(SMALL_MODEL_CHECKPOINTS),
            "code_files": len(code_files) + len(HISTORY_SOURCES),
            "excluded": ["raw images", "failed experiments", "logs", "caches", "website assets"],
        },
        indent=2,
    ).encode()

    if ARCHIVE.exists():
        ARCHIVE.unlink()
    with zipfile.ZipFile(ARCHIVE, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for item in disk_files:
            archive.write(ROOT / item, f"{PREFIX}/{item}")
        for source, archive_name in REFERENCE_FILES.items():
            archive.write(ROOT / source, f"{PREFIX}/{archive_name}")
        for archive_name, content in generated.items():
            archive.writestr(f"{PREFIX}/{archive_name}", content)
        archive.writestr(f"{PREFIX}/MANIFEST.sha256", manifest)
        archive.writestr(f"{PREFIX}/PACKAGE_CONTENTS.json", contents)

    print(f"archive={ARCHIVE}")
    print(f"archive_bytes={ARCHIVE.stat().st_size}")
    print(f"payload_files={len(hashes)} code_files={len(code_files) + len(HISTORY_SOURCES)}")


if __name__ == "__main__":
    main()
