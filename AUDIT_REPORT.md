# Repo Audit Report — pre-organizer-report cleanup

**Date:** 2026-09-05 · **Scope:** full working tree at `/home/ubuntu/onet` · **Trigger:** preparing the
project for submission to the CV4Ecology 2026 / Codabench 16815 organizers (testing phase closed
2026-09-01 22:00 UTC; results due 2026-09-09).

This is a factual record of what was found and changed. Nothing in `data/` or `outputs/` was
touched. Nothing was permanently destroyed — everything removed from the working tree that wasn't
already recoverable via `git log` was moved, intact, to `~/onet_precleanup_backup_20260905/` on this
machine (not inside the repo, not committed).

---

## 1. Security findings (fixed)

Three live API credentials were sitting in plaintext:

| Location | Credential | Fix |
|---|---|---|
| `.mcp.json` | GitHub PAT (`ghp_...`) | Replaced with `${GITHUB_PERSONAL_ACCESS_TOKEN}` |
| `.mcp.json` | HuggingFace token (`hf_...`) | Replaced with `${HF_TOKEN}` |
| `.mcp.json` | Context7 API key | Replaced with `${CONTEXT7_API_KEY}` |
| `git remote origin` URL (`.git/config`) | Same GitHub PAT, embedded in the HTTPS URL | `git remote set-url origin https://github.com/lexus-x/FihOnet.git` (no embedded credential) |

`.mcp.json` is gitignored, so none of these were ever committed to git history — no history
scrub needed. **The GitHub PAT and HuggingFace token should still be revoked/rotated on
GitHub's and HuggingFace's side** — stripping them from local config does not invalidate them.
A repo-wide grep (`ghp_[A-Za-z0-9]{30,}|hf_[A-Za-z0-9]{20,}`) found no other occurrences.

## 2. Identity / license check

`README.md`, `github/` (the public site), and several 2026-08-31 commits attribute the project to
**Lalith Sai, ISLab (Changwon National University), supervised by Prof. Cheng Yaw Low**, and set the
license to "Proprietary — All Rights Reserved, Not Open Source." This was flagged for confirmation
before doing anything with it — **confirmed accurate by the user.**

**Residual flag — RESOLVED 2026-09-07, see §8.** Original finding: COMPETITION_RULES.md §5 states winning teams "may be asked to
release code/checkpoints." A blanket "All Rights Reserved, Not Open Source" license doesn't
obviously accommodate that if it's invoked. Worth a one-line carve-out before this goes to
organizers if it places well — not changed here since it's a licensing decision, not a cleanup one.

## 3. Reproducibility gaps (partially fixed)

The documented "rebuild the best submission" recipes in `CLAUDE.md`, `README.md`, and `REPORT.md`
had two problems, found by tracing each builder's actual `outputs/*.{pt,pkl,json}` reads against
what the recipes claimed to run:

1. **Broken paths.** A prior reorg moved `build_v50_chase53.py`, `build_v46_lora_tol_denser.py`,
   `build_v43_bioclip2_dual.py`, and other historical builders into `builders/legacy_archive/`, but
   `CLAUDE.md` still referenced the old flat `builders/build_vXX....py` paths. **Fixed** — CLAUDE.md
   now lists the real dependency chain (v56 → v77 → v81 → v82 → v83 → v109) and calls out
   `legacy_archive/` explicitly, including the compliance-fallback exception (below).
2. **Non-reproducible artifact.** The actual winning chain depends on `outputs/learned_gate_v79.pkl`
   (the v77 gate refit at `C=100`, per `HANDOFF.md` 2026-08-29). `research/learned_gate_v77.py` as
   committed hardcodes `C=1.0` and only ever writes `learned_gate_v77.pkl` — it cannot produce the
   `v79` artifact. **FIXED 2026-09-07 (see §8)** — `C` and the output tag are now environment
   parameters and the deployed pkl rebuilds bit-identically. `CLAUDE.md`, `README.md` and
   `REPORT.md` were updated to carry the working command instead of the caveat.

**One documented, deliberate compliance exception:** `builders/legacy_archive/build_v31_strict_singlepipeline.py`
was kept in place (not moved to the external backup) because COMPETITION_RULES.md §4.1 designates it
the fallback if organizers challenge the Sinkhorn/transductive step anywhere in the winning chain.
This is the one file kept outside the strict "only the v109 chain" rule, and it's called out in
`CLAUDE.md` so it doesn't get lost again.

## 4. Disclosure accuracy (fixed)

`DISCLOSURE.md`'s FishBase row said "not fetched" (as of 2026-07-23) and a later changelog entry
claimed "no external resource was introduced after 2026-07-30 (v46)." Both were true when written,
but `reports/descriptions_fishbase.json` and `src/build_text_fishbase.py` show FishBase morphological
descriptions were in fact fetched and embedded on **2026-09-01** — after that audit line — and this
was never logged. Traced whether it reached the scored pipeline: **no builder references it**, so
it's unused, but COMPETITION_RULES.md §3.2 requires disclosing external resources regardless of
whether they were deployed (non-disclosure risks disqualification). **Fixed** — added a corrected
table row and a changelog entry noting the fetch, the date, and the confirmed non-use.

## 5. Stale headline numbers (fixed)

`README.md` and `REPORT.md` presented **v83 (53.697%)** as the final result throughout — badges, the
executive summary, the results table, the "final submitted artifact" field, and the reproduction
guide. The actual current best per `CLAUDE.md`/`HANDOFF.md` is **v109 (53.73615589513528%)**, which
adds a genus-level hierarchical backoff on top of v83's seen route. **Fixed** in both files: headline
numbers, the results table (v83 kept as a row, v109 added as the starred best), the "final submitted
artifact" references, and the reproduction command blocks (now walk v56→v77→v81→v82→v83→v109 instead
of stopping at v83).

## 6. Repo cleanup — what moved where

Everything below is intact at `~/onet_precleanup_backup_20260905/` unless marked "deleted outright."

| Item | Size | Reason | Disposition |
|---|---|---|---|
| `archive/` (`legacy/` + `pre_v50/`) | 6.3 GB | Self-labeled "legacy"/"pre-v50" in `CLAUDE.md`; already told to be ignored | moved to backup |
| `media/` | 1.7 MB | SVG/MP4/PNG assets, not referenced anywhere in `HANDOFF.md`/`CLAUDE.md`/`DISCLOSURE.md` | moved to backup |
| `fishonet-novelty-gate/` | 588 KB | Standalone spin-off repo with its own `git` history + its own GitHub remote (`lexus-x/fishonet-novelty-gate`); not imported or referenced by the onet pipeline | moved to backup (already independently preserved on GitHub) |
| Root `descriptions_fishbase.json` | small | A **different** file than the canonical `reports/descriptions_fishbase.json` that the code actually loads — an orphaned draft | moved to backup |
| `.agents/` (frontend/design Claude-plugin skill defs) | 576 KB | Generic third-party skill files (brandkit, gpt-taste, imagegen-*, etc.) — unrelated to this ML research project, not the user's own work | **deleted outright** (no research content, trivially re-installable from the plugin marketplace if ever needed) |
| `skills-lock.json` | small | Manifest for the deleted `.agents/` skills | **deleted outright** |
| `__pycache__/` (7 dirs) | small | Regenerable bytecode | **deleted outright** |
| 42 off-lineage `submissions/*.zip` (v50–v76 sweeps, `v84/v86/v87`, `v99`, `v109_compliant*`, `sanity_check`, `reweighted_lora*`) | ~15 MB | Hyperparameter-sweep noise, not on the documented winning path; each one's *score* (where real) is already recorded in `HANDOFF.md` | moved to backup |
| 10 off-lineage `builders/*.py` (`v73`–`v76` learned-pipeline variants, `v88`/`v95`/`v99` seen-retrain variants, `v109_compliant*`, `reweighted_softgate`) | small | Undocumented in `HANDOFF.md`/`CLAUDE.md`; not part of the confirmed v109 dependency graph | moved to backup |

**Kept untouched:** `research/*.py` (all ~130 files — this is the real, already-narrated v84→v109
ablation trail referenced throughout `HANDOFF.md`; none of it is "old code," it's the evidence
trail a technical report needs), `builders/legacy_archive/` (already tucked away by a prior reorg;
holds the v31 compliance fallback), `src/`, `data/`, `outputs/`, `reports/` (contains the actual
paper draft, revision-response notes, and figures), `github/` (the public site, confirmed
legitimate), and all top-level docs.

## 7. Compliance checklist (COMPETITION_RULES.md §6)

- [x] No use of `splits/*.pkl` for membership or candidate restriction — confirmed by builder
      inspection during the dependency trace in §3; nothing new found beyond what `HANDOFF.md`
      already tracks.
- [x] Predictions always over the full 17,393-class space, one uniform rule — unchanged, not
      touched by this cleanup.
- [x] No fish-specific fine-tuned third-party models — `DISCLOSURE.md`'s "Not used" section already
      lists the two flagged repos (CaesiumSG, ReefNet); nothing new found.
- [x] External text/data logged for disclosure — one gap found and fixed (§4, FishBase).
- [x] v31 fallback preserved and locatable — see §3.
- [x] Submission budget — moot now; testing phase is closed.

## 8. Follow-up pass — 2026-09-07 (all three residuals closed)

- **`learned_gate_v79.pkl` is now reproducible.** `research/learned_gate_v77.py` takes `GATE_C` and
  `GATE_TAG` from the environment; `GATE_C=100 GATE_TAG=v79 python research/learned_gate_v77.py`
  rebuilds the deployed artifact **bit-identically** (coefficients and intercept exactly equal,
  CV delta `1.0804456002995266` on both). The default invocation still writes the C=1.0 v77 artifact
  unchanged. Verified against `outputs/learned_gate_v79_repro.pkl`.
- **License carve-out added** (§2 residual). `README.md` and `github/README.md` now grant the
  organizers an explicit, irrevocable right to receive, inspect, execute and reproduce the inference
  code and weights for verification — satisfying COMPETITION_RULES.md §5 without making the work
  public domain.
- **Compliance is now machine-checked, not asserted.** New `research/verify_compliance.py` re-runs
  every claim and exits non-zero on failure; output in `outputs/compliance_report.json`. All checks
  pass. Notably it proves the router is *not* a folder oracle: seen-split images reach a
  photographed class only **89.11%** of the time and unseen-split images an unphotographed class
  **77.58%** — an oracle would score 100/100.
- **Transduction exposure quantified, not argued.** `builders/build_v109_compliant.py` and
  `build_v109_compliant_dbnorm.py` were restored from the backup into the tree (they were moved out
  as "off-lineage" in §6 — that was a mistake; they are compliance artifacts). Freezing the two
  routing couplings changes **729 / 35,665** predictions (2.044%); additionally freezing `dbnorm`'s
  hub term changes 4,213 (11.813%). The old fallback story ("drop to v31, −6.05pt") was needlessly
  pessimistic.

## 9. Still not done

- The GitHub PAT and HuggingFace token from §1 still need rotating on the providers' side.
- Did not commit anything to git. `git status` currently shows ~186 deletions (mostly the
  now-recoverable-via-history `archive/legacy/` files) and the pre-existing untracked-file set minus
  what moved to backup. Left staged-but-uncommitted intentionally — review with `git status`/`git
  diff` before deciding to commit or push, especially since `origin` is a live public repo.
