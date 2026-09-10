# onet — working notes for Claude

CV4Ecology 2026 fish open-set recognition. **Read `HANDOFF.md` first — it is the source of truth.**
**Read `COMPETITION_RULES.md` before any experiment or submission — never violate it.**
`README.md` is the short entry. `archive/` (legacy v20–v30 code, pre-v50 zips/scripts) was moved out
of the repo to `~/onet_precleanup_backup_20260905/` during the 2026-09-05 audit cleanup — not
present in the working tree; ignore unless the user restores it for history-excavation.

## Environment

```bash
conda activate onet # created by scripts/setup_env.sh
python src/sanity.py # verify torch + CUDA + BioCLIP
```

## Layout

| dir | what |
|---|---|
| `builders/` | `build_v77_learned_gate.py` (**best real, 53.3828683583345%** @ f=0.60 τ=1.8 — v56 heads + 12-feature learned logistic gate; **53% bar cleared**) · `build_v56_overlap_336.py` (51.61643067433057%) · `build_v55_cropmax_336.py` (unscored; holdout +0.992) · `build_v50_chase53.py` (51.44259077526987%) · `build_v46_lora_tol_denser.py` (50.83975886723678%) · `build_v44_tol_denser.py` (50.77246600308426%) · `build_v43_bioclip2_dual.py` (50.75564278704613%) · `build_v41_bioclip2_proto.py` (50.57%) · `build_v40_inat_maxpool.py` (50.49%) · `build_v36_ctftshift_taxabind.py` (49.04%) · `build_v33_sinkhorn.py` (47.78%) · `build_v31_strict_singlepipeline.py` (compliance) |
| `src/` | training + embedding extraction (`ft.py`, `fullft336.py`, `embed*.py`, `build_text_promptens.py`) |
| `research/` | experiment harnesses (`unseen_*.py`) — benchmark here before spending GPU time |
| `outputs/` | embeddings, model checkpoints, frozen constants, diagnostic logs |
| `submissions/` | Codabench zips (v23+) |
| `data/dl/images/` | 99,979 training jpgs |
| `archive/legacy/` | stale v20–v30 code/docs — **ignore** |

## Rebuild the best submission

```bash
conda activate onet && python research/genus_rerank_v103_train.py      #   ^ retrains outputs/rerank_seen_genus_v103.pkl first
conda activate onet && python builders/build_v109_genus_gamble.py      # BEST confirmed real 53.73615589513528% (f=0.60) — final submission
conda activate onet && python builders/build_v83_rerank_both.py        #   ^ v109 reads this build's prediction json as an input
conda activate onet && python builders/build_v82_rerank_leakfree.py    #   ^ v83 reads this one's pkl
conda activate onet && python builders/build_v81_rerank.py
conda activate onet && python builders/build_v77_learned_gate.py       # prior best confirmed real 53.3828683583345% (f=0.60)
conda activate onet && GATE_C=100 GATE_TAG=v79 python research/learned_gate_v77.py
                                                                         #   ^ v81+ consume outputs/learned_gate_v79.pkl (same 12
                                                                         #   features, C=100 refit). VERIFIED bit-identical to the
                                                                         #   deployed pkl (coef + intercept exact, 2026-09-07).
                                                                         #   Plain `python research/learned_gate_v77.py` still
                                                                         #   writes the C=1.0 v77 artifact, unchanged.
conda activate onet && python builders/build_v56_overlap_336.py        # prior 51.61643067433057% (f=0.60)
```

Milestones v31–v50 were moved to `builders/legacy_archive/` in an earlier reorg — not part of the
v109 dependency chain above, kept for historical reference only. **Exception:
`builders/legacy_archive/build_v31_strict_singlepipeline.py` is the designated compliance fallback**
(COMPETITION_RULES.md §4.1) if organizers ever challenge the Sinkhorn/transductive step anywhere in
the winning chain.

## Rules of engagement

- **Never state a score that wasn't measured.** Distinguish holdout from real (Codabench) in every
  claim. Holdout overstates real gains — Sinkhorn was 3.7× overstated. See HANDOFF.md §7.
- **Shift-related fixes can only be validated by a real submission.** Budget submissions accordingly
  (**3/day · 30 total**).
- **Single-pipeline compliance is mandatory**: one uniform rule over all 35,665 images, one argmax
  over the full 17,393-class space, no folder-oracle routing. See HANDOFF.md §6 + COMPETITION_RULES.md.
- **Check HANDOFF.md §4 (WHAT'S DEAD) before proposing anything.**
- **Two routing axes — never conflate them (v77, 2026-08-29):**
  - **NEW INFORMATION in the gate compounds; RE-WEIGHTING does not** (measured twice, 2026-08-29).
    New features change *which* images sit near the threshold, and if they also predict
    head-correctness the swap lifts `a_cond`/`b_cond` too: v77 = **+1.082 routing +0.686 conditionals
    = +1.767pt**. Re-weighting the same features only slides marginal images across a fixed ordering,
    so `a_cond` FALLS and gives ~90% back: v79 (C=1→100) = **+0.481 routing −0.439 conditionals
    = +0.042pt**. **Gate C / f / re-weight / recalibration tuning is EXHAUSTED** — project any such
    change at ~**0.1×** its routing delta. Only new evidence earns a slot.
  - **Moving `f` TRADES at ~4:1** (`overall = 0.49554*TPR + 0.12509*TNR`). Every operating-point move
    lost: v60 (91.1/52.5), v63 (90.6/51.1 → 50.68%), v34, soft_b1.05. `f=0.60` is real-tuned, and
    holdout is *anti-correlated* on this axis — do not sweep `f` on a proxy.
- Encoder / `b` levers stay closed (VLM rerank, external images, trait text, general CLIP, v34 recover,
 unified single-head 2-stage ≤49.4%) — HANDOFF §4/§8. But v77 gained **without** a new encoder: it read
 evidence the pipeline already computed and the gate had never looked at. **Look there first.**
 Best real: **53.73615589513528%** (`submissions/submission_v109_genus_gamble.zip`, v83 + genus-level
 hierarchical backoff on the seen route — new-evidence lever, +0.039pt over v83; unseen route untouched).
 v83 53.69690172437964%, v82 53.64362820692555%, v81 53.43053413710921%, v79 53.42492639842983%,
 v77 53.3828683583345%, v56 51.61643067433057%.
- **The pseudo-novel holdout LEAKS for per-candidate learning (v81, 2026-08-29).** Gold is always one
 of the 1,159 rarest *training* classes = 9.1% of the pool, and that subpopulation is identifiable
 from the features. A top-20 re-ranker learned "prefer gold-eligible" (picks it 53.7% vs the fusion's
 38.4%) → **+13.374 holdout, +0.006 real**. **Class-disjoint CV does NOT detect this.** Per-IMAGE
 decisions (the gate) are safe; **per-CANDIDATE learning — re-rankers, learned fusion weights,
 learned calibrators over the class axis — must be trained on a LEAK-FREE pool.** Fix = restrict the
 holdout candidate pool to gold-eligible classes only + make features pool-size invariant
 (percentile ranks). v82 did this: holdout gain fell +13.374 → +8.154 but real rose +0.006 → **+0.213**.
 **Magnitude of a holdout number is not evidence; the population it is measured on is.**
 **Per-candidate re-ranker transfer anchor: ~0.026-0.030 overall-pt per leak-free proxy-pt**,
 confirmed on BOTH heads (unseen 0.0261, seen 0.0295). **+1.00 overall-pt needs +34 proxy-pt** — use
 this to kill proposals fast. Both heads are now within ~0.65 overall-pt of their re-ranking ceilings;
 per-candidate re-ranking is effectively CLOSED.
 Transfer anchors: dual-B2 **~0.36**; LoRA∩ToL denser **~0.195 overall-pt/proxy-pt**; frozen denser-ToL **~0.039**; maxpool ~0.02.
- Update `HANDOFF.md` when a real score changes. It is how the next session starts.
