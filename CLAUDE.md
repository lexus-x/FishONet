# onet — working notes for Claude

CV4Ecology 2026 fish open-set recognition. **Read `HANDOFF.md` first — it is the source of truth.**
**Read `COMPETITION_RULES.md` before any experiment or submission — never violate it.**
`README.md` is the short entry; do **not** read `archive/legacy/` unless excavating history.

## Environment

```bash
conda activate onet # created by scripts/setup_env.sh
python src/sanity.py # verify torch + CUDA + BioCLIP
```

## Layout

| dir | what |
|---|---|
| `builders/` | **only** `build_v33_sinkhorn.py` (best, 47.78%) + `build_v31_strict_singlepipeline.py` (compliance) |
| `src/` | training + embedding extraction (`ft.py`, `fullft336.py`, `embed*.py`, `build_text_promptens.py`) |
| `research/` | experiment harnesses (`unseen_*.py`) — benchmark here before spending GPU time |
| `outputs/` | embeddings, model checkpoints, frozen constants, diagnostic logs |
| `submissions/` | Codabench zips (v23+) |
| `data/dl/images/` | 99,979 training jpgs |
| `archive/legacy/` | stale v20–v30 code/docs — **ignore** |

## Rebuild the best submission

```bash
conda activate onet && python builders/build_v33_sinkhorn.py
```

## Rules of engagement

- **Never state a score that wasn't measured.** Distinguish holdout from real (Codabench) in every
  claim. Holdout overstates real gains — Sinkhorn was 3.7× overstated. See HANDOFF.md §7.
- **Shift-related fixes can only be validated by a real submission.** Budget submissions accordingly
  (**3/day · 30 total**).
- **Single-pipeline compliance is mandatory**: one uniform rule over all 35,665 images, one argmax
  over the full 17,393-class space, no folder-oracle routing. See HANDOFF.md §6 + COMPETITION_RULES.md.
- **Check HANDOFF.md §4 (WHAT'S DEAD) before proposing anything.**
- Pipeline/gate/routing levers are exhausted (gate efficiency 90.7%, oracle ceiling ~52.7%).
  Remaining headroom is `b` (unseen zero-shot). §8 item 1 (backbone) is closed at BioCLIP-2.5 ViT-H;
  primary lever is richer external text (§8 item 2) with disclosure.
- Update `HANDOFF.md` when a real score changes. It is how the next session starts.
