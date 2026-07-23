# onet — CV4Ecology 2026 Fish Open-Set Recognition

**Start here:** [`HANDOFF.md`](HANDOFF.md) (scores, what’s dead, next steps) · [`COMPETITION_RULES.md`](COMPETITION_RULES.md) (never violate).

Best confirmed real score: **47.78%** (`submissions/submission_v33_sink72.zip`).

## Environment

```bash
bash scripts/setup_env.sh   # once
conda activate onet
python src/sanity.py        # torch + CUDA + BioCLIP
```

## Rebuild best submission

```bash
conda activate onet && python builders/build_v33_sinkhorn.py
# compliance fallback (no Sinkhorn): python builders/build_v31_strict_singlepipeline.py
```

## Layout

| path | role |
|---|---|
| `HANDOFF.md` | source of truth |
| `COMPETITION_RULES.md` | Codabench rules (cached) |
| `builders/` | **v31** + **v33** only |
| `src/` | train / embed / text (active) |
| `research/` | cheap proxies before GPU spend |
| `outputs/` | embeddings, checkpoints, logs |
| `submissions/` | scored Codabench zips (v23+) |
| `data/dl/` | images + labels + descriptions |
| `archive/legacy/` | stale code/docs — **ignore unless excavating** |

## Rules of engagement

- Never state a score that wasn’t measured. Distinguish holdout vs real (Codabench).
- Single-pipeline only: one rule over all images, argmax over full class space, no folder-oracle.
- Check `HANDOFF.md` §4 before proposing levers. Check `COMPETITION_RULES.md` before using external models/data.
