# Paper revision state — response to peer review

Artifacts: `reports/fishonet_openset_paper.pdf`, generator `reports/make_openset_paper.py`,
audit data `outputs/eval_bank_duplicate_audit.json`.

## NEW EXPERIMENT RUN (fixes the reviewer's most serious objection)

The published leakage control compared the external bank against **training** queries
(`research/inat_dup_probe.py` loads `emb_train_h.pt`), while the abstract claimed it ruled out
retrieval of **evaluation** images. The reviewer was right; the control tested the wrong images.

Ran the correct audit: all 35,665 evaluation embeddings vs all 101,106 bank photos, in the
deployed bank encoder's space (`ctftshift`).

**Threshold calibration**
| reference population | n | p50 | p99 | p99.9 | frac > 0.95 |
|---|---|---|---|---|---|
| same species, different photographs (within bank) | 536,318 | 0.684 | 0.922 | 0.973 | 0.302% |
| same physical image, different crop geometry | 150,000 | 0.917 | — | — | 34.2% |

So >0.95 is a defensible near-duplicate threshold: p99.9 territory for genuinely different
photos of one species, routine for one image re-cropped.

**Result**
| eval split | n | p50 | >0.95 | >0.98 | max |
|---|---|---|---|---|---|
| trained-class | 20,097 | 0.779 | 44 (0.219%) | 24 | 0.9995 |
| novel-class | 15,568 | 0.799 | 141 (0.906%) | 48 | 0.9980 |

- Trained split rate (0.219%) is **below** the 0.302% same-species chance baseline → no evidence.
- Novel split rate (0.906%) is **~3x** the baseline → a real near-duplicate population exists.
- **Bound:** 185/35,665 = **0.519 pp** max overall inflation; **0.906 pp** max on novel accuracy,
  which is **8.9%** of the +10.15 pp novel gain attributed to external retrieval.
- Conclusion: the headline gain is not explained by duplication, but the absolute claim must be
  replaced by this quantified bound.

## EFFECT SIZES (reviewer #5 — no uncertainty estimates)

Net images out of 35,665: v50 vs v46 = 215 · v43 vs v41 = 66 · **v56 vs v50 = 62** ·
v46 vs v44 = 24 · v44 vs v43 = **6**.
v56 vs v50 changed 4,092 predictions to net 62 correct = 1.5% net yield.
McNemar needs the discordance split (b, c) which requires labels we do not have; only
b - c is recoverable. State this rather than implying significance.

## REVISION TODO

1. [x] Replace leakage section with the audit above + calibration table + bound. Weaken abstract.
2. [x] Reframe open-set -> generalized zero-shot / large-vocabulary seen-unseen. Retitle (shorter).
3. [x] Weaken overclaims: "encoder is the wall" -> "candidate-space reduction does not explain the
       residual error"; "taxonomic prior actively misinformative" -> "our family-level scoring
       procedure gave no useful prior"; "proxy reliability is a property of the axis" -> "in this
       trajectory, transfer varied by intervention type"; "entire gap was b" -> soften (one snapshot).
4. [x] New "Threats to validity" section: test-set adaptation is the headline threat — 30 sequential
       leaderboard evaluations, routing fraction set on real feedback, AND the aspect-ratio fix was
       derived by measuring the evaluation images (transductive).
5. [x] Add net-image effect sizes; state significance is not computable without labels.
6. [x] "deployed" -> "final evaluated configuration" throughout.
7. [x] Define the label-space accounting once (17,393 / 11,598 / 12,757 / 4,636 + 1,159).
8. [x] Expand related work: adaptive data analysis + reusable holdout, leaderboard overfitting
       (Ladder), benchmark replication (Recht et al.), accuracy-on-the-line, transductive GZSL,
       calibrated stacking, hubness, retrieval-augmented classification, long-tail recognition.
9. [~] Tighten abstract to one central story; shorten title.
10. [x] Note compute cost and that the per-image (inductive) equivalence was verified only at an
        earlier configuration and needs rerunning.


## COLLABORATOR REVIEW PASS — 2026-08-24

### Factual errors found and fixed

| claim | was | is |
|---|---|---|
| training images | 99,979 | **64,259** over 5,795 classes (`data/dl/label_train.json`) |
| adapter-training share | 61,941 / 99,979 = 62.0% | 61,941 / 64,259 = **96.4%** |
| pseudo-novel withheld | 1,159 (classes only) | 1,159 classes = **2,318 images** (each has exactly 2) |

99,979 is the jpg count in `data/dl/images/`, which holds train + **both evaluation splits**
(64,259 + 20,097 + 15,568 = 99,924, plus 55 unreferenced files). It was never a training count.

### Clarifications added (all were real ambiguities, not wording)

- **Closed-set candidate set is 5,795, not 4,636** (`seen classes: 5795` in every build log).
  4,636 is only Eq. 7's loss denominator. The 1,159 withheld classes keep prototypes built from
  their 2 images each and stay on the closed route — they are *not* pushed to the novel route.
- **p_c in Eq. 2 is a genuine class prototype** (L2-normalized mean of the class's training
  embeddings). The paper's "the banks are deliberately not prototypes" made this ambiguous.
- **Eq. 2's taxonomic anchor is gated on `if hspace`** — only the two ViT-H-space members get it.
- **Eq. 7 has no margin term** — stated at the equation, not only in Related Work.
- **Constant provenance is mixed, not "validation-tuned".** Eq. 2/Table 2 weights: proxy coordinate
  search. Routing fraction f and Sinkhorn tau: **real leaderboard feedback** (HANDOFF:253, and the
  proxy ranks f in the wrong order). Stated per-constant.
- **Tune-on-withheld-then-retrain-on-all-5,795** noted as the standard protocol we did not run,
  with the reason (retrain + one evaluation, outside budget).

### NEW EXPERIMENT: the hub offset does not need the test batch

`research/dbnorm_frozen_hub.py` → `outputs/dbnorm_frozen_hub_summary.json`.
dbnorm's across-image term is `s_ic/0.05 − h_c` with `h_c = logsumexp_i(s_ic/0.05)`; `h_c` is the
only place the batch enters. Estimated it on reference pools disjoint from the 11,598 novel
classes, froze it, rebuilt v56 otherwise byte-identical, diffed against the shipped predictions.

| reference pool | n | changed / 35,665 | can change accuracy |
|---|---|---|---|
| withheld pseudo-novel (7-view matched) | 2,318 | **296** (0.83%) | 224 |
| &nbsp;&nbsp;even half | 1,159 | 300 | 226 |
| &nbsp;&nbsp;odd half | 1,159 | 297 | 223 |
| trained-class images (centre view only) | 4,000 | 480 | 357 |

- All four runs reproduce the shipped v56 predictions **exactly (0/35,665)** before the swap.
- The two disjoint halves disagree with **each other** on 64 → converged by n=1,159; the residual
  is systematic (pool composition), not sampling noise. Pools disagree with each other on 459.
- Only the 14,266 ejected images can change; only novel-folder ones can change accuracy (a
  trained-class image on the novel route is wrong either way) → **swing bounded at 0.63 pp**.
- Unscored. Converting the bound to a measurement costs one of thirty submissions.
- Four couplings remain: `zc` member standardization, `z1` gate standardization, the rank quantile
  (strongest), Sinkhorn. Paper now says **five** coupling points, not three.

**The offsets themselves do NOT transfer well — only the predictions do.** Spearman(reference h_c,
test-batch h_c) per leg: pseudo pool **0.39–0.98, median 0.86**; kept pool **0.37–0.91, median 0.69**.
Cause: at tc=0.05 the across-image softmax is concentrated — a median class draws its offset from an
effective sample of **~90 of 35,665 images** (p05=35, p50=90, p95=235, measured on ctftshift×taxon).
So hub structure is only partly intrinsic to the class embedding; the fused decision simply does not
need it recovered precisely, because ten legs are summed. The term matters in aggregate (deleting it
costs 1.60 novel points) while its per-class values are worth 296 predictions. Do **not** state this
as "a validation set can identify the hub classes" — it can't, at the per-class level.

Paper: new §4.7 subsection + Table 3 (all later tables renumbered +1, now 14). PDF is 12 pages.

## STATUS: revision applied 2026-08-22

Done: 1-8 and 10. Item 9 partial — abstract now leads with the bound rather than the absolute
claim, title shortened to "Zero-Shot Species Recognition at 17,393 Classes: What a Held-Out Proxy
Did Not Predict", but the abstract still carries several threads.

NOT addressed (would need new compute / a second benchmark, i.e. real experiments):
- a clean untouched test split, or a second benchmark for the proxy-reliability claim
- re-running the pipeline with flagged near-duplicate photographs removed from the bank
- an inductive (strictly per-image) run at the *current* configuration
- clean BioCLIP / BioCLIP-2 zero-shot + retrieval baselines
- throughput / memory / energy measurements
These are named explicitly in Sec. 11 rather than glossed.
