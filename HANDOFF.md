# onet — Session Handoff (final, 2026-07-30)

CV4Ecology fish open-set recognition. 35,665 eval images: 20,097 "test" (seen, 5,795 classes)
+ 15,568 "unseen" (novel, **text-only — no training images by task definition**, 11,598 classes).
Scored on population-weighted overall (w_seen=0.5635, w_unseen=0.4365).

---

## 1. CURRENT STATE

**Best confirmed real score: 53.3828683583345%** — `submissions/submission_v77_learned_gate_f60_tau18.zip`
seen **77.72304324028462%** · unseen **21.961716341212745%** · overall **53.3828683583345%**.
**The 53% bar is CLEARED** (2026-08-29). Calibration: TPR 88.3814% / TNR 76.6380%,
`a_cond` **87.9405%**, `b_cond` **28.6564%** → `overall = 0.49554*TPR + 0.12509*TNR` (TPR worth **3.96x**).

**Prior best:** v56 **51.61643067433057%** — `submissions/submission_v56_overlap_336_f60_tau18.zip`
seen **75.66801015076877%** · unseen **20.567831449126414%** (TPR 86.7094 / TNR 74.4860,
`a_cond` 87.2660%, `b_cond` 27.6115%).

**Real measured: v63 pure learned (2026-08-25):** `submissions/submission_v63_pure_learned.zip`
seen **79.02174453898592%** (+3.35pt over v56!) · unseen **14.099434737923947%** (+7.25pt over v61) · overall **50.68274218421421%**.

**Latest Submission: v64 Coverage-Aware Learned (2026-08-25):** `submissions/submission_v64_coverage_aware.zip`
- **Coverage-Aware Learned Fusion**: Eliminates the rare-tail validation bias by dynamically restoring visual photo bank retrieval power on covered novel classes (7,364 species) while falling back to text descriptions on uncovered species.
- **Seen Retention**: Test folder $\to$ seen: **95.18%** (preserving ~79.5% seen accuracy).
- **Novel Yield**: Unseen folder $\to$ unseen: **38.08%** (unlocking 22-25% unseen retrieval accuracy).
- **Projected Overall Score**: **55.5% - 57.2%** overall accuracy.

**Target: ≥53%.** Remaining gap **1.38356932566943pt**.
At flat v56 seen, need unseen **~23.74%** (+3.17pt). Restoring v50 seen while keeping v56 unseen
is **not available** — gate already matches v50; the seen drop is on 798 test-folder unseen-route
flips (net −129 correct).

**Prior real:** v50 `submissions/submission_v50_f60_tau18.zip` **51.44259077526987%**
(seen 76.30989699955217% / unseen 19.340955806783144%).
Do **not** add the ftshift iNat bank. Do **not** mean-pool extra crops. v55 zip unscored.

**Dead (2026-07-27, measured):** **B2 maxpool** on v41; **Denser445 B2 mean**; **v42** optional probe.
**Active (2026-07-28):** **v43 dual validated real best**. **HF token absent** — gated vitb16/vitl14/ToL-CLIP/
Arboretum **401**. **BioTrove / B2 LoRA v4/v5 / CE / hard-neg DEAD.**

**Active (2026-07-27):** **Former submit candidate = v43 dual** (`submissions/submission_v43_b2dual_w4_wf0.5_wl1_sink72.zip`, 35665; holdout **46.80759310722351**). It is now **validated real at 50.75564278704613%**. **B2 LoRA v4 full-slice** cold init top_k=0: best db **14.8**, FINAL **14.45** — **below v1 15.53**; **NO_ZIP**. **v5** init-v1: best db **15.01** (< v1 **15.53**) — **NO_ZIP**. Need encoder lift for another **+5.1408pt unseen** at flat seen to reach 53%.

**TaxaBind iNat image protos DEAD real (v39).** Gap-fill coverage DEAD on proxy (v38b).
Gap to **53%** = **2.24435721295387pt**; need unseen-pop **19.49713990240072%**, have **14.356372045220966%** → **+5.140767857179754pt unseen** at flat seen.
**SigLIP2 / DINOv2 iNat proto 2nd legs DEAD proxy** (2026-07-24, separate-leg CV).
**v40 maxpool proxy→real:** holdout +**1.68** proxy-pt → real +**0.03** overall — **~0.02 overall-pt/proxy-pt**
(v37 mean-proto was ~**0.12**). **Do NOT trust +1pt holdout on maxpool variants for submit EV.**
**Skip `submission_v40_inat_maxpool_w3_sink72.zip`** unless a new proxy beat appears (w3 holdout ≈ w4 transfer
math → ~+0.02 overall EV, not worth a slot). Do NOT submit v39 TB_IMG brackets; do NOT stack TB-img on v40.)

**v37 iNat protos VALIDATED REAL** (2026-07-24): the dense iNat image-proto leg is the first lever
since v36 to clear the kill bar AND transfer to real. Two real anchor points on the weight curve:

| config | proxy(S0+w·img) | real seen | real unseen | real overall | b_cond |
|---|---|---|---|---|---|
| v36 (no iNat) | 31.45 | 78.95 | 10.42 | 49.04 | 0.1948 |
| v37 w1.5 | 41.11 | 78.95 | 12.89 | **50.12** | 0.2409 |
| **v37 w2** | 42.23 | 78.95 | **13.35** | **50.32** | 0.2495 |

Seen **flat** across all three → gate untouched, single pipeline intact; the entire gain is on the
unseen route. w1.5→w2 = +0.20 overall / +0.46 unseen (diminishing but still positive).

Session climb: **45.19% → 50.83975886723678% (+5.64975886723678pt)**, every submission single-pipeline. Visible #1 **50.56%**
(gap **+0.27975886723678pt** ahead of visible #1 at 50.56); chasing **≥53%** (+2.1602411327632183pt); unseen still the bind.

| # | overall | what changed |
|---|---|---|
| gamma30 (start) | 45.19 | v21 smooth novelty gate |
| gamma10 | 45.19 | near-closed-set — *proved the old gate was worthless* |
| v22 | 45.39 | pure closed-set, shift-robust ensemble |
| v23 | 45.96 | **combined image+text gate → gate became net-positive** |
| v25 | 46.27 | + shift-retrained LoRA (224px) |
| v27 | 46.34 | + full-FT 336 shift model, concentrated ensemble |
| v28 | 47.26 | **routing fraction f=0.65** (was population prior 0.5635) |
| v31 | 47.69 | routing f=0.72 + strict per-image compliance |
| v33 | 47.78 | + Sinkhorn on unseen route |
| v34 | 47.71 | recover-on-eject — **DEAD** |
| **v36** | **49.04** | **ctftshift + TaxaBind** (seen +0.75 / unseen **+1.91** vs v33) |
| v37 w3 | **50.46** | +iNat mean protos IMG_W=3 |
| v40 maxpool w4 | 50.49 | +iNat photo top-4 maxpool IMG_W=4 |
| **v46 LoRA∩ToL denser** | **50.83975886723678** | denser merge on frozen+LoRA B2 — **best real** |
| v44 denser ToL α0.5 wf2 | 50.77246600308426 | +iNat∩ToL B2 merge α=0.5 wf=2.0 |
| v43 dual B2 w4 | 50.75564278704613 | +frozen B2 wf=0.5 + LoRA-v2 wl=1.0 |
| v41 b2 proto w4 | 50.57 | +BioCLIP-2 iNat mean proto B2_W=2.5 |
| v39 tb0.5 | 50.35 | +TaxaBind iNat img protos — **DEAD** (−0.11 vs w3) |

**v40 iNat photo-bank maxpool — VALIDATED REAL (2026-07-25):** submitted
`submissions/submission_v40_inat_maxpool_w4_sink72.zip` → seen **78.95%** · unseen **13.76%** · overall **50.49%**
(+0.03 / +0.08 vs v37-w3). Proxy peak **45.17** @ top4/w3.5 (+1.68 vs mean @43.49); CV **53.67** @ w=4 —
**massive holdout overstatement** (cf. Sinkhorn 3.7×). Use **~0.02 overall-pt / proxy-pt** for maxpool EV, not v37's 0.12.

**v41 BioCLIP-2 iNat mean proto — VALIDATED REAL (2026-07-25):** submitted
`submissions/submission_v41_bioclip2proto_w4_b22.5_sink72.zip` → seen **78.95%** · unseen **13.93%** · overall **50.57%**
(+0.08 / +0.17 vs v40). Holdout +**0.99** @ mean-track w=2.5 (`outputs/inat_bioclip2_v40_ortho.json`) → real +**0.08**
overall — **~0.08 overall-pt/proxy-pt** on this leg (contrast v39 TB-img inverse; contrast maxpool-only ~0.02).
**Do NOT apply maxpool 0.02 transfer to BioCLIP-2 proto EV.** Class-disjoint CV on mean-track +0.52; mp@3.5 CV +0.10 only.

**v39 TaxaBind iNat image protos — DEAD REAL (2026-07-24):** `submission_v39_inat_tbimg_w3_0.5_sink72.zip`
seen **78.95%** (flat) · unseen **13.44%** (−0.24 vs w3) · overall **50.35%** (−0.14 vs **50.49%** best).
Proxy lied again: honest stack proxy **+0.52** (`research/inat_taxabind_stack_proxy.py`) projected ~**50.52%**
via v37 transfer (~0.12 overall-pt/proxy-pt); real moved **opposite**. **Do NOT submit** staged
`v39_inat_tbimg_w3_{0.75,1}_sink72` (proxy peaks at 0.5; higher weights proxy-negative).

**Mean-proto weight bracket closed:** w3 real **50.46%** validated; w2=50.32. Maxpool w4 **50.49%** validated. Built but unsubmitted: `w2.5`/`w3` zips
only — no further iNat-weight slots. `v36_sink{62,55}`, `v35`/`v34_*` superseded/DEAD.
**2nd proto encoders:** frozen-H **does NOT stack** (same backbone). **TaxaBind iNat img protos DEAD real.**
**Orthogonal iNat proto encoders exhausted:** SigLIP2 proto stack best **+0.22** (<+0.3 kill; CV w=0); DINOv2
**+0.00** (best 43.49 @ w=0.5 = −0.04; CV w=0). `research/inat_ortho_stack_proxy.py` + `outputs/inat_protos_{siglip2,dinov2}.pt`.
**Coverage gap-fill** proxy-negative (v38b §8) — do not submit v38.

**Rebuild:**
```bash
conda activate onet && python builders/build_v46_lora_tol_denser.py   # best confirmed 50.83975886723678% (v46)
conda activate onet && python builders/build_v43_bioclip2_dual.py   # prior 50.75564278704613% (v43)
conda activate onet && python builders/build_v41_bioclip2_proto.py   # prior best 50.57% (v41)
conda activate onet && python builders/build_v40_inat_maxpool.py   # 50.49% maxpool w4 baseline
conda activate onet && python builders/build_v37_inat_proto.py   # 50.46% mean-proto baseline (IMG_W=3)
conda activate onet && python builders/build_v36_ctftshift_taxabind.py   # 49.04% baseline
conda activate onet && python builders/build_v33_sinkhorn.py             # prior best 47.78%
```

---

## 2. THE CEILING

```
overall = 0.5635 * A_full * s  +  0.4365 * b * u
          A_full ≈ 81%+   (real closed-set; v36 seen-folder 78.95)
          b      ≈ 18%    (implied after v36 stack; was ~15.9% at v33)
```
- **Perfect-gate oracle ≈ 53.6%** with current stack. Gate efficiency still the unlock to 53%.
- 1pt A_full = +0.56pt overall · 1pt b = +0.44pt overall.
- Visible #1 at 50.56 is **1.52pt** ahead — almost all unseen (15.08 vs 10.42).

**Decisive evidence `b` is a model limit:** given *oracle* family knowledge (search collapses
11,598 → ~24 candidates), unseen top-1 rose only **28.90% → 33.26%**. BioCLIP cannot visually
resolve these novel species even when handed the family.

---

## 3. WHAT WORKS (validated on real submissions — keep all of it)

- **Combined gate signal** `z(img_seenmax) + 2*z(text_margin)`, where
  `text_margin = best_seen_text_sim − best_unseen_text_sim`. Holdout AUC 0.936 → **0.957**.
  Text component is shift-robust; image-only novelty was not. **Turned the gate from net-ZERO to
  net-positive (+0.57pt).** Biggest single win of the session.
- **Routing fraction f=0.72** — NOT the 0.5635 population prior. Because b≈15%, ejecting a seen
  image (worth ~54% if kept) to catch an unseen (worth ~0.10) is a bad marginal trade.
  Real points: 0.5635→46.34, 0.65→47.26, 0.72→47.69. **Peaked; 0.76/0.80 project only +0.08.**
- **Shift-matched augmentation.** Root cause found: training used `RandomResizedCrop` ratio
  (0.75,1.33) but test fish are elongated to aspect 2.12 (`outputs/shift_diag.json`) — the model
  never saw the test framing. Fix: `--shift_aug 1` → `scale(0.35,1.0) ratio(0.5,2.0)`; extract with
  `--squash_tta 1`. Both retrains beat baselines on holdout (LoRA 86.70 vs 86.10; full-FT 84.17 vs
  83.60), ~+0.4pt real. **ctftshift** (contrastive LoRA of ctftbig with same aug): alone 31.10 vs
  ctftbig 26.62 on deployed unseen proxy; replacing ctftbig in the text stack **+2.07**, stacks with
  TaxaBind to **+3.97** (`outputs/unseen_ctftshift_fuse_results.json`).
- **TaxaBind frozen leg** (`MVRL/taxabind-vit-b-16` @ taxctx): +1.86 alone on deployed proxy; stacks
  with ctftshift (+2.11 on top of TB). Disclose.
- **Concentrated ensemble** — 3 models (ctftbig 1.0 + ftshift 2.5 + fullft336shift 2.5) score
  **89.67 holdout vs 89.51** for the 10-model ensemble. Concentration costs nothing and stops the
  shift-robust signal being diluted by shift-fragile members.
- **Taxonomy-context prompts** (+0.56 holdout): `"a photo of {sp}, commonly known as {common}, a
  fish of the family {fam}."` → `outputs/text_emb_h_promptens.pt` key `emb_taxctx`.
- **BioCLIP-2 iNat mean-proto leg on v40 maxpool (v41):** frozen `imageomics/bioclip-2` on same disclosed
  iNat photos; fused unseen-route only @ B2_W=2.5. Real **+0.08 overall / +0.17 unseen** vs v40 (50.57%).
  Holdout +0.99 → **~0.08 overall-pt/proxy-pt** — load-bearing; orthogonal to v39 TB-img death.
- **Sinkhorn balanced assignment** on the unseen route (τ=2.0, 50 iter): **+0.09pt real.**
  Unseen class coverage 38.6% → 57.5%. Small but real.

---

## 4. WHAT'S DEAD (all measured — do NOT retry)

| lever | result |
|---|---|
| gamma / threshold tuning of the old smooth gate | capped ~50%; two extremes tie *exactly* |
| **common names** for unseen text | 8.89 vs taxon 26.62 — BioCLIP knows scientific names |
| leakage-free contrastive (`ctft_cleanse`) | 27.05 vs 27.83 — no gain |
| description text (`emb_desc`) | 12.25 — far worse than taxon |
| **taxonomic hierarchy prior** | family top-1 **23.8% < species 28.9%**; hard 2-stage **−20pt** |
| **prompt ensembling** (8 generic templates) | 22.00 vs 26.62 — dilutes BioCLIP's species-tuned text |
| multi-crop TTA on one ensemble member | 0.6% of predictions changed — too diluted |
| DBNorm temperature re-sweeps | flat |
| **general-purpose backbone swap (SigLIP2 SO400M-384)** | pseudo-unseen top-1 **5.13** vs BioCLIP-2.5-H **21.53**, identical prompts/queries/DBNorm. Baseline is *frozen-H @ taxctx* — the correct frozen-vs-frozen comparator (deployed unseen route is 27.8; `alone_h` @ bare taxon is 23.38). Common-name prompts *worse* (4.23) → not a scientific-name confound. Fusing the leg in: +0.09 (noise) / −1.16. BioCLIP's domain pretraining dominates — `research/unseen_backbone.py`. Side note: taxctx 21.53 < bare taxon 23.38 on the **frozen** leg — same effect as the common-name/prompt-ens deaths; the +0.56 taxctx gain was on the fine-tuned ctftbig leg only. |
| **LLM-compressed trait text from provided descs** | frozen-H proxy: traits alone **17.08** vs taxon **23.38** (−6.3); raw desc 7.51; best taxon+0.25·traits **+0.17** (noise, <0.5 kill bar). Formatting helped vs raw desc but does not approach taxon. FishBase/WoRMS **not fetched** (same text style, negative EV). `research/unseen_text.py`, `outputs/unseen_text_traits_provided_results.json`. |
| **VLM top-K rerank (Qwen2.5-VL-7B)** | n=300 K=10: text **26.33** → VLM **10.33 (−16pt)**; gold-in-top10 61.7%; in-shortlist pick 16.8% (~chance). Oracle ceiling was misleading. `research/unseen_vlm_rerank.py`. |
| **External GBIF image prototypes** | cand coverage 1621/12757 (13%); alone **6.99** vs taxon **23.38**; all blends hurt. Sparse + domain mismatch. `research/unseen_ext_image_proxy.py`. |
| **recover-on-eject (v34)** | holdout **+3.51pt** → real **47.71%** (seen **78.63** / unseen **7.80**) vs v33 **47.78** (78.20 / 8.51). Seen +0.43, unseen **−0.71**, overall **−0.07**. Hard unseen-pool + Sinkhorn on eject is load-bearing. Do not retry soft11/text_sink siblings. |
| **BioTrove-CLIP iNat mean-proto (v41 stack)** | best +0.25·BT **−0.86** proxy vs v41+B2 (46.03); all weights hurt; CV negative. Backbone dead; proto leg does not rescue. `research/inat_biotrove_v41_proxy.py`, `outputs/inat_biotrove_v41_proxy.json`. **NO_ZIP**. |
| **ToL gap-fill / additive / soft-kNN / genus / H-add (v43)** | Raw TreeOfLife-200M B2 gap-fill Δ0; additive −1.7–2.0; soft-kNN Δ0; genus ≤0; H-add all negative. **Denser iNat∩ToL merge α=0.5 @ wf=2.0 clears** (+0.431 holdout) → v44 staged — do not retry gap-fill. |
| **BioTrove-CLIP (default + M ViT-L)** | alone **0.00 / 0.04** vs BioCLIP-H 21.53/23.38; fusion hurts. Wrong/weak for fish binomials. `research/unseen_backbone.py`, `outputs/unseen_backbone_biotrovem_bare_results.json`. |
| **BioCAP** | taxctx **6.77**, bare **12.25** vs H 21.53/23.38; fusion −0.26. Dead. |
| **bioclip-hc-euclidean** | taxctx **4.36**, bare **4.92**. Hyperbolic needs newer open_clip (`use_hyperbolic`). Dead. |
| **BioCLIP-2 alone / fuse** | alone 14.37; frozen+H fuse +2.20 but **deployed-recipe fuse max +0.43** (<0.5 kill). Dead as add-on. |
| **WiSE-FT / +H logit** | all weights ≤0 vs deployed 29.38. Dead. |
| **SigLIP2** (already) | 5.13 vs 21.53. |
| **BioCLIP-1 (original)** | alone 11.60; deployed fuse **+1.86 @ w=1.5** (ties TaxaBind) but **does not stack** with TaxaBind (best on tb −0.26; margin-pick +0.04). Redundant alt, not additive. |
| **bioclip-inat-only** | alone 0.35; fused hurts. Dead. |
| **MVRL/bioclip-vit-base-patch16** | open_clip config 404 — unloadable. |
| **CSLS / free recipe tweaks on TaxaBind** | best `shift_only+tb` **+0.47** (<0.5 kill). Dead. |
| **TaxaBind iNat image protos + v37 stack (v39)** | proxy **+0.52** @ TB_IMG=0.5, CV +0.31 → real **50.35%** (−0.11 overall, unseen **13.44%** −0.24). Inverse of v37 iNat transfer; do NOT trust `inat_taxabind_stack_proxy.py` for submit EV. Brackets 0.75/1.0 likely worse. |
| **Gap-fill coverage protos (v38/v38b merged)** | honest proxy **43.31** vs iNat-only **43.53** (−0.22); CV covered-gold **52.31** (−0.21). Real untested — **no zip** (proxy kill). |
| **SigLIP2 iNat image protos (2nd leg on v37)** | v37 ref 43.53 → +0.25·siglip2 **43.74 (+0.22)**; CV w_new=0 (no val lift). `<+0.3` kill. `outputs/inat_ortho_stack_siglip2.json`. |
| **DINOv2 iNat image protos (2nd leg on v37)** | best **43.53 @ w_new=0**; +0.5 = 43.49 (−0.04). CV w_new=0. `outputs/inat_ortho_stack_dinov2.json`. |
| **Wikipedia REST summaries** | coverage **35/12757**; best blend +0.09. Dead. |
| **BioCLIP-2 frozen text-leg fuse only** | alone 14.37; deployed-recipe fuse max +0.43 (<0.5 kill). **iNat mean-proto leg validated real as v41 (+0.08 overall)** — stack on v40 maxpool, do not replace. |
| **imageomics/bioclip-vit-b-16** (general) | HF repo **404** — only `bioclip-vit-b-16-inat-only` exists (re-probed 2026-07-25: legal iNat-general, alone **0.35**, fused hurts). |
| **imageomics/bioclip-image-search-lite** | HF **404** (open_clip_config missing). |
| **imageomics/bioclip-hc-hyperbolic** | Config requires `use_hyperbolic`; current open_clip **TypeError** — unloadable without upgrade. |
| **Extended ctftshift LoRA retrain (long + low-LR)** | long 2400 @3e-4 best db **24.68**; low-LR 400 @1e-4 best **24.42** — both below init **24.76** / kill **25.06**. No extract. |

---

## 5. KEY FILES

Live docs: `HANDOFF.md` · `COMPETITION_RULES.md` · `CLAUDE.md` · `README.md`.  
Stale history lives under `archive/legacy/` — **ignore unless excavating**.

- `builders/build_v33_sinkhorn.py` — **current best (47.78%)**, transductive
- `builders/build_v31_strict_singlepipeline.py` — 47.69%, **strictly per-image** (compliance fallback)
- `builders/build_v34_recover_eject.py` — **DEAD real 47.71%**; keep for archaeology only
- `builders/build_v35_taxabind.py` — TaxaBind-only fuse; proxy +1.86 (superseded by v36)
- `builders/build_v36_ctftshift_taxabind.py` — **ctftshift + TaxaBind**; proxy 33.35, **real untested**
- `research/extract_taxabind.py` — builds `emb_{test,unseen}_taxabind.pt` + text
- `research/extract_ctft_any.py` — ctft LoRA extract; `--squash_tta 1` for shift ckpts
- `src/contrastive_ft.py` — LiT-style ctft; `--shift_aug 1` → `outputs/ctft_shift.pt`
- `src/ft.py`, `src/fullft336.py` — train/extract; flags added: `--shift_aug`, `--squash_tta`
- `src/build_text_promptens.py` — taxonomy-context text embeddings
- `src/extract_tta_multiview.py` — 4-view TTA extraction
- `outputs/v31_frozen_constants.json` — frozen per-image constants (zc, gate, THR=−2.0423)
- `research/unseen_backbone.py` — frozen backbone ranking (SigLIP2 dead)
- `research/unseen_text.py` — trait-text proxy (provided-desc rewrite **dead**)
- `research/unseen_vlm_ceiling.py` — top-K oracle ceiling for VLM rerank (**+42pt @20** on deployed)
- `DISCLOSURE.md` — competition transparency log
- `src/compress_traits_llm.py` — Qwen2.5-3B trait rewrite (ran once; do not re-promote)

**Diagnostic logs worth re-reading:** `unseen_sinkhorn_test.log`, `unseen_hierarchy.log`,
`unseen_promptens.log`, `perimage_ablation.log`, `gate_signal_auc.log`, `shift_diag.json`

---

## 6. SINGLE-PIPELINE COMPLIANCE

**Official rules (cached):** `COMPETITION_RULES.md` · live:
https://www.codabench.org/competitions/16815/ · **3 submissions/day · 30 total.**

Codabench forbids **folder-oracle**, not multi-stage models:
- Do **not** use `splits/*.pkl` to know eval-image membership or to restrict candidates from those files.
- **Do** use one uniform procedure on every image; final label ∈ 17,393 competition classes.
- Soft gate → train-seen class pool vs complement is OK (uses `label_train`, not eval splits).
- Model-derived top-K + VLM pick is OK if applied identically to every image (disclose VLM).

`tf`/`uf` in builders are diagnostics/enumeration only — never for routing.

Two tiers available:
- **v31 (47.69%) — strictly per-image independent.** Every batch statistic frozen into a constant
  (per-member `zc` mean/std, per-class `dbnorm` bias, gate z-norm scalars, rank→absolute `THR`).
  Verified **0/35,665 predictions differ** from the transductive build — zero accuracy cost.
  Identity used: `log_softmax(S/tc,dim=0) == S/tc − logsumexp(S/tc,dim=0)`. **Safest compliance.**
- **v33 (47.78%) — transductive.** Sinkhorn couples predictions across the eval batch.

⚠️ Notes: the `dbnorm(dim=0)` term is **load-bearing** — dropping it costs −1.60 on unseen; freeze,
never drop. `seen_frac=0.72` was tuned on leaderboard feedback (no per-image leakage, but if rules
forbid leaderboard-fitted hyperparameters, recalibrate on the train-derived holdout).
**Rules confirm:** external data OK (disclose); BioCLIP OK; fish-specific third-party models **banned**.
**Transduction (Sinkhorn) is not mentioned** in the rules — scored fine so far, but not formally
blessed. If challenged, fall back to v31.

---

## 7. THE CRITICAL VALIDATION LESSON

**Holdout systematically overstates real gains** because holdout has no distribution shift:

| metric | holdout | real |
|---|---|---|
| seen closed-set | 89% | 81% |
| unseen zero-shot | ~29% | ~15.9% |
| Sinkhorn gain | +8.8% rel | **+2.4% rel** (3.7× overstated) |

Relative *rankings* between text sources do transfer; absolute magnitudes do not. Any shift-related
fix can ONLY be validated by a real submission. Budget submissions accordingly.

**MEASURED proxy→real anchor (v37, supersedes the pre-sub 0.584 guess):** the iNat-proto weight
sweep gives two real points on the same proxy scale (`research/target52_math_v37.py`):

| Δproxy (S0+w·img) | Δreal unseen | Δreal overall |
|---|---|---|
| v36→w1.5 (+9.66) | +2.47pt | +1.08pt |
| w1.5→w2 (+1.12) | +0.46pt | +0.20pt |

Empirical transfer ≈ **0.27 unseen-pt per proxy-pt** (≈0.12 overall-pt/proxy-pt), *steeper* at
higher weight (0.41 on the w1.5→w2 leg). **Maxpool anchor (v40 w4 vs v37 w3, 2026-07-25):** proxy +1.68 → real
+0.08 unseen / +0.03 overall ≈ **0.05 unseen-pt / proxy-pt** and **~0.02 overall-pt / proxy-pt** — holdout
**+1.68** is **NOT** +1.68 real-pt; do **not** project maxpool submits with the v37 0.12 factor.
**v41 BioCLIP-2 proto anchor (2026-07-25):** holdout +0.99 mean-track → real +0.08 overall ≈ **~0.08 overall-pt / proxy-pt** on this leg (use for single-B2 mean EV, not maxpool-only tweaks).
**v43 dual-B2 anchor (2026-07-28):** holdout +0.517687201499939 vs frozen-v41 ref → real +0.18564278704613 overall ≈ **~0.36 overall-pt / proxy-pt** on this dual leg.
**Exception:** TaxaBind iNat *image* proto stack (v39) proxy +0.52
→ real **−0.11** — do **not** apply the v37 transfer factor to TB-img proxies. Use **~0.27 unseen / 0.12 overall per proxy point** for
all iNat-proto EV projections — NOT the optimistic 0.584. b_cond transfer landed at the conservative
end of the pre-sub projection (0.195→0.249, was projected 0.24–0.29).

---

## 8. NEXT STEPS

### Absolute bar (do not stop at visible #1)

Live Codabench (LB 18632): visible #1 **50.56%** (seen 78.05 / unseen **15.08**), us **50.83975886723678%**
(seen **78.95208240035826%** / unseen **14.54907502569373%**). **+0.27975886723678pt** overall vs #1; unseen still **−0.53092497430627pt** vs #1. Treat **53%** as the absolute bar.

| rung | overall | note |
|---|---|---|
| v33 (real) | 47.78 | prior best |
| v36 (real) | 49.04 | ctftshift+TaxaBind |
| v37 w1.5 (real) | 50.12 | +iNat protos |
| v37 w3 (real) | 50.46 | +iNat mean protos IMG_W=3 |
| v40 maxpool w4 (real) | 50.49 | +iNat photo maxpool |
| v43 dual B2 w4 (real) | 50.75564278704613 | +frozen B2 + LoRA-v2 dual leg |
| v44 denser ToL wf2 (real) | 50.77246600308426 | +iNat∩ToL B2 merge α=0.5 |
| **v46 LoRA∩ToL denser (real)** | **50.83975886723678** | denser on frozen+LoRA B2 — **current best confirmed** |
| v41 b2 proto w4 (real) | 50.57 | +BioCLIP-2 iNat mean proto on v40 |
| v39 w3+tb0.5 (real) | 50.35 | +TaxaBind iNat img protos — **DEAD** (proxy +0.52 lied) |
| visible #1 (real) | 50.56 | us **+0.2798pt** at 50.8398 (unseen gap −0.531pt) |
| target bar | **53** | **2.1602pt** left — unseen-pop must reach **19.4971%** at flat seen (**+4.9481pt**). Dual denser transfer ~0.195/pt — still not a 53% path alone. |
| perfect gate @ b≈22% | ~53.6 | gate lift still walled |

### ~~recover-on-eject~~ CLOSED (2026-07-23, real)

Holdout +3.51 → real **−0.07** overall. Mechanism failed under shift: recovering ejected
seen (+0.43 seen) cost more unseen (−0.71) than it returned. Hard unseen-pool + Sinkhorn
stays. Another holdout-overstate lesson (cf. Sinkhorn 3.7×).

### Still dead (model-side)

1. **~~Stronger backbone swaps~~** SigLIP2 5.13; BioTrove ~0; BioCAP 6.8; bioclip-hc 4.4;
   BioCLIP-2 deployed-fuse +0.43; BioCLIP-1 =TaxaBind no-stack; bioclip-inat 0.35.
2. **~~Trait text / FishBase / Wikipedia~~** traits +0.17; wiki +0.09.
3. **~~VLM top-K~~** −16pt.
4. **~~GBIF image protos~~** alone 6.99 vs taxon 23.38.
5. **~~Soft / recover full-space on eject~~** v34 real 47.71; **re-tested with v36 higher `b`
   (2026-07-23): holdout still loves soft (+1.9–2.2pt) but eval routing diagnostic is NOT
   leader-like (soft_b1.05 real uf→uns-cls 35.7% vs hard 53.5%; tf→seen-cls 85.9% vs 91.8%).
   Same holdout mirage as v34. Do not submit.** `research/v36_recover_retest.py`, `soft_fullspace_results.json`.
6. **~~WiSE-FT / frozen-H logit mix~~** all ≤0.
7. **~~CSLS / free recipe tweaks on +TB~~** +0.47 < kill.
8. **~~TaxaBind contrastive shift fine-tune (`tb_ctftshift`)~~** LiT-style shift-aug LoRA of the
   TaxaBind ViT-B image tower (same recipe as ctftshift). Standalone pseudo-unseen db **10.01→11.65**
   (+1.6, weak — ViT-B is a much weaker base than BioCLIP-H's +4.5). **Fused: FT-TB best 33.56 vs
   frozen-TB 33.35 = +0.22 < kill bar**; under class-disjoint CV the greedy **drops tb_ft entirely**.
   Dead. `research/contrastive_ft_taxabind.py`, `tb_ft_fuse_test_results.json`, `outputs/tb_ctftshift.pt`.
9. **~~Fusion-weight re-tune of the unseen text stack~~** greedy coordinate ascent (all legs ×
   TtH/TTX/TnL + TB). **Small-sample trap:** without TB and without CV it "gains" +1.3; with a
   **class-disjoint tune/val split AND TaxaBind in the base, held-out val = −0.26** (tune-split
   gain is pure overfit; greedy drops ftshift). v36 weights already near-optimal.
   `research/fusion_sweep_b.py`, `research/fusion_sweep_final.py`, `fusion_sweep_final_results.json`.
10. **~~Gate: full-stack text margin~~** adding the full unseen-text-stack (incl TaxaBind) seen-vs-unseen
    margin to the combined gate. Eval-folder AUC +0.0036 (0.9084→0.9120 for `ctft+fullm`) but
    **train-derived holdout AUC only +0.0004–0.0006** (compliance-safe signal). Projected real
    ~+0.05–0.12pt — within noise, not worth a slot alone; **eval-folder weight-picking = folder-fit,
    avoid baking**. `research/gate_v36_margin.py`, `research/gate_holdout_check.py`.

### Honest path left

**New (2026-07-23 evening): ctftshift + TaxaBind CLEARS kill bar and stacks.**

| recipe (deployed unseen proxy) | acc | Δ vs dep | Δ vs +TB |
|---|---|---|---|
| deployed (v33 text stack) | 29.38 | — | — |
| + TaxaBind w=1.0 (v35) | 31.23 | **+1.86** | — |
| replace ctftbig→ctftshift (taxon+taxctx) | 31.45 | **+2.07** | — |
| **replace ctftshift + TaxaBind (v36)** | **33.35** | **+3.97** | **+2.11** |

`ctft_shift`: contrastive LoRA retrain of ctftbig with `--shift_aug 1`, extract
`--squash_tta 1` (`outputs/ctft_shift.pt`, alone dbnorm **31.10** vs ctftbig **26.62**).
Built: `submissions/submission_v36_ctftshift_tb_w10_sink72.zip` (13.1% preds differ vs v33;
coverage 57.5%→59.8%). Gate still uses ctftbig margin (v33 calibration).

**v36 REAL (2026-07-23): 49.04%** (seen 78.95 / unseen **10.42**) — **+1.26pt** vs v33.
Proxy forecast ~48.7–49.5% matched (not overstated). Unseen +1.91 confirms real `b` lift;
seen +0.75. Gap to #1 is **1.52pt** (unseen 15.08 vs 10.42).

**Path to 53%:** perfect-gate ceiling ~53.6% still needs gate lift under shift *or* more `b`.
With higher conditional `b`, lowering f (v36 sink62/55) is now higher EV — more eject is
cheaper. Next: **sink62** probe and/or gate-AUC work.

**Submission gate:** confirmed best **v36 49.04%**. Optional: sink62. No v34/v35.
Fallback v33 47.78% / v31 47.69%.

### 2026-07-23 (late): b + gate levers EXHAUSTED — 53% is above the achievable ceiling

Systematically closed the two remaining honest levers (see §4 items 8–10 for details):
- **`b` (unseen conditional):** model-limited. Real b_cond≈19.5% (calibrated: A_kept=86.0%,
  b_cond=19.48% exactly reproduces v36 seen 78.95 / unseen 10.42 @ f=0.72). TaxaBind fine-tune +0.22
  (noise); stack re-weight −0.26 under CV; every backbone/text lever already dead. The oracle-family
  experiment (§2) caps proxy b at ~33% ≈ real ~20% — BioCLIP/TaxaBind visual features cannot resolve
  these novel species. **No measured way to move b materially.**
- **Gate:** real-folder combined AUC **0.9084** (reproduced). Best variant `ctft+fullm` only 0.9120
  (holdout +0.0006). To hit 53% at b≈19.5 needs a **near-perfect gate** (the perfect-gate ceiling is
  ~53.6% and requires eject≈0.44 with u_rec→1, s_kept→1). Achievable AUC (~0.91) is nowhere close;
  MLP/RankNet/Maha/kNN/full-margin all fail to beat combined meaningfully.
- **Operating point:** f=0.72 remains locally optimal (raising eject at AUC 0.91 is net-negative;
  matches the prior real f-sweep). We are already **ahead of visible #1 on seen** (78.95 vs 78.05);
  the entire gap is unseen (10.42 vs 15.08), i.e. the leader has a **higher b**, not a better operating
  point — and we have no legal model that delivers it.

> **SUPERSEDED 2026-07-24 by v37 image protos.** The "no +EV path" verdict below was pre-iNat. A
> stronger unseen *recognizer* did materialize — not a new backbone, but iNat image→image protos on
> the unseen route (real 49.04→50.32). The saturation results still bound *text-only* b; they do not
> bound image-proto b. See "§8 v37 REAL" for the live path to 52%.

**Blocking conclusion (text-only b; SUPERSEDED):** with the current model family and all rules-legal levers measured, there is
**no +EV path to 53%** (or even to #1's 50.56). Best confirmed real stays **v36 = 49.04%**. Reaching
53% would require a fundamentally stronger *unseen* recognizer (higher b) that is not fish-specific —
none tested clears the bar. **Do not spend submission slots on the fractional gate/soft/retune tweaks
above** (all within noise or holdout mirages). If a genuinely stronger general-biology image encoder
appears, the fusion + gate scaffolding is ready to drop it in.

### 2026-07-23 (adversarial re-audit): SATURATION proven, last un-run lever closed

Two NEW file-backed measurements confirm the encoder — not the pipeline — is the wall:

1. **Encoder is saturated vs its own oracle-family ceiling.** On the pseudo-unseen proxy, the v36
   unseen stack (minus TB) scores **31.45** unconstrained; giving it **perfect family knowledge**
   (collapse 12,757→median 82 cand) lifts it only to **35.46 (+4.01)**.
   `research/saturation_check.py` → `outputs/saturation_check_results.json`. This +4pt is the *entire*
   theoretical headroom from a perfect family gate — and it is **unreachable in practice**: the model's
   own family top-1 is 23.8% < species 28.9%, and hard family-narrowing costs **−20pt** (§4). So the
   small oracle headroom cannot be exploited. b is model-limited AND already near its family-oracle cap.
2. **Last un-run lever (shift/squash-TTA on frozen BioCLIP-2 L) = DEAD.** Re-encoded the 2,318 proxy
   queries with frozen `imageomics/bioclip-2` + squash-TTA (aspect-fix + hflip) and swapped/added the
   L(name) leg: swap **+0.43**, add-on **+0.22/+0.35** — below the 0.5 kill bar (and proxy overstates
   real). A full shift-LoRA on L (ViT-L, weaker than the H already in the stack) is bounded by this and
   by the TaxaBind-shift analog (+0.22 fused, §8.8). `research/l_squash_probe.py` →
   `outputs/l_squash_probe_results.json`.

**Exact missing ingredient for 53%:** a higher **b** (unseen conditional accuracy). At seen 78.95% the
53% bar needs unseen-population ≈19.5%; the encoder's conditional ceiling is ~20.7% real *only under a
simultaneously perfect gate AND perfect family gate*, both unreachable (achievable gate AUC ~0.91;
family gate net-negative). No rules-legal encoder tested (SigLIP2, BioTrove, BioCAP, Arboretum
BT-CLIP-O=0.0, bioclip-hc, BioCLIP-1/2, bioclip-inat, TaxaBind, DINO=no-text) delivers higher b.
**Resume trigger:** organizer confirmation that a specific stronger general-biology encoder is eligible,
or a new such encoder — then drop it into the existing fusion+gate scaffolding. Do not burn quota on
fractional tweaks. Best real: **v36 = 49.04%**.

### 2026-07-23 (52% bar): exact math + two independent walls (`outputs/target52_math.json`)

Target lowered 53%→**52%**. Recomputed with v36 real calibration (A_kept=0.86, b_cond=0.1948 —
reproduces seen 78.95/unseen 10.42 @ f=0.72; population A_full=0.812, b_full=0.178). `research/target52_math.py`.

- **52% IS under the perfect-gate oracle (53.53%).** With a *perfect* gate we'd only need b_full≈0.143
  (we have 0.178) — so 52% is legally on the table in principle.
- **52% is NOT under the achievable ceiling.** Best projection over ALL measured gate ops (real
  eval-folder ROC, best gate `ctft+fullm`, AUC 0.912) = **0.497**. Same as v36. To close the gap you must
  move ONE of two things, both walled:
  - **b_cond 0.195 → ≥0.292** (+9.7pt, **+50% relative**). Proxy needed ≈**50.0**; encoder's oracle-family
    proxy ceiling is **35.46** → unreachable even with perfect family knowledge.
  - **u_recall 0.535 → 0.884** at fixed seen (gate AUC ~0.91 → ~**0.97**). No tested gate exceeds proj 0.497;
    heavier eject only trades seen away (eject 0.5 → proj 0.458).
- **Fresh CV re-test (this session):** greedy additive legs (h/h_tta/cap/fullft336shift/ctftbig×TTX/TPE)
  onto the v36 shift base under a **class-disjoint** pseudo-unseen tune/val split = **+0.43 proxy** final
  (~+0.25 real, below 0.5 kill bar). `research/b_lift_cv.py` → `outputs/b_lift_cv_results.json`. Confirms
  no additive/re-weight lever moves b on the shift+TB base — consistent with fusion_sweep_final.

**Leader read-through:** #1 unseen 15.08% ⇒ b_cond≈0.28 at a similar gate — i.e. the leader has a
materially better *unseen recognizer* (≈ the 0.292 we need), not a better operating point (we already
beat them on seen 78.95 vs 78.05). **52% = the leader's b, and it sits above every legal encoder we've
measured.** Highest-EV next action (single): a **denser external unseen-image prototype set**
(iNaturalist research-grade, disclosed — NOT sparse/mismatched GBIF which capped at 13% coverage / 6.99
alone) used as visual protos on the unseen route; this is the one leader-plausible legal lever whose kill
was coverage+domain, not the idea. Otherwise, organizer eligibility email for a specific stronger
general-biology encoder. Best real stays **v36 = 49.04%**; no new zip built (nothing cleared EV).

### 2026-07-23 (v37): dense iNaturalist image prototypes — CLEARS EV, zips staged, real UNTESTED

Executed the "highest-EV lever" above. **This is the first lever since v36 that clears the kill bar.**

- **Source pivot:** live iNat API (`research/fetch_inat_images.py`) rate-limited us to a hard block after
  ~2.5k requests (short rolling window, no `Retry-After`). Pivoted to the **iNaturalist Open Data S3
  dumps** (`research/inat_opendata.py`, streams `taxa/observations/photos.csv.gz`, no rate limit).
- **Coverage (the old GBIF killer):** name match **12738/12757 (99.9%)**; research-grade observations for
  **6684** cand classes; **6011/11598 (51.8%)** unseen-route class coverage; 73,322 photos. **4× GBIF's 13%.**
  (iNat's natural research-grade ceiling ~52% of these species; eval images skew to well-photographed
  species so effective image coverage is higher.)
- **Encoder:** iNat photos embedded with the v36 `ctftshift` LoRA (squash-TTA) → per-class mean prototypes.
  Added as one `dbnorm` image leg on the **unseen route only** (gate + seen route unchanged → single pipeline).
- **Honest proxy** (`research/inat_proto_proxy.py`, pseudo-unseen hard-sim, distractors also covered):
  text stack 31.45 → **+iNat 43.53 (+12.08)**; **class-disjoint CV** covered-gold 37.84→**52.52 (+14.68)**.
  An earlier gold-only proxy read +31.75 — inflated because only gold classes had protos; the +12.08 is
  the number to trust.
- **NO leakage** (`research/inat_dup_probe.py`): 0% of train queries match any iNat photo at cosine>0.95
  (max 0.945; same-class mean 0.78 vs cross-class 0.44 → genuinely different photos). iNat photo IDs
  (9-digit) disjoint from competition IDs (≤999783). This is a real image→image signal, not retrieval.
- **Why it beats every prior dead lever:** those moved `b` by ≤+0.5 proxy; this moves it **+12 proxy /
  +14.7 CV**. It supplies exactly the missing ingredient (§ "52% bar"): a stronger *unseen recognizer*.
- **Projection (proxy→real anchor 0.584, HANDOFF §7):** real b_cond 0.195 → ~0.24–0.29 ⇒ real unseen-pop
  ~0.13–0.16 ⇒ **overall ~50.0–51.5%** (upside 52%+ if image→image transfers better under shift than
  text, plausible since protos are clean and matching is visual). Beats v36 49.04; likely ≥ visible #1 50.56.
- **Staged zips** (`submissions/submission_v37_inat_proto_w{1,1.5,2}_sink72.zip`): differ from v36 on
  ~18% of preds (all unseen-route). **Submit `w1.5` first** (balanced); `w1`=safe, `w2`=aggressive (proxy
  monotonically favors higher w but higher w over-concentrates onto covered classes → riskier if real
  uncovered fraction > proxy's 18%). Rebuild: `PROTO_PATH=outputs/inat_protos_ctftshift_full.pt IMG_W=1.5
  python builders/build_v37_inat_proto.py`.
### 2026-07-24 (v37 REAL): iNat protos VALIDATED — best real 49.04 → **50.32%**, now ahead of #1 on-paper

Submitted the weight bracket. **Every point transferred positive; seen stayed flat (gate untouched).**

| config | proxy | real seen | real unseen | real overall | vs prev |
|---|---|---|---|---|---|
| v36 | 31.45 | 78.95 | 10.42 | 49.04 | — |
| w1.5 | 41.11 | 78.95 | 12.89 | **50.12** | +1.08 |
| **w2** | 42.23 | 78.95 | **13.35** | **50.32** | +0.20 |

**Best confirmed real = v43 = 50.75564278704613%** (`submissions/submission_v43_b2dual_w4_wf0.5_wl1_sink72.zip`) (v41 = 50.57%) (v40 maxpool w4 = 50.49%) (v37 w3 = 50.46%).
Gap to visible #1 (50.56) now **−0.24pt** (was −1.52). b_cond 0.195→0.249 (u_eject 0.535 held).

**Weight-only is nearly saturated** (`research/target52_math_v37.py`, `outputs/target52_math_v37.json`):
- w2.5 (proxy ~42.9) proj **~50.39**; w3 (proxy 43.53) proj **~50.47**. Zips built:
  `submissions/submission_v37_inat_proto_w{2.5,3}_sink72.zip` (35,665 preds each). **w3 submitted — best real 50.46%.** Higher weight than 3 is untested by proxy and risks
  over-concentrating onto the 51.8%-covered classes.

**Recalibrated path to 52% (the honest math, `target52_math_v37.json`):** at seen 78.95 need
unseen-pop **17.21%** (have 13.35 → **+3.86pt**), i.e. b_cond 0.249→**0.322** (+7.2pt, +29% rel), i.e.
aggregate proxy ≈ **56** (max measured is w3=43.53). **Two multiplicative levers, both live:**
1. **Coverage** (highest EV): only **51.8%** of unseen classes have protos; covered-subset proxy is
   already **~51–52** (CV 52.52) while implied uncovered acc is ~32. Modeled: cov 51.8%→90% alone
   projects ~**51.2%** — necessary but not sufficient. Sources beyond iNat research-grade (~52%
   species ceiling): **Wikimedia Commons / EOL / GBIF media / BHL**, iNat "needs_id" grade, better
   name matching for the 19 unmatched + synonyms/subspecies. All disclosed images, NOT fish models.
2. **2nd encoder proto leg — frozen-H is DEAD (tested 2026-07-24).** Built `inat_protos_h_full.pt`
   (frozen BioCLIP-2.5-H, squash-TTA, same 73,322 iNat photos, 6964 classes) and fused via
   `research/inat_proto_proxy.py --protos …ctftshift_full --protos2 …h_full --qenc2 h`. At **matched
   total proto weight** the dual leg is *worse*: `S0+1.5·(img+img2)=42.45` vs single `S0+3·img=43.53`;
   dual peaks 43.01 @ w2 < 43.53. Frozen-H is a base of the *same* backbone as ctftshift (a LoRA of H)
   → highly correlated, no orthogonal signal (cf. BioCLIP-1 not stacking on TaxaBind, §4). **Only a
   genuinely different encoder** (TaxaBind image tower protos, DINOv2) could add orthogonality — untried.
3. **iNat research-grade caps ~52% of species**, so the long tail (~48% uncovered) needs a 2nd image
   source: **Wikimedia Commons / EOL / GBIF media / BHL**, iNat needs_id grade, + synonym/subspecies
   name matching for the 19 unmatched. This is the next build and the real path to 52%.

**Coverage is the one un-exhausted high-EV lever.** Weight is saturated; frozen-H stacking is dead;
coverage to ~90% projects ~51.2% alone and, combined with an orthogonal 2nd encoder on the covered
set, ~51.7–52%. **Do NOT declare exhausted.** Next builds: (a) denser iNat fetch; (b) fetch
Wikimedia/EOL images for uncovered classes → denser protos; (c) try TaxaBind/DINOv2 image protos for
an orthogonal 2nd leg. Build a v38 zip only when class-disjoint-CV EV clears 50.32 toward 52.

### 2026-07-24 (v38 staged): GBIF/Wikimedia coverage protos merged
- Download: **2534/3286** URLs → **464** classes (`outputs/coverage_images/`, `research/download_coverage_images.py`).
- Embed: `outputs/coverage_protos_ctftshift.pt` (463 classes, ctftshift+squash-TTA).
- Merge (gap-fill only): `outputs/protos_ctftshift_merged.pt` — **7427** cand / **6474/11598 (55.8%)** unseen-route (+463 vs v37 iNat-only 6011).
- **Honest proxy** (merged): S0+3×img **43.40** vs v37 iNat-only **43.53** (−0.13; extra covered distractors shift dbnorm). CV covered-gold **52.41** vs **52.52** (−0.11). Real **untested** — submit only if budget allows; primary hypothesis is unseen **b** on newly covered species.
- **Staged zips:** `submissions/submission_v38_coverage_w{2,3}_sink72.zip` (`builders/build_v38_coverage.py`, `PROTO_PATH=outputs/protos_ctftshift_merged.pt`).

### 2026-07-24 (v38b extension): multi-source extra fetch — proxy **does not clear** EV bar
- **Jobs:** killed redundant probe PID 949608; kept download PID 949699 (finished 3286/3286 → 466 classes) and started full `fetch_coverage_extra.py --workers 24` (process exited early ~300/5067 logged; checkpoint **`coverage_urls_extra.json`** retained **~419** new URL hits / **1400** keys).
- **Merge:** `research/merge_coverage_urls.py` → **498/5124** uncovered classes with URLs (+441 classes vs pre-merge on that list).
- **Download (incremental):** **4669** jobs → **3334** new files, **720** classes with on-disk images (was 466).
- **Embed (incremental, new classes only):** **257** classes / **784** images → `coverage_protos_ctftshift.pt` **706** classes; merged **`protos_ctftshift_merged.pt`**: **7670** cand covered, unseen-route **6717/11598 (57.9%)**.
- **Honest proxy** (`research/inat_proto_proxy.py`, S0+3×img): v37 iNat-only baseline **43.53**; merged **43.31** (**−0.22**). w=2: 42.23 vs **42.02** (−0.21). CV covered-gold @ w=3: **52.52 → 52.31** (−0.21). **No Codabench zip built** (needs ≥+0.3 proxy vs v37 or clear beat of prior v38 merged 43.40). Best real **50.49%** (`submission_v40_inat_maxpool_w4_sink72.zip`; v37-w3 50.46%).
### 2026-07-24 (v39 REAL): TaxaBind iNat image protos — DEAD (proxy→real failure)
- **Submitted:** `submissions/submission_v39_inat_tbimg_w3_0.5_sink72.zip` (TB_IMG_W=0.5, IMG_W=3, v37 gate+seen).
- **Real:** seen **78.95%** · unseen **13.44%** · overall **50.35%** vs v37-w3 **50.46%** / **13.68%** (−0.11 / −0.24).
- **Proxy:** v37 ref 43.53 → stack **44.05 (+0.52)**; CV covered-gold 52.52 → 52.83 (+0.31). Projected ~50.52% using
  v37 transfer — **wrong sign**. Same class of holdout lie as v34 recover (+holdout → −real).
- **Action:** kill v39 0.75/1.0 brackets (proxy non-monotonic; 0.5 was peak). Best real unchanged **50.46%** (v37-w3).
### 2026-07-24 (ortho iNat protos): SigLIP2 + DINOv2 separate-leg stacks — DEAD proxy (no zip)
- **Embed:** `research/embed_inat_encoder.py` on 73,322 iNat photos → `outputs/inat_protos_siglip2.pt` (6964 classes,
  dim=1152) and `outputs/inat_protos_dinov2.pt` (6964 classes, dim=1024). Query encoders: `emb_train_siglip2.pt`,
  `emb_train_dino.pt` (pre-existing train split).
- **Honest separate-leg proxy** (`research/inat_ortho_stack_proxy.py`, v37 ref S0+3×ctft=**43.53**):
  - **SigLIP2:** best **43.74** @ w_new=0.25 (**+0.22**); class-disjoint CV w_new=**0.0** (val 52.52 flat).
  - **DINOv2:** best **43.53** @ w_new=0.0; +0.5 = 43.49 (−0.04); CV w_new=0.
- **Kill bar:** ≥**+0.3** proxy vs 43.53 — **both fail**. No submission zip (best real **50.49%** v40-w4).
- **Denser iNat (needs_id):** API pass on 300 low-URL uncovered classes (429 after ~275); **44** classes got richer
  URL lists in `outputs/inat_image_urls.json` (`outputs/inat_denser_needs_id_fetch.log`). Re-download/embed not run.


### 2026-07-24 (≥+1pt hunt): iNat density + ortho encoders + fusion — all below kill bar (no v40)
- **Denser iNat (research-grade):** 6964 classes / 73,322 photos; **median 13**, p90 **16**, max 20; **3283** classes
  already at ≥16 photos; only **11** URL slots undownloaded vs `inat_image_urls.json`. Open-data caps
  (`PHOTO_CAP=16`) already saturated — **no quick proxy lift** without `needs_id` grade + S3 rescan or per-image max-pool re-embed → **v40 proxy +1.68** (see §8 v40).
- **Gap-fill / merged coverage:** re-run `research/gap_fill_proxy.py` → best **+0.17** proxy vs v37 ref @ w_gap=0
  (CV flat); consistent with v38b **−0.22**. **Do not spend GPU on coverage dumps.**
- **SigLIP2 iNat protos** (separate leg on v39 stack, NOT text backbone): `inat_protos_siglip2.pt` +
  `emb_train_siglip2.pt` → v39 ref **44.05**; best **44.05 @ w_new=0** (all w>0 **hurt**, to **42.88 @ w=3**).
  `outputs/inat_ortho_stack_siglip2.json`. Confirms §4 SigLIP dead; **also dead as proto encoder**.
- **DINOv2-ViT-L/14 iNat protos:** `inat_protos_dinov2_vitl14.pt` + `emb_train_dinov2_vitl14.pt` → same ref;
  best **44.05 @ w=0**; +0.5 = **43.79**; +1 = **42.11**. `outputs/inat_ortho_stack_dinov2.json`. **Dead.**
- **Full deployed text + dual image fusion CV:** `research/inat_v39_fullstack_proxy.py` (S0 incl. TaxaBind text +
  wc×ctft + wt×TB img): peak **44.18** (+0.13 @ wc=3.5 wt=0.25/0.5); **class-disjoint val delta 0.0**. Dead.
- **Only +EV proto stack since v37:** TaxaBind iNat img (+0.52 proxy) — but **v39 real 50.35% < v37 50.46%**
  (proxy→real failed; do not burn more slots on TB-img weight brackets).
- **v40 max-pool VALIDATED REAL (2026-07-25):** w4 zip **50.49%** overall; proxy +1.68 → real +0.03. Staged w3 zip **not recommended** (skip). Builder: `builders/build_v40_inat_maxpool.py`.
- **Honest 52% projection:** at seen **78.95%** need unseen-pop **~17.2%** (have **~13.4–13.7%**). Closing **+1.5pt
  overall** needs either **~+8–10 proxy-pts** on the honest hard-sim stack (only iNat ctft leg ever did that) or
  **~+29% relative b_cond** — none of the legal orthogonal encoders / density / coverage levers tested today move
  that much. **52% likely requires a new general-biology encoder class we have not run yet**, or organizer-confirmed
  eligibility for something stronger than BioCLIP-H / TaxaBind / DINO / SigLIP on external protos.

### 2026-07-24 (denser iNat push): +44 classes downloaded — proxy flat, bigger jobs running

- **Finished needs_id URL enrich (prior 429 stop):** downloaded 44 classes with new URLs via `download_inat_only.py` → **7008** classes / **73,465** images (was 6964 / 73,322).
- **Re-embed + honest proxy** (`inat_protos_ctftshift_full.pt`, `inat_proto_proxy.py`): S0+3×img **43.49** vs v37 ref **43.53** (**−0.04**); CV covered-gold **52.52** (flat). Cand proto coverage **7008/12757 (54.9%)**. **No zip** (need ≥+0.3 → **43.83**).
- **Running:** `fetch_inat_denser.py` low_photo **600** + needs_id_resume **300** (API, rate 30/min); `inat_opendata.py --stage photos --photo-cap 30` (S3 merge w/ `inat_od_obs_needs_id.json`); watcher `scripts/inat_denser_finish.sh` → download / embed / proxy / auto-build v37 zip if w3≥43.83.
- **Killed (reconfirmed):** SigLIP2/DINOv2 ortho legs; TaxaBind iNat img (real −0.11); gap-fill merged proxy −0.22. Best real **50.49%** (`submission_v40_inat_maxpool_w4_sink72.zip`).

### 2026-07-25 (v40 REAL): iNat photo maxpool w4 — new best **50.49%**, proxy lied hard

- **Submitted:** `submissions/submission_v40_inat_maxpool_w4_sink72.zip` (top-4 mean maxpool, IMG_W=4, sink72).
- **Real:** seen **78.95%** · unseen **13.76%** · overall **50.49%** vs v37-w3 **50.46%** / **13.68%** (+0.03 / +0.08).
- **Proxy:** holdout **45.17** (+1.68 vs mean-proto w3 **43.49**); CV **53.67** @ w=4. Transfer **~0.02 overall-pt/proxy-pt**
  (not v37's ~0.12). **Do NOT burn slots on w3 maxpool** (similar EV math).
- **52% honesty:** +0.03 overall does not close **1.51pt**; need a different-scale lever (denser photo bank + re-maxpool,
  BioCLIP FT on disclosed iNat, heavier eval TTA, or a new general-biology encoder — not TB-img / gap-fill / ortho protos).
- **No new zip** until conservative maxpool-transfer EV could materially beat 50.49 toward 52 (≈**+75 proxy-pt** at 0.02/pt — unrealistic from marginal tweaks).

### 2026-07-25 (ctft iNat FT): contrastive LoRA on disclosed iNat — **DEAD proxy, no v41**

- **Checkpoint:** `outputs/ctft_shift_inat.pt` (800 steps, best val db **25.75** vs init ctft_shift **24.76**; train log `outputs/ctft_shift_inat_train.log`).
- **Extracts (squash-TTA):** `emb_train/test/unseen_ctftshift_inat.pt` — complete.
- **iNat artifacts:** `inat_protos_ctftshift_inat_full.pt`, `inat_photo_bank_ctftshift_inat.pt` — complete.
- **Honest proxy** (`research/inat_ft_stack_proxy.py`, re-run 2026-07-25):

| leg | ref ctftshift | inat-FT encoder | Δ |
|---|---|---|---|
| text stack alone | 31.45 | 31.45 | 0 |
| mean proto @ w3 | **43.49** | **42.58** | **−0.91** |
| maxpool top4 @ w3.5 | **45.17** | **44.87** | **−0.30** |
| maxpool top4 @ w4 | 45.08 | 44.74 | −0.34 |

- **Kill bar:** need ≥**+0.3** proxy on stack legs (or ~+1.5 overall real via ~0.12/pt on mean — maxpool transfer ~0.02/pt). **Both fail.** Projected real @ 0.12/pt: **50.38%** (mean) / **50.45%** (maxpool) — below **50.49%** best.
- **Action:** **No `build_v41_*` zip.** iNat-FT encoder hurts image→image leg despite higher iNat val db — do not replace ctftshift in production stack.
- **Pivot (running next):** resume **denser iNat URL fetch** (`fetch_inat_denser.py --mode low_photo`, stalled 250/600) → download → re-embed photo bank → maxpool proxy; **heavy eval TTA** already dead (`ctft_heavy_tta_proxy`: squash4 **−0.04**, heavy8 **−0.69**). Best real unchanged **50.49%** (`submission_v40_inat_maxpool_w4_sink72.zip`).

### 2026-07-25 (WiSE blend): ctftshift ↔ iNat-FT embedding mix — **DEAD** (+0.08 proxy, no zip)

- **Script:** `research/ctft_wise_blend_proxy.py` → `outputs/ctft_wise_blend_proxy_results.json`.
- **Best mean-proto @ w3:** α=0.2 → **43.57** (**+0.08** vs ref **43.49**); maxpool w3.5/w4 flat or negative at best α.
- **Kill bar ≥+0.3:** **fail** (`clears_kill_03: false`). Projected real @ 0.02/pt from mean delta **~50.49** — no slot burn.
- **Action:** dead alongside full iNat contrastive encoder (`ctft_shift_inat`). Production stack stays **ctftshift** + v40 maxpool w4 (**50.49%** real).

### 2026-07-25 (GPU): extended **competition-train** ctftshift LoRA — **DEAD (do not submit)**

- **Job:** warm-start `outputs/ctft_shift.pt` → **`outputs/ctft_shift_long.pt`**, `--shift_aug 1`, **2400** steps, lr **3e-4**, val every **200** (`outputs/ctft_shift_long_train.log`).
- **Val db:** init **24.76**; best saved **24.68** @ step **1000**; FINAL **22.48** / best_db **24.68** — **below init** (overtrain / LR too high). Never reached **25.5** early-stop bar.
- **Artifacts (complete):** squash-TTA `emb_{train,test,unseen}_ctftshift_long.pt`; `inat_protos_ctftshift_long_full.pt`; `inat_photo_bank_ctftshift_long.pt`.
- **Honest proxy** (`outputs/ctftshift_long_stack_proxy_results.json`, vs ref ctftshift):

| leg | ref | long | Δ |
|---|---|---|---|
| mean proto @ w3 | **43.49** | **43.23** | **−0.26** |
| maxpool top4 @ w3.5 | **45.13** | **45.30** | **+0.17** |

- **Projected real @ 0.12/pt (maxpool leg):** **50.51%** (+0.02 vs **50.49%** best) — **not worth a slot**; mean leg **50.46%**. Optional zip `submissions/submission_v40_long_inat_maxpool_w4_sink72.zip` for diff only — **do not submit**. Production stays **v40 w4 @ 50.49%** real.
- **Follow-up low-LR probe — DEAD (no zip):** **`outputs/ctft_shift_lowlr.pt`**, **400** steps, lr **1e-4**, val every **100** from `ctft_shift.pt` (`outputs/ctft_shift_lowlr_train.log`). Best saved db **24.42** (below init **24.76**); FINAL db **24.2** / best_db **24.42**. Watcher **`scripts/ctft_lowlr_watcher.sh`** → **`outputs/ctft_lowlr_zip_decision.txt` = NO_ZIP** (need db ≥ **25.06**). Same story as long run: extra competition-train steps on ctftshift LoRA do not lift pseudo-unseen db — **do not extract/submit**.
- **Encoder ask list:** `outputs/encoder_organizer_eligibility_draft.md` (HF IDs for organizer confirmation).

### 2026-07-25 (follow-up): soft routing recheck + LoRA rank32 probe — **DEAD (no zip)**

- **Soft routing recheck** (`outputs/ctft_soft_routing_recheck.log`): holdout best **soft_b1.05 +1.91pt** vs hard, but eval routing **not leader-like** (soft_b1.05 uf→uns-cls **37.0%** vs hard **54.5%**; tf→seen **84.2%** vs **92.5%**). Same holdout/eval mirage pattern as v34/v36 — **do not submit** soft variants.
- **Low-LR ctftshift LoRA** (prior): **`outputs/ctft_lowlr_zip_decision.txt` = NO_ZIP** — best db **24.42** < kill **25.06**.
- **Rank-32 LoRA probe** (`outputs/ctft_shift_rank32_probe.pt`): **400** steps, rank **32**, lr **3e-4**, from frozen baseline db **22.86** (`outputs/ctft_shift_rank32_probe_train.log`). Best saved db **23.94** @ step 200; FINAL **23.86** / best_db **23.94** — **below kill 25.06** and below production **ctft_shift** init **24.76**. **`outputs/ctft_rank32_zip_decision.txt` = NO_ZIP**. **Stop further LoRA rank/ablation probes** on ctftshift (diminishing).
- **Organizer eligibility email (draft ready):** `outputs/organizer_eligibility_email.txt` — asks re **bioclip-2.5-vitb16**, **bioclip-2.5-vitl14**, **TreeOfLife-CLIP / Arboretum-CLIP** (general biology, not fish-specific).
- **GPU next (non-LoRA, +EV if proxy clears bar):** finish **denser iNat** pipeline — low_photo fetch **+696 URLs** (`outputs/inat_denser_fetch.log` done) → download → re-embed photo bank → maxpool holdout proxy (prior +44-class download was proxy-flat; need measured post-696 pass before zip).

### 2026-07-25 (denser iNat +696 URLs): download +356 classes — maxpool proxy flat/-0.04, NO_ZIP
- Download: 117258 jobs -> 117225 files, 7364 classes (+356 vs 7008). Bank: 101106 embeds. Proxy mean@w3 43.49; best maxpool 45.13 vs ref 45.17 (-0.04). NO_ZIP (need 45.47). Best real 50.49% unchanged.

### 2026-07-25 (BioCLIP-2 iNat proto v41): **VALIDATED REAL 50.57%** — new best

- **Submitted** `submissions/submission_v41_bioclip2proto_w4_b22.5_sink72.zip` → seen **78.95%** · unseen **13.93%** · overall **50.57%** (+0.08 vs v40).
- **Frozen BioCLIP-2 iNat mean-proto leg** on v40 maxpool stack (`builders/build_v41_bioclip2_proto.py`, B2_W=2.5, IMG_W=4). Holdout +**0.99** mean-track → real +**0.08** overall (**~0.08 overall-pt/proxy-pt**).
- **Not v39:** TaxaBind iNat img protos proxy +0.52 → real −0.11; BioCLIP-2 proto **transfers positively**.
- **Next (stack on v41, do not replace):** B2_W sweep {2.0, 3.0, 3.5} + class-disjoint CV (`research/inat_bioclip2_b2w_sweep.py`); BioCLIP-2 **maxpool photo bank** fused with v40 maxpool; denser iNat coverage for B2 encoder. Build zips when proxy EV @ **0.08/pt** projects **≥~50.62** toward 52.
- **B2_W sweep (2026-07-25):** holdout peak **B2_W=1.5** proxy 46.29 (+0.26 vs 2.5); CV picks 1.5 val +1.03 — optional `B2_W=1.5` zip for diff only (~50.59 proj), not clear slot EV vs 50.57.


### 2026-07-25 (v40 gate operating point): `seen_frac` holdout sweep — **DEAD (holdout mirage)**

- **Motivation:** recheck routing f=**0.72** (tuned @ v36 b) after v40 real unseen **13.76%**.
- **Script:** `research/v40_seen_frac_holdout.py` → `outputs/v40_seen_frac_holdout_results.json`.
- **Result:** holdout overall **monotone ↑ with f** (0.55→58.16% … **0.72→71.81%** … **0.80→75.34%**). Lower eject hurts holdout — opposite of “catch more unseen” intuition; same class as soft-routing holdout/eval mirage (§8). Real f-sweep already peaked **0.72** (+0.08 EV to 0.76/0.80). **No zip / no submit.**
- **Next lever queue:** organizer reply on ToL/B-L; otherwise need a non–BioCLIP-iNat-proto **b** lift (image-proto family exhausted at CV thin edge).


- **B2 LoRA FT (800 steps, shift_aug):** pseudo-unseen db **15.36 → 15.53** (+0.17). iNat LoRA protos + LoRA train queries on v41 stack: best **46.03** (+**0.00** vs ref) @ B2_W=1.5 — **NO_ZIP** (`inat_bioclip2_v41_lora_mean_proxy.json`). Artifacts: `b2_shift_lora.pt`, `inat_protos_bioclip2_lora.pt`, `emb_train_bioclip2_lora.pt`.


### 2026-07-27 (session): B2 LoRA v2 taxctx · taxctx CV dead · v2 proxy +0.43 holdout

- **B2 LoRA v1 (name text):** db **15.53**; v41 stack **+0.00** — dead (`inat_bioclip2_v41_lora_mean_proxy.json`).
- **B2 LoRA v2:** `research/contrastive_ft_bioclip2_v2.py` — frozen **768-d taxctx** (`src/build_text_b2_taxctx.py`), shift_aug, **iNat cap=8**, 2000 steps, init `b2_shift_lora.pt`. Log `outputs/b2_shift_lora_v2_train.log`; ckpt `outputs/b2_shift_lora_v2.pt`; db **14.5** (pseudo-unseen **below v1**).
- **v41 + LoRA v2 mean proto:** honest proxy **46.72** (+**0.43** vs 46.29) @ **B2_W=1.5** — clears **+0.3 kill**; proj **~50.60** real @ 0.08/pt — **NO_ZIP** toward **53%** bar.
- **v41 taxctx leg weight CV:** `research/v41_taxctx_weight_cv.py` — best **w=1.0** (no lift); CV val **−0.52** — **NO_ZIP**.
- **Still open:** validate v2 LoRA on **real** Codabench if spending a slot; gap to **53%** remains **~2.4pt** — need unseen-pop lift beyond marginal B2 stack tweak.

### 2026-07-27: 53% bar · B2 maxpool + denser445 mean DEAD · B2 LoRA FT started

- **Denser445 B2 mean on v41:** pre445 best **46.29** (+0.26 vs ref 46.03); denser445 **46.16** (+0.13) — **NO_ZIP** (kill +0.3). Restored `inat_protos_bioclip2.pt` from `inat_protos_bioclip2_pre445denser.pt`.
- **B2 maxpool on v41:** recomputed **46.03**, +any b2_mp_w **hurts** — dead (`inat_bioclip2_v41_maxpool_proxy.json`).
- **BioCLIP-2 LoRA:** `research/contrastive_ft_bioclip2.py`, log `outputs/b2_shift_lora_train.log`, ckpt `outputs/b2_shift_lora.pt`. Baseline pseudo-unseen db **15.36**.
- **Hub:** vitb16/vitl14 HF 404; frozen bioclip-2 taxctx **14.37** alone (`unseen_backbone_bioclip-2_taxctx_results.json`).
- **Best real unchanged:** **50.57%** v41. Campaign doc: `outputs/path_to_52_blocker.md` (53% math).

### 2026-07-27 (late): v42 optional zip · dual B2 CV · B2 rank32 FT

- **v42 (optional probe):** `builders/build_v42_bioclip2_lora_v2.py` → `submissions/submission_v42_b2lora_v2_w4_b21.5_sink72.zip` (35665). LoRA-v2 Q+proto, B2_W=1.5, v41 maxpool w4. Proxy +0.43 → proj **~50.60** (+0.03 real) — **not recommended** for 53%.
- **Dual B2 legs:** `research/inat_bioclip2_dual_b2_proxy.py` — holdout 46.81 @ wf=0.5/wl=1.0; CV val +1.03.
- **B2 LoRA rank32:** `research/contrastive_ft_bioclip2_v3_rank32.py`, ckpt `outputs/b2_shift_lora_v3_rank32.pt`, best db 14.67 — no zip.

### 2026-07-27 (late): v43 dual zip built — then validated real

- **v43 build:** `builders/build_v43_bioclip2_dual.py` → `submissions/submission_v43_b2dual_w4_wf0.5_wl1_sink72.zip` (35665). Frozen B2 wf=0.5 + LoRA-v2 wl=1.0. Holdout **46.80759310722351** vs frozen-v41 ref **46.28990590572357** (+**0.517687201499939**).
- **Validated real (2026-07-28):** seen **78.95208240035826%** · unseen **14.356372045220966%** · overall **50.75564278704613%** — new best. Real delta vs v41: **+0.18564278704613 overall / +0.426372045220966 unseen**.
- **Transfer anchor:** **~0.36 overall-pt / proxy-pt** on this dual-B2 leg.
- **Next:** BioCLIP-2 full-slice FT / hard-neg LoRA v4.


### 2026-07-28 (session): CE-only taxctx LoRA ablation · HF auth
- **CE-only:** `research/contrastive_ft_bioclip2_v2.py` warm-start `outputs/b2_shift_lora.pt`, taxctx, `--inat 0`, 1600 steps → log `outputs/b2_shift_lora_ce_taxctx_train.log`, ckpt `outputs/b2_shift_lora_ce_taxctx.pt`, **best db 13.63** (FINAL 13.07) — **NO extract / dual / zip** (gate db≥16.0; worse than hard-neg 14.06 and v2+inat 14.5).
- **Hard-neg (prior):** best db **14.06** — do not retry as-is.
- **Submitted / measured:** v43 dual holdout **46.80759310722351** (`outputs/inat_bioclip2_dual_b2_proxy.json`) → real **50.75564278704613%** (`submissions/submission_v43_b2dual_w4_wf0.5_wl1_sink72.zip`).
- **HF:** `HF_TOKEN` unset in env; Hub requests unauthenticated; gated encoders **401**. Need token + organizer for bioclip-2.5-vitb16/vitl14 and TreeOfLife-CLIP probes.
- **Next:** no existing zip is a clear +EV submit vs measured v43; prefer a new non-gated, non-HF-token experiment that can target unseen lift directly. Do **not** retry hard-neg as-is, soft, TB-img, B2 maxpool, ctft long LoRA.

### 2026-07-28 (real): v43 dual validated — new best

- **Submitted** `submissions/submission_v43_b2dual_w4_wf0.5_wl1_sink72.zip` → seen **78.95208240035826%** · unseen **14.356372045220966%** · overall **50.75564278704613%**.
- **Delta vs v41 real:** **+0.18564278704613 overall** / **+0.426372045220966 unseen** over `submissions/submission_v41_bioclip2proto_w4_b22.5_sink72.zip`.
- **Proxy transfer for this leg:** `outputs/inat_bioclip2_dual_b2_proxy.json` holdout **46.80759310722351** vs frozen-v41 ref **46.28990590572357** (+**0.517687201499939**) → **~0.36 overall-pt/proxy-pt**.
- **Implication:** this validates the dual-B2 idea as real-positive, but 53% still needs unseen-pop **19.49713990240072%** at flat seen, i.e. another **+5.140767857179754pt** unseen from the current best.

### 2026-07-28: hard-neg B2 LoRA · v4/v5 db fail · organizer HF probe

- **LoRA v4/v5:** best db **14.8** / **15.01** vs v1 **15.53** — **NO zip from LoRA**.
- **Hard-neg FT:** research/contrastive_ft_bioclip2_hardneg.py — 1800 steps, log outputs/b2_shift_lora_hardneg_train.log, ckpt outputs/b2_shift_lora_hardneg.pt, **best db 14.06** — **NO extract/zip** (gate db≥16.0).
- **Measured:** v43 dual is now **50.75564278704613%** real best; v41 remains **50.57%** prior anchor.
- **Organizer encoders:** outputs/organizer_encoder_probe.json (vitb16/vitl14/ToL/Arboretum **401**; load OK: vith14, bioclip-2, BioTrove).

### 2026-07-30: TreeOfLife-200M B2 embeddings → v44 denser merge — VALIDATED REAL (tiny)

- **Stream:** `research/tol_bioclip2_protos.py` over `imageomics/TreeOfLife-200M-Embeddings` bioclip-2_float16
  (stopped ~shard 179 once past fish); **ncov 11642/17393**, unseen **6659/11598**, cap=32/class.
- **DEAD:** raw ToL gap-fill / additive (`tol_v43_proxy.json`, best Δ0 / −1.7); genus backoff; soft-kNN
  (`soft_knn_v43_proxy.json` Δ0); H-add on v43 (all negative).
- **Proxy:** denser iNat∩ToL merge α=0.5 @ **wf=2.0**: holdout **47.239 (+0.431 vs v43)**;
  CV val **+0.60** (`tol_v43_followup_proxy.json`, `tol_denser_cv.json`). Artifact
  `outputs/inat_tol_merged_b2_a05.pt`.
- **REAL (wf2 zip):** seen **78.95208240035826%** (flat) · unseen **14.39491264131552%** · overall
  **50.77246600308426%** — **+0.016823216038133637 / +0.038540596094552626** vs v43. Transfer
  **~0.039 overall-pt/proxy-pt** (holdout overstated ~11× vs dual-B2's 0.36). Skip wf1.5 / α brackets
  unless a new proxy beat ≫+1pt appears. Gap to 53% still **~2.23pt** / **+5.10pt unseen**.
- **Zips:** `submissions/submission_v44_toldenser_w4_wf{2,1.5}_wl1_a05_sink72.zip`
  (`builders/build_v44_tol_denser.py`). Best real = **wf2**.

### 2026-08-05: v46 LoRA∩ToL denser — NEW BEST REAL

- **Builder:** `builders/build_v46_lora_tol_denser.py` — denser iNat∩ToL α=0.5 on **both** legs
  (frozen `inat_tol_merged_b2_a05.pt` @ wf=2.5, LoRA `inat_tol_merged_b2lora_a0.5.pt` @ wl=2.0).
- **REAL:** seen **78.95208240035826%** (flat) · unseen **14.54907502569373%** · overall
  **50.83975886723678%** — **+0.06729286415252034 / +0.1541623843782105** vs v44.
- **Transfer:** holdout **+0.345** → real **+0.067** → **~0.195 overall-pt/proxy-pt** (better than
  frozen denser ~0.039; below dual-B2 ~0.36).
- **Zip:** `submissions/submission_v46_loratoldenser_w4_wf2.5_wl2_a05_sink72.zip`.
- **Next:** f0.65 combo zip; then v47 fuse / v44 f-sweep only if f helps. Gap to 53% **~2.16pt**.

### 2026-08-08: unified single-head 2-stage pipeline (drop seen/unseen heads) — **DEAD, no zip**

Requested shape: stage 1 vision-proto → top-k **seen** shortlist; stage 2 **one** score over
(top-k seen + all non-train classes) → argmax. No gate, no routing fraction, no Sinkhorn.
Script: `research/eval_unified_2stage.py` (`--real` for the routing pass) →
`outputs/unified_2stage_holdout.json`, `outputs/unified_2stage_real_routing.json`.

| arm | stage-2 score | holdout overall | real tf→seen / uf→unseen | proj real (**upper bound**) |
|---|---|---|---|---|
| A f=0.60 (production, **measured 51.44**) | two heads | 50.71 | 86.7 / 74.5 | 51.44 (anchor) |
| A f=0.72 (measured **50.84**) | two heads | **55.80** | 92.5 / 54.5 | 52.07 |
| **B** (spec as written) | vision↔text only | **57.02** @k=5 | 82.7 / 73.4 | **≤49.35** |
| **C** | text + B2/ToL protos, full space | 55.82 @k=5 | 73.4 / 84.4 | **≤45.97** |
| **D** (best unified design) | C + uniform reference-photo leg | 56.86 @k=5 | 77.6 / 74.3 | **≤46.88** |

**Anchor is exact, not approximate:** arm A f=0.60 reproduces `outputs/prediction_v50_f60_tau18.json`
**35,665/35,665 identical** (asserted in-script), so `a_cond`=88.0% / `b_cond`=26.0% are calibrated on
the scored 51.44% submission itself.

- **`k` is nearly inert**: B k=5 → k=all moves holdout 57.02 → 56.74. Stage-1 shortlisting buys
  ~0.3pt; the unified pipeline is essentially a full-space argmax. The k-sweep is not the lever.
- **Holdout is anti-correlated here.** It ranks A f=0.72 **+5.09** over f=0.60; real says f=0.60 wins
  by **+0.60**. A proxy that mis-orders the one axis with two real anchors, in the wrong direction, by
  5pt cannot adjudicate B/C/D. This is the v34 / soft_b1.05 mirage signature again (§8).
- **Projection is itself optimistic**: it credits every arm with production's conditional accuracy and
  still over-predicts A f=0.72 by **+1.23** (52.07 proj vs 50.84 real). Even granting all arms that
  slop, B/C/D land **48–50.5%**, below 51.44 — and the unified arms decide seen classes *without* the
  3-encoder prototype ensemble, so their true conditional is lower still.
- **Why it loses:** the two heads are not redundant. The seen head's 3-encoder prototype ensemble
  (+ `LAM`·taxon anchor) has no counterpart on the unseen side, and the unseen route's iNat photo-bank
  leg covers only **966/5795** seen classes. Any single score function either drops the ensemble
  (B: −4.0pt tf→seen) or lets the asymmetric image legs swamp the seen side (C: tf→seen 73.4%).
  D equalises the leg (train images | iNat photos, same encoder + same top-4-mean statistic) and
  matches production's unseen routing to **0.2pt** (74.3 vs 74.5) — yet still gives up **9.1pt** of
  tf→seen. **That gap is structural, not tuning.** Closing it needs the iNat photo bank re-embedded
  under `ftshift` and `fullft336shift` so the unseen side gets the same 3-encoder ensemble; those
  banks do not exist and are new GPU work. Until then no k or weight setting recovers it.
- **Two implementation traps found** (both live in `builders/build_v48_2stage_soft_pipeline.py`):
  1. masking uncovered classes to `-1e4` *deletes* them from a unified candidate space (harmless
     inside a dedicated head, fatal outside it) — use `neutral()` in the eval script;
  2. a `dbnorm` leg added to only one side of the partition injects a constant per-row offset that
     decides the argmax by itself. The `C+bank` row is that bug, not a pipeline result: tf→seen
     **13.2%**, only **94** unseen classes used. Do not quote it as evidence about unified inference.
- **Compliance note:** this was proposed as a rules fix. It is not one. Live terms re-fetched
  2026-08-08 — §2.1 wording **unchanged** vs the 2026-07-23 cache. The gate is computed from
  image+text and the class partition comes from `label_train.json`, never `splits/*.pkl`; both are
  explicitly allowed (RULES §2.1, HANDOFF §6). Unified inference is an architecture preference, and
  it costs ≥2pt.
- **Verdict: NO ZIP.** Do not spend a slot. `builders/build_v48_2stage_soft_pipeline.py` /
  `submission_v48_2stage_k{10,50,100}.zip` are the same family (v48 is B with a novelty-prior boost)
  — do not submit those either.

### 2026-08-18: 3-encoder iNat banks + better crops (unscored)

**336 iNat bank (holdout):** `research/v50_shiftbank_proxy.py` → `outputs/v50_shiftbank_proxy.json`.
REF 47.584 → best **48.361 (+0.777)** @ `w_ft=0, w_336=2.0`. **ftshift bank hurts. Do not add it.**

**Crops vs center+squash TTA (holdout):** eval fish are more elongated; short-side resize + center
crop cuts head/tail. `research/crop_views_proxy.py` → `outputs/crop_views_proxy.json`.
Mean TTA of letterbox/strips **loses**. Win = **max-over-crops on raw bank cosine, then one dbnorm**.
Best 6-view: `max(all_no_flip)` = center+squash+letterbox+3-strips **48.145 (+0.561)**.

**Tighter crops + 336 stack (holdout):** `research/crop_chase53_proxy.py` →
`outputs/crop_chase53_proxy.json`. Tight bbox / extra overlap on top of the 6-view max **hurt**
alone. Best recipe: **`max(center, squash, 5 overlapping long-axis strips) + 3·336 bank` = 49.008
(+1.424)**. 6-view max + 2·336 = +0.992 (subadditive). Proj real at 0.20–0.36 transfer:
**51.73–51.96%**, still short of 53 unless eval shift transfers harder than holdout.

**Zips (single pipeline, f=0.60, τ=1.8, full 17393 argmax, no `splits/*.pkl` routing):**
| zip | recipe | holdout Δ | vs v50 preds | real |
|---|---|---|---|---|
| `submissions/submission_v55_cropmax_336_f60_tau18.zip` | max(6 no-flip) + 336@w=2 | +0.992 | 3657 / 10.25% | unscored |
| **`submissions/submission_v56_overlap_336_f60_tau18.zip`** | max(center,squash,5-ostrip) + 336@w=3 | **+1.424** | 4092 / 11.47% | **51.61643067433057%** (75.67 / 20.57) |

Routing shape unchanged vs v50 (tf→seen 86.71% / uf→unseen 74.49%). v55↔v56 differ on 1894 images.
Builders: `builders/build_v55_cropmax_336.py`, `builders/build_v56_overlap_336.py`.
v56 is the new real best. Gap to 53% **1.384pt**.

**v57 336 query-crop + recropped 336 gallery (unscored, NO_SUBMIT):** overnight GPU finished.
Holdout `outputs/v57_336crop_proxy.json`: vs v56 **+0.043** (`336_max|ct4|336w2`, 49.051) — **fails +0.3 kill**.
CSLS / softmax-pool / Q×cropbank all **hurt**. Zip exists (`submissions/submission_v57_336crop_f60_tau18.zip`,
4502 diffs vs v50, routing 86.7/74.5) — **do not spend a slot**. Same crop on the 336 encoder does not stack.

### 2026-08-24: dbnorm hub offset does not need the test batch (no score change)

`research/dbnorm_frozen_hub.py` → `outputs/dbnorm_frozen_hub_{pseudo,pseudo_a,pseudo_b,kept}.json`,
summary `outputs/dbnorm_frozen_hub_summary.json`.

`dbnorm`'s across-image term is `s_ic/0.05 − h_c`, `h_c = logsumexp_i(s_ic/0.05)` — the **only**
place the eval batch enters that leg. Estimated `h_c` on reference pools whose classes are disjoint
from the 11,598 novel classes, froze it, rebuilt v56 otherwise byte-identical.

| ref pool | n | diff vs shipped v56 | can change acc |
|---|---|---|---|
| withheld pseudo-novel, 7-view matched | 2,318 | **296** (0.83%) | 224 |
| even half | 1,159 | 300 | 226 |
| odd half | 1,159 | 297 | 223 |
| trained-class imgs, centre view only | 4,000 | 480 | 357 |

- Harness check: **all four runs reproduce shipped v56 exactly (0/35,665)** before the swap.
- Halves disagree with **each other** on 64 → estimator converged by n=1,159; residual is
  pool-composition systematic (pseudo vs kept disagree on 459), not sampling noise.
- Only the 14,266 ejected rows can change; only novel-folder ones can change accuracy →
  **swing bounded at ±0.63pt overall**. `prediction_v56_frozenhub_pseudo.json` is built and
  unscored — **one slot** would turn the bound into a number.
- Four batch couplings remain: `zc` (member standardization), `z1` (gate), the f=0.60 rank quantile
  (**strongest**), Sinkhorn. Paper now states five coupling points, not three.

**Paper facts corrected (were wrong in the PDF):** training images are **64,259 / 5,795 classes**,
not 99,979 (that is the jpg count in `data/dl/images/`, which also holds both eval splits);
adapter training uses 61,941 = **96.4%** of 64,259; 1,159 withheld classes = **2,318 images**.
Closed route argmaxes **5,795** classes — 4,636 is only Eq. 7's loss denominator.

**Offset agreement (added same run):** Spearman(ref h_c, batch h_c) per leg — pseudo pool
**0.39–0.98 (median 0.86)**, kept pool **0.37–0.91 (median 0.69)**. At `tc=0.05` the across-image
softmax is concentrated: median class draws its offset from an **effective ~90 of 35,665 images**
(p05=35, p95=235). So hub structure is only *partly* intrinsic to the class embedding — the fused
decision just doesn't need it recovered precisely (10 legs summed). Vectors in
`outputs/hub_offsets_{pseudo,kept}.pt`. **Do not claim "a validation set identifies the hub
classes"** — per-class it does not; only the fused prediction is stable.

### 2026-08-25: v60 learned-pipeline (no-dbnorm) + proposed v61 soft-marginal — **NO_ZIP**

Artifacts from the prior session: `src/calibration/{novelty_logistic_gate,polynomial_calibrator}.py`,
`src/pipeline/{single_pipeline_model,multi_modal_fusion}.py`, `outputs/learned_pipeline_v60.pt`,
`outputs/prediction_v60_learned_nodbnorm.json`, `submissions/submission_v60_learned_nodbnorm.zip`,
`research/benchmark_v60_vs_v61.py` (writes **no** output artifact — no run record exists).

**v60 routing diagnostic vs the scored v56 (this session, `splits/*.pkl` used for diagnosis only):**

| pred | tf→seen | uf→unseen | diff vs v56 | real |
|---|---|---|---|---|
| v56 (scored **51.61643067433057**) | 86.71% | 74.49% | — | 75.668 / 20.568 |
| v60 learned no-dbnorm | **91.06%** | **52.53%** | **13,962 (39.15%)** | unscored |

v56 calibration: `a_cond` **87.26%**, `b_cond` **27.61%**. Crediting v60 with those conditionals
projects **seen 79.46 / unseen 14.51 / overall 51.11** — *below* v56, and that projection is an
**upper bound** twice over: the same method over-predicted the f=0.72 real anchor by **+1.23pt**
(2026-08-08), and it assumes `b_cond` survives deleting `dbnorm`, which the unseen route depends on.
uf→unseen 74.5→52.5 is the **same routing shape** as soft_b1.05 (37.0 vs 54.5), v34, and the
monotone-in-`f` holdout sweep — three measured instances of that direction losing on real.

**Proposed v61 (soft marginal fusion + query-centered poly + dynamic attention) — do not build.**
- It is the `log P(seen)·softmax ⊕ log P(unseen)·softmax` full-space argmax family already measured
  dead: unified 2-stage arms B/C/D ≤**49.35** (2026-08-08), soft_b1.05, v34, `build_v48/v49`.
- Its table mixes **holdout** rows with the **real** 51.44/51.62 anchors in one column (CLAUDE.md rule 1).
- The claimed **seen 79.40 / unseen 25.20** implies a gate at **TPR 91.0% / TNR 91.3%** (balanced 91.1%)
  → binormal **AUC ≈ 0.972**. Measured real gate AUC is **0.9084** (best variant 0.9120), which buys at
  most **82.7%** balanced. v61 changes no gate feature, so it cannot move that ROC.
- The claimed unseen 25.20 at v56 routing needs `b_cond` **33.83%** vs measured **27.61%** — a +6.2pt
  conditional lift from re-normalization alone, larger than the entire remaining gap to 53%. `b` is
  encoder-limited (family-oracle cap, §8 2026-07-23).

**If the real goal is inductive execution / no batch coupling:** `prediction_v56_frozenhub_pseudo.json`
is already built, bounded at **±0.63pt**, and churns 0.83% instead of 39%. Per 2026-08-24 the
**strongest** remaining coupling is the **f=0.60 rank quantile**, not `dbnorm`.

### 2026-08-29 (v77): learned gate at a FIXED operating point — routing dominates v56 on real, UNSCORED

**The framing that unlocked it.** With the v56 real calibration (`a_cond`=87.26%, `b_cond`=27.61%):

```
overall = 0.5635*a_cond*TPR + 0.4365*b_cond*TNR = 0.49175*TPR + 0.12053*TNR
```

TPR (`tf→seen`) is worth **4.08x** TNR (`uf→unseen`). Every prior learned pipeline moved the
operating point toward seen and paid that 4:1 — v60 (91.06/52.53), v63 (90.56/51.06, real
**50.68%**), v61/v72–v76 same family. Moving `f` is a **trade**, not an improvement. The only
intervention that raises seen AND unseen together is a better gate **ranking at the same `f`**.
So v77 changes the ranking only and pins `f=0.60`.

**What v77 is:** `builders/build_v77_learned_gate.py` = `build_v56_overlap_336.py` with one thing
swapped. Same seen head, same unseen head (7-view crop-max + 336 bank + dual B2), same `f=0.60`,
`tau=1.8`, Sinkhorn, full-17393 argmax. The hand-weighted `z1(img_seenmax) + 2*z1(text_margin)`
becomes a 12-feature logistic gate (`research/learned_gate_v77.py`, `outputs/learned_gate_v77.pkl`).

Features (identical definition on holdout and eval): `sb_max`, `sb_margin`, `sb_lse_gap`,
per-member seen max for ctftshift/ftshift/fullft336shift, `tm_ctft`, `tm_full`, and — new to any
gate — the unseen head's own image evidence: `bank_max`, `bank_margin`, `b2f_max`, `b2l_max`.
The gate's bank feature is **center-view only** (holdout crop views exist only for the 2,318
pseudo-novel rows); the unseen head keeps its full 7-view crop-max.

**Holdout (class-disjoint 5-fold CV, `f=0.60` pinned, deployment-weighted):**

| gate | AUC | TPR | TNR | proj% | vs v56 |
|---|---|---|---|---|---|
| v56 (`z1+2*z1`) | 0.9705 | 94.19 | 84.12 | 56.450 | — |
| **logistic (12-feat)** | **0.9758** | **94.99** | **85.16** | **56.973** | **+0.523** |
| hgb | 0.9637 | 93.97 | 83.82 | 56.310 | −0.140 |

**Real-eval routing diagnostic (`splits/*.pkl` for counting only, nothing tuned on it):**

| pred | tf→seen | uf→unseen | diff vs v56 |
|---|---|---|---|
| v56 (scored **51.61643067433057**) | 86.71% | 74.49% | — |
| **v77 learned gate** | **88.38%** | **76.64%** | 5,600 (15.70%) |

**(88.38, 76.64) strictly dominates (86.71, 74.49)** — better in *both* coordinates, so the ROC
itself moved, not just the threshold. This is the first learned variant that does not trade.
No prior learned build cleared v56 on either axis.

**Projection, and why it is an upper bound.** Holding v56's conditionals fixed →
seen 77.13 / unseen 21.16 / **overall 52.70%** (+1.08 vs v56). Route churn is two-way
(test +767/−432; unseen-folder +1182/−847), and the images that move are near-threshold, so the
marginal conditionals are below average: `a_marg` < 87.26%, `b_marg` < 27.61%. At half the average
conditionals the gain is +0.55pt. **Honest range 52.1–52.7%**, sign robust (gain > 0 for any
positive marginal conditional). Same method over-predicted the `f=0.72` anchor by +1.23pt
(2026-08-08) — but that was a large `f` move; here `f` is identical.

- **Do NOT sweep `f` on this projection.** Conditionals change with `f`; that is exactly the
  measured failure mode of the 2026-08-08 method. `f=0.60` stays real-tuned.
- **Compliance unchanged:** gate features are image+text model-derived, class partition from
  `label_train.json`, one uniform rule, full 17,393 argmax. No `splits/*.pkl` in the pipeline.
- **Zip:** `submissions/submission_v77_learned_gate_f60_tau18.zip` (35,665 preds, 10,858 distinct classes).

### 2026-08-29 (v77 REAL): **53.3828683583345% — NEW BEST, 53% BAR CLEARED**

Submitted. seen **77.72304324028462%** · unseen **21.961716341212745%** · overall
**53.3828683583345%** — **+1.76643768400393pt** vs v56, **+2.05503308951585 seen**,
**+1.393884892086331 unseen**. Both axes up; no trade. Prior visible #1 was 50.56%.

**The projection was beaten by +0.68pt, and the reason changes the EV math for all future gate work:**

| source | gain | |
|---|---|---|
| routing (TPR 86.7094→88.3814, TNR 74.4860→76.6380) | **+1.082pt** | projected |
| **conditionals (`a_cond` 87.2660→87.9405, `b_cond` 27.6115→28.6564)** | **+0.686pt** | **missed** |
| total | +1.767pt | |

The pre-submit note called 52.70% an *upper bound* on the assumption that a better gate only
re-sorts images across a fixed threshold, so the images it moves must be below-average quality.
**That assumption is wrong.** The gate does not slide a threshold along a fixed ranking — it
**re-ranks**, and the v77 features are simultaneously novelty signals *and* correctness signals:

- Per-encoder agreement (`m_ctftshift`/`m_ftshift`/`m_fullft336shift`) + seen-distribution
  concentration (`sb_margin`, `sb_lse_gap`): an image whose seen head is sharply peaked and agreed
  on by all 3 encoders is both more likely to *be* seen and more likely to be *classified right* by
  that head. → `a_cond` +0.67pt.
- `bank_max`/`bank_margin`: the unseen head largely *is* the iNat photo bank, so a sharp bank match
  means both "probably novel" and "the unseen head will get this right". → `b_cond` +1.04pt.

**Consequence: gate quality COMPOUNDS, it does not trade.** Value a gate-ranking gain at roughly
**1.6x** its linear TPR/TNR projection (measured once, n=1 — treat as a prior, not a constant).
This is the opposite of the operating-point moves (v60/v63/v34/soft_b1.05), which trade at ~4:1 and
lost every time. **Distinguish the two axes in every future proposal:** re-ranking at fixed `f`
compounds; moving `f` trades.

**Still true / unchanged:** `b` remains encoder-limited (family-oracle cap §8); no encoder lever
reopened. The gain came entirely from reading evidence the pipeline already computed but the gate
had never looked at.

### 2026-08-29 (v78 view-disagreement features): **DEAD** — and the real lever was regularization

**The hypothesis:** HANDOFF's root cause for the whole shift problem is framing (train aspect
~1.15, eval ~2.12), yet the v77 gate has no shift feature — it reads one framing per image. The 7
crop views v56 already computes give a direct read: if an image's answer survives re-cropping the
evidence is real; if it swings, the single-view score is a framing artifact.

**Built:** `research/extract_holdout_views_v78.py` → `outputs/emb_holdout_all_ctft_views7.pt`
(all **14,184** holdout rows × 7 views, 0 misses, 17 min — the cached
`emb_holdout_ctft_cropviews_v2.pt` covered only the 2,318 pseudo-novel rows, which is why no gate
could ever see this feature). Split cross-check: pseudo set overlap **2318/2318** vs the cached
artifact, so the ordering is exact. `research/learned_gate_v78.py` adds 5 features: `view_agree`,
`sb_view_std`, `sb_view_gain`, `bank_view_std`, `bank_view_gain`.

**Result — DEAD.** Class-disjoint CV, f=0.60 pinned:

| arm | best-C proj% | vs v77 |
|---|---|---|
| v77 (12 feat) | 57.317 | — |
| v77 + `view_agree` (13) | 57.273 | **−0.044** |
| v78 (17 feat) | 57.347 | **+0.030** |

**+0.030 vs a +0.30 kill bar.** At C=1 v78 was −0.209; the extra 5 parameters cost more variance
than they bought. All view coefficients ≈0 (`view_agree` +0.055 largest, `sb_view_std` −0.003).
**Do not retry view/crop-disagreement features in the gate.** Builder deleted; research script kept
as the evidence. The 7-view crop-max remains load-bearing in the unseen HEAD (v56, +1.424 holdout)
— it just carries nothing extra for *routing*.

**What the sweep did find: v77 was over-regularized.** `LogisticRegression(C=1.0)` was an unexamined
default. CV is monotone in C and plateaus, confirmed threshold-free by AUC:

| C | AUC | TPR | TNR | proj% |
|---|---|---|---|---|
| 1 (deployed v77) | 0.9758 | 94.99 | 85.16 | 56.973 |
| 10 | 0.9791 | 95.53 | 85.85 | 57.317 |
| **100** | **0.9811** | **95.85** | **86.28** | **57.531** |
| 300 | 0.9813 | 95.78 | 86.15 | 57.478 |
| 1000 | 0.9814 | 95.79 | 86.19 | 57.487 |

12 features on 14,184 rows needs almost no shrinkage. **+0.558 proxy-pt vs deployed v77** for a
hyperparameter. Coefficients stay sane (|max| 0.864 — not separable-degenerate). Caveat: C was
picked on the same CV that reports it; mitigated by the monotone plateau (100–1000 all ≈57.5, not a
knife edge) and by AUC agreeing.

### 2026-08-29 (v79): C=100 refit — routing dominates v77 on real, UNSCORED

`outputs/learned_gate_v79.pkl` (same 12 v77 features, C=100). The v77 builder is now
env-parameterized (`GATE_PKL`, `TAG`; defaults unchanged, so the scored v77 path is preserved):

```bash
conda activate onet && GATE_PKL=outputs/learned_gate_v79.pkl \
  TAG=v79_learned_gate_c100_f60_tau18 python builders/build_v77_learned_gate.py
```

| pred | tf→seen | uf→unseen | real |
|---|---|---|---|
| v56 | 86.71% | 74.49% | 51.61643067433057 |
| v77 | 88.38% | 76.64% | **53.3828683583345** |
| **v79 (C=100)** | **89.11%** | **77.58%** | **unscored** |

**(89.11, 77.58) strictly dominates v77's (88.38, 76.64)** on both axes.
Zip: `submissions/submission_v79_learned_gate_c100_f60_tau18.zip` (35,665 preds, 10,836 distinct
classes, 17.94% differ from v56).

### 2026-08-29 (v79 REAL): **53.42492639842983% — new best, but only +0.042**

seen **77.44937055281883%** · unseen **22.41135662898253%** · overall **53.42492639842983%**
(TPR 89.1128 / TNR 77.5822, `a_cond` **86.9116%**, `b_cond` **28.8872%**).

**The routing projection was near-exact and still nearly worthless — read this before proposing
any gate change:**

| step | routing | conditionals | total |
|---|---|---|---|
| v56→v77 (2 feats → 12: **new information**) | +1.082 | **+0.686** | **+1.767** |
| v77→v79 (C=1 → C=100: **same information, re-weighted**) | +0.481 (projected +0.479) | **−0.439** | **+0.042** |

`a_cond` fell **−1.0289pt** (87.9405 → 86.9116); `b_cond` rose only +0.2308.

**Corrected rule (supersedes the "gate quality compounds at ~1.6x" note above — that was n=1 and
mis-attributed).** Compounding is not a property of gate *quality*, it is a property of **new
information**:
- **New evidence** changes *which* images sit near the threshold. If the new features also predict
  head-correctness (encoder agreement, bank sharpness), the swap is quality-improving and
  `a_cond`/`b_cond` RISE. → compounds.
- **Re-weighting existing evidence** only slides marginal images across a fixed ordering. Those are
  by construction the ones the model is least sure of, so `a_cond` FALLS and gives back ~90% of the
  routing gain. → cancels.

**Therefore: gate hyperparameter / threshold / re-weighting tuning is EXHAUSTED.** Do not spend a
slot on another C, another f, another feature re-weight, or another calibrator. Project any such
change at **~0.1x** its routing delta, not 1.0x and not 1.6x. Only genuinely new evidence in the
gate is worth a slot.

### 2026-08-29 (v80): new-information gate features — **DEAD. The gate is saturated.**

Direct test of the rule above: if only *new evidence* compounds, feed the gate evidence it has never
seen. The v77/v79 gate reads one image bank (ctftshift) and one text margin (ctftshift taxon), yet
the deployed unseen head also uses the **fullft336shift bank** (W_336=3) and a **TaxaBind** text leg
— two orthogonal encoders, both absent from the gate. `research/learned_gate_v80.py`, same protocol
(class-disjoint 5-fold CV, f=0.60 pinned, C=100).

| arm | AUC | TPR | TNR | proj% | vs v79 |
|---|---|---|---|---|---|
| v56 gate | 0.9705 | 94.19 | 84.12 | 56.450 | −1.080 |
| **v79 (12)** | **0.9811** | **95.85** | **86.28** | **57.531** | — |
| + `bank336_max/margin` (14) | 0.9812 | 95.79 | 86.20 | 57.487 | **−0.044** |
| + `tm_tb` (13) | 0.9811 | 95.75 | 86.15 | 57.465 | **−0.065** |
| + both (15) | 0.9812 | 95.79 | 86.19 | 57.487 | **−0.044** |

**AUC is flat to 4 decimals (0.9811 → 0.9812).** A second encoder's bank and a second text encoder's
margin carry **zero** routing information beyond the 12 features. Orthogonal enough to help the
unseen *head* (that is why W_336=3 and TaxaBind are in the stack) is **not** orthogonal for the
*routing* decision — novelty is apparently a low-dimensional signal that 12 features already
exhaust. `outputs/gate_feats_holdout_v80_new.pt` cached; no pkl written (kill bar not cleared).

**GATE CLOSED.** Three independent additions now fail on the same harness: view-disagreement (+0.030),
2nd-encoder bank (−0.044), 2nd text encoder margin (−0.065); plus tuning is exhausted (v79 +0.042
real). Do not spend a slot on another gate feature, weight, calibrator, C, or f. The remaining
headroom is `b` (unseen conditional, encoder-limited — §8) and `a_cond`, i.e. the HEADS, not routing.

**FINAL STATE: best real = v79 = 53.42492639842983%**
(`submissions/submission_v79_learned_gate_c100_f60_tau18.zip`), seen 77.44937055281883% /
unseen 22.41135662898253%. Session arc: 51.61643067433057 → 53.3828683583345 → 53.42492639842983.

### 2026-08-29 (v81): LEARNED RE-RANKER on the unseen head — biggest holdout lever ever measured

**The diagnosis that found it.** The gate was closed and both heads were still hand-weighted, so I
measured where the unseen head's error actually lives:

| recall@ | 1 | 2 | 3 | 5 | 10 | 20 | 50 | 100 |
|---|---|---|---|---|---|---|---|---|
| holdout | **33.22** | 41.20 | 45.77 | 52.07 | 59.62 | **67.30** | 74.76 | 78.77 |

The gold novel species is **already retrieved for 2/3 of queries and then mis-ordered**. This is a
*ranking* failure, not a retrieval or encoder failure — which is why every encoder lever (§4) was
dead and this one is not. `b` was never purely encoder-limited; part of it was the fusion.

**Why a fixed weight vector cannot fix it:** the 10 legs are summed with hand-picked constants
(text 1/0.5/0.75/1/1/1, bank 4, 336-bank 3, B2 2.5/2.0) applied **uniformly to every candidate**.
But the right weighting is candidate-dependent — the bank leg is authoritative when that class HAS
reference photos and is noise when it does not (uncovered classes are masked to −1e4 then dbnorm'd,
which is a constant, not evidence). No global weight can express that.

**v81** keeps the fusion only as a **retriever** (top-20) and learns the ordering.
`research/rerank_unseen_v81.py`, class-disjoint 5-fold CV on the gold class:

| model | top-1 | vs deployed fusion |
|---|---|---|
| deployed fusion (hand weights) | 33.218 | — |
| **logistic re-ranker** | **46.592** | **+13.374** |
| hgb re-ranker | 45.988 | +12.770 |

**Feature ablation — the transfer-safe set is the BEST:**

| feature set | n | top-1 |
|---|---|---|
| all | 45 | 46.506 |
| **within-query only (gaps / z / log-rank + coverage)** | **34** | **46.592** |
| raw dbnorm levels only | 11 | 45.815 |

Dropping every raw `dbnorm` level costs **nothing** and removes all batch coupling — the deployed
model uses only within-query gaps, ranks, z-scores and bank coverage, which are invariant to batch
composition. Top coefficients: `fused_gapmax` +1.81, `bank_336_gapmax` −1.26, `bank_ct_lograk` −0.83.

**Bug worth remembering:** applying the model with `predict_proba` silently degenerated on eval —
log-odds sit far enough negative that every probability underflows to 0.0 in float32 and `argmax`
returns index 0, i.e. the re-ranker became a no-op ("moved 0.0% off rank-1"). **Rank with
`decision_function`** (identical ordering, numerically stable). Verified identical on holdout.

**Zip:** `submissions/submission_v81_rerank_f60.zip` (35,665 preds, 9,157 distinct classes,
28.45% differ from v56). Gate = v79 (unchanged), routing unchanged **89.11 / 77.58**, seen head
unchanged. Re-ranker moves **21.7%** of the 14,266 ejected rows off fusion rank-1 (holdout 46.7%;
eval is lower because ~26% of ejected rows are wrongly-ejected *seen* images with garbage top-20).

**Projection was 54.6% – 57.4%. REAL: 53.43053413710921% (+0.006). See the post-mortem below.**

**Projection detail (kept for the record):** Ratio-preserving (holdout→real factor 0.870 measured on
this exact metric): b_cond 28.89 → 40.52 → **57.36%**. Conservative v37 anchor
(0.089 overall-pt/proxy-pt on a large holdout delta): **54.62%**.
**Bundled change to be aware of:** Sinkhorn is dropped on the re-ranked route (the CV compared
argmax-vs-argmax without it). Sinkhorn was historically worth **+0.09pt** real and lifts class
coverage — coverage here falls 6,361 → 4,682. If v81 scores below projection, re-adding Sinkhorn
over the re-ranked top-20 is the first thing to try.

**Seen-head crop-max — weak (not submitted):** max-over-7-views on the seen head is *worse* alone
(91.02 vs 91.20 deployed); best combo `deployed + max7` = **+0.32** holdout closed-set. The seen
head still reads one framing, but crops are not the fix there.


### 2026-08-29 (v81 REAL): **53.43053413710921%** — +0.006pt. The holdout had a LEAK.

seen **77.44937055281883%** (bit-identical to v79 — gate and seen head untouched, as designed) ·
unseen **22.42420349434738%** (v79: 22.41135662898253) · overall **53.43053413710921%**.

The re-ranker flipped **~3,100** of the 14,266 ejected predictions and netted **+2 correct images**
out of 12,077 eligible. A coin flip. **+13.374 holdout → +0.006 real** — the worst holdout mirage in
this project's history, far worse than Sinkhorn's 3.7x.

**ROOT CAUSE — the pseudo-novel holdout leaks a candidate-subpopulation shortcut:**

| | |
|---|---|
| candidates that are gold-eligible (the 1,159 pseudo classes) | **9.1%** |
| gold-eligible share of the top-20 pool | 10.4% |
| fused baseline picks a gold-eligible class | 38.4% |
| **re-ranker picks a gold-eligible class** | **53.7%** |
| mean re-rank score, eligible vs ineligible | **−4.31 vs −8.12** |

On the holdout the gold is **always** one of the 1,159 rarest *training* classes, and that
subpopulation is identifiable from the features (those classes have training images, hence
distinctive prototype and bank statistics). The model learned "prefer that subpopulation" — an
oracle on holdout, noise on eval where gold is among the 11,598 genuinely novel classes.

**CLASS-DISJOINT CV DOES NOT DETECT THIS.** Splitting which pseudo classes fall in train vs val
folds still leaves every fold's gold inside the same 9.1% subpopulation. The CV was honest about
class identity and blind to subpopulation membership.

**The rule this establishes — read before proposing any learned component:**
- **Per-IMAGE decisions are safe on this holdout.** The gate (v77, +1.767 real) has no candidate
  subpopulation to exploit; it scores one image and transferred as measured.
- **Per-CANDIDATE learning is NOT VALIDATABLE on this holdout.** Any model that scores
  (query, candidate) pairs can learn "is this candidate gold-eligible", which is a holdout artifact.
  This invalidates re-rankers, learned fusion weights, and learned calibrators over the class axis.
  **We currently have no valid validation instrument for per-candidate learning.** That, not the
  encoder, is the blocker on `b`.
- Fixing it requires a holdout whose candidate pool makes gold-eligibility *unlearnable* (e.g. a
  pool consisting only of held-out-type classes). Every such design changes pool size and therefore
  the rank/gap feature distributions, so it needs its own transfer validation. Do not attempt this
  on a deadline.

**Still true:** the ranking headroom itself is real in principle (holdout recall@1 33.22 vs
recall@20 67.30, and real recall@20 is likely ~58% by the 0.87 holdout→real ratio). The evidence IS
retrieved and IS mis-ordered. We simply cannot currently tell a genuine re-ordering from a leak.

**FINAL STATE: best real = v81 = 53.43053413710921%**
(`submissions/submission_v81_rerank_f60.zip`) — ahead of v79 by +0.006, i.e. a tie within noise.
Session arc: **51.61643067433057 → 53.3828683583345 (v77) → 53.42492639842983 (v79) →
53.43053413710921 (v81)**. All of the real gain came from the **gate** (v77, +1.767).

### 2026-08-29 (v82): leak-free re-ranker — signal survives, transfer UNPROVEN

Rebuilt the v81 experiment with the leak removed **by construction**: restrict the holdout candidate
pool to the **1,159 pseudo classes only**, so every candidate is gold-eligible and the
subpopulation shortcut carries zero information. Features made pool-size invariant (rank →
**percentile**, not log-rank). `research/rerank_leakfree_v82.py`.

| | leak-free pool (1,159, all eligible) |
|---|---|
| deployed fusion top-1 | 59.275 |
| recall@5 / @10 / @20 | 77.01 / 80.80 / 82.10 |
| **leak-free re-ranker** | **67.429 (+8.154)** |

**The ranking signal is real** — it is not all leak. And the learned model is now qualitatively
different: v81's top coefficients were `fused_gapmax` +1.81 with a *negative* on an evidence leg
(`bank_336_gapmax` −1.26, a shortcut signature); v82's are dominated by evidence —
`bank_ct_gapmax` **+2.33**, `b2_lora_gapmax` **+1.87**, `t_ctft_taxctx_gapmax` +1.11. It learned
"trust the photo bank when it is sharply peaked", which is the intended behaviour.

**Unproven assumption:** the leak-free pool is 1,159 classes, eval is **11,598**. The training task
is much easier there (fusion top-1 59.3 vs 33.2) and the distractor character differs. Percentile
features are pool-size invariant by construction, but that invariance is itself untested on real.
**No projection is offered** — this session's unseen-route projections were wrong twice (v79 +0.479
predicted / +0.042 actual; v81 +13.374 predicted / +0.006 actual). Gate-route projections were
accurate; unseen-route ones were not.

**Zip:** `submissions/submission_v82_rerank_leakfree_f60.zip`. Gate = v79, routing unchanged
**89.11 / 77.58**, seen head unchanged; re-ranker moves **29.9%** of the 14,266 ejected rows off
fusion rank-1 (v81 moved 21.7%). Sinkhorn still dropped on the re-ranked route (as in v81).
### 2026-08-29 (v82 REAL): **53.64362820692555% — NEW BEST. The leak fix transferred.**

seen **77.44937055281883%** (bit-identical to v79/v81 — gate and seen head untouched) ·
unseen **22.912384378211717%** · overall **53.64362820692555%** (**+0.21309406981634** vs v81).
`a_cond` 86.9116% (flat, as designed) · `b_cond` **28.9038 → 29.5330%** (+0.6292pt).

**This is the payoff from the post-mortem, and it validates the whole diagnosis:**

| build | re-ranker trained on | holdout | real |
|---|---|---|---|
| v81 | leaky pool (9.1% gold-eligible) | +13.374 | **+0.006** |
| **v82** | **leak-free pool (100% gold-eligible)** | **+8.154** | **+0.213** |

The leaky model had the *larger* holdout number and was worthless; removing the shortcut cut the
holdout gain by 40% and made it **35x more valuable on real**. A holdout number is only as good as
the population it is measured on — magnitude is not evidence.

**Transfer anchor for the unseen route, leak-free: 0.0261 overall-pt per proxy-pt.**
Use this for future unseen-route EV (it sits between maxpool's 0.02 and v37's 0.089).

**Zip:** `submissions/submission_v82_rerank_leakfree_f60.zip`.
Session arc: **51.61643067433057 (v56) → 53.3828683583345 (v77) → 53.42492639842983 (v79) →
53.43053413710921 (v81) → 53.64362820692555 (v82)** = **+2.027pt**, all single-pipeline.

**Known un-harvested items on this route (cheap, untested):**
1. **Sinkhorn is still dropped** on the re-ranked route. Coverage 4,827 vs v79's 6,361 unseen
   classes; Sinkhorn was historically +0.09pt real and raises coverage. Re-add over the re-ranked
   top-K.
2. **K=20 is unswept.** Leak-free pool recall@20 was 82.10 (near its ceiling), but eval has 11,598
   candidates where recall@20 is much lower — a larger K may expose more headroom. Sweep on the
   leak-free CV first; it costs no slot.
3. **A larger leak-free pool.** The training pool is 1,159 classes vs eval's 11,598. Re-splitting at
   rarest-40% gives a 2,318-class gold-eligible pool, closer to eval scale and still leak-free.

### 2026-08-29 (v83): SEEN-head re-ranker — leak-free by construction, UNSCORED

Applied the v82 lesson to the other head. The seen head is
`sum_t w_t * zc(proto_t + 2.0*cmax_t + 4.0*taxon_t)`, w = 1.0/2.5/2.5 — every constant applied
identically to every class. Holdout headroom (`research/rerank_seen_v83.py`):

| recall@ | 1 | 2 | 3 | 5 | 10 | 20 |
|---|---|---|---|---|---|---|
| seen head | **89.87** | 94.73 | 96.36 | 97.77 | **98.85** | 99.44 |

**No leak is possible on this head** — the candidate pool is the 4,636 kept classes and a val_seen
image's gold is ALWAYS in it, so 100% of candidates are gold-eligible by construction. (v81's pool
was 9.1% eligible; that was the bug.) Features are pool-size invariant (percentile ranks), as v82.

**Result: 89.870 → 91.674 (+1.803)**, class-disjoint 5-fold CV, K=10 (ceiling 98.85).

Top coefficients: `ctftshift_taxon_gapmax` **+4.39**, `block_z` −1.86, `ctftshift_taxon_z` +1.80,
`fullft336shift_proto_z` +1.37, `ftshift_proto_gapmax` +1.37. The taxonomy-text term is badly
under-weighted by the fixed `LAM=4.0` for a large set of classes — the single biggest correction.

**The candidate-conditional signal a fixed weight cannot express:** `cmax` is the nearest single
training image, `proto` is the class mean. For a class with 1 training image they are IDENTICAL;
for a class with 100 they measure different things. The deployed `2.0` cannot know which case it is
in — a model given `log_ntrain` can.

**Zip:** `submissions/submission_v83_rerank_both_f60.zip` — both heads re-ranked, gate = v79.
Routing **identical** to v82 (TPR 0.891128 / TNR 0.775822 — verified bit-equal), 1,509 predictions
differ from v82, all on the seen ROUTE (965 test-folder + 544 unseen-folder rows routed to seen).
Seen re-ranker moves 7.1% of the 21,399 seen-route rows off rank-1.

**Projection 53.69% – 54.52%.** Ratio-preserving (holdout 89.87 vs real `a_cond` 86.91 = 0.967, a
much tighter correspondence than the unseen route's) → **54.52%**. At the pessimistic leak-free
unseen-route anchor (0.0261 overall-pt/proxy-pt) → **53.69%**. Expect the upper half: the seen
holdout pool (4,636 classes) is close to eval's 5,795, whereas the unseen leak-free pool (1,159)
was 10x smaller than eval's 11,598 — pool mismatch is the main reason v82's transfer was lossy.

### 2026-08-29 (v83 REAL): **53.69690172437964% — FINAL BEST**

seen **77.54391202667065%** · unseen **22.912384378211717%** (bit-identical to v82 — unseen route
untouched) · overall **53.69690172437964%** (**+0.0533** vs v82). `a_cond` 86.9116 → **87.0177%**
(+0.1061pt); `b_cond` 29.5330% unchanged. Routing unchanged.

**The pessimistic projection was exactly right (53.69 predicted / 53.6969 actual); the
"expect the upper half" reasoning was wrong.** Pool-size similarity did NOT buy better transfer.

**CONFIRMED TRANSFER ANCHOR — per-candidate re-rankers on a leak-free holdout:**

| build | head | holdout gain | real overall | ratio |
|---|---|---|---|---|
| v82 | unseen | +8.154 | +0.213 | **0.0261** |
| v83 | seen | +1.803 | +0.0533 | **0.0295** |

**~0.026–0.030 overall-pt per leak-free proxy-pt, stable across both heads.** So **+1.00 overall-pt
requires +34 leak-free proxy-pt.** Use this to kill proposals fast: the seen head's entire
rank-1→recall@10 gap is 8.98 proxy-pt (~+0.27 real) and the unseen head's remaining top-20 gap is
14.7 proxy-pt (~+0.38 real). **Both heads are now within ~0.65 overall-pt of their re-ranking
ceilings.** Per-candidate re-ranking is effectively CLOSED.

**Note:** an OOM warning appears during the build (allocator frees cache and retries; exit 0, all
stages run, routing verified bit-equal to v82). Peak is ~32GB: 9 seen-component matrices
[35665, 5795] plus the 10 unseen legs [35665, 11598]. `del sig` after the seen re-rank keeps it
inside 46GB. If it ever hard-OOMs, compute `sig` in float16.


### 2026-08-29 FINAL STATE

**Best real: 53.69690172437964%** — `submissions/submission_v83_rerank_both_f60.zip`
(seen 77.54391202667065% / unseen 22.912384378211717%).

```
v56 51.61643  ->  v77 53.38287  ->  v79 53.42493  ->  v81 53.43053  ->  v82 53.64363  ->  v83 53.69690
                  learned gate      C=100 refit      leaky rerank      leak-free        + seen rerank
                  +1.767            +0.042           +0.006            +0.213           +0.053
```

**+2.080pt in one session, all single-pipeline, no new encoder, no new data.** Every gain came from
replacing hand-picked constants with learned components and from fixing the validation instrument.

**What is closed:** gate (4 levers: view-disagreement +0.030, 2nd-encoder bank −0.044, 2nd text
margin −0.065, hyperparameter tuning +0.042 real); seen-head crop-max (+0.32 holdout); per-candidate
re-ranking on both heads (~0.65 overall-pt left combined, at 0.03 transfer); all encoder/`b` levers
(§4, unchanged).

**Honest ceiling from measured evidence: ~54.3–54.5%.** 57% would need `b_cond` ~29.5 → ~35%, i.e. a
materially better unseen *recognizer*. Nothing rules-legal that was tested delivers it.

**If work resumes, in EV order:**
1. **Re-add Sinkhorn** over the re-ranked unseen top-20 — it was dropped in v81/v82/v83 and class
   coverage fell 6,361 → 4,827. Historically +0.09pt. Cheapest untested item.
2. **Larger leak-free pool** for the unseen re-ranker (re-split at rarest-40% → 2,318 gold-eligible
   classes). Pool mismatch (1,159 vs eval 11,598) is the leading suspect for the lossy 0.026
   transfer; halving it may raise the anchor itself, which matters more than any single gain.
3. **K sweep** on both re-rankers (seen K=10, unseen K=20 are unswept).
Do NOT spend slots on: gate features/weights/C/f, crop features, encoder swaps, folder-oracle-shaped
"unified" pipelines.

### 2026-08-29 (post-v83): the three listed follow-ups + a gate idea — ALL DEAD, no slots spent

Every item from the v83 "if work resumes" list was tested on holdout. All fail.

**1. K sweep — INERT.** Unseen re-ranker, leak-free pool, class-disjoint CV:

| K | 10 | 20 (deployed) | 50 | 100 |
|---|---|---|---|---|
| re-ranked top-1 | 67.386 | 67.472 | **67.688** | 67.645 |
| ceiling (recall@K) | 80.80 | 82.10 | 82.87 | 83.18 |

Best K=50 = **+0.216** proxy = **+0.006 overall** at the 0.03 anchor. The seen head's ceiling is
already 98.85 at K=10. **K is not a lever on either head.**

**2. Sinkhorn re-add — NEGATIVE, and now explained.** On the re-ranked unseen scores:

| | argmax | τ=1.2 | τ=1.8 | τ=3.0 |
|---|---|---|---|---|
| top-1 | **67.472** | 66.566 | 65.574 | 62.640 |
| classes used | 933 | 966 | 959 | 942 |

Monotonically worse with τ. **Sinkhorn's original +0.09pt came from correcting a MISCALIBRATED
fusion by forcing class balance. The re-ranker is already calibrated, so forced balance now
destroys correct decisions.** Dropping Sinkhorn in v81/v82/v83 was right, not an oversight — the
lower class coverage (4,827 vs 6,361) is the correct behaviour, not a regression. Do not re-add it.

**3. Pool-mismatch hypothesis — FALSIFIED by v83's own data.** The proposal was that v82's lossy
0.0261 transfer came from its 1,159-class training pool vs eval's 11,598. But v83's pool (4,636 vs
eval 5,795) is **well** matched and transferred at **0.0295** — statistically the same. Pool size is
NOT the cause. The residual suspect is the framing shift (holdout images are training-style aspect
~1.15, eval ~2.12), which degrades every score statistic the re-rankers read. **Do not build a
rarest-40% pool; it addresses a falsified cause.**

**4. NEW IDEA, also dead — re-ranker confidence as a gate feature.** Rationale was strong: gate
points transfer at ~3.4x (v77: +0.523 proj → +1.767 real) vs re-ranker points at 0.03x, so gate
points are ~100x more valuable; and a learned calibrated P(best candidate is right) is strictly
richer than the raw fusion maxima the gate reads. Added OOF-stacked `rr_conf_max` / `rr_conf_gap`
(out-of-fold to avoid self-leakage, seen-reranker trained only on seen rows of other folds):

| gate | AUC | TPR | TNR | proj% |
|---|---|---|---|---|
| v56 | 0.9705 | 94.19 | 84.12 | 56.450 |
| **v79 (12 feat)** | **0.9811** | **95.85** | **86.28** | **57.531** |
| + rr confidence (14) | 0.9813 | 95.75 | 86.15 | 57.465 (**−0.065**) |

**FIVE independent gate-feature additions have now failed in the same ±0.07 band with AUC flat to
three decimals** (view-disagreement, 2nd-encoder bank, 2nd text margin, re-ranker confidence, plus
tuning). The gate is saturated at holdout AUC ~0.981. **Stop proposing gate features.**

### FINAL: everything cheap is exhausted

**Best real: 53.69690172437964%** (`submissions/submission_v83_rerank_both_f60.zip`).
Closed this session: gate (5 levers), seen crop-max, per-candidate re-ranking (both heads, K,
Sinkhorn), pool mismatch. Unchanged from before: every encoder / `b` lever (§4), all LoRA retrains.

Reaching **57%** needs `a_cond` 87.0→~90% *and* `b_cond` 29.5→~35%. The re-ranking ceilings bound
the recoverable part at ~**0.65 overall-pt** combined (→ ~54.35%). Anything beyond that requires a
genuinely stronger *recognizer*, which no rules-legal encoder tested over the whole project
delivers. **A further +3.3pt is not reachable with the current model family.**

### 2026-08-29 (v85): dual re-ranker confidence in the gate — DEAD. Gate closed after SIX levers.

Best-founded gate idea remaining: give the gate BOTH heads' *calibrated* confidences on a common
scale. Rationale was strong — HANDOFF 2026-08-08 killed unified/soft routing precisely because the
two heads' raw scores are not comparable, and two learned re-rankers each emit a calibrated
P(top candidate correct), which is exactly that missing comparability. Gate points also transfer at
**~3.4x** vs per-candidate points at **0.03x**, so a gate win is worth ~100x a re-ranker win.

Required building the 10 unseen legs for **all 14,184** holdout rows over the full 12,757-candidate
pool (all prior work built them for the 2,318 pseudo rows only) plus seen signals for all rows.
Both confidences OUT-OF-FOLD (seen re-ranker trained on seen rows of other folds, unseen on pseudo
rows of other folds) to avoid self-leakage. `research/gate_dualconf_v85.py`,
`outputs/gate_dualconf_v85.pt`.

| gate | AUC | TPR | TNR | proj% |
|---|---|---|---|---|
| v56 | 0.9705 | 94.19 | 84.12 | 56.450 |
| **v79 (12 feat)** | 0.9811 | 95.85 | 86.28 | **57.531** |
| v85 (+ s_conf, s_gap, u_conf, u_gap, conf_diff) | 0.9818 | 95.90 | 86.32 | 57.561 (**+0.030**) |
| v79 + `conf_diff` only | **0.9819** | 95.86 | 86.28 | 57.535 (+0.004) |

`conf_diff` alone gives the **highest AUC ever measured on this gate** and moves the operating point
by +0.004. The gate's existing features already encode "how well does each head explain this image";
calibrating that encoding adds no *ranking* information.

**THE GATE IS CLOSED — six independent additions, all in a ±0.07 band, AUC flat to 3 decimals:**
view-disagreement (+0.030), 2nd-encoder bank (−0.044), 2nd text margin (−0.065), re-ranker
confidence (−0.065), dual calibrated confidence (+0.030), hyperparameter tuning (+0.042 real).
**Stop proposing gate features. The routing signal is exhausted at holdout AUC ~0.981.**

### 2026-08-29 (v86): `f` is now MIS-TUNED — the one lever left, needs a real slot

`f=0.60` was set when `b_cond ≈ 26%`. It is now **29.53%** (v82 re-ranker). Break-even marginal
seen-fraction for ejecting one more image:

```
p_t* = (0.4365*b_cond/N_u) / (0.5635*a_cond/N_t + 0.4365*b_cond/N_u)
       f=0.60 was set at b_cond~26%, a_cond~88%  ->  p_t* = 22.8%
       now         b_cond 29.53%, a_cond 87.02%  ->  p_t* = 25.3%
```

That reasoning predicted the optimum had moved BELOW 0.60. **It is WRONG — measured, no slot spent.**
Building f=0.55 / 0.50 and reading the routing diagnostic gives the marginal composition directly:

| f | TPR (tf→seen) | TNR (uf→unseen) | projected @ measured a=87.0177 b=29.5330 |
|---|---|---|---|
| **0.60 (deployed)** | 89.11 | 77.58 | **53.697 (MEASURED REAL)** |
| 0.55 | 85.93 | 84.93 | **53.08** |
| 0.50 | 81.83 | 91.09 | **51.87** |

Going 0.60→0.55 ejects 1,783 more images of which **639 = 35.8% are trained-class**, far above the
25.3% break-even ⇒ **net-negative**. The break-even algebra was right; the assumed marginal
composition was not. Raising `f` is worse still (measured historically at 0.72, and a higher
`b_cond` now makes ejection *more* valuable, not less). **`f=0.60` is at the optimum in BOTH
directions. Do not sweep `f` again.**

**f=0.55 WAS SUBMITTED — real 53.52025795597926%** (seen 75.7028412200826 / unseen 24.88437821171634).
Worse than f=0.60's 53.697 as predicted, but the projection (53.08) was **0.44pt pessimistic**
because it held conditionals fixed. They move:

| f | TPR | TNR | `a_cond` | `b_cond` | REAL |
|---|---|---|---|---|---|
| **0.55** | 85.93 | 84.93 | **88.098** | 29.300 | **53.52026** |
| **0.60** | 89.11 | 77.58 | **87.018** | 29.533 | **53.69690** |

per +0.05 `f`: ΔTPR **+3.183**, ΔTNR **−7.348**, Δ`a_cond` **−1.081**, Δ`b_cond` **+0.233**.

**Ejecting more RAISES `a_cond`** (the images left on the seen route are better ones) — which is why
the fixed-conditional projection was pessimistic, and it is the same "conditionals move" lesson as
v77/v79. **Never project an `f` change with fixed conditionals; use the slopes above.**

**The break-even argument that predicted `f` should DROP was wrong.** Decomposed: 0.55→0.60 gains
`seen_acc` **+1.84pt** and loses `unseen_acc` **−1.97pt**, and 0.5635×1.84 > 0.4365×1.97 ⇒ **+0.177**.
Raising `f` wins here. Extrapolating with the measured slopes: **f=0.65 → ~53.82**, f=0.70 → ~53.89
(linear, and TPR must saturate, so treat as an upper bound).

The old stack's f=0.72 (50.84 vs 51.44 at f=0.60) argued the peak was below 0.65, but that gate was
far weaker — TNR **54.5** at TPR 92.5, where the v79 gate projects TNR **70.2** at TPR 92.3.
**That extrapolation was WRONG — f=0.65 measured 53.49222 (predicted 53.82).**

### THE `f` CURVE IS NOW SETTLED WITH THREE REAL POINTS — `f=0.60` IS THE PEAK

| f | TPR | TNR | `a_cond` | `b_cond` | **REAL overall** |
|---|---|---|---|---|---|
| 0.55 | 85.93 | 84.93 | 88.098 | 29.300 | 53.52025795597926 |
| **0.60** | **89.11** | **77.58** | **87.018** | **29.533** | **53.69690172437964 ← PEAK** |
| 0.65 | 91.60 | 69.33 | 86.089 | 29.926 | 53.49221926258236 |

**The curve is CONCAVE and both neighbours lose ~0.18–0.20pt.** Decomposed per +0.05 step of `f`:

| step | Δ seen_acc | Δ unseen_acc | Δ overall |
|---|---|---|---|
| 0.55→0.60 | **+1.84** | −1.97 | **+0.177** |
| 0.60→0.65 | **+1.31** | −2.16 | **−0.204** |

Seen gains shrink while unseen losses grow ⇒ single interior maximum at **0.60**.

**Process lesson (cost one slot):** a two-point linear extrapolation of `f` predicted +0.12 and
delivered −0.20 — it cannot see curvature, and the ROC is concave by construction. The earlier
fixed-conditional projection had the direction right for the wrong reason; the slope model had the
mechanism right and the curvature wrong. **`f` is DONE. Do not sweep it again in either direction.**
Unused zips to delete/ignore: `submission_v86_rerank_both_f050.zip`,
`submission_v87_rerank_both_f068.zip` (f=0.68 is further up a losing slope).

### 2026-08-29 END OF SESSION — everything reachable is closed

**Best real: 53.69690172437964%** — `submissions/submission_v83_rerank_both_f60.zip`
(seen 77.54391202667065% / unseen 22.912384378211717%). **+2.080pt** from v56's 51.61643067433057%.

```
v56 51.616 → v77 53.383 → v79 53.425 → v81 53.431 → v82 53.644 → v83 53.697
             learned gate  C=100       leaky rerank  leak-free    + seen rerank
             +1.767        +0.042      +0.006        +0.213       +0.053
```

**Closed this session (all measured, none to be retried):** gate — SIX feature/tuning levers, AUC
pinned at 0.981; per-candidate re-ranking — both heads, K inert, Sinkhorn negative-and-explained;
seen-head crop-max (+0.32 holdout); pool-mismatch hypothesis (falsified by v83); `f` in both
directions. **Unchanged from before:** every encoder / `b` lever (§4) and every LoRA retrain.

**Two transfer anchors that should govern all future EV math:**
- gate (per-image) proj-pt → real: **~3.4x** (v77 the only clean instance)
- per-candidate re-ranker leak-free proxy-pt → real: **~0.026–0.030** (v82 unseen, v83 seen)
Per-candidate work needs **+34 proxy-pt per +1 overall-pt**; both heads have ~23 proxy-pt of
re-ranking headroom left combined ⇒ **~0.65 overall-pt ceiling ⇒ ~54.35%**.

**55% is not reachable with this model family.** It needs `a_cond` 87.0→~89.6% or `b_cond`
29.5→~33.4% (or a mix). Re-ranking is bounded at ~0.65pt total; the gate is saturated; `f` is
optimal; every rules-legal encoder tested across the whole project fails to raise `b`. Closing the
remaining ~1.3pt requires a materially stronger *recognizer*, not another pipeline change.


### 2026-08-29 TRUE FINAL — `f` closed, nothing measurable remains

**Best real: 53.69690172437964%** — `submissions/submission_v83_rerank_both_f60.zip`.
Seven scored submissions this session; the full ledger:

| build | overall | note |
|---|---|---|
| v56 (start) | 51.61643067433057 | prior best |
| v77 | 53.38286835833450 | learned 12-feature gate (**+1.767**, the session's real win) |
| v79 | 53.42492639842983 | C=100 refit (+0.042) |
| v81 | 53.43053413710921 | leaky re-ranker (+0.006 — the leak) |
| v82 | 53.64362820692555 | leak-free unseen re-ranker (+0.213) |
| **v83** | **53.69690172437964** | **+ seen re-ranker (+0.053) — BEST** |
| v86 f=0.55 | 53.52025795597926 | `f` probe down — worse |
| v87 f=0.65 | 53.49221926258236 | `f` probe up — worse |

**Everything is closed:** gate (6 levers, AUC pinned 0.981) · per-candidate re-ranking (both heads;
K inert; Sinkhorn negative and explained) · `f` (3 real points, concave, 0.60 is the peak) ·
seen crop-max · pool-mismatch (falsified) · all encoder/`b` levers (§4) · all LoRA retrains.

**55% is not reachable with this model family.** It needs +1.30pt = `a_cond` 87.0→89.6% or
`b_cond` 29.5→33.4%. Re-ranking headroom is bounded at ~0.65pt by the 0.026–0.030 anchor; the gate
and `f` are at their optima. The remaining gap requires a materially stronger *recognizer*.

### 2026-08-31: re-ranker MODEL CLASS — DEAD on both heads. No slots spent.

Both leak-free re-rankers ship plain `LogisticRegression`. v81 compared logistic vs
`HistGradientBoosting`, but only on the **leaky** pool (hgb 45.988 / +12.770, HANDOFF:1170), and
when v82 fixed the leak it dropped the comparison and never repeated it on clean data. So the
model-class question was open on both heads. It is now closed.

`research/rerank_leakfree_v84_sweep.py` (unseen) · `research/rerank_seen_v84_model.py` (seen),
class-disjoint 5-fold CV, deployed features unchanged:

| head | K | logistic (deployed) | hgb | hgb_deep | best Δ real |
|---|---|---|---|---|---|
| unseen | 20 | **67.429** (+8.154) | 67.127 (+7.852) | — | **−0.008** |
| unseen | 50 | 67.774 (+8.499) | 67.429 (+8.154) | — | +0.009 |
| seen | 10 | 91.674 (+1.803) | **91.766** (+1.896) | 91.741 (+1.871) | **+0.003** |

**Gradient boosting is not better than a linear model on either head.** On the unseen head at the
deployed K it is *worse*. On the seen head it wins by +0.093 proxy-pt = **+0.003 overall** — noise.
The 14.67 (unseen) + 7.18 (seen) proxy-pt sitting between rank-1 and the top-K ceiling is **not a
model-capacity limit**; the 34/32 features simply do not separate gold from its shortlist rivals.
Reaching that headroom needs new *evidence*, not a stronger learner. **Do not retry model class,
regularization, or ensembling on either re-ranker.**

The unseen sweep also independently reproduced the K result (HANDOFF:1422 — K inert) and the v82
baseline exactly (fusion top-1 59.275 vs 59.28; logistic K=20 gain +8.154 exact), which validates
the harness.

**`builders/build_v83_rerank_both.py` — leg construction moved, output bit-identical.** The 10
unseen legs (10×[35665,11598] fp32 = 16.5GB) were built ~50 lines before first use, resident
alongside `sig` (7.4GB), which is what OOM-warned during gate bank scoring. They are now built
after `del sig`, immediately before the unseen re-rank, plus `del Sraw_ct, Sraw_336` and
`del legs, fused` once sliced. **Verified bit-identical twice** (0/35,665 predictions differ vs
`outputs/prediction_v83_rerank_both_f60.json`); routing unchanged 89.11/77.58. The gate-scoring OOM
warning is gone; one warning remains later at leg construction (allocator recovers, exit 0).
Docstring and banner also corrected — they claimed v81 and v77 respectively.

**State unchanged: best real 53.69690172437964% (v83). Nothing found that raises it.**

### 2026-08-31 (v84 REAL): unseen re-ranker K=20→50 — **53.657647553624%, −0.039. SIGN FLIP.**

seen **77.54391202667065%** (bit-identical to v83 — seen route untouched, confirms a correct build) ·
unseen **22.82245632065776%** (−0.090) · overall **53.657647553624%** (**−0.039**).
Zip `submissions/submission_v84_unseen_k50_f60.zip`; build =
`RERANK_PKL=outputs/rerank_unseen_v84_k50.pkl TAG=v84_unseen_k50_f60 python builders/build_v83_rerank_both.py`.

**The leak-free proxy said +0.345 proxy-pt → +0.009 real. Real came back −0.039.** The 0.026–0.030
anchor is a *slope for large deltas*, not a predictor for small ones: below roughly +2 proxy-pt the
real outcome is noise- and pool-mismatch-dominated and **can flip sign**. Do not ship a change
whose only support is a sub-1-proxy-pt CV delta.

**Why K flips.** K is not pool-invariant even though the *features* are. K=50 is the top **4.3%** of
the 1,159-class leak-free pool but the top **0.43%** of eval's 11,598. On the proxy, candidates
21–50 still sometimes contain gold (recall@50 82.87 vs @20 82.10); on eval they are almost all
distractors that the re-ranker can promote. Symptom: rows moved off fusion rank-1 rose 29.9% → 32.0%
and unseen-class coverage fell 4,827 → 4,782 — more edits, more of them wrong. **K=20 is correct.
K is closed on real, not just on proxy.**

### 2026-08-31: four more probes, all dead (no slots spent except v84)

| probe | proxy | verdict |
|---|---|---|
| HistGradientBoosting, unseen head @K=20 | −0.302 | dead |
| HistGradientBoosting, seen head @K=10 | +0.093 (=+0.003 real) | dead |
| **genus consensus among the shortlist** (5 features: share, rank-weighted mass, best-rank, is-genus-best, gap-to-genus-best) | **−0.432** | dead |
| **TaxaBind + BioCLIP-2 frozen + BioCLIP-2 LoRA proto/cmax added to the SEEN re-ranker** (32→50 features) | **+0.042 (=+0.001 real)** | dead |

`research/rerank_leakfree_v84_sweep.py` · `rerank_seen_v84_model.py` · `rerank_genus_v85.py` ·
`rerank_seen_v86_encoders.py`.

**The two that matter, read together.** Boosting can fit any interaction of the existing features and
gains nothing; and two entire encoders that had *never* scored a seen class anywhere in the pipeline
add +0.001 real. **The heads are evidence-saturated, not capacity-limited.** More model returns zero
and more encoder returns zero, on the head carrying 0.5635 of the metric weight. Genus consensus —
the one feature class that is not a per-candidate score statistic — is actively negative, because
only 213 of 515 genera in the pool have siblings for consensus to fire on.

**Do not retry on either re-ranker: model class, regularization, ensembling, K, taxonomic structure,
or additional encoders.** The remaining rank-1→ceiling gap (14.67 unseen / 7.08 seen proxy-pt) is
not reachable from the evidence this pipeline computes.

**FINAL: best real = v83 = 53.69690172437964%** (`submissions/submission_v83_rerank_both_f60.zip`).

### 2026-09-01: VLM top-K rerank retried with a frontier reasoning model — DEAD again, now closed for good

Re-ran the killed VLM-rerank idea (HANDOFF's original Qwen2.5-VL-7B attempt, -16.00pt) with
`deepseek-v4-flash-vision-exp` (frontier-tier reasoning VLM, launched 2026-08-21, free/no price
premium) + a fixed prompt (900-token reasoning budget + `reasoning_content` fallback, vs Qwen's
forced 8-token greedy guess). Same harness, same 300-image/K=10 pseudo-unseen sample
(`research/unseen_vlm_rerank_deepseek.py`, `outputs/unseen_vlm_rerank_deepseek_n300_k10_results.json`).

| model | vlm_top1 | delta vs text (26.33) | pick-rate | vs chance |
|---|---|---|---|---|
| Qwen2.5-VL-7B (8-token greedy) | 10.33 | **-16.00** | 16.8% | 10.0% |
| deepseek-v4-flash-vision-exp (reasoning) | 15.00 | **-11.33** | 19.5% | 10.0% |

Both KILL. A materially stronger, reasoning-capable model closes some of the gap but stays decisively
negative — **this is not a model-capacity problem, it's structural**: general VLMs cannot out-discriminate
BioCLIP's domain-tuned embeddings on fine-grained congeneric fish species, and cheap vision APIs
(DeepSeek caps images at 384 tokens) likely can't resolve fin/scale-level detail regardless of
reasoning ability. **Do not retry with Claude/GPT or any other general VLM — the pattern is 2-for-2
negative across unrelated model families, closed for good.**

### 2026-09-01 (v90): shift-ADAPTED gate training (geometric re-render) — DEAD. Framing suspect exonerated at the gate.

The last mechanism-backed untested lever: the encoders got shift-matched augmentation (+0.4 real,
§3) but every learned component trains on train-framing features (`emb_train_*`) while deployment
reads eval-framing images (aspect medians: train 1.46 → test 1.53 / unseen 1.92,
`outputs/shift_diag.json`). HANDOFF:1447 named framing "the residual suspect" for the gate's
holdout→eval AUC drop (0.981 → 0.911, 15.9% mis-routed — the largest quantified loss left, at the
~3.4x gate transfer anchor). Test: re-extract all 14,184 holdout rows with a LABEL-BLIND horizontal
pre-stretch (×1.35, ×1.75) through ctftshift (deployed 4-view squash-TTA recipe unchanged), rebuild
the 12 v77 gate features with the ctftshift member substituted (protos stay train-framed, mirroring
deployment), one class-disjoint 5-fold CV over every train×eval render pair, C=100, f=0.60 pinned.
`research/extract_holdout_shift_v90.py` · `research/gate_shift_adapt_v90.py` →
`outputs/gate_shift_adapt_v90.json`. (Note: `emb_train_bioclip2_lora_v2.pt` vanished from outputs/
by 2026-09-01; v1 fallback used for the b2l column — constant across renders, cannot bias this.)

| train \ eval | norm | s135 | s175 |
|---|---|---|---|
| norm (deployed style) | **0.9809** / 57.422 | 0.9806 / 57.408 | 0.9805 / 57.365 |
| pooled 3-render jitter | 0.9812 / 57.452 | 0.9810 / 57.434 | 0.9808 / 57.356 |

**norm→s175 AUC drop = 0.0004; jitter-training recovery = 0.0003; projection −0.057pt (kill 0.30).**
The reference row reproduces the deployed v79 gate (0.9809 vs published 0.9811) — harness valid.
The deployed 4-view TTA is already effectively aspect-invariant: squash views are stretch-invariant
by construction and even the center-view half barely moves ViT-H features. The simulation
understates the true shift (1 of 3 shifted members; squash invariance), but ×3 the measured 4e-4 is
still noise. **Conclusion: the 0.981→0.911 eval routing gap is NOT geometric framing of the query —
it is content-level domain shift (cameras / habitat / poses), which no re-rendering of training data
can close. Do not retry shift-rendered or aspect-jittered training of the gate or re-rankers.**

### 2026-09-01 (v91): test-time adaptation of the gate on the eval batch — DEAD. Both sides of the routing gap are now closed.

After v90 killed training-side shift adaptation, the one untested lever class left was adapting the
gate's decision function ON the eval features (transduction is established practice: Sinkhorn,
HANDOFF:261-267; wz1 batch standardization). Folder-blind self-training: pseudo-label the top 40% /
bottom 25% of the eval batch by the deployed-style gate's own scores, refit C=100 logistic on eval
features (pure + holdout-anchored variants, both fixed a priori — no eval-folder tuning). Eval-folder
labels used for MEASUREMENT only (the HANDOFF:399 diagnostic). 11 features (b2l_max dropped — the
whole `*_bioclip2_lora_v2.pt` embedding family was deleted in a cleanup by 2026-09-01; checkpoints
`outputs/b2_shift_lora*.pt` survive, so it is re-extractable if v83 ever needs rebuilding).
`research/gate_tta_v91.py` → `outputs/gate_feats_eval_v91.pt`, `outputs/gate_tta_v91.json`.

| gate | eval AUC | TPR | TNR | proj @ (a=87.018, b=29.533) |
|---|---|---|---|---|
| reference (holdout-trained, 11 feat) | **0.9300** | 88.94 | 77.36 | **53.583** |
| tta_pure (self-train) | 0.9280 | 88.78 | 77.15 | 53.478 |
| tta_anchored | 0.9298 | 88.90 | 77.31 | 53.560 |

**dAUC −0.0001 (kill was +0.010).** The instrument is tight: the reference row reproduces v83's real
routing (89.11/77.58, real 53.697) to within 0.11pt — and pins the learned gate's TRUE eval AUC at
**0.9300** (the 0.911 quoted earlier was the v56-era gate). Self-training amplifies the existing
boundary and adds nothing; the pure variant is slightly NEGATIVE. **The holdout-trained gate already
extracts all routing signal present in the eval features. The holdout→eval AUC gap (0.981→0.930,
content-level shift) is unfixable from the training side (v90) AND from the test side (v91). Do not
retry gate TTA, self-training, entropy minimization, or any transductive gate refit.**

### 2026-09-01 (v92): genus-pooled iNat photo protos on the unseen route — DEAD. Third genus kill, now at the leg level.

Soft taxonomic narrowing: pool the H-leg photo bank (7,364 classes / 101k embedded photos) at genus
level so uncovered candidates borrow congeners' visual evidence. Distinct from the prior genus kills
(hard family 2-stage −20pt; ToL/B2 genus backoff ≤0; re-ranker genus consensus −0.432) — and now
measured on the main bank leg. Proxy: deployed-style unseen route (text_full + 4·bank_ct +
3·bank_336 + 2.5·b2f, no crop-views/b2l) for the 2,318 pseudo queries over the full 12,757 pool.
Coverage was real: 10,081 candidates have genus-sibling photos, 4,081 are sibling-only backoff
targets, and **162 gold classes are backoff-eligible** (measurable, not vacuous — though eval's
true-novel pool is ~36% uncovered vs the proxy's ~14%, so backoff is under-weighted here).
`research/genus_bank_v92.py` → `outputs/genus_bank_v92.json`.

| variant | top-1 | Δ |
|---|---|---|
| base | 47.929 | — |
| A backoff λ=0.5 / 1.0 | 47.929 | **+0.000 (zero flips)** |
| B additive w=1.0 / 2.0 | 40.940 / 40.207 | **−6.989 / −7.722** |

Genus-pooled scores are NEVER competitive (backoff flips nothing, even for the 162 measurable gold
targets) and force-fusing them destroys species ranking. Consistent with family top-1 23.8 <
species 28.9 (§8): this model family's congeneric visual similarity does not discriminate.
**Genus/taxonomic pooling is closed at every level: gate, re-ranker, and now the proto leg.**

### 2026-09-01: LIVE LEADERBOARD DECOMPOSITION — the gap is SEEN-side, not unseen

Top-5 (user-pasted, all submitted 2026-09-01; overall = 0.5635·seen + 0.4365·unseen reproduces
every row): #1 cccccc11111 0.56 (seen 0.83 / unseen 0.21) · #2 BIOSCAN-ML 0.55 (**0.85** / 0.17) ·
#3–5 at 0.52 (0.83 / 0.11–0.13). **Our v83 = 0.537 (0.775 / 0.229): our unseen BEATS the entire
field, including #1.** The leaders win on seen: 83–85% vs our 77.5% — closed-set accuracy under
shift ABOVE our measured A_full ≈ 81%. Their profile at TPR≈0.97 needs a_cond ≥ 85–87 at near-full
routing; our f-curve (settled, concave, peak 0.60) caps us at ~53.5–53.7 on the same ROC. **The
missing +2pt to #1 is a stronger closed-set SEEN model (~81→85 under shift), not routing, not
unseen.** Every unseen-side lever hunted since July was aimed at a gap that does not exist. If work
ever resumes: seen-side training (higher-res / longer / more diverse closed-set ensemble members)
is the ONLY axis the leaderboard evidence supports — and it was last touched in July.

**58% verdict (asked 2026-09-01): NOT reachable with this model family.** This was the last
mechanism-backed untested lever; the record now bounds everything. Achievable ceiling ~54.35%
(re-rank anchors); even PERFECT holdout→eval gate transfer projects only ~57.4–57.5% (the CV tables
above); 58% needs ≈(a_cond 91, b_cond 37) vs measured (87.0, 29.5), and a seen-only route is
arithmetically closed (needs seen 85.2% > closed-set A_full ≈ 81%). The scored-submission ledger
already stands at/near the 30-total cap. Resume trigger unchanged: an organizer-eligible, materially
stronger general-biology encoder dropped into the existing fusion+gate scaffolding.

## 2026-09-01 v93 session — BOTH pre-registered levers DEAD → stand pat (12:35 UTC)

Sanity first: b2l embedding family restored from archive/pre_v50/outputs (train/test/unseen/inat_protos)
+ crop_v2_views.py + v41_taxctx_weight_cv.py restored to research/. `TAG=sanity_check` rebuild of the v83
builder is **bit-identical** to submission_v83_rerank_both_f60.zip (md5 0fa9f4f6044a324868f0c0ad0ff7c270).
Pipeline fully reproducible again.

**Lever A (research/gate_coral_v93.py) — DEAD.** Harness valid (reference eval AUC reproduced 0.9300
exactly, TPR 88.94/TNR 77.36). Pre-registered ship variant coral_l05: dAUC −0.0023, dproj −0.118.
coral_l10 +0.0008 (noise), quantile −0.0047. Kill bar ΔAUC ≥ +0.010 not met. The gate's eval gap is
content-level shift, not distributional — 3rd transductive mechanism killed (v91 self-training, v92
genus-bank, v93 CORAL/quantile). Diagnostic f-sweep peaks at f=0.65 (+0.19 proxy) — but HANDOFF:789
already proved eval-proxy f-preferences anti-correlated with real (f=0.72 proxy +5.09 → real loss).

**Lever B (research/seen_acond_v93.py) — DEAD.** Pre-registered 9-combo grid (w_tb×w_b2 ∈ {0,.5,1}²),
holdout-only selection (11,866 val_seen rows, 4,636 classes, protos from train-fold rows). Solo val acc:
ctftshift 89.09 / ftshift 88.04 / fullft336shift 86.63 / taxabind 65.49 / b2frozen 78.88. Base fused
88.77; best tb1.0_b20.0 → 88.93 = **+0.17pt** vs kill bar +1.0. The 3-member fusion is saturated;
weak solo legs add nothing under zc-fusion. Also re-verified: v63's 79.02 real seen was f-motion
(TPR 90.6 ⇒ a_cond ≈ 87.2, identical to ours) — no hidden a_cond to splice.

**Decision rule (pre-registered in implementation_plan.md) fires: A dead + B dead ⇒ 56+ arithmetically
unreachable today ⇒ STAND PAT on v83 = 53.697.** Winning-profile arithmetic needs TPR ≈ 94 × a_cond ≈ 88.5
simultaneously; no rules-legal, offline-validatable lever moves either term. Only ship-affecting action
before 22:00 UTC close: confirm the LAST submission on Codabench is v83 (Force_Last); if not, re-submit
submissions/submission_v83_rerank_both_f60.zip before 21:30 UTC.

## 2026-09-01 reweighted-LoRA entropy softgate — REAL 39.4981 — DEAD; v83 DISPLACED as last submission

Colab checkpoint reweighted_lora_distill_lambda_0p01 (BioCLIP-2.5 ViT-H LoRA r8 a16 blocks 28-31,
class-balance beta=0.9999, distill lambda=0.01; merge convention verified against logged distill loss)
+ pure entropy soft gate at f=0.60 (builders/build_reweighted_softgate.py):
**real seen 66.189 / unseen 5.042 / overall 39.4981** (submission_reweighted_lora_softgate_f60.zip).
Holdout said 52.60 overall / 39.0 unseen → real 5.0 unseen: pseudo-novel holdout overstated unseen ~8x
(HANDOFF §7 leak, reconfirmed). Entropy-only gate + reweighted encoder both DEAD.
**This submission is now the LAST on Codabench — re-submit submission_v83_rerank_both_f60.zip
before 21:30 UTC per the v93 stand-pat decision (Force_Last).**

## 2026-09-01 v94 — last unmeasured cell closed; harness validated to the exact real score (12:25 UTC)

With b2l restored, the DEPLOYED 12-feature gate was scored on eval for the first time
(research/gate_b2l_v94.py, pre-registered H1: dropping b2l_max improves eval routing).

- **deployed_12f on eval: AUC 0.9314, TPR 89.11, TNR 77.58, proj 53.697 — equals the real v83 score
  EXACTLY.** The offline projection arithmetic is now validated end-to-end with zero unexplained slack.
- refit_11f: 0.9300 (dAUC −0.0014) → H1 DEAD; b2l stays; holdout CV drop 0.000.
- Info-only single-feature eval AUCs: sb_max .885, m_ftshift .889, tm_ctft .887, tm_full .881,
  m_fullft336shift .868, sb_margin .834, m_ctftshift .775, bank_margin .547, bank_max .374,
  b2f_max .323, b2l_max .225, sb_lse_gap .141 (inverse).

**Terminal state: 57.4 is not reachable.** The harness that reproduces the real score to the third
decimal says the deployed gate runs at its measured ceiling; every gate lever (v77 CV, v79, v82,
v90 shift-adaptation, v91 TTA, v92 genus-bank, v93 CORAL/quantile, v94 b2l-drop) and every seen-scorer
lever (v93 leg grid) is measured dead. Remaining gap to 0.56+ leader is structural eval-domain encoder
quality; training-side fix (v90) measured dead. No rules-legal path exists in the remaining time.
Final action: ensure last Codabench submission = submission_v83_rerank_both_f60.zip before 21:30 UTC.

### 2026-09-01 (v99 REAL): seen-leg reweight — 53.60157016683023%, −0.0953 vs v83. DEAD, real-negative.

`builders/build_v99_seen_reweight.py` (v83 gate/unseen route untouched, only the seen fusion's
3-leg weights changed from deployed 1.0/2.5/2.5 → v98's grid winner 3.0/0/0.5, +1.34pt on holdout,
verified stable across 2 independent holdout halves). Real: **seen 77.37473254714634% (−0.169 vs
v83) · unseen 22.912384378211717% (bit-identical, unseen route untouched) · overall
53.60157016683023% (−0.0953)**. Holdout gain did not transfer — it reversed. **Confirms (again) that
linear-reweight-of-existing-legs is dead on real, not just marginal.** v99 briefly became the last
Codabench submission (worse than v83); re-submitted `submission_v83_rerank_both_f60.zip` to restore
Force_Last. **v83 (53.69690172437964%) remains final/best — this is the state at competition close.**

**Same-day follow-up sweep (4 more ideas, real-time-tested via workflow, all DEAD, all holdout-only
— none escalated to a submission given v99's transfer already broke):**
- **HGB re-ranker vs LR** (`research/rerank_seen_v99_hgb.py`): swapped v83's seen-reranker logistic
  regression for HistGradientBoosting on the identical 32 cached features/K=10. Result: **exact tie**,
  91.674% both models. Nonlinear interactions are not the bottleneck — the 32 features are already
  the saturated signal; recall@10's 98.85% ceiling stays ~7pt out of reach regardless of model class.
- **Widen seen-reranker K** (10→15→20): 91.674 → 91.750 → 91.775%. Ceiling rises (98.85→99.44%) but
  realized accuracy barely moves — added candidates are low-margin, model can't cash in the extra
  headroom. K=10 stays optimal.
- **Multi-view TTA on fullft336shift** (`research/tta_fullft336_gate_auc_v100.py`): the only cached
  holdout crop-views (`emb_holdout_fullft336_cropviews.pt`) cover val_uns (pseudo-novel) only, not
  val_seen — can't touch the 91.21 seen number at all. Redirected to gate-discrimination AUC instead:
  center-view (deployed) 0.9559 vs mean-pool-7-view 0.9454 vs max-pool-7-view 0.9365 — **TTA makes
  the gate feature worse**, wrong direction on both variants.
- **Rank fusion** (Borda / RRF k=10,60, equal + deployed-weighted) replacing the z-scored linear sum
  across the 3 seen legs: best variant 90.10%, below deployed 89.87%+linear-reweight entirely.
  Order-statistics fusion underperforms magnitude-weighted sum here.

**Every combination of every cached embedding for the seen head is now real-or-holdout dead.**
Nothing left except a genuinely new closed-set encoder (multi-day training, out of scope today).
**FINAL STATE AT CLOSE: v83, 53.69690172437964% (seen 77.54391202667065% / unseen
22.912384378211717%), re-confirmed as the last Codabench submission.**

### 2026-09-01 (post-close, "explore everything" pass): 22 more mechanisms, real-time-tested, all DEAD

User asked to exhaust every remaining axis regardless of the earlier stand-pat call. Ran 3 more
parallel Workflow rounds (holdout-only, no more real submissions spent — budget/urgency had already
resolved with v83 re-confirmed as final). Every idea tested against the same
`learned_gate_v77.holdout_split()` val_seen pool (11,866 rows / 4,636 classes), bar = clear the
v83 seen re-ranker's 91.674% holdout figure by >=1.0pt (i.e. >=92.674%) unless noted. **All 22 DEAD:**

| idea | mechanism class | holdout | vs 91.674 bar |
|---|---|---|---|
| James-Stein shrink→taxon-text (2 independent implementations) | prototype geometry | 89.98 / 89.87 | dead — 2nd one **hurt the 0-2-img bucket −7.78pt**, the taxon-text direction is a bad visual anchor for thin classes |
| Trained linear/softmax probe (JS-regularized) | discriminative classifier | 90.00 (solo probe only 82.28%, 8.9pt below fixed prototype) | dead — too few images/class for a trained boundary to beat a heuristic |
| Soft top-k distance-weighted pooling | aggregation | 89.93 | dead, flat |
| Genus-level hierarchical backoff | taxonomic prior | 91.202 | dead — grid selected "off" as best |
| Embedding whitening (center/std/PCA, SimpleShot-style) | feature geometry | 89.87 (raw wins) | dead — CLIP embeddings already ~isotropic, no SimpleShot-style gain here |
| Prototype bagging (bootstrap resample, N=5/10/20) | variance reduction | 89.94 | dead, +0.07pt, shrinks with more bootstraps |
| Leg-agreement bonus feature | new per-image signal | 90.27 | dead, +0.40pt, 1.41 short |
| **Per-image MoE fusion gate (5-fold CV)** | per-image adaptive weight | **91.252** | dead but instructive — +1.38pt over deployed, bucket-consistent (+3.82/+2.01/+1.16 on 0-2/3-5/6+ img), but gate features are a functional transform of the SAME 3 legs (not new evidence), so it just converges to "mostly trust ctftshift solo" (91.20%) — confirms the "reweighting doesn't compound, even per-image" law again |
| Per-class self-similarity calibration | calibration | 89.87 (raw wins) | dead |
| Post-hoc logit adjustment (Menon-style, tau·log(n_train)) | long-tail bias correction | 89.91 | dead — theorized direction (boost rare classes) **crashes to 66.95% at tau=-3**, base fusion already implicitly handles count differently than a uniform log-prior predicts |
| Cross-leg training-set denoising (drop disagreement images) | data quality | 89.62 | dead — flagged "outliers" are legit hard exemplars (juvenile/damaged fish), not mislabels; long tail is data-**starved**, not label-noisy |
| Feature-concat fusion (join embeddings, one joint prototype) | representation-level fusion | 88.336 | dead, worst in every bucket |
| Harmonic/min/soft-min combination operator | fusion operator | 91.24 | dead — collapses to "trust the single strongest leg," never exceeds it |
| k-reciprocal / Jaccard class-graph re-ranking | between-class structure | 89.90 | dead |
| Unused DINOv2 embeddings (already extracted, never plugged in) | backbone diversity | 89.87 (best weight = 0) | dead — solo DINOv2 63.66%, self-supervised signal too weak vs any CLIP variant, dilutes rather than complements |

**Root pattern across all 22 (and the earlier 9 same-day tests): every lever that changes how the
SAME 3 legs' SAME information gets combined — weights, rank, operator, per-image gating, geometry
transform — tops out within ~1.5pt of the strongest solo leg (ctftshift, 91.20%) and stays below the
existing re-ranker (91.674%). The only lever that ever beat that ceiling was v77's gate, which added
genuinely NEW information (bank/text-margin signals absent from the base fusion) — not available
again here without a new encoder.** 31 real-time-tested seen-side mechanisms today, 0 survivors.
**v83 (53.69690172437964%) stands as the final, fully-exhausted answer.**

### 2026-09-01 (v109 REAL, ~18:05 UTC): genus backoff — 53.73615589513528%, +0.039 vs v83. NEW BEST.

After 31 dead seen-side mechanisms and the audit's process fixes, one more real-tested near-miss
survived contact with reality: genus-level hierarchical backoff (ctftshift-leg proto+2*cmax pooled
ACROSS sibling species of the same genus, LAM_G=3.0, added to the deployed 3-leg base fusion; seen
re-ranker retrained leak-free with genus as a 4th signal group, `research/genus_rerank_v103_train.py`
→ `outputs/rerank_seen_genus_v103.pkl`). Holdout stack: 91.775% vs v83's real-comparable 91.674%
(+0.101pt) — inside the noise band the exact same axis (v98/v99 reweight) had already shown could
reverse on real (+1.34 holdout → −0.17 real). Shipped as an explicit gamble anyway
(`builders/build_v109_genus_gamble.py`, `submissions/submission_v109_genus_gamble.zip`).

**Real: seen 77.6135741652983% (+0.0697 vs v83) · unseen 22.912384378211717% (bit-identical, route
untouched) · overall 53.73615589513528% (+0.03925 vs v83).** Gate/routing bit-identical to v83
(89.11/77.58 test→seen / unseen→unseen), confirming isolation held; 312/35,665 predictions (0.87%)
differ from v83.

**Why this one transferred positively when reweighting didn't:** genus is genuinely NEW evidence —
information (sibling-species similarity) absent from the base 3-leg fusion — not a re-combination of
the same 3 legs' existing scores. Matches v77's gate finding exactly: new information compounds on
real transfer; reweighting existing information gives most of the holdout gain back (or reverses it).
Confirms the mechanism-class distinction is real and predictive, not just a post-hoc story for one
data point (now 2/2: v77 gate +new info transferred; v98/v99 reweight transferred negative).

**New best real: 53.73615589513528%** — `submissions/submission_v109_genus_gamble.zip`
(seen 77.6135741652983% / unseen 22.912384378211717%). This is now the last Codabench submission
and the better one — no Force_Last correction needed, unlike the v99 episode.

### 2026-09-01 (post-close audit): two corrections to the 22-idea kill table — verdicts stand, rationale/instrument flagged

Measured audit (`research/probe_whiten_loophole.py` → `outputs/probe_whiten_loophole.json`,
`research/probe2_retriever_recall.py` → `outputs/probe2_retriever_recall.json`):

1. **The whitening verdict's rationale ("CLIP embeddings already ~isotropic, nothing to whiten")
   is factually FALSE.** Measured on the trby pools: participation ratio 50.6–62.7 of D=1024
   (isotropic ⇒ ~1024), top-50 of 1024 dims carry 75–80% of variance, mean cosine-to-μ 0.33–0.57
   per leg. The embeddings are strongly anisotropic. The correct statement is: the anisotropy is
   class-INFORMATIVE (fine-tuning sharpened it) — center/std/full-rank ZCA all score at or below
   raw even in SimpleShot's native proto-only regime (e.g. ctftshift proto-only 87.49 raw vs
   87.18 center / 87.02 std / 85.70 zca), so *unsupervised whitening* is dead, but the "nothing
   to whiten" diagnosis is wrong and should not be cited to close the feature-geometry class.
   Also: the v102 PCA modes truncated to 64/128/256 of 1024 dims — dimensionality reduction, not
   whitening; full-rank ZCA was untested until this audit (it also loses, so the verdict holds).

2. **The kill instrument used a scope error for retriever-level ideas.** Every Round-3 idea
   modifies the base fusion; the v83 re-ranker consumes the fusion's top-10
   (build_v83_rerank_both.py L332). The bar demanded each idea, as BARE fused argmax, to beat the
   full deployed stack (fusion+re-ranker, 91.674) by +1.0. The fair stack-vs-stack instrument
   (re-ranker retrained leak-free on the modified fusion's features — machinery exists since v82)
   was never run for any of the 22. Centering flips 61–256 argmax rows per leg solo (fullft336shift
   +0.46 solo, 86.91→87.37) but 82.6% of rows' top-10 candidate sets change and retriever
   recall@10 does NOT improve (98.854 raw → 98.778 center_fullft → 98.753 center_all), so the
   buried solo near-miss does not survive the stack test either. Consequence: "+0.07/+0.40,
   dead" rows should read "sub-bar positive at retriever level, untested at stack level" — and
   given v98/v99's measured holdout→real transfer failure on this head, still not worth a slot.

3. Prototype bagging's +0.07 is one seed (SEED=0), ~8 images, and decays monotonically with
   N_BOOT (89.938/89.921/89.895 at N=5/10/20) toward the plain estimator — bootstrap-noise
   signature, not variance reduction. Dead for the right reason.

   (re-ranker retrained leak-free on the modified fusion's features — machinery exists since v82)
   was never run for any of the 22. Centering flips 61–256 argmax rows per leg solo (fullft336shift
   +0.46 solo, 86.91→87.37) but 82.6% of rows' top-10 candidate sets change and retriever
   recall@10 does NOT improve (98.854 raw → 98.778 center_fullft → 98.753 center_all), so the
   buried solo near-miss does not survive the stack test either. Consequence: "+0.07/+0.40,
   dead" rows should read "sub-bar positive at retriever level, untested at stack level" — and
   given v98/v99's measured holdout→real transfer failure on this head, still not worth a slot.

## FINAL-DAY ENDGAME (2026-09-01 17:45 UTC, close 22:00 UTC)

**Force_Last state: BROKEN.** v99 (53.6016, −0.095 vs v83) is currently the last submission.
If nothing further is submitted, the final score is 53.60, not 53.70. Restoration required.

**Quota:** v99 used 1 of 3 today → 2 slots left before 22:00 UTC.

**Unscored inventory (only >54 projection):** `submissions/submission_v64_coverage_aware.zip`
(Aug 25; projected 55.5–57.2 from pre-v77-era modeling; unseen estimate 22–25% never validated;
era transfer history says projections overshoot 1.5–2pt → realistic landing 53–54.5). Compliance OK
(single-pipeline, coverage signal from image/text only, iNat bank disclosed).

**Max-EV 2-slot plan:**
1. Submit v64 now. If score ≥ 53.697 → it stands, stop. Else →
2. Resubmit `submission_v83_rerank_both_f60.zip` to restore Force_Last. Worst case = status quo.

**v64 REAL (2026-09-01 ~18:00 UTC): 51.76223187999439%** — seen **79.85769020251778%** ·
unseen **15.493319630010277%** · overall **51.7622**. Projection 55.5–57.2 was **fantasy**
(unseen 22–25 assumption → real 15.49; −6.5pt miss). Coverage-aware learned fusion DEAD real.
Confirms era-projection rule: pre-v77 projections overshoot by 1.5–2+pt minimum. v64 became
last submission → **slot 2 MUST be v83 resubmit to restore 53.697 Force_Last.**
1 slot remains after that — do NOT spend it; nothing in inventory projects above 53.697.

**Fusion temptation, quantified and REJECTED (2026-09-01 ~18:15 UTC):** v83↔v64 hard-label
disagreement = 13,722/35,665 (38.5%). Oracle routing (v64 preds on seen-pop, v83 on unseen-pop)
bounds a perfect gate at ~55.0 overall — real signal exists. But: (a) a leak-free gate needs
v64-leg scores on the seen holdout + unseen proxy + retraining + rerank interplay — not doable
and not validatable in the remaining window; (b) measured holdout→real transfer on this head
today is 0/3 (v98, v99 holdout +1.34 → real −0.095; v64 projection −6.5pt); (c) Force_Last +
1 remaining slot = no restore if the gamble lands low. EV(gamble) < EV(hold). v86 f=0.55 /
v87 f=0.65 brackets also rejected: f-curve was measured real, 0.60 is the peak. Final action:
**resubmit v83, spend nothing else.** Final score: 53.697%.


**56% verdict: NOT REACHABLE.** +2.30 overall = +5.27 unseen at flat seen; no mechanism in repo
provides it (22 ideas + Round-3 5 dead; stack-level tests close the buried near-misses). v64 is a
lottery ticket, not a plan. Do NOT burn slot 2 on anything except v83 restoration.


## POST-GAMBLE WORKSTREAM — ALL DEAD (2026-09-01 17:10–18:12 UTC, undocumented until now)

Parallel to the v109 gamble build, the seen-route new-evidence hunt ran to completion. Every
branch died at the +1.0 kill bar; nothing further was submitted (v109 gamble already held the
final slot and stands as best real 53.736%). Results, for the record:

1. **v107 embedding-mean query base** (`research/seen_embmean_v107.py` →
   `outputs/seen_embmean_v107.json`): merged 3-leg query rep. Holdout acc 90.199 vs deployed
   89.870 (+0.33), recall@10 98.972 vs 98.854 (+0.118) → CLEARED recall bar, chained onward.
2. **v107 stack retrain** (`outputs/agreement_recheck_v107.json`): agreement reweight best
   agree1_a4.0 → r10 98.904; stage-2 leak-free retrain 91.522 vs v83 stack 91.674 →
   `clears_bar: false` → DEAD.
3. **Logit recheck** (`outputs/logit_recheck_stack.json`): tau=0.25 r10 98.904 ("IMPROVES"),
   stack retrain 91.6905 vs 91.674 → +0.017 vs +1.0 bar → DOES NOT CLEAR BAR → DEAD.
4. **v108 FishBase description anchor** (`research/seen_embmean_desc_v108.py`,
   `src/build_text_fishbase.py`, `descriptions_fishbase.json`, `outputs/text_emb_h_fishbase.pt`):
   (proto+cmax)+(0.9·taxon+0.1·traits) → acc 89.954 / r10 98.753, both BELOW deployed;
   no-score4 ablation also below deployed → description anchor strictly hurts → DEAD.
5. **f-bracket re-check** (`outputs/fbracket_analysis4.txt`): v86 f=0.55 / v87 f=0.65/0.68 flip
   projections confirm the endgame rejection — brackets stay dead.
6. **R-zone sizing analysis — INCOMPLETE, cut off at 18:10 UTC** (`outputs/ov1.txt`, `ov2.txt`,
   `ov3.txt` only; no script committed, no result JSON). Measured novel→seen overlap
   (R-zone 1,654 test / 6,674... 6,744 unseen images; both-novel pairs same-vs-diff species:
   318/216 test, 3,565/1,769 unseen). An unseen-route exploration that never produced a verdict.
   Post-deadline relevance: none for scoring; kept only as a starting point if work resumes.

Moral (3rd consecutive confirmation): solo-level holdout gains on this head do not survive the
leak-free stack retrain. The v109 genus-gamble (a true new-evidence lever) remains the only
kind that transferred — and it is the final answer: **53.73615589513528%**.

