# Disclosure log — Fish Species Recognition Challenge (Codabench 16815)

Required by competition External Data Policy and Transparency Statement.
Update this file whenever a new model, dataset, or knowledge source is used.

## Foundation models

| resource | role | status |
|---|---|---|
| `imageomics/bioclip-2.5-vith14` (BioCLIP-2.5 ViT-H/14) | primary image/text backbone; LoRA + full-FT on **provided** train images | in use (v31/v33) |
| `imageomics/bioclip-2` (BioCLIP-2 ViT-L/14) | frozen + LoRA dual iNat proto legs (v41/v43); denser merge with ToL embeddings (v44/v46) | **in use** — deployed text-fuse alone dead (+0.43); **iNat mean-proto + dual validated real**; ToL denser on both legs = **v46 best** |
| `MVRL/taxabind-vit-b-16` | frozen unseen-route fusion leg (v35/v36) | proxy +1.86 alone; **+2.11 stacked with ctftshift** |
| `imageomics/bioclip` (BioCLIP-1) | backbone fuse probe | deployed +1.86 but **no stack** with TaxaBind |
| `BGLab/BioTrove-CLIP` (+ M ViT-L ckpt) | backbone ranking | **dead** (~0%) |
| `imageomics/biocap` | backbone ranking | **dead** (6.8–12.3) |
| `imageomics/bioclip-hc-euclidean` | backbone ranking | **dead** (4.4) |
| `Qwen/Qwen2.5-3B-Instruct` | offline trait rewrite of provided descs | used; traits dead |
| `Qwen/Qwen2.5-VL-7B-Instruct` | uniform top-K rerank (proxy) | used; **dead (−16pt)** |
| `deepseek-v4-flash-vision-exp` (DeepSeek API) | uniform top-K rerank (proxy), reasoning-enabled retry of the Qwen probe | used; **dead (−11.33pt)** |

## Competition data (always OK)

| resource | role |
|---|---|
| `data/dl/images/` + `label_train.json` | seen-class training |
| `data/dl/descriptions.json` | class text (seen+unseen); source for trait compression |
| `data/dl/all_classes.pkl` | label space |

## External data / knowledge (disclose)

| resource | role | status |
|---|---|---|
| GBIF API (`api.gbif.org`) | taxonomy lineage + vernacular common names | already used (`outputs/taxonomy_full.json`, `common_names.json`) |
| Wikimedia Commons MediaSearch (`commons.wikimedia.org`) | CC-licensed species photos for **uncovered** unseen-route classes (v37+ coverage pass) | **in use** — multi-query binomial + fish keyword search |
| Openverse API (`api.openverse.org`) | CC-licensed web images by scientific name | **in use** — coverage URL pass |
| Encyclopedia of Life (`eol.org` API) | supplemental species page images | **in use** — coverage URL pass (sparse) |
| iNaturalist API (`api.inaturalist.org`, all quality grades) | photos for classes missing research-grade S3 coverage | **in use** — second-pass coverage only (general biodiversity platform; disclosed) |
| Wikipedia REST thumbnails (`en.wikipedia.org/api/rest_v1`) | species lead images where available | **in use** — coverage URL pass |
| GBIF occurrence StillImage media | external image prototypes for zero-shot classes | **killed 2026-07-23** — proxy 6.99 vs taxon 23.38; coverage 1621/12757 cand |
| iNaturalist research photos (same S3 join as v37) embedded with **`MVRL/taxabind-vit-b-16`** (`research/embed_inat_taxabind.py`) → `outputs/inat_protos_taxabind.pt` | orthogonal image-prototype leg on unseen route (v39); eval queries use existing `emb_{test,unseen}_taxabind.pt` | **in use (v39)** — stacked @ TB_IMG_W∈{0.5,0.75,1.0} with v37 ctftshift iNat leg; proxy +0.52 @0.5; real untested |
| iNaturalist Open Data (AWS S3 `inaturalist-open-data`, + `api.inaturalist.org`) `quality_grade=research` observation photos | **dense** external image prototypes for unseen classes (v37); joined taxa→observations→photos from the public S3 dumps (API rate-limited so pivoted to S3), photos used only as per-class visual prototypes | **in use (v37)** — unseen-route coverage 6011/11598 (51.8%); **honest** proxy (distractors also covered) 31.45→43.53 (+12.08), class-disjoint CV 37.84→52.52 (+14.68); **0% near-dup leakage** (max cosine 0.945; same-class 0.78 vs cross 0.44). Not fish-specific (general biodiversity platform). Licenses: CC0/CC-BY/CC-BY-NC (open data). |
| `imageomics/TreeOfLife-200M-Embeddings` (`bioclip-2_float16`, CC0 embeddings; no images redistributed) | Pre-computed BioCLIP-2 image embeddings for TreeOfLife-200M; filtered to competition scientific names → mean protos (`outputs/tol_protos_bioclip2.pt`); **denser merge** with iNat B2 protos on intersection (α=0.5) for v44 frozen leg and v46 frozen+LoRA legs | **validated real** — v44 **50.77246600308426%** (~0.039/pt); **v46 best 50.83975886723678%** (~0.195/pt). Raw gap-fill/additive DEAD. General biology corpus (not fish-specific). |
| FishBase morphological descriptions (`reports/descriptions_fishbase.json`, fetched 2026-09-01) | richer per-class trait text, embedded via `src/build_text_fishbase.py` for comparability with the taxon text tower | **fetched and embedded 2026-09-01** (superseding the 2026-07-23 "not fetched" entry below) but **never wired into any builder** — grep of `builders/` finds no reference; **not part of the v109 (or any scored) submission**. Disclosed per COMPETITION_RULES.md §3.2 out of caution even though unused. |
| FishBase / WoRMS morphological fields (original probe) | richer traits if provided-desc rewrite wins on proxy | **killed 2026-07-23** — traits blend +0.17pt (noise); not fetched at that time (see corrected fetch above) |
| Wikipedia REST summaries (`en.wikipedia.org`) | species abstract text for cand classes | **killed 2026-07-23** — coverage 35/12757; best +0.09 |
| `Qwen/Qwen2.5-VL-7B-Instruct` | uniform top-K species rerank | **killed 2026-07-23** — −16pt on n=300 K=10 |
| `deepseek-v4-flash-vision-exp` (DeepSeek API, general-purpose VLM) | uniform top-K species rerank, same recipe as Qwen probe but with a 900-token reasoning budget | **killed 2026-09-01** — −11.33pt on n=300 K=10 (better than Qwen's −16 but still net-negative) |

## Not used (forbidden or not cleared)

- Fish-specific third-party recognition models (CaesiumSG, ReefNet, …)
- Synthetic images of competition species for fine-tuning (not cleared with organizers)
- Folder-oracle / `splits/*.pkl` membership at inference

## Inference notes

### Single-pipeline interpretation (Codabench 16815)

**Forbidden:** using `splits/test.pkl` / `splits/unseen.pkl` to know whether an *eval image*
belongs to the seen or unseen split, or to shrink candidates from those files (folder-oracle).

**Allowed (uniform rule on every image):**
- Soft novelty gate from image/text scores (no split labels) → route to train-seen class pool
  vs complement. Train-derived “which classes have images” is OK; eval-split membership is not.
- Model-derived top-K shortlists (from scores), then VLM/LLM pick among them.
- External data (text or images) with disclosure; same recipe for all eval images.
- Final prediction = one species from the competition 17,393-class list.

v33 Sinkhorn is transductive (gray zone); v31 is the strict per-image fallback.
VLM rerank / external image prototypes: log below when used.

## Change log

| date | change |
|---|---|
| 2026-07-23 | Created. Documented BioCLIP + GBIF. |
| 2026-07-23 | Ran `Qwen/Qwen2.5-3B-Instruct` trait rewrite of provided `descriptions.json` (`outputs/traits_from_desc.json`). Proxy: traits 17.08 vs taxon 23.38; best blend +0.17 — **do not promote**. FishBase/WoRMS not fetched. |
| 2026-07-23 | VLM top-K **oracle ceiling** measured (`research/unseen_vlm_ceiling.py`): deployed top-20 oracle +42.5pt vs top-1. |
| 2026-07-23 | VLM rerank run (`research/unseen_vlm_rerank.py` n=300 K=10): text 26.33 → VLM 10.33 (−16) — **KILL**. |
| 2026-07-23 | External GBIF images (`research/fetch_external_images.py` + `unseen_ext_image_proxy.py`): 1621/12757 cand covered; alone 6.99 vs taxon 23.38 — **KILL**. |
| 2026-07-23 | **No Codabench submission** from loophole A/B experiments. Best remains v33 47.78%. |
| 2026-07-23 | `MVRL/taxabind-vit-b-16` frozen @ taxctx fused into deployed unseen recipe: **29.38 → 31.23 (+1.86pt)** at w=1.0. Built `submissions/submission_v35_taxabind_w10_sink72.zip` (10.4% preds differ vs v33; unseen-class coverage 57.5%→59.2%). Real score untested — do not upload unless projecting past ~50.5. |
| 2026-07-23 | BioCLIP-1 deployed fuse +1.86 (ties TaxaBind) but does not stack. bioclip-inat 0.35 dead. Wikipedia summaries coverage 35/12k — dead. CSLS/free lifts +0.47 — dead. |
| 2026-07-23 | **ctft_shift** (`src/contrastive_ft.py --shift_aug 1`, squash_tta extract): alone 31.10 vs ctftbig 26.62; replace into deployed **+2.07**; **+ TaxaBind → 33.35 (+3.97 / +2.11 on TB)**. Built `submissions/submission_v36_ctftshift_tb_w10_sink72.zip`. Real untested; eyes-open ~49% not 53%. |
| 2026-07-23 (late) | **`tb_ctftshift`** = LiT-style shift-aug LoRA fine-tune of `MVRL/taxabind-vit-b-16` image tower on **provided training images only** (`research/contrastive_ft_taxabind.py`). Standalone pseudo-unseen db 10.01→11.65; **fused +0.22 vs frozen TaxaBind < kill bar → DEAD, not used in any submission.** Logged for completeness. |
| 2026-07-23 (coverage) | Expanded **uncovered-class** URL discovery (`research/fetch_coverage.py`, `fetch_coverage_extra.py`): GBIF multi-key/synonym/variant match, Wikimedia multi-query, Openverse, EOL, iNat non-research API, Wikipedia thumbs. Writes `outputs/coverage_urls_extra.json` then `merge_coverage_urls.py`; downloads to `outputs/coverage_images_extra/`. External prototypes only; not fish-specific models. |
| 2026-07-23 (v37) | **iNaturalist research-grade image prototypes.** Species→photo join built from the **iNaturalist Open Data S3 dumps** (`taxa/observations/photos.csv.gz`, `research/inat_opendata.py`) after the live API (`research/fetch_inat_images.py`) rate-limited/blocked us hard. Name match 12738/12757 (99.9%); research-grade obs for **6684** cand classes; **6011/11598 (51.8%)** unseen-route coverage; 73,322 medium-res photos downloaded from S3 (no rate limit). Embedded with the v36 `ctftshift` encoder (squash-TTA) → per-class visual prototypes (`research/embed_inat.py`); added as a `dbnorm` image leg on the **unseen route only** (single pipeline; gate + seen route unchanged). **Honest proxy** (`research/inat_proto_proxy.py`, distractors also covered): text stack 31.45 → +iNat 43.53 (**+12.08**); class-disjoint CV covered-gold 37.84→52.52 (**+14.68**). (An earlier gold-only proxy read +31.75 was inflated by only gold classes having protos.) **Leakage audit** (`research/inat_dup_probe.py`): 0% of train queries match any iNat photo at cosine>0.95 (max 0.945; same-class mean 0.78 vs cross-class 0.44) — genuinely different photographs, no eval/train image reuse; iNat photo IDs (9-digit) disjoint from competition IDs (≤999783). Builder: `builders/build_v37_inat_proto.py` (IMG_W∈{1.0,1.5,2.0} staged). Real Codabench score **untested**. |

| 2026-07-24 (v38) | **GBIF + Wikimedia Commons coverage extension** for previously uncovered unseen-route classes (`research/fetch_coverage.py`, `research/download_coverage_images.py`). GBIF occurrence API (`api.gbif.org`, StillImage media from multiple publishers). Wikimedia Commons API (`commons.wikimedia.org`, CC-licensed species photos). **520/5587** uncovered classes returned URLs; images downloaded to `outputs/coverage_images/`. Embedded with same `ctftshift` encoder as iNat protos (`research/embed_inat.py`); merged with `outputs/inat_protos_ctftshift_full.pt` via weighted mean (`research/merge_coverage_protos.py` → `outputs/protos_ctftshift_merged.pt`). Same single-pipeline rule: image leg on unseen route only; no folder-oracle. Builder: `builders/build_v38_coverage.py`. |

| 2026-07-24 (v39) | **TaxaBind iNat image prototypes.** Same iNat research-grade photo set as v37, embedded with frozen **`MVRL/taxabind-vit-b-16`** (`research/embed_inat_taxabind.py` → `outputs/inat_protos_taxabind.pt`). Fused on the **unseen route only** as `TB_IMG_W * dbnorm(Q_taxabind @ protos)` alongside the v37 ctftshift iNat leg (`builders/build_v39_inat_taxabind_proto.py`). Honest proxy +0.52 vs v37 @ IMG_W=3; CV +0.31. Real Codabench **untested**; projected ~50.52% overall. |

| 2026-07-24 (ortho iNat protos) | **SigLIP2** (`timm/ViT-SO400M-16-SigLIP2-384` via open_clip) and **DINOv2** (`dinov2_vitl14` via torch.hub) per-class iNat prototypes on the same 73,322 research-grade iNat photos as v37 (`research/embed_inat_encoder.py` → `outputs/inat_protos_siglip2.pt`, `outputs/inat_protos_dinov2.pt`). Proxy ablation only (`research/inat_ortho_stack_proxy.py`); **not deployed** (kill <+0.3 vs v37 ref 43.53). General vision encoders, not fish-specific. |
| 2026-07-24 (denser iNat URLs) | Supplementary iNaturalist API fetch (`quality_grade=research` then `needs_id`) for up to 300 previously low-URL **uncovered** unseen-route classes; **44** classes received expanded URL lists in `outputs/inat_image_urls.json` before HTTP 429 stop. General biodiversity platform; no eval-image matching. Photos not yet re-downloaded/embedded. |

| 2026-07-24 (v38b URL expansion) | Enhanced uncovered-class URL fetch (research/fetch_coverage.py: GBIF synonyms/variants, multi-query Wikimedia, Openverse, EOL). coverage_urls_extra.json merged via merge_coverage_urls.py --prune: 1286/5124 (25.1%) URL hits (was 57); 1193 classes / 4583 images in coverage_images/; ctftshift embed; protos_merged_incr_only.pt (+446 classes) promoted to protos_ctftshift_merged.pt. Honest proxy S0+3ximg: iNat-only 43.53 vs incr-merge 43.31 (full merge 43.18) — no clear +EV vs v37-w3 real 50.46%; no submission_v38b zip. |


| 2026-07-24 (S3 needs_id densify) | Second pass on **iNaturalist Open Data S3** (`research/inat_opendata.py --stage obs_needs_id`, quality_grade=needs_id, obs cap 80) + merged photo URL stage (`--photo-cap 24`, boost low-coverage classes to 30 URLs). Complements API densify (`fetch_inat_denser.py` low_photo / needs_id_resume) without iNat API rate limits. Merges into `outputs/inat_image_urls.json`; download via `download_inat_only.py`. General biodiversity platform; prototypes only. |
| 2026-07-24 (max-pool iNat proxy) | Research ablation: per-class **max cosine** (and top-m mean) over iNat photo embeddings vs mean prototype on pseudo-unseen proxy (`research/inat_maxpool_proxy.py`). Same ctftshift encoder + squash-TTA as v37; no eval-image matching. Submission only if honest proxy clears +0.3 vs mean-proto 43.53 with conservative real EV. |
| 2026-07-24 (denser embed) | Re-downloaded +44 iNat classes from prior needs_id URL enrich; **73,465** photos / **7008** classes. Honest proxy w3 **43.49** (−0.04 vs 43.53) — no submission. Ongoing: API densify (`research/fetch_inat_denser.py`), S3 photo-cap 30 (`inat_opendata.py`). |

| 2026-07-25 (BioCLIP iNat contrastive FT) | **Contrastive LoRA fine-tune** of BioCLIP-2.5 ViT-H/14 on disclosed **iNaturalist** research-grade + pseudo-labeled fish taxa overlapping competition unseen names (`research/contrastive_ft_inat.py`, init from `outputs/ctft_shift.pt`, 97k pairs / 10,691 classes, val db **25.75**). Extracted train/test/unseen with squash-TTA → `emb_*_ctftshift_inat.pt`; rebuilt iNat mean protos + photo bank with this encoder. **Honest stack proxy** (`research/inat_ft_stack_proxy.py`): vs baseline ctftshift ref mean@w3 **43.49** → FT **42.58 (−0.91)**; maxpool top4@w3.5 **45.17 → 44.87 (−0.30)**. **Not used in any submission** (proxy kill; projected real ~50.38–50.45% at v37 transfer). |
| 2026-07-25 (BioCLIP-2 iNat protos) | Frozen **`imageomics/bioclip-2`** (ViT-L/14, general biology CLIP — not fish-specific) embedded 117,225 iNat research-grade photos → `outputs/inat_protos_bioclip2.pt` (7364 classes). Train-query embeddings `outputs/emb_train_bioclip2.pt`. Deployed in **v41** submission (+0.08 real overall vs v40); holdout +0.99 mean-track @ B2_W=2.5. |
| 2026-07-25 (denser iNat +696) | 117225 iNat photos / 7364 classes re-download+re-embed; maxpool proxy 45.13 (-0.04 vs 45.17); not submitted. |

| 2026-07-25 (v41 BioCLIP-2 proto) | **`imageomics/bioclip-2`** (BioCLIP 2 ViT-L/14 via open_clip; **not** fish-specific). HF hub has **no** `bioclip-2.5-vitb16` / `bioclip-2.5-vitl14` repos (only `imageomics/bioclip-2.5-vith14` + `bioclip-2`). Per-class **mean** iNat prototypes on the same 117,225 disclosed iNat photos (`research/embed_inat_encoder.py` → `outputs/inat_protos_bioclip2.pt`). Eval queries: frozen BioCLIP-2 on competition test+unseen (`outputs/emb_{test,unseen}_bioclip2.pt`). Fused on **unseen route only** alongside v40 maxpool leg (`builders/build_v41_bioclip2_proto.py`, B2_W=2.5, IMG_W=4). Honest pseudo-unseen proxy vs v40 ref **45.30**: **+0.99** @ w=2.5 (`outputs/inat_bioclip2_v40_ortho.json`); frozen text-leg fusion **+0.09** (kill fail). **Real Codabench validated 50.57%** overall (seen 78.95% / unseen 13.93%) — current best submission. |

| 2026-07-28 (biotrove iNat proto) | Same iNat research-grade photo bank as v37, embedded with **`BGLab/BioTrove-CLIP`** (`research/embed_inat_encoder.py` → `outputs/inat_protos_biotrove.pt`; train queries `outputs/emb_train_biotrove.pt`). Honest v41-stack proxy only (`research/inat_biotrove_v41_proxy.py`); **dead** (best −0.86 vs v41+B2 @ BT_W=0.25). **Not submitted.** |
| 2026-07-30 (v44/v46 ToL denser) | `imageomics/TreeOfLife-200M-Embeddings` (CC0 pre-computed BioCLIP-2 embeddings; **no images redistributed**) filtered to competition scientific names, merged with iNat BioCLIP-2 protos on the intersection at α=0.5. Validated real: v44 **50.772%**, v46 **50.840%**. Already listed in the resource table above. General biology corpus, not fish-specific. |
| 2026-08-18 (v50–v56 crops + 336 bank) | iNaturalist photo bank re-embedded under the `ftshift` and `fullft336shift` encoders (`outputs/inat_photo_bank_{ftshift,fullft336shift}.pt`) and multi-crop query views (center / squash / 5 overlapping long-axis strips). **Same disclosed iNat photo set as v37** — no new external source. v56 validated real **51.616%**. |
| **2026-08-29 (v77–v83 learned components)** | **No new external models, datasets, or knowledge sources.** The learned gate (12 features), the leak-free unseen re-ranker (34 features) and the seen re-ranker (32 features) are logistic models trained **only** on train-derived holdouts, over evidence the pipeline already computed from resources disclosed above. Best real **53.69690172437964%** (v83). |
| 2026-08-31 (audit) | Full disclosure audit for the final report. Confirmed **no external resource was introduced after 2026-07-30 (v46)**; every subsequent gain (v50→v83) is a learned component or crop/TTA change over already-disclosed data. Re-ranker model-class ablation (`sklearn` HistGradientBoosting vs LogisticRegression) added and closed — no new data. Technical report written to `REPORT.md`. |
| 2026-09-01 | Retried VLM top-K rerank with `deepseek-v4-flash-vision-exp` (general-purpose VLM, free API tier, no fish-specific training) — same n=300 K=10 harness as the 2026-07-23 Qwen probe, but with a 900-token reasoning budget instead of an 8-token forced guess. text 26.33 → VLM 15.00 (**−11.33**) — **KILL**, confirms the Qwen result was not a model-capacity artifact. `research/unseen_vlm_rerank_deepseek.py`. |
| 2026-09-01 | Extending `fullft336shift` full-fine-tune training 4→8 epochs (`src/fullft336.py`, same `--shift_aug 1` recipe as the original run) — original run stopped at epoch 4 while val NCM was still rising (+1.54pt on the last epoch, no plateau). No new external resource; same provided training images, same recipe, just more epochs. In progress. |
| 2026-09-01 | Fetched FishBase morphological descriptions (`reports/descriptions_fishbase.json`) and embedded them (`src/build_text_fishbase.py`) — this **postdates and is not covered by** the 2026-08-31 audit line above, which only certified "no new external resource after v46" as of that date. Confirmed **not deployed** in any builder (`build_v109_genus_gamble.py` included) — kept here purely for §3.2 disclosure completeness. |
| 2026-09-05 (repo audit) | Full pre-report repo audit: stripped 3 plaintext API tokens (GitHub PAT, HuggingFace, Context7) from `.mcp.json` and the git remote URL; moved `archive/` (6.3GB legacy/pre-v50 code+zips), `media/`, the unrelated `fishonet-novelty-gate/` side-repo, and 42 off-lineage submission zips out of the working tree to `~/onet_precleanup_backup_20260905/` (nothing deleted outright except generic third-party Claude-plugin skill defs under `.agents/`, which carried no research content); fixed two stale builder paths in `CLAUDE.md` broken by an earlier reorg; corrected this file's FishBase row (above). Full findings in `AUDIT_REPORT.md`. |
