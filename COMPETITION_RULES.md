# Fish Species Recognition Challenge — Official Rules (cached)

**Source of truth (live):** [Codabench competition 16815](https://www.codabench.org/competitions/16815/)  
**Workshop page:** [cv4e-workshop.github.io/competition.html](https://cv4e-workshop.github.io/competition.html)  
**Contact:** cv4e.workshop@gmail.com  

Cached from Codabench API `/api/competitions/16815/` on **2026-07-23**.  
If the live page disagrees with this file, **the live page wins** — re-fetch and update this doc.

---

## 1. Task & scoring

- Classify each eval image to a species among 17,393 classes (seen + unseen).
- Test set mixes **seen** (in training) and **unseen** (no training images; text descriptions provided).
- Metric: **overall classification accuracy**. Codabench also reports seen / unseen accuracy.
- Rankings / prizes use **overall** accuracy.
- **Most Creative Solution Award** (workshop page, re-fetched 2026-09-07): *"a Most Creative Solution
  Award will recognize a novel or innovative approach regardless of final rank."* Judged on novelty /
  innovation; rank explicitly excluded. No technical-report rubric is published.
- Submission: `prediction.json` mapping filename → species name, zipped as `submission.zip`.

**Timeline:** 1 June 2026 → 1 September 2026 · Results 9 September 2026.

**Hard close — CONFIRMED 2026-08-29 by two sources (API + Codabench UI, agreeing exactly):**
Testing Phase ends **`2026-09-01T22:00:00Z`** = 1 Sep 22:00 UTC = **2 Sep 07:00 KST**
= 1 Sep 15:00 PDT / 18:00 EDT. Organizer: faizan.

**This is NOT AoE — do not plan as if it were.** AoE (UTC−12) would be 2026-09-02 11:59 UTC,
**14 h later** than the real cut-off; assuming AoE forfeits the final submission.

Only open question: whether the 3/day quota resets at 00:00 UTC (assumed) or rolls 24 h.
If UTC: 4 windows remain (29 Aug partial, 30, 31, 1 Sep) → ≤12 submissions before close.

**Quota:** **3 submissions per day · 30 total.** Budget carefully (HANDOFF §7).

---

## 2. HARD PROHIBITIONS (never break)

### 2.1 Folder-oracle / split leakage
Participants **must not** assume whether an eval image is seen or unseen at inference.

- `splits/test.pkl` and `splits/unseen.pkl` are **protocol-only**.
- Do **not** use them to identify membership of an eval image.
- Do **not** restrict predictions to a subset of candidate classes based on those splits.
- **Allowed:** one uniform rule over all eval images; one argmax over the **full** 17,393-class space.
- **Allowed:** a novelty/gate signal computed from the image/text itself (no split labels).

This is why HANDOFF §6 single-pipeline compliance exists. Folder-oracle routing is **disallowed**.

### 2.2 Fish-specific / challenge-domain models & data
**Must not** use models, datasets, or resources **specifically designed for fish species recognition** or the challenge target categories. Prohibited examples:

- Models explicitly trained for fish species recognition
- Models fine-tuned on fish-specific datasets
- Models trained on species-specific challenge-domain benchmarks
- Domain-specific expert systems for the target categories
- Private datasets containing challenge test images or substantially overlapping content

**Do not use** (examples already flagged in-repo):

- `CaesiumSG/sg-fish-bioclip-v2` — anonymous fish-finetune, empty README, possible unseen leakage
- `ReefNet/finetuned-bioclip` — reef/fish-finetuned
- Any other fish-only fine-tune whose training set may overlap competition species

When unsure → email organizers **before** using it.

### 2.3 Fair play
- No obtaining test labels via inspection, reverse-engineering the eval server, or other means.
- No leaderboard manipulation / infrastructure abuse.
- No human label fishing on eval images.

---

## 3. EXPLICITLY ALLOWED

### 3.1 Foundation models
Public models/frameworks are allowed, including (non-exhaustive):

- CLIP / OpenCLIP / SigLIP / DINO and variants
- **BioCLIP and its variants** (e.g. BioCLIP-2, BioCLIP-2.5 ViT-H — what we run)
- General-purpose LLMs and VLMs
- Other publicly available foundation models

You may train, fine-tune, prompt, retrieve from, or otherwise leverage them **if** §2.2 is respected.

**Same category as BioCLIP (allowed):** general biology dual-encoders (BioTrove, BioCAP, bioclip-hc) — biology-pretrained, **not** fish-only.

### 3.2 External data
> “The use of external data is permitted unless explicitly stated otherwise.”

- Must **fully disclose** all external datasets, models, and resources in the final report / transparency statement.
- Undisclosed external resources may → **disqualification**.
- Examples relevant to §8 item 2: FishBase / WoRMS / GBIF morphological text, taxonomy names (already used), richer species descriptions.

Provided `descriptions.json` is competition data (always OK). External *additional* text is OK with disclosure.

### 3.3 Training on provided data
Train on labelled training images + use provided class descriptions for all seen/unseen classes.

---

## 4. GRAY ZONES (interpret carefully)

### 4.1 Transduction / Sinkhorn (v33)
Rules **forbid folder-oracle routing** (§2.1). They **do not mention** batch-coupled / transductive inference (e.g. Sinkhorn over the eval batch).

| Build | Score | Status |
|-------|-------|--------|
| **v31 strict** | 47.69% | Fully per-image; safest compliance |
| **v33 Sinkhorn** | 47.78% | Transductive; scored on leaderboard; **not explicitly blessed** |

If organizers challenge transduction → fall back to v31. Do not add more aggressive transduction without a rules check / organizer email.

### 4.2 Leaderboard-fitted hyperparameters
`seen_frac=0.72` was tuned on Codabench feedback. Rules do not ban this, but if hyperparameter fitting on the leaderboard is questioned, recalibrate on the train-derived holdout.

---

## 5. Transparency & prizes

Leaderboard rank alone ≠ prize. For prizes / verification you may need:

1. Technical report  
2. Complete list of training datasets + external resources  
3. Transparency statement (models, external data, fine-tuning, retrieval/knowledge sources, compute, human intervention)  
4. Inference code sufficient to reproduce  
5. Reproduction docs  

Winning teams may be asked to release code/checkpoints. Keep a disclosure log of every external resource used.

---

## 6. AGENT CHECKLIST (before any experiment or submission)

- [ ] No use of `splits/*.pkl` for membership or candidate restriction at inference
- [ ] Predictions always over the **full** class space with one uniform rule
- [ ] No fish-specific fine-tuned third-party models
- [ ] External text/data → OK, but **log for disclosure**
- [ ] Prefer v31 if transduction is disputed; v33 only with eyes open
- [ ] Submission budget: ≤3/day, ≤30 total — don't burn on noise
- [ ] When model eligibility is unclear → stop and ask organizers (`cv4e.workshop@gmail.com`)

---

## 7. Re-fetch command

```bash
curl -sL 'https://www.codabench.org/api/competitions/16815/' \
  | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['terms'])"
```
