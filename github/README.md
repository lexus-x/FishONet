# FishONet — naming the unseen

Open-set fish species recognition across **17,393 species**, of which **11,598 have no training
photograph at all** — only a scientific name and a written description. Submitted to the
CV4Ecology 2026 challenge (Codabench 16815).

**Final score: 53.69690172437964% overall** — 77.544% on seen species, 22.912% on novel ones.

🌐 **[Live Interactive Showcase → https://fihonet.lalithsai00.workers.dev/](https://fihonet.lalithsai00.workers.dev/)**  
📖 **[Technical Report & Methodology → REPORT.md](https://github.com/lexus-x/FihOnet/blob/main/REPORT.md)**

---

## What this repository is

The public project page: a visual walkthrough of the problem, the pipeline, the results, and the
negative results. It is a static site — no build step, no dependencies, no tracking.

Everything on the page uses real data:

| On the page | Where it comes from |
|---|---|
| Fish photographs | Competition training set, 99,979 images |
| Species names & descriptions | `descriptions.json`, 35,133 species |
| Which photo represents a species | Chosen by the model — the image closest to its BioCLIP class prototype |
| Head scores in the walkthrough | Real cosine similarities from cached embeddings |
| Accuracy figures | Measured leaderboard results |

Projections are labelled as projections. Nothing is invented.

---

## The problem in one paragraph

Give a model a photograph and ask it to name the species from 17,393 candidates. For 5,795 of them
you have labelled photographs. For 11,598 you have nothing but text. The evaluation set mixes both
and never says which is which — and assuming you know is explicitly prohibited. So every image
forces a blind decision (*is this something I have seen?*) before naming is even possible, and a
wrong decision makes the correct answer unreachable.

## How the system works

```
              ┌─ seen head ──── 5,795 classes · image prototypes ─┐
image ─ 6 encoders ─┤                                             ├─ argmax ─ species
              └─ novel head ── 11,598 classes · taxonomy text ────┘
                        ▲
                   learned gate — 12 features, routes the top 60%
```

- **Encoders** — BioCLIP-2.5 ViT-H/14 (three fine-tuned variants), BioCLIP-2 ViT-L/14 (frozen +
  LoRA), TaxaBind ViT-B/16. All fine-tuning used competition training images only.
- **Seen head** — class prototypes + nearest-training-image + taxonomy-text anchor, ensembled over
  three encoders.
- **Novel head** — ten fused legs: taxonomy text across four encoders, two iNaturalist photo-bank
  legs, two BioCLIP-2 prototype legs.
- **Gate** — 12-feature logistic model. Routing is a *quota*, not a threshold: the top 60% of images
  by novelty score go to the seen head, so only the ordering has any effect.
- **Re-rankers** — both hand-weighted sums are demoted to retrievers; a learned per-candidate model
  re-orders the shortlist.

## Results

| Build | Overall | What changed |
|---|---|---|
| v33 | 47.780 | Sinkhorn baseline |
| v37 | 50.320 | iNaturalist image prototypes |
| v56 | 51.616 | Crop-max over 7 views |
| v77 | 53.383 | **12-feature learned gate** (+1.767) |
| v82 | 53.644 | **Leak-free re-ranker validation** (+0.213) |
| **v83** | **53.697** | **Seen re-ranker — final** |
| v84 | 53.658 | Shortlist depth 20→50 — regression |

Everything after v56 was won without a new encoder or dataset.

## The finding that mattered most

A re-ranker scored **+13.374** on our validation set and **+0.006** on the leaderboard.

The validation candidate pool mixed 1,159 classes that could be the answer with 11,598 that never
could. That 9.1% subgroup is detectable from the features, so the model learned *"prefer the
eligible-looking candidates"* instead of ranking evidence. **Class-disjoint cross-validation cannot
detect this** — the shortcut lives in the candidate pool, not the class split.

Restricting the pool so every candidate is eligible made the validation number **worse**
(+13.374 → +8.154) and the real score **better** (+0.006 → +0.213).

Measuring both sides gave a conversion rate — **0.026–0.030 real points per validation point**,
stable across two independent heads — which correctly predicted the final submission's score in
advance (53.69 projected, 53.6969 measured).

> The magnitude of a validation number is not evidence. The population it was measured on is.

## Layout

```
.
├── index.html              # the project page
├── assets/
│   ├── css/site.css        # single stylesheet
│   ├── js/site.js          # no libraries
│   ├── data/
│   │   ├── species.json    # names, descriptions, image paths
│   │   └── demo.json       # real per-head top-5 for the walkthrough
│   └── img/
│       ├── species/        # 18 prototype-selected species photos
│       └── mosaic/         # 72 thumbnails
└── README.md
```

## Running it locally

The page fetches JSON, so it needs to be served over HTTP rather than opened as a file:

```bash
cd fishonet && python -m http.server 8000
# then open http://localhost:8000
```

## Publishing to GitHub Pages

1. Push this folder to a repository.
2. **Settings → Pages → Source: Deploy from a branch**, select `main` and `/ (root)`.
3. Update the page URL at the top of this README.

## Authors & Laboratory Attribution

- **Author:** Lalith Sai ([lexus-x](https://github.com/lexus-x))
- **Laboratory:** Intelligent Systems Laboratory ([ISLab](http://islab.cwnu.ac.kr/)), Changwon National University (CWNU)
- **Supervision:** [Prof. Cheng Yaw Low](https://chengyawlow.github.io/)

## License & Intellectual Property

Copyright (c) 2026 Lalith Sai, Intelligent Systems Laboratory (ISLab, CWNU), and Prof. Cheng Yaw Low. All Rights Reserved. **This work is proprietary and not open source.**

Unauthorized copying, distribution, modification, commercial use, or reproduction of this codebase, architecture, or research materials is strictly prohibited without explicit written permission.

Third-party biodiversity metadata and reference images from iNaturalist, TreeOfLife-200M, and competition materials remain the property of their respective creators and licensors.
