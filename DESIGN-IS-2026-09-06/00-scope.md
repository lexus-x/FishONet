# Scope — FishONet public site design audit

**Audited surface:** `index.html` (repo root) + `assets/css/site.css` + `assets/js/site.js`,
identical to `github/index.html` (gh-pages mirror). Live at:
- https://sai-fishonet.pages.dev/ (primary)
- Cloudflare mirror: https://fihonet.lalithsai00.workers.dev/

Excluded from this audit (separate artifacts, not audited):
- `htmls/index.html` — a distinct internal technical-report dashboard, different design system
- `REPORT.md`, `HANDOFF.md` — prose docs, not a UI surface

**Primary user:** CV4Ecology 2026 competition reviewers/organizers and ML researchers evaluating
the FishONet open-set fish recognition submission (Codabench 16815). Secondary: peers/recruiters
skimming the linked demo from the README badge.

**Primary task:** Understand, in under a minute of scrolling, what the system does (open-set
species ID across 17,393 fish, 2/3 unseen), what its real score is (53.736%, v109), and how the
pipeline works — enough to trust the number and decide whether to read the full report.

**Constraints:**
- Static site, no backend, no build step — plain HTML/CSS/JS
- Three.js (r128, CDN) drives a hero canvas
- Google Fonts, `color-scheme: dark` declared
- Deployed to Cloudflare Pages + GitHub Pages mirror
- Content is fixed (real measured numbers) — copy must not overclaim beyond `HANDOFF.md`/`REPORT.md`

**Reference designs:** none named by user. Implicit competitive set: research-paper landing pages
(Distill.pub-style scrollytelling), other CV4Ecology/Codabench submission pages.

**Scope inference note:** `/design-is` was invoked with no arguments. Root `index.html` was
inferred as the target — it's the only live, user-facing design artifact in an otherwise
ML-research repo, it's the one badge-linked from `README.md` as the "Interactive Demo", and it is
identical to the gh-pages-published copy. Proceeding on this inference per auto-mode guidance;
flag if a different surface was intended.
