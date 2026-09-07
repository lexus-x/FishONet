#!/usr/bin/env python
"""Figures for the 3-page, two-column FishONet technical paper.

Column geometry matches paper.html exactly: single column 84.5mm, full
measure 176mm. Every value is traced to HANDOFF.md, REPORT.md, or a direct
read of data/dl/. Nothing here is estimated.
"""
import os
import json
import pickle
import collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "paper_assets")
FONTS = os.path.join(HERE, "design_assets", "fonts")
os.makedirs(OUT, exist_ok=True)
for _f in os.listdir(FONTS):
    if _f.endswith(".ttf"):
        font_manager.fontManager.addfont(os.path.join(FONTS, _f))

INK   = "#191919"
MUTED = "#6E6E6E"
FAINT = "#A6A6A6"
RULE  = "#DCDCDC"
TEAL  = "#1F6F6B"
TEALL = "#7FB3AD"
CLAY  = "#C05621"
CLAYL = "#E8B48F"
FLAG  = "#8F2D2D"

SANS = "Inter"
MONO = "IBM Plex Mono"

COL_W = 3.33    # in — one column (84.5 mm)
FULL_W = 6.93   # in — full measure (176 mm)

plt.rcParams.update({
    "font.family": SANS, "font.size": 6.2, "text.color": INK,
    "axes.labelcolor": MUTED, "axes.edgecolor": RULE, "axes.linewidth": 0.6,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelsize": 5.6, "ytick.labelsize": 5.6, "axes.labelsize": 5.9,
    "figure.facecolor": "none", "axes.facecolor": "none",
    "savefig.facecolor": "none",
})


def save(fig, name, dpi=440):
    fig.savefig(f"{OUT}/{name}.png", dpi=dpi, bbox_inches="tight",
                transparent=True, pad_inches=0.015)
    plt.close(fig)
    print("  ", name)


def bare(ax, keep=()):
    for k, s in ax.spines.items():
        s.set_visible(k in keep)


# ===================================================== 1 / the task
def fig_task(name="task"):
    """Fine-grained + long-tailed + open-set, read off the actual label file."""
    lab = json.load(open(os.path.join(ROOT, "data/dl/label_train.json")))
    allc = pickle.load(open(os.path.join(ROOT, "data/dl/all_classes.pkl"), "rb"))
    cnt = sorted(collections.Counter(lab.values()).values(), reverse=True)
    n_photo, n_all = len(cnt), len(allc)
    n_zero = n_all - n_photo

    fig, ax = plt.subplots(figsize=(COL_W, 1.45))
    x = np.arange(1, n_photo + 1)
    ax.fill_between(x, 0.55, cnt, color=TEALL, lw=0, alpha=0.55, zorder=2)
    ax.plot(x, cnt, color=TEAL, lw=0.8, zorder=3)
    # the classes with no photograph anywhere: a floor, not a curve
    BAR_LO, BAR_HI = 0.55, 1.02
    ax.fill_between([n_photo, n_all], BAR_LO, BAR_HI, color=CLAY, lw=0, zorder=2)

    ax.set_yscale("log")
    ax.set_ylim(0.55, 620)
    ax.set_xlim(-140, n_all + 140)
    ax.set_yticks([1, 10, 100])
    ax.set_yticklabels(["1", "10", "100"])
    ax.set_xticks([0, 5795, 11000, 17393])
    ax.set_xticklabels(["0", "5,795", "", "17,393"], family=MONO, fontsize=5.4)
    for gy in (1, 10, 100):
        ax.axhline(gy, color=RULE, lw=0.45, zorder=0)
    bare(ax)
    ax.tick_params(which="both", length=0, pad=2)
    ax.set_ylabel("training photographs", fontsize=5.9)
    ax.set_xlabel("species, ranked by how often they were photographed", fontsize=5.9)

    ax.text(150, 430, "289 species (5%) hold 69% of all 64,259 images",
            fontsize=5.5, color=TEAL, va="center")
    ax.annotate("median 2", xy=(2897, 2), xytext=(3400, 22), fontsize=5.5,
                color=MUTED, arrowprops=dict(arrowstyle="-", lw=0.5, color=FAINT,
                                             connectionstyle="arc3,rad=-0.25"))
    ax.text(n_photo + (n_zero / 2), 1.35, f"{n_zero:,} species · no photograph exists",
            ha="center", fontsize=5.6, color=CLAY)
    # centred on the bar's geometric midpoint -- the axis is log, so the arithmetic
    # midpoint sits high and clipped the label against the bar's top edge
    ax.text(n_photo + (n_zero / 2), (BAR_LO * BAR_HI) ** 0.5,
            "text description and taxonomy only",
            ha="center", va="center", fontsize=5.2, color="#FFFFFF", zorder=4)
    save(fig, name)


# ===================================================== 2 / pipeline
def fig_pipeline(name="pipeline"):
    """Two stacked route lanes under one shared decision chain.

    Every string is measured against the real renderer and checked against
    its lane's right edge — the previous horizontal layout guessed at widths
    and the route text ran straight through the merge arrows.
    """
    W, H = 100.0, 44.0
    fig, ax = plt.subplots(figsize=(FULL_W, 1.92))
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    fig.canvas.draw()
    overflow = []

    def T(x, y, s, size=5.4, color=MUTED, right=None, **kw):
        t = ax.text(x, y, s, fontsize=size, color=color, **kw)
        bb = t.get_window_extent(fig.canvas.get_renderer())
        x0, x1 = ax.transData.inverted().transform(
            [(bb.x0, bb.y0), (bb.x1, bb.y1)])[:, 0]
        if right is not None and x1 > right:
            overflow.append((s, round(x1 - right, 2)))
        return x1

    def rule(x0, x1, y, c=RULE, lw=0.6):
        ax.plot([x0, x1], [y, y], color=c, lw=lw, solid_capstyle="butt")

    def arrow(x0, x1, y, c=FAINT):
        ax.annotate("", xy=(x1, y), xytext=(x0, y),
                    arrowprops=dict(arrowstyle="-|>,head_width=0.09,head_length=0.24",
                                    lw=0.6, color=c, shrinkA=0, shrinkB=0))

    # ---------------- shared chain, one line across the top.
    # Fixed anchors, not chained offsets: each stage owns a lane and its
    # sub-line is checked against the next stage's anchor.
    yc = 39.5
    for x, lim, head, sub in (
            (0, 19, "eval image", "35,665 images · no split label"),
            (21, 55, "5 encoders", "3× BioCLIP-2.5 ViT-H · BioCLIP-2 ViT-L · TaxaBind"),
            (57, 100, "quota gate", "12 learned multimodal features · top f = 0.60")):
        e = T(x, yc, head, 6.4, INK, weight="600")
        T(x, yc - 3.3, sub, 5.1, FLAG if x == 0 else MUTED, right=lim)
        if lim < 100:
            arrow(e + 1.6, lim, yc + 0.8)

    # ---------------- fork
    ax.plot([46, 46], [34.2, 31.6], color=FAINT, lw=0.6)
    ax.plot([8, 84], [31.6, 31.6], color=FAINT, lw=0.6)
    ax.plot([8, 8], [31.6, 29.4], color=TEALL, lw=0.9)
    ax.plot([84, 84], [31.6, 29.4], color=CLAYL, lw=0.9)

    # ---------------- the two lanes
    lanes = [
        (0, 48, TEAL, TEALL, "seen route", "5,795 classes",
         ["class prototypes + nearest exemplar",
          "+ taxonomic text  (1.0 / 2.5 / 2.5)",
          "32-feature leak-free re-rank, K = 10",
          "genus-level hierarchical backoff"]),
        (52, 100, CLAY, CLAYL, "novel route", "11,598 classes",
         ["6 text legs + 2 iNaturalist photo banks",
          "+ 2 TreeOfLife prototypes, debiased",
          "34-feature leak-free re-rank, K = 20",
          "no training photograph is available"]),
    ]
    for x0, x1, c, cl, head, count, lines in lanes:
        T(x0, 26.4, head, 6.2, c, weight="600")
        T(x1, 26.4, count, 5.3, FAINT, ha="right", family=MONO)
        rule(x0, x1, 24.6, cl, 0.8)
        for i, s in enumerate(lines):
            T(x0, 20.9 - i * 3.5, s, 5.3, MUTED, right=x1)

    # ---------------- merge
    ax.plot([8, 8], [5.4, 3.4], color=TEALL, lw=0.9)
    ax.plot([84, 84], [5.4, 3.4], color=CLAYL, lw=0.9)
    ax.plot([8, 84], [3.4, 3.4], color=FAINT, lw=0.6)
    ax.plot([46, 46], [3.4, 1.6], color=FAINT, lw=0.6)
    T(46, 0.0, "one argmax over all 17,393 classes", 6.2, INK, ha="center",
      weight="600")

    if overflow:
        print("     ! overflow:", overflow)
    save(fig, name)


# ===================================================== 3 / campaign
LABELS = ["v22", "v31", "v33", "v36", "v46", "v50", "v56",
          "v77", "v79", "v81", "v82", "v83", "v109"]
NUM = [22, 31, 33, 36, 46, 50, 56, 77, 79, 81, 82, 83, 109]
OVERALL = [45.39, 47.69, 47.78, 49.04, 50.84, 51.44, 51.62,
           53.383, 53.425, 53.431, 53.644, 53.697, 53.736]


def fig_campaign(name="campaign"):
    fig, (top, ax) = plt.subplots(
        2, 1, figsize=(COL_W, 1.42), height_ratios=[1, 6.4],
        gridspec_kw=dict(hspace=0.30))

    # -- effort strip: every numbered build, the scored ones marked
    top.set_xlim(0, 110)
    top.set_ylim(0, 1)
    top.axis("off")
    scored = set(NUM)
    for b in range(1, 110):
        c, lw = (INK, 0.85) if b in scored else (RULE, 0.6)
        top.plot([b, b], [0.06, 0.62], color=c, lw=lw, solid_capstyle="butt")
    top.text(0, 0.80, "109 numbered builds", fontsize=5.6, color=MUTED)
    top.text(110, 0.80, "13 measured on the leaderboard", fontsize=5.6,
             color=INK, ha="right")

    # -- the climb through the ones that were actually scored
    x = np.arange(len(LABELS))
    ax.set_xlim(-0.4, len(LABELS) - 0.6)
    ax.set_ylim(44.4, 54.9)
    ax.fill_between(x, 44.4, OVERALL, color=TEALL, alpha=0.22, lw=0)
    ax.plot(x, OVERALL, color=INK, lw=1.0, zorder=4, solid_capstyle="round")
    ax.scatter(x, OVERALL, s=7, color="#FFFFFF", edgecolors=INK,
               linewidths=0.7, zorder=5)
    ax.scatter([x[-1]], [OVERALL[-1]], s=22, color=TEAL, edgecolors="none",
               zorder=6)

    ax.axhline(50.56, color=FAINT, lw=0.6, ls=(0, (2.5, 2.5)), zorder=2)
    ax.text(len(LABELS) - 1.1, 50.76, "public leader at the time  50.56",
            fontsize=5.3, color=MUTED, ha="right")
    ax.text(len(LABELS) - 1, 54.35, "53.736", ha="right", fontsize=9.5,
            color=TEAL, family=MONO, weight="600")
    ax.text(0, 44.85, "45.39", fontsize=6.0, color=MUTED, family=MONO)
    ax.annotate("learned gate\nreplaces the heuristic", xy=(7, 53.383),
                xytext=(4.9, 54.25), fontsize=5.3, color=TEAL, ha="center",
                arrowprops=dict(arrowstyle="-", lw=0.5, color=TEALL,
                                connectionstyle="arc3,rad=0.2"))
    bare(ax)
    ax.set_yticks([46, 48, 50, 52, 54])
    for gy in (46, 48, 50, 52, 54):
        ax.axhline(gy, color=RULE, lw=0.45, zorder=0)
    ax.tick_params(length=0, pad=2)
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, fontsize=5.2, family=MONO, color=FAINT)
    ax.set_ylabel("overall accuracy  %", fontsize=5.9)
    save(fig, name)


# ===================================================== 4 / promise vs paid
# Every change for which BOTH a holdout delta and a real leaderboard delta were
# recorded. (label, holdout Δ overall-pt, real Δ overall-pt, HANDOFF.md line)
PAIRS = [
    ("v34 recover-on-eject",   3.51,  -0.07,  189),
    ("v37 iNat protos",        9.66,   1.08,  290),
    ("v37 w2",                 1.12,   0.20,  291),
    ("v39 TaxaBind img",       0.52,  -0.11,  300),
    ("v40 max-pool",           1.68,   0.03,  294),
    ("v41 B2 proto",           0.99,   0.08,  297),
    ("v43 dual B2",            0.518,  0.186, 298),
    ("v44 denser ToL",         0.431,  0.017, 752),
    ("v46 LoRA∩ToL",           0.345,  0.067, 763),
    ("f = 0.72",               5.09,  -0.60,  789),
    ("v77 learned gate",       0.523,  1.766, 1093),   # holdout: learned_gate_v77.pkl delta
    ("v79 C refit",            0.558,  0.042, 1094),   # holdout: v79.pkl − v77.pkl delta
    ("v81 leaky re-ranker",   13.374,  0.006, 1215),
    ("v82 leak-free",          8.154,  0.213, 1371),
    ("v83 seen re-ranker",     1.803,  0.053, 1372),
    ("v84 K = 50",             0.345, -0.039, 1679),
    ("v99 seen reweight",      1.34,  -0.095, 1892),
    ("v109 genus backoff",     0.101,  0.039, 1965),
]


def fig_promise(name="promise"):
    fig, ax = plt.subplots(figsize=(FULL_W, 2.00))
    LT = 0.1
    ax.set_xscale("symlog", linthresh=LT, linscale=0.6)
    ax.set_yscale("symlog", linthresh=LT, linscale=0.6)
    ax.set_xlim(0.06, 24)
    ax.set_ylim(-1.05, 2.7)

    # the honest line, and the slope the leak-free re-rankers actually paid
    xs = np.logspace(np.log10(0.06), np.log10(24), 200)
    ax.plot(xs, xs, color=FAINT, lw=0.7, ls=(0, (3, 2.5)), zorder=1)
    ax.text(2.25, 2.45, "if the holdout were honest", fontsize=5.4, color=FAINT,
            ha="right", va="center")
    ax.fill_between(xs, 0.026 * xs, 0.030 * xs, color=TEALL, alpha=0.35, lw=0, zorder=1)
    ax.text(23.2, 0.36, "leak-free re-rankers paid 0.026–0.030 per point",
            fontsize=5.0, color=TEAL, ha="right", va="center")
    ax.axhline(0, color=INK, lw=0.6, zorder=2)

    over = under = flip = 0
    ratios = []
    for lab, h, r, _ in PAIRS:
        if r < 0:
            c, s, z = FLAG, 15, 5
            flip += 1
        elif r > h:
            c, s, z = TEAL, 30, 6
            under += 1
        else:
            c, s, z = MUTED, 12, 4
            over += 1
            ratios.append(h / r)
        ax.scatter([h], [r], s=s, color=c, edgecolors="#FFFFFF", linewidths=0.5, zorder=z)

    A = dict(fontsize=5.3, color=INK, zorder=7)
    L = dict(arrowstyle="-", lw=0.45, color=FAINT, shrinkA=0, shrinkB=2)
    ax.annotate("v81  +13.374 promised, +0.006 paid", xy=(13.374, 0.006), xytext=(2.3, 0.115),
                ha="left", va="bottom", arrowprops=L, **A)
    ax.annotate("v82  same idea, leak-free pool", xy=(8.154, 0.213), xytext=(0.95, 0.42),
                ha="left", va="bottom", arrowprops=L, **A)
    ax.annotate("v77  the learned gate — the only\nchange that paid more than it promised",
                xy=(0.523, 1.766), xytext=(0.075, 1.35), ha="left", va="top",
                color=TEAL, fontsize=5.3, zorder=7, linespacing=1.35, arrowprops=L)
    ax.annotate("v37  iNat photo bank", xy=(9.66, 1.08), xytext=(9.2, 1.62), ha="right",
                va="bottom", arrowprops=L, **A)
    ax.annotate("f = 0.72  holdout preferred it by 5 points;\nthe leaderboard docked 0.60",
                xy=(5.09, -0.60), xytext=(0.36, -0.30), ha="left", va="top",
                color=FLAG, fontsize=5.3, zorder=7, linespacing=1.35, arrowprops=L)
    ax.annotate("v34", xy=(3.51, -0.07), xytext=(3.51, -0.24), ha="center", va="top",
                color=FLAG, fontsize=5.0, zorder=7)
    ax.text(0.0, 1.10, "18 changes measured both ways", fontsize=6.2, color=INK,
            weight="600", va="bottom", transform=ax.transAxes)
    ax.text(0.0, 1.02, f"{over} paid less than promised · {flip} flipped sign · "
            f"{under} paid more", fontsize=5.4, color=MUTED, va="bottom",
            transform=ax.transAxes)

    bare(ax, keep=())
    ax.set_xticks([0.1, 0.3, 1, 3, 10])
    ax.set_xticklabels(["0.1", "0.3", "1", "3", "10"], family=MONO, fontsize=5.4)
    ax.set_yticks([-0.5, 0, 0.5, 1, 2])
    ax.set_yticklabels(["−0.5", "0", "0.5", "1", "2"], family=MONO, fontsize=5.4)
    ax.tick_params(which="both", length=0, pad=2)
    for gy in (-0.5, 0.5, 1, 2):
        ax.axhline(gy, color=RULE, lw=0.4, zorder=0)
    ax.set_xlabel("what the holdout promised   (Δ overall accuracy, points)", fontsize=5.9)
    ax.set_ylabel("what the leaderboard paid", fontsize=5.9)
    print(f"     promise: over={over} flip={flip} under={under} "
          f"median over-statement={np.median(ratios):.1f}×")
    save(fig, name)


if __name__ == "__main__":
    print("paper figures:")
    fig_task()
    fig_pipeline()
    fig_campaign()
    fig_promise()
