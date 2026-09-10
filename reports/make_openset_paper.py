#!/usr/bin/env python
"""Build the FishONet open-set recognition paper as a two-column CVPR-style PDF.

Every number in this document is traced to HANDOFF.md, outputs/shift_diag.json,
or a direct reading of the builder / training source. Nothing is estimated.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (BaseDocTemplate, Frame, PageTemplate, Paragraph,
                                Spacer, Table, TableStyle, Image, FrameBreak,
                                KeepTogether, NextPageTemplate)

FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fishonet_figs")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fishonet_openset_paper.pdf")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "font.size": 7.6, "axes.labelsize": 7.8, "xtick.labelsize": 7.2,
    "ytick.labelsize": 7.2, "legend.fontsize": 7.0, "axes.titlesize": 8.2,
})
INK, MID, PALE = "#111111", "#5a5a5a", "#c9c9c9"
BLU, RED, GRN, ORG = "#1f4e79", "#a4292f", "#2b6b4f", "#b86a12"


def _clean(ax, ytitle=None):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=2.8, pad=2.0)
    if ytitle:
        ax.set_ylabel(ytitle)


def _nospine(ax):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)


# =========================================================== Fig 1: pipeline
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(3.30, 3.05))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10.2); ax.axis("off")

    def box(x, y, w, h, title, sub=None, fc="white", ec=INK, tc=INK, bold=True, fs=7.4):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.12",
                                    fc=fc, ec=ec, lw=1.0, zorder=3))
        ax.text(x + w / 2, y + h * (0.62 if sub else 0.5), title, ha="center", va="center",
                fontsize=fs, weight="bold" if bold else "normal", color=tc, zorder=4)
        if sub:
            ax.text(x + w / 2, y + h * 0.26, sub, ha="center", va="center",
                    fontsize=6.3, color=MID, zorder=4)

    def arr(x0, y0, x1, y1, c=INK):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                     mutation_scale=8, lw=1.0, color=c, zorder=5))

    box(2.7, 9.15, 4.6, 0.85, "evaluation image", "35,665 total")
    arr(5.0, 9.15, 5.0, 8.62)
    box(0.9, 7.65, 8.2, 0.95, "routing gate", "rank by  z(image) + 2 z(text margin);  keep top f = 0.60")
    arr(3.1, 7.65, 2.2, 7.05, BLU)
    arr(6.9, 7.65, 7.8, 7.05, ORG)
    ax.text(2.55, 7.34, "21,399", fontsize=6.3, color=BLU, ha="right")
    ax.text(7.45, 7.34, "14,266", fontsize=6.3, color=ORG, ha="left")

    ax.add_patch(FancyBboxPatch((0.05, 3.55), 4.55, 3.45,
                                boxstyle="round,pad=0.05,rounding_size=0.12",
                                fc="#eef3f8", ec=BLU, lw=1.0, zorder=2))
    ax.text(2.33, 6.62, "CLOSED-SET ROUTE", ha="center", fontsize=7.0, weight="bold", color=BLU)
    for i, s in enumerate(["BioCLIP-2.5 ViT-H/14, 3 members",
                           "prototype + best exemplar",
                           "+ taxonomic text anchor"]):
        ax.text(2.33, 6.14 - i * 0.44, s, ha="center", fontsize=6.4, color=MID)
    ax.text(2.33, 4.52, "argmax over", ha="center", fontsize=6.6, color=INK)
    ax.text(2.33, 4.06, "5,795", ha="center", fontsize=10.5, weight="bold", color=BLU)
    ax.text(2.33, 3.74, "trained classes", ha="center", fontsize=6.4, color=MID)

    ax.add_patch(FancyBboxPatch((5.40, 3.55), 4.55, 3.45,
                                boxstyle="round,pad=0.05,rounding_size=0.12",
                                fc="#fdf3e8", ec=ORG, lw=1.0, zorder=2))
    ax.text(7.67, 6.62, "RETRIEVAL ROUTE", ha="center", fontsize=7.0, weight="bold", color=ORG)
    for i, s in enumerate(["6 text legs incl. TaxaBind",
                           "external photo banks, top-4",
                           "BioCLIP-2 prototypes"]):
        ax.text(7.67, 6.14 - i * 0.44, s, ha="center", fontsize=6.4, color=MID)
    ax.text(7.67, 4.52, "argmax over", ha="center", fontsize=6.6, color=INK)
    ax.text(7.67, 4.06, "11,598", ha="center", fontsize=10.5, weight="bold", color=ORG)
    ax.text(7.67, 3.74, "novel classes", ha="center", fontsize=6.4, color=MID)

    arr(2.33, 3.55, 3.55, 2.95, BLU)
    arr(7.67, 3.55, 6.45, 2.95, ORG)
    box(2.3, 1.95, 5.4, 0.95, "prediction",
        "disjoint candidate sets  |  union = 17,393")
    ax.text(5.0, 1.32, "no single 17,393-way decision is ever taken",
            ha="center", fontsize=6.5, style="italic", color=RED)
    ax.text(5.0, 0.52, "batch-coupled at five points: rank threshold, dbnorm, Sinkhorn, two standardizations",
            ha="center", fontsize=6.2, color=MID)
    fig.tight_layout(pad=0.12)
    fig.savefig(f"{FIG}/pipeline.png", dpi=430, bbox_inches="tight")
    plt.close(fig)


# ====================================================== Fig 2: routing flow
NT, NU = 20097, 15568
KT, ET = 17426, 2671          # trained: kept on closed-set route / ejected
EU, KU = 11597, 3971          # novel: ejected to retrieval route / kept


def _ribbon(ax, x0, x1, t0, b0, t1, b1, color, alpha):
    t = np.linspace(0, 1, 120)
    sm = t * t * (3 - 2 * t)
    ax.fill_between(x0 + (x1 - x0) * t, b0 + (b1 - b0) * sm, t0 + (t1 - t0) * sm,
                    color=color, alpha=alpha, lw=0, zorder=2)


def fig_flow():
    fig, ax = plt.subplots(figsize=(3.30, 2.30))
    ax.set_xlim(-0.02, 1.30); ax.set_ylim(-0.10, 1.16); ax.axis("off")
    G = 0.055
    hT, hU = NT / 35665, NU / 35665
    Lt1, Lt0 = 1.0, 1.0 - hT                    # trained block  (left)
    Lu1, Lu0 = Lt0 - G, Lt0 - G - hU            # novel block    (left)
    hS, hR = (KT + KU) / 35665, (ET + EU) / 35665
    Rs1, Rs0 = 1.0, 1.0 - hS                    # closed-set route (right)
    Rr1, Rr0 = Rs0 - G, Rs0 - G - hR            # retrieval route  (right)
    xa, xb = 0.20, 0.80

    for x, y0, y1, c, lab, n in [(0.0, Lt0, Lt1, BLU, "true class\ntrained", NT),
                                 (0.0, Lu0, Lu1, ORG, "true class\nnovel", NU)]:
        ax.add_patch(plt.Rectangle((x, y0), 0.055, y1 - y0, fc=c, ec="none", zorder=4))
        ax.text(x - 0.015, (y0 + y1) / 2, f"{lab}\n{n:,}", ha="right", va="center", fontsize=6.5)
    for x, y0, y1, c, lab, n in [(1.245, Rs0, Rs1, BLU, "closed-set\nroute", KT + KU),
                                 (1.245, Rr0, Rr1, ORG, "retrieval\nroute", ET + EU)]:
        ax.add_patch(plt.Rectangle((x, y0), 0.055, y1 - y0, fc=c, ec="none", zorder=4))
        ax.text(x + 0.070, (y0 + y1) / 2, f"{lab}\n{n:,}", ha="left", va="center", fontsize=6.5)

    tt = Lt1 - KT / 35665                        # trained -> closed-set
    _ribbon(ax, xa, xb, Lt1, tt, Rs1, Rs1 - KT / 35665, BLU, .50)
    _ribbon(ax, xa, xb, tt, Lt0, Rr1, Rr1 - ET / 35665, RED, .55)
    uu = Lu1 - KU / 35665                        # novel -> closed-set (lost)
    _ribbon(ax, xa, xb, Lu1, uu, Rs1 - KT / 35665, Rs0, RED, .55)
    _ribbon(ax, xa, xb, uu, Lu0, Rr1 - ET / 35665, Rr0, ORG, .50)

    ax.text(0.50, Lt1 - KT / 70000 + 0.015, f"{KT:,}  kept", ha="center", fontsize=6.4,
            color="white", weight="bold")
    ax.text(0.50, Lu0 + EU / 70000 - 0.02, f"{EU:,}  ejected", ha="center", fontsize=6.4,
            color="white", weight="bold")
    ax.annotate(f"{ET + KU:,} images ({(ET + KU) / 35665 * 100:.1f}%)\nlost to routing alone",
                xy=(0.50, (tt + Lt0) / 2 - 0.015), xytext=(0.50, -0.055),
                ha="center", fontsize=6.5, color=RED,
                arrowprops=dict(arrowstyle="->", lw=0.8, color=RED))
    ax.text(0.50, 1.11, "true class not in the candidate set it was scored against",
            ha="center", fontsize=6.3, color=RED, style="italic")
    fig.tight_layout(pad=0.12)
    fig.savefig(f"{FIG}/flow.png", dpi=430, bbox_inches="tight")
    plt.close(fig)


# ================================================== Fig 3: marginal trade-off
def fig_marginal():
    fig, ax = plt.subplots(figsize=(3.30, 1.95))
    a = 0.880
    b = np.linspace(0.05, 0.45, 300)
    ax.plot(b * 100, a / (a + b) * 100, lw=1.5, color=INK, zorder=3)
    ax.fill_between(b * 100, a / (a + b) * 100, 100, color=BLU, alpha=.07, lw=0)
    ax.fill_between(b * 100, 50, a / (a + b) * 100, color=ORG, alpha=.07, lw=0)
    for bb, lab, col in [(0.1948, "v36\nf = 0.72", BLU), (0.260, "v50\nf = 0.60", ORG)]:
        q = a / (a + bb) * 100
        ax.plot(bb * 100, q, "o", ms=5.5, mfc="white", mec=col, mew=1.6, zorder=5)
        ax.annotate(lab, (bb * 100, q), textcoords="offset points",
                    xytext=(7, 7), fontsize=6.8, color=col, weight="bold")
    ax.annotate("", xy=(26.0, 77.2), xytext=(19.5, 81.9),
                arrowprops=dict(arrowstyle="->", lw=1.0, color=RED))
    ax.text(28.5, 84.0, "retrieval improves b\n-> eject more", fontsize=6.6, color=RED)
    ax.text(6.5, 88.5, "keeping is better", fontsize=6.7, color=BLU)
    ax.text(30.0, 66.0, "ejecting is better", fontsize=6.7, color=ORG)
    ax.set_xlim(5, 45); ax.set_ylim(62, 95)
    ax.set_xlabel("retrieval-route conditional accuracy  b  (%)")
    _clean(ax, "novel purity needed\nat the margin,  q*  (%)")
    fig.tight_layout(pad=0.15)
    fig.savefig(f"{FIG}/marginal.png", dpi=430, bbox_inches="tight")
    plt.close(fig)


# ==================================================== Fig 4: aspect-ratio shift
def fig_aspect():
    fig, ax = plt.subplots(figsize=(3.30, 1.85))
    rows = [("train\n61,941 img", 0.75, 1.4618, 1.9110, 1.4590),
            ("eval: trained\n20,097 img", 0.9974, 1.4988, 2.1192, 1.5326),
            ("eval: novel\n15,568 img", 1.3531, 1.7439, 2.7586, 1.9177)]
    ax.axvspan(0.75, 1.33, color=RED, alpha=0.13, lw=0, zorder=1)
    ax.axvspan(0.50, 2.00, color=BLU, alpha=0.09, lw=0, zorder=0)
    for i, (lab, p10, p50, p90, mean) in enumerate(rows):
        y = 2 - i
        ax.plot([p10, p90], [y, y], color=INK, lw=2.0, solid_capstyle="butt", zorder=3)
        for e in (p10, p90):
            ax.plot([e, e], [y - .16, y + .16], color=INK, lw=1.2, zorder=3)
        ax.plot(p50, y, "o", ms=4.6, color="white", mec=INK, mew=1.4, zorder=4)
        ax.text(p90 + 0.07, y, f"p90 {p90:.2f}", va="center", fontsize=6.6, color=MID)
    ax.set_yticks([2, 1, 0]); ax.set_yticklabels([r[0] for r in rows], fontsize=6.6)
    ax.set_xlim(0.42, 3.25); ax.set_ylim(-1.02, 2.62)
    ax.set_xlabel("image aspect ratio (width / height)")
    ax.text(1.04, -0.80, "sampled by the\noriginal augmentation", fontsize=6.3,
            color=RED, ha="center", va="center")
    ax.text(2.55, -0.80, "corrected\nrange", fontsize=6.3, color=BLU, ha="center", va="center")
    _clean(ax); _nospine(ax)
    fig.tight_layout(pad=0.15)
    fig.savefig(f"{FIG}/aspect.png", dpi=430, bbox_inches="tight")
    plt.close(fig)


# ================================================ Fig 5: operating points
SUBS = [("v33", 78.20, 8.51), ("v34", 78.63, 7.80), ("v36", 78.95, 10.42),
        ("v37", 78.95, 13.35), ("v39", 78.95, 13.44), ("v40", 78.95, 13.76),
        ("v41", 78.95, 13.93), ("v43", 78.95, 14.36), ("v44", 78.95, 14.39),
        ("v46", 78.95, 14.55), ("v50", 76.31, 19.34), ("v56", 75.67, 20.57)]


def fig_operating():
    fig, ax = plt.subplots(figsize=(3.30, 2.55))
    for lvl in (48, 49, 50, 51, 52, 53):
        xs = np.array([74.0, 81.0])
        ax.plot(xs, (lvl - 0.5635 * xs) / 0.4365, lw=0.7, ls=(0, (4, 3)),
                color=PALE, zorder=1)
        yl = (lvl - 0.5635 * 80.15) / 0.4365
        if 6.6 < yl < 22.0:
            ax.text(80.15, yl + 0.12, f"{lvl}%", fontsize=6.2, color=MID,
                    ha="left", va="bottom")
    ax.plot([s[1] for s in SUBS], [s[2] for s in SUBS], "-", lw=0.9, color=PALE, zorder=2)
    for name, s, u in SUBS:
        dead = name in ("v34", "v39")
        c = RED if dead else INK
        ax.plot(s, u, "o", ms=4.4, color="white", mec=c, mew=1.3, zorder=4)
    for name, dx, dy, ha in [("v33", 0, -0.95, "center"), ("v34", -0.16, -0.95, "center"),
                             ("v36", 0, -0.98, "center"), ("v37", -0.30, -0.30, "left"),
                             ("v46", -0.30, 0.34, "left"), ("v50", 0, 0.68, "center"),
                             ("v56", 0, 0.68, "center")]:
        s, u = next((q[1], q[2]) for q in SUBS if q[0] == name)
        ax.text(s + dx, u + dy, name, fontsize=6.8, ha=ha,
                va="center" if dx else "baseline",
                color=RED if name == "v34" else INK,
                weight="bold" if name in ("v50", "v56") else "normal")
    ax.plot([79.68, 79.68], [13.32, 14.58], lw=0.9, color=MID)
    for yy in (13.32, 14.58):
        ax.plot([79.68, 79.53], [yy, yy], lw=0.9, color=MID)
    ax.text(79.82, 13.95, "v39-v44", fontsize=6.5, color=MID, ha="right", va="center")
    ax.plot(78.05, 15.08, "^", ms=6.5, color=GRN, zorder=5)
    ax.text(77.82, 15.55, "leader", fontsize=6.8, ha="left", color=GRN, weight="bold")
    ax.annotate("", xy=(78.72, 14.30), xytext=(78.72, 10.80),
                arrowprops=dict(arrowstyle="->", lw=1.2, color=BLU))
    ax.text(78.52, 12.3, "external\nretrieval", fontsize=6.7, color=BLU,
            va="center", ha="left", weight="bold")
    ax.annotate("", xy=(76.42, 19.02), xytext=(78.55, 14.98),
                arrowprops=dict(arrowstyle="->", lw=1.2, color=ORG))
    ax.text(76.95, 17.6, "routing\nf .72 to .60", fontsize=6.7, color=ORG,
            ha="left", va="center", weight="bold")
    ax.set_xlim(80.9, 74.9); ax.set_ylim(6.4, 22.6)
    ax.set_xlabel("trained-class top-1 (%)      <-  better")
    _clean(ax, "novel-class top-1 (%)    better  ->")
    fig.tight_layout(pad=0.15)
    fig.savefig(f"{FIG}/operating.png", dpi=430, bbox_inches="tight")
    plt.close(fig)


# ==================================================== Fig 6: progress
def fig_progress():
    labels = ["v22", "v23", "v28", "v31", "v33", "v36", "v37", "v40", "v43", "v46", "v50", "v56"]
    overall = [45.39, 45.96, 47.26, 47.69, 47.78, 49.04, 50.46, 50.49, 50.76, 50.84, 51.44, 51.62]
    novel = [None, None, None, None, 8.51, 10.42, 13.68, 13.76, 14.36, 14.55, 19.34, 20.57]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(3.30, 2.70), sharex=True,
                                 gridspec_kw={"height_ratios": [1.15, 1]})
    x = list(range(len(labels)))
    a1.plot(x, overall, "-o", ms=3.8, lw=1.3, color=INK, zorder=3)
    a1.axhline(50.56, ls=(0, (4, 3)), lw=0.9, color=GRN)
    a1.text(0.1, 50.74, "leaderboard #1  50.56", fontsize=6.4, color=GRN)
    a1.axhline(53.0, ls=(0, (1, 2.5)), lw=1.0, color=RED)
    a1.text(0.1, 53.18, "target  53.00", fontsize=6.4, color=RED)
    a1.set_ylim(44.6, 54.3); _clean(a1, "overall top-1 (%)")
    xs = [i for i, v in enumerate(novel) if v is not None]
    a2.plot(xs, [v for v in novel if v is not None], "-s", ms=3.6, lw=1.3, color=ORG, zorder=3)
    a2.annotate("external bank", xy=(6, 13.68), xytext=(3.1, 17.9), fontsize=6.5,
                color=MID, ha="center",
                arrowprops=dict(arrowstyle="->", lw=0.8, color=MID))
    a2.annotate("f .72 to .60", xy=(10, 19.34), xytext=(7.9, 21.4), fontsize=6.5,
                color=MID, ha="center",
                arrowprops=dict(arrowstyle="->", lw=0.8, color=MID))
    a2.set_ylim(6, 23.5); _clean(a2, "novel-class top-1 (%)")
    a2.set_xticks(x); a2.set_xticklabels(labels, fontsize=6.6)
    fig.tight_layout(pad=0.15, h_pad=0.6)
    fig.savefig(f"{FIG}/progress.png", dpi=430, bbox_inches="tight")
    plt.close(fig)


# ==================================================== Fig 7: encoders
def fig_encoders():
    rows = [("BioCLIP-2.5-H", 23.38, 21.53), ("TaxaBind ViT-B/16", 13.98, 11.48),
            ("BioCLIP 2", None, 14.37), ("BioCAP", 12.25, 6.77),
            ("BioCLIP 1", None, 11.60), ("SigLIP2 SO400M", None, 5.13),
            ("BioCLIP hyperbolic", 4.92, 4.36), ("general CLIP", None, 0.78),
            ("bioclip-inat-only", None, 0.35), ("BioTrove-CLIP", 0.04, 0.00)]
    fig, ax = plt.subplots(figsize=(3.30, 2.45))
    y = [v + 2.3 for v in list(range(len(rows)))[::-1]]
    bb = [(v + 0.20, r[1]) for v, r in zip(y, rows) if r[1] is not None]
    ax.barh([q[0] for q in bb], [q[1] for q in bb], height=0.38, color=INK,
            zorder=3, label="bare binomial")
    tx = [(v - 0.20, r[2]) for v, r in zip(y, rows) if r[2] is not None]
    ax.barh([q[0] for q in tx], [q[1] for q in tx], height=0.38, color="#a8a8a8",
            zorder=3, label="taxonomic context")
    for v, r in zip(y, rows):
        best = max(q for q in (r[1], r[2]) if q is not None)
        off = 0.20 if (r[1] is not None and r[1] >= (r[2] if r[2] is not None else -1)) else -0.20
        ax.text(best + 0.45, v + off, f"{best:.2f}", va="center", fontsize=6.4)
    ax.barh([1.05], [31.10], height=0.46, color=BLU, zorder=3)
    ax.text(31.6, 1.05, "31.10", va="center", fontsize=6.4, color=BLU, weight="bold")
    ax.barh([0.0], [26.62], height=0.46, color="#9dbdd8", zorder=3)
    ax.text(27.1, 0.0, "26.62", va="center", fontsize=6.4, color=BLU)
    ax.axhline(1.85, lw=0.8, color=MID, ls=(0, (3, 2)))
    ax.set_yticks(y + [1.05, 0.0])
    ax.set_yticklabels([r[0] for r in rows] +
                       ["adapter, corrected aug.", "adapter, original aug."], fontsize=6.5)
    for lab in ax.get_yticklabels()[-2:]:
        lab.set_color(BLU)
    ax.set_xlim(0, 36.5)
    ax.set_xlabel("novel-class top-1 on the held-out proxy (%)")
    ax.legend(frameon=False, loc="upper right", handlelength=1.2,
              bbox_to_anchor=(1.02, 1.03), borderpad=0.15)
    _clean(ax); _nospine(ax)
    fig.tight_layout(pad=0.15)
    fig.savefig(f"{FIG}/encoders.png", dpi=430, bbox_inches="tight")
    plt.close(fig)


# ==================================================== Fig 8: ceiling / oracle
def fig_ceiling():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(3.30, 2.00),
                                 gridspec_kw={"width_ratios": [1.05, 1]})
    v = [49.04, 53.53]
    a1.bar([0, 1], v, color=[INK, "#a8a8a8"], width=0.55, zorder=3)
    a1.axhline(53.0, ls=(0, (1, 2.5)), lw=1.0, color=RED)
    a1.text(1.52, 53.05, "target", fontsize=6.5, color=RED, ha="right", va="bottom")
    for i, q in enumerate(v):
        a1.text(i, q + 0.10, f"{q:.2f}", ha="center", fontsize=6.9, weight="bold")
    a1.set_xticks([0, 1])
    a1.set_xticklabels(["as routed\n(v36)", "perfect\ngate"], fontsize=6.5)
    a1.set_ylim(48.0, 54.6); _clean(a1, "overall top-1 (%)")
    a1.set_title("routing head-room", fontsize=7.4)

    a2.bar([0, 1], [28.90, 33.26], color=[INK, "#a8a8a8"], width=0.55, zorder=3)
    a2.annotate("", xy=(1, 33.26), xytext=(0, 28.90),
                arrowprops=dict(arrowstyle="->", lw=1.1, color=RED))
    a2.text(0.5, 35.2, "+4.36 only", fontsize=6.9, ha="center", color=RED, weight="bold")
    for i, q in enumerate([28.90, 33.26]):
        a2.text(i, q + 0.30, f"{q:.2f}", ha="center", fontsize=6.9, weight="bold")
    a2.set_xticks([0, 1])
    a2.set_xticklabels(["11,598\ncandidates", "~24\ncandidates"], fontsize=6.5)
    a2.set_ylim(26, 38.5); _clean(a2, "novel-class top-1 (%)")
    a2.set_title("oracle family probe", fontsize=7.4)
    fig.tight_layout(pad=0.15, w_pad=1.3)
    fig.savefig(f"{FIG}/ceiling.png", dpi=430, bbox_inches="tight")
    plt.close(fig)


# ============================================ Fig 9: proxy-to-real transfer
def fig_transfer():
    pts = [("v37a", 9.66, 1.08), ("v37b", 1.12, 0.20), ("v40", 1.68, 0.03),
           ("v41", 0.99, 0.08), ("v43", 0.52, 0.19), ("v44", 0.43, 0.02),
           ("v46", 0.345, 0.067), ("v56", 1.424, 0.174)]
    neg = [("v34", 3.51, -0.07), ("v39", 0.52, -0.11)]
    off = {"v43": (6, 4), "v44": (7, 3), "v46": (-7, -10), "v41": (6, 4),
           "v37b": (6, 4), "v56": (6, -9), "v40": (6, 3)}
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(3.30, 3.55),
                                 gridspec_kw={"height_ratios": [1.0, 1.12]})

    a1.axhline(0, color=INK, lw=0.9, zorder=2)
    for r, st in ((0.36, (0, (1, 2))), (0.12, (0, (5, 2))), (0.02, (0, (1, 1)))):
        a1.plot([0, 2.0], [0, 2.0 * r], lw=0.9, ls=st, color=MID, zorder=1)
        a1.text(1.99, min(2.0 * r, 0.60) + 0.015, f"{r:g}", fontsize=6.6,
                color=MID, ha="right", va="bottom")
    for n, x, y in pts:
        if x <= 2.0:
            a1.plot(x, y, "o", ms=5.0, color=INK, zorder=4)
            a1.annotate(n, (x, y), textcoords="offset points",
                        xytext=off.get(n, (6, 4)), fontsize=6.8, zorder=5)
    for n, x, y in neg:
        if x <= 2.0:
            a1.plot(x, y, "X", ms=6.6, color=RED, zorder=4)
            a1.annotate(n, (x, y), textcoords="offset points", xytext=(7, -3),
                        fontsize=6.8, color=RED, va="center", zorder=5)
    a1.axhspan(-0.20, 0, color=RED, alpha=0.08, lw=0)
    a1.text(1.99, -0.175, "proxy up, real down", fontsize=6.7, color=RED,
            style="italic", ha="right", va="bottom")
    a1.set_xlim(-0.10, 2.05); a1.set_ylim(-0.20, 0.62)
    a1.set_xlabel("proxy gain (pt)"); _clean(a1, "real gain (pt)")
    a1.set_title("(a)  paired measurements   (v37a off-scale at 9.66, 1.08)",
                 fontsize=7.2, loc="left")

    names = ["dual BioCLIP-2 (v43)", "LoRA / ToL merge (v46)", "crops + 336 bank (v56)",
             "iNat mean prototypes (v37)", "BioCLIP-2 prototypes (v41)",
             "frozen ToL merge (v44)", "top-4 pooling (v40)"]
    vals = [0.359, 0.194, 0.122, 0.120, 0.081, 0.039, 0.018]
    yy = list(range(len(vals)))[::-1]
    a2.barh(yy, vals, height=0.60, color=INK, zorder=3)
    a2.barh([-1.6, -2.6], [-0.020, -0.212], height=0.60, color=RED, zorder=3)
    for y, v in zip(yy, vals):
        a2.text(v + 0.010, y, f"{v:.3f}", va="center", fontsize=6.8)
    for y, v in zip([-1.6, -2.6], [-0.020, -0.212]):
        a2.text(v - 0.014, y, f"{v:.3f}", va="center", ha="right",
                fontsize=6.8, color=RED)
    a2.axvline(0, color=INK, lw=0.9, zorder=4)
    a2.set_yticks(yy + [-1.6, -2.6])
    a2.set_yticklabels(names + ["recover-on-eject (v34)", "TaxaBind protos (v39)"],
                       fontsize=6.8)
    for lab in a2.get_yticklabels()[-2:]:
        lab.set_color(RED)
    a2.axhspan(-3.2, -1.0, color=RED, alpha=0.07, lw=0)
    a2.set_xlim(-0.355, 0.44); a2.set_ylim(-3.2, 6.7)
    a2.set_xlabel("real accuracy points gained per proxy point gained")
    a2.set_title("(b)  transfer rate", fontsize=7.2, loc="left")
    _clean(a2); _nospine(a2)
    fig.tight_layout(pad=0.15, h_pad=1.0)
    fig.savefig(f"{FIG}/transfer.png", dpi=430, bbox_inches="tight")
    plt.close(fig)


for _f in (fig_pipeline, fig_flow, fig_marginal, fig_aspect, fig_operating,
           fig_progress, fig_encoders, fig_ceiling, fig_transfer):
    _f()


# ===================================================================== styles
def S(name, **kw):
    base = dict(fontName="Times-Roman", fontSize=9, leading=10.25, alignment=TA_JUSTIFY,
                spaceAfter=0, spaceBefore=0)
    base.update(kw)
    return ParagraphStyle(name, **base)


TITLE = S("t", fontName="Times-Bold", fontSize=17.5, leading=20.5, alignment=TA_CENTER, spaceAfter=9)
AUTH = S("a", fontSize=10.5, leading=12.5, alignment=TA_CENTER, spaceAfter=2)
ABSH = S("abh", fontName="Times-Bold", fontSize=9.5, leading=11, alignment=TA_CENTER, spaceAfter=4)
ABS = S("ab", fontSize=8.9, leading=10.2)
BODY = S("b", firstLineIndent=11, spaceAfter=0)
BODY0 = S("b0", firstLineIndent=0)
H1 = S("h1", fontName="Times-Bold", fontSize=11, leading=12.6, alignment=0, spaceBefore=7.5, spaceAfter=3)
H2 = S("h2", fontName="Times-Bold", fontSize=9.6, leading=11, alignment=0, spaceBefore=5.5, spaceAfter=2)
CAP = S("cap", fontSize=7.85, leading=9.0, spaceBefore=3.0, spaceAfter=5.0)
EQ = S("eq", fontSize=9.3, leading=12, alignment=TA_CENTER, spaceBefore=4, spaceAfter=4)
REF = S("r", fontSize=7.6, leading=8.8, leftIndent=11, firstLineIndent=-11, spaceAfter=1.9)
BUL = S("bu", fontSize=9, leading=10.25, leftIndent=10, firstLineIndent=-7, spaceAfter=1.8)

TAU = "tau"
ALPHA = "alpha"
DELTA = "delta"


def P(t, s=BODY):
    return Paragraph(t, s)


def bullets(items):
    return [P("&bull;&nbsp; " + i, BUL) for i in items]


def tbl(data, widths, caption, align=None, fs=7.3):
    from reportlab.lib.enums import TA_RIGHT
    amap = {"RIGHT": TA_RIGHT}
    cells = []
    for r, row in enumerate(data):
        out = []
        for c, cell in enumerate(row):
            a = amap.get((align or {}).get(c, "LEFT"), 0)
            stl = ParagraphStyle(f"tc{r}{c}", fontName="Times-Bold" if r == 0 else "Times-Roman",
                                 fontSize=fs, leading=fs * 1.22, alignment=a)
            out.append(Paragraph(str(cell), stl))
        cells.append(out)
    t = Table(cells, colWidths=[w * inch for w in widths], hAlign="LEFT")
    st = [("FONT", (0, 0), (-1, -1), "Times-Roman", fs),
          ("FONT", (0, 0), (-1, 0), "Times-Bold", fs),
          ("LINEABOVE", (0, 0), (-1, 0), 0.9, colors.black),
          ("LINEBELOW", (0, 0), (-1, 0), 0.55, colors.black),
          ("LINEBELOW", (0, -1), (-1, -1), 0.9, colors.black),
          ("TOPPADDING", (0, 0), (-1, -1), 2.1),
          ("BOTTOMPADDING", (0, 0), (-1, -1), 2.1),
          ("LEFTPADDING", (0, 0), (-1, -1), 2.2),
          ("RIGHTPADDING", (0, 0), (-1, -1), 2.2),
          ("VALIGN", (0, 0), (-1, -1), "TOP")]
    t.setStyle(TableStyle(st))
    return KeepTogether([t, P(caption, CAP)])


def figure(path, w, caption):
    from PIL import Image as PILImage
    iw, ih = PILImage.open(path).size
    h = w * ih / iw
    return KeepTogether([Image(path, width=w * inch, height=h * inch), P(caption, CAP)])


story = []

# ===================================================================== title
story += [
    NextPageTemplate("rest"),
    P("Zero-Shot Species Recognition at 17,393 Classes:<br/>What a Held-Out Proxy Did Not Predict", TITLE),
    Spacer(1, 4),
    P("Abstract", ABSH),
    P(
        "We report a measured study of large-vocabulary generalized zero-shot recognition on a "
        "fish-species benchmark "
        "with 17,393 candidate classes, of which 11,598 (66.7%) have no training images at all by task "
        "definition. We describe a final system that partitions the decision between a closed-set "
        "route over the 5,795 trained classes and a retrieval route over the remainder, and we use its "
        "hard 30-submission evaluation budget to measure three things that are more often assumed than "
        "tested. First, we diagnose a <i>framing shift</i>: the training augmentation sampled crop aspect "
        "ratios in [0.75, 1.33], while evaluation images of novel species have mean aspect 1.92 and 90th "
        "percentile 2.76, so the network never saw the test framing; correcting the sampled range and "
        "adding a max-over-crops inference rule produce independent, separately confirmed gains. Second, "
        "we show that a retrieval bank of 101,106 external biodiversity photographs raises top-1 "
        "accuracy on zero-training-image classes from 10.42% to 20.57%, and we audit the evaluation "
        "set against that bank for near-duplicates, finding a small but real population which bounds "
        "the possible inflation at 0.91 of those 10.15 points. Third &mdash; our principal "
        "negative result &mdash; we quantify how well a held-out proxy predicts real gains across seven "
        "confirmed changes and find transfer rates spanning more than an order of magnitude, two changes "
        "that transferred with the wrong sign, and one exactly-anchored case in which the proxy preferred "
        "a configuration by 5.09 points that the real evaluation reversed. Oracle probes localize the "
        "remaining headroom to the encoder rather than the routing policy: granting perfect family-level "
        "knowledge, which collapses 11,598 candidates to roughly two dozen, raises novel-class top-1 only "
        "from 28.90% to 33.26%. We report our full catalogue of measured negative results.", ABS),
    FrameBreak(),
]

# ============================================================ 1 introduction
story += [
    P("1. Introduction", H1),
    P("Species-level visual recognition is open-ended by construction. Any taxonomically complete "
      "candidate list contains far more species than any image collection covers, so a final "
      "recognizer must place images into classes for which it has never seen a single training "
      "example. On the benchmark studied here the imbalance is extreme rather than marginal: 11,598 "
      "of 17,393 candidate classes &mdash; 66.7% &mdash; have no training images by definition of the "
      "task, and they account for 15,568 of the 35,665 evaluation images.", BODY0),
    P("A word on terminology, because it decides which literature this belongs to. All 17,393 labels "
      "are known in advance and every novel class is identified by its scientific name; nothing has to "
      "be rejected as unknown. This is <i>generalized zero-shot</i> recognition in the sense of Xian "
      "<i>et al.</i> [3], with seen and unseen classes scored under one metric, and not open-set "
      "recognition in the sense of Scheirer <i>et al.</i> [1], which requires refusing to answer. We "
      "draw on the open-set literature [1, 2] only for the routing question, which is genuinely "
      "shared: judging whether an image belongs to the trained vocabulary is the decision an open-set "
      "rejector makes, though we act on it by switching branches rather than by abstaining. What makes "
      "the setting "
      "unusual, and what this paper is about, is the combination of that class imbalance with a hard "
      "evaluation budget: three submissions per day, thirty in total. A budget that small changes the "
      "research process itself. Almost every decision must first be screened on an internal held-out "
      "proxy, and only changes that clear a fixed proxy margin are spent on a real evaluation."),
    P("That constraint turns an ordinary engineering convenience into a measurable object of study. A "
      "proxy metric is worth exactly as much as its ability to order configurations the way the real "
      "evaluation orders them, and over thirty real submissions we accumulated enough paired "
      "measurements to ask whether ours did. It largely did not. Across seven changes that each cleared "
      "the proxy bar and were then confirmed on the real evaluation, the fraction of proxy gain that "
      "survived varied by more than an order of magnitude; two further changes moved the proxy up and "
      "the real score down; and one architectural simplification that the proxy preferred by 5.09 "
      "points was worse by 0.60 points in reality. We consider this the most transferable finding in "
      "the paper, and we report it with the anchors that make it checkable rather than as an "
      "impression."),
    P("Alongside it we report two positive results that are ordinary in method but were found by "
      "measurement rather than intuition. One is a quantified mismatch between training augmentation "
      "geometry and evaluation image geometry, which we correct in two independent places. The other is "
      "a retrieval bank assembled from external biodiversity photographs, which roughly doubles accuracy "
      "on the zero-training-image half of the label space, and which we accompany with the "
      "duplicate-detection control such a claim requires. Finally, we use oracle probes to argue that "
      "what remains is a limitation of the visual encoder, not of the routing policy that most of our "
      "engineering effort went into."),
    P("Contributions.", H2),
]
story += bullets([
    "A description of a final open-set system at 17,393-class scale, including the components that "
    "proved load-bearing (a bidirectional score normalization; a combined image-and-text routing signal) "
    "and an explicit account of the three ways in which it is transductive rather than per-image.",
    "A measured diagnosis of framing shift between training augmentation and evaluation imagery "
    "(Sec. 5), with two independent corrections, one at training time and one at inference time, each "
    "separately confirmed on the real evaluation.",
    "An external-bank retrieval route for classes with zero training images (Sec. 6), raising novel-class "
    "top-1 from 10.42% to 20.57%, with a duplicate-detection control against the obvious objection.",
    "A proxy-reliability study (Sec. 8): seven proxy-to-real anchors spanning 0.018 to 0.359 real points "
    "per proxy point, two sign inversions, and one exactly-anchored ranking inversion of 5.09 points.",
    "Oracle probes localizing the residual gap to the encoder (Sec. 7), and a catalogue of twenty "
    "measured negative results (Sec. 9) that we believe saves more time than the positive results do.",
])

# ============================================================ 2 related work
story += [
    P("2. Related Work", H1),
    P("<b>Open-set and zero-shot recognition.</b> Open-set recognition [1, 2] concerns classifiers that "
      "must not be forced to emit a label from a closed training vocabulary. Our novel-class route is "
      "closer to generalized zero-shot learning [3], where seen and unseen classes are scored jointly "
      "under a single metric &mdash; exactly the protocol here (Sec. 3). Where the standard zero-shot "
      "setting supplies attributes or class embeddings, we supply external photographs, which changes "
      "the failure mode from attribute quality to coverage and domain gap.", BODY0),
    P("<b>Vision-language and biology foundation models.</b> CLIP [4] introduced the pattern our system "
      "relies on throughout: classification by cosine similarity between an image embedding and frozen "
      "per-class text embeddings, with no learned linear head. BioCLIP [5] specializes this to biology "
      "using taxonomic text over the TreeOfLife-10M corpus, and BioCLIP&nbsp;2 [6] scales it to "
      "TreeOfLife-200M. Both appear in our system, in different roles and at different scales: the "
      "closed-set route runs a BioCLIP-2.5 ViT-H/14 ensemble, while the novel-class route additionally "
      "uses BioCLIP&nbsp;2 ViT-L/14 prototypes. TaxaBind [11] binds six ecological modalities into a "
      "shared space anchored on ground-level species imagery; we use its image and text towers as one "
      "leg of the novel-class text stack."),
    P("<b>Parameter-efficient adaptation.</b> We adapt the image tower with LoRA [7], which we found "
      "necessary in a specific and slightly unusual way: full fine-tuning of a strong biology encoder "
      "on a narrow fish subset degrades exactly the broad taxonomic structure the novel-class route "
      "depends on. At inference we additionally scale the low-rank term below its trained strength, a "
      "discrete analogue of the weight-space interpolation used by WiSE-FT."),
    P("<b>Margin losses on cosine classifiers.</b> Because the classifier compares an embedding to a "
      "fixed class matrix by cosine similarity, our training objective (Eq. 5) is structurally a "
      "member of the family developed for face recognition, where CosFace [8] subtracts an additive "
      "margin from the target class similarity and ArcFace [9] applies the equivalent margin in "
      "angular space. We use the margin-free form. We note the connection because it locates Eq. 5 "
      "for the reader and because the fixed logit scale it already carries is the same scale term "
      "those losses require; we did not train a margin variant and make no claim about one."),
    P("<b>Prototypes and retrieval.</b> Snell <i>et al.</i> [10] classify by distance to a class mean "
      "embedding, which is precisely what our BioCLIP&nbsp;2 external-prototype leg does. Our iNaturalist "
      "photograph banks are deliberately <i>not</i> prototypes: we retain individual photograph "
      "embeddings and score a class by the mean of its top-4 cosine similarities, which measured better "
      "than either a class mean or a single maximum (Sec. 6)."),
    P("<b>Score normalization and transport.</b> The single most load-bearing normalization in our "
      "system applies a log-softmax across images as well as across classes, penalizing candidate "
      "classes that are similar to everything &mdash; the hubness correction that Sinkhorn-style balanced "
      "assignment achieves in a more principled way. Entropic optimal transport [13] and its use for "
      "balanced online assignment in SwAV [14] are the direct antecedents; we apply a Sinkhorn "
      "projection with a uniform column prior over novel-class rows only."),
    P("<b>Proxy metrics, adaptive evaluation, and benchmark reuse.</b> The paper's central negative "
      "result belongs to a literature we should state clearly, including where it disagrees with us. "
      "Dwork <i>et al.</i> [18, 19] formalize the loss of validity when a held-out set answers a "
      "sequence of adaptively chosen queries, and Blum and Hardt [20] give a leaderboard mechanism "
      "restoring it; our thirty sequential submissions are that regime, run without any such "
      "protection. The prevailing empirical picture is nonetheless more optimistic than our "
      "experience. Roelofs <i>et al.</i> [21] analyse 120 competitions and find public-to-private "
      "leaderboard ordering remarkably reliable, concluding that distribution shift matters more than "
      "test reuse; Recht <i>et al.</i> [22] find replication drops on ImageNet that preserve ordering, "
      "with original-set gains translating into larger new-set gains; and Miller <i>et al.</i> [23] "
      "report strong linear in- to out-of-distribution correlation across many settings. We offer a "
      "single-benchmark counterexample, not a refutation, and note that Teney <i>et al.</i> [24] "
      "document real datasets where the correlation inverts &mdash; the regime our two sign inversions "
      "fall into."),
    P("<b>Transductive and calibrated zero-shot recognition.</b> Our routing fraction is a "
      "seen-versus-unseen calibration parameter of the kind Chao <i>et al.</i> [25] introduced as "
      "calibrated stacking, except that Sec. 4.6 derives its optimum instead of sweeping it. "
      "Transductive zero-shot methods [26] use unlabeled target data during <i>training</i>; our "
      "coupling differs in kind, occurring only at inference across the evaluation batch, and its "
      "closer relatives are batch-level assignment methods such as Sinkhorn label allocation [27]."),
    P("<b>Hubness and score normalization.</b> The across-image term of Eq. 3 is a hubness correction "
      "in the sense of Radovanovi&#263; <i>et al.</i> [28]: in high-dimensional similarity spaces a "
      "few candidates sit close to everything. Its direct ancestor is the inverted softmax of Smith "
      "<i>et al.</i> [29], which divides a similarity by how popular the candidate is across the query "
      "set; querybank normalization [30] reaches the same end without test-query access, which ours "
      "requires. We had rediscovered this operation empirically before finding its lineage, and it "
      "remains the largest single normalization term in the system."),
    P("<b>Retrieval-augmented classification.</b> Classifying against an external corpus at inference "
      "is established: retrieval-augmented classification for long-tailed recognition [31] and "
      "name-only transfer of vision-language models [32] both assemble support sets for classes the "
      "model never trained on. Our contribution is not the mechanism but its scale and its audit "
      "&mdash; 101,106 photographs over 7,364 species, and the near-duplicate check of Sec. 6 that we "
      "believe such claims require and that our own earlier version performed incorrectly."),
    P("<b>Test-time augmentation and fine-grained recognition.</b> Fixed multi-crop evaluation is "
      "standard practice; learned policies [15] are the systematic version. Our contribution here is "
      "not the technique but the diagnosis that motivated a specific geometry (Sec. 5). The task itself "
      "is fine-grained categorization in the sense of Wei <i>et al.</i> [16], and the closest domain "
      "benchmark is FishNet [17], which is closed-set and so does not exercise the regime that dominates "
      "our error budget; on vocabulary scale the nearest comparison is the iNaturalist benchmarking "
      "work of Van Horn <i>et al.</i> [33]."),
]

# ================================================================== 3 task
story += [
    P("3. Task, Data, and Protocol", H1),
    P("<b>Label space and splits.</b> The candidate space contains N = 17,393 species-level classes. "
      "Of these, 5,795 have at least one training image; we call these <i>trained</i> classes. The "
      "remaining 11,598 have none, and no training images for them exist anywhere in the provided data "
      "&mdash; they are specified by scientific name only. The organizers release 64,259 labelled "
      "training images over those 5,795 classes. The evaluation set contains 35,665 images: "
      "20,097 whose true class is trained and 15,568 whose true class is novel.", BODY0),
    tbl([["quantity", "count", "share"],
         ["candidate classes", "17,393", "100%"],
         ["&nbsp;&nbsp;with training images (<i>trained</i>)", "5,795", "33.3%"],
         ["&nbsp;&nbsp;without any (<i>novel</i>)", "11,598", "66.7%"],
         ["labelled training images", "64,259", "100%"],
         ["&nbsp;&nbsp;used for adapter training", "61,941", "96.4%"],
         ["&nbsp;&nbsp;withheld as pseudo-novel", "2,318", "3.6%"],
         ["&nbsp;&nbsp;&nbsp;&nbsp;classes withheld", "1,159", "20% of trained"],
         ["proxy candidate list (novel + pseudo)", "12,757", "&mdash;"],
         ["evaluation images", "35,665", "100%"],
         ["&nbsp;&nbsp;true class trained", "20,097", "56.35%"],
         ["&nbsp;&nbsp;true class novel", "15,568", "43.65%"],
         ["external bank photographs", "117,225", "&mdash;"],
         ["&nbsp;&nbsp;embedded in final bank", "101,545", "86.7%"],
         ["&nbsp;&nbsp;classes represented", "7,364", "&mdash;"]],
        [1.72, 0.86, 0.60],
        "<b>Table 1.</b> Task scale. The evaluation split proportions are exactly the metric weights of "
        "Eq. 1, so the reported score is plain top-1 accuracy decomposed by subset. The proxy candidate list of Sec. 8.1 is the 11,598 novel classes plus the 1,159 withheld pseudo-novel ones, which is where the recurring figure 12,757 comes from.",
        align={1: "RIGHT", 2: "RIGHT"}),
    P("<b>Metric.</b> The reported score is top-1 accuracy over all 35,665 evaluation images. Because "
      "20,097/35,665 = 0.5635 and 15,568/35,665 = 0.4365, this decomposes exactly as", BODY0),
    P(f"Acc = 0.5635 &middot; Acc<sub>seen</sub> + 0.4365 &middot; Acc<sub>novel</sub>&nbsp;&nbsp;&nbsp;(1)", EQ),
    P("so the familiar weighting is not a design choice but the population proportion. One point of "
      "trained-class accuracy is worth 0.56 points overall; one point of novel-class accuracy is worth "
      "0.44. This asymmetry drives the routing analysis in Sec. 4.5.", BODY0),
    P("<b>Constraint.</b> The rules forbid using evaluation-set metadata &mdash; specifically the "
      "provided split files &mdash; to determine whether an image's true class is trained or novel. "
      "One uniform procedure must apply to every image, and the emitted label must lie in the 17,393-class "
      "space. Deriving the trained/novel class <i>partition</i> from the training labels is permitted; "
      "consulting evaluation membership is not. Our system respects this, but is transductive in five "
      "other ways that we state plainly in Sec. 4.7 rather than leave for a reader to discover."),
    P("<b>Budget and submission policy.</b> Three real evaluations per day, thirty in total. A change "
      "reaches the leaderboard only after clearing three conditions: a minimum proxy improvement over "
      "the incumbent (+0.5 points during the encoder and text phase, relaxed to +0.3 once "
      "external-prototype work reduced typical effect sizes); agreement from a class-disjoint "
      "cross-validation split that the new component deserves non-zero weight; and a projected real "
      "gain, computed with that component's own transfer factor, that beats the incumbent. All "
      "internal measurements in this paper are labelled <i>proxy</i> and all leaderboard measurements "
      "<i>real</i>; we never report one as the other, and Sec. 8 is the justification for insisting."),
]

# ================================================================== 4 system
story += [
    P("4. System Description", H1),
    figure(f"{FIG}/pipeline.png", 3.18,
           "<b>Figure 1.</b> System overview. A gate combining an image-space novelty signal with a "
           "text-space margin selects the highest-scoring fraction f of images for the closed-set route; "
           "the two routes then argmax over <i>disjoint</i> candidate sets whose union is the full label "
           "space. Note that no single 17,393-way decision is ever taken."),
    P("4.1. Partitioned decision", H2),
    P("The system does not compute one score vector over 17,393 classes. Images routed to the "
      "closed-set branch take an argmax over the 5,795 trained classes; images routed to the novel "
      "branch take an argmax over the disjoint 11,598. The union is the full label space and every "
      "image receives one label from it, which satisfies the rule, but the distinction matters for "
      "interpreting the results: a routing error is unrecoverable, because the true class is absent "
      "from the candidate set the image was scored against. Sec. 7 quantifies exactly how much this "
      "costs.", BODY0),
    P("4.2. Closed-set route", H2),
    P("Three fine-tuned BioCLIP-2.5 ViT-H/14 encoders are ensembled with weights 1.0 (a contrastive "
      "LoRA trained with shift-corrected augmentation), 2.5 (a classification fine-tune with the same "
      "augmentation), and 2.5 (a 336-pixel full fine-tune). For each member and each trained class c, "
      "the score combines a class mean prototype, the single best training exemplar, and a taxonomic "
      "text anchor:", BODY0),
    P("s<sub>c</sub> = &lt;e, p<sub>c</sub>&gt; + 2.0 &middot; max<sub>i &isin; c</sub> &lt;e, x<sub>i</sub>&gt; "
      "+ 4.0 &middot; &lt;e, t<sub>c</sub><sup>tax</sup>&gt;&nbsp;&nbsp;&nbsp;(2)", EQ),
    P("Here e is the L2-normalized query embedding, x<sub>i</sub> are the L2-normalized embeddings of "
      "the training images of class c, and p<sub>c</sub> is a genuine <i>class prototype</i>: the mean "
      "of those x<sub>i</sub>, renormalized to unit length. t<sub>c</sub><sup>tax</sup> is the text "
      "embedding of the class's taxonomic string. The exemplar term carries twice the weight of the "
      "prototype term, which matters for the long tail: many trained classes have very few images, and "
      "a mean over three photographs is a worse descriptor than the closest of the three. Two details "
      "the equation hides: the taxonomic anchor is added only for the two members whose text tower "
      "shares the ViT-H embedding space, and member score matrices are globally standardized before "
      "summation so that no member's scale dominates.", BODY0),
    P("<b>The closed-set candidate set is all 5,795 trained classes</b>, not the 4,636 that appear in "
      "the adapter's training objective (Eq. 7). The 1,159 classes withheld from adapter training are "
      "still scored here, using prototypes and exemplars built from their two images each with the "
      "trained encoder. Withholding them shapes the encoder; it does not remove them from the label "
      "space or push them onto the novel route.", BODY0),
    P("4.3. Novel-class route", H2),
    P("The novel-class score sums six text legs and three image legs. The text legs pair four encoders "
      "with two prompt styles &mdash; a bare scientific binomial and a taxonomic context string of the "
      "form <i>\"a photo of {species}, commonly known as {common}, a fish of the family {family}\"</i> "
      "&mdash; plus a TaxaBind [11] leg at weight 1.0. The image legs are the two external banks "
      "described in Sec. 6 (weights 4.0 and 3.0) and BioCLIP&nbsp;2 external prototypes (weights 2.5 "
      "frozen and 2.0 LoRA-adapted).", BODY0),
    tbl([["route", "leg", "w"],
         ["trained", "contrastive adapter (ViT-H)", "1.0"],
         ["", "classification fine-tune (ViT-H)", "2.5"],
         ["", "336 px full fine-tune (ViT-H)", "2.5"],
         ["", "&nbsp;&nbsp;<i>within member:</i> prototype", "1.0"],
         ["", "&nbsp;&nbsp;best training exemplar", "2.0"],
         ["", "&nbsp;&nbsp;taxonomic text anchor", "4.0"],
         ["novel", "adapter &times; binomial", "1.0"],
         ["", "adapter &times; taxonomic context", "1.0"],
         ["", "ViT-L &times; name", "0.5"],
         ["", "336 fine-tune &times; binomial", "0.75"],
         ["", "classification FT &times; binomial", "1.0"],
         ["", "TaxaBind &times; taxonomic context", "1.0"],
         ["", "external bank (adapter)", "4.0"],
         ["", "external bank (336 px)", "3.0"],
         ["", "BioCLIP&nbsp;2 prototypes, frozen", "2.5"],
         ["", "BioCLIP&nbsp;2 prototypes, adapted", "2.0"]],
        [0.52, 2.02, 0.46],
        "<b>Table 2.</b> Fusion weights of the final configuration, read from the builder. The two routes share no leg: "
        "the trained route sees no external photographs and the novel route sees no training exemplars. "
        "Sec. 8 shows this asymmetry is why a unified single-score design underperforms.",
        align={2: "RIGHT"}, fs=7.0),
    P("An important empirical detail: the taxonomic-context prompt helps on <i>fine-tuned</i> encoders "
      "(+0.56 proxy) and <i>hurts</i> on frozen ones (21.53 versus 23.38 for the bare binomial). We "
      "report this because it is the same effect that killed common-name prompts (8.89 versus 26.62) "
      "and generic prompt ensembling (22.00 versus 26.62): a biology-pretrained text tower has already "
      "concentrated its representation on scientific names, and paraphrase dilutes it.", BODY0),
    P("4.4. Bidirectional normalization and transport", H2),
    P("Every raw similarity matrix S enters the novel-class sum through", BODY0),
    P("dbnorm(S) = log softmax(S / 0.05, <i>over images</i>)<br/>"
      "+ log softmax(S / 0.5, <i>over classes</i>)&nbsp;&nbsp;&nbsp;(3)", EQ),
    P("The second term is an ordinary per-image posterior. The first is a softmax down each class "
      "column, across all 35,665 evaluation images, which suppresses hub classes that are moderately "
      "similar to everything. This term is not cosmetic: removing it costs 1.60 points of novel-class "
      "accuracy, making it one of the largest single components in the system. Novel-routed rows are "
      f"then passed through a Sinkhorn projection toward a uniform column prior ({TAU} = 1.8, 60 "
      "iterations in the final build). When this step was introduced it was measured at "
      f"{TAU} = 2.0 with 50 iterations, where it raised the number of distinct novel classes actually "
      "predicted from 38.6% to 57.5% of the vocabulary and gained 0.09 real points; we did not re-run "
      "that ablation after retuning.", BODY0),
    P("4.5. Routing gate", H2),
    P("The gate score is a standardized sum of an image-space and a text-space term:", BODY0),
    P("g = z(max<sub>c &isin; trained</sub> s<sub>c</sub>) + 2.0 &middot; z(m<sub>text</sub>)&nbsp;&nbsp;&nbsp;(4)<br/>"
      "where m<sub>text</sub> = max<sub>trained</sub> &lt;e, t&gt; &minus; max<sub>novel</sub> &lt;e, t&gt;", EQ),
    P("The text margin term is what makes this work. An image-only novelty score was not robust to the "
      "distribution shift documented in Sec. 5, and a gate built on it was worth approximately nothing; "
      "adding the text margin raised held-out gate AUC from 0.936 to 0.957 and was worth +0.57 real "
      "points, the single largest gain from any routing change we made.", BODY0),
    P("The threshold is a rank quantile, not a fixed confidence value: the top f = 0.60 of images by g "
      "are routed to the closed-set branch. The optimal f is not the population prior 0.5635, and it "
      "moved as the system improved; Sec. 4.6 derives why. Ejecting a trained-class "
      "image from the closed-set route costs its closed-set accuracy; catching a novel-class image "
      "gains only that route's conditional accuracy b. When b was near 10%, ejecting was a bad trade "
      "and the optimum sat at f = 0.72; once the external banks raised b, ejection became cheap and the "
      "optimum fell to 0.60, a difference of +0.60 real points between the two submissions. We flag "
      "that our records do not fully document the second submission's recipe, so this figure bounds "
      "the routing effect rather than isolating it. A calibrated decomposition "
      "with kept-accuracy 86.0% and b = 19.48% reproduces the measured 78.95 / 10.42 split exactly.", BODY0),
    P("4.6. What routing costs, and how f follows from it", H2),
    P("The choice of f is usually presented as a tuned hyper-parameter. It is not: it follows from a "
      "decomposition that is an identity rather than a fitted model. Write w<sub>T</sub> and "
      "w<sub>U</sub> for the population weights of Eq. 1; <i>s</i> for the fraction of trained-class "
      "images the gate keeps; <i>u</i> for the fraction of novel-class images it ejects; <i>a</i> for "
      "the closed-set route's accuracy on what it keeps; and <i>b</i> for the retrieval route's "
      "accuracy on what it receives. Because a routing error is unrecoverable &mdash; the true class is "
      "simply absent from the candidate set the image is scored against (Sec. 4.1) &mdash; every "
      "correct prediction must survive both stages, and", BODY0),
    P("Acc = w<sub>T</sub> &middot; <i>s</i> &middot; <i>a</i> + w<sub>U</sub> &middot; <i>u</i> "
      "&middot; <i>b</i>&nbsp;&nbsp;&nbsp;(5)", EQ),
    P("We verified this against every scored submission; it reproduces each to four decimal places. "
      "For the final configuration <i>s</i> = 86.71%, <i>u</i> = 74.49%, <i>a</i> = 88.0% and "
      "<i>b</i> = 26.0% give 51.443% against a measured 51.443%.", BODY0),
    figure(f"{FIG}/flow.png", 3.18,
           "<b>Figure 2.</b> Where the 35,665 evaluation images go, using the final routing rates. "
           "The two red streams are routed to the branch that does not contain their true class and "
           "are therefore wrong before either route runs: 2,671 trained-class images ejected plus "
           "3,971 novel-class images kept, 18.6% of the evaluation set. This is the <i>s</i> and "
           "<i>u</i> of Eq. 5 drawn to scale, and it is why routing errors dominate the error budget "
           "that Sec. 7 analyses."),
    P("The same decomposition fixes f. Consider moving the threshold so as to eject one more image, "
      "and let q be the probability that this marginal image is novel. Ejecting it gains q&middot;b "
      "and loses (1&nbsp;&minus;&nbsp;q)&middot;a, so ejecting is worthwhile exactly while", BODY0),
    P("q / (1 &minus; q) &ge; a / b,&nbsp;&nbsp; equivalently &nbsp; q &ge; q* = a / (a + b)"
      "&nbsp;&nbsp;&nbsp;(6)", EQ),
    P("This explains a fact that had puzzled us for most of the project: the optimal f is nowhere near "
      "the population prior 0.5635. At the pre-retrieval configuration <i>a</i> = 86.0% and "
      "<i>b</i> = 19.5% give q* = 81.5% &mdash; four in five marginal images must genuinely be novel "
      "before ejecting one pays &mdash; which is why f sat at 0.72. External retrieval (Sec. 6) raised "
      "<i>b</i> to 26.0%, lowering q* to 77.2% and making ejection cheaper, and the optimum moved to "
      "f = 0.60. So the routing change was not an independent discovery: it is a consequence of the "
      "retrieval work, and the two should not be counted as separate contributions. One caveat on the "
      "+0.60 real points that accompanied the move: our records do not fully document the later "
      "build's recipe, so that figure bounds the routing effect rather than isolating it. Figure 3 "
      "plots the relation.", BODY0),
    figure(f"{FIG}/marginal.png", 3.18,
           "<b>Figure 3.</b> The purity q* that a marginal image must have before ejecting it pays, "
           "as a function of retrieval-route accuracy b (Eq. 6, at a = 88%). Improving the retrieval "
           "route moves the system down this curve, which is what licensed the f = 0.72 to 0.60 move. "
           "The curve is steep at low b, which is why the routing fraction was insensitive to tuning "
           "before external retrieval existed and sensitive to it afterwards."),
    P("4.7. Transductivity, stated plainly", H2),
    P("The final evaluated system is batch-coupled in five independent places, not the two or three a "
      "reader would infer from the equations. The rank threshold of Sec. 4.5 imposes the trained/novel "
      "mix globally rather than deciding per image, and is the strongest of the five. The first term of "
      "Eq.&nbsp;3 is a softmax down each class column across all 35,665 evaluation images. The Sinkhorn "
      "column prior of Sec. 4.4 couples rows explicitly. Less visibly, the closed-set ensemble "
      "standardizes each member's score matrix over the batch before summing, and the gate standardizes "
      "both of its components over the batch before combining them; in both cases an affine rescaling "
      "that is harmless within one term changes the <i>relative</i> weight of terms in a sum. A "
      "prediction therefore depends on the other 35,664 images. This uses no forbidden metadata "
      "&mdash; the class partition comes from training labels and the gate from image and text content "
      "&mdash; but it is not a per-image classifier, and we think papers in this setting should say so.", BODY0),
    P("<b>How much does the hub correction need the test batch?</b> The across-image term of Eq. 3 "
      "expands to s<sub>ic</sub>/0.05 &minus; h<sub>c</sub>, where h<sub>c</sub> = logsumexp over "
      "images of s<sub>&middot;c</sub>/0.05 is a per-class hub offset and is the only place the batch "
      "enters. Nothing requires h<sub>c</sub> to be estimated on the test batch. We recomputed it on "
      "reference image pools whose classes are disjoint from the 11,598 novel classes, froze it, and "
      "rebuilt the final configuration with every other component byte-identical. All four runs "
      "reproduce the shipped predictions exactly before the swap, so the harness is not in question.", BODY0),
    tbl([["reference pool for h<sub>c</sub>", "n", "changed", "of which<br/>can change<br/>accuracy"],
         ["withheld pseudo-novel images", "2,318", "296", "224"],
         ["&nbsp;&nbsp;even half", "1,159", "300", "226"],
         ["&nbsp;&nbsp;odd half", "1,159", "297", "223"],
         ["trained-class images", "4,000", "480", "357"]],
        [1.30, 0.44, 0.52, 0.66],
        "<b>Table 3.</b> Freezing the hub offset on a reference pool instead of the test batch, against "
        "the 35,665 shipped predictions. Only images the gate ejects to the novel route can change at "
        "all (14,266 of them), and only the novel-class subset of those can change accuracy: a "
        "trained-class image on the novel route is wrong either way. The two disjoint halves of the "
        "pseudo pool disagree with <i>each other</i> on 64 images, so the estimator has converged well "
        "before n = 1,159 and the residual disagreement with the transductive build is systematic, not "
        "sampling noise. Pool composition matters more than pool size.",
        align={1: "RIGHT", 2: "RIGHT", 3: "RIGHT"}, fs=7.0),
    P("It is worth separating two questions the table could be read as answering. The predictions are "
      "nearly interchangeable; the <i>offsets</i> are not. Across the ten normalized legs, the rank "
      "correlation between a reference-pool offset vector and the test-batch one ranges from 0.39 to "
      "0.98 (median 0.86) for the pseudo-novel pool and from 0.37 to 0.91 (median 0.69) for the "
      "trained-class pool. The reason is the temperature: at 0.05 the across-image softmax is "
      "concentrated, and a median class draws its offset from an effective sample of about 90 of the "
      "35,665 images. So hub structure is only partly a property of the class embedding, and any "
      "pool-based estimate of it is noisy. What the experiment shows is not that the offsets are "
      "recoverable but that the fused decision does not need them recovered precisely: ten legs are "
      "summed, and per-class errors that survive in each leg do not survive the sum. The term still "
      "matters in aggregate &mdash; deleting it costs 1.60 novel-class points &mdash; while its "
      "per-class values are worth 296 predictions.", BODY0),
    P("The reading we take is that this coupling is removable. An inductive build differs from the "
      "transductive one on 296 of 35,665 predictions, and at most 224 of those can move the score, "
      "bounding the difference at 0.63 overall points. The realistic figure is far below that "
      "ceiling: those 224 are 1.44% of the novel-class images, and novel-class accuracy is 20.57%, "
      "so the large majority of them were already wrong before the swap and change from one wrong "
      "label to another. We report the bound rather than the number because scoring the inductive "
      "build costs one of thirty evaluations. The other four "
      "couplings remain: an earlier configuration froze the threshold to an absolute constant with no "
      "prediction changes at all, but the Sinkhorn prior has no per-image form, and we do not claim "
      "the system as a whole is inductive.", BODY0),
    P("4.8. Provenance of the constants", H2),
    P("Eqs. 2 to 4 and Table 2 contain a number of bare constants: the exemplar weight 2.0 and "
      "taxonomic anchor 4.0 in the closed-set score, the gate's text weight 2.0, the two dbnorm "
      "temperatures, and the fusion weights. These were chosen by coordinate search on the proxy of "
      "Sec. 8.1 and are not derived from anything. We state this plainly because Sec. 8 is an argument "
      "that this proxy is unreliable, and it would be inconsistent to present constants tuned on it as "
      "though they were principled. Three of them do have measured support: removing the "
      "across-image term of Eq. 3 costs 1.60 novel-class points, the gate's text-margin term is worth "
      "+0.57 real points, and the pooling depth of four was selected against a measured curve "
      "(Sec. 6). The remainder are working settings, and we make no claim that they are optimal. "
      "Two constants have a different and worse provenance and should not be described as "
      "validation-tuned: the routing fraction f and the Sinkhorn temperature were set from real "
      "leaderboard feedback, because the proxy ranks them in the wrong order (Sec. 8.3). "
      "Sec. 11 counts that as adaptation to the evaluation set.", BODY0),
    P("4.9. Training", H2),
    P("The contrastive adapter is trained with softmax cross-entropy against frozen text embeddings "
      "&mdash; the CLIP formulation with the text side held fixed:", BODY0),
    P("L = &minus; log [ exp(30 cos(f, t<sub>y</sub>)) / &Sigma;<sub>c=1</sub><sup>4636</sup> "
      "exp(30 cos(f, t<sub>c</sub>)) ]&nbsp;&nbsp;&nbsp;(7)", EQ),
    P("Three details are easy to get wrong. There is <i>no margin term</i>: this is not CosFace or "
      "ArcFace, only their margin-free ancestor, and the scale 30 is a fixed constant rather than "
      "CLIP's learned and clamped <i>logit_scale</i>. The denominator runs over all 4,636 classes "
      "used for training at every step, not over in-batch negatives; there is no text-to-image term "
      "and the loss is not symmetric InfoNCE. And 4,636 is a design decision, not the label space: "
      "the rarest 20% of trained classes (1,159 of 5,795, holding 2,318 of the 64,259 images) are "
      "withheld from training entirely to serve as a pseudo-novel validation signal, so the trained "
      "encoder builds prototypes for 1,159 classes it never saw a gradient from. Those classes remain "
      "on the closed-set route at inference (Sec. 4.2). The alternative &mdash; tuning on the withheld "
      "classes and then retraining on all 5,795 &mdash; is the standard protocol and would very likely "
      "be better; we did not run it, because a retrain plus a leaderboard evaluation was outside the "
      "remaining budget.", BODY0),
]

# ================================================================== 5 shift
story += [
    P("5. Framing Shift: Diagnosis and Two Corrections", H1),
    P("Early in the project the closed-set route underperformed its held-out accuracy by roughly eight "
      "points on the real evaluation, and no amount of ensembling closed the gap. The cause was "
      "geometric rather than semantic. Standard <i>RandomResizedCrop</i> augmentation sampled crop "
      "aspect ratios uniformly in [0.75, 1.33]. We measured the actual aspect distribution of both "
      "evaluation splits (Fig. 4) and found that this range fails to cover even the training median "
      "of 1.46, and badly fails the novel split, whose mean aspect is 1.92 and whose 90th percentile "
      "is 2.76.", BODY0),
    figure(f"{FIG}/aspect.png", 3.18,
           "<b>Figure 4.</b> Measured aspect-ratio distributions (p10&ndash;p90 bars, circles at the "
           "median, red ticks at the mean) against the augmentation ranges. The original sampled range "
           "(red band) does not cover the training median, let alone the novel split, whose fish are "
           "markedly more elongated. The corrected range (blue band) covers all three splits."),
    P("This is a mundane bug with a specific consequence: a short-side resize followed by a center crop "
      "discards the head or tail of an elongated fish, and the network was never trained on that "
      "framing. Fish are elongated; the augmentation defaults inherited from ImageNet-style pipelines "
      "assume they are not. We corrected it twice, independently.", BODY0),
    P("<b>Training-time correction.</b> Widening the sampled range to scale (0.35, 1.0) and ratio "
      "(0.5, 2.0), and extracting with an aspect-squashing test-time view, improved both encoders on "
      "held-out data and was worth approximately +0.4 real points (Table 4). The corrected contrastive "
      "adapter also improved the novel-class route substantially on its own: 31.10 versus 26.62 proxy "
      "when used alone, and +2.07 when substituted into the text stack."),
    tbl([["configuration", "proxy", "corrected", "&Delta;"],
         ["LoRA adapter (224 px)", "86.10", "86.70", "+0.60"],
         ["full fine-tune (336 px)", "83.60", "84.17", "+0.57"],
         ["contrastive adapter, alone", "26.62", "31.10", "+4.48"],
         ["novel-class text stack", "29.38", "31.45", "+2.07"]],
        [1.42, 0.60, 0.68, 0.52],
        "<b>Table 4.</b> Effect of correcting the augmentation aspect range. The first two rows are "
        "closed-set held-out accuracy; the last two are novel-class proxy accuracy. Worth roughly "
        "+0.4 points real in aggregate.",
        align={1: "RIGHT", 2: "RIGHT", 3: "RIGHT"}),
    P("<b>Inference-time correction.</b> The same insight applies at test time and is separable from "
      "the first. We score each query under seven geometries &mdash; a center crop, a full-image squash, "
      "and five overlapping square windows tiled along the long axis &mdash; and take the "
      "<i>maximum</i> similarity across geometries before normalization. The pooling rule matters more "
      "than the geometry set: averaging across views lost, because a view that crops away the "
      "discriminative region contributes a confidently wrong vector that the mean cannot discount. "
      "Max-over-views on raw cosine, followed by a single normalization, gained +1.424 proxy and +0.174 "
      "real (Sec. 8 explains why those two numbers are so far apart).", BODY0),
    P("Two honest caveats. Applying the identical trick a second time at 336-pixel resolution gained "
      "+0.043 proxy, far below the +0.3 submission bar then in force, so the correction does not "
      "compound. And on "
      "square images the five long-axis windows degenerate to one, so the maximum is taken over three "
      "effective views rather than seven; we did not measure the subset of evaluation images for which "
      "this occurs."),
]

# ================================================================== 6 banks
story += [
    P("6. External-Bank Retrieval for Novel Classes", H1),
    P("For 11,598 classes with no training images, the only in-domain signal is a scientific name. "
      "Text-only novel-class accuracy plateaued at 10.42% real, and Sec. 9 lists eight separate "
      "attempts to raise it through better text that all failed. The change that worked was to stop "
      "improving the text and add pixels from outside the competition.", BODY0),
    P("<b>Source and coverage.</b> We assembled photographs from the iNaturalist Open Data archive "
      "[12] on public object storage, streamed rather than API-fetched after live API access rate-limited "
      "us, supplemented by targeted API queries to densify thin classes. Scientific-name matching "
      "succeeded for 12,738 of 12,757 candidate names (99.9%). The download holds 117,225 photographs "
      "over 7,364 classes, capped at 24 per class, of which 101,545 are embedded in the final "
      "336-pixel bank. Coverage of the novel vocabulary was 51.8% (6,011 of 11,598) at first build and "
      "grew with successive fetches; a merged variant reached 57.9% but measured proxy-negative and "
      "was never adopted, which is itself an instance of Sec. 8.", BODY0),
    P("Coverage is the binding constraint, and it is also where an earlier attempt died: prototypes "
      "built from a different biodiversity aggregator reached only 13% coverage and scored 6.99 proxy "
      "against 23.38 for text alone. The lesson is that the idea was never the problem &mdash; density "
      "and domain match were.", BODY0),
    P("<b>Scoring rule.</b> We keep individual photograph embeddings rather than collapsing each class "
      "to a prototype, and score a class by the mean of its top-4 cosine similarities. This sits "
      "deliberately between a prototype and a nearest neighbour, and the choice was measured rather "
      "than assumed: a plain class mean scores 43.49, a single nearest photograph 44.61, and pooling "
      "the top 2, 3 and 4 gives 44.78, 45.13 and 45.25. The curve flattens at four because many "
      "classes hold only a handful of photographs, so a deeper pool re-averages the same images. "
      "Separately, true mean prototypes over a "
      "second encoder (BioCLIP&nbsp;2 ViT-L/14) contribute an additional leg, merged at "
      f"{ALPHA} = 0.5 with prototypes computed from a large tree-of-life embedding archive.", BODY0),
    P("<b>Near-duplicate audit.</b> The obvious objection to any external-image result is that the "
      "system is retrieving the evaluation images themselves. An earlier version of this work tested "
      "the bank against <i>training</i> queries, which does not address that objection; we therefore "
      "compared all 35,665 evaluation embeddings against the primary bank's 101,106 photograph "
      "embeddings directly, in the encoder that bank leg actually uses (the two banks embed the same "
      "photograph set under different encoders, with slightly different counts after load failures), "
      "recording the maximum similarity per evaluation image.", BODY0),
    P("A threshold needs calibrating before those numbers mean anything, and two reference populations "
      "bracket it. Genuinely different photographs of the same species, measured within the bank, reach "
      "0.922 at the 99th percentile and 0.973 at the 99.9th, with only 0.302% of pairs above 0.95. The "
      "same physical image under two crop geometries sits at 0.917 in the median, with 34.2% of pairs "
      "above 0.95. A similarity above 0.95 is therefore an outlier for distinct photographs and "
      "unremarkable for one image seen twice, which makes it a defensible near-duplicate flag.", BODY0),
    tbl([["population", "n", "p50", "&gt;0.95", "max"],
         ["<i>cal.</i>: same species, diff. photo", "536,318", "0.684", "0.302%", "1.000"],
         ["<i>cal.</i>: same image, diff. crop", "150,000", "0.917", "34.2%", "1.000"],
         ["eval, trained-class vs bank", "20,097", "0.779", "0.219%", "0.9995"],
         ["<b>eval, novel-class vs bank</b>", "<b>15,568</b>", "<b>0.799</b>", "<b>0.906%</b>", "<b>0.998</b>"]],
        [1.52, 0.54, 0.40, 0.48, 0.46],
        "<b>Table 5.</b> Near-duplicate audit of the evaluation set against the external bank, with the "
        "two calibration populations that fix the threshold. The trained-class split sits below the "
        "same-species chance rate; the novel-class split sits about three times above it.",
        align={1: "RIGHT", 2: "RIGHT", 3: "RIGHT", 4: "RIGHT"}, fs=6.9),
    P("The result is not the clean negative we had previously claimed. On the trained-class split 44 "
      "of 20,097 images (0.219%) exceed 0.95, which is below the 0.302% rate at which "
      "merely-same-species pairs do, so there is no evidence of duplication there. On the novel-class "
      "split the rate is 0.906%, roughly three times the chance baseline, with 48 images above 0.98 "
      "and a maximum of 0.998. A near-duplicate population exists, and it is concentrated exactly "
      "where the external bank does its work.", BODY0),
    P("What follows is a bound rather than a dismissal. Suppose every flagged image were free, "
      "classified correctly only because its twin sits in the bank: 141 novel-class and 44 "
      "trained-class images, so novel-class accuracy would be inflated by at most 0.91 points and "
      "overall accuracy by at most 0.52. Against the +10.15 points of novel-class accuracy we "
      "attribute to external retrieval, the worst case accounts for 8.9%. The retrieval result "
      "survives that bound; the unqualified claim that duplication plays no part does not, and we "
      "withdraw it. The stronger experiment &mdash; deleting flagged photographs from the bank and "
      "re-submitting &mdash; is the one our evaluation budget did not allow.", BODY0),
    tbl([["configuration", "proxy", "real overall"],
         ["text only, no bank", "31.45", "49.04"],
         ["alternative aggregator (13% coverage)", "6.99", "not built"],
         ["mean prototypes, w = 1.5", "41.11", "50.12"],
         ["mean prototypes, w = 2", "42.23", "50.32"],
         ["mean prototypes, w = 3", "43.53", "50.46"],
         ["top-4 pooled, w = 4", "45.17", "50.49"],
         ["+ second-encoder prototypes", "46.29", "50.57"],
         ["+ dual frozen and adapted", "46.81", "50.76"],
         ["+ archive-merged prototypes", "47.24", "50.77"],
         ["+ merge on both legs", "47.58", "50.84"],
         ["<i>routing f = 0.72 to 0.60</i>", "&mdash;", "<i>51.44</i>"],
         ["<b>+ crop views, 336 px bank</b>", "<b>49.01</b>", "<b>51.62</b>"]],
        [1.90, 0.58, 0.66],
        "<b>Table 6.</b> External-bank development. <i>Proxy</i> is held-out novel-class accuracy; "
        "<i>real</i> is overall leaderboard accuracy. The two columns are on different scales, which "
        "is the subject of Sec. 8. The italic row is not a bank change: the routing fraction moved "
        "between the last two builds, so the final real gain is not attributable to the bank alone. "
        "The alternative-aggregator row is an earlier attempt that failed on coverage, not on method.",
        align={1: "RIGHT", 2: "RIGHT"}, fs=7.0),
    figure(f"{FIG}/operating.png", 3.18,
           "<b>Figure 5.</b> Every scored submission in (trained, novel) accuracy space; dashed lines "
           "are iso-accuracy contours from Eq. 1 and the horizontal axis is reversed so that better is "
           "up and to the left. External retrieval moves the system almost vertically at fixed "
           "trained-class accuracy; the routing change then trades trained for novel along a contour "
           "gradient. Red circles are the two changes that lost real accuracy. The triangle is the "
           "contemporaneous leader, which we beat on the trained route and lost to on the novel route. "
           "Seven submissions share a trained-class accuracy of 78.95 and are bracketed rather than "
           "labelled individually; Table 7 lists them."),
    P("<b>Effect.</b> The external route is the single largest source of improvement in the project. "
      "Novel-class accuracy rose from 10.42% (text only) through 13.68% (mean prototypes) to 20.57% at "
      "the current configuration, and overall accuracy from 49.04% to 51.62%. Figs. 5 and 6 show that "
      "essentially all progress after the closed-set route saturated came from this branch."),
    figure(f"{FIG}/progress.png", 3.18,
           "<b>Figure 6.</b> Real evaluation accuracy across confirmed submissions. Top: overall top-1, "
           "against the contemporaneous visible leaderboard leader and the project target. Bottom: "
           "novel-class top-1, which is where every gain after v33 originates. The two annotated "
           "inflections are the introduction of external retrieval and the routing-fraction change."),
    tbl([["id", "change", "trained", "novel", "overall"],
         ["v22", "closed-set, shift-robust ensemble", "&mdash;", "&mdash;", "45.39"],
         ["v23", "combined image + text gate", "&mdash;", "&mdash;", "45.96"],
         ["v25", "+ shift-retrained adapter", "&mdash;", "&mdash;", "46.27"],
         ["v27", "+ 336 px fine-tune", "&mdash;", "&mdash;", "46.34"],
         ["v28", "routing f = 0.65", "&mdash;", "&mdash;", "47.26"],
         ["v31", "routing f = 0.72, per-image", "&mdash;", "&mdash;", "47.69"],
         ["v33", "+ Sinkhorn on novel route", "78.20", "8.51", "47.78"],
         ["v34", "recover-on-eject", "78.63", "7.80", "47.71"],
         ["v36", "+ corrected adapter, TaxaBind", "78.95", "10.42", "49.04"],
         ["v37", "+ external mean prototypes", "78.95", "13.35", "50.32"],
         ["v37", "&nbsp;&nbsp;prototype weight 3", "&mdash;", "13.68", "50.46"],
         ["v39", "TaxaBind image prototypes", "78.95", "13.44", "50.35"],
         ["v40", "top-4 pooled bank", "78.95", "13.76", "50.49"],
         ["v41", "+ BioCLIP&nbsp;2 prototypes", "78.95", "13.93", "50.57"],
         ["v43", "+ dual frozen / adapted", "78.95", "14.36", "50.76"],
         ["v44", "+ archive merge", "78.95", "14.39", "50.77"],
         ["v46", "merge on both legs", "78.95", "14.55", "50.84"],
         ["v50", "routing f = 0.60", "76.31", "19.34", "51.44"],
         ["<b>v56</b>", "<b>+ crop views, 336 bank</b>", "<b>75.67</b>", "<b>20.57</b>", "<b>51.62</b>"]],
        [0.34, 1.62, 0.48, 0.42, 0.50],
        "<b>Table 7.</b> Complete ledger of scored real evaluations. Rows v34 and v39 are the two "
        "changes that cleared the proxy bar and then lost real accuracy (Sec. 8). Earlier rows predate "
        "the per-subset reporting we later adopted. Trained-class accuracy is flat at 78.95 from v36 "
        "to v46 because every change in that span touched only the novel route.",
        align={2: "RIGHT", 3: "RIGHT", 4: "RIGHT"}, fs=6.9),
]

# ================================================================== 7 ceiling
story += [
    P("7. Where the Ceiling Lies", H1),
    P("Most of our engineering went into routing and fusion. Oracle probes indicate that was the wrong "
      "place, and we report the probes because they are the argument.", BODY0),
    P("<b>The routing gate is near its useful limit.</b> Setting s = u = 1 in Eq. 5 gives the "
      "perfect-gate ceiling, with one correction: <i>a</i> and <i>b</i> are conditional on the images "
      "each route currently receives, and a perfect gate would also hand each route the harder images "
      "it presently discards. Substituting the population-level accuracies measured at that "
      "configuration (81.2% and 17.8% in place of 86.0% and 19.5%) gives 53.5%, against 49.0% as "
      "actually routed. That is the entire prize for solving routing, and it is barely above the 53% "
      "target. Our achievable gate reaches "
      "AUC 0.9084 on real evaluation folders; hitting the 53% target at current b would require "
      "roughly 0.97. Every alternative gate we built (multilayer perceptron, RankNet, Mahalanobis, "
      "k-nearest-neighbour, full-margin variants) failed to beat the simple standardized sum of Eq. 4 "
      "by more than noise.", BODY0),
    tbl([["gate", "signal", "AUC", "real"],
         ["smooth novelty, gamma = 30", "image", "&mdash;", "45.19"],
         ["smooth novelty, gamma = 10", "image", "&mdash;", "45.19"],
         ["none (pure closed-set)", "&mdash;", "&mdash;", "45.39"],
         ["<b>combined (final)</b>", "<b>image + text</b>", "<b>0.957</b>", "<b>45.96</b>"],
         ["best alternative variant", "image + full margin", "0.912", "not built"],
         ["required for 53% target", "&mdash;", "~0.97", "&mdash;"]],
        [1.42, 1.02, 0.44, 0.44],
        "<b>Table 8.</b> Routing-gate ablation. AUC values are held-out; the final real AUC on "
        "evaluation folders is 0.9084. Two settings of the original image-only gate tie to the decimal, "
        "which is what first indicated the signal carried no information. Adding the text margin was "
        "worth +0.57 real points and remains the largest single routing gain.",
        align={2: "RIGHT", 3: "RIGHT"}, fs=7.0),
    figure(f"{FIG}/encoders.png", 3.18,
           "<b>Figure 7.</b> Encoder comparison on the novel-class proxy. Above the rule: frozen "
           "encoders, which is the comparison that matters for a zero-training-image class. Domain "
           "pretraining dominates scale &mdash; a strong general-purpose vision-language model scores "
           "5.13 and a general CLIP 0.78, against 23.38 for the biology encoder. The taxonomic-context "
           "prompt (grey) consistently hurts frozen encoders, which is why both prompt styles ship as "
           "separate legs. Below the rule (blue): the fine-tuned adapters actually adopted, shown for "
           "contrast &mdash; correcting the augmentation geometry (Sec. 5) moved this leg from 26.62 to "
           "31.10, a larger gain than any change of frozen backbone delivered."),
    P("<b>Candidate ambiguity is not what remains.</b> The decisive experiment grants the novel route "
      "oracle family knowledge, collapsing 11,598 candidates to roughly two dozen. Novel-class top-1 "
      "rises only from 28.90% to 33.26%, a gain of 4.36 points from perfect taxonomic information. "
      "Removing nearly all candidate ambiguity therefore leaves the great majority of the residual "
      "error untouched, which bounds what any better candidate-narrowing scheme can deliver. We "
      "stress what this does <i>not</i> show: it does not isolate the encoder. The residual could "
      "equally reflect prototype quality, bank coverage, the domain gap between bank photographs and "
      "evaluation images, or score calibration, and this experiment separates none of them. "
      "Relatedly, our family-level top-1 (23.8%) is lower than our species-level top-1 (28.9%) and "
      "hard family-then-species narrowing costs 20 points &mdash; evidence that the family-scoring "
      "procedure we built is unusable as a prior, not that taxonomy is uninformative in principle.", BODY0),
    figure(f"{FIG}/ceiling.png", 3.18,
           "<b>Figure 8.</b> Left: the final system against the perfect-gate oracle and the target. "
           "Even a flawless router leaves the target barely reachable, so routing is not where the "
           "remaining accuracy is. Right: the oracle-family probe. Collapsing 11,598 candidates to "
           "roughly two dozen by granting perfect taxonomic knowledge buys only 4.36 points, which "
           "bounds what any better candidate-narrowing scheme can deliver."),
    P("<b>Read-through to the leaderboard.</b> At the time of measurement the visible leader scored "
      "50.56% with 78.05% trained-class and 15.08% novel-class accuracy, against our 78.95% and 14.55%. "
      "We were ahead on the closed-set route and behind on the novel route, so on this single snapshot "
      "the difference sits in b rather than in the operating point. We report it as a consistency "
      "check on our own error attribution, not as a decomposition of the other system, whose method "
      "we do not know.", BODY0),
]

# ================================================================== 8 proxy
story += [
    P("8. Proxy Reliability", H1),
    P("<b>The proxy.</b> Novel-class measurements are made on a pseudo-novel split: the 1,159 rarest trained classes are withheld from training entirely (Sec. 4.9), giving 2,318 query images scored against the full 12,757-name candidate list, so that uncovered classes act as distractors. Class-disjoint cross-validation over 954 held-out classes guards the fusion weights. This is a careful proxy, not a careless one, which is the point of what follows.", BODY0),
    P("Every change below cleared the proxy bar of Sec. 3 before being spent on a real evaluation, so this "
      "is not a comparison of good changes against bad ones &mdash; all of them looked good. The "
      "question is only how much of each proxy gain survived contact with the real distribution.", BODY0),
    figure(f"{FIG}/transfer.png", 3.18,
           "<b>Figure 9.</b> Proxy gain against real gain. <b>(a)</b> Every paired measurement we "
           "have, zoomed to the dense region; perfect transfer would place all points on one line "
           "through the origin, and instead they scatter across the plotted slope range while two "
           "changes fall below the axis. One further point, v37a, lies off-scale at (9.66, 1.08). "
           "<b>(b)</b> The same data as transfer rates. These rates are measured against different "
           "proxy definitions and are indicative rather than strictly commensurable, which is itself "
           "part of the problem being reported."),
    P("<b>Magnitude is not preserved.</b> The transfer ratio spans a factor of twenty (Fig. 9b). "
      "Encoder-level changes that both branches share transfer efficiently (0.359 for a dual-encoder "
      "prototype leg); changes that touch only the external-retrieval branch transfer poorly (0.018 for "
      "a pooling-statistic change). A separate and starker instance: a Sinkhorn variant that improved "
      "held-out accuracy by 8.8% relative delivered 2.4% relative on the real evaluation, an "
      "overstatement of 3.7 times.", BODY0),
    P("<b>Sign is not preserved.</b> Two changes cleared the bar and then moved the real score the "
      "wrong way. A recovery rule that re-scored ejected images gained 3.51 proxy points and lost 0.07 "
      "real points, because the trained-class accuracy it recovered (+0.43) cost more novel-class "
      "accuracy (&minus;0.71) than it returned. A TaxaBind-based external prototype leg gained 0.52 "
      "proxy and lost 0.11 real, on a proxy where a BioCLIP&nbsp;2 leg of identical construction "
      "transferred positively.", BODY0),
    P("<b>Ordering is not preserved &mdash; the strongest case.</b> We evaluated a simpler unified "
      "architecture that discards the routing split entirely and scores every image with one function. "
      "It is the design most reviewers would ask for. On held-out data it was far ahead. The anchor "
      "here is exact rather than estimated: the reference arm reproduces a scored real submission on "
      "35,665 of 35,665 predictions, so its conditional accuracies are calibrated on the real result "
      "itself.", BODY0),
    tbl([["arm", "proxy", "real", "verdict"],
         ["routed, f = 0.60", "50.71", "51.44", "final"],
         ["routed, f = 0.72", "55.80", "50.84", "worse in reality"],
         ["unified, vision&ndash;text", "57.02", "&le; 49.35", "not submitted"],
         ["unified, + prototypes", "55.82", "&le; 45.97", "not submitted"],
         ["unified, + reference leg", "56.86", "&le; 46.88", "not submitted"]],
        [1.28, 0.55, 0.60, 0.85],
        "<b>Table 9.</b> The proxy ranks f = 0.72 above f = 0.60 by 5.09 points; the real evaluation "
        "reverses this by 0.60. Real columns for unified arms are optimistic upper bounds, since they "
        "credit each arm with the final system's conditional accuracies. A proxy that inverts the "
        "one axis with two exact real anchors cannot adjudicate the arms that have none.",
        align={1: "RIGHT", 2: "RIGHT"}),
    P("The failure has a structural explanation rather than a tuning one. The two routes are not "
      "redundant: the closed-set branch's three-encoder ensemble has no counterpart on the novel side, "
      "and the external bank covers only 966 of 5,795 trained classes. Any single scoring function "
      "either discards the ensemble or lets asymmetric image legs swamp the trained-class side. The "
      "best unified arm matched the final system's novel-class routing to within 0.2 points and still gave up "
      "9.1 points of trained-class routing. Closing that would require re-embedding the external bank "
      "under all three closed-set encoders, which is new computation rather than a hyperparameter."),
    P("<b>Two implementation traps.</b> We record these because both produced confidently wrong numbers "
      "before being caught. Masking uncovered classes to a large negative constant is harmless inside "
      "a dedicated branch but deletes them from a unified candidate space. And adding a normalized leg "
      "to only one side of a partitioned score injects a constant per-row offset that decides the "
      "argmax by itself; one of our intermediate results was entirely this bug."),
    P("<b>What we conclude.</b> Not that held-out proxies are useless &mdash; we used one for every "
      "decision in this paper, and not that this generalizes beyond the trajectory we ran; ten paired "
      "measurements from one adaptively-tuned development history on one benchmark cannot establish a "
      "law. What we can say is narrower: in this trajectory transfer tracked the <i>kind</i> of "
      "intervention rather than the size of the proxy gain, and it degraded where the proxy "
      "distribution differ. In our case that is the external-retrieval branch, whose photographs come "
      "from a different imaging distribution than the evaluation set, and the routing policy, which "
      "the proxy cannot exercise because the proxy has no distribution shift to route around. Under a "
      "constrained budget the correct allocation is to spend real evaluations on exactly those axes and "
      "trust the proxy on encoder-level changes."),
]

# ================================================================== 9 negative
story += [
    P("9. Catalogue of Negative Results", H1),
    P("We list these compactly because the aggregate is the point: twenty measured failures, several "
      "of which are the first thing a reader would suggest.", BODY0),
    tbl([["text representation", "base", "score"],
         ["taxonomic context added as a leg", "28.90", "<b>29.47</b>"],
         ["generic prompt ensemble added", "28.90", "28.00"],
         ["ensemble replacing the binomial", "28.90", "24.94"],
         ["bare binomial (frozen encoder)", "&mdash;", "23.38"],
         ["taxonomic context (frozen encoder)", "23.38", "21.53"],
         ["LLM-compressed trait text", "23.38", "17.08"],
         ["raw provided descriptions", "23.38", "7.51"],
         ["bare binomial (adapter)", "&mdash;", "26.62"],
         ["taxonomic context alone (adapter)", "26.62", "24.63"],
         ["generic prompt ensemble (adapter)", "26.62", "22.00"],
         ["common names", "26.62", "8.89"],
         ["free-text descriptions as a leg", "27.83", "12.25"],
         ["Wikipedia summaries (0.27% cov.)", "31.23", "31.32"]],
        [1.86, 0.55, 0.62],
        "<b>Table 10.</b> Text-representation ablation. Scores are only comparable within a shared "
        "<i>base</i>; the table groups them accordingly, because these numbers are easy to mix across "
        "baselines. The pattern is consistent: every paraphrase of the scientific name loses to the "
        "scientific name. Taxonomic context wins only as an <i>additional</i> leg on a matched "
        "fine-tuned encoder (row 1), and loses whenever it replaces the binomial or is applied to a "
        "frozen encoder.",
        align={1: "RIGHT", 2: "RIGHT"}, fs=7.0),
    tbl([["attempt", "result"],
         ["General backbone (SigLIP2 SO400M)", "5.13 vs 21.53"],
         ["BioTrove-CLIP (two variants)", "0.00 / 0.04"],
         ["BioCAP", "6.77 vs 21.53"],
         ["Hyperbolic BioCLIP variant", "4.36"],
         ["BioCLIP-1 as second leg", "+1.86, does not stack"],
         ["Common-name prompts", "8.89 vs 26.62"],
         ["Generic prompt ensembling", "22.00 vs 26.62"],
         ["Free-text descriptions", "12.25 vs 26.62"],
         ["LLM-compressed trait text", "17.08 vs 23.38"],
         ["Wikipedia summaries", "35/12,757 coverage"],
         ["Taxonomic hierarchy prior", "family 23.8 &lt; species 28.9"],
         ["Hard two-stage family narrowing", "&minus;20 points"],
         ["Vision-language model top-K rerank", "26.33 &rarr; 10.33"],
         ["Alternative aggregator prototypes", "6.99, 13% coverage"],
         ["SigLIP2 / DINOv2 external prototypes", "+0.22 / +0.00"],
         ["Second same-backbone prototype leg", "correlated, no gain"],
         ["Extended / low-LR adapter training", "below initialization"],
         ["Rank-32 adapter", "23.94 vs 24.76"],
         ["Hard-negative mining", "14.06, worst variant"],
         ["Soft (non-hard) routing", "proxy +1.9, routing degrades"]],
        [1.72, 1.00],
        "<b>Table 11.</b> Measured negative results. Several are worth reading as a group: rows 6&ndash;10 "
        "are all attempts to improve novel-class text and all fail in the same direction, which is what "
        "eventually motivated the pixel-based approach of Sec. 6.",
        align={1: "RIGHT"}, fs=7.0),
    P("<b>The adapter-selection inversion.</b> One result deserves separate mention because it "
      "reproduces the theme of Sec. 8 inside a single component, and because the obvious reading of "
      "it is wrong in an instructive way. Seven adapters for the external-prototype encoder were "
      "scored standalone. Ranked by raw standalone score the best reaches 15.53 while the final "
      "one reaches only 14.50 &mdash; yet fused into the system the first contributes +0.00 and the "
      "second +0.43, which looks like a clean anti-correlation between component quality and system "
      "value. It is not. The two were trained against different text targets and so started from "
      "different baselines, 15.36 and 13.76 respectively. Measured as improvement over its own "
      "starting point the ordering reverses to +0.17 against +0.74, which agrees with the fused "
      "result. The lesson is narrower and more useful than \"component scores mislead\": a raw score "
      "compared across differently-initialized runs misleads, and the delta recovered the right "
      "answer where the level did not.", BODY0),
    tbl([["variant", "base", "best", "delta", "fused"],
         ["name text", "15.36", "15.53", "+0.17", "+0.00"],
         ["full-slice, warm init", "13.76", "15.01", "+1.25", "&mdash;"],
         ["full-slice, cold init", "14.15", "14.80", "+0.65", "&mdash;"],
         ["rank-32", "14.15", "14.67", "+0.52", "&mdash;"],
         ["<b>taxonomic ctx. (final)</b>", "<b>13.76</b>", "<b>14.50</b>", "<b>+0.74</b>", "<b>+0.43</b>"],
         ["hard-negative mining", "13.76", "14.06", "+0.30", "&mdash;"],
         ["no external photographs", "13.76", "13.63", "&minus;0.13", "&mdash;"]],
        [1.24, 0.44, 0.44, 0.46, 0.46],
        "<b>Table 12.</b> Adapter variants for the external-prototype encoder. Ranking by <i>best</i> "
        "puts the name-text variant first; ranking by <i>delta</i> over each run's own baseline puts "
        "the adopted variant first, and only the latter agrees with the fused contribution. A dash "
        "means the variant never cleared its gate, so no fusion measurement was made.",
        align={1: "RIGHT", 2: "RIGHT", 3: "RIGHT", 4: "RIGHT"}, fs=6.9),
]

# ================================================================== 10 impl
story += [
    P("10. Implementation Details", H1),
    tbl([["component", "setting", "component", "setting"],
         ["closed-set backbone", "BioCLIP-2.5 ViT-H/14 &times;3", "optimizer", "AdamW, weight decay 0"],
         ["prototype backbone", "BioCLIP&nbsp;2 ViT-L/14, d = 768", "learning rate", "5e-4, warmup + cosine"],
         ["adapter placement", "MLP c_fc, c_proj, top 16", "batch / steps", "128 / 800, bf16"],
         ["adapter rank / scale", "48 / 96 (ratio 2.0)", "logit scale", "30.0, fixed"],
         ["trainable params", "9,830,400", "augmentation", "scale (.35, 1) ratio (.5, 2)"],
         ["inference scale", "0.4, eff. 0.8 (0.4&times;)", "softmax targets", "4,636 known classes"],
         ["external bank", "117,225 photos / 7,364 cls", "withheld pseudo-novel", "1,159 rarest trained"],
         ["&nbsp;&nbsp;embedded (336 px)", "101,545", "training images", "61,941"],
         ["bank scoring", "mean of top-4 cosines", "routing fraction", "f = 0.60 (21,399)"],
         ["Sinkhorn", "tau = 1.8, 60 iterations", "inference views", "7 (center, squash, strips)"],
         ["dbnorm temperatures", "0.05 images, 0.5 classes", "&nbsp;", "&nbsp;"]],
        [0.76, 1.04, 0.78, 1.02],
        "<b>Table 13.</b> Final evaluated configuration. Shipped adapter hyperparameters differ from the "
        "defaults in our training script; the values here are read from the checkpoint metadata.",
        fs=6.6),
]

# ================================================================== 11 limits
story += [
    P("11. Threats to Validity", H1),
    P("<b>Adaptation to the evaluation set.</b> This is the most serious limitation in the paper and "
      "it touches every real number in it. It arises three separate ways. First, thirty sequential "
      "leaderboard evaluations, each informing the next, which is exactly the adaptive-query regime "
      "formalized by Dwork <i>et al.</i> [18, 19] and the setting the Ladder mechanism [20] exists to "
      "protect; we had no such protection. Second, the routing fraction was chosen using real feedback "
      "(Sec. 4.5). Third, and least defensible, the augmentation correction of Sec. 5 was derived by "
      "measuring the aspect-ratio distribution of the evaluation images themselves. No labels were "
      "touched, but it is transductive, and a reader is entitled to call it optimizing against the "
      "test distribution. The final 51.62% should therefore be read as the accuracy of a configuration "
      "selected with knowledge of this evaluation set, not as an unbiased estimate of generalization.", BODY0),
    P("<b>Effect sizes, and the absence of significance tests.</b> Every real number is a single "
      "evaluation of one configuration, so we report no confidence intervals. Several confirmed gains "
      "are also very small in absolute terms, which the percentage form disguises; Table 14 restates "
      "them as additional images answered correctly out of 35,665. The final change nets 62 images "
      "while altering 4,092 predictions, a yield of 1.5% on the predictions it touched. A paired test "
      "is the right instrument, but McNemar needs the discordant pairs split by direction and a "
      "reported accuracy recovers only their difference; the labels that would give us the split are "
      "not public. We therefore claim no statistical significance for any single-step comparison here, "
      "and the sub-0.2-point rows of Table 7 should be read as directional at best."),
    tbl([["comparison", "delta (pp)", "net images", "predictions changed"],
         ["v50 vs v46", "+0.603", "215", "not recorded"],
         ["v43 vs v41", "+0.186", "66", "not recorded"],
         ["v56 vs v50", "+0.174", "62", "4,092"],
         ["v46 vs v44", "+0.067", "24", "not recorded"],
         ["v44 vs v43", "+0.017", "6", "not recorded"]],
        [0.84, 0.62, 0.62, 1.06],
        "<b>Table 14.</b> Confirmed gains restated as images. Two of these five move fewer than 25 "
        "images out of 35,665. We give them in this form because the percentage form makes them look "
        "sturdier than they are.",
        align={1: "RIGHT", 2: "RIGHT", 3: "RIGHT"}, fs=7.0),
    P("<b>Sequential tuning and component count.</b> The routing fraction, the fusion weights and the "
      "bank weight were tuned one at a time against a proxy that Sec. 8 shows to be unreliable on "
      "exactly those axes, so the final point is not claimed to be jointly optimal. More broadly the "
      "system carries three closed-set encoders, six text legs, two image banks, two prototype sets, "
      "two normalizations and a routing rule. With that many interacting parts a chronological ledger "
      "of submissions (Table 7) is not a controlled factorial experiment, and we have tried to confine "
      "causal language to places where an ablation actually exists.", BODY0),
    P("<b>Batch coupling.</b> The final configuration is transductive at five points (Sec. 4.7). The "
      "rank threshold imposes a fixed 60/40 split on whatever batch it is given, so the method as "
      "evaluated does not classify a single incoming image and would behave differently on a batch "
      "with a different trained/novel mix. Sec. 4.7 removes one of the five &mdash; the hub offset "
      "&mdash; and measures the cost at 296 changed predictions with at most 224 able to move the "
      "score, but the remaining four are not addressed, the inductive build is not itself scored, and "
      "the threshold is the coupling that matters most. Relatedly, the Sinkhorn step imposes a near-uniform prior over 11,598 novel classes "
      "for 15,568 images, which the true label distribution certainly violates. It raised vocabulary "
      "usage from 38.6% to 57.5% and was worth +0.09 real points &mdash; inside the noise band of the "
      "paragraph above."),
    P("<b>Domain gap.</b> The audit of Sec. 6 bounds near-duplication; it does not show that bank "
      "photographs and evaluation images come from the same distribution. They demonstrably do not, "
      "and Sec. 8 argues that this gap is the main reason proxy measurements on that branch mislead. "
      "A held-out split built from the bank itself would test prototype construction without testing "
      "the gap, which is why we did not build one."),
    P("<b>Asymmetric augmentation.</b> Query images are scored under seven views with no horizontal "
      "flips while bank photographs are embedded with flips and squashing; we did not ablate this. As "
      "noted in Sec. 5, the five long-axis views collapse to one on square images."),
    P("<b>Cost.</b> The final configuration runs three ViT-H/14 encoders plus a ViT-L/14 and a "
      "ViT-B/16 over seven crop geometries per query, against 101,106 stored bank embeddings, followed "
      "by a normalization and a Sinkhorn projection over the entire evaluation batch. We did not "
      "measure throughput, memory or energy, which is a real gap in the reporting."),
    P("<b>Scope.</b> One benchmark, one taxonomic domain, one encoder family, and ten paired "
      "proxy-to-real measurements from a single development history. The proxy-reliability result is "
      "the finding we would most expect to generalize and the one we can least demonstrate "
      "generalizes; a second benchmark with a comparable real-evaluation budget is the experiment "
      "that would settle it, and we have not run it."),
    P("12. Conclusion", H1),
    P("We described a generalized zero-shot recognition system operating over 17,393 classes, two-thirds of which "
      "have no training images, and used a thirty-evaluation budget to measure rather than assume the "
      "things that mattered. Two ordinary corrections produced most of the gains: matching augmentation "
      "geometry to the actual aspect statistics of the evaluation imagery, and adding external "
      "photographs as a retrieval bank for classes that have no images of their own, which roughly "
      "doubled accuracy on the novel half of the label space. Oracle probes then located the remaining "
      "gap in the visual encoder rather than in the routing machinery that consumed most of our effort: "
      "perfect family-level knowledge is worth only four points, and the leaderboard gap to the leader "
      "was entirely novel-class accuracy at an operating point we already matched.", BODY0),
    P("The result we consider most useful to others is the negative one. A held-out proxy that cleared "
      "every configuration we shipped nonetheless compressed gains by up to twenty-fold, inverted the "
      "sign of two changes, and preferred by five points an architecture that the real evaluation "
      "rejected by half a point. Proxy reliability was not a fixed property of our validation setup; it "
      "degraded specifically along the axes where the proxy distribution and the evaluation "
      "distribution diverge. In budgeted zero-shot settings we would now spend real evaluations on those "
      "axes first and trust the proxy only where it is measuring something both distributions share.", BODY0),
]

# ================================================================== refs
REFS = [
    'W. J. Scheirer, A. de Rezende Rocha, A. Sapkota, and T. E. Boult. Toward open set recognition. '
    '<i>IEEE TPAMI</i>, 35(7):1757&ndash;1772, 2013.',
    'C. Geng, S.-J. Huang, and S. Chen. Recent advances in open set recognition: A survey. '
    '<i>IEEE TPAMI</i>, 43(10):3614&ndash;3631, 2021.',
    'Y. Xian, C. H. Lampert, B. Schiele, and Z. Akata. Zero-shot learning &mdash; a comprehensive '
    'evaluation of the good, the bad and the ugly. <i>IEEE TPAMI</i>, 41(9):2251&ndash;2265, 2019.',
    'A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, '
    'J. Clark, G. Krueger, and I. Sutskever. Learning transferable visual models from natural language '
    'supervision. In <i>ICML</i>, 2021.',
    'S. Stevens, J. Wu, M. J. Thompson, E. G. Campolongo, C. H. Song, D. E. Carlyn, L. Dong, W. M. Dahdul, '
    'C. Stewart, T. Berger-Wolf, W.-L. Chao, and Y. Su. BioCLIP: A vision foundation model for the tree '
    'of life. In <i>CVPR</i>, 2024.',
    'J. Gu, S. Stevens, E. Campolongo, M. Thompson, N. Zhang, J. Wu, A. Kopanev, Z. Mai, A. White, '
    'J. Balhoff, W. Dahdul, D. Rubenstein, H. Lapp, T. Berger-Wolf, W.-L. Chao, and Y. Su. BioCLIP 2: '
    'Emergent properties from scaling hierarchical contrastive learning. In <i>NeurIPS</i>, 2025.',
    'E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen. LoRA: Low-rank '
    'adaptation of large language models. In <i>ICLR</i>, 2022.',
    'H. Wang, Y. Wang, Z. Zhou, X. Ji, D. Gong, J. Zhou, Z. Li, and W. Liu. CosFace: Large margin cosine '
    'loss for deep face recognition. In <i>CVPR</i>, 2018.',
    'J. Deng, J. Guo, N. Xue, and S. Zafeiriou. ArcFace: Additive angular margin loss for deep face '
    'recognition. In <i>CVPR</i>, 2019.',
    'J. Snell, K. Swersky, and R. Zemel. Prototypical networks for few-shot learning. In <i>NeurIPS</i>, 2017.',
    'S. Sastry, S. Khanal, A. Dhakal, A. Ahmad, and N. Jacobs. TaxaBind: A unified embedding space for '
    'ecological applications. In <i>WACV</i>, 2025.',
    'G. Van Horn, O. Mac Aodha, Y. Song, Y. Cui, C. Sun, A. Shepard, H. Adam, P. Perona, and S. Belongie. '
    'The iNaturalist species classification and detection dataset. In <i>CVPR</i>, 2018.',
    'M. Cuturi. Sinkhorn distances: Lightspeed computation of optimal transport. In <i>NeurIPS</i>, 2013.',
    'M. Caron, I. Misra, J. Mairal, P. Goyal, P. Bojanowski, and A. Joulin. Unsupervised learning of '
    'visual features by contrasting cluster assignments. In <i>NeurIPS</i>, 2020.',
    'A. Lyzhov, Y. Molchanova, A. Ashukha, D. Molchanov, and D. Vetrov. Greedy policy search: A simple '
    'baseline for learnable test-time augmentation. In <i>UAI</i>, 2020.',
    'X.-S. Wei, Y.-Z. Song, O. Mac Aodha, J. Wu, Y. Peng, J. Tang, J. Yang, and S. Belongie. Fine-grained '
    'image analysis with deep learning: A survey. <i>IEEE TPAMI</i>, 44(12):8927&ndash;8948, 2022.',
    'F. F. Khan, X. Li, A. J. Temple, and M. Elhoseiny. FishNet: A large-scale dataset and benchmark for '
    'fish recognition, detection, and functional trait prediction. In <i>ICCV</i>, 2023.',
    '''C. Dwork, V. Feldman, M. Hardt, T. Pitassi, O. Reingold, and A. Roth. The reusable holdout: Preserving validity in adaptive data analysis. <i>Science</i>, 349(6248):636&ndash;638, 2015.''',
    '''C. Dwork, V. Feldman, M. Hardt, T. Pitassi, O. Reingold, and A. Roth. Preserving statistical validity in adaptive data analysis. In <i>STOC</i>, 2015.''',
    '''A. Blum and M. Hardt. The ladder: A reliable leaderboard for machine learning competitions. In <i>ICML</i>, 2015.''',
    '''R. Roelofs, V. Shankar, B. Recht, S. Fridovich-Keil, M. Hardt, J. Miller, and L. Schmidt. A meta-analysis of overfitting in machine learning. In <i>NeurIPS</i>, 2019.''',
    '''B. Recht, R. Roelofs, L. Schmidt, and V. Shankar. Do ImageNet classifiers generalize to ImageNet? In <i>ICML</i>, 2019.''',
    '''J. P. Miller, R. Taori, A. Raghunathan, S. Sagawa, P. W. Koh, V. Shankar, P. Liang, Y. Carmon, and L. Schmidt. Accuracy on the line: On the strong correlation between out-of-distribution and in-distribution generalization. In <i>ICML</i>, 2021.''',
    '''D. Teney, Y. Lin, S. J. Oh, and E. Abbasnejad. ID and OOD performance are sometimes inversely correlated on real-world datasets. In <i>NeurIPS</i>, 2023.''',
    '''W.-L. Chao, S. Changpinyo, B. Gong, and F. Sha. An empirical study and analysis of generalized zero-shot learning for object recognition in the wild. In <i>ECCV</i>, 2016.''',
    '''J. Song, C. Shen, Y. Yang, Y. Liu, and M. Song. Transductive unbiased embedding for zero-shot learning. In <i>CVPR</i>, 2018.''',
    '''M. Radovanovi&#263;, A. Nanopoulos, and M. Ivanovi&#263;. Hubs in space: Popular nearest neighbors in high-dimensional data. <i>JMLR</i>, 11:2487&ndash;2531, 2010.''',
    '''S. L. Smith, D. H. P. Turban, S. Hamblin, and N. Y. Hammerla. Offline bilingual word vectors, orthogonal transformations and the inverted softmax. In <i>ICLR</i>, 2017.''',
    '''S.-V. Bogolin, I. Croitoru, H. Jin, Y. Liu, and S. Albanie. Cross modal retrieval with querybank normalisation. In <i>CVPR</i>, 2022.''',
    '''A. Long, W. Yin, T. Ajanthan, V. Nguyen, P. Purkait, R. Garg, A. Blair, C. Shen, and A. van den Hengel. Retrieval augmented classification for long-tail visual recognition. In <i>CVPR</i>, 2022.''',
    '''V. Udandarao, A. Gupta, and S. Albanie. SuS-X: Training-free name-only transfer of vision-language models. In <i>ICCV</i>, 2023.''',
    '''K. S. Tai, P. Bailis, and G. Valiant. Sinkhorn label allocation: Semi-supervised classification via annealed self-training. In <i>ICML</i>, 2021.''',
    '''G. Van Horn, E. Cole, S. Beery, K. Wilber, S. Belongie, and O. Mac Aodha. Benchmarking representation learning for natural world image collections. In <i>CVPR</i>, 2021.''',
]
story.append(P("References", H1))
for i, r in enumerate(REFS, 1):
    story.append(P(f"[{i}]&nbsp; {r}", REF))

# ================================================================== build
PW, PH = letter
LM = RM = 0.68 * inch
TM, BM = 0.72 * inch, 0.72 * inch
GUT = 0.26 * inch
CW = (PW - LM - RM - GUT) / 2
TITLE_H = 3.30 * inch


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Times-Roman", 8.5)
    canvas.drawCentredString(PW / 2, BM - 0.30 * inch, str(canvas.getPageNumber()))
    canvas.restoreState()


doc = BaseDocTemplate(OUT, pagesize=letter, leftMargin=LM, rightMargin=RM,
                      topMargin=TM, bottomMargin=BM,
                      title="Zero-Shot Species Recognition at 17,393 Classes")

body_h = PH - TM - BM
t_first = PageTemplate(
    id="first",
    frames=[Frame(LM, PH - TM - TITLE_H, PW - LM - RM, TITLE_H, id="ttl",
                  leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0),
            Frame(LM, BM, CW, body_h - TITLE_H, id="c1",
                  leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0),
            Frame(LM + CW + GUT, BM, CW, body_h - TITLE_H, id="c2",
                  leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)],
    onPage=footer)
t_rest = PageTemplate(
    id="rest",
    frames=[Frame(LM, BM, CW, body_h, id="d1",
                  leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0),
            Frame(LM + CW + GUT, BM, CW, body_h, id="d2",
                  leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)],
    onPage=footer)

doc.addPageTemplates([t_first, t_rest])
doc.build(story)
print("WROTE", OUT, os.path.getsize(OUT), "bytes")
