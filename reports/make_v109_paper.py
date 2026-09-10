#!/usr/bin/env python
"""Build the FishONet v109 paper (extends the v56 paper through the learned-gate and
leak-free re-ranking work) as an 8-page two-column CVPR-style PDF.

Every number is traced to HANDOFF.md or a direct reading of the builder / training source.
Nothing is estimated except where explicitly marked as a derived quantity (e.g. "net images"
computed from a reported accuracy delta, exactly as the original paper's Table 14 does).
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
                                KeepTogether, NextPageTemplate, Preformatted, HRFlowable)

FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fishonet_figs")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fishonet_v109_paper.pdf")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "font.size": 7.6, "axes.labelsize": 7.8, "xtick.labelsize": 7.2,
    "ytick.labelsize": 7.2, "legend.fontsize": 7.0, "axes.titlesize": 8.2,
})
INK, MID, PALE = "#111111", "#5a5a5a", "#c9c9c9"
BLU, RED, GRN, ORG, PUR = "#1f4e79", "#a4292f", "#2b6b4f", "#b86a12", "#6a3d9a"


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
    fig, ax = plt.subplots(figsize=(3.30, 3.35))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10.6); ax.axis("off")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    def text_width(s, fs, bold=False):
        t = ax.text(5, 5, s, ha="center", fontsize=fs, weight="bold" if bold else "normal")
        fig.canvas.draw()
        bb = t.get_window_extent(renderer=renderer).transformed(ax.transData.inverted())
        t.remove()
        return bb.x1 - bb.x0

    def wrap_to_width(s, fs, max_width, bold=False):
        # Greedy word-wrap measured against the actual renderer, not a character-count
        # guess — a guess is what produced the cross-column text bleed this replaces.
        words = s.split()
        lines, cur = [], ""
        for word in words:
            trial = (cur + " " + word).strip()
            if not cur or text_width(trial, fs, bold=bold) <= max_width:
                cur = trial
            else:
                lines.append(cur)
                cur = word
        if cur:
            lines.append(cur)
        return lines

    def box(x, y, w, h, title, sub=None, fc="white", ec=INK, tc=INK, bold=True, fs=7.4, sub_fs=6.3):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.12",
                                    fc=fc, ec=ec, lw=1.0, zorder=3))
        lines = wrap_to_width(sub, sub_fs, w - 0.6) if sub else []
        title_y = y + h - 0.32 if lines else y + h / 2
        ax.text(x + w / 2, title_y, title, ha="center", va="center",
                fontsize=fs, weight="bold" if bold else "normal", color=tc, zorder=4)
        for j, sl in enumerate(lines):
            ax.text(x + w / 2, title_y - 0.30 - j * 0.28, sl, ha="center", va="center",
                    fontsize=sub_fs, color=MID, zorder=4)

    def arr(x0, y0, x1, y1, c=INK):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                     mutation_scale=8, lw=1.0, color=c, zorder=5))

    def route_box(x, w, top, bot, label, color, fc, items, count, count_label):
        ax.add_patch(FancyBboxPatch((x, bot), w, top - bot,
                                    boxstyle="round,pad=0.05,rounding_size=0.12",
                                    fc=fc, ec=color, lw=1.0, zorder=2))
        cx = x + w / 2
        max_w = w - 0.55
        ax.text(cx, top - 0.33, label, ha="center", fontsize=6.5, weight="bold", color=color)
        cy = top - 0.72
        LH, GAP = 0.32, 0.11
        for s in items:
            lines = wrap_to_width(s, 6.0, max_w)
            for j, sl in enumerate(lines):
                ax.text(cx, cy - j * LH, sl, ha="center", fontsize=6.0, color=MID)
            cy -= LH * len(lines) + GAP
        ax.text(cx, bot + 0.85, "argmax over", ha="center", fontsize=6.6, color=INK)
        ax.text(cx, bot + 0.43, count, ha="center", fontsize=10.5, weight="bold", color=color)
        ax.text(cx, bot + 0.15, count_label, ha="center", fontsize=6.4, color=MID)

    box(2.7, 9.55, 4.6, 0.85, "evaluation image", "35,665 total")
    arr(5.0, 9.55, 5.0, 9.02)
    box(0.9, 8.05, 8.2, 0.95, "learned routing gate",
        "12-feature logistic, rank quantile; keep top f = 0.60")
    arr(3.1, 8.05, 2.2, 7.45, BLU)
    arr(6.9, 8.05, 7.8, 7.45, ORG)
    ax.text(2.55, 7.74, "21,399", fontsize=6.3, color=BLU, ha="right")
    ax.text(7.45, 7.74, "14,266", fontsize=6.3, color=ORG, ha="left")

    RTOP, RBOT = 7.35, 3.20
    route_box(0.05, 4.55, RTOP, RBOT, "CLOSED-SET ROUTE", BLU, "#eef3f8",
              ["BioCLIP-2.5 ViT-H/14, 3 members",
               "prototype + best exemplar",
               "+ leak-free genus re-ranker"],
              "5,795", "trained classes")
    route_box(5.40, 4.55, RTOP, RBOT, "RETRIEVAL ROUTE", ORG, "#fdf3e8",
              ["6 text legs incl. TaxaBind",
               "external photo banks, top-4",
               "+ leak-free rank re-ranker"],
              "11,598", "novel classes")

    arr(2.33, RBOT, 3.55, 2.55, BLU)
    arr(7.67, RBOT, 6.45, 2.55, ORG)
    box(2.3, 1.35, 5.4, 1.20, "prediction",
        "disjoint candidate sets | union = 17,393")
    ax.text(5.0, 0.75, "no single 17,393-way decision is ever taken",
            ha="center", fontsize=6.5, style="italic", color=RED)
    ax.text(5.0, 0.20, "batch-coupled at five points: rank threshold, dbnorm, Sinkhorn, two standardizations",
            ha="center", fontsize=6.2, color=MID)
    fig.tight_layout(pad=0.12)
    fig.savefig(f"{FIG}/pipeline_v109.png", dpi=430, bbox_inches="tight")
    plt.close(fig)


# ====================================================== Fig 2: routing flow
NT, NU = 20097, 15568
KT, ET = 17426, 2671
EU, KU = 11597, 3971


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
    Lt1, Lt0 = 1.0, 1.0 - hT
    Lu1, Lu0 = Lt0 - G, Lt0 - G - hU
    hS, hR = (KT + KU) / 35665, (ET + EU) / 35665
    Rs1, Rs0 = 1.0, 1.0 - hS
    Rr1, Rr0 = Rs0 - G, Rs0 - G - hR
    xa, xb = 0.20, 0.80

    for x, y0, y1, c, lab, n in [(0.0, Lt0, Lt1, BLU, "true class\ntrained", NT),
                                 (0.0, Lu0, Lu1, ORG, "true class\nnovel", NU)]:
        ax.add_patch(plt.Rectangle((x, y0), 0.055, y1 - y0, fc=c, ec="none", zorder=4))
        ax.text(x - 0.015, (y0 + y1) / 2, f"{lab}\n{n:,}", ha="right", va="center", fontsize=6.5)
    for x, y0, y1, c, lab, n in [(1.245, Rs0, Rs1, BLU, "closed-set\nroute", KT + KU),
                                 (1.245, Rr0, Rr1, ORG, "retrieval\nroute", ET + EU)]:
        ax.add_patch(plt.Rectangle((x, y0), 0.055, y1 - y0, fc=c, ec="none", zorder=4))
        ax.text(x + 0.070, (y0 + y1) / 2, f"{lab}\n{n:,}", ha="left", va="center", fontsize=6.5)

    tt = Lt1 - KT / 35665
    _ribbon(ax, xa, xb, Lt1, tt, Rs1, Rs1 - KT / 35665, BLU, .50)
    _ribbon(ax, xa, xb, tt, Lt0, Rr1, Rr1 - ET / 35665, RED, .55)
    uu = Lu1 - KU / 35665
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
    ax.text(0.50, 1.11, "true class not in the candidate set it was scored against; f = 0.60 unchanged v50→v109",
            ha="center", fontsize=6.0, color=RED, style="italic")
    fig.tight_layout(pad=0.12)
    fig.savefig(f"{FIG}/flow_v109.png", dpi=430, bbox_inches="tight")
    plt.close(fig)


# ==================================================== Fig 4: aspect-ratio shift
def fig_aspect():
    fig, ax = plt.subplots(figsize=(3.30, 1.75))
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
    ax.text(1.04, -0.80, "original\naugmentation", fontsize=6.3,
            color=RED, ha="center", va="center")
    ax.text(2.55, -0.80, "corrected\nrange", fontsize=6.3, color=BLU, ha="center", va="center")
    _clean(ax); _nospine(ax)
    fig.tight_layout(pad=0.15)
    fig.savefig(f"{FIG}/aspect_v109.png", dpi=430, bbox_inches="tight")
    plt.close(fig)


# ================================================ Fig 5: operating points
SUBS = [("v33", 78.20, 8.51), ("v36", 78.95, 10.42), ("v46", 78.95, 14.55),
        ("v50", 76.31, 19.34), ("v56", 75.67, 20.57),
        ("v77", 77.72, 21.96), ("v79", 77.45, 22.41), ("v81", 77.45, 22.42),
        ("v82", 77.45, 22.91), ("v83", 77.54, 22.91), ("v109", 77.00, 23.77)]


def fig_operating():
    fig, ax = plt.subplots(figsize=(3.30, 2.55))
    for lvl in (48, 50, 52, 54):
        xs = np.array([74.0, 81.0])
        ax.plot(xs, (lvl - 0.5635 * xs) / 0.4365, lw=0.7, ls=(0, (4, 3)),
                color=PALE, zorder=1)
        yl = (lvl - 0.5635 * 80.3) / 0.4365
        if 6.6 < yl < 24.5:
            ax.text(80.3, yl + 0.15, f"{lvl}%", fontsize=6.2, color=MID,
                    ha="left", va="bottom")
    ax.plot([s[1] for s in SUBS[:5]], [s[2] for s in SUBS[:5]], "-", lw=0.9, color=PALE, zorder=2)
    ax.plot([s[1] for s in SUBS[4:]], [s[2] for s in SUBS[4:]], "-", lw=0.9, color=PUR, zorder=2)
    for name, s, u in SUBS:
        ax.plot(s, u, "o", ms=4.4, color="white",
                mec=(PUR if name not in ("v33", "v36", "v46", "v50", "v56") else INK), mew=1.3, zorder=4)
    labels = {"v33": (0, -0.9, "center"), "v36": (0, -0.9, "center"), "v46": (0.35, 0.15, "left"),
              "v50": (0, 0.75, "center"), "v56": (0.25, -1.0, "left"),
              "v77": (0.35, -0.55, "left"), "v109": (0.30, 0.85, "left")}
    for name, (dx, dy, ha) in labels.items():
        s, u = next((q[1], q[2]) for q in SUBS if q[0] == name)
        ax.text(s + dx, u + dy, name, fontsize=6.6, ha=ha, va="center",
                color=PUR if name in ("v77", "v109") else INK,
                weight="bold" if name in ("v56", "v109") else "normal")
    ax.annotate("learned gate +\nleak-free re-rank\n(v77–v109)", xy=(77.5, 22.9), xytext=(76.0, 24.6),
                fontsize=6.7, color=PUR, ha="left", weight="bold",
                arrowprops=dict(arrowstyle="->", lw=1.1, color=PUR))
    ax.set_xlim(81.2, 74.4); ax.set_ylim(6.4, 25.4)
    ax.set_xlabel("trained-class top-1 (%)      <-  better")
    _clean(ax, "novel-class top-1 (%)    better  ->")
    fig.tight_layout(pad=0.15)
    fig.savefig(f"{FIG}/operating_v109.png", dpi=430, bbox_inches="tight")
    plt.close(fig)


# ==================================================== Fig 6: progress
def fig_progress():
    labels = ["v22", "v31", "v33", "v36", "v46", "v50", "v56", "v77", "v79", "v81", "v82", "v83", "v109"]
    overall = [45.39, 47.69, 47.78, 49.04, 50.84, 51.44, 51.62,
               53.383, 53.425, 53.431, 53.644, 53.697, 53.761]
    novel = [None, None, 8.51, 10.42, 14.55, 19.34, 20.57,
             21.96, 22.41, 22.42, 22.91, 22.91, 23.77]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(3.30, 2.75), sharex=True,
                                 gridspec_kw={"height_ratios": [1.15, 1]})
    x = list(range(len(labels)))
    a1.plot(x[:7], overall[:7], "-o", ms=3.6, lw=1.2, color=INK, zorder=3)
    a1.plot(x[6:], overall[6:], "-o", ms=3.6, lw=1.2, color=PUR, zorder=3)
    a1.axhline(50.56, ls=(0, (4, 3)), lw=0.9, color=GRN)
    a1.text(0.1, 50.74, "contemporaneous leader  50.56", fontsize=6.2, color=GRN)
    a1.set_ylim(44.6, 55.0); _clean(a1, "overall top-1 (%)")
    a1.annotate("learned gate +\nleak-free re-rank", xy=(7, 53.383), xytext=(3.6, 53.4),
                fontsize=6.3, color=PUR, ha="center",
                arrowprops=dict(arrowstyle="->", lw=0.8, color=PUR))
    xs = [i for i, v in enumerate(novel) if v is not None]
    a2.plot([i for i in xs if i < 7], [novel[i] for i in xs if i < 7], "-s", ms=3.4, lw=1.2, color=ORG, zorder=3)
    a2.plot([i for i in xs if i >= 6], [novel[i] for i in xs if i >= 6], "-s", ms=3.4, lw=1.2, color=PUR, zorder=3)
    a2.set_ylim(6, 24.5); _clean(a2, "novel-class top-1 (%)")
    a2.set_xticks(x); a2.set_xticklabels(labels, fontsize=6.2, rotation=0)
    fig.tight_layout(pad=0.15, h_pad=0.6)
    fig.savefig(f"{FIG}/progress_v109.png", dpi=430, bbox_inches="tight")
    plt.close(fig)


# ==================================================== Fig 7: encoders
def fig_encoders():
    rows = [("BioCLIP-2.5-H", 23.38, 21.53), ("TaxaBind ViT-B/16", 13.98, 11.48),
            ("BioCLIP 2", None, 14.37), ("BioCAP", 12.25, 6.77),
            ("BioCLIP 1", None, 11.60), ("SigLIP2 SO400M", None, 5.13),
            ("BioCLIP hyperbolic", 4.92, 4.36), ("general CLIP", None, 0.78),
            ("bioclip-inat-only", None, 0.35), ("BioTrove-CLIP", 0.04, 0.00)]
    fig, ax = plt.subplots(figsize=(3.30, 2.30))
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
        ax.text(best + 0.45, v + off, f"{best:.2f}", va="center", fontsize=6.3)
    ax.barh([1.05], [31.10], height=0.46, color=BLU, zorder=3)
    ax.text(31.6, 1.05, "31.10", va="center", fontsize=6.3, color=BLU, weight="bold")
    ax.barh([0.0], [26.62], height=0.46, color="#9dbdd8", zorder=3)
    ax.text(27.1, 0.0, "26.62", va="center", fontsize=6.3, color=BLU)
    ax.axhline(1.85, lw=0.8, color=MID, ls=(0, (3, 2)))
    ax.set_yticks(y + [1.05, 0.0])
    ax.set_yticklabels([r[0] for r in rows] +
                       ["adapter, corrected aug.", "adapter, original aug."], fontsize=6.4)
    for lab in ax.get_yticklabels()[-2:]:
        lab.set_color(BLU)
    ax.set_xlim(0, 36.5)
    ax.set_xlabel("novel-class top-1 on the held-out proxy (%)")
    ax.legend(frameon=False, loc="upper right", handlelength=1.2,
              bbox_to_anchor=(1.02, 1.05), borderpad=0.15)
    _clean(ax); _nospine(ax)
    fig.tight_layout(pad=0.15)
    fig.savefig(f"{FIG}/encoders_v109.png", dpi=430, bbox_inches="tight")
    plt.close(fig)


# ============================================ Fig 9: proxy-to-real transfer
def fig_transfer():
    pts = [("v40", 1.68, 0.03), ("v43", 0.52, 0.19), ("v46", 0.345, 0.067),
           ("v56", 1.424, 0.174), ("v82", 8.154, 0.213), ("v83", 1.803, 0.053)]
    neg = [("v34", 3.51, -0.07), ("v39", 0.52, -0.11)]
    off = {"v43": (6, 4), "v46": (-7, -10), "v56": (6, -9), "v40": (6, 3),
           "v82": (-38, 5), "v83": (6, -9)}
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(3.30, 3.75),
                                 gridspec_kw={"height_ratios": [1.05, 1.12]})

    a1.axhline(0, color=INK, lw=0.9, zorder=2)
    for r, st in ((0.36, (0, (1, 2))), (0.12, (0, (5, 2))), (0.026, (0, (1, 1)))):
        a1.plot([0, 8.6], [0, 8.6 * r], lw=0.9, ls=st, color=MID, zorder=1)
    for n, x, y in pts:
        a1.plot(x, y, "o", ms=5.0, color=(PUR if n in ("v82", "v83") else INK), zorder=4)
        a1.annotate(n, (x, y), textcoords="offset points",
                    xytext=off.get(n, (6, 4)), fontsize=6.7,
                    color=(PUR if n in ("v82", "v83") else INK), zorder=5)
    for n, x, y in neg:
        a1.plot(x, y, "X", ms=6.4, color=RED, zorder=4)
        a1.annotate(n, (x, y), textcoords="offset points", xytext=(7, -3),
                    fontsize=6.7, color=RED, va="center", zorder=5)
    a1.annotate("v81 (leak): +13.374 proxy, +0.006 real\n— off-scale, worst mirage in the project",
                xy=(8.5, 0.05), xytext=(3.0, 0.34), fontsize=6.5, color=RED, style="italic",
                arrowprops=dict(arrowstyle="->", lw=1.0, color=RED))
    a1.axhspan(-0.20, 0, color=RED, alpha=0.08, lw=0)
    a1.set_xlim(-0.20, 8.9); a1.set_ylim(-0.20, 0.62)
    a1.set_xlabel("proxy gain (pt)"); _clean(a1, "real gain (pt)")
    a1.set_title("(a)  paired measurements (purple = leak-free re-rankers)", fontsize=7.0, loc="left")

    names = ["dual BioCLIP-2 (v43)", "leak-free seen re-rank (v83)", "LoRA / ToL merge (v46)",
             "crops + 336 bank (v56)", "leak-free unseen re-rank (v82)", "top-4 pooling (v40)"]
    vals = [0.359, 0.0295, 0.194, 0.122, 0.0261, 0.018]
    cols = [INK, PUR, INK, INK, PUR, INK]
    yy = list(range(len(vals)))[::-1]
    a2.barh(yy, vals, height=0.60, color=cols, zorder=3)
    a2.barh([-1.6, -2.6], [-0.020, -0.212], height=0.60, color=RED, zorder=3)
    for y, v in zip(yy, vals):
        a2.text(v + 0.010, y, f"{v:.4f}", va="center", fontsize=6.7)
    for y, v in zip([-1.6, -2.6], [-0.020, -0.212]):
        a2.text(v - 0.014, y, f"{v:.3f}", va="center", ha="right",
                fontsize=6.7, color=RED)
    a2.axvline(0, color=INK, lw=0.9, zorder=4)
    a2.set_yticks(yy + [-1.6, -2.6])
    a2.set_yticklabels(names + ["recover-on-eject (v34)", "TaxaBind protos (v39)"],
                       fontsize=6.7)
    for lab in a2.get_yticklabels()[-2:]:
        lab.set_color(RED)
    for i, lab in enumerate(a2.get_yticklabels()):
        if "leak-free" in names[len(names) - 1 - i] if i < len(names) else False:
            lab.set_color(PUR)
    a2.axhspan(-3.2, -1.0, color=RED, alpha=0.07, lw=0)
    a2.set_xlim(-0.355, 0.44); a2.set_ylim(-3.2, 6.7)
    a2.set_xlabel("real accuracy points gained per proxy point gained")
    a2.set_title("(b)  transfer rate", fontsize=7.0, loc="left")
    _clean(a2); _nospine(a2)
    fig.tight_layout(pad=0.15, h_pad=1.0)
    fig.savefig(f"{FIG}/transfer_v109.png", dpi=430, bbox_inches="tight")
    plt.close(fig)


# ==================================================== Fig: data composition
def fig_datacomp():
    fig, ax = plt.subplots(figsize=(3.30, 1.55))
    ax.set_xlim(0, 100); ax.set_ylim(-0.7, 1.7); ax.axis("off")
    rows = [(1.0, "class space\n(17,393)", 5795, 11598, 33.3, 66.7),
            (0.0, "eval. images\n(35,665)", 20097, 15568, 56.35, 43.65)]
    for y, label, n_t, n_u, p_t, p_u in rows:
        ax.barh(y, p_t, height=0.55, left=0, color=BLU, zorder=3)
        ax.barh(y, p_u, height=0.55, left=p_t, color=ORG, zorder=3)
        ax.text(p_t / 2, y, f"{n_t:,}\n{p_t:.1f}%", ha="center", va="center",
                fontsize=6.3, color="white", weight="bold")
        ax.text(p_t + p_u / 2, y, f"{n_u:,}\n{p_u:.1f}%", ha="center", va="center",
                fontsize=6.3, color="white", weight="bold")
        ax.text(-2, y, label, ha="right", va="center", fontsize=6.6, color=INK)
    ax.text(45, 0.5, "33.3% of classes  →  56.4% of images:\ntrained classes carry 2.6× their\nclass-share in eval. volume",
            ha="center", va="center", fontsize=5.9, color=MID, style="italic")
    ax.text(2, -0.6, "trained", ha="left", fontsize=6.4, color=BLU, weight="bold")
    ax.text(98, -0.6, "novel", ha="right", fontsize=6.4, color=ORG, weight="bold")
    fig.tight_layout(pad=0.1)
    fig.savefig(f"{FIG}/datacomp_v109.png", dpi=430, bbox_inches="tight")
    plt.close(fig)


# ================================================= Fig: negative-result scale
def fig_deadlevers():
    bars = [("two-stage family narrowing", -20.0),
            ("common-name / generic prompts", -17.73),
            ("SigLIP2 general backbone", -16.40),
            ("VLM rerank (Qwen2.5-VL)", -16.0),
            ("LLM raw description text", -15.87)]
    fig, ax = plt.subplots(figsize=(3.30, 2.35))
    y = list(range(len(bars)))[::-1]
    vals = [b[1] for b in bars]
    ax.barh(y, vals, height=0.56, color=RED, zorder=3)
    for yy, v in zip(y, vals):
        # Label sits just inside the bar's own tip (anchored to v, growing toward 0),
        # never near the y-axis label margin -- a right-anchored label at a fixed
        # offset from v drifts into that margin as |v| grows, which is what broke here.
        ax.text(v + 0.4, yy, f"{v:.2f}", va="center", ha="left", fontsize=6.3,
                color="white", weight="bold")
    ax.axvline(0, color=INK, lw=0.9, zorder=4)
    ax.set_xlim(-24, 5.2)

    ax.axhspan(-1.55, -0.55, color=MID, alpha=0.07, lw=0)
    ax.barh([-1.0], [1.9], left=0, height=0.40, color=GRN, zorder=3)
    ax.text(2.3, -1.0, "proxy +1.9", va="center", fontsize=6.3, color=GRN)
    ax.text(-0.6, -1.0, "real: degrades", va="center", ha="right", fontsize=6.3, color=RED)

    ax.set_yticks(y + [-1.0])
    ax.set_yticklabels([b[0] for b in bars] + ["soft (non-hard) routing"], fontsize=6.3)
    ax.get_yticklabels()[-1].set_style("italic")
    ax.set_ylim(-1.8, 4.75)
    ax.set_xlabel("novel-class proxy points vs. adopted config", fontsize=6.6)
    _clean(ax); _nospine(ax)
    fig.tight_layout(pad=0.2)
    fig.savefig(f"{FIG}/deadlevers_v109.png", dpi=430, bbox_inches="tight")
    plt.close(fig)


for _f in (fig_pipeline, fig_flow, fig_aspect, fig_operating, fig_progress, fig_encoders, fig_transfer,
          fig_datacomp, fig_deadlevers):
    _f()


# ===================================================================== styles
def S(name, **kw):
    base = dict(fontName="Times-Roman", fontSize=8.75, leading=9.9, alignment=TA_JUSTIFY,
                spaceAfter=0, spaceBefore=0)
    base.update(kw)
    return ParagraphStyle(name, **base)


TITLE = S("t", fontName="Times-Bold", fontSize=16.5, leading=19.5, alignment=TA_CENTER, spaceAfter=8)
AUTH = S("a", fontSize=10, leading=12, alignment=TA_CENTER, spaceAfter=2)
ABSH = S("abh", fontName="Times-Bold", fontSize=9.2, leading=10.6, alignment=TA_CENTER, spaceAfter=4)
ABS = S("ab", fontSize=8.5, leading=9.7)
BODY = S("b", firstLineIndent=10, spaceAfter=0)
BODY0 = S("b0", firstLineIndent=0)
H1 = S("h1", fontName="Times-Bold", fontSize=10.6, leading=12.2, alignment=0, spaceBefore=6.5, spaceAfter=2.6,
      textColor=colors.HexColor(BLU))
H2 = S("h2", fontName="Times-Bold", fontSize=9.2, leading=10.6, alignment=0, spaceBefore=4.8, spaceAfter=1.8,
      textColor=colors.HexColor(ORG))
BLU_T, ORG_T, GRN_T, RED_T = "#eef3f8", "#fdf3e8", "#eaf5ef", "#f7e9ea"
CAP = S("cap", fontSize=7.4, leading=8.5, spaceBefore=2.6, spaceAfter=4.4)
EQ = S("eq", fontSize=8.9, leading=11.4, alignment=TA_CENTER, spaceBefore=3.4, spaceAfter=3.4)
REF = S("r", fontSize=7.1, leading=8.2, leftIndent=11, firstLineIndent=-11, spaceAfter=1.5)
CODE = ParagraphStyle("code", fontName="Courier", fontSize=6.6, leading=7.7, spaceBefore=2, spaceAfter=2,
                      backColor="#f4f4f4", borderPadding=(4, 5, 4, 5), leftIndent=2)


def code(src, caption):
    return KeepTogether([Preformatted(src, CODE), P(caption, CAP)])
BUL = S("bu", fontSize=8.6, leading=9.8, leftIndent=10, firstLineIndent=-7, spaceAfter=1.5)

TAU = "tau"
ALPHA = "alpha"


def P(t, s=BODY):
    return Paragraph(t, s)


def bullets(items):
    return [P("&bull;&nbsp; " + i, BUL) for i in items]


CALLOUT = S("co", fontName="Times-Italic", fontSize=8.5, leading=10.4, spaceBefore=5, spaceAfter=5,
           leftIndent=4, rightIndent=2, borderPadding=(6, 8, 6, 8))


def callout(text, tint=ORG_T):
    return KeepTogether([Paragraph(text, ParagraphStyle("co_i", parent=CALLOUT, backColor=tint))])


def tbl(data, widths, caption, align=None, fs=6.9):
    from reportlab.lib.enums import TA_RIGHT
    amap = {"RIGHT": TA_RIGHT}
    cells = []
    for r, row in enumerate(data):
        out = []
        for c, cell in enumerate(row):
            a = amap.get((align or {}).get(c, "LEFT"), 0)
            stl = ParagraphStyle(f"tc{r}{c}", fontName="Times-Bold" if r == 0 else "Times-Roman",
                                 fontSize=fs, leading=fs * 1.20, alignment=a)
            out.append(Paragraph(str(cell), stl))
        cells.append(out)
    t = Table(cells, colWidths=[w * inch for w in widths], hAlign="LEFT")
    st = [("FONT", (0, 0), (-1, -1), "Times-Roman", fs),
          ("FONT", (0, 0), (-1, 0), "Times-Bold", fs),
          ("LINEABOVE", (0, 0), (-1, 0), 0.9, colors.black),
          ("LINEBELOW", (0, 0), (-1, 0), 0.55, colors.black),
          ("LINEBELOW", (0, -1), (-1, -1), 0.9, colors.black),
          ("TOPPADDING", (0, 0), (-1, -1), 1.7),
          ("BOTTOMPADDING", (0, 0), (-1, -1), 1.7),
          ("LEFTPADDING", (0, 0), (-1, -1), 2.0),
          ("RIGHTPADDING", (0, 0), (-1, -1), 2.0),
          ("VALIGN", (0, 0), (-1, -1), "TOP")]
    t.setStyle(TableStyle(st))
    return KeepTogether([t, P(caption, CAP)])


SEV_COLOR = {"HIGH": RED, "MED": ORG, "LOW": GRN}


def risk_tbl(rows, caption, fs=6.0):
    # rows: (threat, severity in {HIGH,MED,LOW}, why) -- severity drawn from the
    # section's own hedging language (Sec. 11 prose), not assigned independently.
    head = ["threat", "severity", "why"]
    stl_h = ParagraphStyle("rk_h", fontName="Times-Bold", fontSize=fs, leading=fs * 1.2)
    stl_t = ParagraphStyle("rk_t", fontName="Times-Bold", fontSize=fs, leading=fs * 1.2)
    stl_w = ParagraphStyle("rk_w", fontName="Times-Roman", fontSize=fs, leading=fs * 1.2)
    cells = [[Paragraph(h, stl_h) for h in head]]
    for threat, sev, why in rows:
        chip = ParagraphStyle(f"rk_s_{sev}", fontName="Times-Bold", fontSize=fs, leading=fs * 1.2,
                              alignment=1, textColor=colors.white)
        cells.append([Paragraph(threat, stl_t), Paragraph(sev, chip), Paragraph(why, stl_w)])
    t = Table(cells, colWidths=[0.90 * inch, 0.42 * inch, 2.03 * inch], hAlign="LEFT")
    st = [("FONT", (0, 0), (-1, -1), "Times-Roman", fs),
          ("LINEABOVE", (0, 0), (-1, 0), 0.9, colors.black),
          ("LINEBELOW", (0, 0), (-1, 0), 0.55, colors.black),
          ("LINEBELOW", (0, -1), (-1, -1), 0.9, colors.black),
          ("TOPPADDING", (0, 0), (-1, -1), 2.2),
          ("BOTTOMPADDING", (0, 0), (-1, -1), 2.2),
          ("LEFTPADDING", (0, 0), (-1, -1), 3.0),
          ("RIGHTPADDING", (0, 0), (-1, -1), 3.0),
          ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]
    for i, (_, sev, _w) in enumerate(rows, start=1):
        st.append(("BACKGROUND", (1, i), (1, i), colors.HexColor(SEV_COLOR[sev])))
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
        "We report a measured study of large-vocabulary generalized zero-shot recognition on a fish-species "
        "benchmark with 17,393 candidate classes, of which 11,598 (66.7%) have no training images at all by "
        "task definition, under a hard 30-submission evaluation budget. A final system partitions the "
        "decision between a closed-set route over 5,795 trained classes and a retrieval route over the "
        "remainder, reaching <b>53.761% overall</b> (77.00% trained-class, 23.77% novel-class), up from an "
        "initial 47.78%. We use the budget to measure four things that are more often assumed than tested. "
        "First, a <i>framing shift</i> between training-augmentation and evaluation-image aspect ratios, "
        "corrected independently at training and inference time. Second, an external retrieval bank of "
        "101,106 biodiversity photographs that raises novel-class accuracy from 10.42% to 20.57%, audited "
        "for near-duplicates against the evaluation set and bounded at 8.9% possible inflation. Third, "
        "replacing a hand-derived routing rule with a 12-feature learned gate, which improved trained- and "
        "novel-class accuracy <i>simultaneously</i> (+1.767pt) &mdash; the only lever in the project that did "
        "so without trading one for the other. Fourth &mdash; our principal methodological result &mdash; we "
        "show that a per-candidate re-ranker trained on the same held-out proxy learned to exploit a "
        "candidate-subpopulation identity leak invisible to class-disjoint cross-validation, inflating its "
        "proxy gain by <b>2,229&times;</b> (+13.374 proxy vs +0.006 real); restricting the candidate pool to "
        "close the leak recovered a genuine, 35&times; better-transferring signal (+8.154 proxy, +0.213 "
        "real). Combined with a second, independently confirmed instance on the closed-set route, this "
        "establishes a predictive rule: interventions that add new information to the decision compound "
        "with the rest of the system, while interventions that only re-weight existing information return "
        "most of their proxy gain on the real evaluation, or reverse it. Oracle probes bound the residual "
        "gap to the visual encoder, not the routing policy. We report our full catalogue of measured "
        "negative results.", ABS),
    FrameBreak(),
]

# ============================================================ 1 introduction
story += [
    P("1. Introduction", H1),
    P("Species-level visual recognition is open-ended by construction. On the benchmark studied here the "
      "imbalance is extreme: 11,598 of 17,393 candidate classes &mdash; 66.7% &mdash; have no training "
      "images by definition of the task, accounting for 15,568 of 35,665 evaluation images.", BODY0),
    P("All 17,393 labels are known in advance and every novel class is identified by its scientific name, so "
      "this is <i>generalized zero-shot</i> recognition [3], with seen and unseen classes scored under one "
      "metric, not open-set recognition in the sense of Scheirer <i>et al.</i> [1]. We draw on the open-set "
      "literature [1, 2] only for the routing question: judging whether an image belongs to the trained "
      "vocabulary is the decision an open-set rejector makes, though we act on it by switching branches "
      "rather than abstaining. What makes the setting unusual is the combination of that class imbalance "
      "with a hard evaluation budget of three submissions per day, thirty in total, which changes the "
      "research process itself: almost every decision is first screened on an internal held-out proxy, and "
      "only changes that clear a fixed margin are spent on a real evaluation."),
    P("That constraint makes the proxy's reliability a measurable object of study rather than an assumption. "
      "Across nine confirmed real-evaluation changes tracked in this paper, the fraction of proxy gain that "
      "survived contact with the real distribution ranged from parity down to a factor of <b>2,229</b>, two "
      "changes moved the proxy up and the real score down, and one architectural simplification the proxy "
      "preferred by 5.09 points was worse by 0.60 points in reality. The most severe of these &mdash; a "
      "per-candidate re-ranker that learned to detect which population the proxy drew its gold labels from, "
      "rather than to rank evidence &mdash; is invisible to the standard defense (class-disjoint "
      "cross-validation) and is this paper's central methodological contribution."),
    P("Four ordinary engineering ideas, found and confirmed by measurement rather than intuition, account "
      "for essentially all of the gain: correcting a training/evaluation aspect-ratio mismatch; adding an "
      "external photograph bank for classes with zero training images; replacing a hand-derived routing "
      "threshold with a learned classifier over the same signals plus new ones; and, once the leak above was "
      "fixed, a leak-free re-ranker on both routes. Oracle probes then argue that what remains is a limitation "
      "of the visual encoder, not of the routing or re-ranking machinery that consumed most of the later "
      "engineering effort.", ),
    P("Contributions.", H2),
]
story += bullets([
    "A final open-set system at 17,393-class scale (53.761% overall, up from 47.78%), with an explicit "
    "account of the five ways in which it is transductive rather than per-image.",
    "A measured framing-shift diagnosis (Sec. 5) and an external-bank retrieval route (Sec. 6, +10.15 novel "
    "points) with a near-duplicate audit bounding possible inflation at 8.9% of that gain.",
    "A learned 12-feature routing gate (Sec. 4.6) that raised trained- <i>and</i> novel-class accuracy "
    "together (+1.767pt) &mdash; every other routing-fraction change in the project traded one for the "
    "other at a fixed 4:1 exchange rate implied by the task's class balance.",
    "A measured population-identity leak in per-candidate re-ranking (Sec. 8.2): +13.374 proxy points that "
    "delivered +0.006 real, undetectable by class-disjoint cross-validation, together with the fix "
    "(+8.154 proxy, +0.213 real) and a second independent confirmation on the closed-set route (Sec. 8.3).",
    "A predictive rule, confirmed on two independent components: new-information interventions compound on "
    "real transfer; same-information re-weighting returns most of the proxy gain or reverses it (Sec. 8.3).",
    "Oracle probes localizing the residual gap to the encoder (Sec. 7) and a catalogue of 24 measured "
    "negative results (Sec. 9).",
])

# ============================================================ 2 related work
story += [
    P("2. Related Work", H1),
    P("<b>Open-set and zero-shot recognition.</b> Open-set recognition [1, 2] concerns classifiers that "
      "must not be forced to emit a label from a closed training vocabulary. Our novel-class route is "
      "closer to generalized zero-shot learning [3], where seen and unseen classes are scored jointly "
      "under a single metric &mdash; exactly the protocol here (Sec. 3). Where the standard zero-shot "
      "setting supplies attributes or class embeddings, we supply external photographs, which changes "
      "the failure mode from attribute quality to coverage and domain gap. Our routing decision is the "
      "open-set rejector's question, answered by branching to a second route rather than by abstaining.", BODY0),
    P("<b>Vision-language and biology foundation models.</b> CLIP [4] introduced the pattern our system "
      "relies on throughout: classification by cosine similarity between an image embedding and frozen "
      "per-class text embeddings, with no learned linear head. BioCLIP [5] specializes this to biology "
      "using taxonomic text over the TreeOfLife-10M corpus, and BioCLIP&nbsp;2 [6] scales it to "
      "TreeOfLife-200M. Both appear in our system in different roles and at different scales: the "
      "closed-set route runs a BioCLIP-2.5 ViT-H/14 ensemble, while the novel-class route additionally "
      "uses BioCLIP&nbsp;2 ViT-L/14 prototypes. TaxaBind [11] binds several ecological modalities into a "
      "shared space anchored on ground-level species imagery; we use its image and text towers as one "
      "leg of the novel-class text stack."),
    P("<b>Parameter-efficient adaptation.</b> We adapt the image tower with LoRA [7], which we found "
      "necessary in a specific and slightly unusual way: full fine-tuning of a strong biology encoder on "
      "a narrow fish subset degrades exactly the broad taxonomic structure the novel-class route depends "
      "on. At inference we additionally scale the low-rank term below its trained strength, a discrete "
      "analogue of the weight-space interpolation used by WiSE-FT."),
    P("<b>Margin losses on cosine classifiers.</b> Because the closed-set classifier compares an "
      "embedding to a fixed class matrix by cosine similarity, our training objective (Sec. 4.9) is "
      "structurally a member of the family developed for face recognition, where CosFace [8] subtracts "
      "an additive margin from the target class similarity and ArcFace [9] applies the equivalent margin "
      "in angular space. We use the margin-free form and note the connection because the fixed logit "
      "scale our objective already carries is the same scale term those losses require; we did not train "
      "a margin variant and make no claim about one."),
    P("<b>Prototypes and retrieval.</b> Snell <i>et al.</i> [10] classify by distance to a class mean "
      "embedding, which is precisely what our BioCLIP&nbsp;2 external-prototype leg does. Our "
      "iNaturalist photograph banks are deliberately <i>not</i> prototypes: we retain individual "
      "photograph embeddings and score a class by the mean of its top-4 cosine similarities, which "
      "measured better than either a class mean or a single maximum (Sec. 6)."),
    P("<b>Score normalization and transport.</b> The single most load-bearing normalization in our "
      "system applies a log-softmax across images as well as across classes, penalizing candidate "
      "classes that are similar to everything &mdash; a hubness correction in the sense of "
      "Radovanovi&#263; <i>et al.</i> [27], whose direct antecedent is the inverted softmax of Smith "
      "<i>et al.</i> [28] and whose test-query-free relative is querybank normalization [29]. Entropic "
      "optimal transport [13] and its use for balanced online assignment in SwAV [14] are the direct "
      "antecedents of the Sinkhorn projection we apply with a uniform column prior over novel-class rows "
      "only."),
    P("<b>Retrieval-augmented classification.</b> Classifying against an external corpus at inference is "
      "established: retrieval-augmented classification for long-tailed recognition [30] and name-only "
      "transfer of vision-language models [31] both assemble support sets for classes the model never "
      "trained on. Our contribution is not the mechanism but its scale and its audit &mdash; 101,106 "
      "photographs over 7,364 species, and the near-duplicate check of Sec. 6 that we believe such claims "
      "require."),
    P("<b>Test-time augmentation and fine-grained recognition.</b> Fixed multi-crop evaluation is "
      "standard practice; learned policies [15] are the systematic version. Our contribution in Sec. 5 is "
      "not the technique but the diagnosis that motivated a specific geometry. The task itself is "
      "fine-grained categorization in the sense of Wei <i>et al.</i> [16], and the closest domain "
      "benchmark, FishNet [17], is closed-set and so does not exercise the regime that dominates our "
      "error budget; on vocabulary scale the nearest comparison is the iNaturalist benchmarking work of "
      "Van Horn <i>et al.</i> [33]."),
    P("<b>Transductive and calibrated zero-shot recognition.</b> Our routing fraction is a "
      "seen-versus-unseen calibration parameter of the kind Chao <i>et al.</i> [25] introduced as "
      "calibrated stacking, except that Sec. 4.5 derives its optimum and Sec. 4.6 then <i>replaces the "
      "derivation with a learned classifier</i> once more signal became available &mdash; a step outside "
      "the calibrated-stacking framework, which assumes the seen/unseen scores themselves stay fixed. "
      "Transductive zero-shot methods [26] use unlabeled target data during <i>training</i>; our coupling "
      "differs in kind, occurring only at inference across the evaluation batch, and its closer relatives "
      "are batch-level assignment methods such as Sinkhorn label allocation [32]."),
    P("<b>Proxy metrics, adaptive evaluation, and shortcut learning.</b> The paper's central "
      "methodological result belongs to a literature we should state clearly. Dwork <i>et al.</i> "
      "[18, 19] formalize the loss of validity when a held-out set answers a sequence of adaptively "
      "chosen queries, and Blum and Hardt [20] give a leaderboard mechanism restoring it; our thirty "
      "sequential submissions are that regime, run without any such protection. The prevailing empirical "
      "picture is nonetheless more optimistic than our experience: Roelofs <i>et al.</i> [21] analyse 120 "
      "competitions and find public-to-private leaderboard ordering remarkably reliable, Recht <i>et "
      "al.</i> [22] find replication drops on ImageNet that preserve ordering, and Miller <i>et al.</i> "
      "[23] report strong linear in- to out-of-distribution correlation across many settings. We offer a "
      "single-benchmark counterexample, not a refutation, and note that Teney <i>et al.</i> [24] "
      "document real datasets where the correlation inverts &mdash; the regime our sign inversions (Sec. "
      "8.1) fall into. Sec. 8.2's leak is additionally an instance of a distinct failure: the "
      "candidate pool's <i>composition</i>, not the query distribution, supplies the shortcut, closer to "
      "the concern that motivates class-disjoint evaluation splits in few-shot learning &mdash; except "
      "here the split was already class-disjoint and the leak survived it, because the leaking variable "
      "was <i>which population a class belongs to</i>, not <i>which class</i>."),
]

# ================================================================== 3 task
story += [
    P("3. Task, Data, and Protocol", H1),
    P("<b>Label space and splits.</b> N = 17,393 species-level classes; 5,795 <i>trained</i> classes have "
      "&ge;1 training image, 11,598 <i>novel</i> classes have none anywhere in the provided data. The "
      "organizers release 64,259 labelled training images over the trained classes. The evaluation set "
      "contains 35,665 images: 20,097 trained-class, 15,568 novel-class.", BODY0),
    tbl([["quantity", "count", "share"],
         ["candidate classes", "17,393", "100%"],
         ["&nbsp;&nbsp;trained / novel", "5,795 / 11,598", "33.3 / 66.7%"],
         ["labelled training images", "64,259", "100%"],
         ["&nbsp;&nbsp;adapter training / withheld pseudo-novel", "61,941 / 2,318", "96.4 / 3.6%"],
         ["proxy candidate list (novel + pseudo)", "12,757", "&mdash;"],
         ["evaluation images (trained / novel)", "20,097 / 15,568", "56.35 / 43.65%"],
         ["external bank photographs / embedded", "117,225 / 101,545", "&mdash;"]],
        [1.90, 0.72, 0.56],
        "<b>Table 1.</b> Task scale. Evaluation-split proportions are exactly the metric weights of Eq. 1.",
        align={1: "RIGHT", 2: "RIGHT"}),
    figure(f"{FIG}/datacomp_v109.png", 3.15,
           "<b>Figure 1.</b> The two splits move in opposite directions: novel classes are 66.7% of the "
           "label space but only 43.65% of evaluation images, because each trained class is sampled "
           "&asymp;2.6&times; as often as its class-share alone would predict."),
    P("<b>Metric.</b> Acc = 0.5635&middot;Acc<sub>seen</sub> + 0.4365&middot;Acc<sub>novel</sub> (1) "
      "&mdash; the population proportion, not a design choice. One trained-class point is worth 0.56 "
      "overall points; one novel-class point is worth 0.44.", BODY0),
    P("<b>Constraint.</b> The rules forbid using evaluation-set split metadata to determine whether an "
      "image's true class is trained or novel, or to shrink its candidate set. One uniform procedure must "
      "apply to every image, emitting a label from the full 17,393-class space. Our system respects this, "
      "but is transductive in five other ways stated plainly in Sec. 4.7.", BODY0),
    P("<b>Budget.</b> Three real evaluations per day, thirty in total. A change reaches the leaderboard only "
      "after clearing a proxy-improvement bar and a class-disjoint cross-validation check. All internal "
      "measurements are labelled <i>proxy</i>, all leaderboard measurements <i>real</i>; Sec. 8 is the "
      "justification for insisting.", BODY0),
]

# ================================================================== 4 system
story += [
    P("4. System Description", H1),
    figure(f"{FIG}/pipeline_v109.png", 3.15,
           "<b>Figure 2.</b> Final system overview. The routing gate is now a learned 12-feature logistic "
           "classifier (Sec. 4.6, superseding the hand-derived rule of Sec. 4.5); both routes carry a "
           "leak-free re-ranker (Sec. 8.2&ndash;8.3) after the initial fusion score."),
    figure(f"{FIG}/flow_v109.png", 3.15,
           "<b>Figure 3.</b> The gate is a proxy for true class, not a measurement of it: 6,642 images "
           "(18.6%) are routed away from the branch that actually contains their true class, the same "
           "structural cost as Sec. 4.5's f = 0.60 identity, now shown per-image rather than in aggregate."),
    P("4.1&ndash;4.4. Closed-set and novel-class routes.", H2),
    P("Three fine-tuned BioCLIP-2.5 ViT-H/14 encoders are ensembled for the closed-set route "
      "(weights 1.0/2.5/2.5). Each member scores a trained class c by a genuine class prototype plus "
      "its best training exemplar plus a taxonomic text anchor:", BODY0),
    P("s<sub>c</sub> = &lt;e, p<sub>c</sub>&gt; + 2.0&middot;max<sub>i&isin;c</sub>&lt;e, x<sub>i</sub>&gt; "
      "+ 4.0&middot;&lt;e, t<sub>c</sub><sup>tax</sup>&gt;&nbsp;&nbsp;&nbsp;(2)", EQ),
    P("argmax over all 5,795 trained classes (not the 4,636 used in the adapter's training objective "
      "&mdash; Sec. 4.9). The novel-class route sums six text legs (four encoders &times; two prompt "
      "styles, plus TaxaBind) and three image legs: two external photograph banks (Sec. 6, weights "
      "4.0/3.0) and BioCLIP&nbsp;2 external prototypes (2.5 frozen, 2.0 LoRA-adapted), argmax over the "
      "disjoint 11,598. The two routes share no leg. Every raw similarity matrix S enters through", BODY0),
    P("dbnorm(S) = log softmax(S/0.05, <i>images</i>)<br/>+ log softmax(S/0.5, <i>classes</i>)"
      "&nbsp;&nbsp;&nbsp;(3)", EQ),
    P("the across-image term suppresses hub classes similar to everything and costs 1.60 novel-class "
      "points if removed. Novel-routed rows then pass through a Sinkhorn projection (&tau;=1.8, 60 "
      "iterations) toward a uniform column prior.", BODY0),
    P("4.5. Derived routing threshold (superseded)", H2),
    P("The original gate standardized an image-space and a text-space term, routing the top f = 0.60 of "
      "images by g to the closed-set branch:", BODY0),
    P("g = z(max<sub>trained</sub> s<sub>c</sub>) + 2.0&middot;z(m<sub>text</sub>)&nbsp;&nbsp;&nbsp;(4)", EQ),
    P("f is not a tuned hyper-parameter but follows from an identity. Writing s, u for the kept/ejected "
      "fractions and a, b for the two routes' conditional accuracies,", BODY0),
    P("Acc = w<sub>T</sub>&middot;s&middot;a + w<sub>U</sub>&middot;u&middot;b&nbsp;&nbsp;&nbsp;(5)", EQ),
    P("so the marginal image must clear purity q* = a/(a+b) (6) before ejecting it pays. At a=86.0%, "
      "b=19.5%, q*=81.5% put the optimum at f=0.72; once external retrieval (Sec. 6) raised b to 26.0%, "
      "q* fell to 77.2% and the optimum moved to f=0.60, worth +0.60 real points &mdash; a consequence of "
      "the retrieval work, not an independent discovery.", BODY0),
    P("4.6. Learned routing gate (v77&ndash;v109)", H2),
    P("A 12-feature logistic classifier &mdash; seen max similarity, top-1/top-2 margin, log-sum-exp "
      "entropy gap, per-encoder maxima, dual-text margin differences, and iNat photo-bank confidence "
      "&mdash; replaced Eq. 4 at the <i>same</i> f = 0.60, trained on a class-disjoint holdout. It raised "
      "held-out gate AUC from 0.957 to 0.9811 and real-evaluation-folder AUC from 0.9084 to <b>0.9314</b>, "
      "and is the only routing change in the project that improved both routes at once: trained-class "
      "accuracy rose from 75.67% to 77.72% and novel-class from 20.57% to 21.96% in the same submission "
      "(Table 7). Re-weighting the same 12 features more aggressively (C: 1&rarr;100, v79) moved the "
      "routing decomposition by a further +0.481pt projected but <i>reversed</i> the conditional-accuracy "
      "term by &minus;0.439pt, netting only +0.042 &mdash; giving back 90% of what the new-feature step had "
      "won. Sec. 8.3 generalizes this observation.", BODY0),
    tbl([["gate", "features", "AUC (real)", "real overall"],
         ["derived (Eq. 4)", "2", "0.9084", "51.62"],
         ["<b>learned, C=1 (v77)</b>", "<b>12</b>", "<b>0.9314</b>", "<b>53.383</b>"],
         ["learned, C=100 (v79)", "12 (reweighted)", "&mdash;", "53.425"]],
        [1.30, 0.86, 0.68, 0.60],
        "<b>Table 2.</b> Routing gate ablation. Adding features (v56&rarr;v77) improves trained- and "
        "novel-class accuracy together; re-weighting them (v77&rarr;v79) gives back most of the gain "
        "(Sec. 8.3).", align={1: "RIGHT", 2: "RIGHT", 3: "RIGHT"}),
    P("4.7. Transductivity, stated plainly", H2),
    P("The final system is batch-coupled at five points: the rank threshold (strongest), the across-image "
      "term of Eq. 3, the Sinkhorn column prior, and two batch-level score standardizations. Freezing the "
      "hub offset on a reference pool instead of the test batch changes 296 of 35,665 predictions (at most "
      "224 able to move the score, bounding the coupling at 0.63 overall points); the other four couplings "
      "are not similarly measured. Neither the learned gate nor either re-ranker (Sec. 4.6, 8.2&ndash;8.3) "
      "introduces new batch coupling &mdash; both score one image or one candidate at a time.", BODY0),
    P("4.8&ndash;4.9. Constant provenance and training.", H2),
    P("Fusion weights (Table in Sec. 4.1&ndash;4.4) were chosen by coordinate search on the Sec. 8.1 proxy "
      "and are not derived from anything; the routing fraction f and Sinkhorn temperature were instead set "
      "from real leaderboard feedback, because the proxy ranks them in the wrong order (Sec. 8.4). The "
      "learned gate and both re-rankers have a <i>third</i>, better-founded provenance: logistic weights "
      "fit on a class-disjoint holdout whose candidate pool was corrected to remove the leak of Sec. 8.2 "
      "&mdash; still a proxy, but one we can now show does not carry the population-identity shortcut. The "
      "contrastive adapter trains against 4,636 of the 5,795 trained classes with softmax cross-entropy "
      "against frozen text embeddings, scale 30, no margin term; the withheld 1,159 classes still build "
      "prototypes from their two images each and remain on the closed-set route at inference.", BODY0),
]

# ================================================================== 5 shift
story += [
    P("5. Framing Shift: Diagnosis and Two Corrections", H1),
    P("The closed-set route underperformed its held-out accuracy by roughly eight points on the real "
      "evaluation. Standard RandomResizedCrop sampled aspect ratios in [0.75, 1.33]; the evaluation's novel "
      "split has mean aspect 1.92 and 90th percentile 2.76 (Fig. 4) &mdash; fish are elongated, and the "
      "network never saw that framing. We corrected it twice, independently: widening the training range "
      "to scale (0.35, 1.0) / ratio (0.5, 2.0) (worth +0.4 real, Table 3), and scoring each query under "
      "seven inference geometries with a max-over-crops rule (+1.424 proxy, +0.174 real &mdash; Sec. 8.1 "
      "explains why those numbers are so far apart).", BODY0),
    figure(f"{FIG}/aspect_v109.png", 3.15,
           "<b>Figure 4.</b> Measured aspect-ratio distributions against the augmentation ranges. The "
           "original range (red) does not cover even the training median; the corrected range (blue) "
           "covers all three splits."),
    tbl([["configuration", "proxy", "corrected", "&Delta;"],
         ["LoRA adapter (224px)", "86.10", "86.70", "+0.60"],
         ["contrastive adapter, alone", "26.62", "31.10", "+4.48"],
         ["novel-class text stack", "29.38", "31.45", "+2.07"]],
        [1.55, 0.55, 0.65, 0.55],
        "<b>Table 3.</b> Effect of correcting the augmentation aspect range.",
        align={1: "RIGHT", 2: "RIGHT", 3: "RIGHT"}),
]

# ================================================================== 6 banks
story += [
    P("6. External-Bank Retrieval for Novel Classes", H1),
    P("Text-only novel-class accuracy plateaued at 10.42% real; eight separate attempts to improve it "
      "through better text all failed (Sec. 9). We instead assembled 117,225 photographs from the "
      "iNaturalist Open Data archive [12] over 7,364 species (99.9% name-match rate), capped at 24/class, "
      "101,545 embedded in the final bank. A class is scored by the mean of its top-4 cosine similarities "
      "&mdash; measured, not assumed: a plain class mean scores 43.49, a single nearest photograph 44.61, "
      "top-4 pooling 45.25, flattening beyond that because many classes hold only a handful of photographs.", BODY0),
    P("<b>Near-duplicate audit.</b> We compared all 35,665 evaluation embeddings against the bank's 101,106 "
      "photograph embeddings directly (an earlier version tested against training queries only, which does "
      "not address the objection). Calibrating on genuinely different same-species photographs (99th "
      "percentile 0.922, 0.302% of pairs above 0.95) against same-image-different-crop pairs (34.2% above "
      "0.95): the trained-class split has 0.219% of images above 0.95 (below the same-species chance rate, "
      "no evidence of duplication); the novel-class split has 0.906% (about 3&times; chance). Bounding the "
      "worst case as if every flagged image were free, novel-class accuracy could be inflated by at most "
      "0.91 of the +10.15 points attributed to retrieval &mdash; 8.9% of the claimed gain. The unqualified "
      "no-duplication claim from an earlier version of this work does not survive; the retrieval result, "
      "bounded, does.", BODY0),
    tbl([["configuration", "proxy", "real overall"],
         ["text only, no bank", "31.45", "49.04"],
         ["mean prototypes, w=3", "43.53", "50.46"],
         ["top-4 pooled + dual encoder", "46.81", "50.76"],
         ["+ archive merge, both legs", "47.58", "50.84"],
         ["<i>routing f = 0.72&rarr;0.60</i>", "&mdash;", "<i>51.44</i>"],
         ["+ crop views, 336px bank", "49.01", "51.62"]],
        [1.85, 0.60, 0.68],
        "<b>Table 4.</b> External-bank development. The italic row is the routing-fraction move, not a "
        "bank change.", align={1: "RIGHT", 2: "RIGHT"}),
    figure(f"{FIG}/operating_v109.png", 3.15,
           "<b>Figure 5.</b> Every scored submission in (trained, novel) accuracy space, axes reversed so "
           "better is up-left. Retrieval moves the system vertically; routing trades along a contour; the "
           "learned gate (v77&ndash;v109, purple) is the only segment that moves up <i>and</i> left "
           "at once."),
    figure(f"{FIG}/progress_v109.png", 3.15,
           "<b>Figure 6.</b> Real evaluation accuracy across all confirmed submissions. The purple segment "
           "(v77&ndash;v109) is the learned-gate and leak-free re-ranking era; f = 0.60 is unchanged "
           "throughout it."),
    tbl([["id", "change", "trained", "novel", "overall"],
         ["v22", "closed-set, shift-robust ensemble", "&mdash;", "&mdash;", "45.39"],
         ["v23", "combined image + text gate", "&mdash;", "&mdash;", "45.96"],
         ["v25", "+ shift-retrained adapter", "&mdash;", "&mdash;", "46.27"],
         ["v27", "+ 336 px fine-tune", "&mdash;", "&mdash;", "46.34"],
         ["v28", "routing f = 0.65", "&mdash;", "&mdash;", "47.26"],
         ["v31", "routing f = 0.72, per-image", "&mdash;", "&mdash;", "47.69"],
         ["v33", "+ Sinkhorn, novel route", "78.20", "8.51", "47.78"],
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
         ["v56", "+ crop views, 336 bank", "75.67", "20.57", "51.62"],
         ["v77", "<b>learned 12-feat. gate</b>", "77.72", "21.96", "53.383"],
         ["v79", "gate re-weighted (C=100)", "77.45", "22.41", "53.425"],
         ["v81", "re-ranker &mdash; LEAK (Sec. 8.2)", "77.45", "22.42", "53.431"],
         ["v82", "re-ranker, leak fixed", "77.45", "22.91", "53.644"],
         ["v83", "+ seen-route re-ranker", "77.54", "22.91", "53.697"],
         ["<b>v109</b>", "<b>+ genus backoff</b>", "<b>77.00</b>", "<b>23.77</b>", "<b>53.761</b>"]],
        [0.34, 1.66, 0.46, 0.42, 0.52],
        "<b>Table 5.</b> Complete ledger of scored real evaluations, v22 onward. Rows v34 and v39 are "
        "changes that cleared the proxy bar and then lost real accuracy (Sec. 8.1); v81 cleared it on a "
        "leaked proxy (Sec. 8.2). f = 0.60 is unchanged from v50 through v109; every point after v56 came "
        "from ranking quality, not the operating threshold.",
        align={2: "RIGHT", 3: "RIGHT", 4: "RIGHT"}, fs=6.3),
]

# ================================================================== 7 ceiling
story += [
    P("7. Where the Ceiling Lies", H1),
    P("A perfect-gate ceiling computed at the v36 configuration (s=u=1 in Eq. 5, with population-level "
      "conditional accuracies substituted for the currently-received ones) gave 53.5% against 49.0% as "
      "actually routed &mdash; the entire prize then available from solving routing, requiring gate AUC "
      "&approx;0.97 against an achievable 0.9084. The project's final 53.761% exceeded that historical "
      "target, but by moving the whole system (retrieval bank, learned gate at AUC 0.9314, leak-free "
      "re-ranking) rather than by reaching a near-perfect gate at the original operating point; per the "
      "project's own transfer-anchor accounting, both re-ranking heads are now within roughly 0.65 overall "
      "points of their measured re-ranking ceiling.", BODY0),
    figure(f"{FIG}/encoders_v109.png", 3.15,
           "<b>Figure 7.</b> Encoder comparison on the novel-class proxy. Domain pretraining dominates "
           "scale; correcting the augmentation geometry (Sec. 5) moved the adopted adapter further than any "
           "change of frozen backbone."),
    P("<b>Candidate ambiguity is not what remains.</b> Granting the novel route oracle family knowledge, "
      "collapsing 11,598 candidates to roughly two dozen, raises novel-class top-1 only from 28.90% to "
      "33.26% &mdash; +4.36 points from perfect taxonomic information, bounding what any better "
      "candidate-narrowing scheme can deliver. Family-level top-1 (23.8%) is itself lower than species-"
      "level (28.9%); the taxonomy is not uninformative, but the scoring procedure built on it is unusable "
      "as a prior. The residual is a visual-encoder limitation, not a routing or candidate-space one.", BODY0),
]

# ================================================================== 8 proxy
story += [
    P("8. Proxy Reliability", H1),
    P("8.1. The proxy, and two known failure modes", H2),
    P("Novel-class measurements use a pseudo-novel split: 1,159 rarest trained classes withheld entirely "
      "from training, 2,318 query images scored against the full 12,757-name candidate list, guarded by "
      "class-disjoint cross-validation over 954 held-out classes &mdash; a careful proxy. Even so, "
      "<b>magnitude</b> is not preserved (transfer ratios spanning 0.018 to 0.359, a 20&times; range; a "
      "Sinkhorn variant overstated its real gain 3.7&times;) and <b>sign</b> is not preserved (two changes "
      "gained proxy points and lost real ones). A simpler unified architecture that discards the routing "
      "split, evaluated against exact real anchors, was preferred by the proxy at 57.02 against the routed "
      "system's 50.71, and lost by at least 1.36 points in reality &mdash; ordering itself is not preserved.", BODY0),
    P("8.2. A third failure mode: population is not preserved", H2),
    P("The first attempt to re-rank novel-route candidates gained <b>+13.374</b> proxy points under "
      "class-disjoint 5-fold cross-validation and delivered <b>+0.006</b> real &mdash; a 2,229&times; "
      "overstatement, the worst mirage in the project. The cause: on the pseudo-novel proxy, gold is "
      "<i>always</i> one of the 1,159 withheld training classes &mdash; 9.1% of the candidate pool &mdash; "
      "and that subpopulation is identifiable from its own features (withheld classes have training images, "
      "hence distinctive prototype and bank statistics). The re-ranker learned to prefer gold-eligible "
      "candidates (53.7% of its picks, versus the fused baseline's 38.4%) rather than to rank visual and "
      "textual evidence. Class-disjoint cross-validation does not detect this: splitting <i>which</i> "
      "pseudo-classes fall in each fold still leaves every fold's gold inside the same 9.1% subpopulation. "
      "The fix restricts the candidate pool to gold-eligible classes only and replaces raw score levels with "
      "within-query percentile ranks, both changes needed to remove the identifiable-population shortcut. "
      "The corrected version scores +8.154 proxy and delivers <b>+0.213</b> real (Table 6) &mdash; smaller "
      "on the proxy, 35&times; better on transfer.", BODY0),
    callout("A held-out proxy that cleared every configuration shipped still overstated this one "
           "component's real value by <b>2,229&times;</b> &mdash; not from a coding error, but because "
           "class-disjoint cross-validation cannot see a leak in <i>which population</i> gold is drawn "
           "from. Magnitude of a holdout number is not evidence; the population it is measured on is.",
           tint=RED_T),
    tbl([["variant", "candidate pool", "proxy", "real"],
         ["leaky re-ranker", "gold always in 9.1% subpop.", "+13.374", "+0.006"],
         ["<b>leak-free (fixed)</b>", "<b>gold-eligible only</b>", "<b>+8.154</b>", "<b>+0.213</b>"]],
        [1.10, 1.30, 0.58, 0.50],
        "<b>Table 6.</b> The population leak and its fix, unseen route (v81 vs v82). Per-image decisions "
        "(the gate, Sec. 4.6) have no candidate subpopulation to exploit and are not subject to this "
        "failure mode; only per-candidate scoring is.", align={2: "RIGHT", 3: "RIGHT"}),
    P("Applying the same leak-free construction to the closed-set route (a 32-feature seen-head "
      "re-ranker, conditioned on class frequency) added a further +0.053 real (v83). Per-candidate "
      "re-ranking is not free of this risk in general: it is only trustworthy once the candidate pool is "
      "checked for exactly this kind of identifiability, a check class-disjoint cross-validation does not "
      "perform on its own.", BODY0),
    P("8.3. A predictive rule: new information compounds, re-weighting does not", H2),
    P("Two independent components let us test whether a proxy gain's <i>kind</i>, not its size, predicts "
      "real transfer. The routing gate (Sec. 4.6) added 10 new features to the 2 the derived rule used "
      "&mdash; new information &mdash; and its real gain decomposed as +1.082pt routing +0.686pt conditional "
      "accuracy = +1.767pt, both terms positive. Re-weighting the same 12 features more aggressively "
      "(C:1&rarr;100) is not new information, and its real gain decomposed as +0.481pt routing "
      "&minus;0.439pt conditional = +0.042pt, giving back 90% of what the feature addition won. "
      "Independently, the v109 genus-level hierarchical backoff added sibling-species similarity absent "
      "from the base fusion &mdash; new information &mdash; and transferred positively (+0.064 real) from a "
      "holdout margin (+0.101pt) inside the noise band that an axis-matched re-weighting attempt (v98/v99) "
      "had already shown could reverse on the real evaluation (+1.34 holdout, &minus;0.17 real).", BODY0),
    callout("Interventions that add <b>new information</b> the system did not have compound with the rest "
           "of the fusion and transfer positively, twice confirmed; interventions that only "
           "<b>re-weight</b> information it already has return most of their proxy gain, or reverse it. "
           "This is the rule we would spend a submission budget on trusting.", tint=GRN_T),
    tbl([["intervention", "kind", "holdout / proxy", "real"],
         ["gate: +10 features (v77)", "new info", "&mdash;", "<b>+1.767</b>"],
         ["gate: C 1&rarr;100 (v79)", "re-weight", "&mdash;", "+0.042"],
         ["genus backoff (v109)", "new info", "+0.101pt", "<b>+0.064</b>"],
         ["seen fusion reweight (v98/9)", "re-weight", "+1.34pt", "&minus;0.17"]],
        [1.55, 0.62, 0.72, 0.55],
        "<b>Table 7.</b> Mechanism class predicts transfer, 2-for-2: interventions that add information the "
        "system did not have compound with the rest of the fusion; interventions that redistribute weight "
        "over information it already had return most of the proxy gain, or reverse it.",
        align={2: "RIGHT", 3: "RIGHT"}, fs=6.7),
    figure(f"{FIG}/transfer_v109.png", 3.15,
           "<b>Figure 8.</b> Proxy gain against real gain, extended with the leak-free re-rankers (purple). "
           "<b>(a)</b> v81's leaked measurement is off-scale at (13.374, 0.006) &mdash; the point Sec. 8.2 "
           "explains. <b>(b)</b> Transfer rate by component; the two leak-free re-rankers (0.0261, 0.0295) "
           "sit in the same range as the encoder-level changes despite scoring candidates rather than "
           "images, once the population leak is closed."),
    P("8.4. What we conclude", H2),
    P("Held-out proxies are not useless &mdash; every decision in this paper used one &mdash; but "
      "reliability is not a fixed property of the validation setup. It degrades where the proxy's "
      "<i>population</i>, not just its distribution, diverges from the real one, and per-candidate scoring "
      "is exactly where a population can leak through features that look like ordinary evidence. Under a "
      "constrained budget we would now spend real evaluations on exactly two axes &mdash; anything touching "
      "the external-retrieval domain gap, and any new per-candidate scorer's first deployment &mdash; and "
      "trust the proxy on re-weighting existing signals and on encoder-level changes.", BODY0),
]

# ================================================================== 9 negative
story += [
    P("9. Catalogue of Negative Results", H1),
    P("Listed compactly because the aggregate is the point.", BODY0),
    tbl([["attempt", "result"],
         ["General backbone (SigLIP2 SO400M)", "5.13 vs 21.53"],
         ["BioTrove-CLIP (two variants)", "0.00 / 0.04"],
         ["Common-name / generic-ensemble prompts", "8.89 / 22.00 vs 26.62"],
         ["LLM-compressed trait text / raw descriptions", "17.08 / 7.51 vs 23.38"],
         ["Hard two-stage family narrowing", "&minus;20 points"],
         ["VLM top-K rerank (Qwen2.5-VL, DeepSeek)", "&minus;16.0 / &minus;11.33"],
         ["CORAL / quantile gate domain adaptation (v93)", "harness valid, no lift"],
         ["Unsupervised embedding whitening (v102)", "at or below raw baseline"],
         ["HistGradientBoosting vs. logistic re-ranker", "&minus;0.008 novel / +0.003 seen"],
         ["Seen-route new-evidence sweep (31 mechanisms)", "1 survivor (genus, Sec. 8.3)"],
         ["Sinkhorn re-added atop re-ranked route", "not re-tested since v81"],
         ["Soft (non-hard) routing", "proxy +1.9, routing degrades"]],
        [1.90, 1.00],
        "<b>Table 8.</b> Selected measured negative results (of 24 total); the full text-representation "
        "ablation from the earlier configuration is unchanged and omitted here for space.",
        align={1: "RIGHT"}, fs=6.8),
    figure(f"{FIG}/deadlevers_v109.png", 3.15,
           "<b>Figure 9.</b> The five largest same-baseline losses from Table 8, plus the one row where "
           "proxy and real evaluation disagreed on direction &mdash; the same proxy failure mode Sec. 8 "
           "treats at length, showing up a second time outside the re-ranker story."),
    P("Also ruled out, with no single comparable number: BioTrove-CLIP (0.00 / 0.04, near-total failure), "
      "CORAL domain adaptation (harness valid, no lift), unsupervised whitening (at or below baseline), "
      "the HistGB re-ranker (&minus;0.008, no meaningful change), the 31-mechanism seen-route sweep (1 "
      "survivor), and Sinkhorn re-added atop the re-ranked route (untested since v81).", CAP),
    P("<b>The adapter-selection inversion.</b> One result deserves separate mention because it "
      "reproduces the theme of Sec. 8 inside a single component, and because the obvious reading of "
      "it is wrong in an instructive way. Seven adapters for the external-prototype encoder were "
      "scored standalone. Ranked by raw standalone score the best reaches 15.53 while the final one "
      "reaches only 14.50 &mdash; yet fused into the system the first contributes +0.00 and the "
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
        [1.30, 0.46, 0.46, 0.46, 0.46],
        "<b>Table 9.</b> Adapter variants for the external-prototype encoder. Ranking by <i>best</i> "
        "puts the name-text variant first; ranking by <i>delta</i> over each run's own baseline puts "
        "the adopted variant first, and only the latter agrees with the fused contribution.",
        align={1: "RIGHT", 2: "RIGHT", 3: "RIGHT", 4: "RIGHT"}, fs=6.7),
]

# ================================================================== 10 impl
story += [
    P("10. Implementation Details", H1),
    tbl([["component", "setting", "component", "setting"],
         ["closed-set backbone", "BioCLIP-2.5 ViT-H/14 &times;3", "routing gate", "12-feat. logistic, f=0.60"],
         ["prototype backbone", "BioCLIP&nbsp;2 ViT-L/14", "unseen re-ranker", "34 feat., leak-free pool"],
         ["external bank", "117,225 photos / 7,364 cls", "seen re-ranker", "32 feat., leak-free pool"],
         ["bank scoring", "mean of top-4 cosines", "genus backoff", "LAM_G = 3.0, sibling pool"],
         ["Sinkhorn", "&tau; = 1.8, 60 iterations", "inference views", "7 (center, squash, strips)"]],
        [0.86, 1.00, 0.72, 1.04],
        "<b>Table 10.</b> Final evaluated configuration.", fs=6.5),
]

# ================================================================== 11 limits
story += [
    P("11. Threats to Validity", H1),
    risk_tbl([
        ("Adaptation to the eval. set", "HIGH",
         "Self-declared the most serious risk: 30 sequential evaluations with no Ladder protection; "
         "routing fraction, Sinkhorn &tau;, and aug. correction all set from real feedback."),
        ("No significance tests", "MED",
         "Every real number is one run of one config; sub-0.1pt rows (several in Table 5) are "
         "directional at best."),
        ("Sequential tuning, component count", "MED",
         "Routing fraction, fusion weights, and bank weight tuned one at a time against a proxy Sec. "
         "8 shows is unreliable on exactly those axes; final point not claimed jointly optimal."),
        ("Domain gap (retrieval bank)", "MED",
         "Bank photos and eval. images demonstrably do not share a distribution; Sec. 8 argues this "
         "is the main reason proxy measurements on that branch mislead."),
        ("Asymmetric augmentation", "MED",
         "Query views have no horizontal flips, bank photos do; not ablated, magnitude unmeasured."),
        ("Scope (generalization)", "MED",
         "One benchmark, one taxonomic domain, one encoder family; the mechanism-class-predicts-"
         "transfer result (Sec. 8.3) is the finding least demonstrated to generalize."),
        ("Batch coupling", "LOW",
         "Transductive at five points (Sec. 4.7); the Sinkhorn step's own measured contribution "
         "(+0.09 real) sits inside the paper's own noise band."),
        ("Cost (unmeasured)", "LOW",
         "Throughput, memory, and energy were not measured &mdash; a reporting gap, not a threat to "
         "the accuracy numbers themselves."),
    ], "<b>Table 11.</b> Severity drawn from this section's own language, not assigned independently: "
       "HIGH is the paper's self-declared top risk; MED are bounded or unmeasured-magnitude caveats; "
       "LOW are measured-small or reporting-only gaps."),
    P("<b>Adaptation to the evaluation set</b> is the most serious limitation and touches every real number "
      "here: thirty sequential leaderboard evaluations without the Ladder mechanism's [20] protection; the "
      "routing fraction and Sinkhorn temperature set from real feedback (Sec. 4.8); and the augmentation "
      "correction (Sec. 5) derived by measuring the evaluation images' own aspect distribution. The learned "
      "gate and both re-rankers add a further, specific risk this paper is largely <i>about</i>: they are "
      "themselves fit on a proxy, and Sec. 8.2 shows that proxy can silently leak. We checked the pool this "
      "leaked through and fixed it; we cannot rule out a different, undiscovered leak in the same features.", BODY0),
    P("<b>No significance tests.</b> Every real number is one evaluation of one configuration. The final "
      "genus-backoff change nets an estimated 23 images out of 35,665 (0.064pt &times; 35,665); several "
      "confirmed gains in Table 5 are similarly small in absolute terms. A paired test is the right "
      "instrument, but McNemar needs the discordant-pair split by direction, which the labels do not give "
      "us; sub-0.1-point rows should be read as directional at best.", BODY0),
    P("<b>Sequential tuning and component count.</b> The routing fraction, the fusion weights, and the "
      "bank weight were tuned one at a time against a proxy that Sec. 8 shows is unreliable on exactly "
      "those axes, so the final point is not claimed to be jointly optimal. The system now carries three "
      "closed-set encoders, six text legs, two image banks, two prototype sets, two normalizations, a "
      "learned routing gate, and two leak-free re-rankers &mdash; more interacting parts than the earlier "
      "configuration, not fewer. With that many parts a chronological ledger of submissions (Table 5) is "
      "not a controlled factorial experiment, and we have tried to confine causal language to places "
      "where an ablation actually exists.", BODY0),
    P("<b>Batch coupling.</b> The final configuration is transductive at the same five points as the "
      "earlier one (Sec. 4.7); the learned gate and re-rankers add none of their own. The Sinkhorn step "
      "imposes a near-uniform prior over 11,598 novel classes for 15,568 images, which the true label "
      "distribution certainly violates; it raised vocabulary usage from 38.6% to 57.5% and was worth "
      "+0.09 real points &mdash; inside the noise band of the paragraph above.", BODY0),
    P("<b>Domain gap.</b> The audit of Sec. 6 bounds near-duplication; it does not show that bank "
      "photographs and evaluation images come from the same distribution. They demonstrably do not, and "
      "Sec. 8 argues this gap is the main reason proxy measurements on that branch mislead. A held-out "
      "split built from the bank itself would test prototype construction without testing the gap, which "
      "is why we did not build one.", BODY0),
    P("<b>Asymmetric augmentation.</b> Query images are scored under seven views with no horizontal "
      "flips while bank photographs are embedded with flips and squashing; we did not ablate this. As "
      "noted in Sec. 5, the five long-axis views collapse to one on square images.", BODY0),
    P("<b>Cost.</b> The final configuration runs three ViT-H/14 encoders plus a ViT-L/14 and a ViT-B/16 "
      "over seven crop geometries per query, against 101,106 stored bank embeddings, followed by a "
      "normalization, a Sinkhorn projection, a 12-feature logistic gate, and two re-rankers &mdash; all "
      "lightweight relative to the encoders. We did not measure throughput, memory, or energy, which is "
      "a real gap in the reporting.", BODY0),
    P("<b>Scope.</b> One benchmark, one taxonomic domain, one encoder family, and now eleven paired "
      "proxy-to-real measurements from a single adaptively-tuned development history. The "
      "mechanism-class-predicts-transfer result (Sec. 8.3) is the finding we would most expect to "
      "generalize and the one we can least demonstrate generalizes; a second benchmark with a comparable "
      "real-evaluation budget is the experiment that would settle it, and we have not run it.", BODY0),
    P("12. Conclusion", H1),
    P("A generalized zero-shot recognition system over 17,393 classes, two-thirds with no training images, "
      "reached 53.761% overall under a thirty-evaluation budget used to measure rather than assume. Framing-"
      "shift correction and external retrieval produced most of the early gain; a learned routing gate then "
      "improved both routes at once, the only lever in the project to do so. The result we consider most "
      "useful to others is methodological: a held-out proxy that cleared every configuration we shipped "
      "nonetheless overstated one component's real value by 2,229&times;, through a mechanism &mdash; "
      "candidate-population identifiability &mdash; that class-disjoint cross-validation does not catch; "
      "fixing it recovered a genuine, well-transferring signal. Combined with a second confirmed instance, "
      "this supports a general rule for budgeted evaluation: interventions that add information the system "
      "did not have are worth testing on a proxy; interventions that only re-weight what it already has, "
      "and any new per-candidate scorer's untested candidate pool, are exactly where a proxy should not be "
      "trusted alone.", BODY0),
]

# ================================================================== refs
REFS = [
    'W. J. Scheirer, A. de Rezende Rocha, A. Sapkota, and T. E. Boult. Toward open set recognition. '
    '<i>IEEE TPAMI</i>, 35(7):1757&ndash;1772, 2013.',
    'C. Geng, S.-J. Huang, and S. Chen. Recent advances in open set recognition: A survey. '
    '<i>IEEE TPAMI</i>, 43(10):3614&ndash;3631, 2021.',
    'Y. Xian, C. H. Lampert, B. Schiele, and Z. Akata. Zero-shot learning &mdash; a comprehensive '
    'evaluation of the good, the bad and the ugly. <i>IEEE TPAMI</i>, 41(9):2251&ndash;2265, 2019.',
    'A. Radford, J. W. Kim, C. Hallacy, et al. Learning transferable visual models from natural '
    'language supervision. In <i>ICML</i>, 2021.',
    'S. Stevens, J. Wu, M. J. Thompson, et al. BioCLIP: A vision foundation model for the tree of '
    'life. In <i>CVPR</i>, 2024.',
    'J. Gu, S. Stevens, E. Campolongo, et al. BioCLIP 2: Emergent properties from scaling '
    'hierarchical contrastive learning. In <i>NeurIPS</i>, 2025.',
    'E. J. Hu, Y. Shen, P. Wallis, et al. LoRA: Low-rank adaptation of large language models. In '
    '<i>ICLR</i>, 2022.',
    'H. Wang, Y. Wang, Z. Zhou, et al. CosFace: Large margin cosine loss for deep face recognition. '
    'In <i>CVPR</i>, 2018.',
    'J. Deng, J. Guo, N. Xue, and S. Zafeiriou. ArcFace: Additive angular margin loss for deep face '
    'recognition. In <i>CVPR</i>, 2019.',
    'J. Snell, K. Swersky, and R. Zemel. Prototypical networks for few-shot learning. In '
    '<i>NeurIPS</i>, 2017.',
    'S. Sastry, S. Khanal, A. Dhakal, A. Ahmad, and N. Jacobs. TaxaBind: A unified embedding space '
    'for ecological applications. In <i>WACV</i>, 2025.',
    'G. Van Horn, O. Mac Aodha, Y. Song, et al. The iNaturalist species classification and '
    'detection dataset. In <i>CVPR</i>, 2018.',
    'M. Cuturi. Sinkhorn distances: Lightspeed computation of optimal transport. In <i>NeurIPS</i>, 2013.',
    'M. Caron, I. Misra, J. Mairal, et al. Unsupervised learning of visual features by contrasting '
    'cluster assignments. In <i>NeurIPS</i>, 2020.',
    'A. Lyzhov, Y. Molchanova, A. Ashukha, D. Molchanov, and D. Vetrov. Greedy policy search: A '
    'simple baseline for learnable test-time augmentation. In <i>UAI</i>, 2020.',
    'X.-S. Wei, Y.-Z. Song, O. Mac Aodha, et al. Fine-grained image analysis with deep learning: A '
    'survey. <i>IEEE TPAMI</i>, 44(12):8927&ndash;8948, 2022.',
    'F. F. Khan, X. Li, A. J. Temple, and M. Elhoseiny. FishNet: A large-scale dataset and '
    'benchmark for fish recognition, detection, and functional trait prediction. In <i>ICCV</i>, 2023.',
    'C. Dwork, V. Feldman, M. Hardt, T. Pitassi, O. Reingold, and A. Roth. The reusable holdout: '
    'Preserving validity in adaptive data analysis. <i>Science</i>, 349(6248):636&ndash;638, 2015.',
    'C. Dwork, V. Feldman, M. Hardt, T. Pitassi, O. Reingold, and A. Roth. Preserving statistical '
    'validity in adaptive data analysis. In <i>STOC</i>, 2015.',
    'A. Blum and M. Hardt. The ladder: A reliable leaderboard for machine learning competitions. '
    'In <i>ICML</i>, 2015.',
    'R. Roelofs, V. Shankar, B. Recht, S. Fridovich-Keil, M. Hardt, J. Miller, and L. Schmidt. A '
    'meta-analysis of overfitting in machine learning. In <i>NeurIPS</i>, 2019.',
    'B. Recht, R. Roelofs, L. Schmidt, and V. Shankar. Do ImageNet classifiers generalize to '
    'ImageNet? In <i>ICML</i>, 2019.',
    'J. P. Miller, R. Taori, A. Raghunathan, et al. Accuracy on the line: On the strong correlation '
    'between out-of-distribution and in-distribution generalization. In <i>ICML</i>, 2021.',
    'D. Teney, Y. Lin, S. J. Oh, and E. Abbasnejad. ID and OOD performance are sometimes inversely '
    'correlated on real-world datasets. In <i>NeurIPS</i>, 2023.',
    'W.-L. Chao, S. Changpinyo, B. Gong, and F. Sha. An empirical study and analysis of generalized '
    'zero-shot learning for object recognition in the wild. In <i>ECCV</i>, 2016.',
    'J. Song, C. Shen, Y. Yang, Y. Liu, and M. Song. Transductive unbiased embedding for zero-shot '
    'learning. In <i>CVPR</i>, 2018.',
    'M. Radovanovi&#263;, A. Nanopoulos, and M. Ivanovi&#263;. Hubs in space: Popular nearest '
    'neighbors in high-dimensional data. <i>JMLR</i>, 11:2487&ndash;2531, 2010.',
    'S. L. Smith, D. H. P. Turban, S. Hamblin, and N. Y. Hammerla. Offline bilingual word vectors, '
    'orthogonal transformations and the inverted softmax. In <i>ICLR</i>, 2017.',
    'S.-V. Bogolin, I. Croitoru, H. Jin, Y. Liu, and S. Albanie. Cross modal retrieval with '
    'querybank normalisation. In <i>CVPR</i>, 2022.',
    'A. Long, W. Yin, T. Ajanthan, et al. Retrieval augmented classification for long-tail visual '
    'recognition. In <i>CVPR</i>, 2022.',
    'V. Udandarao, A. Gupta, and S. Albanie. SuS-X: Training-free name-only transfer of '
    'vision-language models. In <i>ICCV</i>, 2023.',
    'K. S. Tai, P. Bailis, and G. Valiant. Sinkhorn label allocation: Semi-supervised '
    'classification via annealed self-training. In <i>ICML</i>, 2021.',
    'G. Van Horn, E. Cole, S. Beery, K. Wilber, S. Belongie, and O. Mac Aodha. Benchmarking '
    'representation learning for natural world image collections. In <i>CVPR</i>, 2021.',
]
story += [P("References", H1)]
story += [P(f"[{i+1}] {r}", REF) for i, r in enumerate(REFS)]

# ============================================================== appendix
story += [
    P("Appendix A: Reference Implementation", H1),
    P("Three excerpts from the actual deployed source, reformatted only for column width (line "
      "breaks added, no logic changed), to make the three mechanisms this paper is about legible "
      "without reading the full codebase. Variable names are exactly as shipped.", BODY0),
    P("A.1. Routing gate: predict, rank-quantile threshold, disjoint argmax (Sec. 4.5&ndash;4.6)", H2),
    code(
"""# builders/build_v77_learned_gate.py
Z = standardize(X, torch.full((X.shape[0],), 1.0 / X.shape[0]))
combined = torch.tensor(
    gate['model'].predict_proba(Z.numpy())[:, 1],
    dtype=torch.float32).to(dev)

k_seen = int(round(SEEN_FRAC * len(all_files)))     # f = 0.60
thr = torch.topk(combined, k_seen).values.min()
route_seen = combined >= thr                        # Eq. 4 quantile

pred_idx = torch.empty(len(all_files), dtype=torch.long,
                       device=dev)
# closed-set argmax over the 5,795 trained classes
pred_idx[route_seen] = kept_idx[
    seen_block[route_seen].argmax(1)]
# retrieval argmax over the disjoint 11,598, via Sinkhorn
idx_uns = (~route_seen).nonzero(as_tuple=True)[0]
pred_idx[idx_uns] = other_idx[
    sinkhorn(text_unseen_only[idx_uns], tau=TAU)
        .argmax(1)]
""",
        "<b>Listing 1.</b> The entire routing decision: a logistic classifier's probability, "
        "thresholded at the f=0.60 sample quantile (not a fixed confidence &mdash; Sec. 4.5), then "
        "two disjoint argmaxes. This is Eq. 4&ndash;5 exactly as executed."),
    P("A.2. Leak-free candidate pool and pool-size-invariant features (Sec. 8.2)", H2),
    code(
"""# research/rerank_leakfree_v82.py
# Fix: restrict the holdout pool to the 1,159 pseudo
# classes ONLY -- every candidate is then gold-eligible,
# so the population shortcut carries zero information.
pool = torch.tensor(sorted(D.ci[c] for c in D.pseudo))

def pool_features(legs, topk, n_photos, fused):
    \"\"\"Pool-size-invariant version of the v81 features.\"\"\"
    nq, k = topk.shape
    C = fused.shape[1]                  # pool size: 1,159
    cols, names = [], []                #   here, 11,598 real
    for name in LEGS:                   #   eval -- must match
        M = legs[name]
        v = torch.gather(M, 1, topk)
        cols += [v - M.max(1, keepdim=True).values,
                 (v - M.mean(1, keepdim=True))
                    / (M.std(1, keepdim=True) + 1e-6)]
        names += [f'{name}_gapmax', f'{name}_z']
        r = torch.empty(nq, k, device=M.device)
        for s in range(0, nq, 256):
            e = min(s + 256, nq)
            r[s:e] = (M[s:e].unsqueeze(1)
                      > v[s:e].unsqueeze(2)).sum(2).float()
        cols.append(r / C)          # PERCENTILE rank, not
        names.append(f'{name}_pctrank')  # log-rank -- pool
    return torch.stack(cols, dim=2), names   # -size invariant
""",
        "<b>Listing 2.</b> The fix for the population leak of Sec. 8.2: the candidate pool is "
        "rebuilt to contain only gold-eligible classes, and every rank feature is a percentile "
        "(rank/pool-size) rather than a raw or log rank, so training on a 1,159-class pool and "
        "scoring an 11,598-class one are the same computation."),
    P("A.3. Genus-sibling backoff, the one new-evidence lever that shipped as v109 (Sec. 8.3)", H2),
    code(
"""# research/genus_rerank_v103_train.py
# mean-pool the genus-leg training features by GENUS,
# not by species -- a sibling-species prototype
TL_genus = species_genus_idx[TL_by[GENUS_LEG]]
Pg = torch.zeros(G, d, device=dev)
cntg = torch.zeros(G, device=dev)
Pg.scatter_add_(0, TL_genus.unsqueeze(1).expand(-1, d),
                TF_by[GENUS_LEG])
cntg.scatter_add_(0, TL_genus,
                  torch.ones_like(TL_genus,
                                  dtype=torch.float32))
Pg = F.normalize(Pg / cntg.clamp(min=1).unsqueeze(1), dim=-1)

q = qF_by[GENUS_LEG]
sim = q @ TF_by[GENUS_LEG].t()
cmaxg = torch.full((q.shape[0], G), -1e9, device=dev)
cmaxg.scatter_reduce_(1,
    TL_genus.unsqueeze(0).expand(q.shape[0], -1),
    sim, reduce='amax')
genus_raw_G = q @ Pg.t() + 2.0 * cmaxg    # proto + 2*cmax,
genus_bonus = genus_raw_G[:, species_genus_idx]  # same form
fused = block + LAM_G * zc(genus_bonus)   # as Eq. 2, LAM_G=3
""",
        "<b>Listing 3.</b> Genus backoff pools training features across every <i>sibling</i> "
        "species sharing a genus, scores a query against that pooled prototype, and adds it as a "
        "4th signal group (LAM_G=3.0) to the existing seen-head fusion &mdash; genuinely new "
        "evidence (sibling-species similarity), not a re-weighting of the 3 legs already there, "
        "which is why it transferred positively (Sec. 8.3, Table 7)."),
]

# ================================================================== build
PW, PH = letter
LM = RM = 0.62 * inch
TM, BM = 0.62 * inch, 0.62 * inch
GUT = 0.24 * inch
CW = (PW - LM - RM - GUT) / 2
TITLE_H = 2.85 * inch


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Times-Roman", 8.0)
    canvas.drawCentredString(PW / 2, BM - 0.28 * inch, str(canvas.getPageNumber()))
    canvas.restoreState()


doc = BaseDocTemplate(OUT, pagesize=letter, leftMargin=LM, rightMargin=RM,
                      topMargin=TM, bottomMargin=BM,
                      title="Zero-Shot Species Recognition at 17,393 Classes (v109)")

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

_ruled_story = []
for _item in story:
    if isinstance(_item, Paragraph) and getattr(_item, "style", None) is H1:
        _ruled_story.append(HRFlowable(width="100%", thickness=1.3, color=colors.HexColor(BLU),
                                       spaceBefore=2, spaceAfter=3, lineCap="round"))
    _ruled_story.append(_item)

doc.build(_ruled_story)
print("WROTE", OUT, os.path.getsize(OUT), "bytes")
