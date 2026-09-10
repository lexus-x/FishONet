#!/usr/bin/env python
"""Figures for the FishONet technical report, in the Specimen Ledger design system.

Every number here is copied from make_v109_paper.py, which traces each one to
HANDOFF.md or to a direct reading of the builder / training source. No number is
invented, estimated, or recomputed here -- this module only restyles.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "report_assets")
FONTS = os.path.join(OUT, "fonts")
os.makedirs(OUT, exist_ok=True)

for _f in os.listdir(FONTS):
    if _f.endswith(".ttf"):
        font_manager.fontManager.addfont(os.path.join(FONTS, _f))

# ------------------------------------------------------------------ tokens
ABYSS   = "#0A1F26"
MUTED   = "#5C6B68"
RULE    = "#C3CEC9"
PAPER   = "#EFF3F1"
PANEL   = "#E4EAE6"
TRAINED = "#1B6E7A"   # a training photograph exists
NOVEL   = "#C25E00"   # no photograph anywhere
FLAG    = "#A32E38"   # failed / at risk
GAIN    = "#2F6B4F"

MAIN_W = 5.35   # inches, the reading measure (136mm)
WIDE_W = 6.85   # inches, rail + measure (174mm)

plt.rcParams.update({
    # Google serves this face declaring itself "Archivo SemiBold"; it is Archivo.
    "font.family": "Archivo SemiBold",
    "font.size": 7.2,
    "text.color": ABYSS,
    "axes.labelcolor": ABYSS,
    "axes.edgecolor": RULE,
    "axes.linewidth": 0.8,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelsize": 6.8,
    "ytick.labelsize": 6.8,
    "axes.labelsize": 7.0,
    "legend.fontsize": 6.8,
    "figure.facecolor": "none",
    "axes.facecolor": "none",
    "savefig.facecolor": "none",
})


def _clean(ax, ytitle=None, keep_left=True):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if not keep_left:
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    ax.tick_params(length=2.4, pad=2.4)
    if ytitle:
        ax.set_ylabel(ytitle)


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.png", dpi=340, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print("wrote", name)


# =========================================================== the register
def _hex(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], dtype=np.uint8)


def fig_register(cols=165, cell=12, gut=3, name="register"):
    """One cell per candidate class, 17,393 of them. The signature element.

    A tick per species is thinner than a printer dot at page width, so the
    register is laid out as a plate instead: every cell is one species, teal
    where a training photograph exists, amber where none does anywhere.
    """
    n_trained, n_novel = 5795, 11598
    total = n_trained + n_novel
    rows = int(np.ceil(total / cols))

    rng = np.random.default_rng(17393)
    state = np.zeros(cols * rows, dtype=np.uint8)      # 0 = absent from catalogue
    state[:total] = 1                                  # 1 = novel
    photographed = rng.choice(total, n_trained, replace=False)
    state[photographed] = 2                            # 2 = photographed
    grid = state.reshape(rows, cols)

    img = np.full((rows * cell, cols * cell, 3), _hex(PAPER), dtype=np.uint8)
    colors = {1: _hex(NOVEL), 2: _hex(TRAINED)}
    for value, rgb in colors.items():
        ys, xs = np.nonzero(grid == value)
        for y, x in zip(ys, xs):
            y0, x0 = y * cell, x * cell
            img[y0:y0 + cell - gut, x0:x0 + cell - gut] = rgb

    h_in = WIDE_W * (rows * cell) / (cols * cell)
    fig, ax = plt.subplots(figsize=(WIDE_W, h_in))
    ax.imshow(img, interpolation="nearest")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(f"{OUT}/{name}.png", dpi=300, bbox_inches="tight",
                pad_inches=0, transparent=True)
    plt.close(fig)
    print("wrote", name, f"{cols}x{rows} cells")


# ================================================= 1 / data composition
def fig_composition():
    fig, ax = plt.subplots(figsize=(MAIN_W, 1.42))
    ax.set_xlim(0, 100); ax.set_ylim(-0.62, 1.62); ax.axis("off")
    rows = [(1.0, "candidate classes", "17,393", 5795, 11598, 33.3, 66.7),
            (0.0, "evaluation images", "35,665", 20097, 15568, 56.35, 43.65)]
    for y, label, tot, a, b, pa, pb in rows:
        ax.add_patch(Rectangle((0, y - 0.22), pa, 0.44, fc=TRAINED, ec="none"))
        ax.add_patch(Rectangle((pa, y - 0.22), pb, 0.44, fc=NOVEL, ec="none"))
        ax.text(-1.5, y + 0.05, label, ha="right", va="center", fontsize=7.0, color=ABYSS)
        ax.text(-1.5, y - 0.13, tot, ha="right", va="center", fontsize=6.4,
                color=MUTED, family="IBM Plex Mono")
        ax.text(pa / 2, y, f"{a:,}   {pa:.1f}%", ha="center", va="center",
                fontsize=6.6, color="white", family="IBM Plex Mono")
        ax.text(pa + pb / 2, y, f"{b:,}   {pb:.1f}%", ha="center", va="center",
                fontsize=6.6, color="white", family="IBM Plex Mono")
    ax.text(0, 1.44, "photographed", fontsize=6.6, color=TRAINED, weight="600")
    ax.text(100, 1.44, "never photographed", fontsize=6.6, color=NOVEL,
            weight="600", ha="right")
    ax.text(50, 0.5, "two thirds of the catalogue, but under half the images",
            fontsize=6.3, color=MUTED, ha="center", va="center", style="italic")
    save(fig, "composition")


# ============================================================ 2 / routing
NT, NU = 20097, 15568
KT, ET = 17426, 2671
EU, KU = 11597, 3971


def _ribbon(ax, x0, x1, t0, b0, t1, b1, color, alpha):
    t = np.linspace(0, 1, 160)
    sm = t * t * (3 - 2 * t)
    ax.fill_between(x0 + (x1 - x0) * t, b0 + (b1 - b0) * sm, t0 + (t1 - t0) * sm,
                    color=color, alpha=alpha, lw=0, zorder=2)


def fig_routing():
    fig, ax = plt.subplots(figsize=(MAIN_W, 2.15))
    ax.set_xlim(-0.30, 1.34); ax.set_ylim(-0.12, 1.10); ax.axis("off")
    G = 0.055
    hT, hU = NT / 35665, NU / 35665
    Lt1, Lt0 = 1.0, 1.0 - hT
    Lu1, Lu0 = Lt0 - G, Lt0 - G - hU
    hS, hR = (KT + KU) / 35665, (ET + EU) / 35665
    Rs1, Rs0 = 1.0, 1.0 - hS
    Rr1, Rr0 = Rs0 - G, Rs0 - G - hR
    xa, xb = 0.16, 0.84

    for x, y0, y1, c, lab, n in [(0.0, Lt0, Lt1, TRAINED, "true class\nphotographed", NT),
                                 (0.0, Lu0, Lu1, NOVEL, "true class\nnever photographed", NU)]:
        ax.add_patch(Rectangle((x, y0), 0.045, y1 - y0, fc=c, ec="none", zorder=4))
        ax.text(x - 0.02, (y0 + y1) / 2 + 0.03, lab, ha="right", va="center",
                fontsize=6.5, color=ABYSS)
        ax.text(x - 0.02, (y0 + y1) / 2 - 0.055, f"{n:,}", ha="right", va="center",
                fontsize=6.4, color=MUTED, family="IBM Plex Mono")
    for x, y0, y1, c, lab, n in [(1.255, Rs0, Rs1, TRAINED, "closed-set route", KT + KU),
                                 (1.255, Rr0, Rr1, NOVEL, "retrieval route", ET + EU)]:
        ax.add_patch(Rectangle((x, y0), 0.045, y1 - y0, fc=c, ec="none", zorder=4))
        ax.text(x + 0.06, (y0 + y1) / 2 + 0.03, lab, ha="left", va="center",
                fontsize=6.5, color=ABYSS)
        ax.text(x + 0.06, (y0 + y1) / 2 - 0.055, f"{n:,}", ha="left", va="center",
                fontsize=6.4, color=MUTED, family="IBM Plex Mono")

    tt = Lt1 - KT / 35665
    _ribbon(ax, xa, xb, Lt1, tt, Rs1, Rs1 - KT / 35665, TRAINED, .34)
    _ribbon(ax, xa, xb, tt, Lt0, Rr1, Rr1 - ET / 35665, FLAG, .48)
    uu = Lu1 - KU / 35665
    _ribbon(ax, xa, xb, Lu1, uu, Rs1 - KT / 35665, Rs0, FLAG, .48)
    _ribbon(ax, xa, xb, uu, Lu0, Rr1 - ET / 35665, Rr0, NOVEL, .34)

    ax.text(0.50, Lt1 - KT / 71000 + 0.012, f"{KT:,} kept", ha="center",
            fontsize=6.4, color="white", family="IBM Plex Mono")
    ax.text(0.50, Lu0 + EU / 71000 - 0.018, f"{EU:,} ejected", ha="center",
            fontsize=6.4, color="white", family="IBM Plex Mono")
    ax.annotate(f"{ET + KU:,} images ({(ET + KU) / 35665 * 100:.1f}%) reach a route\n"
                f"whose candidate set cannot contain their species",
                xy=(0.50, (tt + Lt0) / 2 - 0.012), xytext=(0.50, -0.10),
                ha="center", fontsize=6.4, color=FLAG,
                arrowprops=dict(arrowstyle="-", lw=0.8, color=FLAG))
    save(fig, "routing")


# =========================================================== 3 / the climb
def fig_climb():
    labels = ["v22", "v31", "v33", "v36", "v46", "v50", "v56", "v77", "v79", "v81", "v82", "v83", "v109"]
    overall = [45.39, 47.69, 47.78, 49.04, 50.84, 51.44, 51.62,
               53.383, 53.425, 53.431, 53.644, 53.697, 53.736]
    novel = [None, None, 8.51, 10.42, 14.55, 19.34, 20.57,
             21.96, 22.41, 22.42, 22.91, 22.91, 22.91]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(WIDE_W, 2.62), sharex=True,
                                 gridspec_kw={"height_ratios": [1.15, 1]})
    x = list(range(len(labels)))
    a1.plot(x[:7], overall[:7], "-o", ms=3.4, lw=1.3, color=ABYSS, zorder=3)
    a1.plot(x[6:], overall[6:], "-o", ms=3.4, lw=1.3, color=TRAINED, zorder=3)
    a1.axhline(50.56, ls=(0, (4, 3)), lw=0.9, color=MUTED)
    a1.text(0.08, 50.78, "contemporaneous leader   50.56", fontsize=6.2, color=MUTED)
    a1.text(12, 53.95, "53.736", fontsize=7.4, color=TRAINED, ha="right",
            family="IBM Plex Mono", weight="600")
    a1.set_ylim(44.6, 55.0)
    _clean(a1, "overall top-1  %")
    a2.plot([i for i in x if i >= 2 and i < 7], [novel[i] for i in x if i >= 2 and i < 7],
            "-s", ms=3.2, lw=1.3, color=NOVEL, zorder=3)
    a2.plot([i for i in x if i >= 6], [novel[i] for i in x if i >= 6],
            "-s", ms=3.2, lw=1.3, color=TRAINED, zorder=3)
    a2.set_ylim(6, 24.5)
    _clean(a2, "novel-class top-1  %")
    a2.set_xticks(x); a2.set_xticklabels(labels, fontsize=6.3, family="IBM Plex Mono")
    fig.tight_layout(pad=0.2, h_pad=0.5)
    save(fig, "climb")


# ======================================================= 4 / the trade
SUBS = [("v33", 78.20, 8.51), ("v36", 78.95, 10.42), ("v46", 78.95, 14.55),
        ("v50", 76.31, 19.34), ("v56", 75.67, 20.57),
        ("v77", 77.72, 21.96), ("v79", 77.45, 22.41), ("v81", 77.45, 22.42),
        ("v82", 77.45, 22.91), ("v83", 77.54, 22.91), ("v109", 77.61, 22.91)]


def fig_tradeoff():
    fig, ax = plt.subplots(figsize=(MAIN_W, 2.45))
    for lvl in (48, 50, 52, 54):
        xs = np.array([74.0, 81.0])
        ax.plot(xs, (lvl - 0.5635 * xs) / 0.4365, lw=0.7, ls=(0, (3, 3)),
                color=RULE, zorder=1)
        yl = (lvl - 0.5635 * 80.4) / 0.4365
        if 6.6 < yl < 24.5:
            ax.text(80.4, yl + 0.2, f"{lvl}%", fontsize=6.1, color=MUTED,
                    ha="left", va="bottom", family="IBM Plex Mono")
    early = SUBS[:5]
    late = SUBS[4:]
    ax.plot([s[1] for s in early], [s[2] for s in early], "-", lw=1.0, color=RULE, zorder=2)
    ax.plot([s[1] for s in late], [s[2] for s in late], "-", lw=1.2, color=TRAINED, zorder=2)
    for name, s, u in SUBS:
        late_pt = name not in ("v33", "v36", "v46", "v50", "v56")
        ax.plot(s, u, "o", ms=4.2, color="white",
                mec=TRAINED if late_pt else ABYSS, mew=1.3, zorder=4)
    lab = {"v33": (0, -0.95, "center"), "v36": (0, -0.95, "center"),
           "v46": (0.32, 0.2, "left"), "v50": (0, 0.8, "center"),
           "v56": (0.24, -1.0, "left"), "v77": (0.32, -0.6, "left"),
           "v109": (0.28, 0.85, "left")}
    for name, (dx, dy, ha) in lab.items():
        s, u = next((q[1], q[2]) for q in SUBS if q[0] == name)
        ax.text(s + dx, u + dy, name, fontsize=6.5, ha=ha, va="center",
                family="IBM Plex Mono",
                color=TRAINED if name in ("v77", "v109") else ABYSS)
    ax.annotate("the learned gate moves\nup and left at once",
                xy=(77.9, 22.3), xytext=(80.4, 24.9),
                fontsize=6.5, color=TRAINED, ha="left", va="top",
                arrowprops=dict(arrowstyle="-", lw=0.9, color=TRAINED))
    ax.set_xlim(81.3, 74.4); ax.set_ylim(6.4, 25.6)
    ax.set_xlabel("trained-class top-1  %          better <-")
    _clean(ax, "novel-class top-1  %          better ->")
    fig.tight_layout(pad=0.2)
    save(fig, "tradeoff")


# ========================================================= 5 / framing
def fig_framing():
    fig, ax = plt.subplots(figsize=(MAIN_W, 1.62))
    rows = [("training crops\n61,941 img", 0.75, 1.4618, 1.9110),
            ("eval, photographed\n20,097 img", 0.9974, 1.4988, 2.1192),
            ("eval, novel\n15,568 img", 1.3531, 1.7439, 2.7586)]
    ax.axvspan(0.75, 1.33, color=FLAG, alpha=0.11, lw=0, zorder=1)
    ax.axvspan(0.50, 2.00, color=TRAINED, alpha=0.09, lw=0, zorder=0)
    for i, (lab, p10, p50, p90) in enumerate(rows):
        y = 2 - i
        ax.plot([p10, p90], [y, y], color=ABYSS, lw=1.8, solid_capstyle="butt", zorder=3)
        for e in (p10, p90):
            ax.plot([e, e], [y - .15, y + .15], color=ABYSS, lw=1.1, zorder=3)
        ax.plot(p50, y, "o", ms=4.2, color="white", mec=ABYSS, mew=1.3, zorder=4)
        ax.text(p90 + 0.06, y, f"p90 {p90:.2f}", va="center", fontsize=6.4,
                color=MUTED, family="IBM Plex Mono")
    ax.set_yticks([2, 1, 0]); ax.set_yticklabels([r[0] for r in rows], fontsize=6.5)
    ax.set_xlim(0.42, 3.25); ax.set_ylim(-1.0, 2.62)
    ax.set_xlabel("aspect ratio, width / height")
    ax.text(1.04, -0.78, "what the network\nwas trained on", fontsize=6.2,
            color=FLAG, ha="center", va="center")
    ax.text(2.52, -0.78, "corrected range", fontsize=6.2, color=TRAINED,
            ha="center", va="center")
    _clean(ax, keep_left=False)
    fig.tight_layout(pad=0.2)
    save(fig, "framing")


# ========================================================= 6 / encoders
def fig_encoders():
    rows = [("BioCLIP-2.5-H", 21.53), ("BioCLIP 2", 14.37), ("TaxaBind ViT-B/16", 11.48),
            ("BioCLIP 1", 11.60), ("BioCAP", 6.77), ("SigLIP2 SO400M", 5.13),
            ("BioCLIP hyperbolic", 4.36), ("general CLIP", 0.78),
            ("bioclip-inat-only", 0.35), ("BioTrove-CLIP", 0.00)]
    fig, ax = plt.subplots(figsize=(MAIN_W, 2.30))
    y = list(range(len(rows)))[::-1]
    vals = [r[1] for r in rows]
    ax.barh(y, vals, height=0.55, color=RULE, zorder=3)
    ax.barh([y[0]], [vals[0]], height=0.55, color=TRAINED, zorder=4)
    for yy, v in zip(y, vals):
        ax.text(v + 0.5, yy, f"{v:.2f}", va="center", fontsize=6.3,
                color=MUTED, family="IBM Plex Mono")
    ax.barh([-1.7], [31.10], height=0.62, color=TRAINED, zorder=3)
    ax.text(31.6, -1.7, "31.10", va="center", fontsize=6.4, color=TRAINED,
            family="IBM Plex Mono", weight="600")
    ax.text(0.4, -1.7, "adopted adapter, corrected framing", va="center",
            fontsize=6.4, color="white", zorder=5)
    ax.axhline(-0.85, lw=0.8, color=RULE)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=6.5)
    ax.set_xlim(0, 36.5); ax.set_ylim(-2.4, 9.9)
    ax.set_xlabel("novel-class top-1 on the held-out proxy  %")
    _clean(ax, keep_left=False)
    fig.tight_layout(pad=0.2)
    save(fig, "encoders")


# ========================================================= 7 / transfer
def fig_transfer():
    pts = [("v40", 1.68, 0.03), ("v43", 0.52, 0.19), ("v46", 0.345, 0.067),
           ("v56", 1.424, 0.174), ("v82", 8.154, 0.213), ("v83", 1.803, 0.053)]
    neg = [("v34", 3.51, -0.07), ("v39", 0.52, -0.11)]
    off = {"v43": (6, 4), "v46": (-7, -10), "v56": (6, -9), "v40": (6, 3),
           "v82": (-36, 6), "v83": (6, -9)}
    fig, ax = plt.subplots(figsize=(MAIN_W, 2.05))
    ax.axhline(0, color=ABYSS, lw=0.9, zorder=2)
    ax.axhspan(-0.20, 0, color=FLAG, alpha=0.07, lw=0)
    for r, st in ((0.36, (0, (1, 2))), (0.12, (0, (5, 2))), (0.026, (0, (1, 1)))):
        ax.plot([0, 8.6], [0, 8.6 * r], lw=0.8, ls=st, color=RULE, zorder=1)
    for n, x, y in pts:
        ax.plot(x, y, "o", ms=4.6, color=TRAINED, zorder=4)
        ax.annotate(n, (x, y), textcoords="offset points", xytext=off.get(n, (6, 4)),
                    fontsize=6.4, color=ABYSS, family="IBM Plex Mono", zorder=5)
    for n, x, y in neg:
        ax.plot(x, y, "x", ms=5.6, color=FLAG, mew=1.6, zorder=4)
        ax.annotate(n, (x, y), textcoords="offset points", xytext=(7, -3),
                    fontsize=6.4, color=FLAG, family="IBM Plex Mono", va="center", zorder=5)
    ax.annotate("v81, before the leak was found:\n+13.374 proxy, +0.006 real",
                xy=(8.52, 0.045), xytext=(2.7, 0.36), fontsize=6.4, color=FLAG,
                arrowprops=dict(arrowstyle="->", lw=0.9, color=FLAG))
    ax.set_xlim(-0.20, 8.9); ax.set_ylim(-0.20, 0.60)
    ax.set_xlabel("gain on the held-out proxy  points")
    _clean(ax, "gain on the real evaluation  points")
    fig.tight_layout(pad=0.2)
    save(fig, "transfer")


# ==================================================== 8 / negative results
def fig_negatives():
    bars = [("two-stage family narrowing", -20.0),
            ("common-name / generic prompts", -17.73),
            ("SigLIP2 general backbone", -16.40),
            ("VLM rerank, Qwen2.5-VL", -16.0),
            ("LLM raw description text", -15.87)]
    fig, ax = plt.subplots(figsize=(MAIN_W, 2.05))
    y = list(range(len(bars)))[::-1]
    vals = [b[1] for b in bars]
    ax.barh(y, vals, height=0.54, color=FLAG, zorder=3)
    for yy, v in zip(y, vals):
        ax.text(v + 0.45, yy, f"{v:.2f}", va="center", ha="left", fontsize=6.3,
                color="white", family="IBM Plex Mono")
    ax.axvline(0, color=ABYSS, lw=0.9, zorder=4)
    ax.axhspan(-1.58, -0.52, color=PANEL, alpha=0.75, lw=0)
    ax.barh([-1.05], [1.9], height=0.40, color=GAIN, zorder=3)
    ax.text(2.25, -1.05, "proxy  +1.9", va="center", fontsize=6.3, color=GAIN,
            family="IBM Plex Mono")
    ax.text(-0.55, -1.05, "real: degrades", va="center", ha="right", fontsize=6.3,
            color=FLAG, family="IBM Plex Mono")
    ax.set_yticks(y + [-1.05])
    ax.set_yticklabels([b[0] for b in bars] + ["soft, non-hard routing"], fontsize=6.5)
    ax.get_yticklabels()[-1].set_style("italic")
    ax.set_xlim(-24, 5.6); ax.set_ylim(-1.85, 4.75)
    ax.set_xlabel("novel-class proxy points against the adopted configuration")
    _clean(ax, keep_left=False)
    fig.tight_layout(pad=0.2)
    save(fig, "negatives")


if __name__ == "__main__":
    fig_register()
    fig_composition()
    fig_routing()
    fig_climb()
    fig_tradeoff()
    fig_framing()
    fig_encoders()
    fig_transfer()
    fig_negatives()
