#!/usr/bin/env python
"""Figures for the FishONet design report.

Rules for this figure system:
  * no outlined boxes, no chart chrome that is not carrying information
  * every form's geometry is proportional to a real measured quantity
  * two hues only in data: teal = a training photograph exists,
    clay = none exists anywhere. Red is reserved for measured failure.

Every number is copied from make_v109_paper.py, which traces each one to
HANDOFF.md or to a direct reading of builder / training source. Nothing here
is estimated; where only summary statistics exist (framing percentiles) the
figure draws only those statistics and does not invent a distribution shape.
"""
import os
import json
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.collections import LineCollection
from matplotlib.path import Path
from matplotlib.patches import PathPatch, Circle

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "design_assets")
FONTS = os.path.join(OUT, "fonts")
os.makedirs(OUT, exist_ok=True)
for _f in os.listdir(FONTS):
    if _f.endswith(".ttf"):
        font_manager.fontManager.addfont(os.path.join(FONTS, _f))

# ------------------------------------------------------------------ tokens
GROUND = "#FAF7F2"
DARK   = "#14100D"
INK    = "#221A15"
MUTED  = "#7A6E65"
FAINT  = "#A99C91"
RULE   = "#E3DAD0"
TEAL   = "#1F6F6B"
TEAL_L = "#4E9B93"
CLAY   = "#C96442"
CLAY_L = "#E0A183"
FLAG   = "#8F2D2D"
GAIN   = "#3F6B4A"

SANS = "Inter"
MONO = "IBM Plex Mono"
SERIF = "Instrument Serif"

COL_W = 5.27    # inches — the reading measure (134mm)
FULL_W = 6.85   # inches — full text width (174mm)

plt.rcParams.update({
    "font.family": SANS,
    "font.size": 7.0,
    "text.color": INK,
    "axes.labelcolor": MUTED,
    "axes.edgecolor": RULE,
    "axes.linewidth": 0.7,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelsize": 6.4,
    "ytick.labelsize": 6.4,
    "axes.labelsize": 6.8,
    "figure.facecolor": "none",
    "axes.facecolor": "none",
    "savefig.facecolor": "none",
})


def save(fig, name, dpi=360):
    fig.savefig(f"{OUT}/{name}.png", dpi=dpi, bbox_inches="tight",
                transparent=True, pad_inches=0.02)
    plt.close(fig)
    print("  ", name)


def rgb(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)])


def bare(ax):
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])


def hairline(ax, x0, y0, x1, y1, c=RULE, lw=0.6):
    ax.plot([x0, x1], [y0, y1], color=c, lw=lw, zorder=1,
            solid_capstyle="round")


def gradient_shape(ax, verts, c_left, c_right, alpha=1.0, horizontal=True, zorder=2):
    """Fill an arbitrary polygon with a smooth two-colour gradient."""
    verts = np.asarray(verts)
    patch = PathPatch(Path(verts), facecolor="none", edgecolor="none", zorder=zorder)
    ax.add_patch(patch)
    grad = np.linspace(0, 1, 512)
    grad = np.vstack([grad] * 2) if horizontal else np.vstack([grad] * 2).T
    a, b = rgb(c_left), rgb(c_right)
    img = a[None, None, :] * (1 - grad[..., None]) + b[None, None, :] * grad[..., None]
    im = ax.imshow(img, extent=[verts[:, 0].min(), verts[:, 0].max(),
                                verts[:, 1].min(), verts[:, 1].max()],
                   origin="lower", aspect="auto", zorder=zorder, alpha=alpha,
                   interpolation="bilinear")
    im.set_clip_path(patch)
    return im


def smoothstep(n=200):
    t = np.linspace(0, 1, n)
    return t, t * t * (3 - 2 * t)


def ribbon(ax, x0, x1, a0, a1, b0, b1, c_left, c_right, alpha=0.95, zorder=2):
    """A flowing band from (a0,a1) at x0 to (b0,b1) at x1, gradient-filled."""
    t, s = smoothstep()
    xs = x0 + (x1 - x0) * t
    top = a1 + (b1 - a1) * s
    bot = a0 + (b0 - a0) * s
    verts = np.concatenate([np.stack([xs, top], 1),
                            np.stack([xs[::-1], bot[::-1]], 1)])
    gradient_shape(ax, verts, c_left, c_right, alpha=alpha, zorder=zorder)


# ======================================================= 1 / the catalogue
def fig_fan(name="fan"):
    """The catalogue as one swept arc, divided by the true proportion.

    A hairline per species is far below one printed pixel at any page width,
    so the arc is divided rather than drawn line-by-line; the faint radial
    texture is decorative and carries no count.
    """
    allc = pickle.load(open(os.path.join(HERE, "..", "data/dl/all_classes.pkl"), "rb"))
    lab = json.load(open(os.path.join(HERE, "..", "data/dl/label_train.json")))
    seen = set(lab.values())
    n_photo = sum(1 for c in allc if c in seen)          # ground truth
    n_none = len(allc) - n_photo
    frac = n_photo / len(allc)

    fig, ax = plt.subplots(figsize=(FULL_W, FULL_W * 0.40))
    span = np.deg2rad(150)
    a_hi = np.pi / 2 + span / 2
    a_lo = np.pi / 2 - span / 2
    a_mid = a_hi - span * frac
    r0, r1 = 0.52, 1.0

    def sector(a_from, a_to, colour, alpha):
        aa = np.linspace(a_from, a_to, 400)
        verts = np.concatenate([
            np.stack([np.cos(aa) * r1, np.sin(aa) * r1], 1),
            np.stack([np.cos(aa[::-1]) * r0, np.sin(aa[::-1]) * r0], 1)])
        ax.fill(verts[:, 0], verts[:, 1], color=colour, alpha=alpha, lw=0, zorder=2)
        return aa

    sector(a_hi, a_mid, TEAL, 0.92)
    sector(a_mid, a_lo, CLAY, 0.92)
    # faint radial texture, purely optical
    for aa in np.linspace(a_lo, a_hi, 260):
        ax.plot([np.cos(aa) * r0, np.cos(aa) * r1], [np.sin(aa) * r0, np.sin(aa) * r1],
                color=GROUND, lw=0.35, alpha=0.30, zorder=3)

    am = (a_hi + a_mid) / 2
    ac = (a_mid + a_lo) / 2
    for a, num, lab_txt, colour in ((am, f"{n_photo:,}", "have at least one\ntraining photograph", TEAL),
                                    (ac, f"{n_none:,}", "have none, anywhere\nin the data", CLAY)):
        rx, ry = np.cos(a) * 1.10, np.sin(a) * 1.10
        ha = "right" if np.cos(a) < 0 else "left"
        nudge = -0.06 if ha == "right" else 0.06
        ax.text(rx + nudge, ry + 0.05, num, ha=ha, va="bottom", fontsize=13.5,
                color=colour, family=MONO, weight="600")
        ax.text(rx + nudge, ry + 0.025, lab_txt, ha=ha, va="top", fontsize=6.5,
                color=MUTED, linespacing=1.5)
    ax.text(0, 0.30, f"{len(allc):,}", ha="center", va="center", fontsize=16,
            color=INK, family=MONO, weight="600")
    ax.text(0, 0.20, "species in the catalogue", ha="center", va="center",
            fontsize=6.8, color=MUTED)
    ax.set_xlim(-1.72, 1.72); ax.set_ylim(0.10, 1.52)
    ax.set_aspect("equal")
    bare(ax)
    save(fig, name, dpi=420)


# ====================================================== 2 / composition
def fig_composition(name="composition"):
    """Two proportional bands, joined by flows that taper rather than cross."""
    fig, ax = plt.subplots(figsize=(COL_W, 2.05))
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.34, 1.34)
    bare(ax)

    top_y, bot_y, h = 0.92, 0.16, 0.075
    t_share, i_share = 5795 / 17393, 20097 / 35665
    t, s_ = smoothstep()

    # tapering flows first, so the bands sit over them
    ax.fill(np.concatenate([[0], t_share + (i_share - t_share) * s_, [0]]),
            np.concatenate([[bot_y + h], top_y - (top_y - bot_y - h) * (1 - s_), [top_y]]),
            color=TEAL, alpha=0.16, lw=0, zorder=1)
    ax.fill(np.concatenate([[1], t_share + (i_share - t_share) * s_, [1]]),
            np.concatenate([[bot_y + h], top_y - (top_y - bot_y - h) * (1 - s_), [top_y]]),
            color=CLAY, alpha=0.16, lw=0, zorder=1)

    for y, split in ((top_y, t_share), (bot_y, i_share)):
        ax.fill_between([0, split], y, y + h, color=TEAL, lw=0, zorder=3)
        ax.fill_between([split, 1], y, y + h, color=CLAY, lw=0, zorder=3)

    def pair(y, x, num, txt, colour, above):
        sign = 1 if above else -1
        base = y + h + 0.045 if above else y - 0.045
        ax.text(x, base, num, ha="center", va="bottom" if above else "top",
                fontsize=11.5, color=colour, family=MONO, weight="600")
        ax.text(x, base + sign * 0.235, txt, ha="center",
                va="bottom" if above else "top", fontsize=6.4, color=MUTED)

    pair(top_y, t_share / 2, "5,795", "with photographs", TEAL, True)
    pair(top_y, (1 + t_share) / 2, "11,598", "with none", CLAY, True)
    pair(bot_y, i_share / 2, "20,097", "of photographed species", TEAL, False)
    pair(bot_y, (1 + i_share) / 2, "15,568", "of species with none", CLAY, False)

    ax.text(-0.03, top_y + h / 2, "the catalogue\n17,393 species", ha="right",
            va="center", fontsize=6.5, color=INK, linespacing=1.5)
    ax.text(-0.03, bot_y + h / 2, "the evaluation set\n35,665 images", ha="right",
            va="center", fontsize=6.5, color=INK, linespacing=1.5)
    save(fig, name)


# ========================================================== 3 / pipeline
def fig_pipeline(name="pipeline"):
    """The system as one continuous flow. Stream width is image count."""
    total, keep, eject = 35665, 21399, 14266
    fig, ax = plt.subplots(figsize=(FULL_W, 2.55))
    ax.set_xlim(0, 1.0); ax.set_ylim(-0.80, 0.88)
    bare(ax)

    H = 0.26
    kh = H * 2 * keep / total
    eh = H * 2 * eject / total
    x_in, gate, x_sp, x_end = 0.005, 0.30, 0.52, 0.70
    gap = 0.055

    ribbon(ax, x_in, gate, -H, H, -H, H, "#CFD9D6", "#AFC3BF", alpha=0.75)
    ribbon(ax, gate, x_sp, H - kh, H, gap, gap + kh, "#AFC3BF", TEAL)
    ribbon(ax, gate, x_sp, -H, -H + eh, -gap - eh, -gap, "#D9BCAC", CLAY)
    ribbon(ax, x_sp, x_end, gap, gap + kh, gap, gap + kh, TEAL, TEAL)
    ribbon(ax, x_sp, x_end, -gap - eh, -gap, -gap - eh, -gap, CLAY, CLAY)

    ax.text(x_in, H + 0.28, "35,665", fontsize=13.5, color=INK, family=MONO, weight="600")
    ax.text(x_in, H + 0.155, "every evaluation image", fontsize=6.8, color=MUTED)

    ax.plot([gate, gate], [-H - 0.04, H + 0.04], color=INK, lw=0.8, zorder=6)
    ax.text(gate, -H - 0.12, "a learned gate reads twelve signals\noff each image and keeps the top 60%",
            fontsize=6.6, color=INK, ha="center", va="top", linespacing=1.6)

    ax.text(x_end + 0.02, gap + kh / 2 + 0.03, "21,399", fontsize=12,
            color=TEAL, family=MONO, weight="600", va="bottom")
    ax.text(x_end + 0.02, gap + kh / 2 + 0.018, "scored against the 5,795\nspecies with photographs",
            fontsize=6.5, color=MUTED, va="top", linespacing=1.5)
    ax.text(x_end + 0.02, -gap - eh / 2 + 0.03, "14,266", fontsize=12,
            color=CLAY, family=MONO, weight="600", va="bottom")
    ax.text(x_end + 0.02, -gap - eh / 2 + 0.018, "scored against the 11,598\nspecies with none",
            fontsize=6.5, color=MUTED, va="top", linespacing=1.5)

    ax.text(0.5, -0.71, "the two candidate sets are disjoint  ·  no image is ever scored "
                        "against all 17,393 at once",
            ha="center", fontsize=6.6, color=MUTED, style="italic")
    save(fig, name)


# =========================================================== 4 / routing
def fig_routing(name="routing"):
    NT, NU = 20097, 15568
    KT, ET = 17426, 2671
    EU, KU = 11597, 3971
    T = 35665
    fig, ax = plt.subplots(figsize=(COL_W, 2.30))
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.14, 1.06)
    bare(ax)
    G = 0.05
    hT, hU = NT / T, NU / T
    Lt1, Lt0 = 1.0, 1.0 - hT
    Lu1, Lu0 = Lt0 - G, Lt0 - G - hU
    hS, hR = (KT + KU) / T, (ET + EU) / T
    Rs1, Rs0 = 1.0, 1.0 - hS
    Rr1, Rr0 = Rs0 - G, Rs0 - G - hR
    xa, xb = 0.20, 0.80

    tt = Lt1 - KT / T
    ribbon(ax, xa, xb, tt, Lt1, Rs1 - KT / T, Rs1, TEAL, TEAL, alpha=0.42)
    ribbon(ax, xa, xb, Lt0, tt, Rr1 - ET / T, Rr1, FLAG, FLAG, alpha=0.55)
    uu = Lu1 - KU / T
    ribbon(ax, xa, xb, uu, Lu1, Rs0, Rs1 - KT / T, FLAG, FLAG, alpha=0.55)
    ribbon(ax, xa, xb, Lu0, uu, Rr0, Rr1 - ET / T, CLAY, CLAY, alpha=0.42)

    for y0, y1, c in ((Lt0, Lt1, TEAL), (Lu0, Lu1, CLAY)):
        ax.fill_betweenx([y0, y1], -0.005, 0.008, color=c, lw=0)
    for y0, y1, c in ((Rs0, Rs1, TEAL), (Rr0, Rr1, CLAY)):
        ax.fill_betweenx([y0, y1], 0.992, 1.005, color=c, lw=0)

    ax.text(-0.02, (Lt0 + Lt1) / 2, "20,097\nphotographed species", ha="right",
            va="center", fontsize=6.5, color=INK, linespacing=1.5)
    ax.text(-0.02, (Lu0 + Lu1) / 2, "15,568\nspecies with none", ha="right",
            va="center", fontsize=6.5, color=INK, linespacing=1.5)
    ax.text(1.02, (Rs0 + Rs1) / 2, "21,397\nclosed-set route", ha="left",
            va="center", fontsize=6.5, color=INK, linespacing=1.5)
    ax.text(1.02, (Rr0 + Rr1) / 2, "14,268\nretrieval route", ha="left",
            va="center", fontsize=6.5, color=INK, linespacing=1.5)

    ax.text(0.5, -0.10, f"{ET + KU:,} images  ({(ET + KU) / T * 100:.1f}%)  reach a route "
                        f"that cannot contain their species",
            ha="center", fontsize=6.6, color=FLAG)
    save(fig, name)


# ============================================================= 5 / climb
LABELS = ["v22", "v31", "v33", "v36", "v46", "v50", "v56", "v77", "v79", "v81", "v82", "v83", "v109"]
OVERALL = [45.39, 47.69, 47.78, 49.04, 50.84, 51.44, 51.62,
           53.383, 53.425, 53.431, 53.644, 53.697, 53.736]
NOVELACC = [None, None, 8.51, 10.42, 14.55, 19.34, 20.57,
            21.96, 22.41, 22.42, 22.91, 22.91, 22.91]


def fig_climb(name="climb"):
    fig, ax = plt.subplots(figsize=(FULL_W, 2.55))
    x = np.arange(len(LABELS))
    ax.set_xlim(-0.35, len(LABELS) - 0.65)
    ax.set_ylim(44.2, 55.4)

    xs = np.linspace(0, len(LABELS) - 1, 400)
    ys = np.interp(xs, x, OVERALL)
    verts = np.concatenate([np.stack([xs, ys], 1),
                            np.stack([xs[::-1], np.full_like(xs, 44.2)], 1)])
    gradient_shape(ax, verts, "#EDE3DA", TEAL_L, alpha=0.30, zorder=1)

    ax.plot(x, OVERALL, color=INK, lw=1.4, zorder=4, solid_capstyle="round")
    ax.scatter(x, OVERALL, s=13, color=GROUND, edgecolors=INK, linewidths=1.0, zorder=5)
    ax.scatter([x[-1]], [OVERALL[-1]], s=42, color=TEAL, edgecolors="none", zorder=6)

    ax.axhline(50.56, color=FAINT, lw=0.7, ls=(0, (3, 3)), zorder=2)
    ax.text(0.05, 50.85, "public leader at the time   50.56", fontsize=6.3, color=MUTED)

    ax.text(len(LABELS) - 1, 54.6, "53.736", ha="right", fontsize=14, color=TEAL,
            family=MONO, weight="600")
    ax.text(len(LABELS) - 1, 54.35, "final", ha="right", va="top", fontsize=6.4, color=MUTED)
    ax.text(0, 44.9, "45.39", fontsize=8.5, color=MUTED, family=MONO)

    ax.annotate("the gate is learned\nrather than derived", xy=(7, 53.383),
                xytext=(4.15, 54.3), fontsize=6.5, color=TEAL, ha="center",
                arrowprops=dict(arrowstyle="-", lw=0.7, color=TEAL_L,
                                connectionstyle="arc3,rad=0.2"))
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_yticks([46, 48, 50, 52, 54])
    ax.tick_params(length=0, pad=3)
    ax.set_xticks(x); ax.set_xticklabels(LABELS, fontsize=6.2, family=MONO, color=FAINT)
    for gy in (46, 48, 50, 52, 54):
        ax.axhline(gy, color=RULE, lw=0.5, zorder=0)
    ax.set_ylabel("overall accuracy  %", fontsize=6.6)
    save(fig, name)


# ============================================================= 6 / trade
SUBS = [("v33", 78.20, 8.51), ("v36", 78.95, 10.42), ("v46", 78.95, 14.55),
        ("v50", 76.31, 19.34), ("v56", 75.67, 20.57), ("v77", 77.72, 21.96),
        ("v79", 77.45, 22.41), ("v81", 77.45, 22.42), ("v82", 77.45, 22.91),
        ("v83", 77.54, 22.91), ("v109", 77.61, 22.91)]


def fig_trade(name="trade"):
    fig, ax = plt.subplots(figsize=(COL_W, 2.75))
    gx = np.linspace(74.4, 81.3, 300)
    gy = np.linspace(6.4, 25.6, 300)
    GX, GY = np.meshgrid(gx, gy)
    score = 0.5635 * GX + 0.4365 * GY
    ax.imshow(score, extent=[74.4, 81.3, 6.4, 25.6], origin="lower", aspect="auto",
              cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
                  "warm", ["#FAF7F2", "#EFE6DC", "#CFDCD9", TEAL_L]),
              alpha=0.34, zorder=0)
    cs = ax.contour(GX, GY, score, levels=[48, 50, 52, 54], colors=[FAINT],
                    linewidths=0.55, zorder=1)
    ax.clabel(cs, fmt="%d%%", fontsize=5.8, inline=True, colors=[MUTED])

    early = [s for s in SUBS if s[0] in ("v33", "v36", "v46", "v50", "v56")]
    late = [s for s in SUBS if s not in early]
    ax.plot([s[1] for s in early], [s[2] for s in early], color=FAINT, lw=1.0, zorder=3)
    ax.plot([early[-1][1]] + [s[1] for s in late], [early[-1][2]] + [s[2] for s in late],
            color=TEAL, lw=1.3, zorder=3)
    for nm, sx, sy in SUBS:
        is_late = (nm, sx, sy) in late
        ax.scatter(sx, sy, s=26, color=TEAL if is_late else GROUND,
                   edgecolors=TEAL if is_late else INK, linewidths=1.0, zorder=5)
    for nm, dx, dy, ha in (("v33", 0, -0.95, "center"), ("v36", 0, -0.95, "center"),
                           ("v46", 0.28, 0.25, "left"), ("v50", 0, 0.85, "center"),
                           ("v56", 0.22, -0.95, "left"), ("v77", 0.3, -0.7, "left"),
                           ("v109", 0.26, 0.85, "left")):
        s = next(q for q in SUBS if q[0] == nm)
        ax.text(s[1] + dx, s[2] + dy, nm, fontsize=6.3, ha=ha, va="center",
                family=MONO, color=TEAL if nm in ("v77", "v109") else INK)
    ax.annotate("every other move slides along a contour;\nthe learned gate crosses them",
                xy=(77.9, 22.2), xytext=(80.9, 24.9), fontsize=6.4, color=TEAL,
                ha="left", va="top",
                arrowprops=dict(arrowstyle="-", lw=0.7, color=TEAL_L,
                                connectionstyle="arc3,rad=-0.25"))
    ax.set_xlim(81.3, 74.4); ax.set_ylim(6.4, 25.6)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0, pad=3)
    ax.set_xlabel("accuracy on photographed species  %        better", fontsize=6.6)
    ax.set_ylabel("accuracy on species with no photograph  %", fontsize=6.6)
    save(fig, name)


# =========================================================== 7 / framing
def fig_framing(name="framing"):
    rows = [("what the encoder was trained on", 0.75, 1.4618, 1.9110, "#BFB2A6"),
            ("evaluation, photographed species", 0.9974, 1.4988, 2.1192, TEAL),
            ("evaluation, species with none", 1.3531, 1.7439, 2.7586, CLAY)]
    fig, ax = plt.subplots(figsize=(COL_W, 1.85))
    ax.set_xlim(0.55, 3.05); ax.set_ylim(-1.05, 2.65)
    bare(ax)
    ax.fill_betweenx([-0.35, 2.45], 0.75, 1.33, color=FLAG, alpha=0.075, lw=0, zorder=0)
    for i, (lab, p10, p50, p90, c) in enumerate(rows):
        y = 2 - i
        verts = np.array([[p10, y - 0.055], [p50, y - 0.14], [p90, y - 0.055],
                          [p90, y + 0.055], [p50, y + 0.14], [p10, y + 0.055]])
        gradient_shape(ax, verts, c, c, alpha=0.30, zorder=2)
        ax.plot([p10, p90], [y, y], color=c, lw=1.1, zorder=3, solid_capstyle="round")
        ax.scatter([p50], [y], s=22, color=c, edgecolors="none", zorder=4)
        ax.text(p10 - 0.035, y, lab, ha="right", va="center", fontsize=6.4, color=INK)
        ax.text(p90 + 0.035, y, f"{p90:.2f}", ha="left", va="center", fontsize=6.2,
                color=MUTED, family=MONO)
    ax.text(1.04, -0.80, "the augmentation range", ha="center", fontsize=6.3, color=FLAG)
    ax.text(2.35, -0.80, "aspect ratio, width / height", ha="center", fontsize=6.3, color=MUTED)
    for xt in (1.0, 1.5, 2.0, 2.5, 3.0):
        ax.text(xt, -0.48, f"{xt:.1f}", ha="center", fontsize=6.0, color=FAINT, family=MONO)
    save(fig, name)


# ========================================================== 8 / encoders
def fig_encoders(name="encoders"):
    rows = [("BioCLIP-2.5-H  +  corrected framing", 31.10, True),
            ("BioCLIP-2.5-H", 21.53, False), ("BioCLIP 2", 14.37, False),
            ("BioCLIP 1", 11.60, False), ("TaxaBind ViT-B/16", 11.48, False),
            ("BioCAP", 6.77, False), ("SigLIP2 SO400M", 5.13, False),
            ("BioCLIP hyperbolic", 4.36, False), ("general CLIP", 0.78, False),
            ("bioclip-inat-only", 0.35, False), ("BioTrove-CLIP", 0.00, False)]
    fig, ax = plt.subplots(figsize=(COL_W, 2.35))
    y = np.arange(len(rows))[::-1]
    bare(ax)
    for (lab, v, hero), yy in zip(rows, y):
        c = TEAL if hero else "#C4B8AC"
        ax.plot([0, v], [yy, yy], color=c, lw=0.8, zorder=2, alpha=0.8)
        ax.scatter([v], [yy], s=44 if hero else 20, color=c, edgecolors="none", zorder=3)
        ax.text(-0.7, yy, lab, ha="right", va="center", fontsize=6.4,
                color=INK if hero else MUTED)
        ax.text(v + 0.7, yy, f"{v:.2f}", ha="left", va="center", fontsize=6.3,
                color=c if hero else MUTED, family=MONO,
                weight="600" if hero else "400")
    ax.set_xlim(-0.5, 36); ax.set_ylim(-1.0, len(rows) - 0.3)
    ax.text(17, -0.95, "accuracy on species with no photograph, held-out proxy  %",
            ha="center", fontsize=6.4, color=MUTED)
    save(fig, name)


# ========================================================== 9 / transfer
def fig_transfer(name="transfer"):
    pts = [("v40", 1.68, 0.03), ("v43", 0.52, 0.19), ("v46", 0.345, 0.067),
           ("v56", 1.424, 0.174), ("v82", 8.154, 0.213), ("v83", 1.803, 0.053)]
    neg = [("v34", 3.51, -0.07), ("v39", 0.52, -0.11)]
    off = {"v43": (7, 4), "v46": (-9, -11), "v56": (7, -10), "v40": (7, 3),
           "v82": (-4, 10), "v83": (7, -10)}
    fig, ax = plt.subplots(figsize=(COL_W, 2.15))
    xs = np.linspace(0, 9, 200)
    ax.fill_between(xs, 0, xs * 0.36, color=TEAL, alpha=0.10, lw=0, zorder=0)
    ax.fill_between(xs, -0.22, 0, color=FLAG, alpha=0.07, lw=0, zorder=0)
    for r in (0.36, 0.12, 0.026):
        ax.plot([0, 9], [0, 9 * r], lw=0.55, color=FAINT, ls=(0, (2, 3)), zorder=1)
    ax.axhline(0, color=INK, lw=0.7, zorder=2)
    for n, x, y in pts:
        ax.scatter(x, y, s=30, color=TEAL, edgecolors="none", zorder=4)
        ax.annotate(n, (x, y), textcoords="offset points", xytext=off.get(n, (7, 4)),
                    fontsize=6.3, color=INK, family=MONO, zorder=5)
    for n, x, y in neg:
        ax.scatter(x, y, s=30, color=FLAG, marker="X", edgecolors="none", zorder=4)
        ax.annotate(n, (x, y), textcoords="offset points", xytext=(8, -2),
                    fontsize=6.3, color=FLAG, family=MONO, va="center", zorder=5)
    ax.scatter([8.9], [0.006], s=30, color=FLAG, marker="X", edgecolors="none", zorder=4)
    ax.annotate("v81, before the leak was found",
                xy=(8.9, 0.006), xytext=(8.6, 0.16), fontsize=6.4, color=FLAG,
                ha="right", va="bottom",
                arrowprops=dict(arrowstyle="-", lw=0.7, color=FLAG))
    ax.text(2.4, 0.545, "gains that transferred", fontsize=6.3, color=TEAL)
    ax.text(0.15, -0.155, "gains that reversed", fontsize=6.3, color=FLAG)
    ax.set_xlim(-0.25, 9.6); ax.set_ylim(-0.22, 0.60)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0, pad=2)
    ax.set_xlabel("gain measured on the held-out proxy  points", fontsize=6.6)
    ax.set_ylabel("gain measured on the real evaluation  points", fontsize=6.6)
    save(fig, name)


# ========================================================= 10 / negatives
def fig_negatives(name="negatives"):
    bars = [("two-stage family narrowing", -20.0),
            ("common-name and generic prompts", -17.73),
            ("SigLIP2 as a general backbone", -16.40),
            ("VLM re-rank, Qwen2.5-VL", -16.0),
            ("raw LLM description text", -15.87)]
    fig, ax = plt.subplots(figsize=(COL_W, 1.95))
    y = np.arange(len(bars))[::-1]
    bare(ax)
    for (lab, v), yy in zip(bars, y):
        ax.plot([v, 0], [yy, yy], color=FLAG, lw=0.8, alpha=0.55, zorder=2)
        ax.scatter([v], [yy], s=26, color=FLAG, edgecolors="none", zorder=3)
        ax.text(0.7, yy, lab, ha="left", va="center", fontsize=6.4, color=MUTED)
        ax.text(v - 0.7, yy, f"{v:.2f}", ha="right", va="center", fontsize=6.3,
                color=FLAG, family=MONO)
    ax.plot([1.9, 0], [-1.25, -1.25], color=GAIN, lw=0.8, alpha=0.55, zorder=2)
    ax.scatter([1.9], [-1.25], s=26, color=GAIN, edgecolors="none", zorder=3)
    ax.text(2.6, -1.25, "soft routing — proxy said +1.9,\nthe real evaluation fell",
            ha="left", va="center", fontsize=6.4, color=GAIN)
    ax.axvline(0, color=INK, lw=0.7, zorder=4)
    ax.set_xlim(-23, 27); ax.set_ylim(-2.0, len(bars) - 0.3)
    ax.text(-11, -1.95, "accuracy points against the adopted configuration",
            ha="center", fontsize=6.3, color=MUTED)
    save(fig, name)


# =========================================================== 11 / the gap
def fig_gap(name="gap"):
    """The leak, drawn on one shared scale.

    The real gains are nearly invisible beside the proxy gains, which is the
    finding. A magnified strip repeats them at 20x, labelled as such, rather
    than silently rescaling the main bars.
    """
    fig, ax = plt.subplots(figsize=(COL_W, 2.05))
    ax.set_xlim(-3.4, 15.4); ax.set_ylim(-1.25, 2.05)
    bare(ax)
    rows = ((1.45, 13.374, 0.006, "as first built", FLAG),
            (0.55, 8.154, 0.213, "leak closed", TEAL))
    for y, proxy, real, lab, c in rows:
        ax.plot([0, proxy], [y, y], color=c, lw=4.5, alpha=0.20,
                solid_capstyle="butt", zorder=2)
        ax.plot([0, real], [y, y], color=c, lw=4.5, solid_capstyle="butt", zorder=3)
        ax.text(proxy + 0.25, y, f"{proxy:+.3f} on the proxy", va="center",
                fontsize=6.5, color=MUTED, family=MONO)
        ax.text(-0.35, y, lab, ha="right", va="center", fontsize=6.8, color=INK)
    ax.plot([0, 0], [0.15, 1.85], color=INK, lw=0.7, zorder=4)

    # magnified repeat of the real gains only
    ax.text(-0.35, -0.42, "the same real gains,\nmagnified 20 times", ha="right",
            va="center", fontsize=6.3, color=MUTED, linespacing=1.5)
    for (y, _proxy, real, _lab, c), yy in zip(rows, (-0.18, -0.66)):
        ax.plot([0, real * 20], [yy, yy], color=c, lw=4.5,
                solid_capstyle="butt", zorder=3)
        ax.text(real * 20 + 0.25, yy, f"{real:+.3f} on the real evaluation",
                va="center", fontsize=6.5, color=c, family=MONO, weight="600")
    ax.text(0, 1.95, "solid = what survived the real evaluation", fontsize=6.4,
            color=MUTED, style="italic", va="bottom")
    save(fig, name)


if __name__ == "__main__":
    print("figures:")
    fig_fan()
    fig_composition()
    fig_pipeline()
    fig_routing()
    fig_climb()
    fig_trade()
    fig_framing()
    fig_encoders()
    fig_transfer()
    fig_negatives()
    fig_gap()
