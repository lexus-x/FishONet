#!/usr/bin/env python3
"""Generate organization-grade FishONet Challenge technical report (single PDF)."""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch, mm
from reportlab.platypus import (
    CondPageBreak,
    Image,
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path("/home/ubuntu/onet")
FIG = ROOT / "reports" / "fishonet_figs"
OUT = ROOT / "reports" / "FishONet_Challenge_Technical_Report.pdf"
FIG.mkdir(parents=True, exist_ok=True)

# Measured Codabench overall scores (HANDOFF.md) — never invent scores
CLIMB = [
    ("γ30", 45.19),
    ("γ10", 45.19),
    ("v22", 45.39),
    ("v23", 45.96),
    ("v25", 46.27),
    ("v27", 46.34),
    ("v28", 47.26),
    ("v31", 47.69),
    ("v33", 47.78),
    ("v34†", 47.71),
    ("v36", 49.04),
    ("v37w3", 50.46),
    ("v39†", 50.35),
    ("v40", 50.49),
    ("v41", 50.57),
    ("v43★", 50.76),
]

# Decomposed real scores where available
DECOMP = [
    # label, seen, unseen, overall
    ("v33", 78.20, 8.51, 47.78),
    ("v34†", 78.63, 7.80, 47.71),
    ("v36", 78.95, 10.42, 49.04),
    ("v37 w1.5", 78.95, 12.89, 50.12),
    ("v37 w2", 78.95, 13.35, 50.32),
    ("v37 w3", 78.95, 13.68, 50.46),  # implied from climb; w3 overall 50.46
    ("v39†", 78.95, 13.44, 50.35),
    ("v40", 78.95, 13.76, 50.49),
    ("v41", 78.95, 13.93, 50.57),
    ("v43★", 78.952, 14.356, 50.756),
]

# Proxy→real transfer factors (measured)
TRANSFER = [
    ("iNat mean (v37)", 0.12),
    ("Maxpool (v40)", 0.02),
    ("B2 mean (v41)", 0.08),
    ("Dual B2 (v43)", 0.36),
    ("TB-img (v39)†", -0.21),  # inverse direction qualitatively
]

NAVY = colors.HexColor("#0B1F33")
TEAL = colors.HexColor("#1A6B7A")
SLATE = colors.HexColor("#334155")
LIGHT = colors.HexColor("#F1F5F9")
ACCENT = colors.HexColor("#0E7490")
WARN = colors.HexColor("#B45309")
GOOD = colors.HexColor("#047857")
MUTED = colors.HexColor("#64748B")


def style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#94A3B8")
    ax.spines["bottom"].set_color("#94A3B8")
    ax.tick_params(colors="#475569", labelsize=8)
    ax.grid(axis="y", color="#E2E8F0", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def make_figures():
    # Fig 1 — climb timeline
    labels, scores = zip(*CLIMB)
    fig, ax = plt.subplots(figsize=(9.2, 3.6), dpi=160)
    x = np.arange(len(scores))
    dead = {i for i, l in enumerate(labels) if "†" in l}
    best = {i for i, l in enumerate(labels) if "★" in l}
    cols = []
    for i in range(len(scores)):
        if i in best:
            cols.append("#0E7490")
        elif i in dead:
            cols.append("#B45309")
        else:
            cols.append("#1E3A5F")
    ax.plot(x, scores, color="#94A3B8", linewidth=1.4, zorder=1)
    ax.scatter(x, scores, c=cols, s=42, zorder=3, edgecolors="white", linewidths=0.6)
    for i, (lab, sc) in enumerate(CLIMB):
        if i in best or lab in ("v36", "v23", "v28", "v37w3"):
            ax.annotate(
                f"{sc:.2f}",
                (i, sc),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=7,
                color="#0F172A",
            )
    ax.axhline(50.56, color="#DC2626", linestyle="--", linewidth=1.0, label="Visible #1 (50.56%)")
    ax.axhline(53.0, color="#047857", linestyle=":", linewidth=1.2, label="Target bar (53%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Overall accuracy (%)", fontsize=9)
    ax.set_title("Codabench climb: single-pipeline submissions", fontsize=11, pad=8, color="#0F172A")
    ax.set_ylim(44.5, 54.2)
    style_ax(ax)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    p1 = FIG / "fig1_climb.png"
    fig.savefig(p1, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # Fig 2 — seen vs unseen decomposition
    labs = [r[0] for r in DECOMP]
    seen = [r[1] for r in DECOMP]
    uns = [r[2] for r in DECOMP]
    fig, ax = plt.subplots(figsize=(9.2, 3.8), dpi=160)
    x = np.arange(len(labs))
    w = 0.38
    ax.bar(x - w / 2, seen, w, label="Seen accuracy", color="#1E3A5F", zorder=2)
    ax.bar(x + w / 2, uns, w, label="Unseen accuracy", color="#0E7490", zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(labs, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Accuracy (%)", fontsize=9)
    ax.set_title("Seen vs unseen accuracy on Codabench (measured)", fontsize=11, pad=8)
    ax.set_ylim(0, 100)
    style_ax(ax)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    p2 = FIG / "fig2_seen_unseen.png"
    fig.savefig(p2, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # Fig 3 — contribution waterfall from early → best
    steps = [
        ("Start γ30", 45.19),
        ("+ combined gate (v23)", 45.96),
        ("+ shift FT / ens (v27)", 46.34),
        ("+ routing f (v31)", 47.69),
        ("+ Sinkhorn (v33)", 47.78),
        ("+ ctft+TaxaBind (v36)", 49.04),
        ("+ iNat mean (v37)", 50.46),
        ("+ maxpool (v40)", 50.49),
        ("+ B2 proto (v41)", 50.57),
        ("+ dual B2 (v43)", 50.756),
    ]
    vals = [s[1] for s in steps]
    deltas = [vals[0]] + [vals[i] - vals[i - 1] for i in range(1, len(vals))]
    fig, ax = plt.subplots(figsize=(9.2, 3.8), dpi=160)
    cum = 0.0
    xs = np.arange(len(steps))
    for i, d in enumerate(deltas):
        bottom = cum if i > 0 else 0
        if i == 0:
            ax.bar(i, d, color="#1E3A5F", zorder=2)
            cum = d
        else:
            col = "#047857" if d >= 0 else "#B45309"
            ax.bar(i, d, bottom=bottom, color=col, zorder=2)
            cum = bottom + d
        ax.text(i, cum + 0.12, f"{vals[i]:.2f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(xs)
    ax.set_xticklabels([s[0] for s in steps], rotation=35, ha="right", fontsize=7)
    ax.set_ylabel("Overall (%)", fontsize=9)
    ax.set_title("Cumulative contribution of validated real levers", fontsize=11, pad=8)
    ax.set_ylim(44, 52.5)
    style_ax(ax)
    fig.tight_layout()
    p3 = FIG / "fig3_waterfall.png"
    fig.savefig(p3, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # Fig 4 — score composition pie / stacked contribution
    # overall = 0.5635*seen + 0.4365*unseen
    seen_c = 0.5635 * 78.952
    uns_c = 0.4365 * 14.356
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5), dpi=160)
    axes[0].pie(
        [seen_c, uns_c],
        labels=[f"Seen contribution\n{seen_c:.2f} pt", f"Unseen contribution\n{uns_c:.2f} pt"],
        colors=["#1E3A5F", "#0E7490"],
        startangle=90,
        wedgeprops=dict(width=0.45, edgecolor="white"),
        textprops=dict(fontsize=8, color="#0F172A"),
    )
    axes[0].set_title("v43 overall = 50.76%\n(population-weighted)", fontsize=10)
    # Gap chart
    need_uns = 19.497
    have = 14.356
    axes[1].barh([0], [have], color="#0E7490", height=0.45, label="Current unseen")
    axes[1].barh([0], [need_uns - have], left=[have], color="#FDE68A", height=0.45, label="Gap to 53% @ flat seen")
    axes[1].axvline(need_uns, color="#047857", linestyle="--", linewidth=1.2)
    axes[1].set_yticks([])
    axes[1].set_xlabel("Unseen accuracy (%)", fontsize=9)
    axes[1].set_xlim(0, 25)
    axes[1].set_title("Unseen headroom to 53% overall\n(+5.14 pt unseen needed)", fontsize=10)
    axes[1].legend(frameon=False, fontsize=7, loc="lower right")
    style_ax(axes[1])
    fig.tight_layout()
    p4 = FIG / "fig4_composition_gap.png"
    fig.savefig(p4, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # Fig 5 — proxy→real transfer
    names = [t[0] for t in TRANSFER]
    facs = [t[1] for t in TRANSFER]
    fig, ax = plt.subplots(figsize=(7.5, 3.4), dpi=160)
    cols = ["#047857" if f > 0 else "#B45309" for f in facs]
    ax.barh(names[::-1], facs[::-1], color=cols[::-1], height=0.55, zorder=2)
    ax.axvline(0, color="#64748B", linewidth=0.8)
    ax.set_xlabel("Overall points per proxy point (measured)", fontsize=9)
    ax.set_title("Holdout→Codabench transfer factors (evidence for EV budgeting)", fontsize=10)
    style_ax(ax)
    fig.tight_layout()
    p5 = FIG / "fig5_transfer.png"
    fig.savefig(p5, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # Fig 6 — architecture schematic (matplotlib boxes)
    fig, ax = plt.subplots(figsize=(9.2, 4.2), dpi=160)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    def box(x, y, w, h, text, fc="#E0F2FE", ec="#0E7490"):
        r = plt.Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, linewidth=1.2, zorder=2)
        ax.add_patch(r)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7.5, color="#0F172A", wrap=True)

    box(0.3, 4.4, 2.2, 1.1, "Eval image\n(35,665)", "#F8FAFC", "#334155")
    box(3.0, 4.4, 2.6, 1.1, "Vision stack\nBioCLIP / ctftshift\n+ TTA / ens", "#DBEAFE", "#1E40AF")
    box(6.2, 4.4, 3.2, 1.1, "Text / proto bank\ntaxctx + TaxaBind\niNat mean/max + B2 dual", "#CCFBF1", "#0F766E")
    box(0.3, 2.5, 4.0, 1.2, "Novelty gate\nz(img_max) + 2·z(text_margin)\nroute frac f = 0.72", "#FEF3C7", "#B45309")
    box(5.0, 2.5, 4.4, 1.2, "Unseen route\nfull 17,393-class argmax\n+ Sinkhorn (τ=2, 50 it)", "#FCE7F3", "#9D174D")
    box(2.5, 0.5, 5.0, 1.2, "prediction.json → submission.zip\nCodabench overall = 0.5635·seen + 0.4365·unseen", "#ECFDF5", "#047857")
    ax.annotate("", xy=(3.0, 4.95), xytext=(2.5, 4.95), arrowprops=dict(arrowstyle="->", color="#64748B"))
    ax.annotate("", xy=(6.2, 4.95), xytext=(5.6, 4.95), arrowprops=dict(arrowstyle="->", color="#64748B"))
    ax.annotate("", xy=(2.3, 3.7), xytext=(1.4, 4.4), arrowprops=dict(arrowstyle="->", color="#64748B"))
    ax.annotate("", xy=(7.2, 3.7), xytext=(7.8, 4.4), arrowprops=dict(arrowstyle="->", color="#64748B"))
    ax.annotate("", xy=(5.0, 3.1), xytext=(4.3, 3.1), arrowprops=dict(arrowstyle="->", color="#64748B"))
    ax.annotate("", xy=(5.0, 1.7), xytext=(5.0, 2.5), arrowprops=dict(arrowstyle="->", color="#64748B"))
    ax.set_title("Inference pipeline (single uniform rule; no folder-oracle)", fontsize=11, color="#0F172A", pad=4)
    p6 = FIG / "fig6_pipeline.png"
    fig.savefig(p6, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return {
        "climb": str(p1),
        "seen_unseen": str(p2),
        "waterfall": str(p3),
        "gap": str(p4),
        "transfer": str(p5),
        "pipeline": str(p6),
    }


def build_styles():
    base = getSampleStyleSheet()
    styles = {
        "cover_title": ParagraphStyle(
            "cover_title",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=26,
            leading=30,
            textColor=NAVY,
            alignment=TA_CENTER,
            spaceAfter=8,
        ),
        "cover_sub": ParagraphStyle(
            "cover_sub",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=12,
            leading=16,
            textColor=TEAL,
            alignment=TA_CENTER,
            spaceAfter=6,
        ),
        "cover_meta": ParagraphStyle(
            "cover_meta",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=9,
            leading=12,
            textColor=MUTED,
            alignment=TA_CENTER,
        ),
        "h1": ParagraphStyle(
            "h1",
            parent=base["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=14,
            leading=18,
            textColor=NAVY,
            spaceBefore=14,
            spaceAfter=8,
            borderPadding=3,
        ),
        "h2": ParagraphStyle(
            "h2",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=11,
            leading=14,
            textColor=TEAL,
            spaceBefore=10,
            spaceAfter=5,
        ),
        "body": ParagraphStyle(
            "body",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=13,
            textColor=SLATE,
            alignment=TA_JUSTIFY,
            spaceAfter=6,
        ),
        "caption": ParagraphStyle(
            "caption",
            parent=base["Normal"],
            fontName="Helvetica-Oblique",
            fontSize=8,
            leading=10,
            textColor=MUTED,
            alignment=TA_CENTER,
            spaceBefore=3,
            spaceAfter=10,
        ),
        "bullet": ParagraphStyle(
            "bullet",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=9,
            leading=12,
            textColor=SLATE,
            leftIndent=8,
        ),
        "table_cell": ParagraphStyle(
            "table_cell",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=7.5,
            leading=9.5,
            textColor=SLATE,
        ),
        "table_hdr": ParagraphStyle(
            "table_hdr",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=7.5,
            leading=9.5,
            textColor=colors.white,
        ),
        "kpi": ParagraphStyle(
            "kpi",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=10,
            leading=13,
            textColor=NAVY,
            alignment=TA_CENTER,
        ),
        "footer": ParagraphStyle(
            "footer",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=8,
            textColor=MUTED,
            alignment=TA_CENTER,
        ),
    }
    return styles


def P(text, style):
    return Paragraph(text, style)


def hdr_footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(TEAL)
    canvas.setLineWidth(0.6)
    canvas.line(18 * mm, A4[1] - 12 * mm, A4[0] - 18 * mm, A4[1] - 12 * mm)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(MUTED)
    canvas.drawString(18 * mm, A4[1] - 10 * mm, "FishONet Challenge — Technical Report")
    canvas.drawRightString(A4[0] - 18 * mm, A4[1] - 10 * mm, "CV4Ecology 2026 · Codabench 16815")
    canvas.line(18 * mm, 12 * mm, A4[0] - 18 * mm, 12 * mm)
    canvas.drawCentredString(A4[0] / 2, 8 * mm, f"Page {doc.page}")
    canvas.restoreState()


def make_table(headers, rows, col_widths, styles):
    data = [[P(h, styles["table_hdr"]) for h in headers]]
    for row in rows:
        data.append([P(str(c), styles["table_cell"]) for c in row])
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("BACKGROUND", (0, 1), (-1, -1), colors.white),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CBD5E1")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return t


def kpi_row(styles):
    data = [
        [
            P("<b>Best overall</b><br/>50.756%", styles["kpi"]),
            P("<b>Seen / Unseen</b><br/>78.952% / 14.356%", styles["kpi"]),
            P("<b>vs visible #1</b><br/>+0.196 pt", styles["kpi"]),
            P("<b>Gap to 53%</b><br/>2.244 pt", styles["kpi"]),
        ]
    ]
    t = Table(data, colWidths=[42 * mm, 48 * mm, 42 * mm, 42 * mm])
    t.setStyle(
        TableStyle(
            [
                ("BOX", (0, 0), (-1, -1), 0.8, TEAL),
                ("INNERGRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#99F6E4")),
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F0FDFA")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    return t


def build_pdf(figs):
    styles = build_styles()
    doc = SimpleDocTemplate(
        str(OUT),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=16 * mm,
        title="FishONet Challenge — Technical Report",
        author="We / Our team",
        subject="CV4Ecology 2026 fish open-set recognition",
    )
    story = []

    # Cover
    story.append(Spacer(1, 28 * mm))
    story.append(P("FishONet Challenge", styles["cover_title"]))
    story.append(
        P(
            "Open-Set Fish Species Recognition — Technical Results Report",
            styles["cover_sub"],
        )
    )
    story.append(
        P(
            "CV4Ecology Workshop Competition · Codabench Competition 16815<br/>"
            "Single-pipeline compliant system · Measured Codabench evidence only",
            styles["cover_meta"],
        )
    )
    story.append(Spacer(1, 8 * mm))
    story.append(kpi_row(styles))
    story.append(Spacer(1, 8 * mm))
    story.append(
        P(
            f"Document date: {date.today().isoformat()} · Classification: Internal technical report<br/>"
            "Narrative voice: we / our · No individual author attribution",
            styles["cover_meta"],
        )
    )
    story.append(Spacer(1, 10 * mm))
    story.append(
        P(
            "<b>Abstract.</b> We present our end-to-end approach and measured results for the "
            "FishONet open-set recognition challenge. The evaluation set comprises 35,665 images "
            "(20,097 seen / 15,568 unseen) over 17,393 species classes, scored with population-weighted "
            "overall accuracy (0.5635·seen + 0.4365·unseen). Our best confirmed Codabench submission "
            "(v43 dual BioCLIP-2) reaches <b>50.7556%</b> overall (seen 78.952%, unseen 14.356%), "
            "exceeding the visible leaderboard #1 overall by +0.196 pt while remaining 2.244 pt short "
            "of a 53% target bar. Gains since our early ~45% baseline are driven by a combined "
            "image–text novelty gate, routing fraction tuning, Sinkhorn assignment on the unseen route, "
            "shift-robust BioCLIP fine-tunes, TaxaBind text fusion, and external iNaturalist prototypes "
            "(mean → maxpool → BioCLIP-2 → dual frozen+LoRA). We document submission changelogs with "
            "rationales, ablations that failed to transfer, and the remaining unseen headroom.",
            styles["body"],
        )
    )
    story.append(PageBreak())

    # TOC-ish
    story.append(P("1. Problem formulation and evaluation protocol", styles["h1"]))
    story.append(
        P(
            "The Fish Species Recognition Challenge requires predicting a species name for every "
            "evaluation image among 17,393 candidates. Seen classes have training images; unseen "
            "classes are text-only by task definition (no training photographs). Organizers prohibit "
            "folder-oracle use of <font face='Courier'>splits/test.pkl</font> / "
            "<font face='Courier'>splits/unseen.pkl</font> to decide membership or shrink the "
            "candidate set. Our system therefore applies one uniform rule to all 35,665 images and "
            "argmaxes over the full class space.",
            styles["body"],
        )
    )
    story.append(
        P(
            "Official scoring emphasizes overall accuracy. Codabench also reports seen and unseen "
            "accuracies. With population weights w_seen=0.5635 and w_unseen=0.4365, overall is dominated "
            "by seen accuracy, but competitive movement after ~48% is almost entirely in the unseen "
            "term b. Submission quota is 3/day and 30 total — we treat every Codabench slot as "
            "expensive evidence.",
            styles["body"],
        )
    )

    story.append(
        make_table(
            ["Quantity", "Value", "Notes"],
            [
                ["Eval images", "35,665", "20,097 seen + 15,568 unseen"],
                ["Seen classes", "5,795", "With training images"],
                ["Unseen classes", "11,598", "Text-only; no train photos"],
                ["Total classes", "17,393", "Full argmax space"],
                ["Score", "Overall accuracy", "0.5635·seen + 0.4365·unseen"],
                ["Quota", "3/day · 30 total", "Budget via holdout EV + transfer factors"],
                ["Timeline", "1 Jun – 1 Sep 2026", "Results 9 Sep 2026"],
            ],
            [38 * mm, 42 * mm, 94 * mm],
            styles,
        )
    )
    story.append(P("Table 1. Challenge quantities used throughout this report.", styles["caption"]))

    story.append(P("2. System overview", styles["h1"]))
    story.append(
        P(
            "Our production stack is a BioCLIP-family vision–language system with shift-robust "
            "fine-tuning, concentrated ensembling, taxonomic text prompts (taxctx), database "
            "normalization (dbnorm), TaxaBind text features, and iNaturalist-derived class prototypes. "
            "A novelty gate routes a fraction f=0.72 of images toward a closed-set-like path and "
            "the remainder toward an unseen-oriented path with Sinkhorn-balanced assignment. "
            "Compliance is mandatory: one pipeline, full 17,393-way argmax, no fish-specific "
            "third-party recognizers, disclosed external text/images.",
            styles["body"],
        )
    )
    story.append(Image(figs["pipeline"], width=170 * mm, height=78 * mm))
    story.append(P("Figure 1. Inference architecture used for all v33+ submissions.", styles["caption"]))

    story.append(P("2.1 What consistently worked", styles["h2"]))
    bullets = [
        "Combined gate z(img_seenmax) + 2·z(text_margin) — gate became net-positive at v23.",
        "Routing fraction f=0.72 (not the 0.5635 population prior) — v28→v31.",
        "Sinkhorn on the unseen route (tau=2.0, 50 iterations) — +0.09 pt real at v33; coverage 38.6%→57.5%.",
        "ctftshift + TaxaBind text — largest single jump (v36: +1.26 overall / +1.91 unseen vs v33).",
        "iNat mean prototypes (v37), photo maxpool (v40), BioCLIP-2 mean proto (v41), dual frozen+LoRA-v2 BioCLIP-2 (v43).",
        "dbnorm (dim=0) is load-bearing (−1.60 unseen if dropped).",
    ]
    story.append(
        ListFlowable(
            [ListItem(P(b, styles["bullet"]), leftIndent=10, bulletColor=TEAL) for b in bullets],
            bulletType="bullet",
            start="•",
        )
    )

    story.append(P("3. Codabench climb and evidence", styles["h1"]))
    story.append(
        P(
            "Figure 2 plots every major measured overall score from our early gamma baselines "
            "(~45.2%) to the current best (v43, 50.76%). Orange markers denote real regressions "
            "relative to the contemporaneous best (v34 recover-on-eject; v39 TaxaBind image protos). "
            "The red dashed line is visible leaderboard #1 at 50.56%; the green dotted line is our "
            "internal 53% target.",
            styles["body"],
        )
    )
    story.append(Image(figs["climb"], width=170 * mm, height=66 * mm))
    story.append(P("Figure 2. Measured Codabench overall accuracy across submissions.", styles["caption"]))

    story.append(Image(figs["waterfall"], width=170 * mm, height=70 * mm))
    story.append(
        P(
            "Figure 3. Cumulative contribution of levers that transferred to Codabench (dead levers omitted).",
            styles["caption"],
        )
    )

    story.append(Image(figs["seen_unseen"], width=170 * mm, height=70 * mm))
    story.append(
        P(
            "Figure 4. Seen accuracy is essentially flat (~78.95%) after v36; competitive gains are unseen-side.",
            styles["caption"],
        )
    )

    story.append(PageBreak())
    story.append(P("4. Submission changelog (files, results, and why)", styles["h1"]))
    story.append(
        P(
            "Table 2 lists the principal submission artifacts under "
            "<font face='Courier'>submissions/</font>, the measured Codabench outcome when submitted, "
            "and the scientific rationale. Scores are stated only when measured on Codabench; "
            "brackets built but held or projected are marked accordingly.",
            styles["body"],
        )
    )

    changelog = [
        [
            "v33 sink72",
            "submission_v33_sink72.zip",
            "47.78",
            "78.20 / 8.51",
            "Add Sinkhorn balanced assignment on unseen route; lift class coverage.",
        ],
        [
            "v34 recover",
            "(recover-on-eject)",
            "47.71†",
            "78.63 / 7.80",
            "Holdout +3.51 lied; recovering ejected seen hurt unseen (−0.71). Dead.",
        ],
        [
            "v36 ctft+TB",
            "submission_v36_ctftshift_tb_sink72.zip",
            "49.04",
            "78.95 / 10.42",
            "Shift-FT BioCLIP + TaxaBind text; largest single real jump (+1.26).",
        ],
        [
            "v37 w1.5",
            "submission_v37_inat_proto_w1.5_sink72.zip",
            "50.12",
            "78.95 / 12.89",
            "iNat mean image prototypes on unseen route; first post-v36 real lift.",
        ],
        [
            "v37 w2",
            "submission_v37_inat_proto_w2_sink72.zip",
            "50.32",
            "78.95 / 13.35",
            "Weight sweep; diminishing but positive (+0.20 overall vs w1.5).",
        ],
        [
            "v37 w3",
            "submission_v37_inat_proto_w3_sink72.zip",
            "50.46",
            "~13.7 unseen",
            "Higher IMG_W; validated mean-proto anchor before maxpool.",
        ],
        [
            "v39 TB-img",
            "submission_v39_inat_tbimg_w3_0.5_sink72.zip",
            "50.35†",
            "78.95 / 13.44",
            "Proxy +0.52 projected gain; real moved opposite (−0.11). Dead.",
        ],
        [
            "v40 maxpool",
            "submission_v40_inat_maxpool_w4_sink72.zip",
            "50.49",
            "78.95 / 13.76",
            "Top-k photo maxpool bank; tiny real +0.03; transfer ~0.02.",
        ],
        [
            "v41 B2 proto",
            "submission_v41_bioclip2proto_w4_b22.5_sink72.zip",
            "50.57",
            "78.95 / 13.93",
            "BioCLIP-2 iNat mean proto @ B2_W=2.5; +0.08; transfer ~0.08.",
        ],
        [
            "v42 LoRA (opt.)",
            "submission_v42_b2lora_v2_w4_b21.5_sink72.zip",
            "proj ~+0.03",
            "—",
            "Optional LoRA-v2 alone; not the 53% path; superseded by dual.",
        ],
        [
            "v43 dual ★",
            "submission_v43_b2dual_w4_wf0.5_wl1_sink72.zip",
            "50.756",
            "78.952 / 14.356",
            "Frozen B2 wf=0.5 + LoRA-v2 wl=1.0; best real; transfer ~0.36.",
        ],
        [
            "v43 w3 (hold)",
            "submission_v43_b2dual_w3_wf0.5_wl1_sink72.zip",
            "held",
            "—",
            "Proxy weaker than w4 (~44.95 vs ~45.56); do not burn a slot.",
        ],
    ]
    story.append(
        make_table(
            ["ID", "Artifact", "Overall %", "Seen/Unseen", "Why (change & outcome)"],
            changelog,
            [22 * mm, 48 * mm, 20 * mm, 28 * mm, 56 * mm],
            styles,
        )
    )
    story.append(
        P(
            "Table 2. Submission changelog. † = real regression vs contemporaneous best. ★ = current best.",
            styles["caption"],
        )
    )

    story.append(P("4.1 Detailed rationale by era", styles["h2"]))
    story.append(
        P(
            "<b>Gate & routing (γ→v31).</b> Early novelty gates were effectively worthless (γ30≈γ10≈45.19%). "
            "v23’s combined image+text gate made novelty detection net-positive. Raising the seen routing "
            "fraction from the population prior to 0.65 then 0.72 (v28/v31) recognized that with weak "
            "unseen accuracy, over-ejecting valuable seen images is a bad trade.",
            styles["body"],
        )
    )
    story.append(
        P(
            "<b>Assignment & compliance (v33–v34).</b> Sinkhorn improved unseen coverage with a small but "
            "real +0.09 pt. Soft recover-on-eject (v34) is a cautionary tale: holdout gains of +3.51 pt "
            "reversed on Codabench (−0.07 overall) because seen recovery came at a larger unseen cost. "
            "We thereafter insist on measured transfer factors before spending slots.",
            styles["body"],
        )
    )
    story.append(
        P(
            "<b>Representation stack (v36).</b> ctftshift (shift-augmented fine-tune) plus TaxaBind text "
            "delivered the largest single leap (+1.26 overall, +1.91 unseen). Seen rose to ~78.95% and "
            "has remained essentially flat through later prototypes — evidence that subsequent work is "
            "almost purely an unseen-b campaign.",
            styles["body"],
        )
    )
    story.append(
        P(
            "<b>External prototypes (v37–v41).</b> Dense iNaturalist mean prototypes transferred "
            "(~0.12 overall-pt per proxy-pt). Maxpool added only +0.03 real despite large holdout "
            "deltas (~0.02 transfer). BioCLIP-2 mean prototypes stacked orthogonally (+0.08 real; "
            "~0.08 transfer). TaxaBind image prototypes (v39) are the inverse case: positive proxy, "
            "negative real — we treat that proxy family as unreliable for EV.",
            styles["body"],
        )
    )
    story.append(
        P(
            "<b>Dual BioCLIP-2 (v43).</b> Combining a frozen BioCLIP-2 prototype leg (weight 0.5) with "
            "a LoRA-v2 taxon-text leg (weight 1.0) at maxpool weight 4 produced our best measured "
            "result: 50.7556% overall, +0.186 vs v41, with unusually healthy dual-leg transfer "
            "(~0.36 overall-pt/proxy-pt). A w3 dual bracket underperformed w4 on proxy and is held.",
            styles["body"],
        )
    )

    story.append(PageBreak())
    story.append(P("5. Proxy discipline and failed levers", styles["h1"]))
    story.append(
        P(
            "Because Codabench quota is scarce, we maintain holdout proxies and calibrate them with "
            "real anchors. Figure 5 summarizes measured transfer factors. Using the wrong factor "
            "(e.g., applying v37’s 0.12 to maxpool or TaxaBind-image proxies) systematically "
            "overstates expected value and wastes slots.",
            styles["body"],
        )
    )
    story.append(Image(figs["transfer"], width=150 * mm, height=68 * mm))
    story.append(P("Figure 5. Measured holdout→Codabench transfer by lever family.", styles["caption"]))

    story.append(
        make_table(
            ["Lever", "Holdout signal", "Real outcome", "Status"],
            [
                ["Recover-on-eject (v34)", "+3.51 overall", "−0.07 overall", "DEAD"],
                ["TaxaBind iNat img (v39)", "+0.52 stack proxy", "−0.11 overall", "DEAD"],
                ["SigLIP2 / DINO proto stack", "< +0.3 / ~0", "Not submitted", "DEAD proxy"],
                ["Coverage gap-fill (v38)", "proxy-negative", "Not submitted", "DEAD"],
                ["VLM top-K rerank", "oracle misleading", "−16 pt on n=300", "DEAD"],
                ["Trait / Wikipedia text", "≪ taxon prompts", "Not load-bearing", "DEAD"],
                ["B2 maxpool / denser445", "≤0 / worse denser", "NO_ZIP", "DEAD"],
                ["B2 LoRA v4/v5 / hard-neg / CE", "db below v1", "NO_ZIP", "DEAD"],
                ["BioTrove-CLIP proto", "−0.86 vs v41+B2", "NO_ZIP", "DEAD"],
                ["Gated BioCLIP-2.5 B/L, ToL", "401 without token", "Blocked", "BLOCKED"],
            ],
            [42 * mm, 38 * mm, 38 * mm, 56 * mm],
            styles,
        )
    )
    story.append(P("Table 3. Negative or blocked results (do not retry without new evidence).", styles["caption"]))

    story.append(P("6. Scoring decomposition and remaining headroom", styles["h1"]))
    story.append(
        P(
            "For v43, population-weighted contributions are approximately 44.49 pt from seen and "
            "6.27 pt from unseen, summing to 50.76%. At flat seen (78.952%), reaching 53% overall "
            "requires unseen accuracy of 19.497% — a +5.14 pt unseen lift. Perfect-gate oracle "
            "estimates with the current stack remain near ~53.6%, implying gate efficiency and "
            "unseen representation—not closed-set accuracy—are the binding constraints.",
            styles["body"],
        )
    )
    story.append(Image(figs["gap"], width=170 * mm, height=65 * mm))
    story.append(P("Figure 6. Score composition and unseen gap to the 53% target.", styles["caption"]))

    story.append(
        make_table(
            ["Benchmark", "Overall %", "Seen %", "Unseen %", "Δ overall vs us"],
            [
                ["Our best (v43)", "50.756", "78.952", "14.356", "—"],
                ["Visible #1", "50.56", "78.05", "15.08", "−0.196 (we ahead overall)"],
                ["Target bar", "53.00", "78.952 (flat)", "19.497 (needed)", "+2.244 needed"],
                ["Early baseline (γ30)", "45.19", "—", "—", "−5.566 from start"],
            ],
            [40 * mm, 28 * mm, 32 * mm, 40 * mm, 34 * mm],
            styles,
        )
    )
    story.append(P("Table 4. Competitive position (measured Codabench / defined targets).", styles["caption"]))

    story.append(P("7. Compliance and disclosure", styles["h1"]))
    story.append(
        P(
            "We operate under the official Codabench rules. Folder-oracle routing is never used. "
            "Fish-specific third-party recognizers (e.g., CaesiumSG/sg-fish-bioclip-v2, "
            "ReefNet/finetuned-bioclip) are excluded. External morphological/taxonomy text and "
            "iNaturalist photographs used as prototypes are disclosed for the transparency statement. "
            "When model eligibility is unclear (gated general-biology encoders), we stop and ask "
            "organizers (cv4e.workshop@gmail.com) before use. If transduction is questioned, the "
            "strict single-pipeline builder "
            "<font face='Courier'>builders/build_v31_strict_singlepipeline.py</font> (47.69%) is "
            "the compliance reference; production best remains the dual BioCLIP-2 builder.",
            styles["body"],
        )
    )

    story.append(
        make_table(
            ["Rule", "Our practice"],
            [
                ["No folder-oracle", "One gate from image/text signals; full 17,393 argmax"],
                ["No fish-only 3rd-party models", "BioCLIP / general VLM family only"],
                ["No test-label fishing", "No manual labelling of eval images"],
                ["Disclose external data", "iNat photos + taxonomy text logged"],
                ["Quota discipline", "Submit only +EV after calibrated transfer"],
            ],
            [55 * mm, 119 * mm],
            styles,
        )
    )
    story.append(P("Table 5. Compliance checklist.", styles["caption"]))

    story.append(P("8. Rebuild instructions", styles["h1"]))
    story.append(
        P(
            "Environment: <font face='Courier'>conda activate onet</font>. Best submission rebuild:",
            styles["body"],
        )
    )
    story.append(
        P(
            "<font face='Courier' size='8'>python builders/build_v43_bioclip2_dual.py</font><br/>"
            "→ <font face='Courier' size='8'>submissions/submission_v43_b2dual_w4_wf0.5_wl1_sink72.zip</font>",
            styles["body"],
        )
    )
    story.append(
        P(
            "Prior anchors: "
            "<font face='Courier' size='8'>build_v41_bioclip2_proto.py</font> (50.57%), "
            "<font face='Courier' size='8'>build_v40_inat_maxpool.py</font> (50.49%), "
            "<font face='Courier' size='8'>build_v36_ctftshift_taxabind.py</font> (49.04%), "
            "<font face='Courier' size='8'>build_v33_sinkhorn.py</font> (47.78%).",
            styles["body"],
        )
    )

    story.append(P("9. Conclusions", styles["h1"]))
    story.append(
        P(
            "We advanced from ~45% to <b>50.76%</b> overall under a strict single-pipeline regime, "
            "with all late-stage progress coming from unseen-route representation (prototypes and "
            "dual BioCLIP-2) rather than seen closed-set accuracy. Visible #1 is behind us on overall "
            "but ahead on unseen (~15.08% vs 14.36%); closing the internal 53% bar still requires "
            "approximately +5.14 pt unseen at flat seen. Local stack levers are largely exhausted; "
            "remaining plausible upside is stronger eligible general-biology encoders (pending "
            "authentication and organizer confirmation), not further low-signal weight brackets.",
            styles["body"],
        )
    )
    story.append(Spacer(1, 6 * mm))
    story.append(
        P(
            "— End of report —<br/>Source of truth for live scores: repository HANDOFF.md · "
            "Rules: COMPETITION_RULES.md / Codabench 16815",
            styles["cover_meta"],
        )
    )

    doc.build(story, onFirstPage=hdr_footer, onLaterPages=hdr_footer)
    return OUT


def main():
    print("Generating figures…")
    figs = make_figures()
    print("Building PDF…")
    path = build_pdf(figs)
    print(f"Wrote {path} ({path.stat().st_size / 1024:.1f} KiB)")


if __name__ == "__main__":
    main()
