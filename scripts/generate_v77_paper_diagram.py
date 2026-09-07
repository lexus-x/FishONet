"""Generate publication-quality paper-style architecture diagram for v77.

Saves high-res diagram to reports/fishonet_figs/v77_architecture_diagram.png,
reports/fishonet_figs/v77_architecture_diagram.pdf, and the artifact directory.
"""
from __future__ import annotations

import os
import shutil
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(ROOT, 'reports', 'fishonet_figs')
ART_DIR = '/home/ubuntu/.gemini/antigravity-ide/brain/2e825206-9290-4e80-9127-413b851dbc24'
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(ART_DIR, exist_ok=True)


def draw_box(ax, xy, width, height, title, subtitle, color, edgecolor, text_color='#111827', title_size=10.5, sub_size=8.5, radius=0.012):
    x, y = xy
    box = FancyBboxPatch(
        (x, y), width, height,
        boxstyle=f"round,pad=0.010,rounding_size={radius}",
        facecolor=color, edgecolor=edgecolor, linewidth=1.6,
        zorder=3
    )
    ax.add_patch(box)
    
    if subtitle:
        ax.text(x + width / 2, y + height * 0.64, title, ha='center', va='center', fontsize=title_size, fontweight='bold', color=text_color, zorder=4)
        ax.text(x + width / 2, y + height * 0.28, subtitle, ha='center', va='center', fontsize=sub_size, color='#374151', zorder=4, linespacing=1.2)
    else:
        ax.text(x + width / 2, y + height / 2, title, ha='center', va='center', fontsize=title_size, fontweight='bold', color=text_color, zorder=4)


def draw_arrow(ax, start, end, color='#4B5563', lw=1.6):
    ax.annotate(
        '', xy=end, xytext=start,
        arrowprops=dict(
            arrowstyle='-|>', color=color, lw=lw,
            mutation_scale=13, shrinkA=2, shrinkB=2
        ),
        zorder=2
    )


def main():
    plt.rcParams['font.family'] = 'DejaVu Sans'
    fig, ax = plt.subplots(figsize=(16, 10.5), dpi=300)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.axis('off')

    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#FFFFFF')

    # Main Title Header
    ax.text(0.5, 0.965, "FishONet Architecture: 12-D Learned Cross-Modal Soft Gate & Multi-Modal Unseen Routing",
            ha='center', va='center', fontsize=15, fontweight='bold', color='#0F172A')
    ax.text(0.5, 0.938, "Dual-Axis Decision Space · 12-Feature MLE Logistic Gate · 7-View Overlap Photo Bank · Entropic Sinkhorn Transport (53.38% Real Score)",
            ha='center', va='center', fontsize=10, color='#475569', style='italic')

    # Query Input Box (Top Center)
    draw_box(ax, (0.37, 0.83), 0.26, 0.075, "Query Image $x$", "Aspect-Ratio Aware Multi-View Crops\n(Center, Squash, 5-Strip Overlaps)", '#F8FAFC', '#94A3B8')

    # Feed-forward arrows from input
    draw_arrow(ax, (0.42, 0.83), (0.19, 0.73)) # To Gate Features
    draw_arrow(ax, (0.50, 0.83), (0.50, 0.73)) # To Seen Head
    draw_arrow(ax, (0.58, 0.83), (0.81, 0.73)) # To Unseen Head

    # -------------------------------------------------------------
    # Column 1 (Left): 12-D Learned Novelty Gate
    # -------------------------------------------------------------
    draw_box(ax, (0.04, 0.61), 0.30, 0.12, "12-D Cross-Modal Feature Assembly",
             "• Seen Ensemble: sb_max, sb_margin, sb_lse_gap\n• Encoder Maxima: m_ctftshift, m_ftshift, m_336shift\n• Debate Margins: tm_ctft, tm_full\n• Photo Bank Sharpness: bank_max, bank_margin\n• BioCLIP-2 Protos: b2f_max, b2l_max",
             '#EFF6FF', '#3B82F6', sub_size=7.8)
    
    draw_arrow(ax, (0.19, 0.61), (0.19, 0.52))

    draw_box(ax, (0.04, 0.44), 0.30, 0.075, "12-D Learned Logistic Soft Gate",
             "$P(\\mathrm{seen} \\mid x) = \\sigma(\\mathbf{w}^T \\mathbf{z} + b)$\nStandardScaler $Z$-score + MLE Weights (AUC: 0.9758)",
             '#DBEAFE', '#2563EB')

    draw_arrow(ax, (0.19, 0.44), (0.19, 0.35))

    draw_box(ax, (0.07, 0.27), 0.24, 0.075, "Rank-Order Hard Routing",
             "Rank by $P(\\mathrm{seen} \\mid x)$ with fixed $f=0.60$\nIf $P \\geq t_{0.60} \\rightarrow$ Seen Head (Top 60%)\nIf $P < t_{0.60} \\rightarrow$ Unseen Head (40%)",
             '#EFF6FF', '#1D4ED8', sub_size=7.8)

    # -------------------------------------------------------------
    # Column 2 (Center): Seen Specialist Ensemble Head
    # -------------------------------------------------------------
    draw_box(ax, (0.37, 0.61), 0.26, 0.12, "Seen Specialist Backbone Ensemble",
             "• BioCLIP-2.5 ctftshift (w = 1.0)\n• BioCLIP-2.5 ftshift (w = 2.5)\n• BioCLIP-2.5 fullft336shift (w = 2.5)\n• 5,795 Training Prototypes $P_s$\n• Instance Nearest-Neighbor $c_{\\max}$",
             '#F0FDF4', '#22C55E', sub_size=8.0)

    draw_arrow(ax, (0.50, 0.61), (0.50, 0.52))

    draw_box(ax, (0.37, 0.44), 0.26, 0.075, "Seen Scoring Function",
             "$S_{\\mathrm{seen}}(x, s) = \\sum_m w_m \\cdot Z(\\mathbf{q}_m \\mathbf{P}_s + 2 c_{\\max} + \\lambda \\mathbf{q}_m \\mathbf{T}_s)$\n($\\lambda = 4.0$, Taxonomy Guidance)",
             '#DCFCE7', '#16A34A', sub_size=8.0)

    draw_arrow(ax, (0.50, 0.44), (0.50, 0.35))

    draw_box(ax, (0.39, 0.27), 0.22, 0.075, "Seen Candidate Argmax",
             "$\\hat{y} = \\mathrm{argmax}_{s} S_{\\mathrm{seen}}(x, s)$\nOver 5,795 Seen Candidate Classes",
             '#F0FDF4', '#15803D', sub_size=8.2)

    # -------------------------------------------------------------
    # Column 3 (Right): Multi-Modal Unseen Head
    # -------------------------------------------------------------
    draw_box(ax, (0.66, 0.61), 0.30, 0.12, "Multi-Modal Unseen Candidate Legs",
             "• 7-View Overlap Crop-Max Photo Bank ($w_{\\mathrm{ctft}} = 4.0$)\n• 336px iNaturalist Photo Bank ($w_{336} = 3.0$)\n• BioCLIP-2 Tree-of-Life: Frozen ($w=2.5$) + LoRA ($w=2.0$)\n• 4-Encoder Text Ensemble + TaxaBind ($w=1.0$)",
             '#FFFBEB', '#F59E0B', sub_size=7.8)

    draw_arrow(ax, (0.81, 0.61), (0.81, 0.52))

    draw_box(ax, (0.66, 0.44), 0.30, 0.075, "Multi-Modal Score Fusion",
             "$S_{\\mathrm{unseen}}(x, u) = S_{\\mathrm{text}}(u) + S_{\\mathrm{img}}(u) + S_{\\mathrm{ToL}}(u)$\nOver 11,598 Unseen Candidate Classes",
             '#FEF3C7', '#D97706', sub_size=8.0)

    draw_arrow(ax, (0.81, 0.44), (0.81, 0.35))

    draw_box(ax, (0.68, 0.27), 0.26, 0.075, "Entropic Optimal Transport (Sinkhorn)",
             "$\\mathbf{P}^* = \\mathrm{Sinkhorn}(S_{\\mathrm{unseen}} / \\tau, \\tau=1.8)$\nEnforces Marginal Class Uniformity",
             '#FFFBEB', '#B45309', sub_size=8.0)

    # -------------------------------------------------------------
    # Gating Routing Switch Connections to Final Output
    # -------------------------------------------------------------
    draw_arrow(ax, (0.19, 0.27), (0.35, 0.15), color='#2563EB', lw=2.0)
    draw_arrow(ax, (0.50, 0.27), (0.47, 0.16), color='#16A34A', lw=2.0)
    draw_arrow(ax, (0.81, 0.27), (0.53, 0.16), color='#D97706', lw=2.0)

    # Final Combined Prediction Box
    draw_box(ax, (0.28, 0.04), 0.44, 0.10, "Final Calibrated Open-Set Prediction (17,393 Classes)",
             "$\\hat{y} = \\mathrm{argmax}_{s} S_{\\mathrm{seen}}(x, s)$ if $P(\\mathrm{seen}) \\geq t_{0.60}$, else $\\mathrm{argmax}_{u} \\mathbf{P}^*_{x, u}$\nOverall Test Score: 53.3828% (Seen: 76.64% · Unseen: 23.36% / Lift: +1.767pt)",
             '#F1F5F9', '#334155', title_size=11, sub_size=8.5, radius=0.015)

    out_png = os.path.join(FIG_DIR, 'v77_architecture_diagram.png')
    out_pdf = os.path.join(FIG_DIR, 'v77_architecture_diagram.pdf')
    art_png = os.path.join(ART_DIR, 'v77_architecture_diagram.png')

    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.savefig(out_pdf, bbox_inches='tight')
    shutil.copy(out_png, art_png)
    print(f"Saved architecture diagram to:\n  - {out_png}\n  - {out_pdf}\n  - {art_png}")


if __name__ == '__main__':
    main()
