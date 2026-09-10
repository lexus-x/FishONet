"""Generate publication-quality paper-style architecture diagram for v61.

Saves high-res diagram to reports/fishonet_figs/v61_architecture_diagram.png
and artifact directory.
"""
from __future__ import annotations

import os
import shutil
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ArrowStyle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(ROOT, 'reports', 'fishonet_figs')
ART_DIR = '/home/ubuntu/.gemini/antigravity-ide/brain/ec8a5980-c985-4e97-8745-3a4db6b3ef9c'
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(ART_DIR, exist_ok=True)


def draw_box(ax, xy, width, height, title, subtitle, color, edgecolor, text_color='#111827', title_size=11, sub_size=9, radius=0.015):
    x, y = xy
    box = FancyBboxPatch(
        (x, y), width, height,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        facecolor=color, edgecolor=edgecolor, linewidth=1.8,
        zorder=3
    )
    ax.add_patch(box)
    
    if subtitle:
        ax.text(x + width / 2, y + height * 0.65, title, ha='center', va='center', fontsize=title_size, fontweight='bold', color=text_color, zorder=4)
        ax.text(x + width / 2, y + height * 0.30, subtitle, ha='center', va='center', fontsize=sub_size, color='#374151', zorder=4, linespacing=1.2)
    else:
        ax.text(x + width / 2, y + height / 2, title, ha='center', va='center', fontsize=title_size, fontweight='bold', color=text_color, zorder=4)


def draw_arrow(ax, start, end, color='#4B5563', lw=1.8, style='->'):
    ax.annotate(
        '', xy=end, xytext=start,
        arrowprops=dict(
            arrowstyle='-|>', color=color, lw=lw,
            mutation_scale=14, shrinkA=3, shrinkB=3
        ),
        zorder=2
    )


def main():
    plt.rcParams['font.family'] = 'DejaVu Sans'
    fig, ax = plt.subplots(figsize=(16, 10), dpi=300)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.axis('off')

    # Background canvas
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#FFFFFF')

    # Main Paper Title Header
    ax.text(0.5, 0.965, "v61 Architecture: Soft Marginal Probabilistic Inference & Inductive Polynomial Calibration",
            ha='center', va='center', fontsize=15, fontweight='bold', color='#111827')
    ax.text(0.5, 0.935, "Strictly Inductive (Zero DBNorm) · Calibrated Novelty Logistic Gate · Query-Centered Degree-3 Polynomial S-Curves · 17,393 Class Space",
            ha='center', va='center', fontsize=10.5, color='#4B5563', style='italic')

    # Step 1: Input Image Box
    draw_box(ax, (0.40, 0.83), 0.20, 0.07, "Evaluation Image Query $x$", "Aspect Ratio + 7-View Crops", '#F3F4F6', '#9CA3AF', radius=0.01)

    # Connections from Input to 3 Branches
    draw_arrow(ax, (0.42, 0.83), (0.17, 0.74)) # To Gate
    draw_arrow(ax, (0.50, 0.83), (0.50, 0.74)) # To Seen Head
    draw_arrow(ax, (0.58, 0.83), (0.83, 0.74)) # To Unseen Head

    # Branch 1: Novelty Gating (Left)
    draw_box(ax, (0.05, 0.65), 0.24, 0.09, "1. Novelty Feature Extraction", "10 Discrepancy & Energy Signals\n[Seen Margin, Energy, Taxon Diff]", '#EFF6FF', '#3B82F6')
    draw_arrow(ax, (0.17, 0.65), (0.17, 0.57))
    draw_box(ax, (0.05, 0.49), 0.24, 0.08, "Calibrated Logistic Gate", "$P(\\mathrm{seen} \\mid x) = \\sigma(\\mathbf{w}^T \\mathbf{z} + b)$\nStandardScaler + Cost-Optimal Calibration", '#DBEAFE', '#2563EB')
    draw_arrow(ax, (0.17, 0.49), (0.17, 0.41))
    draw_box(ax, (0.07, 0.35), 0.20, 0.06, "Novelty Likelihood", "$\\log P(\\mathrm{seen} \\mid x)$\n$\\log(1 - P(\\mathrm{seen} \\mid x))$", '#EFF6FF', '#3B82F6', title_size=10, sub_size=8.5)

    # Branch 2: Seen Specialist (Center)
    draw_box(ax, (0.37, 0.65), 0.26, 0.09, "2. Seen Specialist Head", "3-Encoder Prototype Ensemble\n$(\\mathrm{ctftshift}, \\mathrm{ftshift}, 336\\mathrm{shift})$", '#F0FDF4', '#22C55E')
    draw_arrow(ax, (0.50, 0.65), (0.50, 0.57))
    draw_box(ax, (0.37, 0.49), 0.26, 0.08, "Instance Top-1 + Taxon Anchor", "$S_{\\mathrm{seen}}(x, s) = \\mathbf{q}\\mathbf{p}_s + 2 c_{\\max} + \\lambda \\mathbf{q}\\mathbf{t}_s$\nOver 5,795 Seen Candidate Classes", '#DCFCE7', '#16A34A')
    draw_arrow(ax, (0.50, 0.49), (0.50, 0.41))
    draw_box(ax, (0.39, 0.35), 0.22, 0.06, "Calibrated Seen Logits", "$\\log \\mathrm{Softmax}(S_{\\mathrm{seen}}(x) / \\tau_s)$\n$\\tau_s = 1.80$ (NLL Calibrated)", '#F0FDF4', '#22C55E', title_size=10, sub_size=8.5)

    # Branch 3: Unseen Retrieval (Right)
    draw_box(ax, (0.69, 0.65), 0.27, 0.09, "3. Multi-Modal Unseen Legs", "7-View Overlap CTFT + 336 Bank\nBioCLIP-2 ToL Protos + TaxaBind", '#FEF3C7', '#F59E0B')
    draw_arrow(ax, (0.83, 0.65), (0.83, 0.57))
    draw_box(ax, (0.69, 0.49), 0.27, 0.08, "Query-Centered Degree-3 S-Curves", "$g_m(\\tilde{s}) = c_3 \\tilde{s}^3 + c_1 \\tilde{s} + c_0$\n$\\tilde{s} = (s - \\mu_i) / \\sigma_i$ (No DBNorm)", '#FEF9C3', '#D97706')
    draw_arrow(ax, (0.83, 0.49), (0.83, 0.41))
    draw_box(ax, (0.71, 0.35), 0.23, 0.06, "Calibrated Unseen Logits", "$\\log \\mathrm{Softmax}(S_{\\mathrm{unseen}}(x) / \\tau_u)$\n$\\tau_u = 1.65$ (Dynamic Attention)", '#FEF3C7', '#F59E0B', title_size=10, sub_size=8.5)

    # Convergence into Continuous Soft Marginal Fusion
    draw_arrow(ax, (0.17, 0.35), (0.35, 0.24))
    draw_arrow(ax, (0.50, 0.35), (0.50, 0.24))
    draw_arrow(ax, (0.83, 0.35), (0.65, 0.24))

    # Fusion Box (Wide Center)
    draw_box(
        ax, (0.18, 0.13), 0.64, 0.11,
        "Continuous Soft Marginal Probabilistic Fusion",
        "Seen:   $\\log P(c \\mid x) = \\log P(\\mathrm{seen} \\mid x) + \\log \\mathrm{Softmax}(S_{\\mathrm{seen}}(x, c) / \\tau_s)$\nUnseen: $\\log P(c \\mid x) = \\log(1 - P(\\mathrm{seen} \\mid x)) + \\log \\mathrm{Softmax}(S_{\\mathrm{unseen}}(x, c) / \\tau_u)$",
        '#FDF4FF', '#A855F7', radius=0.015, title_size=12, sub_size=9.5
    )

    draw_arrow(ax, (0.50, 0.14), (0.50, 0.08))

    # Output Box
    draw_box(
        ax, (0.32, 0.02), 0.36, 0.06,
        "Unified Argmax Over Full 17,393 Class Space",
        "$\\hat{c}^* = \\arg\\max_{c \\in \\mathcal{S} \\cup \\mathcal{U}} \\log P(c \\mid x)$ (Self-Correcting & Zero Leakage)",
        '#1E293B', '#0F172A', text_color='#FFFFFF', radius=0.01, title_size=11, sub_size=9
    )

    fig_png = os.path.join(FIG_DIR, 'v61_architecture_diagram.png')
    art_png = os.path.join(ART_DIR, 'v61_architecture_diagram.png')
    
    plt.tight_layout()
    plt.savefig(fig_png, dpi=300, bbox_inches='tight')
    plt.savefig(art_png, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[SUCCESS] Publication diagram generated:")
    print(f"  ├─ {fig_png}")
    print(f"  └─ {art_png}")


if __name__ == '__main__':
    main()
