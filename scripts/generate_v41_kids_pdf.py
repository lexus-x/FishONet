import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY

# Output paths
OUT_DIR = "/home/ubuntu/onet"
SCRATCH_DIR = "/tmp/v41_pdf_scratch"
os.makedirs(SCRATCH_DIR, exist_ok=True)

PDF_PATH = os.path.join(OUT_DIR, "v41_visual_guide_for_kids.pdf")
ARTIFACT_PDF_PATH = "/home/ubuntu/.gemini/antigravity-ide/brain/79d1ceb2-a89b-4b02-b751-a7000ef33648/v41_visual_guide_for_kids.pdf"

# Generated AI Infographic Paths
AI_IMG_1 = os.path.join(SCRATCH_DIR, "gen_img1.png")  # Aspect Shift Crop
AI_IMG_2 = os.path.join(SCRATCH_DIR, "gen_img2.png")  # Smart Bouncer Gate
AI_IMG_3 = os.path.join(SCRATCH_DIR, "gen_img3.png")  # Photo-Bank Maxpool
AI_IMG_4 = os.path.join(SCRATCH_DIR, "gen_img4.png")  # Sinkhorn Traffic Cop

# ---------------------------------------------------------
# Step 1: Generate Visual Impact Charts & Diagram Helpers
# ---------------------------------------------------------

def generate_impact_chart():
    fig, ax = plt.subplots(figsize=(10, 4.0), dpi=300)
    stages = [
        "Base CLIP\n(Zero-Shot)", "Aspect-Shift\nAdaptation", "Soft Novelty\nGate", 
        "72% Safe\nRouting", "Photo-Bank\nMaxpool", "Sinkhorn\nTransport", 
        "BioCLIP-2\nPrototype", "v41 Final\nSystem"
    ]
    values = [45.19, 46.34, 46.91, 47.69, 49.20, 50.49, 50.57, 50.57]
    deltas = [0, "+1.15", "+0.57", "+0.78", "+1.51", "+1.29", "+0.08", "TOTAL"]
    bar_colors = ['#6c757d', '#4a90e2', '#50e3c2', '#f5a623', '#bd10e0', '#7ed321', '#9013fe', '#1b2a4a']
    
    x = np.arange(len(stages))
    bars = ax.bar(x, values, color=bar_colors, width=0.55, edgecolor='black', linewidth=1)
    
    ax.set_ylim(40, 53)
    ax.set_ylabel("Accuracy (%)", fontsize=11, fontweight='bold', color='#1b2a4a')
    ax.set_title("v41 Accuracy Progression (Technique-by-Technique Impact)", fontsize=12, fontweight='bold', pad=12, color='#1b2a4a')
    ax.set_xticks(x)
    ax.set_xticklabels(stages, fontsize=8.5, fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
        
    for idx, (bar, val, d) in enumerate(zip(bars, values, deltas)):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2., height + 0.3,
            f"{val:.2f}%\n({d})" if idx > 0 and idx < 7 else f"{val:.2f}%",
            ha='center', va='bottom', fontsize=8, fontweight='bold',
            color='#1b2a4a' if idx < 7 else '#d0021b'
        )
        
    plt.tight_layout()
    chart_path = os.path.join(SCRATCH_DIR, "impact_chart.png")
    plt.savefig(chart_path, bbox_inches='tight')
    plt.close()
    return chart_path


def generate_accurate_system_architecture():
    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=300)
    ax.axis('off')
    ax.text(5, 5.0, "v41 Master System Architecture Flowchart", ha='center', va='center', fontsize=12, fontweight='bold', color='#1b2a4a')
    
    b_input = patches.FancyBboxPatch((0.3, 3.8), 1.8, 0.85, boxstyle="round,pad=0.08", facecolor="#e2e8f0", edgecolor="#475569", lw=1.5)
    ax.add_patch(b_input)
    ax.text(1.2, 4.225, "Input Image x_i\n(2.12:1 Ratio)", ha='center', va='center', fontsize=8, fontweight='bold', color='#1e293b')
    
    ax.annotate("", xy=(2.4, 4.225), xytext=(2.1, 4.225), arrowprops=dict(arrowstyle="->", lw=1.5, color="#475569"))
    
    b_squash = patches.FancyBboxPatch((2.4, 3.8), 1.8, 0.85, boxstyle="round,pad=0.08", facecolor="#dbeafe", edgecolor="#2563eb", lw=1.5)
    ax.add_patch(b_squash)
    ax.text(3.3, 4.225, "Aspect Squash\n(224x224 + TTA)", ha='center', va='center', fontsize=8, fontweight='bold', color='#1e40af')
    
    ax.annotate("", xy=(4.5, 4.225), xytext=(4.2, 4.225), arrowprops=dict(arrowstyle="->", lw=1.5, color="#2563eb"))

    b_b1 = patches.FancyBboxPatch((4.5, 4.35), 2.2, 0.55, boxstyle="round,pad=0.05", facecolor="#fae8ff", edgecolor="#c026d3", lw=1.5)
    ax.add_patch(b_b1)
    ax.text(5.6, 4.625, "BioCLIP-2.5 (ViT-H/14)", ha='center', va='center', fontsize=7.5, fontweight='bold', color='#701a75')

    b_b2 = patches.FancyBboxPatch((4.5, 3.65), 2.2, 0.55, boxstyle="round,pad=0.05", facecolor="#e0e7ff", edgecolor="#4f46e5", lw=1.5)
    ax.add_patch(b_b2)
    ax.text(5.6, 3.925, "TaxaBind (ViT-B/16)", ha='center', va='center', fontsize=7.5, fontweight='bold', color='#3730a3')

    b_b3 = patches.FancyBboxPatch((4.5, 2.95), 2.2, 0.55, boxstyle="round,pad=0.05", facecolor="#fef3c7", edgecolor="#d97706", lw=1.5)
    ax.add_patch(b_b3)
    ax.text(5.6, 3.225, "BioCLIP-2 (ViT-L/14)", ha='center', va='center', fontsize=7.5, fontweight='bold', color='#92400e')

    ax.annotate("", xy=(7.0, 3.925), xytext=(6.7, 3.925), arrowprops=dict(arrowstyle="->", lw=1.5, color="#1e293b"))

    b_gate = patches.FancyBboxPatch((7.0, 3.3), 2.5, 1.1, boxstyle="round,pad=0.08", facecolor="#dcfce7", edgecolor="#16a34a", lw=2)
    ax.add_patch(b_gate)
    ax.text(8.25, 3.85, "Novelty Gate G(x)\nz1(img) + 2.0*z1(margin)\nCutoff: Top 72%", ha='center', va='center', fontsize=7.5, fontweight='bold', color='#14532d')

    ax.annotate("Top 72%\n(Seen)", xy=(8.25, 2.7), xytext=(8.25, 3.3), arrowprops=dict(arrowstyle="->", lw=1.8, color="#16a34a"), fontsize=7.5, fontweight='bold', color='#15803d', ha='right')
    ax.annotate("Bottom 28%\n(Unseen)", xy=(4.0, 2.0), xytext=(7.0, 3.5), arrowprops=dict(arrowstyle="->", lw=1.8, color="#dc2626"), fontsize=7.5, fontweight='bold', color='#b91c1c')

    b_seen = patches.FancyBboxPatch((7.0, 1.5), 2.5, 1.0, boxstyle="round,pad=0.08", facecolor="#ecfdf5", edgecolor="#059669", lw=1.5)
    ax.add_patch(b_seen)
    ax.text(8.25, 2.0, "SEEN HEAD (5,795 Species)\nDirect Prototype Argmax", ha='center', va='center', fontsize=7.5, fontweight='bold', color='#065f46')

    b_unseen_box = patches.FancyBboxPatch((0.3, 0.3), 6.2, 1.9, boxstyle="round,pad=0.1", facecolor="#fff1f2", edgecolor="#e11d48", lw=1.5)
    ax.add_patch(b_unseen_box)
    ax.text(3.4, 2.0, "UNSEEN HEAD (11,598 Novel Species) - Logit Fusion", ha='center', va='center', fontsize=8, fontweight='bold', color='#9f1239')

    ax.text(1.3, 1.4, "1. Fused Text Base", ha='center', va='center', fontsize=7, color='#881337', bbox=dict(boxstyle="round,pad=0.25", fc="#ffffff", ec="#fda4af"))
    ax.text(3.4, 1.4, "2. Photo-Bank (TOPM=4)", ha='center', va='center', fontsize=7, color='#881337', bbox=dict(boxstyle="round,pad=0.25", fc="#ffffff", ec="#fda4af"))
    ax.text(5.5, 1.4, "3. BioCLIP-2 Protos", ha='center', va='center', fontsize=7, color='#881337', bbox=dict(boxstyle="round,pad=0.25", fc="#ffffff", ec="#fda4af"))

    b_db = patches.FancyBboxPatch((0.5, 0.5), 2.7, 0.55, boxstyle="round,pad=0.05", facecolor="#fffbeb", edgecolor="#d97706", lw=1.2)
    ax.add_patch(b_db)
    ax.text(1.85, 0.775, "Dual Temp Norm (dbnorm)", ha='center', va='center', fontsize=7, fontweight='bold', color='#78350f')

    b_sink = patches.FancyBboxPatch((3.5, 0.5), 2.8, 0.55, boxstyle="round,pad=0.05", facecolor="#f3e8ff", edgecolor="#9333ea", lw=1.2)
    ax.add_patch(b_sink)
    ax.text(4.9, 0.775, "Sinkhorn Transport (50 iter)", ha='center', va='center', fontsize=7, fontweight='bold', color='#581c87')

    ax.annotate("", xy=(3.5, 0.775), xytext=(3.2, 0.775), arrowprops=dict(arrowstyle="->", lw=1.2, color="#78350f"))
    ax.annotate("", xy=(7.8, 0.65), xytext=(6.5, 0.775), arrowprops=dict(arrowstyle="->", lw=1.5, color="#1e293b"))
    ax.annotate("", xy=(8.25, 1.0), xytext=(8.25, 1.5), arrowprops=dict(arrowstyle="->", lw=1.5, color="#1e293b"))

    b_out = patches.FancyBboxPatch((7.2, 0.2), 2.1, 0.8, boxstyle="round,pad=0.08", facecolor="#1e293b", edgecolor="#0f172a", lw=2)
    ax.add_patch(b_out)
    ax.text(8.25, 0.6, "FINAL PREDICTION\nSpecies Class ID", ha='center', va='center', fontsize=8, fontweight='bold', color='#ffffff')

    ax.set_xlim(0, 9.8)
    ax.set_ylim(0.1, 5.2)
    plt.tight_layout()
    sys_arch_path = os.path.join(SCRATCH_DIR, "system_architecture.png")
    plt.savefig(sys_arch_path, bbox_inches='tight')
    plt.close()
    return sys_arch_path

# ---------------------------------------------------------
# Step 2: PDF Document Construction via ReportLab
# ---------------------------------------------------------

def build_pdf():
    print("Generating Matplotlib charts & processing AI infographics...")
    impact_chart_img = generate_impact_chart()
    sys_arch_img = generate_accurate_system_architecture()
    
    print("Setting up ReportLab document...")
    doc = SimpleDocTemplate(
        PDF_PATH,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36
    )
    
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle('DocTitle', parent=styles['Heading1'], fontName='Helvetica-Bold', fontSize=18, leading=22, textColor=colors.HexColor('#1b2a4a'), alignment=TA_CENTER, spaceAfter=4)
    subtitle_style = ParagraphStyle('DocSubtitle', parent=styles['Normal'], fontName='Helvetica', fontSize=10, leading=13, textColor=colors.HexColor('#4a5568'), alignment=TA_CENTER, spaceAfter=10)
    section_heading = ParagraphStyle('SecHeading', parent=styles['Heading2'], fontName='Helvetica-Bold', fontSize=12, leading=15, textColor=colors.HexColor('#1b2a4a'), spaceBefore=8, spaceAfter=4)
    body_style = ParagraphStyle('BodyTextCustom', parent=styles['Normal'], fontName='Helvetica', fontSize=8.5, leading=12, textColor=colors.HexColor('#2d3748'), spaceAfter=3)
    impact_badge_style = ParagraphStyle('ImpactBadge', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=8.5, leading=11, textColor=colors.HexColor('#742a2a'))

    story = []
    
    # ---------------------------------------------------------
    # Page 1: Header Banner & Master System Architecture
    # ---------------------------------------------------------
    story.append(Paragraph("🐟 How We Recognized 35,665 Fish Species with AI!", title_style))
    story.append(Paragraph("<b>The Simple 10-Year-Old's Visual Guide to the v41 System</b><br/><i>CV4Ecology Competition 16815 • Peak Accuracy: 50.57%</i>", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#1b2a4a'), spaceAfter=6))
    
    intro_bullets = [
        "• <b>The Goal:</b> Classify 35,665 mystery fish photos across 17,393 target species.",
        "• <b>The Challenge:</b> Over half of the fish (11,598 novel species) have 0 training photos—only text encyclopedia descriptions!",
        "• <b>The Record Result:</b> Our v41 System achieved state-of-the-art 50.57% peak accuracy using 4 core intuitions."
    ]
    for b in intro_bullets:
        story.append(Paragraph(b, body_style))
    story.append(Spacer(1, 6))
    
    story.append(Paragraph("🏛️ Master System Architecture Flowchart", section_heading))
    story.append(Image(sys_arch_img, width=540, height=280))
    story.append(Spacer(1, 6))

    # Page Break for Visual AI Feature Infographics
    story.append(PageBreak())
    
    # ---------------------------------------------------------
    # Page 2: High-Resolution AI Visual Infographics
    # ---------------------------------------------------------
    story.append(Paragraph("🎨 Core Visual Intuitions (High-Resolution Infographics)", section_heading))
    
    # AI Infographic 1: Aspect Crop
    if os.path.exists(AI_IMG_1):
        story.append(Paragraph("<b>1. Aspect-Shift Crop & Squashing (The Tailor)</b>", body_style))
        story.append(Image(AI_IMG_1, width=520, height=260))
        story.append(Paragraph("• <i>Mechanism:</i> Standard square crops chop off fish snouts and tails. Wide rectangular fitting ($2.12:1$) + aspect squashing preserves full fish anatomy (+2.07 pt unseen gain).", body_style))
        story.append(Spacer(1, 8))

    # AI Infographic 2: Smart Bouncer Gate
    if os.path.exists(AI_IMG_2):
        story.append(Paragraph("<b>2. Model-Derived Soft Novelty Gating (The Smart Bouncer)</b>", body_style))
        story.append(Image(AI_IMG_2, width=520, height=260))
        story.append(Paragraph("• <i>Mechanism:</i> Evaluates face recognition (<code>img_seenmax</code>) and dress code contrast (<code>text_margin</code>). Routes top 72% to Known Room and bottom 28% to Unseen Room (+0.57 pt overall gain, 0.957 AUC).", body_style))

    story.append(PageBreak())

    # AI Infographic 3: Photo-Bank Maxpooling
    if os.path.exists(AI_IMG_3):
        story.append(Paragraph("<b>3. iNaturalist Photo-Bank Maxpooling (The 4-Photo Album)</b>", body_style))
        story.append(Image(AI_IMG_3, width=520, height=260))
        story.append(Paragraph("• <i>Mechanism:</i> Looking at 4 real reference photos showing male, female, and juvenile fish morphs beats reading 1 text paragraph (+3.43 pt unseen gain).", body_style))
        story.append(Spacer(1, 8))

    # AI Infographic 4: Sinkhorn Transport
    if os.path.exists(AI_IMG_4):
        story.append(Paragraph("<b>4. Transductive Sinkhorn Optimal Transport (The Traffic Cop)</b>", body_style))
        story.append(Image(AI_IMG_4, width=520, height=260))
        story.append(Paragraph("• <i>Mechanism:</i> Traffic police balancing algorithm forces a fair share of predictions across all 11,598 novel candidate species, stopping single-species hoarding (+2.77 pt unseen gain).", body_style))

    story.append(PageBreak())

    # ---------------------------------------------------------
    # Page 4: Step-by-Step Flow-in-Flow Trace & 10 Techniques Summary
    # ---------------------------------------------------------
    story.append(Paragraph("🔄 Step-by-Step 'Flow in Flow' Execution Trace (Sample Image #3521)", section_heading))
    
    flow_steps = [
        "<b>Step 1: Aspect Squashing:</b> Image #3521 (Regal Blue Tang / <i>P. hepatus</i>, $2.12:1$ ratio) squashed to $224 \\times 224$ + H-flip TTA.",
        "<b>Step 2: Multi-Modal Vectors:</b> BioCLIP-2.5 ($1024$-d) + TaxaBind ($512$-d) + BioCLIP-2 ($768$-d) extracted simultaneously.",
        "<b>Step 3: Gating Check:</b> $G(x) = -1.15 + 2.0 \\cdot (-0.42) = -1.99 < \\theta_{\\text{gate}} (-0.21) \\implies$ <b>Routed to UNSEEN HEAD!</b>",
        "<b>Step 4: Logit Fusion:</b> Text Base ($2.40$) + Photo-Bank Maxpool ($4.16$) + BioCLIP-2 Proto ($2.85$) $\\implies$ Raw Logit ($9.41$).",
        "<b>Step 5: Dual Normalization:</b> <code>dbnorm</code> scales logits across column ($\\tau=0.05$) & row ($\\tau=0.50$) variance.",
        "<b>Step 6: Sinkhorn Balancing:</b> 50 Sinkhorn iterations balance batch target allocations.",
        "<b>Step 7: Final Prediction:</b> Class #8412 ➔ <b>Paracanthurus hepatus (Regal Blue Tang)</b> 🎉"
    ]

    for step_text in flow_steps:
        p_step = Paragraph(step_text, body_style)
        t_box = Table([[p_step]], colWidths=[530])
        t_box.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f8fafc')),
            ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#cbd5e0')),
            ('PADDING', (0,0), (-1,-1), 4),
        ]))
        story.append(KeepTogether([t_box, Spacer(1, 3)]))

    story.append(Spacer(1, 6))

    # 10 Techniques Master Lookup
    story.append(Paragraph("🚀 Summary Lookup: All 10 Core Techniques of v41", section_heading))
    
    techniques = [
        ("1. Aspect-Shift Crop & Squash", "Wide crop (0.5, 2.0) + squash_tta 1", "+2.07 pt unseen gain"),
        ("2. Multi-Modal Backbone Ensemble", "BioCLIP-2.5 + TaxaBind + BioCLIP-2", "+1.15 pt baseline gain"),
        ("3. Model-Derived Soft Novelty Gate", "z1(img_seenmax) + 2.0*z1(text_margin)", "+0.57 pt overall (0.957 AUC)"),
        ("4. Asymmetric 72% Routing Cutoff", "Safe-bet routing fraction f = 0.72", "+1.35 pt overall gain"),
        ("5. iNaturalist Photo-Bank Maxpooling", "TOPM=4 over 73,322 iNat reference photos", "+3.43 pt unseen gain"),
        ("6. BioCLIP-2 Mean Prototype Leg", "Secondary mean-prototype weight B2_W = 2.5", "+0.08 pt overall (50.57% peak)"),
        ("7. Dual-Temperature Normalization", "dbnorm tau_col=0.05, tau_row=0.50", "Calibrated logit scales"),
        ("8. Transductive Sinkhorn Transport", "Traffic police batch balancing (50 iter)", "+2.77 pt unseen gain"),
        ("9. Strict Non-Transductive Fallback", "Pre-computed frozen threshold theta = -2.0423", "100% single-pipeline compliance"),
        ("10. Hard Vote Ensembling", "5-way hard vote across v44/v43/v41/v40/v37", "Final competition stability")
    ]

    for name, mech, impact in techniques:
        t_row = Paragraph(f"• <b>{name}:</b> {mech} ➔ <b>{impact}</b>", body_style)
        story.append(t_row)

    story.append(Spacer(1, 8))
    
    # Impact Progression Chart & Summary Table
    story.append(Paragraph("📊 Accuracy Impact Progression Summary", section_heading))
    story.append(Image(impact_chart_img, width=540, height=190))
    story.append(Spacer(1, 6))
    
    story.append(Paragraph("• <b>Conclusion:</b> By combining wide-aspect visual cropping, intelligent bouncer gating, real 4-photo albums, and traffic cop batch balancing, the <b>v41 System</b> achieved state-of-the-art <b>50.57% overall accuracy</b> under strict competition compliance!", body_style))

    print(f"Building document at {PDF_PATH}...")
    doc.build(story)
    
    os.system(f"cp '{PDF_PATH}' '{ARTIFACT_PDF_PATH}'")
    print(f"PDF successfully generated and saved to:\n  - {PDF_PATH}\n  - {ARTIFACT_PDF_PATH}")

if __name__ == "__main__":
    build_pdf()
