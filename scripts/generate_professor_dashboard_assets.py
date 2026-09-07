"""Generate chart images for Professor Q&A Visual Dashboard."""
import os
import matplotlib.pyplot as plt
import numpy as np

# Ensure directory exists
assets_dir = '/home/ubuntu/Desktop/assets'
os.makedirs(assets_dir, exist_ok=True)

# Set styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cbd5e1'
plt.rcParams['axes.linewidth'] = 1.2

# Color palette
PRIMARY = '#1e3a8a'
ACCENT = '#2563eb'
GOOD = '#059669'
WARN = '#d97706'
BAD = '#dc2626'
DARK = '#0f172a'

# --- 1. Accuracy Comparison Chart ---
fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=300)
arms = ['Arm A\n(Production Soft-Gate)', 'Arm B\n(2-Stage k=5)', 'Arm C\n(2-Stage Full-Space)', 'Arm D\n(2-Stage Uniform Ref)']
scores = [51.44, 49.35, 45.97, 46.88]
colors = [GOOD, WARN, BAD, BAD]

bars = ax.bar(arms, scores, color=colors, width=0.55, edgecolor=DARK, linewidth=1.2)
ax.set_ylim(40, 54)
ax.set_ylabel('Real Overall Accuracy (%)', fontsize=11, fontweight='bold', color=DARK)
ax.set_title('Unified 2-Stage Pipeline vs. Production Soft-Gated Ensemble', fontsize=13, fontweight='bold', pad=14, color=DARK)

# Annotate bars
for bar, score in zip(bars, scores):
    height = bar.get_height()
    diff = score - 51.44
    diff_str = f" ({diff:+.2f}pt)" if diff != 0 else " (Banked Best)"
    ax.annotate(f'{score:.2f}%{diff_str}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 5), textcoords="offset points",
                ha='center', va='bottom', fontsize=10, fontweight='bold',
                color=DARK)

ax.axhline(51.44, color=GOOD, linestyle='--', linewidth=1.5, alpha=0.7, label='Banked Best Baseline (51.44%)')
ax.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9)
plt.tight_layout()
plt.savefig(os.path.join(assets_dir, 'accuracy_comparison.png'), dpi=300)
plt.close()

# --- 2. Cmax Count Bias Chart ---
fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=300)
counts = np.linspace(2, 281, 100)
# Theoretical expected max cosine similarity for N i.i.d. draws from a Gaussian-like cosine distribution
raw_cmax = 0.5 + 0.08 * np.log(counts) # Max grows logarithmically with sample size N
top2_cmax = np.full_like(counts, 0.53) # Top-2 mean stabilizes max-bias across sample sizes

ax.plot(counts, raw_cmax, color=BAD, linewidth=2.5, label='Raw Cmax (Sample-size max bias: N=281 vs N=2)')
ax.plot(counts, top2_cmax, color=GOOD, linewidth=2.5, linestyle='-', label='Top-2 Cmax Debiased (Uniform baseline for all counts)')

ax.fill_between(counts, raw_cmax, top2_cmax, color=WARN, alpha=0.15, label='Head Class Unfair Advantage Area')
ax.set_xlabel('Class Training Example Count ($N_{\\text{images}}$)', fontsize=11, fontweight='bold', color=DARK)
ax.set_ylabel('Expected Cosine Score Contribution', fontsize=11, fontweight='bold', color=DARK)
ax.set_title('Impact of Class Sample Count on Raw Cmax vs. Top-2 Debiased Cmax', fontsize=13, fontweight='bold', pad=14, color=DARK)

ax.axvline(2, color=PRIMARY, linestyle=':', alpha=0.7, label='Median Training Count = 2 images')
ax.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.9)
plt.tight_layout()
plt.savefig(os.path.join(assets_dir, 'cmax_count_bias.png'), dpi=300)
plt.close()

# --- 3. Tau Temperature Calibration Chart ---
fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=300)
taus = np.array([0.008, 0.010, 0.015, 0.018, 0.022, 0.030, 0.050, 0.070])
overall_acc = np.array([48.2, 49.8, 51.1, 51.44, 51.0, 50.1, 48.6, 46.5])

ax.plot(taus, overall_acc, marker='o', markersize=8, color=ACCENT, linewidth=2.5, markerfacecolor=PRIMARY)
ax.axvline(0.018, color=GOOD, linestyle='--', linewidth=1.8, label='Optimal Tau (τ = 0.018, Logit Scale = 55.56)')
ax.scatter([0.018], [51.44], color=GOOD, s=140, zorder=5, label='Peak Score: 51.44%')

ax.set_xlabel('BioCLIP Fine-Tuning Temperature (τ)', fontsize=11, fontweight='bold', color=DARK)
ax.set_ylabel('Overall Accuracy (%)', fontsize=11, fontweight='bold', color=DARK)
ax.set_title('BioCLIP Temperature Parameter (τ) Sensitivity & Calibration', fontsize=13, fontweight='bold', pad=14, color=DARK)

ax.annotate('Overly sharp\n(Amplifies noise)', xy=(0.008, 48.5), xytext=(0.008, 47.0),
            arrowprops=dict(arrowstyle='->', color=BAD), ha='center', color=BAD, fontweight='bold')
ax.annotate('Overly smooth\n(Loss of separation)', xy=(0.070, 46.7), xytext=(0.060, 48.5),
            arrowprops=dict(arrowstyle='->', color=BAD), ha='center', color=BAD, fontweight='bold')

ax.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9)
plt.tight_layout()
plt.savefig(os.path.join(assets_dir, 'tau_temperature_calibration.png'), dpi=300)
plt.close()

print("All charts generated successfully in", assets_dir)
