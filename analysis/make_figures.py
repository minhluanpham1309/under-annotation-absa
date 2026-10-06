"""Figures for the IASC manuscript (600 dpi PNG, print-safe). python analysis/make_figures.py"""
import os, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
A = f"{ROOT}/results/analysis_cpu"; F = f"{ROOT}/results/figures"; os.makedirs(F, exist_ok=True)
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 8.5, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True})
ASP = ['overall', 'amenities', 'food', 'service', 'location', 'ambiance', 'room', 'price', 'cleanliness']

# ---------- Figure 1: evaluation protocol
fig, ax = plt.subplots(figsize=(6.5, 2.5)); ax.axis("off"); ax.set_xlim(-1, 101); ax.set_ylim(0, 40)
def box(x, y, w, h, t, fc="#f3f2ee"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fc, ec=MUTED, lw=0.8))
    ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=6.6, color=INK, wrap=True)
def arr(x1, y1, x2, y2):
    ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.8))
box(0, 26, 21, 11, "Source corpus\n70,913 sentences\nEN / VI / ZH / JA")
box(0, 3, 21, 11, "Human-verified target\n892 VI sentences\n2 annotators, κ = 0.93")
box(26, 14.5, 19, 11, "Fine-tune\nmBERT / XLM-R / mT5\n(+ TF-IDF fusion)")
box(50.5, 26, 21, 11, "In-domain test\nthresholds tuned\non source validation")
box(50.5, 3, 21, 11, "Target test\nsame thresholds\n(main protocol)")
box(77, 26, 23, 11, "Ranking: F1, subset,\nprimary aspect\n+ McNemar / bootstrap", fc="#e8f0fb")
box(77, 3, 23, 11, "Diagnosis: P/R split,\nerror types, cardinality,\nthreshold re-tuning", fc="#fdeee6")
arr(21.5, 31.5, 26, 22); arr(21.5, 8.5, 50, 8.5); arr(45.5, 22, 50, 31.5); arr(45.5, 18, 50, 8.5)
arr(72, 31.5, 76.5, 31.5); arr(72, 8.5, 76.5, 8.5)
fig.savefig(f"{F}/fig1_protocol.png", dpi=600, bbox_inches="tight"); plt.close(fig)

# ---------- Figure 2: cardinality vs sentence length
lm = pd.read_csv(f"{A}/E4_cardinality_by_length.csv")
labels = ["1–5", "6–10", "11–15", "16–20", "21–30", ">30"]
x = np.arange(len(labels))
fig, ax = plt.subplots(figsize=(3.4, 2.4))
ax.plot(x, lm.vn_hotel_vi_card, "-o", color=BLUE, lw=2, ms=4.5, label="Source corpus (Vietnamese test)")
ax.plot(x, lm.target_card, "-s", color=ORANGE, lw=2, ms=4.5, label="Human-verified target set")
for xi, (a, b) in enumerate(zip(lm.vn_hotel_vi_card, lm.target_card)):
    if xi in (0, len(x) - 1):
        ax.text(xi, a - 0.28, f"{a:.2f}", ha="center", fontsize=7, color=INK)
        ax.text(xi, b + 0.14, f"{b:.2f}", ha="center", fontsize=7, color=INK)
ax.set_xticks(x, labels); ax.set_xlabel("Sentence length (words)"); ax.set_ylabel("Gold labels per sentence")
ax.set_ylim(0.6, 4.3); ax.legend(frameon=False, fontsize=7, loc="upper left")
fig.savefig(f"{F}/fig2_cardinality_length.png", dpi=600, bbox_inches="tight"); plt.close(fig)

# ---------- Figure 3: per-aspect precision / recall of XLM-R, in-domain vs target
pa = pd.read_csv(f"{ROOT}/results/analysis_3seed/M6_per_aspect.csv"); pa = pa[pa.model == "XLM-R"]  # mean over 3 seeds
fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.5), sharey=True)
w = 0.38; x = np.arange(len(ASP))
for ax, met in zip(axs, ["P", "R"]):
    i = pa[pa.split == "in-domain"].set_index("aspect").loc[ASP, met].values
    t = pa[pa.split == "target"].set_index("aspect").loc[ASP, met].values
    ax.bar(x - w / 2 - 0.01, i, w, color=BLUE, label="In-domain test")
    ax.bar(x + w / 2 + 0.01, t, w, color=ORANGE, label="Target set")
    ax.set_xticks(x, ASP, rotation=45, ha="right"); ax.set_ylim(0, 1.05)
    ax.set_title("Precision" if met == "P" else "Recall", fontsize=9, color=INK)
axs[0].set_ylabel("Score (XLM-R)"); fig.legend(*axs[0].get_legend_handles_labels(), frameon=False, fontsize=7.5, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.06))
fig.tight_layout(); fig.savefig(f"{F}/fig3_per_aspect_PR.png", dpi=600, bbox_inches="tight"); plt.close(fig)

# ---------- Figure 4: error composition
et0 = pd.read_csv(f"{A}/E7_error_types.csv"); ems = pd.read_csv(f"{ROOT}/results/analysis_3seed/M3_errors.csv")
et = pd.concat([et0[et0.model == "TF-IDF+SVM"][["model", "split", "missed_share"]], ems[["model", "split", "missed_share"]]])
order = ["TF-IDF+SVM", "mBERT", "XLM-R", "mT5"]
fig, ax = plt.subplots(figsize=(3.4, 2.4))
y = np.arange(len(order))
for k, (sp, off, col) in enumerate([("in-domain", -0.2, BLUE), ("target", 0.2, ORANGE)]):
    v = et[et.split == sp].set_index("model").loc[order, "missed_share"].values * 100
    ax.barh(y + off, v, 0.36, color=col, label="In-domain test" if sp == "in-domain" else "Target set")
    for yi, vi in zip(y, v):
        ax.text(vi + 1, yi + off, f"{vi:.0f}%", va="center", fontsize=6.8, color=INK)
ax.set_yticks(y, ["TF-IDF + SVM", "mBERT", "XLM-R", "mT5-small"]); ax.invert_yaxis()
ax.set_xlim(0, 100); ax.set_xlabel("Missed labels as % of all label errors")
ax.axvline(50, color=MUTED, lw=0.6, ls="--"); ax.legend(frameon=False, fontsize=7, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 1.0))
fig.savefig(f"{F}/fig4_error_composition.png", dpi=600, bbox_inches="tight"); plt.close(fig)
print("figures ->", F)
