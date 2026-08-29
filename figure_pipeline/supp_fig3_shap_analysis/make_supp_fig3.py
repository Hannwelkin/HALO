#!/usr/bin/env python3
"""
Supplementary Figure 3: SHAP rank vs Gain rank correlation.
No legend, no point labels – just the scatter and correlation box.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr
from halo.paths import RESULTS, FIGURES

# ---------- paths ----------
SHAP_RANK_PATH = RESULTS / "other_analysis" / "shap_vs_gain_comparison.csv"
PLOT_DIR = FIGURES / "supplementary"
PLOT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = PLOT_DIR / "supp_fig3.png"

# ---------- style ----------
plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
})

MAIN_BLUE = "#1f77b4"
PASTEL_PEACH = "#F7C9A9"
BAR_EDGE_COLOR = "black"

def clean_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_shap_vs_gain(ax, df_rank):
    df = df_rank.copy()
    df["metric"] = df["feature"].apply(
        lambda x: "cosine" if x.startswith("cos") else "euclidean"
    )
    colors = {"cosine": MAIN_BLUE, "euclidean": PASTEL_PEACH}

    for metric, group in df.groupby("metric"):
        ax.scatter(
            group["gain_rank"],
            group["shap_rank"],
            color=colors[metric],
            alpha=0.6,
            s=40,
            edgecolor=BAR_EDGE_COLOR,
            linewidth=0.2,
        )

    # Diagonal y = x
    ax.plot([0, 1000], [0, 1000], "k--", alpha=0.5, linewidth=1)

    ax.legend().remove()

    ax.set_xlabel("Gain rank (lower = more important)")
    ax.set_ylabel("SHAP rank (lower = more important)")

    # Fixed limits: upper bound exactly 1000
    ax.set_xlim(-50, 1000)
    ax.set_ylim(-50, 1000)

    # Correlation box remains the same
    pearson_r = np.corrcoef(df["gain_rank"], df["shap_rank"])[0, 1]
    spearman_rho, _ = spearmanr(df["gain_rank"], df["shap_rank"])
    ax.text(
        0.95, 0.95,
        f"Pearson r = {pearson_r:.2f}\nSpearman ρ = {spearman_rho:.2f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=14,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="none", alpha=0.9),
    )

    ax.grid(False)
    clean_axis(ax)


def main():
    if not SHAP_RANK_PATH.exists():
        raise FileNotFoundError(f"File not found: {SHAP_RANK_PATH}")

    df = pd.read_csv(SHAP_RANK_PATH)
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_shap_vs_gain(ax, df)

    fig.tight_layout()
    fig.savefig(OUTPUT_FILE, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()