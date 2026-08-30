#!/usr/bin/env python3
"""
Generate Figure 7:

Benchmarking comparison -- HALO vs. narrow-feature RF baselines
representative of CoSynE and INDIGO's original feature scope.

Internal nested pair-held-out CV results (mean +/- SD across five outer folds).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from halo.paths import FIGURES

PLOT_DIR = FIGURES / "main"
PLOT_DIR.mkdir(parents=True, exist_ok=True)
FIG7_PNG = PLOT_DIR / "fig7.png"

# ==========================
# Global style
# ==========================
TITLE_SIZE = 16
LABEL_SIZE = 16
TICK_SIZE = 13
BAR_LABEL_SIZE = 13
LEGEND_SIZE = 13

plt.rcParams.update(
    {
        "font.size": LABEL_SIZE,
        "axes.titlesize": TITLE_SIZE,
        "axes.labelsize": LABEL_SIZE,
        "xtick.labelsize": TICK_SIZE,
        "ytick.labelsize": TICK_SIZE,
    }
)

# colors (house palette)
main_blue     = "#1f77b4"
pastel_teal   = "#8DC5C1"
pastel_red    = "#E8A5A5"
pastel_green  = "#A7D7A0"
pastel_blue   = "#7BAFD4"
pastel_yellow = "#F4E3A3"
pastel_peach  = "#F7C9A9"

BAR_EDGE_COLOR = "black"
BAR_EDGE_WIDTH = 0.6
ERRORBAR_KW = dict(ecolor="black", elinewidth=1.2, capsize=4, capthick=1.2)

def clean_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ==========================
# Data (internal nested pair-held-out CV, mean +/- SD, 5 outer folds)
# ==========================
models = ["HALO", "RF\n(CoSynE-style)", "RF\n(INDIGO-style)"]
model_colors = [main_blue, pastel_teal, pastel_red]

accuracy_mean = [0.70, 0.66, 0.65]
accuracy_sd   = [0.02, 0.02, 0.01]

rocauc_mean = [0.78, 0.72, 0.72]
rocauc_sd   = [0.03, 0.03, 0.01]


def plot_metric_panel(ax, title, means, sds, ylabel, panel_letter):
    x = np.arange(len(models))

    bars = ax.bar(
        x,
        means,
        yerr=sds,
        color=model_colors,
        edgecolor=BAR_EDGE_COLOR,
        linewidth=BAR_EDGE_WIDTH,
        error_kw=ERRORBAR_KW,
        width=0.6,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=TICK_SIZE)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1.0)

    for xi, mean, sd in zip(x, means, sds):
        ax.text(
            xi,
            mean + sd + 0.03,
            f"{mean:.2f}",
            ha="center",
            va="bottom",
            fontsize=BAR_LABEL_SIZE,
        )

    ax.set_title(
        rf"$\mathbf{{{panel_letter}.}}$  {title}",
        fontsize=TITLE_SIZE,
        pad=10,
    )

    clean_axis(ax)


# ==========================
# Assemble Figure 7
# ==========================
def main():
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12, 5.5))

    plot_metric_panel(axA, "Accuracy", accuracy_mean, accuracy_sd, "Accuracy", "A")
    plot_metric_panel(axB, "ROC-AUC", rocauc_mean, rocauc_sd, "ROC-AUC", "B")

    fig.suptitle(
        "HALO vs. CoSynE- and INDIGO-representative baselines\n(nested pair-held-out CV)",
        fontsize=TITLE_SIZE,
        y=1.03,
    )

    fig.tight_layout()
    fig.savefig(FIG7_PNG, dpi=600, bbox_inches="tight")
    plt.close(fig)

    print("Saved Fig 7 PNG to:", FIG7_PNG)


if __name__ == "__main__":
    main()
