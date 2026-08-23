#!/usr/bin/env python3
"""
figure_pipeline/fig6_external_validation_acdb/make_fig6.py

Generate Figure 6:

External validation sanity check on ACDB dataset using HALO

Panel outputs:
- fig6_panelB.png (ROC)
- fig6_panelC.png (PR)
- fig6_panelD.png (Boxplot of predicted probabilities by true class)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score

from halo.paths import FIGURE_PIPELINE, RESULTS

BASE_DIR = RESULTS / "external_validation" / "external_validation_acdb"
OUT_DIR = FIGURE_PIPELINE / "fig6_external_validation_acdb" / "fig6_panels"
OUT_DIR.mkdir(parents=True, exist_ok=True)

roc_path = BASE_DIR / "external_roc_curve_acdb.csv"
pr_path = BASE_DIR / "external_pr_curve_acdb.csv"
cm_path = BASE_DIR / "external_confusion_matrix_acdb.csv"
pred_path = BASE_DIR / "external_predictions_acdb.csv"

# Color palette
main_blue   = "#1f77b4"
pastel_green  = "#A7D7A0"
pastel_red    = "#E8A5A5"
warm_gray     = "#C7C7C7"


def load_data():
    roc_df = pd.read_csv(roc_path).copy()
    pr_df = pd.read_csv(pr_path).copy()
    cm_df = pd.read_csv(cm_path, index_col=0).copy()
    pred_df = pd.read_csv(pred_path).copy()

    # verify the Interaction Type column exists
    if "Interaction Type" not in pred_df.columns:
        raise ValueError("'Interaction Type' column not found in external_predictions CSV.")
    
    return roc_df, pr_df, cm_df, pred_df


def make_fig6():
    roc_df, pr_df, cm_df, pred_df = load_data()

    # ===== Basic metrics =====
    # Build y_true directly from the categorical column
    y_true = (pred_df["Interaction Type"] == "synergy").astype(int).values
    p_synergy = pred_df["p_synergy"].values

    auc = roc_auc_score(y_true, p_synergy)
    ap = average_precision_score(y_true, p_synergy)

    # Confusion matrix from CSV (rows: true antag / syn, cols: pred antag / syn)
    cm = cm_df.values.astype(int)
    tn, fp, fn, tp = cm.ravel()  # tn_csv, fp_csv, fn_csv, tp_csv

    # Precompute ROC + PR arrays
    fpr = roc_df["fpr"].values
    tpr = roc_df["tpr"].values
    rec = pr_df["recall"].values
    prec = pr_df["precision"].values
    pos_rate = y_true.mean()

    # Boolean masks for true labels
    mask_syn = y_true == 1
    mask_ant = y_true == 0

    # ==========================
    # Main figure: 3 panels (B, C, D)
    # ==========================
    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.2])

    ax_roc     = fig.add_subplot(gs[0, 0])   # Panel B
    ax_pr      = fig.add_subplot(gs[0, 1])   # Panel C
    ax_boxplot = fig.add_subplot(gs[1, :])   # Panel D (bottom, full width) - changed to boxplot

    # --------------------------
    # Panel B: ROC curve
    # --------------------------
    ax_roc.plot(fpr, tpr, label=f"AUC = {auc:.2f}", color=main_blue, lw=2)
    ax_roc.plot([0, 1], [0, 1], linestyle="--", color=warm_gray, lw=1)
    ax_roc.set_xlim(0, 1)
    ax_roc.set_ylim(0, 1)
    ax_roc.set_xlabel("False positive rate")
    ax_roc.set_ylabel("True positive rate")
    ax_roc.set_title(r"$\mathbf{B.}$" + " ROC on external set", loc="center", pad=14)
    ax_roc.legend(frameon=False)
    ax_roc.grid(alpha=0.2, color=warm_gray)

    # --------------------------
    # Panel C: Precision–Recall
    # --------------------------
    ax_pr.plot(rec, prec, color=main_blue, lw=2, label=f"AP = {ap:.2f}")
    ax_pr.hlines(pos_rate, 0, 1, linestyle="--", color=warm_gray, lw=1,
                 label=f"Pos. prevalence = {pos_rate:.2f}")
    ax_pr.set_xlim(0, 1)
    ax_pr.set_ylim(0, 1)
    ax_pr.set_xlabel("Recall (synergy)")
    ax_pr.set_ylabel("Precision (synergy)")
    ax_pr.set_title(r"$\mathbf{C.}$" + " Precision–recall curve", loc="center", pad=14)
    ax_pr.legend(frameon=False)
    ax_pr.grid(alpha=0.2, color=warm_gray)

    # --------------------------
    # Panel D: Boxplot of predicted P(synergy) by true Interaction Type
    # --------------------------
    # Prepare data for boxplot
    data_to_plot = [p_synergy[mask_ant], p_synergy[mask_syn]]
    bp = ax_boxplot.boxplot(data_to_plot, 
                            labels=["Antagonism", "Synergy"],
                            patch_artist=True,
                            showmeans=True,
                            meanline=True,
                            meanprops={'color': 'black', 'linestyle': '--', 'linewidth': 1.5},
                            boxprops={'facecolor': pastel_red, 'edgecolor': 'black'},
                            whiskerprops={'color': 'black'},
                            capprops={'color': 'black'},
                            medianprops={'color': 'black', 'linewidth': 2})

    # Color the boxes distinctly (Antag=Red, Syn=Green)
    bp['boxes'][0].set_facecolor(pastel_red)
    bp['boxes'][1].set_facecolor(pastel_green)

    ax_boxplot.set_ylim(-0.02, 1.02)
    ax_boxplot.set_ylabel("Predicted P(synergy)")
    ax_boxplot.set_xlabel("True Interaction Type")
    ax_boxplot.set_title(r"$\mathbf{D.}$" + " Predicted synergy probability by true class", 
                         loc="center", pad=14)
    ax_boxplot.grid(alpha=0.2, color=warm_gray, axis='y')

    # Add counts (N) below the x-axis labels
    n_ant = mask_ant.sum()
    n_syn = mask_syn.sum()
    ax_boxplot.text(1, -0.08, f"N = {n_ant}", ha='center', va='top', transform=ax_boxplot.transData)
    ax_boxplot.text(2, -0.08, f"N = {n_syn}", ha='center', va='top', transform=ax_boxplot.transData)

    # layout
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.6, wspace=0.4, left=0.10, right=0.90, top=0.94, bottom=0.09)

    # =======================================================
    # Save PANELS B, C, D independently
    # =======================================================

    # Panel B: ROC
    figB, axB = plt.subplots(figsize=(4.5, 4))
    axB.plot(fpr, tpr, color=main_blue, lw=2, label=f"AUC = {auc:.2f}")
    axB.plot([0, 1], [0, 1], linestyle="--", color=warm_gray, lw=1)
    axB.set_xlim(0, 1)
    axB.set_ylim(0, 1)
    axB.set_xlabel("False positive rate")
    axB.set_ylabel("True positive rate")
    axB.set_title(r"$\mathbf{B.}$" + " ROC on external set")
    axB.legend(frameon=False)
    axB.grid(alpha=0.2, color=warm_gray)
    figB.tight_layout()
    out_png_B = OUT_DIR / "fig6_panelB_ROC.png"
    figB.savefig(out_png_B, dpi=600)
    plt.close(figB)

    # Panel C: PR curve
    figC, axC = plt.subplots(figsize=(4.5, 4))
    axC.plot(rec, prec, color=main_blue, lw=2, label=f"AP = {ap:.2f}")
    axC.hlines(pos_rate, 0, 1, linestyle="--", color=warm_gray, lw=1,
               label=f"Pos. prevalence = {pos_rate:.2f}")
    axC.set_xlim(0, 1)
    axC.set_ylim(0, 1)
    axC.set_xlabel("Recall (synergy)")
    axC.set_ylabel("Precision (synergy)")
    axC.set_title(r"$\mathbf{C.}$" + " Precision–recall curve")
    axC.legend(frameon=False)
    axC.grid(alpha=0.2, color=warm_gray)
    figC.tight_layout()
    out_png_C = OUT_DIR / "fig6_panelC_PR.png"
    figC.savefig(out_png_C, dpi=600)
    plt.close(figC)

    # Panel D: Boxplot (independent)
    figD, axD = plt.subplots(figsize=(6, 4))
    bpD = axD.boxplot(data_to_plot, 
                      labels=["Antagonism", "Synergy"],
                      patch_artist=True,
                      showmeans=True,
                      meanline=True,
                      meanprops={'color': 'black', 'linestyle': '--', 'linewidth': 1.5},
                      boxprops={'facecolor': pastel_red, 'edgecolor': 'black'},
                      whiskerprops={'color': 'black'},
                      capprops={'color': 'black'},
                      medianprops={'color': 'black', 'linewidth': 2})
    bpD['boxes'][0].set_facecolor(pastel_red)
    bpD['boxes'][1].set_facecolor(pastel_green)
    axD.set_ylim(-0.02, 1.02)
    axD.set_ylabel("Predicted P(synergy)")
    axD.set_xlabel("True Interaction Type")
    # Instead of using separate text annotations, simply redefine the tick labels
    axD.set_xticklabels([f"Antagonism\nN = {n_ant}", f"Synergy\nN = {n_syn}"])
    axD.set_title(r"$\mathbf{D.}$" + " Predicted synergy probability by true class")
    axD.grid(alpha=0.2, color=warm_gray, axis='y')
    # axD.text(1, -0.08, f"N = {n_ant}", ha='center', va='top', transform=axD.transData)
    # axD.text(2, -0.08, f"N = {n_syn}", ha='center', va='top', transform=axD.transData)

    figD.tight_layout()
    out_png_D = OUT_DIR / "fig6_panelD_boxplot.png"
    figD.savefig(out_png_D, dpi=600)
    plt.close(figD)

    print("Saved independent panels:")
    print("  ", out_png_B)
    print("  ", out_png_C)
    print("  ", out_png_D)

    plt.close(fig)


if __name__ == "__main__":
    make_fig6()


