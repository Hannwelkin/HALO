#!/usr/bin/env python3
"""
Figure 4 (reordered & restyled):

- A: CC domain contributions (HALO, pie)
- B: CC vs strain‑space contributions (M2, pie)
- C: Top feature groups by normalized gain (HALO, bar)

Layout:
  Top row: A (left) and B (right) – equal width
  Bottom row: C (full width) – slightly shorter
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from halo.paths import MODEL_RESULTS, FEATURES, FIGURES

# ---------- paths ----------
RESULT_DIR_EXP06D = MODEL_RESULTS / "exp06d_lgbm_bin_nosspace_elementwise_reduced_nestedcv"
FI_PATH_EXP06D = RESULT_DIR_EXP06D / "feature_importances_cv1.csv"

RESULT_DIR_EXP06B = MODEL_RESULTS / "exp06b_lgbm_bin_sspace_elementwise_reduced_nestedcv"
CC_VS_SS_SUMMARY_PATH = RESULT_DIR_EXP06B / "cc_vs_ss_importance_summary.csv"

PLOT_DIR = FIGURES / "main"
PLOT_DIR.mkdir(parents=True, exist_ok=True)
FIG4_PNG = PLOT_DIR / "fig4.png"

FEATURE_META_PATH = FEATURES / "feature_metadata_cc_s_full.csv"

# ---------- style ----------
TITLE_SIZE = 16
LABEL_SIZE = 14
TICK_SIZE = 11
BAR_LABEL_SIZE = 13

plt.rcParams.update({
    "font.size": LABEL_SIZE,
    "axes.titlesize": TITLE_SIZE,
    "axes.labelsize": LABEL_SIZE,
    "xtick.labelsize": TICK_SIZE,
    "ytick.labelsize": TICK_SIZE,
})

MAIN_BLUE = "#1f77b4"
PASTEL_PEACH = "#F7C9A9"
BAR_EDGE_COLOR = "black"
BAR_EDGE_WIDTH = 0.3

# ---------- helpers ----------
def short_label(text):
    replacements = {
        "Chemical genetics": "Chem. genetics",
        "Mechanisms of action": "Mech. of action",
        "Small molecule roles": "Small mol. roles",
        "Small molecule pathways": "Small mol. pathways",
        "Structural keys": "Struct. keys",
        "Metabolic genes": "Metab. genes",
        "Side effects": "Side effects",
        "2D fingerprints": "2D fingerprints",
        "Indications": "Indications",
        "Transcription": "Transcript.",
        "Therapeutic areas": "Therap. areas",
        "Diseases & toxicology": "Disease/tox.",
        "Cancer cell lines": "Cancer cell lines",
        "Signaling pathways": "Signaling paths.",
    }
    return replacements.get(text, text)

def clean_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# ---------- data loading ----------
def load_importances(fi_path: Path) -> pd.DataFrame:
    if not fi_path.exists():
        raise FileNotFoundError(fi_path)
    df = pd.read_csv(fi_path).copy()
    if "importance_gain_norm" not in df.columns:
        total_gain = df["importance_gain"].sum()
        if total_gain <= 0:
            raise ValueError("Total importance_gain is non-positive; cannot normalize.")
        df["importance_gain_norm"] = df["importance_gain"] / total_gain
    df = df.sort_values("importance_gain", ascending=False).reset_index(drop=True)
    return df

def load_feature_meta():
    if not FEATURE_META_PATH.exists():
        raise FileNotFoundError(FEATURE_META_PATH)
    meta = pd.read_csv(FEATURE_META_PATH).copy()
    if "original_name" not in meta.columns:
        raise ValueError("feature_metadata_cc_s_full.csv must have 'original_name' column")
    meta = meta.set_index("original_name")
    cc_dims = meta[meta["space"] == "CC"]["dimension"].dropna()
    if cc_dims.empty:
        raise ValueError("No CC dimensions found in metadata (space == 'CC').")
    n_cc_dims = int(cc_dims.max()) + 1
    return meta, n_cc_dims

# ---------- decoding ----------
def decode_elementwise_feature(feat_name: str, meta: pd.DataFrame, n_cc_dims: int):
    name = feat_name.lower()
    if name.startswith("cos_elem_"):
        metric = "cosine"
        try:
            idx = int(name.split("_")[-1])
        except ValueError:
            idx = None
    elif name.startswith("euc_elem_"):
        metric = "euclidean"
        try:
            idx = int(name.split("_")[-1])
        except ValueError:
            idx = None
    else:
        return {"metric": "unknown", "space": "Unknown", "space_name": "Unknown",
                "cc_level": None, "cc_sublevel": None, "cc_level_name": None,
                "cc_sublevel_name": None, "group_label": "Unknown", "base_feature": None}
    if idx is None:
        return {"metric": metric, "space": "Unknown", "space_name": "Unknown",
                "cc_level": None, "cc_sublevel": None, "cc_level_name": None,
                "cc_sublevel_name": None, "group_label": "Unknown", "base_feature": None}
    if idx < n_cc_dims:
        base_feature = f"dim_{idx}"
    else:
        s_idx = idx - n_cc_dims
        base_feature = f"s_{s_idx}"
    if base_feature not in meta.index:
        return {"metric": metric, "space": "Unknown", "space_name": "Unknown",
                "cc_level": None, "cc_sublevel": None, "cc_level_name": None,
                "cc_sublevel_name": None, "group_label": "Unknown", "base_feature": base_feature}
    row = meta.loc[base_feature]
    return {
        "metric": metric,
        "space": row.get("space", "Unknown"),
        "space_name": row.get("space_name", "Unknown"),
        "cc_level": row.get("cc_level", None),
        "cc_sublevel": row.get("cc_sublevel", None),
        "cc_level_name": row.get("cc_level_name", None),
        "cc_sublevel_name": row.get("cc_sublevel_name", None),
        "group_label": row.get("group_label", row.get("space_name", "Unknown")),
        "base_feature": base_feature,
    }

# ---------- Panel A (new): CC domain contributions (pie) ----------
def plot_panel_A_pie(ax, df_importance, meta, n_cc_dims):
    """CC domain contributions (HALO) – pie chart."""
    df_imp = df_importance.copy()
    df_imp = df_imp[df_imp["feature"].str.startswith(("cos_elem_", "euc_elem_"))]
    decoded = df_imp["feature"].apply(lambda f: decode_elementwise_feature(f, meta, n_cc_dims))
    df_imp = pd.concat([df_imp, pd.DataFrame(list(decoded))], axis=1)

    def level_group(row):
        if (row["space_name"] == "Strain-space" or row["group_label"] == "Strain-space" or row["cc_level"] == "Strain"):
            return "Strain-space"
        if isinstance(row["cc_level_name"], str) and row["cc_level_name"]:
            return row["cc_level_name"]
        if row["space_name"] == "Chemical Checker":
            return "Chemical Checker"
        return "Unknown"

    df_imp["level_group"] = df_imp.apply(level_group, axis=1)
    grouped = df_imp.groupby("level_group")["importance_gain_norm"].sum().reset_index()
    grouped = grouped[grouped["level_group"] != "Strain-space"].copy()
    desired_order = ["Chemistry", "Targets", "Networks", "Cells", "Clinics"]
    grouped["level_group"] = pd.Categorical(grouped["level_group"], categories=desired_order, ordered=True)
    grouped = grouped.sort_values("level_group")
    colors = plt.cm.Greens(np.linspace(0.4, 0.8, len(grouped)))

    wedges, texts, autotexts = ax.pie(
        grouped["importance_gain_norm"],
        labels=grouped["level_group"],
        autopct="%1.1f%%",
        colors=colors,
        startangle=90,
        textprops={"fontsize": 14},
        wedgeprops={"edgecolor": "white", "linewidth": 0.5},
    )
    ax.set_title(r"$\mathbf{A.}$  CC domain contributions", fontsize=TITLE_SIZE, pad=12)
    for autotext in autotexts:
        autotext.set_color("white")
        autotext.set_fontweight("bold")

# ---------- Panel B (new): CC vs SS contributions (pie) ----------
def plot_panel_B_pie(ax, cc_vs_ss_summary):
    """CC vs strain‑space gain contributions (M2) – pie chart."""
    df = cc_vs_ss_summary.copy()
    desired_order = ["CC", "SS"]
    value_map = dict(zip(df["group"], df["fraction_of_total_gain"]))
    plot_df = pd.DataFrame({
        "group": desired_order,
        "fraction_of_total_gain": [value_map.get(g, 0.0) for g in desired_order],
    })
    colors = ["#1f77b4", "#F7C9A9"]
    wedges, texts, autotexts = ax.pie(
        plot_df["fraction_of_total_gain"],
        labels=plot_df["group"],
        autopct="%1.1f%%",
        colors=colors,
        startangle=90,
        textprops={"fontsize": 14},
        wedgeprops={"edgecolor": "white", "linewidth": 0.5},
    )
    ax.set_title(r"$\mathbf{B.}$  CC vs strain‑space gain", fontsize=TITLE_SIZE, pad=12)
    for autotext in autotexts:
        autotext.set_color("white")
        autotext.set_fontweight("bold")

# ---------- Panel C (new): top feature groups (bar) ----------
def plot_panel_C_bar(ax, df_importance, meta, n_cc_dims, top_n=10):
    """
    Collective feature groups (CC sublevels + Strain-space) by normalized gain.
    Each bar gets a distinct shade of green.
    """
    df_imp = df_importance.copy()
    df_imp = df_imp[df_imp["feature"].str.startswith(("cos_elem_", "euc_elem_"))]
    decoded = df_imp["feature"].apply(lambda f: decode_elementwise_feature(f, meta, n_cc_dims))
    df_imp = pd.concat([df_imp, pd.DataFrame(list(decoded))], axis=1)

    def sub_group(row):
        if (row["space_name"] == "Strain-space" or row["group_label"] == "Strain-space" or row["cc_level"] == "Strain"):
            return "Strain-space"
        sub = row.get("cc_sublevel_name")
        if isinstance(sub, str) and sub:
            return sub
        return row["group_label"]

    df_imp["sub_group"] = df_imp.apply(sub_group, axis=1)
    grouped = df_imp.groupby("sub_group")["importance_gain_norm"].sum().sort_values(ascending=False).head(top_n)
    plot_df = grouped.sort_values(ascending=True).reset_index()
    plot_df.columns = ["sub_group", "importance_gain_norm"]
    plot_df["sub_group"] = plot_df["sub_group"].map(short_label)

    # Generate distinct green shades
    n_bars = len(plot_df)
    colors = plt.cm.Greens(np.linspace(0.4, 0.9, n_bars))  # lighter to darker

    ax.barh(plot_df["sub_group"], plot_df["importance_gain_norm"],
            color=colors, edgecolor=BAR_EDGE_COLOR, linewidth=BAR_EDGE_WIDTH)

    xmax = plot_df["importance_gain_norm"].max()
    ax.set_xlim(0, xmax * 1.16)
    for y, v in enumerate(plot_df["importance_gain_norm"]):
        ax.text(v + xmax * 0.02, y, f"{v:.1%}", va="center", ha="left", fontsize=12)

    ax.set_xlabel("Total normalized gain importance")
    ax.set_title(r"$\mathbf{C.}$  Top feature groups by normalized gain", fontsize=TITLE_SIZE, pad=8)
    ax.tick_params(axis="y", labelsize=13)
    clean_axis(ax)

# ---------- Main ----------
# def main():
#     df_imp_exp06d = load_importances(FI_PATH_EXP06D)
#     meta, n_cc_dims = load_feature_meta()

#     if not CC_VS_SS_SUMMARY_PATH.exists():
#         raise FileNotFoundError(f"CC vs SS summary not found: {CC_VS_SS_SUMMARY_PATH}")
#     cc_vs_ss_summary = pd.read_csv(CC_VS_SS_SUMMARY_PATH).copy()

#     # Simple figure
#     fig = plt.figure(figsize=(12, 9))

#     # GridSpec – clean and simple
#     gs = fig.add_gridspec(
#         2, 2,
#         height_ratios=[1, 0.9],
#         width_ratios=[1, 1],
#         hspace=0.3,
#         wspace=0.25
#     )

#     axA = fig.add_subplot(gs[0, 0])  # CC domain pie
#     axB = fig.add_subplot(gs[0, 1])  # CC vs SS pie
#     axC = fig.add_subplot(gs[1, :])  # Bar chart

#     # Plot
#     plot_panel_A_pie(axA, df_imp_exp06d, meta, n_cc_dims)
#     plot_panel_B_pie(axB, cc_vs_ss_summary)
#     plot_panel_C_bar(axC, df_imp_exp06d, meta, n_cc_dims, top_n=10)

#     # Make pies circular
#     axA.axis('equal')
#     axB.axis('equal')

#     # Adjust margins – this is the ONLY fix needed
#     fig.subplots_adjust(
#         left=0.18,    # left margin
#         right=0.88,   # right margin
#         top=0.92,     # top margin
#         bottom=0.10,  # bottom margin
#     )

#     fig.savefig(FIG4_PNG, dpi=600)
#     plt.close(fig)
#     print("Saved Fig 4 PNG to:", FIG4_PNG)



def main():
    df_imp_exp06d = load_importances(FI_PATH_EXP06D)
    meta, n_cc_dims = load_feature_meta()

    if not CC_VS_SS_SUMMARY_PATH.exists():
        raise FileNotFoundError(f"CC vs SS summary not found: {CC_VS_SS_SUMMARY_PATH}")
    cc_vs_ss_summary = pd.read_csv(CC_VS_SS_SUMMARY_PATH).copy()

    fig = plt.figure(figsize=(12, 9))

    gs = fig.add_gridspec(
        2, 2,
        height_ratios=[1, 0.9],
        width_ratios=[1, 1],
        hspace=0.3,
        wspace=0.25
    )

    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, :])

    plot_panel_A_pie(axA, df_imp_exp06d, meta, n_cc_dims)
    plot_panel_B_pie(axB, cc_vs_ss_summary)
    plot_panel_C_bar(axC, df_imp_exp06d, meta, n_cc_dims, top_n=10)

    axA.axis('equal')
    axB.axis('equal')

    # Symmetrical margins – NOW CENTRED
    fig.subplots_adjust(
        left=0.22,
        right=0.88,
        top=0.92,
        bottom=0.10,
    )

    fig.savefig(FIG4_PNG, dpi=600)
    plt.close(fig)
    print("Saved Fig 4 PNG to:", FIG4_PNG)


if __name__ == "__main__":
    main()

