"""
scripts/other_analysis/halo_model_comparisons.py

Reproducible statistical comparisons: HALO vs M1, M2, M3, M4.

Model / evaluation-scheme summary (see manuscript Table, Fig 2A):
  HALO - no S-space, fold-internal ("within CV folds") feature selection, cv1 scheme
  M1   - S-space, fold-internal feature selection, cv2 scheme (drug-pair + strain held out)
  M2   - S-space, fold-internal feature selection, cv1 scheme (same scheme as HALO)
  M3   - no S-space, pre-CV (global) feature selection, cv1 scheme (same scheme as HALO)
  M4   - no S-space, training-set-only feature selection, standard stratified split (no grouping)

Comparison logic:
  HALO vs M2 -> same cv1 scheme, same feature-selection philosophy -> paired test
                (exact paired if M2 raw per-fold values available; Welch's-from-summary
                approximation otherwise -- see NOTE in compare_halo_m2())
  HALO vs M3 -> same cv1 scheme, differs only in feature-selection timing -> paired test
                (this is the leakage comparison for Reviewer 2)
  HALO vs M4 -> different CV scheme entirely (no pair grouping) -> unpaired test
  HALO vs M1 -> different task (cv2 = pair+strain held out, a harder/different question)
                -> NOT a fair head-to-head; report M1's descriptive fold stats only,
                   no formal p-value against HALO (see manuscript framing of M1 as
                   answering a different generalization question)

"""

import re
import json
import numpy as np
import pandas as pd
from scipy import stats

from halo.paths import MODEL_RESULTS

EXP_DIRS = {
    "HALO": MODEL_RESULTS / "exp06d_lgbm_bin_nosspace_elementwise_reduced_nestedcv",
    "M1":   MODEL_RESULTS / "exp06c_lgbm_bin_sspace_elementwise_reduced_nestedcv_cv2",
    "M2":   MODEL_RESULTS / "exp06b_lgbm_bin_sspace_elementwise_reduced_nestedcv",
    "M3":   MODEL_RESULTS / "exp06e_lgbm_bin_nosspace_elementwise_preCV_reduced_nestedcv",
    "M4_LOG": MODEL_RESULTS / "exp09d.log",
}

METRICS = ["roc_auc_test", "accuracy_test", "f1_weighted_test", "f1_macro_test"]
METRIC_LABELS = {
    "roc_auc_test": "ROC-AUC",
    "accuracy_test": "Accuracy",
    "f1_weighted_test": "F1-weighted",
    "f1_macro_test": "F1-macro",
}


# ===== 1) Parse M4's single-split log (NOT k-fold -- one "Global Metrics" block) =====
def parse_m4_log(log_path):
    """
    M4 uses a single stratified train/test split (no CV, no pair grouping -- see
    manuscript Table/Fig 2A: cv_scheme = "standard"). The log therefore contains
    ONE block of results, not one row per fold:

        Global Metrics:
        accuracy_test      = 0.7633
        f1_macro_test      = 0.7631
        f1_weighted_test   = 0.7631
        ...
        AUC Scores:
        roc_auc_test        = 0.8514

    Returns a dict of {metric_name: value}, not a per-fold DataFrame.
    """
    metrics = {}
    pat = re.compile(r"^(roc_auc_test|accuracy_test|f1_macro_test|f1_weighted_test)\s*=\s*([0-9.]+)")
    with open(log_path, "r") as f:
        for line in f:
            m = pat.match(line.strip())
            if m:
                metrics[m.group(1)] = float(m.group(2))
    if not metrics:
        raise ValueError(
            "No metrics found in M4 log. Check that the log still has the "
            "'Global Metrics' / 'AUC Scores' block intact."
        )
    return metrics


def one_sample_test(fold_scores, single_value, label_a="HALO", label_b="M4", metric="metric"):
    """
    Use when comparing a model with repeated fold scores (fold_scores) against a
    model evaluated on a single split (single_value, e.g. M4). Treats fold_scores
    as a sample and asks whether single_value is consistent with that distribution.
    """
    fold_scores = np.asarray(fold_scores, dtype=float)
    t_stat, p = stats.ttest_1samp(fold_scores, single_value)
    print(f"\n--- {label_a} (n={len(fold_scores)} folds) vs {label_b} (single split) on {metric} ---")
    print(f"{label_a}: mean={fold_scores.mean():.4f}, sd={fold_scores.std(ddof=1):.4f}")
    print(f"{label_b}: {single_value:.4f}")
    print(f"One-sample t-test: t={t_stat:.3f}, p={p:.4f}")
    return {"test": "one_sample", "t_stat": t_stat, "p": p}


# ===== 2) Placeholder: parse M1's per-fold data out of cv2_info_fold*.json =====
def parse_m1_folds(m1_dir, n_folds=5):
    """
    TODO: structure of cv2_info_fold*.json not yet confirmed -- fill in once
    the JSON schema is known. Expected to return a DataFrame with one row per
    fold and columns matching METRICS.
    """
    raise NotImplementedError(
        "Paste the contents of one cv2_info_fold1.json so this parser can be written."
    )


# ===== 3) Compare two models' fold-level metric arrays =====
def compare_models(scores_a, scores_b, paired=True, label_a="A", label_b="B", metric="accuracy"):
    """
    paired=True  -> use when both models share the same CV split scheme
    paired=False -> use when CV schemes differ (e.g. HALO vs M4)
    """
    scores_a, scores_b = np.asarray(scores_a, dtype=float), np.asarray(scores_b, dtype=float)
    print(f"\n--- {label_a} vs {label_b} on {metric} ---")
    print(f"{label_a}: mean={scores_a.mean():.4f}, sd={scores_a.std(ddof=1):.4f}, n={len(scores_a)}")
    print(f"{label_b}: mean={scores_b.mean():.4f}, sd={scores_b.std(ddof=1):.4f}, n={len(scores_b)}")

    if paired:
        if len(scores_a) != len(scores_b):
            raise ValueError("Paired test requires equal-length fold arrays.")
        t_stat, t_p = stats.ttest_rel(scores_a, scores_b)
        try:
            w_stat, w_p = stats.wilcoxon(scores_a, scores_b)
        except ValueError:
            w_stat, w_p = np.nan, np.nan
        print(f"Paired t-test:        t={t_stat:.3f}, p={t_p:.4f}")
        print(f"Wilcoxon signed-rank: W={w_stat:.3f}, p={w_p:.4f}")
        return {"test": "paired", "t_stat": t_stat, "t_p": t_p, "wilcoxon_p": w_p}
    else:
        u_stat, u_p = stats.mannwhitneyu(scores_a, scores_b, alternative="two-sided")
        t_stat, t_p = stats.ttest_ind(scores_a, scores_b, equal_var=False)  # Welch's
        print(f"Mann-Whitney U:  U={u_stat:.3f}, p={u_p:.4f}")
        print(f"Welch's t-test:  t={t_stat:.3f}, p={t_p:.4f}")
        return {"test": "unpaired", "u_p": u_p, "welch_p": t_p}


# ===== 4) Approximate paired test from summary stats only (fallback for M2) =====
def welch_from_summary(mean1, std1, n1, mean2, std2, n2, label_a="A", label_b="B", metric="metric"):
    """
    NOTE: this is an approximation used ONLY because M2's raw per-fold values
    were not retained (cv_metrics_summary_cv1.csv has mean/std, not per-fold rows).
    If M2 is re-run and raw per-fold metrics are logged, replace this call with
    compare_models(..., paired=True) using the real fold values -- exact paired
    test is always preferable to this summary-stats approximation.
    """
    se = np.sqrt(std1**2 / n1 + std2**2 / n2)
    t = (mean1 - mean2) / se
    df = (std1**2/n1 + std2**2/n2)**2 / (
        (std1**2/n1)**2/(n1-1) + (std2**2/n2)**2/(n2-1)
    )
    p = 2 * (1 - stats.t.cdf(abs(t), df))
    print(f"\n--- {label_a} vs {label_b} on {metric} (APPROXIMATE: summary-stats Welch's t-test) ---")
    print(f"{label_a}: mean={mean1:.4f} (sd={std1:.4f})  {label_b}: mean={mean2:.4f} (sd={std2:.4f})")
    print(f"Welch t={t:.3f}, df={df:.2f}, p={p:.4f}")
    return {"test": "welch_from_summary", "t_stat": t, "df": df, "p": p}


# ===== 5) DeLong's test for comparing two ROC-AUCs directly from raw predictions =====
def _compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def _fast_delong(preds_sorted_transposed, label_1_count):
    m = label_1_count
    n = preds_sorted_transposed.shape[1] - m
    positive_examples = preds_sorted_transposed[:, :m]
    negative_examples = preds_sorted_transposed[:, m:]
    k = preds_sorted_transposed.shape[0]
    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)
    for r in range(k):
        tx[r, :] = _compute_midrank(positive_examples[r, :])
        ty[r, :] = _compute_midrank(negative_examples[r, :])
        tz[r, :] = _compute_midrank(preds_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def delong_roc_test(y_true, prob_a, prob_b):
    y_true = np.asarray(y_true)
    order = np.argsort(-y_true)
    y_sorted = y_true[order]
    label_1_count = int(y_sorted.sum())
    preds = np.vstack([np.asarray(prob_a)[order], np.asarray(prob_b)[order]])
    aucs, delongcov = _fast_delong(preds, label_1_count)
    diff = aucs[0] - aucs[1]
    var = delongcov[0, 0] + delongcov[1, 1] - 2 * delongcov[0, 1]
    z = diff / np.sqrt(var)
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return aucs[0], aucs[1], p


# ===== 6) Descriptive-only report for M1 (different task, no formal test vs HALO) =====
# NOTE: M1's metrics_per_fold.csv uses different column names than HALO/M2/M3
# (no "_test" suffix: "roc_auc", "accuracy", "f1_macro", "f1_weighted").
M1_METRIC_MAP = {
    "roc_auc_test": "roc_auc",
    "accuracy_test": "accuracy",
    "f1_weighted_test": "f1_weighted",
    "f1_macro_test": "f1_macro",
}


def report_m1_descriptive(m1_df):
    print("\n--- M1 descriptive fold stats (cv2: drug-pair + strain held out) ---")
    print("NOTE: M1 answers a different generalization question than HALO (unseen")
    print("strains as well as unseen pairs), so no formal p-value is computed against")
    print("HALO here -- fold compositions are not comparable. Reported descriptively only.")
    for metric in METRICS:
        col = M1_METRIC_MAP[metric]
        if col in m1_df.columns:
            vals = m1_df[col].astype(float)
            print(f"{METRIC_LABELS.get(metric, metric)}: mean={vals.mean():.4f}, sd={vals.std(ddof=1):.4f}, "
                  f"min={vals.min():.4f}, max={vals.max():.4f}, n={len(vals)}")


if __name__ == "__main__":
    # ---- HALO (reference model) ----
    halo = pd.read_csv(EXP_DIRS["HALO"] / "metrics_per_fold_cv1.csv")

    # ---- M3: paired test vs HALO (leakage comparison, Reviewer 2) ----
    m3 = pd.read_csv(EXP_DIRS["M3"] / "metrics_per_fold_cv1.csv")
    print("=" * 60)
    print("HALO vs M3 (paired -- same cv1 scheme, leakage comparison)")
    print("=" * 60)
    for metric in METRICS:
        compare_models(halo[metric], m3[metric], paired=True,
                        label_a="HALO", label_b="M3", metric=METRIC_LABELS[metric])

    # ---- M2: approximate paired test vs HALO (summary stats only, for now) ----
    # TODO: replace with exact paired test once M2 is re-run with per-fold logging.
    m2_summary = pd.read_csv(EXP_DIRS["M2"] / "cv_metrics_summary_cv1.csv")
    print("\n" + "=" * 60)
    print("HALO vs M2 (APPROXIMATE -- summary stats only, same cv1 scheme)")
    print("=" * 60)
    m2_metric_map = {"roc_auc_test": "roc_auc", "accuracy_test": "accuracy",
                      "f1_weighted_test": "f1_weighted", "f1_macro_test": "f1_macro"}
    for metric in METRICS:
        halo_vals = halo[metric].astype(float)
        row = m2_summary[m2_summary["metric"] == m2_metric_map[metric]].iloc[0]
        welch_from_summary(
            halo_vals.mean(), halo_vals.std(ddof=1), len(halo_vals),
            row["mean"], row["std"], 5,
            label_a="HALO", label_b="M2", metric=METRIC_LABELS[metric],
        )

    # ---- M4: one-sample test vs HALO (M4 is a SINGLE split, not k-fold) ----
    # NOTE: if M4 gets re-run as a true k-fold CV (not a single stratified split),
    # switch this block to load a metrics_per_fold.csv the same way M1/M3 do, and
    # use compare_models(..., paired=False) instead of one_sample_test -- the
    # CV scheme still differs from HALO's pair-held-out grouping, so it stays
    # an UNPAIRED test either way, just with 5 values on each side instead of 1.
    m4 = parse_m4_log(EXP_DIRS["M4_LOG"])
    print("\n" + "=" * 60)
    print("HALO vs M4 (one-sample -- M4 has no folds, single stratified split)")
    print("=" * 60)
    for metric in METRICS:
        if metric in m4:
            one_sample_test(halo[metric], m4[metric],
                             label_a="HALO", label_b="M4", metric=METRIC_LABELS[metric])

    # ---- M1: descriptive only, no formal test vs HALO (different task) ----
    # NOTE: cv2_info_fold*.json only describes fold CONSTRUCTION (held-out strains/
    # pairs, candidate split search scores) -- it has no performance metrics.
    # Use metrics_per_fold.csv (seen alongside the json files) for the real numbers.
    print("\n" + "=" * 60)
    print("M1 (cv2: drug-pair + strain held out)")
    print("=" * 60)
    m1 = pd.read_csv(EXP_DIRS["M1"] / "metrics_per_fold.csv")
    report_m1_descriptive(m1)

    # ---- Optional: DeLong's test wherever raw predictions are available ----
    # preds_halo = pd.read_csv(EXP_DIRS["HALO"] / "test_predictions_cv1.csv")
    # preds_m3   = pd.read_csv(EXP_DIRS["M3"] / "test_predictions_cv1.csv")
    # auc_halo, auc_m3, p = delong_roc_test(preds_halo["y_true"], preds_halo["y_prob"], preds_m3["y_prob"])
    # print(f"DeLong HALO vs M3: AUC {auc_halo:.3f} vs {auc_m3:.3f}, p={p:.4f}")