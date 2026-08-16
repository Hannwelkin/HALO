"""
scripts/other_analysis/halo_m4_comparison.py

Extract per-fold metrics + compute p-values for model comparisons.

Three pieces:
  1. parse_log_for_folds()   -> pull per-fold metrics out of the M4 log file
  2. compare_models()        -> paired/unpaired test on fold-level metric arrays
  3. delong_roc_test()       -> statistically rigorous AUC comparison from raw predictions

"""

import re
import numpy as np
import pandas as pd
from scipy import stats

from halo.paths import MODEL_RESULTS

# ===== 1) Parse per-fold metrics out of a plain-text log file =====
def parse_log_for_folds(log_path, metric_names=("accuracy", "f1", "roc_auc", "auc")):
    """
    Generic line-based parser. Looks for lines that mention a fold number and
    a metric value, e.g.:
        "Fold 1 - accuracy: 0.71, f1: 0.68, roc_auc: 0.76"
        "[Fold 3] Accuracy=0.69 AUC=0.75"
    Adjust `fold_pat` / `metric_pat` below if your log format differs -
    paste me a few real lines and I'll tune the regex exactly.
    """
    fold_pat = re.compile(r"fold[\s_#:]*([0-9]+)", re.IGNORECASE)
    rows = []
    current_fold = None

    with open(log_path, "r") as f:
        for line in f:
            fold_match = fold_pat.search(line)
            if fold_match:
                current_fold = int(fold_match.group(1))

            if current_fold is None:
                continue

            found = {}
            for name in metric_names:
                # matches "accuracy: 0.71", "accuracy=0.71", "accuracy 0.71"
                m = re.search(rf"{name}[\s:=]+([0-9]*\.?[0-9]+)", line, re.IGNORECASE)
                if m:
                    found[name] = float(m.group(1))

            if found:
                found["fold"] = current_fold
                rows.append(found)

    if not rows:
        raise ValueError(
            "No fold/metric pairs found. Paste a few log lines back to me "
            "so I can adjust the regex to your exact log format."
        )

    df = pd.DataFrame(rows)
    # collapse multiple partial matches per fold into one row per fold
    df = df.groupby("fold").first().reset_index()
    return df


# ===== 2) Compare two models' fold-level metric arrays =====
def compare_models(scores_a, scores_b, paired=True, label_a="A", label_b="B", metric="accuracy"):
    """
    paired=True  -> use when both models share the same CV split scheme
                     (e.g. HALO vs M1/M2/M3, all pair-held-out nested CV)
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
        return {"test": "paired", "t_p": t_p, "wilcoxon_p": w_p}
    else:
        u_stat, u_p = stats.mannwhitneyu(scores_a, scores_b, alternative="two-sided")
        t_stat, t_p = stats.ttest_ind(scores_a, scores_b, equal_var=False)  # Welch's
        print(f"Mann-Whitney U:  U={u_stat:.3f}, p={u_p:.4f}")
        print(f"Welch's t-test:  t={t_stat:.3f}, p={t_p:.4f}")
        return {"test": "unpaired", "u_p": u_p, "welch_p": t_p}


# ===== 3. DeLong's test for comparing two ROC-AUCs directly from predictions =====
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
    """
    y_true: array of 0/1 labels (same for both models, same test instances)
    prob_a, prob_b: predicted probability of the positive class for model A and B
    Returns: auc_a, auc_b, p_value
    """
    y_true = np.asarray(y_true)
    order = np.argsort(-y_true)  # positives first
    y_sorted = y_true[order]
    label_1_count = int(y_sorted.sum())

    preds = np.vstack([np.asarray(prob_a)[order], np.asarray(prob_b)[order]])
    aucs, delongcov = _fast_delong(preds, label_1_count)

    diff = aucs[0] - aucs[1]
    var = delongcov[0, 0] + delongcov[1, 1] - 2 * delongcov[0, 1]
    z = diff / np.sqrt(var)
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return aucs[0], aucs[1], p




if __name__ == "__main__":
    # === 1) HALO per-fold metrics ===
    halo = pd.read_csv(MODEL_RESULTS / "exp06d_lgbm_bin_nosspace_elementwise_reduced_nestedcv" / "metrics_per_fold_cv1.csv")  
    print(halo.head())

    # === 2) Parse M4 log for per-fold metrics ===
    m4 = parse_log_for_folds(MODEL_RESULTS / "exp09d.log")   
    print(m4)

    # === 3) Paired comparison: HALO vs M3 (same CV scheme) ===
    compare_models(halo["accuracy"], m3["accuracy"], paired=True,
                   label_a="HALO", label_b="M3", metric="accuracy")

    # === 4) Unpaired comparison: HALO vs M4 (different CV scheme) ===
    compare_models(halo["accuracy"], m4["accuracy"], paired=False,
                   label_a="HALO", label_b="M4", metric="accuracy")

                   