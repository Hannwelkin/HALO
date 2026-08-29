"""
scripts/other_analysis/quantify_feature_selection_leakage.py

Uses the per-fold feature_importances_cv1_fold{1-5}.csv files already saved
for both exp06d (HALO) and exp06e (M3) -- any feature appearing in one of
these files was selected for that fold, so the row set IS the selected
feature set for that fold.
"""

import numpy as np
import pandas as pd
from itertools import combinations
from scipy import stats

from halo.paths import MODEL_RESULTS

HALO_DIR = MODEL_RESULTS / "exp06d_lgbm_bin_nosspace_elementwise_reduced_nestedcv"
M3_DIR = MODEL_RESULTS / "exp06e_lgbm_bin_nosspace_elementwise_preCV_reduced_nestedcv"

N_FOLDS = 5


def load_fold_feature_sets(model_dir, label):
    """
    Load feature_importances_cv1_fold{1-5}.csv and return a dict of
    {fold_number: set(feature_names)}.
    """
    fold_sets = {}
    for fold in range(1, N_FOLDS + 1):
        path = model_dir / f"feature_importances_cv1_fold{fold}.csv"
        if not path.exists():
            raise FileNotFoundError(f"[{label}] Missing expected file: {path}")
        df = pd.read_csv(path)

        # Try to auto-detect the feature-name column
        candidate_cols = [c for c in df.columns if "feature" in c.lower()]
        if not candidate_cols:
            raise ValueError(
                f"[{label}] Could not find a feature-name column in {path}. "
                f"Available columns: {list(df.columns)}. "
                f"Update `candidate_cols` logic to match."
            )
        feat_col = candidate_cols[0]
        fold_sets[fold] = set(df[feat_col].astype(str))

    return fold_sets


def jaccard(set_a, set_b):
    if not set_a and not set_b:
        return 1.0
    return len(set_a & set_b) / len(set_a | set_b)


def report_counts(fold_sets, label):
    counts = [len(fold_sets[f]) for f in sorted(fold_sets)]
    print(f"\n--- {label}: selected feature counts per fold ---")
    for f in sorted(fold_sets):
        print(f"  Fold {f}: {len(fold_sets[f])} features")
    print(f"  Mean: {np.mean(counts):.1f}, SD: {np.std(counts, ddof=1):.1f}, "
          f"Min: {min(counts)}, Max: {max(counts)}")
    return counts


def report_within_model_overlap(fold_sets, label):
    """Pairwise Jaccard overlap across all fold pairs within one model."""
    pairs = list(combinations(sorted(fold_sets), 2))
    jaccards = []
    print(f"\n--- {label}: fold-to-fold overlap (Jaccard index) ---")
    for f1, f2 in pairs:
        j = jaccard(fold_sets[f1], fold_sets[f2])
        jaccards.append(j)
        print(f"  Fold {f1} vs Fold {f2}: Jaccard = {j:.3f}")
    print(f"  Mean Jaccard across all fold pairs: {np.mean(jaccards):.3f} "
          f"(SD: {np.std(jaccards, ddof=1):.3f})")
    return jaccards


def report_cross_model_overlap(halo_sets, m3_sets):
    """
    For each HALO fold, how much does it overlap with M3's (effectively
    global, reused) feature set? Also checks whether M3's fold sets are
    indeed identical/near-identical across folds, confirming global reuse.
    """
    print("\n--- M3 fold-to-fold identity check (expected: ~identical, since global reuse) ---")
    m3_pairs = list(combinations(sorted(m3_sets), 2))
    for f1, f2 in m3_pairs:
        j = jaccard(m3_sets[f1], m3_sets[f2])
        identical = m3_sets[f1] == m3_sets[f2]
        print(f"  M3 Fold {f1} vs Fold {f2}: Jaccard = {j:.3f}, identical = {identical}")

    print("\n--- HALO fold vs. M3 (using M3 fold 1 as representative global set) ---")
    m3_global = m3_sets[1]  # representative, should be ~same across all M3 folds
    overlap_fracs = []
    for f in sorted(halo_sets):
        halo_set = halo_sets[f]
        overlap = halo_set & m3_global
        frac_of_m3_in_halo = len(overlap) / len(m3_global) if m3_global else 0.0
        frac_of_halo_in_m3 = len(overlap) / len(halo_set) if halo_set else 0.0
        overlap_fracs.append(frac_of_m3_in_halo)
        print(f"  HALO Fold {f}: {len(halo_set)} features, "
              f"{len(overlap)} shared with M3's global set "
              f"({frac_of_m3_in_halo:.1%} of M3's set; {frac_of_halo_in_m3:.1%} of HALO's set)")

    m3_only = m3_global - set().union(*halo_sets.values())
    print(f"\n  Features in M3's global set NEVER selected in ANY HALO fold: {len(m3_only)} "
          f"({len(m3_only)/len(m3_global):.1%} of M3's set)")
    print("  These are candidate 'leakage-driven' features -- present in M3's global")
    print("  selection but never independently selected under fold-internal selection.")

    return overlap_fracs, m3_only


def compare_counts_statistically(halo_counts, m3_counts):
    print("\n--- Statistical comparison: selected feature count per fold ---")
    print(f"HALO: mean={np.mean(halo_counts):.1f}, sd={np.std(halo_counts, ddof=1):.1f}")
    print(f"M3:   mean={np.mean(m3_counts):.1f}, sd={np.std(m3_counts, ddof=1):.1f}")
    # Paired since both use the same 5 outer fold splits
    t_stat, p = stats.ttest_rel(halo_counts, m3_counts)
    print(f"Paired t-test on feature count (HALO vs M3): t={t_stat:.3f}, p={p:.4f}")


if __name__ == "__main__":
    halo_sets = load_fold_feature_sets(HALO_DIR, "HALO")
    m3_sets = load_fold_feature_sets(M3_DIR, "M3")

    halo_counts = report_counts(halo_sets, "HALO (fold-internal selection)")
    m3_counts = report_counts(m3_sets, "M3 (global/pre-CV selection)")

    report_within_model_overlap(halo_sets, "HALO")
    report_within_model_overlap(m3_sets, "M3")

    report_cross_model_overlap(halo_sets, m3_sets)

    compare_counts_statistically(halo_counts, m3_counts)

