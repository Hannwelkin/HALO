#!/usr/bin/env python3
"""
Experiment: exp06d_rf_bin_nosspace_elementwise_reduced_nestedcv (HALO — RF baseline)

Purpose
This is a controlled algorithm-swap baseline for HALO, used to benchmark
against CoSynE and INDIGO (both Random Forest-based methods), while holding
everything else in HALO's pipeline constant: same dataset, same elementwise
CC features, same feature selection procedure, same nested CV protocol
(CV1: drug-pair held-out, 5 outer folds x 3 inner folds x 32 sampled configs).

Config
- model: RandomForestClassifier (sklearn)
- task: binary classification
- feature_design: elementwise similarity (identical to HALO/exp06d)
- sspace: disabled
- feature_selection: enabled (within CV folds) -- NOTE: feature selection
  itself is still performed via a small LightGBM importance-ranking model
  (select_features_lgbm), UNCHANGED from exp06d. Only the final classifier
  is swapped to Random Forest. This isolates the classifier as the single
  variable being compared, rather than conflating feature selection method
  and classifier choice.

- CV:
  - nested_cv: enabled
  - Outer split:
    - CV1 scheme: drug pair held-out
  - Inner split:
    - StratifiedGroupKFold, groups = Drug Pair
    - random search over 32 sampled hyperparameter configs
    - selection metric: mean validation accuracy across inner folds
  - Final fit: refit best model on full outer-train, evaluate once on outer-test

Data integrity note
All preprocessing (missing values, dtypes, column validation, etc.) is performed upstream in preprocessing
notebooks/scripts. This script assumes the processed inputs are clean and consistent.

Class encoding
The binary target is encoded as {0,1} where 1 corresponds to synergy and is treated as the
positive class for F1 and ROC-AUC.
"""

import json
import itertools
import numpy as np
import pandas as pd
import lightgbm as lgb  # still used only inside select_features_lgbm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    GroupKFold,
    StratifiedGroupKFold,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_fscore_support,
)

from halo.paths import CC_FEATURES, PROCESSED, MODEL_RESULTS
from halo.mappers.feature_mapper import FeatureMapper
from halo.shared_utils.data_io import classify_interaction


# helper function -- UNCHANGED from exp06d (feature selection stays LightGBM-based
# on purpose; see Config note above)
def select_features_lgbm(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    feat_cols: list[str],
    corr_min: float = 0.01,
    keep_top_frac: float = 0.30,
) -> list[str]:
    """
    Feature selection performed using training data only.

    Steps
    1) drop zero-variance features
    2) correlation prefilter using |corr(feature, y)| >= corr_min
       fallback: keep all variance-filtered features if none pass
    3) LightGBM importance ranking and keep top fraction
    """
    var_series = X_train.var()
    kept_after_var = [c for c in feat_cols if var_series[c] > 0.0]

    if len(kept_after_var) == 0:
        raise ValueError("No features remained after variance filtering.")

    kept_after_corr = []
    y_train_s = pd.Series(y_train, index=X_train.index)

    for col in kept_after_var:
        corr = X_train[col].corr(y_train_s)
        if corr is not None and np.isfinite(corr) and abs(corr) >= corr_min:
            kept_after_corr.append(col)

    if not kept_after_corr:
        kept_after_corr = kept_after_var.copy()

    fs_model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=2000,
        random_state=777,
        n_jobs=1,
        learning_rate=0.03,
        max_depth=3,
        num_leaves=15,
        min_data_in_leaf=200,
        feature_fraction=0.4,
        bagging_fraction=0.8,
        bagging_freq=1,
        lambda_l2=50.0,
        lambda_l1=0.0,
        max_bin=127,
        min_gain_to_split=0.05,
    )

    fs_model.fit(X_train[kept_after_corr], y_train)

    feat_imp = pd.Series(
        fs_model.feature_importances_,
        index=kept_after_corr
    ).sort_values(ascending=False)

    n_keep = max(1, int(len(feat_imp) * keep_top_frac))
    selected_features = feat_imp.index[:n_keep].tolist()
    return selected_features


def main():
    # ==========================
    # 0) Basic config
    # ==========================
    SCHEME = "CV1"
    corr_min = 0.01
    keep_top_frac = 0.30

    cc_path = CC_FEATURES / "cc_features_concat_25x128.csv"
    combos_path = PROCESSED / "halo_training_dataset.csv"

    out_dir = MODEL_RESULTS / "exp06d_rf_bin_nosspace_elementwise_reduced_nestedcv"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        "\n=== EXP06d — Random Forest baseline (for CoSynE/INDIGO comparison) ===\n"
    )
    print("Output dir :", out_dir)

    # ==========================
    # 1) Load full CC-only dataset and rebuild elementwise features
    # ==========================
    cc_df = pd.read_csv(cc_path).copy()
    combinations_df = pd.read_csv(combos_path).copy()

    features_cc = cc_df.copy()
    df = FeatureMapper().elementwise_similarity(combinations_df, features_cc)
    print("Full df shape:", df.shape)

    df = df[df["Interaction Type"].isin(["synergy", "antagonism"])].copy()
    print("\nAfter filtering to synergy/antagonism:", df.shape)
    print(df["Interaction Type"].value_counts())

    # ==========================
    # 2) Feature columns
    # ==========================
    drop_cols = [
        "Drug A",
        "Drug B",
        "Drug A Inchikey",
        "Drug B Inchikey",
        "Strain",
        "Specie",
        "Bliss Score",
        "Interaction Type",
        "Source",
        "Drug Pair",
    ]
    feat_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feat_cols].copy()

    y = df["Interaction Type"].copy()
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    pairs = df["Drug Pair"].astype(str).values
    strains = df["Strain"].astype(str).values
    n = len(df)

    rng = np.random.default_rng(42)

    inv_label_map = {
        int(code): cls for cls, code in zip(le.classes_, le.transform(le.classes_))
    }

    # ==========================
    # 3) Outer splits (CV1) -- UNCHANGED
    # ==========================
    def make_splits_cv1(n_splits=5, verbose=True):
        try:
            outer_cv = StratifiedGroupKFold(
                n_splits=n_splits, shuffle=True, random_state=42
            )
            split_gen = outer_cv.split(X, y_enc, groups=pairs)
        except TypeError:
            outer_cv = GroupKFold(n_splits=n_splits)
            split_gen = outer_cv.split(X, y_enc, groups=pairs)

        splits = []
        for fold_idx, (tr_idx, te_idx) in enumerate(split_gen, 1):
            splits.append((tr_idx, te_idx))
        return splits

    outer_splits = make_splits_cv1(n_splits=5, verbose=True)

    # ==========================
    # 4) Inner CV (nested) + outer loop
    # ==========================
    class SilentLogger:
        def info(self, msg):
            pass

        def warning(self, msg):
            pass

    lgb.register_logger(SilentLogger())

    fold_results = []
    cm_total = None
    all_test_dfs = []
    all_train_dfs = []
    per_fold_importances = []
    per_fold_best = []

    synergy_code = le.transform(["synergy"])[0]
    ant_code = le.transform(["antagonism"])[0]

    def sample_one_params():
        """
        Random Forest hyperparameter search space, chosen to mirror the
        approximate compute budget of exp06d's LightGBM search (32 sampled
        configs) rather than to exactly replicate CoSynE/INDIGO's own
        (unreported) tuning procedures.
        """
        # NOTE: rng.choice() on a mixed-type Python list silently upcasts
        # everything to one dtype (e.g. turns 0.3 into the string "0.3").
        # Sampling an index and indexing into the plain Python list avoids
        # that coercion and preserves the actual types sklearn expects.
        max_depth_opts = [None, 8, 12, 20]
        max_features_opts = ["sqrt", "log2", 0.3, 0.5]
        class_weight_opts = [None, "balanced"]

        return dict(
            n_estimators=int(rng.choice([200, 400, 600, 800])),
            max_depth=max_depth_opts[rng.integers(len(max_depth_opts))],
            min_samples_leaf=int(rng.choice([1, 2, 5, 10])),
            min_samples_split=int(rng.choice([2, 5, 10])),
            max_features=max_features_opts[rng.integers(len(max_features_opts))],
            class_weight=class_weight_opts[rng.integers(len(class_weight_opts))],
        )
    
    for fold_idx, (tr_idx, te_idx) in enumerate(outer_splits, 1):
        print("\n" + "#" * 72)
        print(f"########## OUTER FOLD {fold_idx}/{len(outer_splits)} ##########")
        print("#" * 72 + "\n")

        X_tr = X.iloc[tr_idx].reset_index(drop=True)
        X_te = X.iloc[te_idx].reset_index(drop=True)
        y_tr = y_enc[tr_idx]
        y_te = y_enc[te_idx]
        grp_tr = pairs[tr_idx]

        df_tr = df.iloc[tr_idx].reset_index(drop=True)
        df_te = df.iloc[te_idx].reset_index(drop=True)

        param_samples = [sample_one_params() for _ in range(32)]

        try:
            inner_cv = StratifiedGroupKFold(
                n_splits=3, shuffle=True, random_state=111
            )

            def inner_splitter():
                return inner_cv.split(X_tr, y_tr, groups=grp_tr)

        except Exception:
            inner_cv = GroupKFold(n_splits=3)

            def inner_splitter():
                return inner_cv.split(X_tr, y_tr, groups=grp_tr)

        def cv_acc_for_params(params):
            """
            Same inner-CV structure as exp06d: feature selection fit on
            inner-train only, RF trained with candidate params, evaluated
            on inner-validation accuracy. No early stopping (RF doesn't
            support it) -- n_estimators is treated as a tunable hyperparameter
            instead.
            """
            accs = []
            for tr_f, val_f in inner_splitter():
                Xf_tr = X_tr.iloc[tr_f].reset_index(drop=True)
                Xf_val = X_tr.iloc[val_f].reset_index(drop=True)
                yf_tr = y_tr[tr_f]
                yf_val = y_tr[val_f]

                selected_inner = select_features_lgbm(
                    X_train=Xf_tr,
                    y_train=yf_tr,
                    feat_cols=feat_cols,
                    corr_min=corr_min,
                    keep_top_frac=keep_top_frac,
                )

                Xf_tr_sel = Xf_tr[selected_inner]
                Xf_val_sel = Xf_val[selected_inner]

                m = RandomForestClassifier(
                    random_state=777,
                    n_jobs=4,
                    **params,
                )
                m.fit(Xf_tr_sel, yf_tr)

                y_pred_fold = m.predict(Xf_val_sel)
                accs.append(accuracy_score(yf_val, y_pred_fold))
            return float(np.mean(accs))

        print(
            "\n--- Nested CV (outer fold "
            f"{fold_idx}): inner search over {len(param_samples)} configs ---"
        )
        scores = [(cv_acc_for_params(ps), ps) for ps in param_samples]
        scores.sort(reverse=True, key=lambda t: t[0])
        best_acc, best_params = scores[0]
        print(
            f"Best inner-CV ACC (fold {fold_idx}): {best_acc:.3f}\n"
            f"Best params: {best_params}"
        )

        per_fold_best.append(
            {
                "fold": fold_idx,
                "best_acc_inner_cv": float(best_acc),
                "best_params": best_params,
            }
        )

        # ==========================
        # 5) Final refit on outer-train with outer-train-only feature selection
        # ==========================
        selected_outer = select_features_lgbm(
            X_train=X_tr,
            y_train=y_tr,
            feat_cols=feat_cols,
            corr_min=corr_min,
            keep_top_frac=keep_top_frac,
        )

        X_tr_sel = X_tr[selected_outer]
        X_te_sel = X_te[selected_outer]

        m_final = RandomForestClassifier(
            random_state=777,
            n_jobs=4,
            **best_params,
        )
        m_final.fit(X_tr_sel, y_tr)

        pos_idx = np.flatnonzero(m_final.classes_ == synergy_code)[0]

        # ==========================
        # 5b) Save feature importances (per fold)
        #     RF gives a single impurity-based (MDI) importance -- no
        #     separate split/gain distinction like LightGBM's booster.
        # ==========================
        imp_mdi = m_final.feature_importances_

        feat_importance_df = pd.DataFrame(
            {
                "feature": selected_outer,
                "importance_mdi": imp_mdi,
            }
        )

        total_imp_fold = feat_importance_df["importance_mdi"].sum()
        if total_imp_fold > 0:
            feat_importance_df["importance_mdi_norm"] = (
                feat_importance_df["importance_mdi"] / total_imp_fold
            )
        else:
            feat_importance_df["importance_mdi_norm"] = 0.0

        fi_fold_path = out_dir / f"feature_importances_{SCHEME.lower()}_fold{fold_idx}.csv"
        feat_importance_df.to_csv(fi_fold_path, index=False)
        print("Saved per-fold feature importances to:", fi_fold_path)

        feat_importance_df["fold"] = fold_idx
        per_fold_importances.append(feat_importance_df)

        # ==========================
        # 6) Final evaluation on held-out outer test
        # ==========================
        p_te = m_final.predict_proba(X_te_sel)[:, pos_idx]
        y_pred = (p_te >= 0.5).astype(int)

        y_te_bin = (y_te == synergy_code).astype(int)

        accuracy_test = accuracy_score(y_te, y_pred)
        f1_macro_test = f1_score(y_te, y_pred, average="macro")
        f1_weighted_test = f1_score(y_te, y_pred, average="weighted")
        roc_auc_test = roc_auc_score(y_te_bin, p_te)

        print(f"\n=== Held-out Test (fold {fold_idx}) ===")
        print(f"accuracy_test={accuracy_test:.4f}")
        print(f"roc_auc_test={roc_auc_test:.4f}")
        print(f"f1_macro_test={f1_macro_test:.4f}")
        print(f"f1_weighted_test={f1_weighted_test:.4f}")

        print("\nConfusion matrix:\n", confusion_matrix(y_te, y_pred))
        print(
            "\nReport:\n",
            classification_report(y_te, y_pred, target_names=le.classes_),
        )

        prec, rec, f1s, _ = precision_recall_fscore_support(
            y_te,
            y_pred,
            labels=[ant_code, synergy_code],
        )
        precision_antag, precision_syn = prec
        recall_antag, recall_syn = rec
        f1_antag, f1_syn = f1s

        # ==========================
        # 7) Overfitting check
        # ==========================
        p_tr = m_final.predict_proba(X_tr_sel)[:, pos_idx]
        y_tr_pred = (p_tr >= 0.5).astype(int)
        y_tr_bin = (y_tr == synergy_code).astype(int)

        accuracy_train = accuracy_score(y_tr, y_tr_pred)
        f1_macro_train = f1_score(y_tr, y_tr_pred, average="macro")
        f1_weighted_train = f1_score(y_tr, y_tr_pred, average="weighted")
        roc_auc_train = roc_auc_score(y_tr_bin, p_tr)

        print("\n=== Overfitting check (CC-only, outer fold", fold_idx, ") ===")
        print(
            "Train AUC:", round(roc_auc_train, 3),
            "| Test AUC:", round(roc_auc_test, 3),
        )
        print(
            "Train Acc:", round(accuracy_train, 3),
            "| Test Acc:", round(accuracy_test, 3),
        )
        print(
            "Train F1w:", round(f1_weighted_train, 3),
            "| Test F1w:", round(f1_weighted_test, 3),
        )

        # ==========================
        # 7b) Save train/test-level info for this fold
        # ==========================
        test_out_fold = pd.DataFrame(
            {
                "fold": fold_idx,
                "index": df_te.index,
                "Drug_Pair": df_te["Drug Pair"].astype(str),
                "Strain": df_te["Strain"].astype(str),
                "y_true_int": y_te,
                "y_true_label": [inv_label_map[int(v)] for v in y_te],
                "y_pred_int": y_pred,
                "y_pred_label": [inv_label_map[int(v)] for v in y_pred],
                "p_synergy": p_te,
            }
        )
        all_test_dfs.append(test_out_fold)

        train_out_fold = pd.DataFrame(
            {
                "fold": fold_idx,
                "index": df_tr.index,
                "Drug_Pair": df_tr["Drug Pair"].astype(str),
                "Strain": df_tr["Strain"].astype(str),
                "y_true_int": y_tr,
                "y_true_label": [inv_label_map[int(v)] for v in y_tr],
                "y_pred_int": y_tr_pred,
                "y_pred_label": [inv_label_map[int(v)] for v in y_tr_pred],
                "p_synergy": p_tr,
            }
        )
        all_train_dfs.append(train_out_fold)

        order = ["antagonism", "synergy"]
        order_idx = le.transform(order)
        cm = confusion_matrix(y_te, y_pred, labels=order_idx)

        if cm_total is None:
            cm_total = cm.copy()
        else:
            cm_total += cm

        fig, ax = plt.subplots(figsize=(6, 5))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=order)
        disp.plot(cmap="Blues", ax=ax, values_format="d")
        ax.set_title(
            f"Confusion Matrix — RF baseline, True vs Predicted "
            f"(CC-only reduced features, fold {fold_idx})",
            fontsize=13,
            pad=15,
        )
        ax.set_xlabel("Predicted Label", fontsize=11)
        ax.set_ylabel("True Label", fontsize=11)
        plt.tight_layout()
        fig_path = out_dir / f"confusion_matrix_{SCHEME.lower()}_fold{fold_idx}.png"
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        print("\nSaved per-fold confusion matrix plot to:", fig_path)

        fold_results.append(
            dict(
                fold=fold_idx,
                roc_auc_test=roc_auc_test,
                accuracy_test=accuracy_test,
                f1_weighted_test=f1_weighted_test,
                f1_macro_test=f1_macro_test,
                precision_antag=precision_antag,
                recall_antag=recall_antag,
                f1_antag=f1_antag,
                precision_syn=precision_syn,
                recall_syn=recall_syn,
                f1_syn=f1_syn,
                roc_auc_train=roc_auc_train,
                accuracy_train=accuracy_train,
                f1_weighted_train=f1_weighted_train,
                f1_macro_train=f1_macro_train,
                n_train=len(tr_idx),
                n_test=len(te_idx),
            )
        )

    # ==========================
    # 8) Aggregate across folds
    # ==========================
    best_params_path = out_dir / f"best_params_{SCHEME.lower()}.json"
    best_overall = max(per_fold_best, key=lambda d: d["best_acc_inner_cv"])
    with open(best_params_path, "w") as f:
        json.dump(
            {
                "scheme": SCHEME,
                "model": "RandomForestClassifier",
                "n_folds": len(per_fold_best),
                "best_acc_inner_cv": float(best_overall["best_acc_inner_cv"]),
                "best_params": best_overall["best_params"],
                "per_fold": per_fold_best,
            },
            f,
            indent=2,
            default=str,  # handles numpy scalar types (e.g. max_depth=None, np.str_ choices)
        )
    print("Saved best params (incl. per-fold) to:", best_params_path)

    if per_fold_importances:
        fi_all = pd.concat(per_fold_importances, ignore_index=True)

        all_selected_features = sorted(fi_all["feature"].unique().tolist())
        rows = []
        for feat in all_selected_features:
            sub = fi_all[fi_all["feature"] == feat]
            vals_mdi = sub["importance_mdi"].values
            vals_mdi_norm = sub["importance_mdi_norm"].values

            rows.append(
                {
                    "feature": feat,
                    "importance_mdi": float(np.mean(vals_mdi)),
                    "importance_mdi_norm": float(np.mean(vals_mdi_norm)),
                    "selected_in_n_folds": int(sub["fold"].nunique()),
                }
            )

        fi_mean = (
            pd.DataFrame(rows)
            .sort_values("importance_mdi", ascending=False)
            .reset_index(drop=True)
        )

        total_imp = fi_mean["importance_mdi"].sum()
        if total_imp > 0:
            fi_mean["importance_mdi_norm"] = fi_mean["importance_mdi"] / total_imp

        fi_path = out_dir / f"feature_importances_{SCHEME.lower()}.csv"
        fi_mean.to_csv(fi_path, index=False)
        print("Saved averaged feature importances to:", fi_path)

    test_out_all = pd.concat(all_test_dfs, ignore_index=True)
    train_out_all = pd.concat(all_train_dfs, ignore_index=True)

    test_out_path = out_dir / f"test_predictions_{SCHEME.lower()}.csv"
    train_out_path = out_dir / f"train_predictions_{SCHEME.lower()}.csv"

    test_out_all.to_csv(test_out_path, index=False)
    train_out_all.to_csv(train_out_path, index=False)

    print("Saved aggregated test predictions to:", test_out_path)
    print("Saved aggregated train predictions to:", train_out_path)

    metrics_per_fold_df = pd.DataFrame(fold_results)
    metrics_per_fold_path = out_dir / f"metrics_per_fold_{SCHEME.lower()}.csv"
    metrics_per_fold_df.to_csv(metrics_per_fold_path, index=False)
    print("Saved per-fold metrics to:", metrics_per_fold_path)

    def mean_std(key):
        vals = [fr[key] for fr in fold_results]
        return float(np.mean(vals)), float(np.std(vals))

    acc_mean, acc_std = mean_std("accuracy_test")
    f1w_mean, f1w_std = mean_std("f1_weighted_test")
    f1m_mean, f1m_std = mean_std("f1_macro_test")
    auc_mean, auc_std = mean_std("roc_auc_test")

    print("\n" + "=" * 72)
    print(f"=== Summary over {len(fold_results)} CV1 outer folds (RF baseline) ===")
    print(f"roc_auc_test_mean={auc_mean:.4f}  roc_auc_test_std={auc_std:.4f}")
    print(f"accuracy_test_mean={acc_mean:.4f}  accuracy_test_std={acc_std:.4f}")
    print(f"f1_weighted_test_mean={f1w_mean:.4f}  f1_weighted_test_std={f1w_std:.4f}")
    print(f"f1_macro_test_mean={f1m_mean:.4f}  f1_macro_test_std={f1m_std:.4f}")
    print("=" * 72)

    prec_a_mean, prec_a_std = mean_std("precision_antag")
    rec_a_mean, rec_a_std = mean_std("recall_antag")
    f1_a_mean, f1_a_std = mean_std("f1_antag")

    prec_s_mean, prec_s_std = mean_std("precision_syn")
    rec_s_mean, rec_s_std = mean_std("recall_syn")
    f1_s_mean, f1_s_std = mean_std("f1_syn")

    print("Per-class metrics over outer folds:")
    print(
        f"ANTAGONISM: "
        f"precision_mean={prec_a_mean:.4f} (±{prec_a_std:.4f}), "
        f"recall_mean={rec_a_mean:.4f} (±{rec_a_std:.4f}), "
        f"f1_mean={f1_a_mean:.4f} (±{f1_a_std:.4f})"
    )
    print(
        f"SYNERGY   : "
        f"precision_mean={prec_s_mean:.4f} (±{prec_s_std:.4f}), "
        f"recall_mean={rec_s_mean:.4f} (±{rec_s_std:.4f}), "
        f"f1_mean={f1_s_mean:.4f} (±{f1_s_std:.4f})"
    )
    print("=" * 72)

    metrics_test = {
        "scheme": SCHEME,
        "model": "RandomForestClassifier",
        "n_folds": len(fold_results),
        "accuracy_test": acc_mean,
        "accuracy_test_std": acc_std,
        "f1_macro_test": f1m_mean,
        "f1_macro_test_std": f1m_std,
        "f1_weighted_test": f1w_mean,
        "f1_weighted_test_std": f1w_std,
        "roc_auc_test": auc_mean,
        "roc_auc_test_std": auc_std,
        "n_test": int(sum(fr["n_test"] for fr in fold_results)),
    }
    metrics_test_df = pd.DataFrame([metrics_test])
    metrics_test_path = out_dir / f"metrics_test_{SCHEME.lower()}.csv"
    metrics_test_df.to_csv(metrics_test_path, index=False)
    print("Saved aggregated test metrics to:", metrics_test_path)

    auc_tr_mean, auc_tr_std = mean_std("roc_auc_train")
    acc_tr_mean, acc_tr_std = mean_std("accuracy_train")
    f1w_tr_mean, f1w_tr_std = mean_std("f1_weighted_train")
    f1m_tr_mean, f1m_tr_std = mean_std("f1_macro_train")

    metrics_train = {
        "scheme": SCHEME,
        "model": "RandomForestClassifier",
        "n_folds": len(fold_results),
        "accuracy_train": acc_tr_mean,
        "accuracy_train_std": acc_tr_std,
        "f1_macro_train": f1m_tr_mean,
        "f1_macro_train_std": f1m_tr_std,
        "f1_weighted_train": f1w_tr_mean,
        "f1_weighted_train_std": f1w_tr_std,
        "roc_auc_train": auc_tr_mean,
        "roc_auc_train_std": auc_tr_std,
        "n_train": int(sum(fr["n_train"] for fr in fold_results)),
    }
    metrics_train_df = pd.DataFrame([metrics_train])
    metrics_train_path = out_dir / f"metrics_train_{SCHEME.lower()}.csv"
    metrics_train_df.to_csv(metrics_train_path, index=False)
    print("Saved aggregated train metrics to:", metrics_train_path)

    order = ["antagonism", "synergy"]
    cm_all = cm_total if cm_total is not None else np.zeros((2, 2), dtype=int)

    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_all, display_labels=order)
    disp.plot(cmap="Blues", ax=ax, values_format="d")

    ax.set_title(
        "Confusion Matrix — RF baseline, aggregated over outer folds "
        "(CC-only reduced features, CV1)",
        fontsize=13,
        pad=15,
    )
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)

    plt.tight_layout()
    fig_path = out_dir / f"confusion_matrix_{SCHEME.lower()}.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    print("\nSaved aggregated confusion matrix plot to:", fig_path)
    print("\n=== EXP06d RF baseline DONE ===\n")


if __name__ == "__main__":
    main()