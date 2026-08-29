"""
scripts/models/external_validation/halo_rf_indigo_external_validation_on_acdb.py

Indigo Random Forest baseline's external validation on the rest of the ACDB dataset
(combinations that are not Bliss method and were not in the training or test set).

Mirrors halo_external_validation_on_acdb.py exactly, except:
- final classifier is RandomForestClassifier instead of LightGBM
- best_params are loaded from the RF nested-CV run (exp06d_rf_...)
- feature selection (select_features_lgbm) is UNCHANGED -- still LightGBM-based,
  to keep this a controlled classifier-only comparison against the original
  HALO external validation.
"""

import json
import numpy as np
import pandas as pd
import lightgbm as lgb  # still used only inside select_features_lgbm
import matplotlib
matplotlib.use("Agg")

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    classification_report,
)

from halo.paths import INTERIM, PROCESSED, RESULTS, MODEL_RESULTS, CC_FEATURES
from halo.mappers.feature_mapper import FeatureMapper


SCHEME = "CV1"
corr_min = 0.01
keep_top_frac = 0.30

# NOTE: point at the RF run's output dir, not the LightGBM one
exp06d_rf_indigo_out = MODEL_RESULTS / "exp06d_rf_bin_indigo_nosspace_elementwise_reduced_nestedcv"
ext_out = RESULTS / "external_validation" / "external_validation_acdb_rf_indigo"
ext_out.mkdir(parents=True, exist_ok=True)

best_params_path = exp06d_rf_indigo_out / "best_params_cv1.json"

external_base_path = INTERIM / "source_c_acdb" / "acdb_cleaned_data_validation.csv"

cc_path = CC_FEATURES / "cc_features_d3_only.csv"
combos_path = PROCESSED / "halo_training_dataset.csv"


def select_features_lgbm(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    feat_cols: list[str],
    corr_min: float = 0.01,
    keep_top_frac: float = 0.30,
) -> list[str]:
    """
    Feature selection using training data only. UNCHANGED from the original
    HALO (LightGBM) pipeline -- kept identical so this benchmark isolates
    the classifier, not the feature selection method.
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
        index=kept_after_corr,
    ).sort_values(ascending=False)

    n_keep = max(1, int(len(feat_imp) * keep_top_frac))
    return feat_imp.index[:n_keep].tolist()


# ==========================
# 1) Load internal training data and rebuild full CC-only features
# ==========================
if not cc_path.exists():
    raise FileNotFoundError(f"CC features file not found at: {cc_path}")
if not combos_path.exists():
    raise FileNotFoundError(f"Training dataset not found at: {combos_path}")

cc_df = pd.read_csv(cc_path).copy()
combos_df = pd.read_csv(combos_path).copy()

fm = FeatureMapper()
df = fm.elementwise_similarity(combos_df, cc_df)

print("Full df shape:", df.shape)

df = df[df["Interaction Type"].isin(["synergy", "antagonism"])].copy()

print("\nAfter filtering to synergy/antagonism:", df.shape)
print(df["Interaction Type"].value_counts())

drop_cols = [
    "Drug A",
    "Drug B",
    "Drug A Inchikey",
    "Drug B Inchikey",
    "Strain",
    "Specie",
    "Score",
    "Bliss Score",
    "Method",
    "Interaction Type",
    "Drug Pair",
    "PMID",
    "Source",
]
feat_cols = [c for c in df.columns if c not in drop_cols]

X = df[feat_cols].copy()
y_text = df["Interaction Type"].copy()

le = LabelEncoder()
y_enc = le.fit_transform(y_text)

pairs = df["Drug Pair"].astype(str).values
n = len(df)

inv_label_map = {
    int(code): cls for cls, code in zip(le.classes_, le.transform(le.classes_))
}
synergy_code = le.transform(["synergy"])[0]
ant_code = le.transform(["antagonism"])[0]


# ==========================
# 2) Load best hyperparameters (RF run)
# ==========================
if not best_params_path.exists():
    raise FileNotFoundError(f"best_params JSON not found at: {best_params_path}")

with open(best_params_path) as f:
    best_params_data = json.load(f)

best_params = best_params_data["best_params"]
print("\nLoaded RF best_params from:", best_params_path)
print(best_params)


# ==========================
# 3) Outer splits (CV1) -- identical to HALO
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
        if verbose:
            print("=" * 72)
            print(f"CV1 outer fold {fold_idx}/{n_splits} (Drug Pair grouping):")
            print(f"Train size: {len(tr_idx)} ({len(tr_idx) / n * 100:.2f}%)")
            print(f"Test size : {len(te_idx)} ({len(te_idx) / n * 100:.2f}%)")
            print(f"Test + Train set: {len(tr_idx) + len(te_idx)}")
            print("-" * 72)
    return splits


outer_splits = make_splits_cv1(n_splits=5, verbose=True)


# ==========================
# 4) Run outer CV with fixed RF best_params
#    and outer-train-only feature selection
# ==========================
lgb.register_logger(
    type(
        "SilentLogger",
        (),
        {
            "info": lambda *a, **k: None,
            "warning": lambda *a, **k: None,
        },
    )()
)

fold_results = []
cm_total = None
all_test_dfs = []
all_train_dfs = []

for fold_idx, (tr_idx, te_idx) in enumerate(outer_splits, 1):
    print("\n" + "#" * 72)
    print(f"########## OUTER FOLD {fold_idx}/{len(outer_splits)} ##########")
    print("#" * 72 + "\n")

    X_tr = X.iloc[tr_idx].reset_index(drop=True)
    X_te = X.iloc[te_idx].reset_index(drop=True)
    y_tr = y_enc[tr_idx]
    y_te = y_enc[te_idx]

    df_tr = df.iloc[tr_idx].reset_index(drop=True)
    df_te = df.iloc[te_idx].reset_index(drop=True)

    selected_outer = select_features_lgbm(
        X_train=X_tr,
        y_train=y_tr,
        feat_cols=feat_cols,
        corr_min=corr_min,
        keep_top_frac=keep_top_frac,
    )

    X_tr_sel = X_tr[selected_outer].copy()
    X_te_sel = X_te[selected_outer].copy()

    m_final = RandomForestClassifier(
        random_state=777,
        n_jobs=4,
        **best_params,
    )
    m_final.fit(X_tr_sel, y_tr)

    pos_idx = np.flatnonzero(m_final.classes_ == synergy_code)[0]

    p_te = m_final.predict_proba(X_te_sel)[:, pos_idx]
    y_pred = (p_te >= 0.5).astype(int)

    y_te_bin = (y_te == synergy_code).astype(int)

    accuracy_test = accuracy_score(y_te, y_pred)
    f1_macro_test = f1_score(y_te, y_pred, average="macro")
    f1_weighted_test = f1_score(y_te, y_pred, average="weighted")
    roc_auc_test = roc_auc_score(y_te_bin, p_te)

    print(f"\n=== Held-out Test (fold {fold_idx}) ===")
    print(f"ROC AUC : {roc_auc_test:.3f}")
    print(f"Acc     : {accuracy_test:.3f}")
    print(f"F1 (w)  : {f1_weighted_test:.3f}")
    print("\nConfusion matrix:\n", confusion_matrix(y_te, y_pred))
    print(
        "\nReport:\n",
        classification_report(y_te, y_pred, target_names=le.classes_),
    )

    p_tr = m_final.predict_proba(X_tr_sel)[:, pos_idx]
    y_tr_pred = (p_tr >= 0.5).astype(int)
    y_tr_bin = (y_tr == synergy_code).astype(int)

    accuracy_train = accuracy_score(y_tr, y_tr_pred)
    f1_weighted_train = f1_score(y_tr, y_tr_pred, average="weighted")
    roc_auc_train = roc_auc_score(y_tr_bin, p_tr)

    print("\n=== Overfitting check (fold", fold_idx, ") ===")
    print("Train AUC:", round(roc_auc_train, 3), "| Test AUC:", round(roc_auc_test, 3))
    print("Train Acc:", round(accuracy_train, 3), "| Test Acc:", round(accuracy_test, 3))
    print(
        "Train F1w:", round(f1_weighted_train, 3),
        "| Test F1w:", round(f1_weighted_test, 3),
    )

    fold_results.append(
        dict(
            fold=fold_idx,
            roc_auc_test=roc_auc_test,
            accuracy_test=accuracy_test,
            f1_weighted_test=f1_weighted_test,
            roc_auc_train=roc_auc_train,
            accuracy_train=accuracy_train,
            f1_weighted_train=f1_weighted_train,
            n_train=len(tr_idx),
            n_test=len(te_idx),
            n_selected_features=len(selected_outer),
        )
    )

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
    all_test_dfs.append(test_out_fold)
    all_train_dfs.append(train_out_fold)

    order = ["antagonism", "synergy"]
    order_idx = le.transform(order)
    cm = confusion_matrix(y_te, y_pred, labels=order_idx)
    cm_total = cm if cm_total is None else cm_total + cm

test_out_all = pd.concat(all_test_dfs, ignore_index=True)
train_out_all = pd.concat(all_train_dfs, ignore_index=True)
test_out_all.to_csv(ext_out / "internal_test_predictions_cv1.csv", index=False)
train_out_all.to_csv(ext_out / "internal_train_predictions_cv1.csv", index=False)

metrics_per_fold_df = pd.DataFrame(fold_results)
metrics_per_fold_df.to_csv(ext_out / "internal_metrics_per_fold_cv1.csv", index=False)
print("\nSaved internal CV metrics & predictions to:", ext_out)


# ==========================
# 5) Train FINAL RF model on ALL training data
#    with feature selection on ALL internal training data only
# ==========================
selected_features_final = select_features_lgbm(
    X_train=X,
    y_train=y_enc,
    feat_cols=feat_cols,
    corr_min=corr_min,
    keep_top_frac=keep_top_frac,
)

X_final = X[selected_features_final].copy()

final_model = RandomForestClassifier(
    random_state=777,
    n_jobs=4,
    **best_params,
)
final_model.fit(X_final, y_enc)
pos_idx_final = np.flatnonzero(final_model.classes_ == synergy_code)[0]
print("\nTrained FINAL RF model on all training data.")
print("Selected features for FINAL model:", len(selected_features_final))


# ==========================
# 6) Build elementwise CC features for external set
# ==========================
if not external_base_path.exists():
    raise FileNotFoundError(f"External base dataset not found at: {external_base_path}")

ext_base = pd.read_csv(external_base_path).copy()
print("\nLoaded external base dataset:", external_base_path)
print("Shape:", ext_base.shape)

required_cols = ["Drug A", "Drug B", "Drug A Inchikey", "Drug B Inchikey"]
missing_req = [c for c in required_cols if c not in ext_base.columns]
if missing_req:
    raise ValueError(f"External base dataset is missing required columns: {missing_req}")

ext_base["Drug A Inchikey"] = ext_base["Drug A Inchikey"].astype(str).str.upper().str.strip()
ext_base["Drug B Inchikey"] = ext_base["Drug B Inchikey"].astype(str).str.upper().str.strip()

if "Drug Pair" not in ext_base.columns:
    ext_base["Drug Pair"] = ext_base.apply(
        lambda x: "::".join(sorted([x["Drug A Inchikey"], x["Drug B Inchikey"]])),
        axis=1,
    )

print(f"\nCombination methods in the ACDB external validation set:")
print(ext_base["Method"].value_counts())

train_pairs = set(combos_df["Drug Pair"].astype(str))
before_n = len(ext_base)
ext_base = ext_base[~ext_base["Drug Pair"].astype(str).isin(train_pairs)].copy()
print(f"Dropped {before_n - len(ext_base)} / {before_n} ACDB rows overlapping training pairs.")

print(ext_base.groupby("Method")["Interaction Type"].value_counts())
unique_compounds = pd.unique(pd.concat([ext_base["Drug A Inchikey"], ext_base["Drug B Inchikey"]]))
print(f"Number of unique antibacterials in acdb validation set: {len(unique_compounds)}")

ext_elem = fm.elementwise_similarity(ext_base, cc_df)

print("Elementwise external matrix shape (before label filtering):", ext_elem.shape)

if "Interaction Type" not in ext_elem.columns:
    raise ValueError("External elementwise matrix lacks 'Interaction Type' column.")

ext_elem = ext_elem[ext_elem["Interaction Type"].isin(["synergy", "antagonism"])].copy()
print("External elementwise after filtering to synergy/antagonism:", ext_elem.shape)


# ==========================
# 7) Align final selected features & predict on external set
# ==========================
missing_in_ext = set(selected_features_final) - set(ext_elem.columns)
if missing_in_ext:
    raise ValueError(
        "External elementwise dataset is missing final selected feature columns. "
        f"Example missing cols: {sorted(list(missing_in_ext))[:10]}"
    )

X_ext = ext_elem[selected_features_final].copy()
y_ext_text = ext_elem["Interaction Type"].copy()
y_ext = le.transform(y_ext_text)
y_ext_bin = (y_ext == synergy_code).astype(int)

print("\nExternal set size (after alignment):", len(ext_elem))

p_synergy_ext = final_model.predict_proba(X_ext)[:, pos_idx_final]
y_pred_ext = (p_synergy_ext >= 0.5).astype(int)
y_pred_ext_label = [inv_label_map[int(v)] for v in y_pred_ext]

ext_elem["y_true_int"] = y_ext
ext_elem["y_true_label"] = y_ext_text.values
ext_elem["p_synergy"] = p_synergy_ext
ext_elem["y_pred_int"] = y_pred_ext
ext_elem["y_pred_label"] = y_pred_ext_label

cm_ext = confusion_matrix(y_ext, y_pred_ext, labels=[ant_code, synergy_code])
tn, fp, fn, tp = cm_ext.ravel()

acc_ext = accuracy_score(y_ext, y_pred_ext)
f1_ext = f1_score(y_ext, y_pred_ext)
try:
    auc_ext = roc_auc_score(y_ext_bin, p_synergy_ext)
except ValueError:
    auc_ext = float("nan")

print("\n=== External evaluation (ACDB, RF baseline) ===")
print("n =", len(ext_elem))
print("Accuracy:", acc_ext)
print("F1      :", f1_ext)
print("ROC AUC :", auc_ext)
print("Confusion matrix [[TN, FP], [FN, TP]]:\n", cm_ext)
print(
    "\nReport:\n",
    classification_report(y_ext, y_pred_ext, target_names=le.classes_),
)


# ==========================
# 7b) Per-method metrics (FICI vs Loewe) + combined
# ==========================
def compute_metrics(y_true, y_pred, y_prob):
    cm = confusion_matrix(y_true, y_pred, labels=[ant_code, synergy_code])
    tn, fp, fn, tp = cm.ravel()
    y_true_bin = (y_true == synergy_code).astype(int)
    try:
        auc = roc_auc_score(y_true_bin, y_prob)
    except ValueError:
        auc = float("nan")
    return dict(
        n=len(y_true),
        accuracy=accuracy_score(y_true, y_pred),
        f1=f1_score(y_true, y_pred, zero_division=0),
        f1_macro=f1_score(y_true, y_pred, average="macro", zero_division=0),
        roc_auc=auc,
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
    )


per_method_rows = []

for method_name, sub in ext_elem.groupby("Method"):
    y_true_sub = sub["y_true_int"].to_numpy()
    y_pred_sub = sub["y_pred_int"].to_numpy()
    p_sub = sub["p_synergy"].to_numpy()

    row = dict(method=method_name)
    row.update(compute_metrics(y_true_sub, y_pred_sub, p_sub))
    per_method_rows.append(row)

    print(f"\n=== External evaluation (ACDB — {method_name}, RF baseline) ===")
    print(f"n = {row['n']}")
    print(f"Accuracy: {row['accuracy']:.3f}")
    print(f"F1      : {row['f1']:.3f}")
    print(f"ROC AUC : {row['roc_auc']}")

combined_row = dict(method="combined")
combined_row.update(compute_metrics(y_ext, y_pred_ext, p_synergy_ext))
per_method_rows.append(combined_row)

metrics_by_method_df = pd.DataFrame(per_method_rows)
metrics_by_method_path = ext_out / "external_metrics_by_method_acdb_rf.csv"
metrics_by_method_df.to_csv(metrics_by_method_path, index=False)
print("\nSaved per-method + combined RF metrics to:", metrics_by_method_path)
print(metrics_by_method_df)


# ==========================
# 8) Save everything needed for plotting
# ==========================
ext_pred_path = ext_out / "external_predictions_acdb_rf.csv"
ext_elem.to_csv(ext_pred_path, index=False)
print("\nSaved external per-pair predictions to:", ext_pred_path)

metrics_ext = pd.DataFrame(
    [{
        "dataset": "acdb_external_rf",
        "model": "RandomForestClassifier",
        "n": len(ext_elem),
        "accuracy": acc_ext,
        "f1": f1_ext,
        "roc_auc": auc_ext,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "n_selected_features_final": int(len(selected_features_final)),
    }]
)
metrics_ext_path = ext_out / "external_metrics_acdb_rf.csv"
metrics_ext.to_csv(metrics_ext_path, index=False)
print("Saved external summary RF metrics to:", metrics_ext_path)

fpr, tpr, roc_thr = roc_curve(y_ext_bin, p_synergy_ext)
prec, rec, pr_thr = precision_recall_curve(y_ext_bin, p_synergy_ext)

roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": roc_thr})
pr_df = pd.DataFrame({"recall": rec, "precision": prec})
pr_thr_df = pd.DataFrame({"threshold": pr_thr})

roc_df.to_csv(ext_out / "external_roc_curve_acdb_rf.csv", index=False)
pr_df.to_csv(ext_out / "external_pr_curve_acdb_rf.csv", index=False)
pr_thr_df.to_csv(ext_out / "external_pr_thresholds_acdb_rf.csv", index=False)

print("Saved ROC and PR curve data to:", ext_out)

cm_df = pd.DataFrame(
    cm_ext,
    index=["true_antagonism", "true_synergy"],
    columns=["pred_antagonism", "pred_synergy"],
)
cm_df.to_csv(ext_out / "external_confusion_matrix_acdb_rf.csv")

print("\n=== External evaluation script (RF, ACDB) DONE ===")