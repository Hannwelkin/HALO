#!/usr/bin/env python3
"""
** Experiment: exp09b_lgbm_bin_sspace_elementwise_reduced_simplecv_bliss005 **

- task: binary classification using a refined Bliss-score threshold of ±0.05 to exclude neutral interactions
- feature_design: reduced elementwise similarity (exp05b)
- use_sspace: true (already baked into features)
- nested_cv: false (simple CV, intentionally leaky)
- purpose: estimate the "upper bound" accuracy achievable with selected features
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import (
    StratifiedKFold,
    train_test_split,
    RandomizedSearchCV
)
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix
)
import lightgbm as lgb

# Make pipeline importable
sys.path.append("/home/hannie/cc_ml/pipeline")
from shared_utils.metrics import classification_metrics, overfitting_report


def main():
    print("\n=== EXP09d: Simple CV + Reduced Elementwise Features only from CC, Bliss cutoff ±0.05 ===\n")

    # ==========================
    # 1) Load reduced feature file from exp05
    # ==========================
    reduced_path = Path(
        "/home/hannie/cc_ml/models/results/"
        "exp05d_lgbm_bin_nosspace_elementwise_featselect_bliss005/"
        "elementwise_features_filtered_cv1_cc_only.csv"
    )

    if not reduced_path.exists():
        raise FileNotFoundError(f"Reduced feature file not found: {reduced_path}")

    df = pd.read_csv(reduced_path).copy()
    print("Loaded df:", df.shape)
    print(df["Interaction Type"].value_counts())

    # ==========================
    # 2) Keep binary classes only
    # ==========================
    df = df[df["Interaction Type"].isin(["synergy", "antagonism"])].copy()
    print("\nFiltered (binary classes):", df.shape)

    # ==========================
    # 3) Feature columns
    # ==========================
    drop_cols = [
        "Drug A", "Drug B",
        "Drug A Inchikey", "Drug B Inchikey",
        "Strain", "Specie",
        "Bliss Score",
        "Interaction Type",
        "Source", "Drug Pair",
    ]

    feat_cols = [c for c in df.columns if c not in drop_cols]
    print("Feature columns:", len(feat_cols))

    X = df[feat_cols].copy()
    y = df["Interaction Type"].copy()

    # Encode target
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    print("Classes:", list(le.classes_))

    # ==========================
    # 4) Simple CV train/test split (LEAKY ON PURPOSE)
    # ==========================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.20, random_state=42, stratify=y_enc
    )

    # ==========================
    # 5) LightGBM baseline + randomized search
    # ==========================
    base_clf = lgb.LGBMClassifier(
        objective="binary",
        metric="binary_logloss",
        n_jobs=1,
        random_state=42,
    )

    param_dist = {
        "learning_rate": [0.02, 0.05],
        "n_estimators": [300, 600],
        "max_depth": [3, 4, 5, 6],
        "num_leaves": [7, 15, 31],
        "min_child_samples": [50, 100, 200],
        "feature_fraction": [0.3, 0.4, 0.6, 0.8],
        "subsample": [0.6, 0.8],
        "subsample_freq": [1],
        "lambda_l1": [0.0, 0.1, 1.0, 5.0],
        "lambda_l2": [0.0, 0.1, 1.0, 5.0],
        "max_bin": [63, 127, 255],
        "min_split_gain": [0.0, 0.05, 0.1],
    }

    print("\n--- Running RandomizedSearchCV (simple CV) ---\n")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    search = RandomizedSearchCV(
        estimator=base_clf,
        param_distributions=param_dist,
        n_iter=40,
        cv=cv,
        scoring="f1",
        verbose=1,
        n_jobs=1,
        random_state=42,
    )

    search.fit(X_train, y_train)
    best_model = search.best_estimator_

    print("\nBest parameters:", search.best_params_)

    # ==========================
    # 6) Final evaluation
    # ==========================
    y_pred = best_model.predict(X_test)
    y_score = best_model.predict_proba(X_test)  # full (n_samples, 2) matrix

    print("\n=== TEST METRICS (LEAKY SIMPLE CV) ===")
    classification_metrics(
        y_test,
        y_pred,
        y_score=y_score,
        class_names=le.classes_,
    )

    # ==========================
    # 7) Overfitting report
    # ==========================
    print("\n=== Overfitting Report ===")
    overfitting_report(
        best_model,
        X_train, y_train,
        X_test, y_test,
        task="classification",
        average="macro"
    )

    print("\n=== EXP09d DONE ===\n")


if __name__ == "__main__":
    main()




