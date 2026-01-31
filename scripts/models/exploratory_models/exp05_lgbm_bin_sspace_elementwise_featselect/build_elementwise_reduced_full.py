#!/usr/bin/env python3
"""
** Experiment: build_elementwise_reduced_full **

- task: feature selection for multiclass classification and regression, same logic as binary
- feature_design: elementwise similarity (cos_elem_*, euc_elem_*)
- use_sspace: true
- goal: reduce elementwise feature space
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.append("/home/hannie/cc_ml/pipeline")
from feature_mapper.feature_mapper import FeatureMapper

base_dir = Path("/home/hannie/cc_ml/pipeline/preprocessing/data_to_use")
cc_path = base_dir / "features_25_levels_into_1.csv"
ss_path = base_dir / "sspace.csv"
combos_path = base_dir / "combinations_combined.csv"

# 1) Load original data
cc_df = pd.read_csv(cc_path).copy()
ss_df = pd.read_csv(ss_path).copy()
combinations_df = pd.read_csv(combos_path).copy()

features_cc_s = (
    cc_df
    .merge(ss_df, on="inchikey", how="inner", suffixes=("", "_s"))
)

# 2) Full elementwise matrix for ALL combos (3 classes + Bliss)
df_full = FeatureMapper().elementwise_similarity(combinations_df, features_cc_s)

# 3) Load selected feature names from exp05
feat_list_path = Path(
    "/home/hannie/cc_ml/models/results/"
    "exp05_lgbm_bin_sspace_elementwise_featselect/"
    "selected_features_cv1.txt"
)
with open(feat_list_path) as f:
    selected_features = [line.strip() for line in f if line.strip() and not line.startswith("#")]

meta_cols = [
    "Drug A", "Drug B",
    "Drug A Inchikey", "Drug B Inchikey",
    "Strain", "Specie",
    "Bliss Score",
    "Interaction Type",
    "Source", "Drug Pair",
]

df_out = df_full[meta_cols + selected_features].copy()

out_path = Path(
    "/home/hannie/cc_ml/models/results/"
    "exp05_lgbm_bin_sspace_elementwise_featselect/"
    "elementwise_features_filtered_cv1_full.csv"
)
df_out.to_csv(out_path, index=False)
print("Wrote full reduced matrix to:", out_path)
print("Shape:", df_out.shape)
print(df_out["Interaction Type"].value_counts())
