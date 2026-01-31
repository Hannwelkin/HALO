# src/paths.py
from pathlib import Path

# HALO repository
ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data"
RAW = DATA / "a_raw"
EXTRACTED = DATA / "b_extracted"
INTERIM = DATA / "c_interim"
PROCESSED = DATA / "d_processed"

FEATURES = DATA / "features"
CC_FEATURES = FEATURES / "chemicalchecker_cc"
SS_FEATURES = FEATURES / "strain_space_ss"

REFERENCE = DATA / "reference"
DRUG_LISTS = REFERENCE / "drug_lists"

RESULTS = ROOT / "results"
MODEL_RESULTS = RESULTS / "models"

FEATURE_PIPELINE = ROOT / "feature_pipeline"
FIGURE_PIPELINE = ROOT / "figure_pipeline"
FIGURES = ROOT / "figures"
