# **HALO**

**HALO** is a research codebase for **bioactivity-driven prediction of antibacterial drug synergy** using machine learning and multi-scale chemical and biological features.

The repository contains the full pipeline used in the study, including:
- data preprocessing and curation,
- feature construction (Chemical Checker and strain-space),
- model training and evaluation,
- figure and table generation for the manuscript.

---

## Repository structure
HALO_repo/
├── src/
├── data/
├── feature_pipeline/
├── figure_pipeline/
├── scripts/
├── notebooks/
├── results/
├── figures/
├── paper/
└── README.md



### Directory overview

- **`src/`**  
  Core reusable source code: mappers, shared utilities, and helper functions used across scripts and notebooks.

- **`data/`**  
  Data at different stages of the pipeline:
  - raw and extracted datasets,
  - curated reference tables (e.g. drug lists),
  - final feature matrices used for modeling.

- **`feature_pipeline/`**  
  Feature construction pipelines:
  - `chemicalchecker/`: fetching and assembling Chemical Checker features  
  - `strain_space/`: construction of strain-space (S-space) features following the Chemical Checker protocol

- **`scripts/`**  
  Scripted entry points to reproduce main experiments, model training, and external validation runs.

- **`notebooks/`**  
  Exploratory, preprocessing, and intermediate analysis notebooks used during development and data curation.

- **`figure_pipeline/`**  
  Scripts used to generate individual figure panels and tables programmatically.

- **`figures/`**  
  Exported final figures used in the manuscript (generated outputs).

- **`results/`**  
  Model outputs, logs, metrics, and intermediate experiment artifacts (generated).

- **`paper/`**  
  Manuscript source files and supplementary materials.

---

## Feature construction overview

HALO uses two complementary feature spaces:

1. **Chemical Checker (CC) features**  
Multi-level chemical and biological descriptors spanning levels **A–E**, with **128 dimensions per sublevel**.

- Features are fetched via the Chemical Checker API.
- Raw CC signatures are assembled into model-ready matrices.
- Final CC feature files are stored under:
`data/features/chemicalchecker_cc/`
including:
- `chemicalchecker_data.csv` (raw per-level signatures)
- `features_25_levels_into_1.csv`
- `features_15_levels_into_1.csv`

2. **Strain-space (S-space) features**  
Data-driven embeddings derived from **drug–strain fitness profiles**.

- Constructed using the Chemical Checker signature protocol:
- type-0 (sign0) → type-I (sign1) → type-II (sign2)
- The final 128-dimensional strain-space embeddings are used directly in modeling.
- Intermediate and final artifacts are managed under:
`feature_pipeline/strain_space/`

Detailed documentation for both pipelines is provided in:
- `feature_pipeline/chemicalchecker/README.md`
- `feature_pipeline/strain_space/README.md`

---

## Setup

### Requirements
- Python ≥ 3.9
- Conda or virtualenv recommended
- Core dependencies include:  
  `pandas`, `numpy`, `scikit-learn`, `lightgbm`, `chemicalchecker`, `requests`

### Chemical Checker
This project relies on the **Chemical Checker** framework.

You must install it separately:
```bash
git clone https://github.com/sbnb-irb/chemical_checker.git
‍‍‍‍‍‍```

Set the environment variable CC_CONFIG to point to a local cc_config.json file before running strain-space construction:
`export CC_CONFIG=/path/to/cc_config.json`
A reference configuration used in this study is included in:
`feature_pipeline/chemicalchecker/cc_config.json`
Users must adapt paths to their local setup.

---

## Reproducibility notes
- Generated outputs (cached Chemical Checker artifacts, model results, figures) are not required to run the pipeline and may be excluded from version control.

- Scripts and notebooks assume paths relative to the repository root.

- Large external datasets and proprietary databases are not redistributed and must be obtained independently.

---

## Citation
If you use this codebase, please cite the associated manuscript (details to be added upon publication).

---

## Contact
For questions or issues related to this repository, please open a GitHub issue or contact the authors.
