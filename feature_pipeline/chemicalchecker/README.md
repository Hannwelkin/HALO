# ChemicalChecker dependency

This directory documents the external **ChemicalChecker** dependency used in this project.

ChemicalChecker is **not included** in this repository and must be installed separately.

---

## Installation

Install ChemicalChecker from the official repository:

https://github.com/sbnb-irb/chemical_checker/

Follow the installation instructions provided by the ChemicalChecker authors, including database setup.

---

## Contents of this directory

This directory contains two files related to ChemicalChecker usage in this project:

- `fetch_cc_features.py`  
  Script used to fetch **original ChemicalChecker (CC) features** via the CC API.  
  The output is stored in `data/features/chemicalchecker_cc/chemicalchecker_data.csv`.

  This file has the following structure:
  - rows: drugs (indexed by InChIKey)
  - columns:
    - `level`: CC sublevel identifier (`A1`–`E5`, 25 total)
    - `dim_0` … `dim_127`: 128-dimensional signature for that level

  From this file, level-wise CC features are assembled into:
  - `features_25_levels_into_1.csv`: concatenation of all 25 CC sublevels (A1–E5)
  - `features_15_levels_into_1.csv`: concatenation of CC levels A–C (15 first sublevels)
  
- `cc_config.json`  
Reference ChemicalChecker configuration file used during strain-space construction  
in `feature_pipeline/strain_space/notebooks/sspace.ipynb`.

---

## `cc_config.json`

ChemicalChecker requires a configuration file (`cc_config.json`) that defines local paths to:
- ChemicalChecker databases
- temporary files
- log directories

These paths are **machine-specific**.

We include `cc_config.json` **only as a reference configuration used in this study**.  
Users must adapt the paths and set `CC_CONFIG` to point to a local copy.

---

## Required environment variable

Before running any ChemicalChecker-based step, set:

```bash
export CC_CONFIG=/absolute/path/to/your/cc_config.json

