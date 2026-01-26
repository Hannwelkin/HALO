# Dataflow

## Sources
- source_a: Brochado et al. dataset (Training set)
- source_b: Cacace et al. dataset (Training set)
- source_c: ACDB (Training set)
- source_d: Chandrasekaran et al dataset (External validation set)

## Pipeline stages
1. raw: original downloads in data/a_raw/
2. extracted: selected tables from raw in data/b_extracted
2. interim: per-source cleaned outputs in data/c_interim/
3. processed: merged final modeling dataset in data/d_processed/


## Current truth (what exists today)
- Raw preprocessing + filtering (all 4 sources): `notebooks/01_raw_preprocess_and_filter.ipynb`
  - Outputs: `data/b_interim/<source_name>/...` (cleaned per-source files)
  - Also 

- Merge, preprocessing + filtering (integrated dataset): `notebooks/02_postmerge_preprocess_and_filter.ipynb`
  - Produces merged dataset, preprocess, ...
  - Inputs: `data/b_interim/<source_name>/...`
  - Outputs: 
        - `data/c_processed/halo_dataset_v1.csv` (final dataset for modeling)
        - `data/c_processed/something.csv`(for external validation)


## Outputs used in the paper
- Final dataset: data/c_processed/halo_dataset_v1.csv
- External validation set: data/c_processed/soemthing.csv
<!-- - Key tables: results/tables/<...>
- Key figures: figures/main/<...> -->

