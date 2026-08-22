"""
scripts/other_analysis/check_external_overlap.py

- Overlap checker for any external validation set against HALO's internal training+test data. 
- Matches on InChIKey
- Pairs are treated as unordered (A+B == B+A).
"""

import pandas as pd

from halo.paths import INTERIM, PROCESSED

def check_overlap(internal_df, external_df, internal_label="internal", external_label="external"):
    """
    Compare an external validation set against the HALO's internal training+test set.

    Returns a dict with:
      - n_pair_overlap: number of external pairs that already exist internally
      - overlapping_pairs: the actual overlapping pair keys
      - pct_pair_overlap: % of external pairs affected
      - n_compound_overlap: number of individual compounds in external set
                             also seen internally (weaker signal than pair overlap)
      - novel_compounds: compounds in external set NEVER seen internally
    """
    internal_pairs = set(internal_df["Drug Pair"])  # 'Drug Pair' column is already sorted alphabetically 
    external_pairs = set(external_df["Drug Pair"])

    overlapping_pairs = external_pairs & internal_pairs
    pct_pair_overlap = 100 * len(overlapping_pairs) / len(external_pairs) if external_pairs else 0.0

    internal_compounds = set(pd.concat([internal_df["Drug A Inchikey"], internal_df["Drug B Inchikey"]]).dropna().unique())
    external_compounds = set(pd.concat([external_df["Drug A Inchikey"], external_df["Drug B Inchikey"]]).dropna().unique())

    compound_overlap = external_compounds & internal_compounds
    novel_compounds = external_compounds - internal_compounds

    print(f"\n{'='*70}")
    print(f"Overlap check: {external_label} vs. {internal_label}")
    print(f"{'='*70}")
    print(f"External set: {len(external_df)} rows, {len(external_pairs)} unique pairs, "
          f"{len(external_compounds)} unique compounds")
    print(f"Internal set: {len(internal_df)} rows, {len(internal_pairs)} unique pairs, "
          f"{len(internal_compounds)} unique compounds")
    print(f"\n--- Pair-level overlap (the leakage-relevant check) ---")
    print(f"Overlapping pairs: {len(overlapping_pairs)} / {len(external_pairs)} "
          f"({pct_pair_overlap:.1f}% of external set)")
    if overlapping_pairs:
        print("WARNING: some external pairs already exist in the internal set.")
        print("These rows may not represent genuinely novel pair-level generalization.")
    else:
        print("No pair-level overlap -- external set is fully novel at the pair level.")
    print(f"\n--- Compound-level overlap (weaker signal, informational only) ---")
    print(f"Compounds also seen internally: {len(compound_overlap)} / {len(external_compounds)}")
    print(f"Compounds NEVER seen internally (fully novel chemistry): {len(novel_compounds)}")

    return {
        "n_pair_overlap": len(overlapping_pairs),
        "overlapping_pairs": overlapping_pairs,
        "pct_pair_overlap": pct_pair_overlap,
        "n_compound_overlap": len(compound_overlap),
        "novel_compounds": novel_compounds,
    }


if __name__ == "__main__":

    halo = pd.read_csv(PROCESSED / "halo_training_dataset.csv")
    indigo = pd.read_csv(INTERIM / "source_d_chandrasekaran" / "chandrasekaran_cleaned_data.csv")
    acdb = pd.read_csv(INTERIM / "source_c_acdb" / "acdb_cleaned_data_validation.csv")


    check_overlap(internal_df=halo, external_df=indigo, internal_label="HALO internal", external_label="INDIGO")
    
    check_overlap(internal_df= halo, external_df=acdb, internal_label="HALO internal", external_label="ACDB FICI/Loewe")
