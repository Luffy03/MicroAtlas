"""
preprocess.py
=============
Parse BBBC021 metadata and build treatment-level mapping tables.

Responsibilities:
  - Parse BBBC021_v1_image.csv (image paths, plate/well/compound/concentration)
  - Parse BBBC021_v1_moa.csv (compound + concentration -> MoA class)
  - Parse BBBC021_v1_compound.csv (compound -> SMILES)
  - Identify DMSO control wells
  - Build unified treatment table with MoA labels
  - Print dataset statistics

Usage:
  python biomarker_discovery/preprocess.py
  python biomarker_discovery/preprocess.py --stats
"""

import argparse
import sys
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config


def load_image_metadata():
    """Load BBBC021_v1_image.csv.

    Columns: TableNumber, ImageNumber,
             Image_FileName_DAPI, Image_PathName_DAPI,
             Image_FileName_Tubulin, Image_PathName_Tubulin,
             Image_FileName_Actin, Image_PathName_Actin,
             Image_Metadata_Plate_DAPI, Image_Metadata_Well_DAPI,
             Replicate, Image_Metadata_Compound,
             Image_Metadata_Concentration
    """
    if not config.IMAGE_CSV.exists():
        raise FileNotFoundError(
            f"Image metadata not found: {config.IMAGE_CSV}\n"
            f"Run: python download.py --metadata_only"
        )
    df = pd.read_csv(config.IMAGE_CSV)
    print(f"Loaded image metadata: {len(df):,} rows")
    return df


def load_moa_labels():
    """Load BBBC021_v1_moa.csv.

    Columns: compound, concentration, moa
    """
    if not config.MOA_CSV.exists():
        raise FileNotFoundError(
            f"MoA labels not found: {config.MOA_CSV}\n"
            f"Run: python download.py --metadata_only"
        )
    df = pd.read_csv(config.MOA_CSV)
    print(f"Loaded MoA labels: {len(df)} compound-concentrations, "
          f"{df['moa'].nunique()} classes")
    return df


def load_compound_smiles():
    """Load BBBC021_v1_compound.csv.

    Columns: compound, smiles
    """
    if not config.COMPOUND_CSV.exists():
        return pd.DataFrame(columns=["compound", "smiles"])
    df = pd.read_csv(config.COMPOUND_CSV)
    print(f"Loaded compound SMILES: {len(df)} compounds")
    return df


def build_unified_table(image_df, moa_df, compound_df):
    """Build unified treatment table joining images with MoA labels.

    Returns DataFrame with columns:
      ImageNumber, plate, well, compound, concentration,
      moa, smiles, replicate,
      path_DAPI, path_Actin, path_Tubulin
    """
    df = image_df.copy()

    # Standardize column names
    col_map = {
        "Image_Metadata_Plate_DAPI": "plate",
        "Image_Metadata_Well_DAPI": "well",
        "Image_Metadata_Compound": "compound",
        "Image_Metadata_Concentration": "concentration",
        "Replicate": "replicate",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    # Normalize compound names for joining
    df["compound_clean"] = df["compound"].astype(str).str.strip().str.lower()

    # Normalize concentration for joining (MoA CSV uses numeric concentrations)
    df["concentration_num"] = pd.to_numeric(df["concentration"], errors="coerce")

    # Join MoA labels (on compound + concentration)
    moa_df = moa_df.copy()
    moa_df["compound_clean"] = moa_df["compound"].astype(str).str.strip().str.lower()
    moa_df["concentration_num"] = pd.to_numeric(
        moa_df["concentration"], errors="coerce"
    )

    df = df.merge(
        moa_df[["compound_clean", "concentration_num", "moa"]].drop_duplicates(),
        on=["compound_clean", "concentration_num"],
        how="left",
    )

    # Fill missing MoA with "unknown"
    df["moa"] = df["moa"].fillna("unknown")

    # Join SMILES
    if not compound_df.empty:
        compound_df = compound_df.copy()
        compound_df["compound_clean"] = (
            compound_df["compound"].astype(str).str.strip().str.lower()
        )
        df = df.merge(
            compound_df[["compound_clean", "smiles"]].drop_duplicates(),
            on="compound_clean",
            how="left",
        )

    # Build local image paths
    df = _build_image_paths(df)

    # Identify DMSO controls
    df["is_dmso"] = df["compound_clean"].str.lower().str.contains("dmso", na=False)

    # Clean up
    df = df.drop(columns=["compound_clean", "concentration_num"], errors="ignore")

    return df


def _build_image_paths(df):
    """Build local file paths for each channel image.

    BBBC021 image paths from the CSV point to the original download location.
    We map them to our organized directory structure:
      images/{plate}/{channel}/{filename}
    """
    channel_file_cols = {
        "DAPI": "Image_FileName_DAPI",
        "Actin": "Image_FileName_Actin",
        "Tubulin": "Image_FileName_Tubulin",
    }

    for ch, col in channel_file_cols.items():
        path_col = f"path_{ch}"
        if col in df.columns and "plate" in df.columns:
            df[path_col] = df.apply(
                lambda row: str(
                    config.IMAGES_DIR / row["plate"] / ch / row[col]
                ),
                axis=1,
            )
        else:
            df[path_col] = ""

    return df


def build_treatment_summary(df):
    """Build treatment-level summary table.

    One row per unique (compound, concentration) pair.
    """
    # Group by compound + concentration
    groups = df.groupby(["compound", "concentration"])

    records = []
    for (cpd, conc), group in groups:
        rec = {
            "compound": cpd,
            "concentration": conc,
            "n_images": len(group),
            "n_plates": group["plate"].nunique(),
            "moa": group["moa"].iloc[0],
            "is_dmso": group["is_dmso"].iloc[0],
        }
        if "smiles" in group.columns:
            rec["smiles"] = group["smiles"].iloc[0]
        records.append(rec)

    summary = pd.DataFrame(records)
    return summary


def get_dmso_profiles(df):
    """Get all DMSO (negative control) field profiles for normalization."""
    dmso = df[df["is_dmso"]]
    print(f"  DMSO controls: {len(dmso)} fields of view")
    return dmso


def print_statistics(df, summary_df):
    """Print comprehensive dataset statistics."""
    print(f"\n{'=' * 60}")
    print("BBBC021 Dataset Statistics")
    print(f"{'=' * 60}")

    print(f"\nTotal fields of view: {len(df):,}")
    print(f"Unique plates: {df['plate'].nunique()}")
    print(f"Unique compounds: {df['compound'].nunique()}")
    print(f"Unique concentrations: {sorted(df['concentration'].dropna().unique())}")

    # DMSO
    n_dmso = df["is_dmso"].sum()
    print(f"\nDMSO control fields: {n_dmso}")

    # MoA distribution
    moa_df = df[df["moa"] != "unknown"]
    print(f"\nFields with MoA label: {len(moa_df):,} ({100*len(moa_df)/len(df):.1f}%)")

    moa_counts = moa_df["moa"].value_counts()
    print(f"\nMoA Class Distribution ({moa_df['moa'].nunique()} classes):")
    print("-" * 50)
    for moa, count in moa_counts.items():
        n_cpds = moa_df[moa_df["moa"] == moa]["compound"].nunique()
        print(f"  {moa:<35s} {n_cpds:3d} compounds, {count:5d} fields")

    # Concentration distribution
    conc_counts = df["concentration"].value_counts().sort_index()
    print(f"\nConcentration Distribution:")
    for conc, count in conc_counts.items():
        print(f"  {conc:>10} : {count:5d} fields")

    print(f"\n{'=' * 60}")


def save_processed_data(df, summary_df, output_dir=None):
    """Save processed tables to CSV."""
    output_dir = Path(output_dir) if output_dir else config.RESULTS_DIR / "preprocess"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Full image-level table
    full_path = output_dir / "image_table.csv"
    df.to_csv(full_path, index=False)
    print(f"  Saved: {full_path}")

    # Treatment summary
    summary_path = output_dir / "treatment_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"  Saved: {summary_path}")

    # MoA-only summary (compounds with known MoA)
    moa_summary = summary_df[summary_df["moa"] != "unknown"]
    moa_path = output_dir / "moa_compounds.csv"
    moa_summary.to_csv(moa_path, index=False)
    print(f"  Saved: {moa_path} ({len(moa_summary)} treatments with MoA)")


def run_preprocessing():
    """Run full preprocessing pipeline."""
    print("=" * 60)
    print("BBBC021 Preprocessing")
    print("=" * 60)

    # Load raw data
    image_df = load_image_metadata()
    moa_df = load_moa_labels()
    compound_df = load_compound_smiles()

    # Build unified table
    print(f"\nBuilding unified treatment table...")
    df = build_unified_table(image_df, moa_df, compound_df)

    # Summary
    summary_df = build_treatment_summary(df)

    # Statistics
    print_statistics(df, summary_df)

    # Save
    print(f"\nSaving processed data...")
    config.ensure_dirs()
    save_processed_data(df, summary_df)

    return df, summary_df


def main():
    parser = argparse.ArgumentParser(description="Preprocess BBBC021 metadata")
    parser.add_argument("--stats", action="store_true",
                        help="Print statistics only")
    args = parser.parse_args()

    df, summary_df = run_preprocessing()

    if args.stats:
        print_statistics(df, summary_df)


if __name__ == "__main__":
    main()
