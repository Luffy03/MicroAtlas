"""
preprocess.py
=============
Build the field/treatment-level mapping tables for the U2OS-Cell-Painting dataset.

Instead of joining a platemap TSV with a compound-metadata TSV, the
U2OS-Cell-Painting dataset ships a single authoritative table, fl_data.csv, that already gives
plate / well / site / compound / MoA and the per-channel TIFF filenames. So this
stage simply:
  - enumerates the nucleus (DNA) images actually on disk to find which
    (plate, well, site) fields have been downloaded + extracted,
  - joins them with fl_data.csv to attach compound + MoA,
  - marks DMSO negative-control fields (moa == "dmso"),
  - emits the same image_table.csv schema the downstream pipeline expects.

Because every compound is imaged at a single concentration (10 uM), the
treatment level is equivalent to the compound level.

Usage:
  python src/morphology_profiling/preprocess.py
  python src/morphology_profiling/preprocess.py --plates P015080 P015081
  python src/morphology_profiling/preprocess.py --stats
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config


# =========================================================================
# Metadata loading
# =========================================================================

def load_fl_table():
    """Load fl_data.csv (plate, well, site, bf_site, compound, C1..C5, moa)."""
    if not config.FL_DATA_CSV.exists():
        raise FileNotFoundError(
            f"fl_data.csv not found: {config.FL_DATA_CSV}\n"
            f"Run: python download.py --metadata_only"
        )
    df = pd.read_csv(config.FL_DATA_CSV, dtype=str)
    # Normalize the join keys.
    df["plate"] = df["plate"].str.strip()
    df["well"] = df["well"].str.strip()
    df["site_i"] = df["site"].str.lstrip("s").astype(int)
    df["moa"] = df["moa"].fillna("").str.strip()
    df["compound"] = df["compound"].fillna("").str.strip()
    print(f"Loaded fl_data.csv: {len(df)} FL fields, "
          f"{df['compound'].nunique()} compounds, {df['moa'].nunique()} MoA labels")
    return df


# =========================================================================
# Field enumeration
# =========================================================================

_FIELD_RE = re.compile(r"^([A-P]\d{2})_s(\d+)_w\d+", re.IGNORECASE)


def build_fields_from_table(fl_df, plates=None):
    """Enumerate every FL field listed in fl_data.csv (disk-independent).

    This is the default: fl_data.csv is authoritative, so the full label table
    persists even after a plate's images are deleted in the disk-frugal loop
    (download -> segment -> featurize -> delete images). segment.py and
    feature_extraction.py enumerate work from whatever images are on disk, so
    image_table.csv is only needed to attach MoA labels during aggregation.

    Returns a DataFrame with columns: plate, well, site, field.
    """
    df = fl_df[["plate", "well", "site_i"]].rename(columns={"site_i": "site"}).copy()
    if plates is not None:
        df = df[df["plate"].isin(set(plates))]
    df = df.drop_duplicates(["plate", "well", "site"]).reset_index(drop=True)
    df["field"] = df["well"] + "_s" + df["site"].astype(str)
    n_plates = df["plate"].nunique() if len(df) else 0
    print(f"Enumerated {len(df)} FL fields across {n_plates} plate(s) from fl_data.csv")
    return df


def discover_fields(plates=None):
    """Enumerate downloaded nucleus images to list available (plate, well, site).

    Returns a DataFrame with columns: plate, well, site, field.
    field == "{well}_s{site}" (the shared field key used across the pipeline).
    """
    if not config.IMAGES_DIR.exists():
        raise FileNotFoundError(
            f"No images directory: {config.IMAGES_DIR}\nRun download.py first."
        )

    if plates is None:
        plates = [d.name for d in sorted(config.IMAGES_DIR.iterdir()) if d.is_dir()]

    records = []
    for plate in plates:
        nuc_dir = config.IMAGES_DIR / plate / config.NUCLEUS_CHANNEL
        if not nuc_dir.exists():
            print(f"  [WARN] No {config.NUCLEUS_CHANNEL} dir for plate {plate}")
            continue
        for f in sorted(nuc_dir.glob("*.tif")):
            m = _FIELD_RE.match(f.name)
            if not m:
                continue
            well, site = m.group(1).upper(), int(m.group(2))
            records.append({
                "plate": plate,
                "well": well,
                "site": site,
                "field": f"{well}_s{site}",
            })

    df = pd.DataFrame(records)
    n_plates = df["plate"].nunique() if len(df) else 0
    print(f"Discovered {len(df)} fields across {n_plates} plate(s)")
    return df


# =========================================================================
# Unified table
# =========================================================================

def build_unified_table(fields_df, fl_df):
    """Join discovered fields with fl_data.csv to attach compound + MoA.

    Returns DataFrame with columns:
      plate, well, site, field, broad_sample, compound, concentration,
      moa, pert_iname, smiles, is_dmso, path_<CHANNEL> ...
    """
    if fields_df.empty:
        return fields_df

    meta = fl_df[["plate", "well", "site_i", "compound", "moa"]].rename(
        columns={"site_i": "site"}
    )
    df = fields_df.merge(meta, on=["plate", "well", "site"], how="left")

    df["compound"] = df["compound"].fillna("").str.strip()
    df["moa"] = df["moa"].fillna("").str.strip()

    # DMSO negative controls (moa == "dmso" in fl_data.csv).
    df["is_dmso"] = df["moa"].str.lower() == config.DMSO_MOA_LABEL

    # Human-readable treatment label. DMSO -> "DMSO"; unmatched -> broad id.
    df["compound"] = df["compound"].replace("", np.nan)
    df.loc[df["is_dmso"], "compound"] = "DMSO"
    df["compound"] = df["compound"].fillna("unknown")

    # MoA label. Keep the dataset's verbatim class names for the 10 MoAs;
    # DMSO -> "DMSO"; unmatched fields -> "unknown". Downstream excludes DMSO
    # via is_dmso and skips "unknown", so clustering sees exactly 10 classes.
    df.loc[df["is_dmso"], "moa"] = "DMSO"
    df["moa"] = df["moa"].replace("", "unknown")

    # Single-concentration design: constant nominal concentration.
    df["concentration"] = config.DEFAULT_CONCENTRATION
    df.loc[df["is_dmso"], "concentration"] = 0.0

    # Columns kept for schema compatibility with the shared downstream pipeline.
    df["broad_sample"] = df["compound"]
    df["pert_iname"] = df["compound"]
    df["smiles"] = ""

    df = _build_image_paths(df)
    return df


def _build_image_paths(df):
    """Add path_<CHANNEL> columns mirroring download.py's on-disk layout."""
    channel_idx = {ch: i + 1 for i, ch in enumerate(config.CHANNEL_NAMES)}
    for ch in config.CHANNEL_NAMES:
        idx = channel_idx[ch]
        df[f"path_{ch}"] = df.apply(
            lambda r: str(config.IMAGES_DIR / r["plate"] / ch /
                          f"{r['well']}_s{int(r['site'])}_w{idx}.tif"),
            axis=1,
        )
    return df


def build_treatment_summary(df):
    """One row per treatment (compound). Single concentration -> compound level."""
    groups = df.groupby(["compound", "concentration"])
    records = []
    for (cpd, conc), group in groups:
        rec = {
            "compound": cpd,
            "concentration": conc,
            "n_images": len(group),
            "n_wells": group["well"].nunique(),
            "n_plates": group["plate"].nunique(),
            "moa": group["moa"].iloc[0],
            "is_dmso": group["is_dmso"].iloc[0],
            "broad_sample": group["broad_sample"].iloc[0],
            "smiles": group["smiles"].iloc[0],
        }
        records.append(rec)
    return pd.DataFrame(records)


def print_statistics(df, summary_df):
    print(f"\n{'=' * 60}")
    print("U2OS-Cell-Painting Dataset Statistics")
    print(f"{'=' * 60}")

    print(f"\nTotal fields of view: {len(df):,}")
    print(f"Unique plates: {df['plate'].nunique()}")
    print(f"Unique wells: {df['well'].nunique()}")
    print(f"Unique compounds (excl. DMSO): "
          f"{df.loc[~df['is_dmso'], 'compound'].nunique()}")

    n_dmso = df["is_dmso"].sum()
    print(f"\nDMSO control fields: {n_dmso}")

    moa_df = df[(df["moa"] != "unknown") & (df["moa"] != "DMSO")]
    print(f"\nFields with MoA label: {len(moa_df):,} "
          f"({100*len(moa_df)/max(len(df),1):.1f}%)")
    print(f"\nMoA Class Distribution ({moa_df['moa'].nunique()} classes):")
    print("-" * 50)
    moa_counts = moa_df.groupby("moa")["compound"].nunique().sort_values(ascending=False)
    for moa, n_cpds in moa_counts.items():
        n_fields = (moa_df["moa"] == moa).sum()
        print(f"  {moa:<40s} {n_cpds:2d} cpds, {n_fields:5d} fields")

    print(f"\n{'=' * 60}")


def save_processed_data(df, summary_df, output_dir=None):
    output_dir = Path(output_dir) if output_dir else config.RESULTS_DIR / "preprocess"
    output_dir.mkdir(parents=True, exist_ok=True)

    full_path = output_dir / "image_table.csv"
    df.to_csv(full_path, index=False)
    print(f"  Saved: {full_path}")

    summary_path = output_dir / "treatment_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"  Saved: {summary_path}")

    moa_summary = summary_df[(summary_df["moa"] != "unknown") &
                             (summary_df["moa"] != "DMSO")]
    moa_path = output_dir / "moa_compounds.csv"
    moa_summary.to_csv(moa_path, index=False)
    print(f"  Saved: {moa_path} ({len(moa_summary)} treatments with MoA)")


def run_preprocessing(plates=None, on_disk=False):
    print("=" * 60)
    print("U2OS-Cell-Painting Preprocessing")
    print("=" * 60)

    fl_df = load_fl_table()
    if on_disk:
        fields_df = discover_fields(plates=plates)
    else:
        fields_df = build_fields_from_table(fl_df, plates=plates)

    if fields_df.empty:
        print("\n[ERROR] No fields found. Check fl_data.csv / downloaded images.")
        return fields_df, pd.DataFrame()

    print("\nBuilding unified image table...")
    df = build_unified_table(fields_df, fl_df)
    summary_df = build_treatment_summary(df)

    print_statistics(df, summary_df)

    print("\nSaving processed data...")
    config.ensure_dirs()
    save_processed_data(df, summary_df)
    return df, summary_df


def main():
    parser = argparse.ArgumentParser(description="Preprocess U2OS-Cell-Painting metadata")
    parser.add_argument("--plates", type=str, nargs="+", default=None,
                        help="Restrict to specific plates (default: all 18)")
    parser.add_argument("--on_disk", action="store_true",
                        help="Only include fields whose images are on disk "
                             "(default: build the full label table from fl_data.csv)")
    parser.add_argument("--stats", action="store_true",
                        help="Print statistics only")
    args = parser.parse_args()

    df, summary_df = run_preprocessing(plates=args.plates, on_disk=args.on_disk)

    if args.stats and len(df):
        print_statistics(df, summary_df)


if __name__ == "__main__":
    main()
