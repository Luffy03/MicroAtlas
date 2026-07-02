"""
feature_aggregation.py
======================
Aggregate single-cell features into field-level and treatment-level profiles.

Pipeline:
  1. Load per-plate feature CSVs
  2. Field-level aggregation: median + MAD per feature per field
  3. Join with treatment metadata (compound, concentration, MoA)
  4. Robust z-score normalization vs DMSO controls
  5. Feature selection: remove low-variance + high-correlation features
  6. Output: treatment-level profiles for biomarker discovery

Usage:
  python biomarker_discovery/feature_aggregation.py
  python biomarker_discovery/feature_aggregation.py --plates Week1_22123 Week1_22141
"""

import argparse
import multiprocessing
import sys
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config


def load_all_features(plates=None, model_name="cellpose4"):
    """Load and concatenate feature CSVs from multiple plates.

    Args:
        plates: list of plate names, or None for all available
        model_name: segmentation model name

    Returns:
        DataFrame with all single-cell features
    """
    feat_dir = config.features_dir(model_name)
    if plates is None:
        plates = [
            f.stem.replace("features_", "").replace(".csv", "")
            for f in sorted(feat_dir.glob("features_*.csv"))
        ]

    if not plates:
        raise FileNotFoundError(
            f"No feature files found in {feat_dir}\n"
            f"Run feature_extraction.py first."
        )

    dfs = []
    for plate in plates:
        path = feat_dir / f"features_{plate}.csv"
        if not path.exists():
            print(f"  [WARN] Missing: {path}")
            continue
        df = pd.read_csv(path)
        print(f"  Loaded {plate}: {len(df):,} cells, {len(df.columns)} features")
        dfs.append(df)

    if not dfs:
        raise FileNotFoundError("No feature files loaded.")

    result = pd.concat(dfs, ignore_index=True)
    print(f"  Total: {len(result):,} cells from {len(dfs)} plates")
    return result


def get_feature_columns(df):
    """Get numeric feature column names (exclude metadata columns)."""
    meta_cols = {'cell_id', 'plate', 'field', 'compound', 'concentration',
                 'moa', 'smiles', 'is_dmso', 'replicate', 'well'}
    return [c for c in df.columns
            if c not in meta_cols and df[c].dtype in (np.float64, np.int64, float, int)]


def _worker_aggregate_field(key_group, feat_cols, min_cells):
    """Worker for multiprocessing Pool. Aggregates one field."""
    (plate, field), field_data = key_group

    if len(field_data) < min_cells:
        return None

    rec = {
        'plate': plate,
        'field': field,
        'n_cells': len(field_data),
    }

    for col in feat_cols:
        vals = field_data[col].values.astype(np.float64)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            rec[f'{col}_median'] = np.nan
            rec[f'{col}_mad'] = np.nan
            continue
        med = np.median(vals)
        rec[f'{col}_median'] = med
        rec[f'{col}_mad'] = float(np.median(np.abs(vals - med)))

    return rec


def aggregate_field_level(features_df, n_workers=8):
    """Aggregate single-cell features to field-level profiles.

    For each field: compute median + MAD of each feature across cells.
    Fields with fewer than MIN_CELLS_PER_FIELD cells are excluded.

    Returns:
        DataFrame with one row per field, ~2x feature count columns
    """
    feat_cols = get_feature_columns(features_df)
    print(f"\n  Aggregating {len(feat_cols)} features to field level "
          f"({n_workers} workers)...")

    # Pre-group data
    grouped = list(features_df.groupby(['plate', 'field']))

    # Parallel aggregation
    worker_fn = partial(
        _worker_aggregate_field,
        feat_cols=feat_cols,
        min_cells=config.MIN_CELLS_PER_FIELD,
    )

    records = []
    if n_workers > 1 and len(grouped) > 1:
        with multiprocessing.Pool(processes=n_workers) as pool:
            for rec in tqdm(
                pool.imap_unordered(worker_fn, grouped),
                total=len(grouped),
                desc="  Field aggregation",
                unit="field",
            ):
                if rec is not None:
                    records.append(rec)
    else:
        for kg in tqdm(grouped, desc="  Field aggregation", unit="field"):
            rec = worker_fn(kg)
            if rec is not None:
                records.append(rec)

    field_profiles = pd.DataFrame(records)
    print(f"  Field profiles: {len(field_profiles)} fields, "
          f"{len(get_feature_columns(field_profiles))} features")
    return field_profiles


def join_treatment_metadata(field_profiles, image_df=None):
    """Join field profiles with compound/concentration/MoA metadata.

    Args:
        field_profiles: field-level DataFrame with 'plate' and 'field' columns
        image_df: preprocessed image table from preprocess.py

    Returns:
        field_profiles with compound, concentration, moa columns added
    """
    if image_df is None:
        # Try to load from preprocessed output
        preproc_dir = config.RESULTS_DIR / "preprocess"
        table_path = preproc_dir / "image_table.csv"
        if not table_path.exists():
            print(f"  [WARN] No image table found. Run preprocess.py first.")
            # Add default metadata columns so downstream code doesn't crash
            field_profiles = field_profiles.copy()
            field_profiles['compound'] = ''
            field_profiles['concentration'] = np.nan
            field_profiles['moa'] = 'unknown'
            field_profiles['is_dmso'] = False
            field_profiles['well'] = ''
            return field_profiles
        image_df = pd.read_csv(table_path)

    # Build field -> treatment mapping from image_df
    # Map: (plate, field_key) -> compound, concentration, moa, is_dmso
    # field_key = plate_well_site (e.g. G10_s1), extracted from DAPI filename
    import re
    field_meta = {}
    for _, row in image_df.iterrows():
        plate = row.get('plate', '')
        dapi_file = row.get('Image_FileName_DAPI', '')
        if dapi_file:
            # Extract field key: G10_s1_w1UUID.tif -> G10_s1
            m = re.match(r'^(.+?)_w\d+', str(dapi_file))
            field_key = m.group(1) if m else str(dapi_file).replace('.tif', '').replace('.tiff', '')
            key = (plate, field_key)
            field_meta[key] = {
                'compound': row.get('compound', ''),
                'concentration': row.get('concentration', np.nan),
                'moa': row.get('moa', 'unknown'),
                'is_dmso': row.get('is_dmso', False),
                'well': row.get('well', ''),
            }

    # Join
    field_profiles = field_profiles.copy()
    field_profiles['compound'] = ''
    field_profiles['concentration'] = np.nan
    field_profiles['moa'] = 'unknown'
    field_profiles['is_dmso'] = False
    field_profiles['well'] = ''

    for idx in field_profiles.index:
        plate = field_profiles.at[idx, 'plate']
        field = field_profiles.at[idx, 'field']
        key = (plate, field)
        if key in field_meta:
            meta = field_meta[key]
            field_profiles.at[idx, 'compound'] = meta['compound']
            field_profiles.at[idx, 'concentration'] = meta['concentration']
            field_profiles.at[idx, 'moa'] = meta['moa']
            field_profiles.at[idx, 'is_dmso'] = meta['is_dmso']
            field_profiles.at[idx, 'well'] = meta['well']

    # Stats
    n_annotated = (field_profiles['moa'] != 'unknown').sum()
    print(f"  Joined metadata: {n_annotated}/{len(field_profiles)} fields "
          f"have MoA labels")
    return field_profiles


def robust_zscore_normalize(field_profiles):
    """Normalize features using robust z-score vs DMSO controls.

    For each feature:
      z = (x - median_DMSO) / MAD_DMSO

    Args:
        field_profiles: DataFrame with field-level features + is_dmso column

    Returns:
        DataFrame with z-scored features (original features replaced)
    """
    feat_cols = get_feature_columns(field_profiles)
    dmso_mask = field_profiles['is_dmso'] == True
    n_dmso = dmso_mask.sum()
    print(f"\n  Robust z-score normalization vs {n_dmso} DMSO fields...")

    if n_dmso < 5:
        print(f"  [WARN] Very few DMSO controls ({n_dmso}). "
              f"Falling back to global median/MAD.")
        dmso_mask = pd.Series(True, index=field_profiles.index)

    result = field_profiles.copy()

    for col in feat_cols:
        dmso_vals = field_profiles.loc[dmso_mask, col].values.astype(np.float64)
        dmso_vals = dmso_vals[np.isfinite(dmso_vals)]

        if len(dmso_vals) == 0:
            result[col] = np.nan
            continue

        dmso_med = np.median(dmso_vals)
        dmso_mad = np.median(np.abs(dmso_vals - dmso_med))

        if dmso_mad < 1e-10:
            # Avoid division by zero: use std instead
            dmso_mad = np.std(dmso_vals) + 1e-10

        all_vals = result[col].values.astype(np.float64)
        result[col] = (all_vals - dmso_med) / dmso_mad

    print(f"  Normalized {len(feat_cols)} features")
    return result


def select_features(field_profiles, variance_threshold=None,
                     corr_threshold=None):
    """Feature selection: remove low-variance and high-correlation features.

    Args:
        field_profiles: normalized DataFrame
        variance_threshold: min variance to keep (default from config)
        corr_threshold: max pairwise correlation (default from config)

    Returns:
        DataFrame with selected features only
    """
    if variance_threshold is None:
        variance_threshold = config.VARIANCE_THRESHOLD
    if corr_threshold is None:
        corr_threshold = config.CORRELATION_THRESHOLD

    feat_cols = get_feature_columns(field_profiles)
    X = field_profiles[feat_cols].values.astype(np.float64)

    # Step 1: Remove NaN-heavy columns
    nan_frac = np.isnan(X).mean(axis=0)
    valid_mask = nan_frac < 0.5
    valid_cols = [c for c, v in zip(feat_cols, valid_mask) if v]
    print(f"  After NaN filter: {len(valid_cols)}/{len(feat_cols)} features")

    # Step 2: Remove low-variance features
    X_valid = field_profiles[valid_cols].values.astype(np.float64)
    X_valid = np.nan_to_num(X_valid, nan=0.0)
    variances = np.var(X_valid, axis=0)
    var_mask = variances > variance_threshold
    selected_cols = [c for c, v in zip(valid_cols, var_mask) if v]
    print(f"  After variance filter ({variance_threshold}): "
          f"{len(selected_cols)}/{len(valid_cols)} features")

    # Step 3: Remove highly correlated features
    if len(selected_cols) > 1:
        X_sel = np.nan_to_num(field_profiles[selected_cols].values, nan=0.0)
        corr = np.corrcoef(X_sel.T)
        # Make symmetric, set diagonal to 0
        np.fill_diagonal(corr, 0)

        drop = set()
        for i in range(corr.shape[0]):
            if i in drop:
                continue
            for j in range(i + 1, corr.shape[1]):
                if j in drop:
                    continue
                if abs(corr[i, j]) > corr_threshold:
                    # Drop the one with lower variance (keep the more informative)
                    if variances[selected_cols.index(selected_cols[i])] < \
                       variances[selected_cols.index(selected_cols[j])]:
                        drop.add(i)
                    else:
                        drop.add(j)

        final_cols = [c for i, c in enumerate(selected_cols) if i not in drop]
        print(f"  After correlation filter ({corr_threshold}): "
              f"{len(final_cols)}/{len(selected_cols)} features")
    else:
        final_cols = selected_cols

    # Return selected features
    meta_cols = [c for c in field_profiles.columns if c not in feat_cols]
    result = field_profiles[meta_cols + final_cols].copy()
    print(f"\n  Final: {len(final_cols)} features selected")
    return result


def aggregate_treatment_level(field_profiles):
    """Aggregate field-level profiles to treatment level.

    For each (compound, concentration): average across replicate fields.

    Returns:
        DataFrame with one row per treatment
    """
    feat_cols = get_feature_columns(field_profiles)

    # Group by compound + concentration
    # Fill NaN concentration so groupby doesn't drop those rows
    df = field_profiles.copy()
    df['concentration'] = df['concentration'].fillna(-1)
    groups = df.groupby(['compound', 'concentration'])

    records = []
    for (cpd, conc), group in groups:
        rec = {
            'compound': cpd,
            'concentration': conc,
            'n_fields': len(group),
            'n_cells_total': group['n_cells'].sum() if 'n_cells' in group.columns else np.nan,
            'moa': group['moa'].iloc[0],
            'is_dmso': group['is_dmso'].iloc[0] if 'is_dmso' in group.columns else False,
        }
        for col in feat_cols:
            vals = group[col].values.astype(np.float64)
            vals = vals[np.isfinite(vals)]
            rec[col] = np.mean(vals) if len(vals) > 0 else np.nan
        records.append(rec)

    treatment_df = pd.DataFrame(records)
    if len(treatment_df) == 0:
        print(f"\n  Treatment profiles: 0 treatments (no metadata joined)")
        return treatment_df
    # Restore NaN concentration where it was filled with -1
    treatment_df.loc[treatment_df['concentration'] == -1, 'concentration'] = np.nan
    n_moa = (treatment_df['moa'] != 'unknown').sum() if 'moa' in treatment_df.columns else 0
    print(f"\n  Treatment profiles: {len(treatment_df)} treatments "
          f"({n_moa} with MoA)")
    return treatment_df


def run_aggregation(plates=None, model_name="cellpose4", n_workers=8):
    """Run full aggregation pipeline.

    1. Load features
    2. Aggregate to field level
    3. Join treatment metadata
    4. Robust z-score normalize
    5. Feature selection
    6. Aggregate to treatment level
    7. Save all outputs
    """
    print("=" * 60)
    print(f"Feature Aggregation Pipeline [{model_name}]")
    print("=" * 60)

    # 1. Load
    features_df = load_all_features(plates, model_name=model_name)

    # 2. Field-level aggregation
    field_profiles = aggregate_field_level(features_df, n_workers=n_workers)

    # 3. Join treatment metadata
    field_profiles = join_treatment_metadata(field_profiles)

    # 4. Normalize
    field_profiles_norm = robust_zscore_normalize(field_profiles)

    # 5. Feature selection
    field_profiles_sel = select_features(field_profiles_norm)

    # 6. Treatment-level aggregation
    treatment_profiles = aggregate_treatment_level(field_profiles_sel)

    # 7. Save
    output_dir = config.aggregation_dir(model_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    field_path = output_dir / "field_profiles.csv"
    field_profiles_sel.to_csv(field_path, index=False)
    print(f"\n  Saved: {field_path} ({len(field_profiles_sel)} fields)")

    field_norm_path = output_dir / "field_profiles_normalized.csv"
    field_profiles_norm.to_csv(field_norm_path, index=False)
    print(f"  Saved: {field_norm_path}")

    treatment_path = output_dir / "treatment_profiles.csv"
    treatment_profiles.to_csv(treatment_path, index=False)
    print(f"  Saved: {treatment_path} ({len(treatment_profiles)} treatments)")

    # DMSO profiles (for perturbation analysis)
    dmso_fields = field_profiles_norm[field_profiles_norm['is_dmso'] == True]
    dmso_path = output_dir / "dmso_profiles.csv"
    dmso_fields.to_csv(dmso_path, index=False)
    print(f"  Saved: {dmso_path} ({len(dmso_fields)} DMSO fields)")

    return field_profiles_sel, treatment_profiles


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate features into profiles"
    )
    parser.add_argument("--models", type=str, nargs="+", default=None,
                        help="Models (e.g. cellpose4 cellpose3). "
                             "Use 'all' for all 5 models.")
    parser.add_argument("--plates", type=str, nargs="+", default=None,
                        help="Specific plates to aggregate (default: all)")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of parallel workers (default: 8)")
    args = parser.parse_args()

    config.ensure_dirs()

    # Determine models
    if args.models is None:
        models_to_run = ["cellpose4"]
    elif "all" in args.models:
        models_to_run = config.MODELS
    else:
        models_to_run = args.models

    for model_name in models_to_run:
        run_aggregation(plates=args.plates, model_name=model_name,
                        n_workers=args.workers)


if __name__ == "__main__":
    main()
