"""
feature_extraction.py
=====================
Pure Python single-cell feature extraction for U2OS-Cell-Painting N-channel images.

Extracts per-cell features across all fluorescence channels defined in
config.CHANNEL_NAMES (U2OS-Cell-Painting: DNA, Mito, AGP, RNA, ER).
Optimized: regionprops called once, cell pixels pre-extracted, ndimage batched.

  - AreaShape:  regionprops shape descriptors
  - Intensity:  batched ndimage stats + per-cell percentiles, per channel
  - Texture:    Haralick GLCM on config.TEXTURE_CHANNELS
  - Granularity: multi-scale opening on the nucleus channel
  - RadialDistribution: per-channel binned stats
  - Correlation: inter-channel Pearson/Mander's (all channel pairs)

Channel images for a field are derived from its field key
("{well}_s{site}") as {channel}/{field_key}_w{idx}.tif, with idx following
config.CHANNEL_NAMES order (nucleus = w1).

No CellProfiler dependency. Only numpy/scipy/skimage/mahotas.

Usage:
  python src/morphology_profiling/feature_extraction.py --plate P015080
  python src/morphology_profiling/feature_extraction.py --all
  python src/morphology_profiling/feature_extraction.py --status
"""

import argparse
import multiprocessing
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from tifffile import imread
from skimage.measure import regionprops
from scipy.ndimage import (
    mean as ndmean, standard_deviation as ndstd,
    minimum as ndmin, maximum as ndmax, sum as ndsum,
    grey_opening,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config


# =========================================================================
# AreaShape features (~14 per cell)
# =========================================================================

def compute_areashape_features(props):
    """Compute morphological shape features from pre-computed regionprops.

    Accepts shared props to avoid redundant regionprops() calls.
    """
    if not props:
        return []

    pfx = 'Cell_AreaShape_'
    records = []
    for prop in props:
        rec = {}
        area = prop.area
        perimeter = prop.perimeter

        rec[pfx + 'Area'] = area
        rec[pfx + 'Perimeter'] = perimeter
        rec[pfx + 'Eccentricity'] = prop.eccentricity
        rec[pfx + 'Solidity'] = prop.solidity
        rec[pfx + 'Extent'] = prop.extent
        rec[pfx + 'MajorAxisLength'] = prop.major_axis_length
        rec[pfx + 'MinorAxisLength'] = prop.minor_axis_length
        rec[pfx + 'Orientation'] = prop.orientation
        rec[pfx + 'EquivalentDiameter'] = prop.equivalent_diameter

        # Compactness = 4*pi*area / perimeter^2
        if perimeter > 0:
            rec[pfx + 'Compactness'] = 4 * np.pi * area / (perimeter ** 2)
        else:
            rec[pfx + 'Compactness'] = 0.0

        # FormFactor = 4*pi*area / perimeter^2
        rec[pfx + 'FormFactor'] = rec[pfx + 'Compactness']

        # MaxFeretDiameter approximation
        major = prop.major_axis_length
        minor = prop.minor_axis_length
        rec[pfx + 'MaxFeretDiameter'] = max(major, minor)

        # Euler number
        rec[pfx + 'EulerNumber'] = 1  # single connected component

        records.append(rec)

    return records


# =========================================================================
# Intensity features (~15 per channel x 3 channels = 45)
# =========================================================================

def compute_intensity_features(labels, images, channel_names, props,
                                cell_pixels, label_ids):
    """Compute per-channel intensity statistics for each cell.

    Batched ndimage stats (7 calls total) + per-cell percentiles from
    pre-extracted cell_pixels.
    """
    if not props:
        return []

    # Find boundaries for edge intensity (once for all cells)
    from skimage.segmentation import find_boundaries
    boundaries = find_boundaries(labels, mode='inner')
    edge_labels = labels * boundaries
    has_edges = np.any(boundaries)

    # Batch compute all ndimage stats (7 full-image scans total)
    batch_obj = []
    batch_edge = []
    for image in images:
        obj_mean = ndmean(image, labels, label_ids)
        obj_std = ndstd(image, labels, label_ids)
        obj_min_val = ndmin(image, labels, label_ids)
        obj_max_val = ndmax(image, labels, label_ids)
        obj_sum_val = ndsum(image, labels, label_ids)
        batch_obj.append((obj_mean, obj_std, obj_min_val, obj_max_val, obj_sum_val))
        if has_edges:
            edge_mean = ndmean(image, edge_labels, label_ids)
            edge_std = ndstd(image, edge_labels, label_ids)
            batch_edge.append((edge_mean, edge_std))

    records = []
    for idx, prop in enumerate(props):
        rec = {}
        for ch_idx, ch_name in enumerate(channel_names):
            pfx_i = 'Cell_Intensity_'
            means, stds, mins, maxs, sums = batch_obj[ch_idx]

            obj_mean = float(means[idx])
            obj_std = float(stds[idx])
            obj_min = float(mins[idx])
            obj_max = float(maxs[idx])
            obj_sum = float(sums[idx])

            rec[pfx_i + f'Mean_{ch_name}'] = obj_mean
            rec[pfx_i + f'Std_{ch_name}'] = obj_std
            rec[pfx_i + f'Min_{ch_name}'] = obj_min
            rec[pfx_i + f'Max_{ch_name}'] = obj_max
            rec[pfx_i + f'IntegratedIntensity_{ch_name}'] = obj_sum
            rec[pfx_i + f'Range_{ch_name}'] = obj_max - obj_min

            # Edge intensity (lookup from batched result)
            if has_edges:
                e_means, e_stds = batch_edge[ch_idx]
                rec[pfx_i + f'MeanEdge_{ch_name}'] = float(e_means[idx])
                rec[pfx_i + f'StdEdge_{ch_name}'] = float(e_stds[idx])
            else:
                rec[pfx_i + f'MeanEdge_{ch_name}'] = obj_mean
                rec[pfx_i + f'StdEdge_{ch_name}'] = 0.0

            # Per-pixel percentiles (from pre-extracted cell_pixels)
            obj_pix = cell_pixels[idx][ch_idx]
            if len(obj_pix) > 0:
                med = np.median(obj_pix)
                rec[pfx_i + f'LowerQuartile_{ch_name}'] = float(np.percentile(obj_pix, 25))
                rec[pfx_i + f'Median_{ch_name}'] = float(med)
                rec[pfx_i + f'MAD_{ch_name}'] = float(np.median(np.abs(obj_pix - med)))
                rec[pfx_i + f'UpperQuartile_{ch_name}'] = float(np.percentile(obj_pix, 75))
            else:
                rec[pfx_i + f'LowerQuartile_{ch_name}'] = obj_mean
                rec[pfx_i + f'Median_{ch_name}'] = obj_mean
                rec[pfx_i + f'MAD_{ch_name}'] = 0.0
                rec[pfx_i + f'UpperQuartile_{ch_name}'] = obj_mean

        # Location (same for all channels)
        rec['Cell_Location_Center_X'] = float(prop.centroid[1])
        rec['Cell_Location_Center_Y'] = float(prop.centroid[0])
        records.append(rec)

    return records


# =========================================================================
# Texture features (~13 x 2 channels = 26)
# =========================================================================

def compute_texture_features(images, channel_names, props,
                              scales=(8,), gray_levels=256):
    """Compute Haralick texture features via mahotas GLCM.

    Applied to DAPI and Actin channels (Tubulin typically noisy for Haralick).
    Single scale (8) for speed. Uses shared props.
    """
    import mahotas
    if not props:
        return []

    # Texture channels are configurable (nucleus + a bright organelle channel)
    texture_channels = [ch for ch in channel_names if ch in config.TEXTURE_CHANNELS]
    ch_indices = [channel_names.index(ch) for ch in texture_channels]

    records = []
    for idx, prop in enumerate(props):
        rec = {}
        # Skip very small objects (< 30 pixels)
        if prop.area < 30:
            for ch in texture_channels:
                for s in scales:
                    for feat_idx in range(13):
                        rec[f'Cell_Texture_{config.HARALICK_FEATURES[feat_idx]}_{ch}_s{s}'] = 0.0
            records.append(rec)
            continue

        for ch, ch_idx in zip(texture_channels, ch_indices):
            # Need 2D crop for GLCM — extract directly from image
            obj_img = images[ch_idx][prop.slice]
            crop = obj_img.copy()

            # Normalize to [0, gray_levels-1]
            vmin, vmax = crop.min(), crop.max()
            if vmax > vmin:
                crop = ((crop - vmin) * (gray_levels - 1) / (vmax - vmin)).astype(np.uint8)
            else:
                crop = np.zeros_like(crop, dtype=np.uint8)

            # Apply mask (zero out pixels outside cell)
            crop[~prop.image] = 0

            for scale in scales:
                try:
                    feats = mahotas.features.haralick(
                        crop, distance=scale, return_mean=True,
                        ignore_zeros=True,
                    )
                except Exception:
                    feats = np.zeros(13)

                for feat_idx in range(min(13, len(feats))):
                    fname = config.HARALICK_FEATURES[feat_idx] if feat_idx < len(config.HARALICK_FEATURES) else f"Haralick_{feat_idx}"
                    rec[f'Cell_Texture_{fname}_{ch}_s{scale}'] = float(feats[feat_idx])

        records.append(rec)

    return records


# =========================================================================
# Granularity features (~10, DAPI only)
# =========================================================================

def compute_granularity_features(labels, nucleus_image, nucleus_name="DNA",
                                  spectrum_length=5):
    """Compute multi-scale granularity features on the nucleus channel.

    Uses grey_opening at 5 scales (3,5,7,9,11) to decompose texture.
    Reduced from 10 for speed — first 5 scales capture most information.
    """
    if labels.max() == 0:
        return []

    label_ids = np.arange(1, labels.max() + 1)
    img_f = nucleus_image.astype(np.float64)

    # Compute opened images at each scale
    opened_images = []
    for i in range(1, spectrum_length + 1):
        size = 2 * i + 1
        opened = grey_opening(img_f, size=size)
        opened_images.append(opened)

    # Per-object granularity spectrum
    obj_start_means = ndmean(img_f, labels, label_ids)
    obj_current_means = obj_start_means.copy()
    n_obj = len(label_ids)
    gran_matrix = np.zeros((n_obj, spectrum_length))

    for i in range(spectrum_length):
        obj_new_means = ndmean(opened_images[i], labels, label_ids)
        gran_matrix[:, i] = (
            (obj_current_means - obj_new_means) * 100
            / np.maximum(obj_start_means, np.finfo(float).eps)
        )
        obj_current_means = obj_new_means

    records = []
    for j in range(n_obj):
        rec = {}
        for i in range(spectrum_length):
            rec[f'Cell_Granularity_{i+1}_{nucleus_name}'] = float(gran_matrix[j, i])
        records.append(rec)

    return records


# =========================================================================
# Radial distribution features (~4 x 3 channels = 12)
# =========================================================================

def compute_radial_features(images, channel_names, props, cell_pixels,
                             n_bins=4):
    """Compute radial intensity distribution per cell per channel.

    For each cell, bin pixels by distance from centroid (inner to outer rings).
    Report fraction of total intensity in each bin.
    Uses shared props + pre-extracted cell_pixels.
    """
    if not props:
        return []

    records = []
    for idx, prop in enumerate(props):
        rec = {}
        cy, cx = prop.centroid[0], prop.centroid[1]
        coords = np.argwhere(prop.image)  # (y, x) relative to slice

        if len(coords) == 0:
            for ch in channel_names:
                for b in range(n_bins):
                    rec[f'Cell_RadialDistribution_{ch}_Bin{b+1}'] = 0.0
            records.append(rec)
            continue

        # Absolute coordinates
        y_abs = coords[:, 0] + prop.slice[0].start
        x_abs = coords[:, 1] + prop.slice[1].start

        # Distance from centroid
        dist = np.sqrt((y_abs - cy) ** 2 + (x_abs - cx) ** 2)
        max_dist = dist.max()
        if max_dist == 0:
            max_dist = 1.0

        # Bin by normalized distance [0, 1]
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(dist / max_dist, bin_edges[1:-1])

        for ch_idx, ch_name in enumerate(channel_names):
            obj_pixels = cell_pixels[idx][ch_idx]
            total_intensity = obj_pixels.sum()

            if total_intensity > 0:
                for b in range(n_bins):
                    mask_b = (bin_indices == b)
                    bin_intensity = obj_pixels[mask_b].sum()
                    rec[f'Cell_RadialDistribution_{ch_name}_Bin{b+1}'] = (
                        float(bin_intensity / total_intensity)
                    )
            else:
                for b in range(n_bins):
                    rec[f'Cell_RadialDistribution_{ch_name}_Bin{b+1}'] = 0.0

        records.append(rec)

    return records


# =========================================================================
# Correlation features (~12)
# =========================================================================

def compute_correlation_features(images, channel_names, props, cell_pixels):
    """Compute inter-channel correlation (Pearson, Mander's) per cell.

    All pairs of channels. Vectorized numpy for speed (no per-cell pearsonr).
    Uses shared props + pre-extracted cell_pixels.
    """
    from itertools import combinations

    if not props:
        return []

    pairs = list(combinations(range(len(channel_names)), 2))

    records = []
    for ci, prop in enumerate(props):
        rec = {}
        for (i, j) in pairs:
            ch1 = channel_names[i]
            ch2 = channel_names[j]
            pix1 = cell_pixels[ci][i]
            pix2 = cell_pixels[ci][j]

            if len(pix1) < 3:
                rec[f'Cell_Correlation_Pearson_{ch1}_{ch2}'] = 0.0
                rec[f'Cell_Correlation_Manders1_{ch1}_{ch2}'] = 0.0
                rec[f'Cell_Correlation_Manders2_{ch1}_{ch2}'] = 0.0
                rec[f'Cell_Correlation_Overlap_{ch1}_{ch2}'] = 0.0
                continue

            # Vectorized Pearson (avoid scipy.stats.pearsonr overhead)
            m1 = pix1.mean()
            m2 = pix2.mean()
            d1 = pix1 - m1
            d2 = pix2 - m2
            num = np.sum(d1 * d2)
            den = np.sqrt(np.sum(d1 * d1) * np.sum(d2 * d2))
            r = num / den if den > 0 else 0.0
            rec[f'Cell_Correlation_Pearson_{ch1}_{ch2}'] = (
                float(r) if np.isfinite(r) else 0.0
            )

            # Mander's coefficients
            dot = np.sum(pix1 * pix2)
            norm1 = np.sqrt(np.sum(pix1 ** 2))
            norm2 = np.sqrt(np.sum(pix2 ** 2))
            if norm1 > 0 and norm2 > 0:
                rec[f'Cell_Correlation_Manders1_{ch1}_{ch2}'] = float(dot / (norm1 ** 2))
                rec[f'Cell_Correlation_Manders2_{ch1}_{ch2}'] = float(dot / (norm2 ** 2))
                rec[f'Cell_Correlation_Overlap_{ch1}_{ch2}'] = float(
                    dot / np.sqrt(norm1 ** 2 * norm2 ** 2)
                )
            else:
                rec[f'Cell_Correlation_Manders1_{ch1}_{ch2}'] = 0.0
                rec[f'Cell_Correlation_Manders2_{ch1}_{ch2}'] = 0.0
                rec[f'Cell_Correlation_Overlap_{ch1}_{ch2}'] = 0.0

        records.append(rec)

    return records


# =========================================================================
# Multiprocessing worker (module-level for pickling)
# =========================================================================

def _field_channel_paths(plate_img_dir, field_key):
    """Return channel image paths for a field, in config.CHANNEL_NAMES order.

    Channel files follow the normalized naming {channel}/{field_key}_w{idx}.tif
    where idx = 1..N follows config.CHANNEL_NAMES order (nucleus = w1).
    """
    plate_img_dir = Path(plate_img_dir)
    return [plate_img_dir / ch / f"{field_key}_w{idx}.tif"
            for idx, ch in enumerate(config.CHANNEL_NAMES, start=1)]


def _worker_extract_field(field_key, plate_name, plate_img_dir, mask_dir):
    """Worker for multiprocessing Pool. Extracts features for one field."""
    channel_paths = _field_channel_paths(plate_img_dir, field_key)
    mask_path = Path(mask_dir) / f"{field_key}_mask.tif"

    try:
        df = extract_features_for_field(channel_paths, mask_path)
        if len(df) > 0:
            df['plate'] = plate_name
            df['field'] = field_key
        return df, field_key, None
    except Exception as e:
        return pd.DataFrame(), field_key, str(e)


# =========================================================================
# Main extraction entry point
# =========================================================================

def extract_features_for_field(channel_paths, mask_path):
    """Extract all features for a single field of view.

    Optimized: regionprops called once, cell pixels pre-extracted once,
    ndimage stats batched. All shared across feature functions.

    Args:
        channel_paths: list of channel image TIFF paths, in
                       config.CHANNEL_NAMES order (nucleus channel first)
        mask_path: path to instance mask TIFF (uint32)

    Returns:
        pandas DataFrame (one row per cell), or empty DataFrame
    """
    # Load channel images (in config.CHANNEL_NAMES order)
    images = []
    for p in channel_paths:
        img = imread(str(p)).astype(np.float64)
        if img.ndim > 2:
            img = img[..., 0]
        images.append(img)
    labels = imread(str(mask_path)).astype(np.uint32)

    if labels.max() == 0:
        return pd.DataFrame()

    channel_names = config.CHANNEL_NAMES
    nucleus_idx = channel_names.index(config.NUCLEUS_CHANNEL)

    # === Compute regionprops ONCE (eliminates redundant calls) ===
    props = regionprops(labels)
    if not props:
        return pd.DataFrame()

    label_ids = np.arange(1, labels.max() + 1)

    # === Pre-extract cell pixel arrays ONCE ===
    cell_pixels = []  # cell_pixels[cell_idx][ch_idx] = 1D float64 array
    for prop in props:
        ch_pixels = []
        for image in images:
            ch_pixels.append(image[prop.slice][prop.image].astype(np.float64))
        cell_pixels.append(ch_pixels)

    # Extract all feature groups (shared props + cell_pixels)
    areashape = compute_areashape_features(props)
    intensity = compute_intensity_features(labels, images, channel_names,
                                            props, cell_pixels, label_ids)
    texture = compute_texture_features(images, channel_names, props)
    granularity = compute_granularity_features(
        labels, images[nucleus_idx], config.NUCLEUS_CHANNEL)
    radial = compute_radial_features(images, channel_names, props, cell_pixels)
    correlation = compute_correlation_features(images, channel_names,
                                                props, cell_pixels)

    # Merge all features per cell
    n_cells = len(areashape)
    rows = []
    for i in range(n_cells):
        row = {}
        row['cell_id'] = i + 1
        for feat_list in [areashape, intensity, texture, granularity,
                          radial, correlation]:
            if feat_list and i < len(feat_list):
                row.update(feat_list[i])
        rows.append(row)

    return pd.DataFrame(rows)


def _get_field_key(filename):
    """Extract field key from a normalized channel filename.

    E.g. A01_s1_w1.tif -> A01_s1
    """
    import re
    m = re.match(r'^(.+?)_w\d+', filename)
    return m.group(1) if m else Path(filename).stem


# =========================================================================
# Field-level checkpoint helpers (resume interrupted plate extraction)
# =========================================================================

def _checkpoint_dir(model_name, plate_name):
    """Per-plate checkpoint directory holding one file per completed field."""
    return config.features_dir(model_name) / "_checkpoint" / plate_name


def _field_done(ckpt_dir, field_key):
    """A field is done if it has a result CSV or an empty-field marker."""
    return (ckpt_dir / f"{field_key}.csv").exists() or \
        (ckpt_dir / f"{field_key}.empty").exists()


def _write_field_checkpoint(ckpt_dir, field_key, df):
    """Persist one field's result so it is not recomputed on resume.

    Non-empty fields are written atomically to {field_key}.csv; fields with
    no cells get a zero-byte {field_key}.empty marker.
    """
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if df is not None and len(df) > 0:
        final = ckpt_dir / f"{field_key}.csv"
        tmp = ckpt_dir / f"{field_key}.csv.tmp"
        df.to_csv(tmp, index=False)
        tmp.replace(final)  # atomic rename -> no half-written CSV on crash
    else:
        (ckpt_dir / f"{field_key}.empty").touch()


def _load_checkpoint(ckpt_dir, field_keys=None):
    """Concatenate completed field CSVs from the checkpoint directory.

    If field_keys is given, only those fields' CSVs are read (used to load
    just the resumed-from-disk fields, avoiding re-reading files this run
    already holds in memory). Otherwise every *.csv in the dir is read.
    """
    if not ckpt_dir.exists():
        return pd.DataFrame()
    if field_keys is not None:
        files = [ckpt_dir / f"{k}.csv" for k in field_keys]
    else:
        files = sorted(ckpt_dir.glob("*.csv"))
    dfs = []
    for f in files:
        if not f.exists():
            continue
        try:
            dfs.append(pd.read_csv(f))
        except Exception:
            continue
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def _clear_checkpoint(ckpt_dir):
    """Remove a plate's checkpoint directory after the final CSV is saved."""
    if not ckpt_dir.exists():
        return
    for f in ckpt_dir.glob("*"):
        try:
            f.unlink()
        except OSError:
            pass
    try:
        ckpt_dir.rmdir()
    except OSError:
        pass


def _expected_plate_fields(plate_name):
    """Field keys a plate should have, per the authoritative image table.

    Returns None when the table is unavailable so callers fall back to the
    persisted masks.
    """
    table_path = config.RESULTS_DIR / "preprocess" / "image_table.csv"
    if not table_path.exists():
        return None
    try:
        df = pd.read_csv(table_path, usecols=["plate", "field"])
    except Exception:
        return None
    return set(df.loc[df["plate"] == plate_name, "field"].astype(str))


def plate_features_complete(plate_name, model_name):
    """Check whether a saved feature CSV covers every field of its plate.

    Images are deleted per plate, so at check time only the masks and the
    image table survive. A plate whose images were partially removed before
    extraction yields a CSV covering a handful of fields; comparing field
    coverage against the mask count catches that, whereas testing for file
    existence does not.

    Returns:
        (complete, n_fields_in_csv, n_fields_expected)
    """
    path = config.features_dir(model_name) / f"features_{plate_name}.csv"
    expected = _expected_plate_fields(plate_name)
    if expected:
        n_expected = len(expected)
    else:
        mask_dir = config.masks_dir(model_name) / plate_name
        n_expected = len(list(mask_dir.glob("*_mask.tif"))) \
            if mask_dir.exists() else 0
    if not path.exists():
        return False, 0, n_expected
    try:
        got = pd.read_csv(path, usecols=["field"])["field"].astype(str).nunique()
    except Exception:
        return False, 0, n_expected
    return got >= n_expected, got, n_expected


def verify_completeness(models):
    """Audit every saved feature CSV against the fields it should cover.

    Returns the number of incomplete plates (0 == clean).
    """
    print("\n=== Feature Completeness Audit ===")
    n_bad = 0
    n_csvs = 0
    for model_name in models:
        feat_dir = config.features_dir(model_name)
        csvs = sorted(feat_dir.glob("features_*.csv")) if feat_dir.exists() else []
        if not csvs:
            continue
        n_csvs += len(csvs)
        print(f"\n  {model_name}:")
        for f in csvs:
            plate = f.stem.replace("features_", "")
            complete, got, expect = plate_features_complete(plate, model_name)
            if not complete:
                n_bad += 1
            print(f"    {plate}: {got}/{expect} fields "
                  f"[{'OK' if complete else 'INCOMPLETE'}]")
    if not n_csvs:
        print("\n  No feature CSVs found; nothing to audit.")
    elif n_bad:
        print(f"\n  {n_bad}/{n_csvs} plate(s) INCOMPLETE. Re-download those "
              f"plates (download.py --plate P) and re-extract with --force.")
    else:
        print(f"\n  All {n_csvs} plate(s) complete.")
    return n_bad


def extract_plate_features(plate_name, model_name="cellpose4",
                           image_df=None, n_workers=8, force=False):
    """Extract features for all fields in a plate.

    Fields are enumerated from the nucleus-channel directory using the
    normalized naming scheme {channel}/{well}_s{site}_w{idx}.tif. A field
    is kept only if all N channels and its mask exist.

    Field-level resume: each completed field is written to a per-plate
    checkpoint directory; an interrupted run continues from the fields that
    have not been processed yet. Pass force=True to recompute all fields.

    Args:
        plate_name: plate identifier
        model_name: segmentation model name (for mask path)
        image_df: optional preprocessed image table (unused, kept for API)
        n_workers: number of parallel workers (default 8)
        force: ignore and clear any existing checkpoint, recompute all fields

    Returns:
        DataFrame with all cell features for the plate
    """
    from tqdm import tqdm

    plate_img_dir = config.IMAGES_DIR / plate_name
    mask_dir = config.masks_dir(model_name) / plate_name
    ckpt_dir = _checkpoint_dir(model_name, plate_name)
    if force:
        _clear_checkpoint(ckpt_dir)

    if not plate_img_dir.exists():
        print(f"  [ERROR] Image directory not found: {plate_img_dir}")
        return pd.DataFrame()
    if not mask_dir.exists():
        print(f"  [ERROR] Mask directory not found: {mask_dir}")
        return pd.DataFrame()

    # Enumerate fields from the nucleus-channel directory
    nucleus_dir = plate_img_dir / config.NUCLEUS_CHANNEL
    if not nucleus_dir.exists():
        print(f"  [ERROR] Nucleus channel dir not found: {nucleus_dir}")
        return pd.DataFrame()

    nucleus_files = sorted(nucleus_dir.glob("*.tif"))
    print(f"  Plate {plate_name}: {len(nucleus_files)} fields "
          f"(from {config.NUCLEUS_CHANNEL} enumeration)")

    # The enumeration above trusts the disk; the image table is authoritative.
    # A plate whose images were partially deleted looks perfectly normal here,
    # so report the shortfall instead of silently featurizing the remainder.
    expected_fields = _expected_plate_fields(plate_name)
    if expected_fields:
        found = {_get_field_key(p.name) for p in nucleus_files}
        n_absent = len(expected_fields - found)
        if n_absent:
            print(f"  [ERROR] {plate_name}: {n_absent}/{len(expected_fields)} "
                  f"fields have no {config.NUCLEUS_CHANNEL} image on disk. "
                  f"The plate images are incomplete (deleted or never "
                  f"extracted); run download.py --plate {plate_name} first, "
                  f"otherwise this plate's features will be truncated.")

    # Build task list of field keys; keep only fields with all channels + mask
    tasks = []
    skipped = 0
    resumed = 0
    prior_csv_fields = []  # resumed non-empty fields to re-read from disk
    for nuc_path in nucleus_files:
        field_key = _get_field_key(nuc_path.name)
        channel_paths = _field_channel_paths(plate_img_dir, field_key)
        mask_path = mask_dir / f"{field_key}_mask.tif"

        if not all(p.exists() for p in channel_paths):
            skipped += 1
            continue
        if not mask_path.exists():
            skipped += 1
            continue

        # Field-level resume: skip fields already checkpointed
        if _field_done(ckpt_dir, field_key):
            resumed += 1
            if (ckpt_dir / f"{field_key}.csv").exists():
                prior_csv_fields.append(field_key)
            continue

        tasks.append(field_key)

    if skipped > 0:
        print(f"  Skipped {skipped} fields (missing channels or mask)")
    if resumed > 0:
        print(f"  Resuming: {resumed} fields already done "
              f"(checkpoint), {len(tasks)} remaining")
    if not tasks:
        if resumed > 0:
            # All valid fields already checkpointed -> assemble final result
            result = _load_checkpoint(ckpt_dir, prior_csv_fields)
            if len(result) > 0:
                print(f"  Total: {len(result):,} cells, "
                      f"{len(result.columns)} features (from checkpoint)")
            return result
        print(f"  No valid tasks.")
        return pd.DataFrame()

    # Parallel extraction
    t_start = time.time()
    new_dfs = []  # this run's freshly computed fields (kept in memory)
    n_errors = 0

    if n_workers > 1 and len(tasks) > 1:
        # --- Multiprocessing path ---
        worker_fn = partial(
            _worker_extract_field,
            plate_name=plate_name,
            plate_img_dir=str(plate_img_dir),
            mask_dir=str(mask_dir),
        )
        with multiprocessing.Pool(processes=n_workers) as pool:
            for df, field_key, err in tqdm(
                pool.imap_unordered(worker_fn, tasks),
                total=len(tasks),
                desc=f"  {plate_name}",
                unit="field",
            ):
                if err is not None:
                    n_errors += 1
                    continue
                # Persist each completed field immediately (crash-safe resume)
                _write_field_checkpoint(ckpt_dir, field_key, df)
                if len(df) > 0:
                    new_dfs.append(df)
    else:
        # --- Single-process path ---
        for field_key in tqdm(tasks, desc=f"  {plate_name}", unit="field"):
            channel_paths = _field_channel_paths(plate_img_dir, field_key)
            mask_path = mask_dir / f"{field_key}_mask.tif"

            try:
                df = extract_features_for_field(channel_paths, mask_path)
                if len(df) > 0:
                    df['plate'] = plate_name
                    df['field'] = field_key
                _write_field_checkpoint(ckpt_dir, field_key, df)
                if len(df) > 0:
                    new_dfs.append(df)
            except Exception as e:
                n_errors += 1

    elapsed = time.time() - t_start
    n_new = len(new_dfs)
    rate = n_new / elapsed if elapsed > 0 else 0
    print(f"  {n_new} new fields in {elapsed:.1f}s ({rate:.1f} fields/s), "
          f"{n_errors} errors")

    # Assemble the full plate: freshly computed (in memory) + resumed (on disk).
    # Only resumed fields are re-read, so a fresh run does no extra CSV parsing.
    parts = list(new_dfs)
    if prior_csv_fields:
        prior_df = _load_checkpoint(ckpt_dir, prior_csv_fields)
        if len(prior_df) > 0:
            parts.append(prior_df)
    if parts:
        result = pd.concat(parts, ignore_index=True)
        print(f"  Total: {len(result):,} cells, {len(result.columns)} features")
        return result
    else:
        print(f"  No features extracted.")
        return pd.DataFrame()


def save_plate_features(df, plate_name, model_name="cellpose4"):
    """Save feature DataFrame to CSV."""
    output_path = config.features_dir(model_name) / f"features_{plate_name}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"  Saved: {output_path} ({len(df):,} rows, {len(df.columns)} cols)")
    return output_path


def list_status():
    """Show feature extraction status per model."""
    print("\n=== Feature Extraction Status ===")

    for model_name in config.MODELS:
        feat_dir = config.features_dir(model_name)
        if not feat_dir.exists():
            continue

        import pandas as pd
        csvs = sorted(feat_dir.glob("features_*.csv"))
        if not csvs:
            continue

        total_rows = 0
        for f in csvs:
            n_rows = sum(1 for _ in open(f)) - 1
            total_rows += n_rows
        print(f"  {model_name}: {len(csvs)} plates, {total_rows:,} cells")


def main():
    parser = argparse.ArgumentParser(
        description="Extract per-cell features from U2OS-Cell-Painting images"
    )
    parser.add_argument("--models", type=str, nargs="+", default=None,
                        help="Models (e.g. cellpose4 cellpose3). "
                             "Use 'all' for all 5 models.")
    parser.add_argument("--plate", type=str, default=None,
                        help="Extract features for a single plate")
    parser.add_argument("--plates", type=str, nargs="+", default=None,
                        help="Extract features for multiple plates")
    parser.add_argument("--all", action="store_true",
                        help="Extract features for all plates with masks")
    parser.add_argument("--force", action="store_true",
                        help="Re-extract even if features file exists")
    parser.add_argument("--status", action="store_true",
                        help="Show extraction status")
    parser.add_argument("--verify", action="store_true",
                        help="Audit field coverage of every saved feature CSV "
                             "and exit non-zero if any plate is truncated")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of parallel workers (default: 8)")
    args = parser.parse_args()

    config.ensure_dirs()

    if args.status:
        list_status()
        return

    # Determine models
    if args.models is None:
        models_to_run = ["cellpose4"]
    elif "all" in args.models:
        models_to_run = config.MODELS
    else:
        models_to_run = args.models

    if args.verify:
        sys.exit(1 if verify_completeness(models_to_run) else 0)

    for model_name in models_to_run:
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")

        feat_dir = config.features_dir(model_name)
        mask_dir = config.masks_dir(model_name)

        plates = []
        if args.plate:
            plates = [args.plate]
        elif args.plates:
            plates = args.plates
        elif args.all:
            if not mask_dir.exists():
                print(f"No masks found for {model_name}. Run segment.py first.")
                continue
            plates = [d.name for d in sorted(mask_dir.iterdir())
                      if d.is_dir() and list(d.glob("*_mask.tif"))]

        for plate in plates:
            output_path = feat_dir / f"features_{plate}.csv"
            if output_path.exists() and not args.force:
                complete, got, expect = plate_features_complete(plate, model_name)
                if complete:
                    print(f"  [SKIP] {plate}: features already exist")
                else:
                    print(f"  [INCOMPLETE] {plate}: features cover only "
                          f"{got}/{expect} fields. Re-download the plate "
                          f"images, then re-run with --force. Existing CSV "
                          f"left untouched.")
                continue

            print(f"\nExtracting features for plate: {plate} [{model_name}]")
            df = extract_plate_features(plate, model_name=model_name,
                                        n_workers=args.workers,
                                        force=args.force)
            if len(df) > 0:
                save_plate_features(df, plate, model_name=model_name)
                # Final CSV written -> drop the per-field checkpoint
                _clear_checkpoint(_checkpoint_dir(model_name, plate))


if __name__ == "__main__":
    main()
