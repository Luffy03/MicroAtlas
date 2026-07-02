"""
feature_extraction.py
=====================
Pure Python single-cell feature extraction for BBBC021 3-channel images.

Extracts ~106 features per cell across 3 channels (DAPI, Actin, Tubulin).
Optimized: regionprops called once, cell pixels pre-extracted, ndimage batched.

  - AreaShape:  regionprops shape descriptors (13)
  - Intensity:  batched ndimage stats + per-cell percentiles (12 x 3 + 2 = 38)
  - Texture:    Haralick GLCM on DAPI+Actin, 1 scale (13 x 2 = 26)
  - Granularity: multi-scale opening on DAPI, 5 scales (5)
  - RadialDistribution: per-channel binned stats (4 x 3 = 12)
  - Correlation: inter-channel Pearson/Mander's (12)

No CellProfiler dependency. Only numpy/scipy/skimage/mahotas.

Usage:
  python biomarker_discovery/feature_extraction.py --plate Week1_22123
  python biomarker_discovery/feature_extraction.py --all
  python biomarker_discovery/feature_extraction.py --status
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

    # Only use first 2 channels for texture
    texture_channels = [ch for ch in channel_names if ch in ("DAPI", "Actin")]
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

def compute_granularity_features(labels, dapi_image,
                                  spectrum_length=5):
    """Compute multi-scale granularity features on DAPI channel.

    Uses grey_opening at 5 scales (3,5,7,9,11) to decompose texture.
    Reduced from 10 for speed — first 5 scales capture most information.
    """
    if labels.max() == 0:
        return []

    label_ids = np.arange(1, labels.max() + 1)
    img_f = dapi_image.astype(np.float64)

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
            rec[f'Cell_Granularity_{i+1}_DAPI'] = float(gran_matrix[j, i])
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

def _worker_extract_field(args_tuple, plate_name, plate_img_dir, mask_dir):
    """Worker for multiprocessing Pool. Extracts features for one field."""
    dapi_name, actin_name, tubulin_name, field_key = args_tuple

    dapi_path = Path(plate_img_dir) / "DAPI" / dapi_name
    actin_path = Path(plate_img_dir) / "Actin" / actin_name
    tubulin_path = Path(plate_img_dir) / "Tubulin" / tubulin_name
    mask_path = Path(mask_dir) / f"{field_key}_mask.tif"

    try:
        df = extract_features_for_field(dapi_path, actin_path,
                                        tubulin_path, mask_path)
        if len(df) > 0:
            df['plate'] = plate_name
            df['field'] = field_key
        return df, field_key, None
    except Exception as e:
        return pd.DataFrame(), field_key, str(e)


# =========================================================================
# Main extraction entry point
# =========================================================================

def extract_features_for_field(dapi_path, actin_path, tubulin_path, mask_path):
    """Extract all features for a single field of view.

    Optimized: regionprops called once, cell pixels pre-extracted once,
    ndimage stats batched. All shared across feature functions.

    Args:
        dapi_path: path to DAPI channel TIFF
        actin_path: path to Actin channel TIFF
        tubulin_path: path to Tubulin channel TIFF
        mask_path: path to instance mask TIFF (uint32)

    Returns:
        pandas DataFrame (one row per cell), or empty DataFrame
    """
    # Load images
    dapi = imread(str(dapi_path)).astype(np.float64)
    actin = imread(str(actin_path)).astype(np.float64)
    tubulin = imread(str(tubulin_path)).astype(np.float64)
    labels = imread(str(mask_path)).astype(np.uint32)

    if labels.max() == 0:
        return pd.DataFrame()

    images = [dapi, actin, tubulin]
    channel_names = config.CHANNEL_NAMES

    # === Compute regionprops ONCE (eliminates 5 redundant calls) ===
    props = regionprops(labels)
    if not props:
        return pd.DataFrame()

    label_ids = np.arange(1, labels.max() + 1)

    # === Pre-extract cell pixel arrays ONCE (eliminates ~1800 array slices) ===
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
    granularity = compute_granularity_features(labels, dapi)
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
    """Extract field key from BBBC021 filename.

    E.g. G10_s1_w1BEDC2073-...tif -> G10_s1
    """
    import re
    m = re.match(r'^(.+?)_w\d+', filename)
    return m.group(1) if m else Path(filename).stem


def _build_channel_file_map(plate_name):
    """Build DAPI filename -> (Actin filename, Tubulin filename) from image CSV."""
    if not config.IMAGE_CSV.exists():
        return {}

    df = pd.read_csv(config.IMAGE_CSV)
    plate_col = "Image_Metadata_Plate_DAPI"
    if plate_col in df.columns:
        df = df[df[plate_col] == plate_name]

    mapping = {}
    for _, row in df.iterrows():
        dapi = row.get("Image_FileName_DAPI", "")
        actin = row.get("Image_FileName_Actin", "")
        tubulin = row.get("Image_FileName_Tubulin", "")
        if dapi and actin and tubulin:
            mapping[dapi] = (actin, tubulin)
    return mapping


def extract_plate_features(plate_name, model_name="cellpose4",
                           image_df=None, n_workers=8):
    """Extract features for all fields in a plate.

    Uses image CSV to correctly map DAPI files to Actin/Tubulin files,
    since each channel has a different UUID in the filename.

    Args:
        plate_name: plate identifier
        model_name: segmentation model name (for mask path)
        image_df: optional preprocessed image table
        n_workers: number of parallel workers (default 8)

    Returns:
        DataFrame with all cell features for the plate
    """
    from tqdm import tqdm

    plate_img_dir = config.IMAGES_DIR / plate_name
    mask_dir = config.masks_dir(model_name) / plate_name

    if not plate_img_dir.exists():
        print(f"  [ERROR] Image directory not found: {plate_img_dir}")
        return pd.DataFrame()
    if not mask_dir.exists():
        print(f"  [ERROR] Mask directory not found: {mask_dir}")
        return pd.DataFrame()

    # Build DAPI -> (Actin, Tubulin) file mapping from image CSV
    channel_map = _build_channel_file_map(plate_name)
    print(f"  Channel mapping: {len(channel_map)} fields from image CSV")

    # Find all DAPI files
    dapi_dir = plate_img_dir / "DAPI"
    if not dapi_dir.exists():
        print(f"  [ERROR] DAPI directory not found: {dapi_dir}")
        return pd.DataFrame()

    dapi_files = sorted(dapi_dir.glob("*.tif"))
    print(f"  Plate {plate_name}: {len(dapi_files)} fields (from DAPI enumeration)")

    # Build task list: (dapi_name, actin_name, tubulin_name, field_key)
    tasks = []
    skipped = 0
    for dapi_path in dapi_files:
        dapi_name = dapi_path.name
        field_key = _get_field_key(dapi_name)

        if dapi_name not in channel_map:
            skipped += 1
            continue

        actin_name, tubulin_name = channel_map[dapi_name]
        actin_path = plate_img_dir / "Actin" / actin_name
        tubulin_path = plate_img_dir / "Tubulin" / tubulin_name
        mask_path = mask_dir / f"{field_key}_mask.tif"

        if not actin_path.exists() or not tubulin_path.exists():
            skipped += 1
            continue
        if not mask_path.exists():
            skipped += 1
            continue

        tasks.append((dapi_name, actin_name, tubulin_name, field_key))

    if skipped > 0:
        print(f"  Skipped {skipped} fields (missing files)")
    if not tasks:
        print(f"  No valid tasks.")
        return pd.DataFrame()

    # Parallel extraction
    t_start = time.time()
    all_dfs = []
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
            results = list(tqdm(
                pool.imap_unordered(worker_fn, tasks),
                total=len(tasks),
                desc=f"  {plate_name}",
                unit="field",
            ))

        for df, field_key, err in results:
            if err is not None:
                n_errors += 1
                continue
            if len(df) > 0:
                all_dfs.append(df)
    else:
        # --- Single-process path ---
        for task in tqdm(tasks, desc=f"  {plate_name}", unit="field"):
            dapi_name, actin_name, tubulin_name, field_key = task
            dapi_path = dapi_dir / dapi_name
            actin_path = plate_img_dir / "Actin" / actin_name
            tubulin_path = plate_img_dir / "Tubulin" / tubulin_name
            mask_path = mask_dir / f"{field_key}_mask.tif"

            try:
                df = extract_features_for_field(
                    dapi_path, actin_path, tubulin_path, mask_path)
                if len(df) > 0:
                    df['plate'] = plate_name
                    df['field'] = field_key
                    all_dfs.append(df)
            except Exception as e:
                n_errors += 1

    elapsed = time.time() - t_start
    n_done = len(all_dfs)
    rate = n_done / elapsed if elapsed > 0 else 0
    print(f"  {n_done} fields in {elapsed:.1f}s ({rate:.1f} fields/s), "
          f"{n_errors} errors")

    if all_dfs:
        result = pd.concat(all_dfs, ignore_index=True)
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
        description="Extract per-cell features from BBBC021 images"
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
                print(f"  [SKIP] {plate}: features already exist")
                continue

            print(f"\nExtracting features for plate: {plate} [{model_name}]")
            df = extract_plate_features(plate, model_name=model_name,
                                        n_workers=args.workers)
            if len(df) > 0:
                save_plate_features(df, plate, model_name=model_name)


if __name__ == "__main__":
    main()
