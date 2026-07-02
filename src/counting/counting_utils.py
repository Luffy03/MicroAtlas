"""
Counting evaluation shared utilities.

Functions for loading ground truth, locating images, counting instances,
computing metrics, and saving results.
"""
import os
import csv
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.metrics import r2_score


# ============================================================================
#  Dataset image directory mapping
# ============================================================================
def get_image_dir_map(bbbc_root):
    """Return {dataset_name: image_dir_path} for all BBBC datasets.

    Parameters
    ----------
    bbbc_root : str or Path
        Root directory containing BBBC datasets

    Returns
    -------
    dict : {dataset_name: image_dir_path}
    """
    root = Path(bbbc_root)
    return {
        "BBBC001": root / "BBBC001" / "BBBC001_v1_images_tif" / "human_ht29_colon_cancer_1_images",
        "BBBC039": root / "BBBC039" / "images",
        "BBBC041": root / "BBBC041" / "malaria" / "images",
    }


# ============================================================================
#  Ground truth loading
# ============================================================================
def load_counts(counts_path):
    """Load ground truth counts from a *_v1_counts.txt file.

    For BBBC001 which has 2 manual counts, the GT is the average.

    Parameters
    ----------
    counts_path : str or Path
        Path to counts.txt file

    Returns
    -------
    dict : {filename: gt_count}  (filename is the image basename as in counts.txt)
    """
    counts = {}
    with open(counts_path) as f:
        lines = f.read().strip().split("\n")
    header = lines[0]
    n_cols = len(header.split("\t"))

    for line in lines[1:]:
        if not line.strip():
            continue
        parts = line.split("\t")
        fname = parts[0]
        if n_cols == 2:
            # Single manual count (BBBC039, BBBC041)
            gt = float(parts[1])
        else:
            # Two manual counts (BBBC001) — average them
            vals = [float(p) for p in parts[1:] if p]
            gt = float(np.mean(vals))
        counts[fname] = gt
    return counts


def locate_image(base_dir, filename):
    """Locate an image file by searching recursively.

    The counts.txt filenames may be in a subdirectory.
    This function handles case-insensitive matching
    and tries common extensions when the filename has none.

    Parameters
    ----------
    base_dir : str or Path
        Directory to search in
    filename : str
        Image filename from counts.txt (may omit extension)

    Returns
    -------
    Path or None
    """
    base_dir = Path(base_dir)
    if not base_dir.is_dir():
        return None
    target = filename.lower()

    # Collect all file names (once) into a {lower_name: path} map
    file_map = {}
    for f in base_dir.rglob("*"):
        if f.is_file():
            file_map[f.name.lower()] = f

    # Direct name match
    if target in file_map:
        return file_map[target]

    # If filename has no extension, try common image extensions
    ext = Path(filename).suffix
    if not ext:
        for ext_candidate in [".tif", ".TIF", ".tiff", ".TIFF", ".png", ".PNG", ".jpg", ".JPG", ".jpeg", ".JPEG"]:
            candidate = target + ext_candidate
            if candidate in file_map:
                return file_map[candidate]

    return None


# ============================================================================
#  Instance counting
# ============================================================================
def count_instances(mask):
    """Count cells in an instance segmentation mask.

    Parameters
    ----------
    mask : np.ndarray
        Instance segmentation mask, each cell has a unique integer label

    Returns
    -------
    int : number of instances (excluding background 0)
    """
    return len(np.unique(mask)) - 1


# ============================================================================
#  Metrics computation
# ============================================================================
def compute_counting_metrics(gt_counts, pred_counts):
    """Compute counting evaluation metrics.

    Parameters
    ----------
    gt_counts : list or np.ndarray of float
        Ground truth cell counts
    pred_counts : list or np.ndarray of float
        Predicted cell counts

    Returns
    -------
    dict : {MAE, RMSE, R2, Pearson_r, MPE, n_images}
        - MPE: Mean Percentage Error = mean(|pred - gt| / gt) * 100
    """
    gt = np.array(gt_counts, dtype=float)
    pred = np.array(pred_counts, dtype=float)
    n = len(gt)

    abs_error = np.abs(pred - gt)
    mae = float(np.mean(abs_error))
    rmse = float(np.sqrt(np.mean((pred - gt) ** 2)))

    # R²
    if np.var(gt) > 1e-12:
        r2 = float(r2_score(gt, pred))
    else:
        r2 = 0.0

    # Pearson r
    if n >= 3 and np.std(gt) > 1e-12 and np.std(pred) > 1e-12:
        r_val, _ = pearsonr(pred, gt)
        pearson_r = float(r_val)
    else:
        pearson_r = 0.0

    # MPE: mean percentage error
    # avoid division by zero
    safe_gt = np.where(gt == 0, np.nan, gt)
    mpe = float(np.nanmean(np.abs(pred - gt) / safe_gt * 100))

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Pearson_r": pearson_r,
        "MPE": mpe,
        "n_images": n,
    }


def compute_std_ci95(gt_counts, pred_counts):
    """Compute standard deviation and 95% confidence interval from per-image data.

    Computed metrics:
      - MAE_std / MAE_ci95 : based on per-image absolute errors |pred - gt|
      - MPE_std / MPE_ci95 : based on per-image percentage errors

    Parameters
    ----------
    gt_counts : list or np.ndarray
        Ground truth counts
    pred_counts : list or np.ndarray
        Predicted counts

    Returns
    -------
    dict : {MAE_std, MAE_ci95, MPE_std, MPE_ci95}
    """
    gt = np.array(gt_counts, dtype=float)
    pred = np.array(pred_counts, dtype=float)
    n = len(gt)

    # Absolute errors (for MAE)
    abs_errors = np.abs(pred - gt)
    mae_std = float(np.std(abs_errors, ddof=1))
    mae_ci95 = 1.96 * mae_std / np.sqrt(n)

    # Percentage errors (for MPE)
    safe_gt = np.where(gt == 0, np.nan, gt)
    pct_errors = np.abs(pred - gt) / safe_gt * 100
    mpe_std = float(np.nanstd(pct_errors, ddof=1))
    mpe_ci95 = 1.96 * mpe_std / np.sqrt(n)

    return {
        "MAE_std": mae_std,
        "MAE_ci95": mae_ci95,
        "MPE_std": mpe_std,
        "MPE_ci95": mpe_ci95,
    }


# ============================================================================
#  Result saving
# ============================================================================
def save_per_image_csv(output_dir, model_name, dataset_name, results):
    """Save per-image results as CSV.

    Parameters
    ----------
    output_dir : str or Path
        Root output directory
    model_name : str
        Model name (e.g. "cellpose4")
    dataset_name : str
        Dataset name (e.g. "BBBC001")
    results : list of dict
        Each item: {"filename": str, "gt_count": float, "pred_count": float}
    """
    out_dir = Path(output_dir) / "results_per_image"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{model_name}_{dataset_name}.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "gt_count", "pred_count"])
        for r in results:
            writer.writerow([r["filename"], r["gt_count"], r["pred_count"]])

    print(f"  Per-image CSV saved: {csv_path}")


def append_per_image_csv(output_dir, model_name, dataset_name, filename, gt_count, pred_count):
    """Append one row to per-image CSV. Creates file with header if new."""
    out_dir = Path(output_dir) / "results_per_image"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{model_name}_{dataset_name}.csv"
    file_exists = csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["filename", "gt_count", "pred_count"])
        writer.writerow([filename, gt_count, pred_count])


def save_metrics_txt(output_dir, model_name, dataset_name, metrics):
    """Save aggregate metrics as TXT (mirrors tracking _save_metrics_txt style).

    Parameters
    ----------
    output_dir : str or Path
        Root output directory
    model_name : str
        Model name
    dataset_name : str
        Dataset name
    metrics : dict
        Metrics dict from compute_counting_metrics()
    """
    out_dir = Path(output_dir) / "results_metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    txt_path = out_dir / f"{model_name}_{dataset_name}_metrics.txt"

    lines = []
    lines.append("=" * 50)
    lines.append(f"Counting Evaluation: {model_name} on {dataset_name}")
    lines.append("=" * 50)
    lines.append("")
    for key in ["MAE", "RMSE", "R2", "Pearson_r", "MPE", "n_images"]:
        val = metrics.get(key, "N/A")
        if isinstance(val, float):
            if key == "MPE":
                lines.append(f"  {key:12s} = {val:.2f}%")
            else:
                lines.append(f"  {key:12s} = {val:.4f}")
        else:
            lines.append(f"  {key:12s} = {val}")
    lines.append("")
    lines.append("=" * 50)

    with open(txt_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Metrics TXT saved: {txt_path}")




# ============================================================================
#  Image loading helper (common for BBBC datasets)
# ============================================================================

def load_bbbc_image(image_path):
    """Load a BBBC image, handling grayscale, TIFF, and PNG/JPG.

    Parameters
    ----------
    image_path : Path
        Path to the image file

    Returns
    -------
    img : np.ndarray
        2D (H,W) grayscale or 3D (H,W,3) RGB image
    """
    from PIL import Image
    ext = image_path.suffix.lower()
    if ext in (".tif", ".tiff"):
        from cellpose import io
        img = io.imread(str(image_path))
    else:
        # PNG / JPG
        img = np.array(Image.open(str(image_path)))
        if img.ndim == 3 and img.shape[2] == 4:
            img = img[:, :, :3]  # drop alpha
    return img

