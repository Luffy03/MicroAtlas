"""
Evaluation metrics for comparing segmentation results.

Provides metrics for:
- Overlap-based: Dice coefficient, IoU
- Boundary-based: Hausdorff distance, average contour distance
- Cell-level: detection agreement, matching accuracy
- Distributional: cell count, area distribution, gene detection
"""

import logging
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import directed_hausdorff
from scipy.sparse import csr_matrix
from skimage import measure
from tqdm import tqdm

logger = logging.getLogger(__name__)


def dice_coefficient(
    mask_a: np.ndarray, mask_b: np.ndarray
) -> float:
    """Compute the Dice coefficient between two binary masks.

    Dice = 2 * |A & B| / (|A| + |B|)

    Args:
        mask_a: Binary mask (0/1) or label array (0=background).
        mask_b: Binary mask (0/1) or label array (0=background).

    Returns:
        Dice coefficient in [0, 1].
    """
    binary_a = (mask_a > 0).astype(bool)
    binary_b = (mask_b > 0).astype(bool)

    intersection = np.logical_and(binary_a, binary_b).sum()
    sum_a = binary_a.sum()
    sum_b = binary_b.sum()

    if sum_a + sum_b == 0:
        return 1.0 if sum_a == 0 and sum_b == 0 else 0.0

    return 2.0 * intersection / (sum_a + sum_b)


def iou_score(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Compute Intersection over Union (IoU / Jaccard Index).

    IoU = |A & B| / |A | B|

    Args:
        mask_a: Binary mask (0/1) or label array.
        mask_b: Binary mask (0/1) or label array.

    Returns:
        IoU in [0, 1].
    """
    binary_a = (mask_a > 0).astype(bool)
    binary_b = (mask_b > 0).astype(bool)

    intersection = np.logical_and(binary_a, binary_b).sum()
    union = np.logical_or(binary_a, binary_b).sum()

    if union == 0:
        return 1.0

    return intersection / union


def hausdorff_distance(
    polygon_a: np.ndarray, polygon_b: np.ndarray
) -> float:
    """Compute the Hausdorff distance between two cell boundary polygons.

    Args:
        polygon_a: (N, 2) array of (x, y) vertices.
        polygon_b: (M, 2) array of (x, y) vertices.

    Returns:
        Hausdorff distance (in pixels).
    """
    d_ab = directed_hausdorff(polygon_a, polygon_b)[0]
    d_ba = directed_hausdorff(polygon_b, polygon_a)[0]
    return max(d_ab, d_ba)


def average_contour_distance(
    polygon_a: np.ndarray, polygon_b: np.ndarray
) -> float:
    """Compute the average contour distance between two polygons.

    For each point in A, finds the nearest distance to B, and vice versa,
    then averages.

    Args:
        polygon_a: (N, 2) array.
        polygon_b: (M, 2) array.

    Returns:
        Average contour distance.
    """
    from scipy.spatial import KDTree

    tree_a = KDTree(polygon_a)
    tree_b = KDTree(polygon_b)

    dists_ab = tree_b.query(polygon_a)[0]
    dists_ba = tree_a.query(polygon_b)[0]

    return float(np.mean(np.concatenate([dists_ab, dists_ba])))


def cell_matching(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    iou_threshold: float = 0.5,
) -> Dict[str, any]:
    """Match cells between two segmentation masks by IoU overlap.

    For each cell in mask_a, finds the best-matching cell in mask_b
    based on highest IoU.

    Performance note: uses per-cell bincount (not all-pairs), so this
    runs in O(H*W) instead of O(N_a * N_b).

    Args:
        mask_a: Label array (first segmentation).
        mask_b: Label array (second segmentation).
        iou_threshold: Minimum IoU to consider a match.

    Returns:
        Dict with:
            - matches: list of (label_a, label_b, iou) tuples
            - unmatched_a: list of labels in a with no match
            - unmatched_b: list of labels in b with no match
            - precision: fraction of mask_a cells with a match
            - recall: fraction of mask_b cells with a match
            - f1: harmonic mean of precision and recall
    """
    props_a = measure.regionprops(mask_a)
    props_b = measure.regionprops(mask_b)

    # Build area lookup for mask_b cells
    area_b = {p.label: p.area for p in props_b}

    t0 = time.time()
    matches = []
    matched_b = set()

    for prop in tqdm(props_a, desc="Cell matching", unit="cell"):
        label_a = prop.label
        area_a = prop.area

        # Extract bounding box of this cell
        minr, minc, maxr, maxc = prop.bbox

        # Get mask_b labels at this cell's pixels (within bounding box)
        cell_mask = mask_a[minr:maxr, minc:maxc] == label_a
        b_overlap = mask_b[minr:maxr, minc:maxc][cell_mask]

        if len(b_overlap) == 0:
            continue

        # bincount: O(cell_pixels) — count overlapping pixels per mask_b label
        max_label = int(b_overlap.max())
        if max_label == 0:
            continue
        counts = np.bincount(b_overlap.astype(np.int64), minlength=max_label + 1)
        counts[0] = 0  # ignore background

        # Find best matching mask_b cell
        best_iou = 0.0
        best_label_b = 0
        for label_b in np.flatnonzero(counts):
            if label_b in matched_b:
                continue
            overlap = counts[label_b]
            iou_val = overlap / (area_a + area_b.get(label_b, 0) - overlap)
            if iou_val > best_iou:
                best_iou = iou_val
                best_label_b = int(label_b)

        if best_iou >= iou_threshold and best_label_b > 0:
            matches.append((label_a, best_label_b, best_iou))
            matched_b.add(best_label_b)

    # Unmatched cells
    all_labels_a = {p.label for p in props_a}
    all_labels_b = {p.label for p in props_b}
    matched_a_labels = {m[0] for m in matches}
    matched_b_labels = {m[1] for m in matches}

    unmatched_a = all_labels_a - matched_a_labels
    unmatched_b = all_labels_b - matched_b_labels

    # Statistics
    precision = len(matches) / max(len(all_labels_a), 1)
    recall = len(matches) / max(len(all_labels_b), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-10)

    elapsed = time.time() - t0
    logger.info(
        f"Cell matching: {len(matches)}/{len(all_labels_a)} matched, "
        f"{len(unmatched_a)}/{len(all_labels_a)} unmatched (a), "
        f"{len(unmatched_b)}/{len(all_labels_b)} unmatched (b), "
        f"F1={f1:.3f} "
        f"({elapsed:.0f}s)"
    )

    return {
        "matches": matches,
        "unmatched_a": sorted(unmatched_a),
        "unmatched_b": sorted(unmatched_b),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def compare_cell_counts(
    masks_dict: Dict[str, np.ndarray],
) -> pd.DataFrame:
    """Compare cell counts across multiple segmentation methods.

    Args:
        masks_dict: Dict of {method_name: mask_array}.

    Returns:
        DataFrame with method, cell_count, density (cells/1000px^2).
    """
    results = []
    for method, mask in masks_dict.items():
        n_cells = len(np.unique(mask)) - 1  # exclude background
        total_area_px = mask.size
        density = n_cells / (total_area_px / 1e6)  # cells per million pixels

        results.append({
            "method": method,
            "cell_count": n_cells,
            "density_per_mpix": round(density, 2),
        })

    df = pd.DataFrame(results)
    logger.info(f"Cell counts:\n{df.to_string()}")
    return df


def compare_area_distribution(
    masks_dict: Dict[str, np.ndarray],
) -> pd.DataFrame:
    """Compare cell area distributions across methods.

    Args:
        masks_dict: Dict of {method_name: mask_array}.

    Returns:
        DataFrame with method, mean_area, std_area, median_area, etc.
    """
    results = []
    for method, mask in masks_dict.items():
        props = measure.regionprops(mask)
        areas = [p.area for p in props]

        if len(areas) > 0:
            results.append({
                "method": method,
                "n_cells": len(areas),
                "mean_area": np.mean(areas),
                "std_area": np.std(areas),
                "median_area": np.median(areas),
                "min_area": np.min(areas),
                "max_area": np.max(areas),
                "total_area_covered": np.sum(areas),
            })

    df = pd.DataFrame(results)
    logger.info(f"Area distribution:\n{df.to_string()}")
    return df


def compare_gene_detection(
    expression_matrices: Dict[str, csr_matrix],
    gene_names: List[str],
) -> pd.DataFrame:
    """Compare gene detection sensitivity across methods.

    Args:
        expression_matrices: Dict of {method_name: csr_matrix (n_cells, n_genes)}.
        gene_names: List of gene names.

    Returns:
        DataFrame with method, mean_genes_per_cell, median_genes_per_cell,
        mean_transcripts_per_cell, n_cells.
    """
    results = []
    for method, mat in expression_matrices.items():
        if mat.shape[0] == 0:
            continue

        # Count genes detected per cell
        genes_per_cell = np.array(mat.getnnz(axis=1)).flatten()

        # Total transcripts per cell
        transcripts_per_cell = np.array(mat.sum(axis=1)).flatten()

        results.append({
            "method": method,
            "n_cells": mat.shape[0],
            "mean_genes_per_cell": round(float(np.mean(genes_per_cell)), 2),
            "median_genes_per_cell": float(np.median(genes_per_cell)),
            "mean_transcripts_per_cell": round(float(np.mean(transcripts_per_cell)), 1),
            "median_transcripts_per_cell": float(np.median(transcripts_per_cell)),
        })

    df = pd.DataFrame(results)
    logger.info(f"Gene detection:\n{df.to_string()}")
    return df
