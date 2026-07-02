"""
Post-processing utilities for segmentation masks.

Functions for merging tiled masks, filtering cells by size/shape,
converting masks to polygon boundaries, and computing cell centroids.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import ndimage
from skimage import measure

logger = logging.getLogger(__name__)


def merge_tile_masks(
    tile_masks: List[np.ndarray],
    tile_positions: List[Tuple[int, int, int, int]],
    full_shape: Tuple[int, int],
) -> np.ndarray:
    """Merge tiled segmentation masks into a single full-size mask.

    Handles overlapping regions by keeping the higher label value
    (i.e., the more confident segmentation).

    Args:
        tile_masks: List of mask arrays for each tile.
        tile_positions: List of (y_start, x_start, y_end, x_end) tuples.
        full_shape: (height, width) of the full image.

    Returns:
        Merged uint16 mask array of shape full_shape.
    """
    full_mask = np.zeros(full_shape, dtype=np.uint16)
    label_offset = 0

    for mask, (y0, x0, y1, x1) in zip(tile_masks, tile_positions):
        # Relabel the tile mask to avoid ID conflicts
        tile_labels = np.unique(mask)
        tile_labels = tile_labels[tile_labels > 0]

        if len(tile_labels) == 0:
            continue

        # Create a mapping from old labels to new unique labels
        max_existing = np.max(full_mask[y0:y1, x0:x1])
        relabeled = mask.copy()
        for old_label in tile_labels:
            relabeled[relabeled == old_label] = old_label + label_offset

        # Merge: for overlapping regions, keep the larger label
        overlap = (full_mask[y0:y1, x0:x1] > 0) & (relabeled > 0)
        if overlap.any():
            # Keep the one with larger area or higher confidence
            # Simple strategy: keep cellpose-sam result (relabeled)
            final = np.where(overlap, relabeled, full_mask[y0:y1, x0:x1])
            final = np.where(relabeled > 0, relabeled, final)
            full_mask[y0:y1, x0:x1] = final
        else:
            full_mask[y0:y1, x0:x1] = np.maximum(
                full_mask[y0:y1, x0:x1], relabeled
            )

        label_offset += len(tile_labels)

    return full_mask


def filter_masks(
    mask: np.ndarray,
    min_size: int = 15,
    max_size_fraction: float = 0.4,
) -> np.ndarray:
    """Filter masks by minimum size and maximum size fraction.

    Removes:
    - Cells smaller than min_size pixels.
    - Cells larger than max_size_fraction of total image area.

    Args:
        mask: uint32 label array.
        min_size: Minimum number of pixels per mask.
        max_size_fraction: Maximum mask size as fraction of image.

    Returns:
        Filtered uint32 mask array.
    """
    total_pixels = mask.size
    max_size = int(total_pixels * max_size_fraction)

    # Pre-compute max label for lookup table size
    max_label = mask.max()

    # Check processed labels (using np.unique for small count)
    unique_labels, counts = np.unique(mask, return_counts=True)
    label_counts = dict(zip(unique_labels, counts))

    # Background (label 0) should always remain
    valid_labels = {0}
    for lbl, cnt in label_counts.items():
        if lbl == 0:
            continue
        if cnt < min_size or cnt > max_size:
            continue
        valid_labels.add(lbl)

    # Fast vectorized filter using lookup table (O(1) per pixel)
    # Instead of np.isin (which creates large intermediate arrays)
    lookup = np.zeros(max_label + 1, dtype=mask.dtype)
    for lbl in valid_labels:
        lookup[lbl] = lbl
    filtered = lookup[mask]

    # Relabel sequentially using another lookup table
    filtered = _relabel_masks_vectorized(filtered)

    n_removed = len(unique_labels) - len(valid_labels)
    n_kept = len(valid_labels) - 1  # exclude background
    if n_removed > 0:
        logger.info(
            f"Filtered masks: removed {n_removed} cells, kept {n_kept} cells"
        )

    return filtered


def _relabel_masks_vectorized(mask: np.ndarray) -> np.ndarray:
    """Relabel mask IDs sequentially using a vectorized lookup table.

    Much faster than per-label loop for large images with many labels.

    Args:
        mask: uint32 label array (H, W).

    Returns:
        Relabeled uint32 array with sequential IDs 1, 2, 3, ...
    """
    unique_labels = np.unique(mask)
    max_label = unique_labels.max()

    # Build lookup table: index by old label, get new label
    lookup = np.zeros(max_label + 1, dtype=mask.dtype)
    new_id = 1
    for old in unique_labels:
        if old != 0:
            lookup[old] = new_id
            new_id += 1
    # Background stays 0 (lookup[0] is already 0)

    return lookup[mask]


def masks_to_polygons(
    mask: np.ndarray,
    simplify_tolerance: float = 0.5,
) -> Dict[int, np.ndarray]:
    """Convert a segmentation mask to polygon boundaries.

    Uses marching squares algorithm from scikit-image.

    Args:
        mask: uint16 label array.
        simplify_tolerance: Douglas-Peucker simplification tolerance.

    Returns:
        Dictionary mapping cell_id -> polygon vertices array (N, 2).
    """
    polygons = {}
    unique_labels = np.unique(mask)
    unique_labels = unique_labels[unique_labels > 0]

    for label in unique_labels:
        # Extract single cell mask
        single_mask = (mask == label).astype(np.uint8)

        # Find contours
        contours = measure.find_contours(single_mask, level=0.5)
        if len(contours) > 0:
            # Take the longest contour (main boundary)
            contour = max(contours, key=len)

            # Simplify if requested
            if simplify_tolerance > 0 and len(contour) > 10:
                contour = _simplify_polygon(contour, simplify_tolerance)

            # Format: (N, 2) with columns (y, x) -> convert to (x, y)
            polygon = np.fliplr(contour)
            polygons[int(label)] = polygon

    logger.info(f"Converted {len(polygons)} masks to polygons")
    return polygons


def _simplify_polygon(
    vertices: np.ndarray, tolerance: float
) -> np.ndarray:
    """Simplify a polygon using Douglas-Peucker algorithm."""
    from shapely.geometry import LineString

    line = LineString(vertices)
    simplified = line.simplify(tolerance, preserve_topology=True)
    return np.array(simplified.coords)


def compute_cell_centroids(
    mask: np.ndarray,
) -> Dict[int, Tuple[float, float]]:
    """Compute the centroid (y, x) for each cell in the mask.

    Uses vectorized ndimage operations with broadcast coordinate views
    to avoid O(N^2) overhead of regionprops on large masks.

    Args:
        mask: uint16 label array.

    Returns:
        Dictionary mapping cell_id -> (y_centroid, x_centroid).
    """
    h, w = mask.shape
    max_label = mask.max()
    if max_label == 0:
        logger.info("No cells found in mask")
        return {}

    indices = np.arange(1, max_label + 1)

    # Count pixels per label (exclude background index 0)
    counts = np.bincount(mask.ravel())[1:]

    # Sum of y and x coordinates per label.
    # np.broadcast_to creates a zero-copy view, avoiding full (H, W) allocation.
    # scipy.ndimage.sum then iterates in C over labels, accumulating per-pixel.
    y_grid = np.broadcast_to(np.arange(h, dtype=np.float64)[:, None], (h, w))
    y_sum = ndimage.sum(y_grid, labels=mask, index=indices)

    x_grid = np.broadcast_to(np.arange(w, dtype=np.float64)[None, :], (h, w))
    x_sum = ndimage.sum(x_grid, labels=mask, index=indices)

    centroids = {}
    for i, label in enumerate(indices):
        if counts[i] > 0:
            centroids[int(label)] = (
                float(y_sum[i] / counts[i]),
                float(x_sum[i] / counts[i]),
            )

    logger.info(f"Computed centroids for {len(centroids)} cells")
    return centroids


def compute_cell_areas(mask: np.ndarray) -> Dict[int, float]:
    """Compute the area (in pixels) for each cell.

    Args:
        mask: uint16 label array.

    Returns:
        Dictionary mapping cell_id -> area_in_pixels.
    """
    labels, counts = np.unique(mask, return_counts=True)
    areas = {}
    for lbl, cnt in zip(labels, counts):
        if lbl > 0:
            areas[int(lbl)] = int(cnt)
    return areas


def compute_morphology_features(
    mask: np.ndarray,
) -> Dict[int, Dict[str, float]]:
    """Compute morphological features for each cell.

    Features: area, perimeter, circularity, eccentricity, solidity.

    Args:
        mask: uint16 label array.

    Returns:
        Dictionary mapping cell_id -> {feature_name: value}.
    """
    features = {}
    props = measure.regionprops(mask)

    for prop in props:
        label = int(prop.label)
        area = prop.area
        perimeter = prop.perimeter
        # Circularity: 4*pi*area / perimeter^2 (1.0 = perfect circle)
        circularity = (4 * np.pi * area) / (perimeter**2) if perimeter > 0 else 0
        circularity = min(circularity, 1.0)

        features[label] = {
            "area": area,
            "perimeter": perimeter,
            "circularity": circularity,
            "eccentricity": prop.eccentricity,
            "solidity": prop.solidity,
            "major_axis_length": prop.major_axis_length,
            "minor_axis_length": prop.minor_axis_length,
        }

    logger.info(f"Computed morphology features for {len(features)} cells")
    return features
