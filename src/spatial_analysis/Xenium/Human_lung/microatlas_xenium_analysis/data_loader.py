"""
Data loading module for Xenium spatial transcriptomics data.

Provides unified access to Xenium output data including:
- Morphology and H&E images
- Transcript coordinates and gene identities
- Cell metadata and boundaries
- Gene expression matrix (cells x genes)
- Gene panel information
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
import pandas as pd
import tifffile

from . import config

logger = logging.getLogger(__name__)


# =============================================================================
# Image Loading
# =============================================================================

def load_morphology_image(
    use_focus: bool = True,
    crop_roi: Optional[Dict[str, int]] = None,
) -> np.ndarray:
    """Load the Xenium morphology image as a 3-channel uint8 RGB image.

    The morphology image captures tissue structure using Xenium's staining.
    If use_focus=True, loads the focus-merged image (recommended).
    Otherwise loads the raw z-stack (first plane).

    Args:
        use_focus: If True, load the focus-merged morphology image.
        crop_roi: Optional dict with keys {x_start, y_start, width, height}.
                  If provided, only that region is loaded.

    Returns:
        RGB image as uint8 array of shape (H, W, 3).
    """
    if use_focus:
        img = _load_focus_morphology(crop_roi)
    else:
        img = _load_raw_morphology(crop_roi)

    logger.info(f"Loaded morphology image: shape={img.shape}, dtype={img.dtype}")
    return img


def _load_focus_morphology(crop_roi: Optional[Dict[str, int]] = None) -> np.ndarray:
    """Load the focus-merged morphology image (multi-channel)."""
    focus_file = config.MORPHOLOGY_FOCUS_DIR / "morphology_focus_0000.ome.tif"
    if not focus_file.exists():
        logger.warning(f"Focus image not found at {focus_file}, falling back to raw")
        return _load_raw_morphology(crop_roi)

    # Shape: (C, H, W) where C=4 channels, dtype=uint16
    img = tifffile.imread(str(focus_file))
    logger.info(f"Raw focus morphology shape: {img.shape}, dtype={img.dtype}")

    # Convert to uint8 RGB
    # Typically channel 0 = DAPI/nuclear, combine channels for RGB
    img = _convert_to_rgb(img, crop_roi)
    return img


def _load_raw_morphology(crop_roi: Optional[Dict[str, int]] = None) -> np.ndarray:
    """Load the raw morphology z-stack (first plane)."""
    if not config.MORPHOLOGY_IMAGE_PATH.exists():
        raise FileNotFoundError(
            f"Morphology image not found: {config.MORPHOLOGY_IMAGE_PATH}"
        )

    # Read only the first z-plane (shape: (H, W))
    # The file has 12 z-planes as separate pages
    with tifffile.TiffFile(str(config.MORPHOLOGY_IMAGE_PATH)) as tf:
        # Page 0 is the first z-plane
        z_plane = tf.pages[0].asarray()
        logger.info(f"Raw morphology z-plane shape: {z_plane.shape}, dtype={z_plane.dtype}")

    # Apply cropping if requested
    if crop_roi is not None:
        z_plane = _apply_crop(z_plane, crop_roi)

    # Convert single channel uint16 to 3-channel uint8
    img = _normalize_to_uint8(z_plane)
    img = np.stack([img, img, img], axis=-1)
    return img


def load_he_image(crop_roi: Optional[Dict[str, int]] = None) -> np.ndarray:
    """Load the H&E stained image as a 3-channel uint8 RGB array.

    Args:
        crop_roi: Optional dict with keys {x_start, y_start, width, height}.

    Returns:
        RGB image as uint8 array of shape (H, W, 3).
    """
    if not config.HE_IMAGE_PATH.exists():
        raise FileNotFoundError(f"H&E image not found: {config.HE_IMAGE_PATH}")

    # H&E image shape: (H, W, 3), dtype=uint8
    img = tifffile.imread(str(config.HE_IMAGE_PATH))
    logger.info(f"Raw H&E image shape: {img.shape}, dtype={img.dtype}")

    if img.ndim == 3 and img.shape[-1] == 3:
        # Already RGB uint8
        if crop_roi is not None:
            img = _apply_crop(img, crop_roi)
    elif img.ndim == 2:
        # Single channel - broadcast to RGB
        img = _normalize_to_uint8(img)
        img = np.stack([img, img, img], axis=-1)
        if crop_roi is not None:
            img = _apply_crop(img, crop_roi)
    else:
        raise ValueError(f"Unexpected H&E image shape: {img.shape}")

    logger.info(f"Loaded H&E image: shape={img.shape}, dtype={img.dtype}")
    return img


def _convert_to_rgb(
    img: np.ndarray, crop_roi: Optional[Dict[str, int]] = None
) -> np.ndarray:
    """Convert multi-channel uint16 image to 3-channel uint8 RGB.

    For focus morphology with 4 channels:
    - If C=4, use channels 0,1,2 as RGB (or appropriate mapping)
    - If C=1, broadcast to 3 channels
    """
    if img.ndim == 3 and img.shape[0] in (3, 4):
        # Multi-channel: take first 3 channels as RGB
        rgb = img[:3].transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)
    elif img.ndim == 3 and img.shape[0] == 1:
        rgb = np.stack([img[0], img[0], img[0]], axis=-1)
    elif img.ndim == 2:
        rgb = np.stack([img, img, img], axis=-1)
    else:
        raise ValueError(f"Cannot convert image of shape {img.shape} to RGB")

    # Normalize each channel independently to uint8
    rgb_uint8 = np.zeros_like(rgb, dtype=np.uint8)
    for c in range(rgb.shape[-1]):
        rgb_uint8[..., c] = _normalize_to_uint8(rgb[..., c])

    # Apply cropping
    if crop_roi is not None:
        rgb_uint8 = _apply_crop(rgb_uint8, crop_roi)

    return rgb_uint8


def _normalize_to_uint8(channel: np.ndarray) -> np.ndarray:
    """Normalize a single-channel image to uint8 [0, 255] using percentile clipping."""
    if channel.dtype == np.uint8:
        return channel
    if channel.dtype == np.uint16:
        # Use percentile-based normalization for better contrast
        low, high = np.percentile(channel, [1, 99])
        if high > low:
            channel = np.clip((channel.astype(np.float32) - low) / (high - low) * 255, 0, 255)
        else:
            channel = np.zeros_like(channel, dtype=np.float32)
    else:
        # For float types, assume [0, 1] range
        channel = np.clip(channel * 255, 0, 255)
    return channel.astype(np.uint8)


def _apply_crop(img: np.ndarray, crop_roi: Dict[str, int]) -> np.ndarray:
    """Apply a cropping region of interest to an image."""
    xs = crop_roi["x_start"]
    ys = crop_roi["y_start"]
    w = crop_roi["width"]
    h = crop_roi["height"]

    if img.ndim == 2:
        return img[ys : ys + h, xs : xs + w]
    elif img.ndim == 3:
        return img[ys : ys + h, xs : xs + w, :]
    else:
        raise ValueError(f"Cannot crop image of shape {img.shape}")


# =============================================================================
# Transcript Loading
# =============================================================================

def load_transcripts(
    min_qv: float = 0.0,
    usecols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Load transcript data from the Xenium output.

    Columns:
        transcript_id, cell_id, overlaps_nucleus, feature_name (gene),
        x_location, y_location, z_location, qv, fov_name,
        nucleus_distance, codeword_index

    Args:
        min_qv: Minimum transcript quality value threshold.
        usecols: List of columns to load (None = all columns).

    Returns:
        DataFrame with transcript information.
    """
    if not config.TRANSCRIPTS_PATH.exists():
        raise FileNotFoundError(f"Transcripts file not found: {config.TRANSCRIPTS_PATH}")

    logger.info(f"Loading transcripts from {config.TRANSCRIPTS_PATH}")
    df = pd.read_parquet(config.TRANSCRIPTS_PATH, columns=usecols)

    if min_qv > 0 and "qv" in df.columns:
        n_before = len(df)
        df = df[df["qv"] >= min_qv]
        logger.info(f"Filtered transcripts by qv>={min_qv}: {n_before} -> {len(df)}")

    logger.info(f"Loaded {len(df)} transcripts with {df['feature_name'].nunique()} genes")
    return df


# =============================================================================
# Cell Information Loading
# =============================================================================

def load_cells_df() -> pd.DataFrame:
    """Load cell summary information.

    Returns:
        DataFrame with columns: cell_id, x_centroid, y_centroid,
        transcript_counts, cell_area, nucleus_area, etc.
    """
    if not config.CELLS_PATH.exists():
        raise FileNotFoundError(f"Cells file not found: {config.CELLS_PATH}")

    df = pd.read_csv(config.CELLS_PATH, compression="gzip")
    logger.info(f"Loaded {len(df)} cells")
    return df


def load_cell_boundaries() -> pd.DataFrame:
    """Load cell boundary polygon data.

    Returns:
        DataFrame with columns: cell_id, vertex_x, vertex_y
        Each cell has multiple vertices defining its boundary polygon.
    """
    if not config.CELL_BOUNDARIES_PATH.exists():
        raise FileNotFoundError(f"Cell boundaries not found: {config.CELL_BOUNDARIES_PATH}")

    df = pd.read_parquet(config.CELL_BOUNDARIES_PATH)
    logger.info(f"Loaded cell boundaries: {len(df)} vertices for {df['cell_id'].nunique()} cells")
    return df


def rasterize_cell_boundaries(
    output_shape: Optional[Tuple[int, int]] = None,
    crop_roi: Optional[Dict[str, int]] = None,
    label_start: int = 1,
) -> np.ndarray:
    """Rasterize Xenium cell boundary polygons into a label mask.

    Converts polygon vertex data (cell_boundaries.parquet) into a dense
    segmentation mask, where each unique cell_id gets a unique integer label.

    IMPORTANT: Xenium boundary coordinates are in **micrometers** (not pixels).
    They are converted to pixels using config.PIXEL_SIZE_UM before rasterizing.
    The mask dimensions must match the morphology image.

    Args:
        output_shape: (H, W) of the output mask. If None, inferred from
                      max vertex coordinates.
        crop_roi: Optional crop region {x_start, y_start, width, height}.
                  If set, only boundaries within the ROI are rasterized.
        label_start: Starting label value (default 1).

    Returns:
        uint32 label mask of shape (H, W).
    """
    from skimage.draw import polygon as sk_polygon

    df = load_cell_boundaries()

    # Convert vertex coordinates from micrometers to pixels
    pixel_size = config.PIXEL_SIZE_UM
    df["vertex_x"] = (df["vertex_x"] / pixel_size).astype(np.int32)
    df["vertex_y"] = (df["vertex_y"] / pixel_size).astype(np.int32)

    # Determine mask dimensions from max pixel coordinates
    h_max = int(df["vertex_y"].max()) + 1
    w_max = int(df["vertex_x"].max()) + 1
    logger.info(f"Xenium boundaries span ({h_max}, {w_max}) pixels")

    # Apply cropping (in pixel coordinates now)
    if crop_roi is not None:
        xs, ys = crop_roi["x_start"], crop_roi["y_start"]
        w, h = crop_roi["width"], crop_roi["height"]

        # Filter vertices within ROI
        df = df[
            (df["vertex_x"] >= xs)
            & (df["vertex_x"] < xs + w)
            & (df["vertex_y"] >= ys)
            & (df["vertex_y"] < ys + h)
        ].copy()
        # Shift to local ROI coordinates
        df["vertex_x"] -= xs
        df["vertex_y"] -= ys

        # After cropping, some cells may have incomplete boundaries
        # Keep only cells with at least 3 vertices (valid polygon)
        cell_vertex_counts = df.groupby("cell_id").size()
        valid_cells = cell_vertex_counts[cell_vertex_counts >= 3].index
        df = df[df["cell_id"].isin(valid_cells)]

        h_out, w_out = h, w
    else:
        h_out, w_out = h_max, w_max

    if output_shape is not None:
        h_out, w_out = output_shape

    logger.info(
        f"Rasterizing {df['cell_id'].nunique()} cells onto "
        f"mask of size ({h_out}, {w_out})..."
    )

    # Assign sequential labels to each cell_id
    cell_ids = df["cell_id"].unique()
    cell_id_to_label = {cid: i + label_start for i, cid in enumerate(cell_ids)}
    n_cells = len(cell_ids)

    # Group vertices by cell
    grouped = df.groupby("cell_id")

    # Create empty mask
    mask = np.zeros((h_out, w_out), dtype=np.uint32)

    # Rasterize each cell's polygon
    # Use batch processing: track progress every 10K cells
    batch_idx = 0
    BATCH_LOG = 10_000

    for cid, group in grouped:
        label = cell_id_to_label[cid]
        vertices_x = group["vertex_x"].values
        vertices_y = group["vertex_y"].values

        # Clip to image bounds
        vertices_x = np.clip(vertices_x, 0, w_out - 1)
        vertices_y = np.clip(vertices_y, 0, h_out - 1)

        # Draw filled polygon
        rr, cc = sk_polygon(vertices_y, vertices_x, shape=(h_out, w_out))
        mask[rr, cc] = label

        batch_idx += 1
        if batch_idx % BATCH_LOG == 0:
            logger.debug(f"  Rasterized {batch_idx}/{n_cells} cells...")

    logger.info(f"Rasterization complete: {n_cells} cells rasterized")
    return mask


def load_nucleus_boundaries() -> pd.DataFrame:
    """Load nucleus boundary polygon data."""
    if not config.NUCLEUS_BOUNDARIES_PATH.exists():
        raise FileNotFoundError(f"Nucleus boundaries not found: {config.NUCLEUS_BOUNDARIES_PATH}")

    df = pd.read_parquet(config.NUCLEUS_BOUNDARIES_PATH)
    logger.info(f"Loaded nucleus boundaries: {len(df)} vertices")
    return df


# =============================================================================
# Gene Expression Matrix Loading
# =============================================================================

def load_expression_matrix() -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load the gene-cell expression matrix from the Xenium H5 file.

    The file stores the matrix in CSC (Compressed Sparse Column) format:
        - shape: (n_features, n_barcodes)
        - data: non-zero expression values
        - indices: feature (gene) indices
        - indptr: column pointers

    Returns:
        Tuple of (matrix_csc, cell_ids, gene_names):
            matrix_csc: scipy.sparse.csc_matrix of shape (n_cells, n_genes)
            cell_ids: numpy array of cell ID strings
            gene_names: list of gene name strings
    """
    if not config.EXPRESSION_MATRIX_PATH.exists():
        raise FileNotFoundError(
            f"Expression matrix not found: {config.EXPRESSION_MATRIX_PATH}"
        )

    from scipy.sparse import csc_matrix

    with h5py.File(str(config.EXPRESSION_MATRIX_PATH), "r") as f:
        m = f["matrix"]
        shape = m["shape"][:]
        data = m["data"][:]
        indices = m["indices"][:]
        indptr = m["indptr"][:]
        barcodes = m["barcodes"][:].astype(str)
        gene_names = [n.decode() for n in m["features"]["name"][:]]

    # shape = (n_features, n_barcodes)
    # Transpose to (n_cells, n_genes) for convenience
    mat = csc_matrix((data, indices, indptr), shape=shape).tocsc()
    # Transpose to cells x genes
    mat = mat.T.tocsr()

    logger.info(
        f"Loaded expression matrix: {mat.shape} "
        f"({len(barcodes)} cells x {len(gene_names)} genes), "
        f"{mat.nnz} non-zero entries"
    )
    return mat, barcodes, gene_names


def load_gene_panel() -> pd.DataFrame:
    """Load the gene panel information.

    Returns:
        DataFrame with gene info (gene_id, gene_name, feature_type, genome).
    """
    if not config.GENE_PANEL_PATH.exists():
        raise FileNotFoundError(f"Gene panel not found: {config.GENE_PANEL_PATH}")

    with open(config.GENE_PANEL_PATH, "r") as f:
        data = json.load(f)

    # The payload contains the gene entries
    genes = data.get("payload", data if isinstance(data, list) else [])
    if isinstance(genes, dict):
        # Sometimes it's a dict with 'features' key
        genes = genes.get("features", [list(genes.keys())])

    df = pd.DataFrame(genes)
    logger.info(f"Loaded gene panel: {len(df)} genes")
    return df


def load_experiment_metadata() -> Dict:
    """Load experiment metadata from experiment.xenium JSON."""
    if not config.EXPERIMENT_PATH.exists():
        logger.warning(f"Experiment metadata not found: {config.EXPERIMENT_PATH}")
        return {}

    with open(config.EXPERIMENT_PATH, "r") as f:
        metadata = json.load(f)
    return metadata


# =============================================================================
# Utility: Pixel <-> Micrometer Conversion
# =============================================================================

def um_to_pixels(coords_um: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Convert micrometer coordinates to pixel coordinates."""
    return coords_um / config.PIXEL_SIZE_UM


def pixels_to_um(coords_px: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Convert pixel coordinates to micrometer coordinates."""
    return coords_px * config.PIXEL_SIZE_UM


# =============================================================================
# Convenience: Load All Data
# =============================================================================

def load_all(
    load_images: bool = True,
    crop_roi: Optional[Dict[str, int]] = None,
    min_qv: float = 20,
) -> Dict:
    """Load all Xenium data into a single dictionary.

    Args:
        load_images: If True, load morphology and H&E images.
        crop_roi: Optional cropping region for images.
        min_qv: Minimum transcript quality score.

    Returns:
        Dictionary with keys: transcripts, cells, cell_boundaries,
        nucleus_boundaries, expression_matrix, cell_ids, gene_names,
        morphology_image, he_image, experiment_metadata, metrics.
    """
    data = {}

    # Always load non-image data
    data["transcripts"] = load_transcripts(min_qv=min_qv)
    data["cells"] = load_cells_df()
    data["cell_boundaries"] = load_cell_boundaries()
    data["nucleus_boundaries"] = load_nucleus_boundaries()
    data["expression_matrix"], data["cell_ids"], data["gene_names"] = (
        load_expression_matrix()
    )
    data["gene_panel"] = load_gene_panel()
    data["experiment_metadata"] = load_experiment_metadata()

    # Load images (expensive)
    if load_images:
        data["morphology_image"] = load_morphology_image(crop_roi=crop_roi)
        data["he_image"] = load_he_image(crop_roi=crop_roi)

    return data
