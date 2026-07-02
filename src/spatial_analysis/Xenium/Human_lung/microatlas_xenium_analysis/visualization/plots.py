"""
Visualization module for the cellpose-sam Xenium analysis pipeline.

Provides publication-quality plots for:
- Segmentation overlay on tissue images
- Spatial gene expression maps
- Cell type spatial maps
- Segmentation comparison dashboard
- Expression heatmaps
- Spatial variable gene plots
- Moran's I scatter plots
- Biomarker volcano plots
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from matplotlib.patches import Polygon
from matplotlib.colors import LinearSegmentedColormap
from scipy.sparse import csr_matrix

from .. import config

logger = logging.getLogger(__name__)

# Color maps for cell types
CELL_TYPE_COLORS = {
    # Pancreas cell types (legacy)
    "Alpha": "#E63946",
    "Beta": "#457B9D",
    "Delta": "#2A9D8F",
    "PP": "#E9C46A",
    "Acinar": "#F4A261",
    "Ductal": "#264653",
    "Stellate": "#8ECAE6",
    # Lung cell types
    "Epithelial": "#E76F51",
    "Club_Ciliated": "#72B7B2",
    "Alveolar": "#A8DADC",
    "Endothelial": "#219EBC",
    "Stromal": "#8338EC",
    "Immune": "#FFB703",
    "Macrophage": "#E63946",
    "Lymphoid": "#FF79C6",
    "T_Cell": "#55A630",
    "Mast": "#9C6644",
    "Tumor_Epithelial": "#FB8500",
    # Fallback
    "Unknown": "#CCCCCC",
}

# Style settings
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 8,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
})


def _save_or_show(fig, save_path: Optional[Path] = None, dpi: int = 300):
    """Save figure to path or show if no path provided."""
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=dpi, bbox_inches="tight")
        logger.info(f"Saved figure to {save_path}")
        plt.close(fig)
    else:
        plt.show()


def _downscale_for_display(
    image: np.ndarray,
    mask: np.ndarray,
    max_dim: int = 2000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Downscale image and mask to max_dim for display.

    Uses area-based interpolation for image and nearest-neighbor for mask
    to preserve label integrity.
    """
    from skimage.transform import resize

    h, w = mask.shape[:2]
    if max(h, w) <= max_dim:
        return image, mask

    scale = max_dim / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)

    img_small = resize(
        image, (new_h, new_w), preserve_range=True, anti_aliasing=True,
        order=1  # bilinear for image
    ).astype(image.dtype)

    mask_small = resize(
        mask, (new_h, new_w), preserve_range=True, anti_aliasing=False,
        order=0  # nearest-neighbor for labels
    ).astype(mask.dtype)

    return img_small, mask_small



def plot_segmentation_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    title: str = "Cellpose-SAM Segmentation",
    alpha: float = 0.3,
    save_path: Optional[Path] = None,
    outline_color: str = "cyan",
    linewidth: float = 0.3,
) -> None:
    """Save segmentation overlay image directly, no plot frills."""
    from skimage.segmentation import mark_boundaries

    # Downscale large images
    img_small, mask_small = _downscale_for_display(image, mask, max_dim=2000)

    # Normalize image to float [0,1] for mark_boundaries
    if img_small.dtype == np.uint8:
        img_float = img_small.astype(np.float32) / 255.0
    else:
        img_float = img_small.astype(np.float32)
        img_float /= img_float.max() if img_float.max() > 0 else 1.0

    # Use mark_boundaries for vectorized boundary detection
    boundaries = mark_boundaries(
        img_float, mask_small,
        color=_parse_color(outline_color),
        outline_color=None,
        mode='subpixel',
        background_label=0,
    )

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        from PIL import Image
        img_uint8 = (np.clip(boundaries, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(img_uint8).save(str(save_path))
        logger.info(f"Saved overlay image to {save_path}")


def _parse_color(color_str: str) -> Tuple[float, float, float]:
    """Parse a matplotlib color string to RGB tuple in [0,1]."""
    import matplotlib.colors as mcolors
    return mcolors.to_rgb(color_str)


def plot_gene_expression_spatial(
    transcripts_df: pd.DataFrame,
    gene_name: str,
    cell_mask: Optional[np.ndarray] = None,
    title: Optional[str] = None,
    point_size: float = 1.0,
    cmap: str = "hot",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot spatial distribution of a single gene's transcripts.

    Args:
        transcripts_df: DataFrame with 'x_pixel', 'y_pixel', 'feature_name'.
        gene_name: Gene to visualize.
        cell_mask: Optional cell mask to show as background.
        title: Plot title.
        point_size: Size of transcript points.
        cmap: Colormap for density.
        save_path: Optional save path.

    Returns:
        Matplotlib figure.
    """
    gene_transcripts = transcripts_df[
        transcripts_df["feature_name"] == gene_name
    ]

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    # Show cell mask as background if provided
    if cell_mask is not None:
        ax.imshow(
            (cell_mask > 0).astype(np.uint8),
            cmap="gray",
            alpha=0.3,
            interpolation="none",
        )

    if len(gene_transcripts) == 0:
        ax.text(0.5, 0.5, f"No transcripts found for {gene_name}",
                transform=ax.transAxes, ha="center", va="center")
    else:
        # Plot scatter
        sc = ax.scatter(
            gene_transcripts["x_pixel"].values,
            gene_transcripts["y_pixel"].values,
            s=point_size,
            c="red",
            alpha=0.6,
            edgecolors="none",
        )
        ax.set_title(title or f"Spatial expression of {gene_name} "
                               f"({len(gene_transcripts)} transcripts)")

    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")
    ax.set_aspect("equal")
    ax.invert_yaxis()

    _save_or_show(fig, save_path)
    return fig


def plot_cell_type_map(
    cell_mask: np.ndarray,
    cell_type_df: pd.DataFrame,
    mask_labels: np.ndarray,
    title: str = "Cell Type Map",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot a spatial map of cell types.

    Uses vectorized lookup table for color assignment and automatic
    downscaling for large images.

    Args:
        cell_mask: Label array (H, W).
        cell_type_df: DataFrame with 'cell_id', 'cell_type' columns.
        mask_labels: Array mapping row index to mask label.
        title: Plot title.
        save_path: Optional save path.

    Returns:
        Matplotlib figure.
    """
    # Downscale mask for display if needed
    from skimage.transform import resize

    h, w = cell_mask.shape
    max_dim = 2000
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        cell_mask_small = resize(
            cell_mask, (new_h, new_w), preserve_range=True,
            anti_aliasing=False, order=0
        ).astype(cell_mask.dtype)
    else:
        cell_mask_small = cell_mask

    # Build label -> cell_type mapping using vectorized lookup
    label_to_type = {}
    for i, label in enumerate(mask_labels):
        if i < len(cell_type_df):
            label_to_type[int(label)] = cell_type_df.iloc[i]["cell_type"]

    # Pre-build RGB lookup table for all possible labels
    # (faster than per-label boolean indexing)
    max_label = int(cell_mask_small.max())
    rgb_lookup = np.zeros((max_label + 1, 3), dtype=np.uint8)

    unique_in_mask = np.unique(cell_mask_small)
    for label in unique_in_mask:
        if label == 0:
            continue
        cell_type = label_to_type.get(int(label), "Unknown")
        color = CELL_TYPE_COLORS.get(cell_type, "#CCCCCC")
        rgb_lookup[int(label)] = tuple(int(color[i : i + 2], 16) for i in (1, 3, 5))

    # Single vectorized lookup: O(H*W) instead of O(N_cells * H*W)
    rgb_mask = rgb_lookup[cell_mask_small]

    # Create legend handles
    unique_types = sorted(
        set(cell_type_df["cell_type"].unique()) | {"Unknown"}
    )
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(
            facecolor=CELL_TYPE_COLORS.get(ct, "#CCCCCC"),
            label=f"{ct} ({(cell_type_df['cell_type']==ct).sum()})",
        )
        for ct in unique_types
        if ct in cell_type_df["cell_type"].values
    ]

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    ax.imshow(rgb_mask, interpolation="none")
    ax.set_title(title)
    ax.axis("off")
    ax.legend(handles=legend_elements, loc="upper right",
              fontsize=6, framealpha=0.8)

    _save_or_show(fig, save_path)
    return fig


def plot_comparison_dashboard(
    images: Dict[str, np.ndarray],
    masks: Dict[str, np.ndarray],
    comparison_report: Dict[str, pd.DataFrame],
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Create a multi-panel dashboard comparing segmentation methods.

    Args:
        images: Dict of {method: image_array} (or None for no image).
        masks: Dict of {method: mask_array}.
        comparison_report: Dict from SegmentationComparator.summary_report().
        save_path: Optional save path.

    Returns:
        Matplotlib figure.
    """
    methods = list(masks.keys())
    n_methods = len(methods)

    # Determine layout
    n_rows = 2 + (n_methods + 1) // 2  # Image row + mask rows + tables

    fig = plt.figure(figsize=(5 * min(n_methods, 3), 4 * n_rows))

    row = 0

    # Row 1: Segmentation overlays
    for i, method in enumerate(methods):
        ax = fig.add_subplot(n_rows, min(n_methods, 3), i + 1)
        if method in images and images[method] is not None:
            ax.imshow(images[method])
        # Overlay outlines
        from skimage import measure
        unique_labels = np.unique(masks[method])
        unique_labels = unique_labels[unique_labels > 0]
        for label in unique_labels:
            contours = measure.find_contours(
                (masks[method] == label).astype(np.uint8), level=0.5
            )
            for contour in contours[:1]:  # First contour only
                ax.plot(contour[:, 1], contour[:, 0], "c-", linewidth=0.5, alpha=0.7)
        ax.set_title(f"{method} ({len(unique_labels)} cells)")
        ax.axis("off")

    # Row 2: Table with comparison metrics
    ax_table = fig.add_subplot(n_rows, 1, 2)
    ax_table.axis("off")

    if "cell_counts" in comparison_report:
        table_data = comparison_report["cell_counts"].round(2)
        cell_text = table_data.values.tolist()
        col_labels = table_data.columns.tolist()
        table = ax_table.table(
            cellText=cell_text,
            colLabels=col_labels,
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.5)
        ax_table.set_title(
            f"Cell Count Comparison (Dice={comparison_report.get('pairwise_overlap', pd.DataFrame()).to_string(index=False)})",
            fontsize=10,
        )

    plt.tight_layout()
    _save_or_show(fig, save_path)
    return fig


def plot_heatmap_gene_expression(
    expression_matrix: csr_matrix,
    gene_names: List[str],
    cell_annotations: Optional[pd.Series] = None,
    n_top_genes: int = 50,
    title: str = "Gene Expression Heatmap",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot a heatmap of gene expression across cells.

    Args:
        expression_matrix: CSR matrix (n_cells, n_genes).
        gene_names: List of gene names.
        cell_annotations: Optional Series of cell type labels for row colors.
        n_top_genes: Maximum number of genes to show.
        title: Plot title.
        save_path: Optional save path.

    Returns:
        Matplotlib figure.
    """
    # Select most variable genes
    if expression_matrix.shape[1] > n_top_genes:
        gene_var = np.array(expression_matrix.std(axis=0)).flatten()
        top_idx = np.argsort(gene_var)[-n_top_genes:]
        top_idx = np.sort(top_idx)
        genes = [gene_names[i] for i in top_idx]
        mat = expression_matrix[:, top_idx].toarray()
    else:
        genes = gene_names
        mat = expression_matrix.toarray()

    # Log1p transform
    mat = np.log1p(mat)

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Create heatmap
    im = ax.imshow(mat.T, aspect="auto", cmap="viridis", interpolation="nearest")

    # Annotations
    ax.set_yticks(np.arange(len(genes)))
    ax.set_yticklabels(genes, fontsize=6)
    ax.set_xlabel("Cells")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="log1p(expression)")

    _save_or_show(fig, save_path)
    return fig


def plot_spatial_variable_genes(
    svg_df: pd.DataFrame,
    n_top: int = 20,
    title: str = "Spatially Variable Genes",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot top spatially variable genes ranked by -log10(p-value).

    Args:
        svg_df: DataFrame with 'gene', 'morans_i', 'p_value' columns.
        n_top: Number of top genes to display.
        title: Plot title.
        save_path: Optional save path.

    Returns:
        Matplotlib figure.
    """
    top = svg_df.head(n_top).copy()
    top["neg_log10_p"] = -np.log10(top["p_value"].values + 1e-100)

    fig, ax = plt.subplots(1, 1, figsize=(8, max(4, n_top * 0.3)))

    colors = ["#457B9D" if v >= 0 else "#E63946" for v in top["morans_i"].values]
    bars = ax.barh(range(len(top)), top["neg_log10_p"].values, color=colors)

    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["gene"].values)
    ax.set_xlabel("-log10(p-value)")
    ax.set_title(title)
    ax.invert_yaxis()

    # Add Moran's I as text
    for i, (_, row) in enumerate(top.iterrows()):
        ax.text(
            row["neg_log10_p"] + 0.1, i,
            f"I={row['morans_i']:.3f}",
            va="center",
            fontsize=7,
        )

    _save_or_show(fig, save_path)
    return fig


def plot_morans_i_scatter(
    morans_df: pd.DataFrame,
    title: str = "Moran's I Spatial Autocorrelation",
    highlight_genes: Optional[List[str]] = None,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot a scatter of Moran's I values for all genes.

    Args:
        morans_df: DataFrame with 'gene', 'morans_i', 'p_value'.
        title: Plot title.
        highlight_genes: Optional list of genes to highlight.
        save_path: Optional save path.

    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    is_sig = morans_df["p_value"] < 0.05
    ax.scatter(
        morans_df.loc[~is_sig, "morans_i"].values,
        -np.log10(morans_df.loc[~is_sig, "p_value"].values + 1e-100),
        c="#CCCCCC", s=10, alpha=0.5, label="Not significant",
    )
    ax.scatter(
        morans_df.loc[is_sig, "morans_i"].values,
        -np.log10(morans_df.loc[is_sig, "p_value"].values + 1e-100),
        c="#E63946", s=20, alpha=0.8, label="Significant (p<0.05)",
    )

    # Highlight specific genes
    if highlight_genes:
        for gene in highlight_genes:
            row = morans_df[morans_df["gene"] == gene]
            if len(row) > 0:
                ax.annotate(
                    gene,
                    (row["morans_i"].values[0],
                     -np.log10(row["p_value"].values[0] + 1e-100)),
                    fontsize=8,
                    arrowprops=dict(arrowstyle="->", color="black", lw=0.5),
                )

    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Moran's I")
    ax.set_ylabel("-log10(p-value)")
    ax.set_title(title)
    ax.legend(fontsize=8)

    _save_or_show(fig, save_path)
    return fig


def plot_biomarker_volcano(
    de_results: pd.DataFrame,
    title: str = "Differential Expression - Biomarker Volcano Plot",
    pval_cutoff: float = 0.05,
    logfc_cutoff: float = 0.5,
    n_label: int = 15,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Create a volcano plot for differential expression results.

    Args:
        de_results: DataFrame from BiomarkerDiscovery.differential_expression().
        title: Plot title.
        pval_cutoff: Adjusted p-value significance threshold.
        logfc_cutoff: Log2 fold change significance threshold.
        n_label: Number of top genes to label.
        save_path: Optional save path.

    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    log_p = -np.log10(de_results["p_value_adj"].values + 1e-100)
    log2fc = de_results["log2fc"].values

    # Categorize points
    is_up = (de_results["p_value_adj"] < pval_cutoff) & (log2fc > logfc_cutoff)
    is_down = (de_results["p_value_adj"] < pval_cutoff) & (log2fc < -logfc_cutoff)
    is_ns = ~(is_up | is_down)

    ax.scatter(log2fc[is_ns], log_p[is_ns], c="#CCCCCC", s=5, alpha=0.5, label="NS")
    ax.scatter(log2fc[is_up], log_p[is_up], c="#E63946", s=15, alpha=0.7, label=f"Up (n={is_up.sum()})")
    ax.scatter(log2fc[is_down], log_p[is_down], c="#457B9D", s=15, alpha=0.7, label=f"Down (n={is_down.sum()})")

    # Threshold lines
    ax.axhline(-np.log10(pval_cutoff), color="gray", linestyle="--", alpha=0.5)
    ax.axvline(logfc_cutoff, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(-logfc_cutoff, color="gray", linestyle="--", alpha=0.5)

    # Label top genes
    de_sorted = de_results.sort_values("p_value_adj")
    for i in range(min(n_label, len(de_sorted))):
        row = de_sorted.iloc[i]
        ax.annotate(
            row["gene"],
            (row["log2fc"], -np.log10(row["p_value_adj"] + 1e-100)),
            fontsize=6,
            alpha=0.8,
            arrowprops=dict(arrowstyle="-", color="gray", lw=0.3),
        )

    ax.set_xlabel("log2(Fold Change)")
    ax.set_ylabel("-log10(adjusted p-value)")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="upper right")

    _save_or_show(fig, save_path)
    return fig


def plot_boundary_comparison(
    image: np.ndarray,
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    name_a: str = "Cellpose-SAM",
    name_b: str = "Xenium Official",
    title: Optional[str] = None,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Compare boundaries from two segmentation methods on the same image.

    Shows the image with method A outlines overlaid in one color and
    method B outlines in another color, side-by-side + merge overlay.

    Args:
        image: RGB uint8 image of shape (H, W, 3).
        mask_a: Label array for method A.
        mask_b: Label array for method B.
        name_a: Display name for method A.
        name_b: Display name for method B.
        title: Optional figure title.
        save_path: Optional save path.

    Returns:
        Matplotlib figure.
    """
    from skimage import measure
    from skimage.color import color_dict

    # Downscale if necessary
    img = _downscale_for_display(image)

    # Downscale masks to match
    scale = img.shape[0] / image.shape[0] if image.ndim >= 2 else 1.0
    if abs(scale - 1.0) > 0.01:
        from skimage.transform import resize
        mask_a_small = (
            resize(mask_a.astype(np.float32), img.shape[:2], order=0, preserve_range=True)
            .astype(mask_a.dtype)
        )
        mask_b_small = (
            resize(mask_b.astype(np.float32), img.shape[:2], order=0, preserve_range=True)
            .astype(mask_b.dtype)
        )
    else:
        mask_a_small = mask_a
        mask_b_small = mask_b

    # Extract outlines using skimage segmentation.mark_boundaries
    from skimage.segmentation import mark_boundaries

    # Create binary edge maps
    edge_a = _mask_to_edges(mask_a_small)
    edge_b = _mask_to_edges(mask_b_small)

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    if title:
        fig.suptitle(title, fontsize=14, y=0.98)

    # Panel 1: Method A overlay
    ax = axes[0, 0]
    overlay_a = mark_boundaries(
        img, mask_a_small, color=(0, 1, 0), outline_color=None
    )
    ax.imshow(overlay_a)
    ax.set_title(f"{name_a} ({len(np.unique(mask_a_small)) - 1} cells)", fontsize=11)
    ax.axis("off")

    # Panel 2: Method B overlay
    ax = axes[0, 1]
    overlay_b = mark_boundaries(
        img, mask_b_small, color=(1, 0, 0), outline_color=None
    )
    ax.imshow(overlay_b)
    ax.set_title(f"{name_b} ({len(np.unique(mask_b_small)) - 1} cells)", fontsize=11)
    ax.axis("off")

    # Panel 3: Combined overlay (both boundaries)
    ax = axes[1, 0]
    # Create RGB overlay: green for A, red for B, yellow for both
    edge_overlay = np.zeros_like(img, dtype=np.float32)
    edge_overlay[..., 1] = edge_a.astype(np.float32) * 255  # green channel
    edge_overlay[..., 0] = edge_b.astype(np.float32) * 255  # red channel

    combined = img.astype(np.float32) * 0.35 + edge_overlay * 0.8
    combined = np.clip(combined, 0, 255).astype(np.uint8)

    # Where both overlap, mark as yellow
    both = edge_a & edge_b
    combined[both, 0] = 255
    combined[both, 1] = 255
    combined[both, 2] = 0

    ax.imshow(combined)
    ax.set_title(f"Overlap: {name_a} (green) vs {name_b} (red)", fontsize=11)
    ax.axis("off")

    # Panel 4: Difference map
    ax = axes[1, 1]
    unique = np.zeros_like(img, dtype=np.uint8)
    # Both covered
    covered = (mask_a_small > 0) & (mask_b_small > 0)
    unique[covered] = [200, 200, 200]
    # Only in A
    only_a = (mask_a_small > 0) & (mask_b_small == 0)
    unique[only_a] = [0, 200, 0]
    # Only in B
    only_b = (mask_b_small > 0) & (mask_a_small == 0)
    unique[only_b] = [200, 0, 0]

    ax.imshow(unique)
    n_only_a = only_a.sum()
    n_only_b = only_b.sum()
    n_both = covered.sum()
    ax.set_title(
        f"Only {name_a}: {n_only_a:,}px | Only {name_b}: {n_only_b:,}px | Both: {n_both:,}px",
        fontsize=10,
    )
    ax.axis("off")

    plt.tight_layout()
    _save_or_show(fig, save_path)
    return fig


def _mask_to_edges(mask: np.ndarray) -> np.ndarray:
    """Extract edge pixels from a label mask (1=edge, 0=interior)."""
    from scipy.ndimage import binary_dilation, binary_erosion

    binary = (mask > 0).astype(np.uint8)
    # Edges = dilated - eroded (morphological boundary)
    dilated = binary_dilation(binary, iterations=1)
    eroded = binary_erosion(binary, iterations=1)
    edges = (dilated.astype(bool) ^ eroded.astype(bool))
    return edges
