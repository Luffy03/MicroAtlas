"""
Generate all Jupyter notebooks for the Cellpose-SAM Xenium analysis pipeline.

Run this script to create the 6 analysis notebooks:
    01_data_exploration
    02_cellpose_sam_segmentation
    03_transcript_mapping
    04_cell_typing
    05_segmentation_comparison
    06_biomarker_analysis
"""

import json
import os
from pathlib import Path

NOTEBOOK_DIR = Path(__file__).parent / "notebooks"
NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)


def make_cell(source_lines, cell_type="code", outputs=None):
    """Create a notebook cell dict."""
    if isinstance(source_lines, str):
        source_lines = [source_lines]
    # Ensure each line ends with newline
    source = [s if s.endswith("\n") else s + "\n" for s in source_lines]

    cell = {
        "cell_type": cell_type,
        "metadata": {},
        "source": source,
    }
    if cell_type == "code":
        cell["outputs"] = outputs or []
        cell["execution_count"] = None
    return cell


def make_md(source_lines):
    """Create a markdown cell."""
    return make_cell(source_lines, cell_type="markdown")


def make_notebook(cells):
    """Create a complete notebook dict."""
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0",
            },
        },
        "cells": cells,
    }


def save_notebook(name, notebook):
    """Save a notebook to disk."""
    path = NOTEBOOK_DIR / name
    with open(path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    print(f"Created: {path}")


# =============================================================================
# Notebook 1: Data Exploration
# =============================================================================

def notebook_01_data_exploration():
    cells = [
        make_md([
            "# Xenium Human Pancreas Data Exploration",
            "",
            "This notebook provides an overview of the Xenium spatial transcriptomics dataset.",
            "We visualize the morphology image, examine transcript distributions, and explore",
            "the gene expression matrix.",
        ]),
        make_cell([
            "import sys; sys.path.insert(0, '..')",
            "import numpy as np",
            "import pandas as pd",
            "import matplotlib.pyplot as plt",
            "import seaborn as sns",
            "",
            "from cpsam_xenium_analysis import config, data_loader as dl",
            "",
            "%matplotlib inline",
            "plt.rcParams.update({'figure.dpi': 150, 'font.size': 8})",
        ]),
        make_md(["## 1. Load Experiment Metadata"]),
        make_cell([
            "metadata = dl.load_experiment_metadata()",
            "print(f'Region: {metadata.get(\"region_name\", \"N/A\")}')",
            "print(f'Panel: {metadata.get(\"panel_name\", \"N/A\")}')",
            "print(f'Cells detected: {metadata.get(\"num_cells\", \"N/A\"):,}')",
            "print(f'Pixel size: {metadata.get(\"pixel_size\", \"N/A\")} um/px')",
        ]),
        make_md(["## 2. Load Morphology Image (Cropped for Speed)"]),
        make_cell([
            "crop = config.CROP_ROI  # 4000x4000 region for demo",
            "morph_img = dl.load_morphology_image(use_focus=True, crop_roi=crop)",
            "print(f'Morphology image shape: {morph_img.shape}, dtype={morph_img.dtype}')",
        ]),
        make_cell([
            "fig, ax = plt.subplots(1, 1, figsize=(10, 10))",
            "ax.imshow(morph_img)",
            "ax.set_title('Morphology Image (Focus-Merged, Cropped ROI)')",
            "ax.axis('off')",
            "plt.show()",
        ]),
        make_md(["## 3. Load H&E Image"]),
        make_cell([
            "he_img = dl.load_he_image(crop_roi=crop)",
            "print(f'H&E image shape: {he_img.shape}, dtype={he_img.dtype}')",
        ]),
        make_cell([
            "fig, ax = plt.subplots(1, 1, figsize=(10, 10))",
            "ax.imshow(he_img)",
            "ax.set_title('H&E Image (Cropped ROI)')",
            "ax.axis('off')",
            "plt.show()",
        ]),
        make_md(["## 4. Transcript Statistics"]),
        make_cell([
            "transcripts = dl.load_transcripts(min_qv=20)",
            "print(f'Total transcripts: {len(transcripts):,}')",
            "print(f'Unique genes: {transcripts[\"feature_name\"].nunique()}')",
            "print(f'Assigned to cells: {(transcripts[\"cell_id\"]!=\"UNASSIGNED\").sum():,} ({100*(transcripts[\"cell_id\"]!=\"UNASSIGNED\").mean():.1f}%)')",
            "",
            "transcripts.head()",
        ]),
        make_cell([
            "# Top 20 expressed genes",
            "top_genes = transcripts['feature_name'].value_counts().head(20)",
            "fig, ax = plt.subplots(1, 1, figsize=(8, 6))",
            "top_genes.plot(kind='barh', ax=ax)",
            "ax.set_xlabel('Transcript count')",
            "ax.set_title('Top 20 Expressed Genes')",
            "plt.tight_layout()",
            "plt.show()",
        ]),
        make_md(["## 5. Xenium Cell Summary"]),
        make_cell([
            "cells = dl.load_cells_df()",
            "print(f'Total Xenium cells: {len(cells):,}')",
            "print()",
            "print('Metrics summary:')",
            "print(cells[['cell_area', 'nucleus_area', 'transcript_counts', 'total_counts']].describe())",
        ]),
        make_cell([
            "fig, axes = plt.subplots(1, 3, figsize=(15, 4))",
            "",
            "axes[0].hist(cells['cell_area'], bins=50, color='steelblue', edgecolor='white')",
            "axes[0].set_xlabel('Cell area (um^2)')",
            "axes[0].set_ylabel('Count')",
            "axes[0].set_title('Cell Area Distribution')",
            "",
            "axes[1].hist(cells['transcript_counts'], bins=50, color='coral', edgecolor='white')",
            "axes[1].set_xlabel('Transcript counts')",
            "axes[1].set_title('Transcripts per Cell')",
            "",
            "axes[2].scatter(cells['cell_area'], cells['transcript_counts'], s=1, alpha=0.3)",
            "axes[2].set_xlabel('Cell area (um^2)')",
            "axes[2].set_ylabel('Transcript counts')",
            "axes[2].set_title('Area vs Transcripts')",
            "",
            "plt.tight_layout()",
            "plt.show()",
        ]),
        make_md(["## 6. Gene Expression Matrix"]),
        make_cell([
            "expr_matrix, cell_ids, gene_names = dl.load_expression_matrix()",
            "print(f'Expression matrix shape: {expr_matrix.shape}')",
            "print(f'  Cells: {expr_matrix.shape[0]:,}')",
            "print(f'  Genes: {expr_matrix.shape[1]}')",
            "print(f'  Non-zero entries: {expr_matrix.nnz:,}')",
            "print(f'  Sparsity: {100 * expr_matrix.nnz / (expr_matrix.shape[0] * expr_matrix.shape[1]):.2f}%')",
            "print()",
            "print(f'First 10 genes: {gene_names[:10]}')",
        ]),
        make_cell([
            "# Gene detection rate",
            "detection_rate = np.array((expr_matrix > 0).sum(axis=0)).flatten() / expr_matrix.shape[0]",
            "top_detected = pd.DataFrame({'gene': gene_names, 'detection_rate': detection_rate}).sort_values('detection_rate', ascending=False).head(15)",
            "",
            "fig, ax = plt.subplots(1, 1, figsize=(8, 5))",
            "ax.barh(range(len(top_detected)), top_detected['detection_rate'].values, color='teal')",
            "ax.set_yticks(range(len(top_detected)))",
            "ax.set_yticklabels(top_detected['gene'].values)",
            "ax.set_xlabel('Fraction of cells expressing gene')",
            "ax.set_title('Top 15 Most Widely Expressed Genes')",
            "ax.invert_yaxis()",
            "plt.tight_layout()",
            "plt.show()",
        ]),
    ]
    return make_notebook(cells)


# =============================================================================
# Notebook 2: Cellpose-SAM Segmentation
# =============================================================================

def notebook_02_segmentation():
    cells = [
        make_md([
            "# Cellpose-SAM Segmentation on Xenium Images",
            "",
            "This notebook runs Cellpose-SAM segmentation on both the morphology",
            "(nuclear stain) and H&E images, and visualizes the results.",
        ]),
        make_cell([
            "import sys; sys.path.insert(0, '..')",
            "import numpy as np",
            "import matplotlib.pyplot as plt",
            "from pathlib import Path",
            "",
            "from cpsam_xenium_analysis import config, data_loader as dl",
            "from cpsam_xenium_analysis.segmentation import CPSAMSegmentor, filter_masks",
            "from cpsam_xenium_analysis.visualization import plots as vis",
            "",
            "%matplotlib inline",
        ]),
        make_md(["## 1. Load Images (Cropped ROI)"]),
        make_cell([
            "crop = config.CROP_ROI",
            "morph_img = dl.load_morphology_image(use_focus=True, crop_roi=crop)",
            "he_img = dl.load_he_image(crop_roi=crop)",
            "print(f'Morphology: {morph_img.shape}')",
            "print(f'H&E: {he_img.shape}')",
        ]),
        make_md(["## 2. Run Cellpose-SAM on Morphology Image"]),
        make_cell([
            "# Initialize segmentor",
            "segmentor = CPSAMSegmentor(gpu=True, use_bfloat16=True)",
            "",
            "# Run segmentation on morphology image",
            "mask_morph, flows_morph, styles_morph = segmentor.segment_morphology(morph_img)",
            "mask_morph = filter_masks(mask_morph, min_size=15)",
            "",
            "n_cells = len(np.unique(mask_morph)) - 1",
            "print(f'Found {n_cells} cells in morphology image')",
        ]),
        make_cell([
            "# Visualize segmentation overlay",
            "fig = vis.plot_segmentation_overlay(",
            "    morph_img, mask_morph,",
            "    title=f'Cellpose-SAM on Morphology ({n_cells} cells)'",
            ")",
        ]),
        make_md(["## 3. Run Cellpose-SAM on H&E Image"]),
        make_cell([
            "mask_he, flows_he, styles_he = segmentor.segment_he(he_img)",
            "mask_he = filter_masks(mask_he, min_size=15)",
            "",
            "n_cells_he = len(np.unique(mask_he)) - 1",
            "print(f'Found {n_cells_he} cells in H&E image')",
        ]),
        make_cell([
            "# Visualize H&E segmentation",
            "fig = vis.plot_segmentation_overlay(",
            "    he_img, mask_he,",
            "    title=f'Cellpose-SAM on H&E ({n_cells_he} cells)',",
            "    outline_color='yellow',",
            ")",
        ]),
        make_md(["## 4. Side-by-Side Comparison"]),
        make_cell([
            "fig, axes = plt.subplots(1, 2, figsize=(16, 8))",
            "",
            "axes[0].imshow(morph_img)",
            "axes[0].imshow((mask_morph > 0).astype(float), cmap='jet', alpha=0.3)",
            "axes[0].set_title(f'Morphology + CPSAM ({n_cells} cells)')",
            "axes[0].axis('off')",
            "",
            "axes[1].imshow(he_img)",
            "axes[1].imshow((mask_he > 0).astype(float), cmap='jet', alpha=0.3)",
            "axes[1].set_title(f'H&E + CPSAM ({n_cells_he} cells)')",
            "axes[1].axis('off')",
            "",
            "plt.tight_layout()",
            "plt.show()",
        ]),
        make_md(["## 5. Save Masks for Downstream Analysis"]),
        make_cell([
            "np.save(config.OUTPUT_DIR / 'masks_cpsam_morphology.npy', mask_morph)",
            "np.save(config.OUTPUT_DIR / 'masks_cpsam_he.npy', mask_he)",
            "print('Masks saved!')",
        ]),
    ]
    return make_notebook(cells)


# =============================================================================
# Notebook 3: Transcript Mapping
# =============================================================================

def notebook_03_transcript_mapping():
    cells = [
        make_md([
            "# Transcript-to-Cell Assignment and Expression Matrix",
            "",
            "This notebook assigns Xenium transcripts to Cellpose-SAM segmented cells",
            "and builds gene expression matrices for downstream analysis.",
        ]),
        make_cell([
            "import sys; sys.path.insert(0, '..')",
            "import numpy as np",
            "import pandas as pd",
            "import matplotlib.pyplot as plt",
            "",
            "from cpsam_xenium_analysis import config, data_loader as dl",
            "from cpsam_xenium_analysis.integration import CoordinateAligner, TranscriptMapper",
            "",
            "%matplotlib inline",
        ]),
        make_md(["## 1. Load Data"]),
        make_cell([
            "crop = config.CROP_ROI",
            "",
            "# Load transcripts",
            "transcripts = dl.load_transcripts(min_qv=20)",
            "print(f'Loaded {len(transcripts):,} transcripts')",
            "",
            "# Load expression matrix for gene names",
            "xenium_matrix, xenium_cell_ids, gene_names = dl.load_expression_matrix()",
            "print(f'Gene panel: {len(gene_names)} genes')",
            "",
            "# Load Xenium cells for alignment",
            "cells_df = dl.load_cells_df()",
            "print(f'Xenium cells: {len(cells_df):,}')",
            "",
            "# Load cellpose-sam masks",
            "mask_morph = np.load(config.OUTPUT_DIR / 'masks_cpsam_morphology.npy')",
            "mask_he = np.load(config.OUTPUT_DIR / 'masks_cpsam_he.npy')",
            "print(f'CPSAM Morphology mask: {mask_morph.shape}, cells={len(np.unique(mask_morph))-1}')",
            "print(f'CPSAM H&E mask: {mask_he.shape}, cells={len(np.unique(mask_he))-1}')",
        ]),
        make_md(["## 2. Coordinate Alignment"]),
        make_cell([
            "aligner = CoordinateAligner()",
            "",
            "# Convert transcript coordinates to pixels and crop to ROI",
            "transcripts_px = aligner.transcripts_to_pixel_coords(transcripts, crop_roi=crop)",
            "print(f'Transcripts in ROI: {len(transcripts_px):,}')",
            "transcripts_px.head()",
        ]),
        make_md(["## 3. Assign Transcripts to Cells (Morphology)"]),
        make_cell([
            "mapper = TranscriptMapper(gene_names)",
            "",
            "# Morphology-based assignment",
            "assigned_morph = mapper.assign_transcripts(transcripts_px, mask_morph)",
            "cell_labels_morph, _, expr_morph = mapper.build_expression_matrix(assigned_morph)",
            "",
            "print(f'Cells with expression: {len(cell_labels_morph):,}')",
            "print(f'Expression matrix shape: {expr_morph.shape}')",
            "print(f'Non-zero entries: {expr_morph.nnz:,}')",
        ]),
        make_md(["## 4. Compare with Xenium Original Expression"]),
        make_cell([
            "# Map Xenium cells to CPSAM cells",
            "cell_to_mask = aligner.build_cell_id_to_mask_map(cells_df, mask_morph, crop_roi=crop)",
            "",
            "# Aggregate Xenium expression per CPSAM cell",
            "_, _, xenium_agg = mapper.build_expression_from_xenium_matrix(",
            "    xenium_cell_ids, xenium_matrix, cell_to_mask",
            ")",
            "print(f'Aggregated Xenium matrix: {xenium_agg.shape}')",
        ]),
        make_cell([
            "# Compare transcript counts per cell",
            "cpsam_counts = np.array(expr_morph.sum(axis=1)).flatten()",
            "xenium_counts = np.array(xenium_agg.sum(axis=1)).flatten()",
            "",
            "fig, axes = plt.subplots(1, 2, figsize=(12, 5))",
            "",
            "axes[0].hist(cpsam_counts, bins=50, alpha=0.6, label='CPSAM assignment')",
            "axes[0].hist(xenium_counts, bins=50, alpha=0.6, label='Xenium aggregated')",
            "axes[0].set_xlabel('Transcripts per cell')",
            "axes[0].set_ylabel('Count')",
            "axes[0].legend()",
            "axes[0].set_title('Transcript Count Distribution')",
            "",
            "axes[1].scatter(xenium_counts, cpsam_counts, s=1, alpha=0.3)",
            "axes[1].set_xlabel('Xenium aggregated transcripts/cell')",
            "axes[1].set_ylabel('CPSAM assigned transcripts/cell')",
            "axes[1].set_title('Correlation')",
            "",
            "plt.tight_layout()",
            "plt.show()",
        ]),
        make_md(["## 5. Visually Compare Gene Expression Spatial Patterns"]),
        make_cell([
            "# Plot spatial expression of INS (Beta cell marker)",
            "fig = dl.visualization.plots.plot_gene_expression_spatial(",
            "    assigned_morph, 'INS', cell_mask=mask_morph,",
            "    title='INS expression on CPSAM morphology segmentation'",
            ")",
        ]),
        make_md(["## 6. Also Process H&E Segmentation"]),
        make_cell([
            "assigned_he = mapper.assign_transcripts(transcripts_px, mask_he)",
            "cell_labels_he, _, expr_he = mapper.build_expression_matrix(assigned_he)",
            "",
            "print(f'CPSAM H&E cells with expression: {len(cell_labels_he):,}')",
            "print(f'Expression matrix shape: {expr_he.shape}')",
        ]),
        make_cell([
            "# Save expression matrices",
            "from cpsam_xenium_analysis.integration.transcript_mapping import save_expression_npz",
            "",
            "save_expression_npz(expr_morph, cell_labels_morph, gene_names, ",
            "    config.OUTPUT_DIR / 'expr_cpsam_morphology')",
            "save_expression_npz(expr_he, cell_labels_he, gene_names, ",
            "    config.OUTPUT_DIR / 'expr_cpsam_he')",
            "print('Expression matrices saved!')",
        ]),
    ]
    return make_notebook(cells)


# =============================================================================
# Notebook 4: Cell Typing
# =============================================================================

def notebook_04_cell_typing():
    cells = [
        make_md([
            "# Cell Type Identification",
            "",
            "This notebook assigns cell types to segmented cells using known",
            "pancreatic cell type markers. It uses both a marker-based scoring",
            "approach and unsupervised clustering with annotation.",
        ]),
        make_cell([
            "import sys; sys.path.insert(0, '..')",
            "import numpy as np",
            "import pandas as pd",
            "import matplotlib.pyplot as plt",
            "",
            "from cpsam_xenium_analysis import config",
            "from cpsam_xenium_analysis.analysis import CellTyper",
            "from cpsam_xenium_analysis.visualization import plots as vis",
            "from cpsam_xenium_analysis.integration.transcript_mapping import save_expression_npz",
            "",
            "%matplotlib inline",
        ]),
        make_cell([
            "# Load expression data for morphology-based segmentation",
            "gene_names = np.load(config.OUTPUT_DIR / 'expr_cpsam_morphology_gene_names.npy', allow_pickle=True).tolist()",
            "cell_labels = np.load(config.OUTPUT_DIR / 'expr_cpsam_morphology_cell_ids.npy')",
            "from scipy.sparse import load_npz",
            "expr_mat = load_npz(str(config.OUTPUT_DIR / 'expr_cpsam_morphology_matrix.npz'))",
            "mask_morph = np.load(config.OUTPUT_DIR / 'masks_cpsam_morphology.npy')",
            "",
            "print(f'Expression matrix: {expr_mat.shape}')",
            "print(f'Gene panel: {len(gene_names)} genes')",
            "print(f'Cell mask labels: {len(cell_labels)} unique cells')",
        ]),
        make_md(["## 1. Marker-Based Cell Typing"]),
        make_cell([
            "typer = CellTyper()",
            "print('Pancreatic cell type markers:')",
            "for ct, genes in typer.markers.items():",
            "    available = [g for g in genes if g in gene_names]",
            "    print(f'  {ct}: {available}')",
        ]),
        make_cell([
            "cell_types = typer.score_cell_types(expr_mat, gene_names)",
            "print(cell_types['cell_type'].value_counts())",
            "cell_types.head()",
        ]),
        make_cell([
            "# Visualize cell type map",
            "fig = vis.plot_cell_type_map(",
            "    mask_morph, cell_types, cell_labels,",
            "    title='Cell Type Map (Morphology-based segmentation)'",
            ")",
        ]),
        make_md(["## 2. Marker Gene Expression per Cell Type"]),
        make_cell([
            "# Expression of key markers across cell types",
            "marker_genes = ['GCG', 'INS', 'SST', 'PRSS1', 'KRT19', 'PECAM1', 'PTPRC']",
            "marker_genes = [g for g in marker_genes if g in gene_names]",
            "",
            "type_order = cell_types.groupby('cell_type')['cell_id'].count().sort_values(ascending=False).index",
            "",
            "fig, axes = plt.subplots(1, len(marker_genes), figsize=(4*len(marker_genes), 4))",
            "if len(marker_genes) == 1:",
            "    axes = [axes]",
            "",
            "for i, gene in enumerate(marker_genes):",
            "    gene_idx = gene_names.index(gene)",
            "    gene_expr = expr_mat[:, gene_idx].toarray().flatten()",
            "    data = pd.DataFrame({'cell_type': cell_types['cell_type'], 'expression': gene_expr})",
            "    means = data.groupby('cell_type')['expression'].mean().reindex(type_order)",
            "    axes[i].bar(range(len(means)), means.values)",
            "    axes[i].set_xticks(range(len(means)))",
            "    axes[i].set_xticklabels(means.index, rotation=45, ha='right', fontsize=6)",
            "    axes[i].set_title(f'{gene}')",
            "    axes[i].set_ylabel('Mean expression')",
            "",
            "plt.tight_layout()",
            "plt.show()",
        ]),
        make_md(["## 3. Cluster-Based Cell Typing (requires scanpy)"]),
        make_cell([
            "# Unsupervised clustering for validation",
            "try:",
            "    cluster_types = typer.cluster_and_annotate(expr_mat, gene_names)",
            "    print('Cluster-based annotation complete')",
            "    ",
            "    # Compare with marker-based annotation",
            "    agreement = (cluster_types['cell_type'] == cell_types['cell_type']).mean()",
            "    print(f'Agreement between marker-based and cluster-based: {agreement:.1%}')",
            "except Exception as e:",
            "    print(f'Clustering skipped (install scanpy if desired): {e}')",
        ]),
        make_md(["## 4. H&E-based Cell Typing (if available)"]),
        make_cell([
            "if (config.OUTPUT_DIR / 'expr_cpsam_he_matrix.npz').exists():",
            "    cell_labels_he = np.load(config.OUTPUT_DIR / 'expr_cpsam_he_cell_ids.npy')",
            "    expr_he = load_npz(str(config.OUTPUT_DIR / 'expr_cpsam_he_matrix.npz'))",
            "    mask_he = np.load(config.OUTPUT_DIR / 'masks_cpsam_he.npy')",
            "    cell_types_he = typer.score_cell_types(expr_he, gene_names)",
            "    ",
            "    print('\\nH&E cell type distribution:')",
            "    print(cell_types_he['cell_type'].value_counts())",
            "else:",
            "    print('H&E expression data not found. Run transcript mapping first.')",
        ]),
    ]
    return make_notebook(cells)


# =============================================================================
# Notebook 5: Segmentation Comparison
# =============================================================================

def notebook_05_segmentation_comparison():
    cells = [
        make_md([
            "# Segmentation Method Comparison",
            "",
            "Systematic comparison of Xenium native segmentation vs ",
            "Cellpose-SAM (morphology) vs Cellpose-SAM (H&E).",
        ]),
        make_cell([
            "import sys; sys.path.insert(0, '..')",
            "import numpy as np",
            "import pandas as pd",
            "import matplotlib.pyplot as plt",
            "",
            "from cpsam_xenium_analysis import config, data_loader as dl",
            "from cpsam_xenium_analysis.comparison import SegmentationComparator",
            "from cpsam_xenium_analysis.segmentation.post_process import compute_morphology_features",
            "",
            "%matplotlib inline",
        ]),
        make_md(["## 1. Load All Segmentation Results"]),
        make_cell([
            "# Load CPSAM masks",
            "mask_morph = np.load(config.OUTPUT_DIR / 'masks_cpsam_morphology.npy')",
            "mask_he = np.load(config.OUTPUT_DIR / 'masks_cpsam_he.npy')",
            "",
            "# Load Xenium cell boundaries as a mask (for comparison)",
            "from cpsam_xenium_analysis.integration import CoordinateAligner",
            "cells_df = dl.load_cells_df()",
            "cell_boundaries = dl.load_cell_boundaries()",
            "",
            "print(f'CPSAM Morphology: {len(np.unique(mask_morph))-1} cells')",
            "print(f'CPSAM H&E: {len(np.unique(mask_he))-1} cells')",
            "print(f'Xenium: {len(cells_df)} cells in full dataset')",
        ]),
        make_cell([
            "# Build Xenium reference mask within the crop ROI",
            "crop = config.CROP_ROI",
            "xs, ys = crop['x_start'], crop['y_start']",
            "xe, ye = xs + crop['width'], ys + crop['height']",
            "",
            "# Filter Xenium cells to ROI",
            "aligner = CoordinateAligner()",
            "cells_px = aligner.cell_centroids_to_pixels(cells_df, crop_roi=crop)",
            "print(f'Xenium cells in ROI: {len(cells_px)}')",
        ]),
        make_md(["## 2. Compare Cell Counts and Areas"]),
        make_cell([
            "comparator = SegmentationComparator()",
            "comparator.add_segmentation('cpsam_morphology', mask_morph)",
            "comparator.add_segmentation('cpsam_he', mask_he)",
            "",
            "report = comparator.summary_report()",
            "",
            "print('=== Cell Counts ===')",
            "print(report['cell_counts'])",
            "print()",
            "print('=== Area Distribution ===')",
            "print(report['area_distribution'])",
            "print()",
            "print('=== Pairwise Overlap ===')",
            "print(report['pairwise_overlap'])",
        ]),
        make_md(["## 3. Per-Cell Matching Analysis"]),
        make_cell([
            "comparison = comparator.per_cell_comparison(",
            "    'cpsam_morphology', 'cpsam_he', iou_threshold=0.3",
            ")",
            "",
            "print(f'Matched cells: {len(comparison[\"matching\"][\"matches\"])}')",
            "print(f'Mean IoU: {comparison[\"mean_iou\"]:.3f}')",
            "print(f'Median IoU: {comparison[\"median_iou\"]:.3f}')",
            "print(f'Mean boundary distance: {comparison[\"mean_boundary_distance\"]:.1f} px')",
            "print(f'Precision: {comparison[\"matching\"][\"precision\"]:.3f}')",
            "print(f'Recall: {comparison[\"matching\"][\"recall\"]:.3f}')",
            "print(f'F1: {comparison[\"matching\"][\"f1\"]:.3f}')",
        ]),
        make_md(["## 4. Morphology Feature Comparison"]),
        make_cell([
            "features_morph = compute_morphology_features(mask_morph)",
            "features_he = compute_morphology_features(mask_he)",
            "",
            "df_morph = pd.DataFrame(features_morph).T",
            "df_he = pd.DataFrame(features_he).T",
            "",
            "fig, axes = plt.subplots(2, 2, figsize=(12, 10))",
            "",
            "for ax, col, title in zip(axes.flatten(), ",
            "    ['area', 'circularity', 'eccentricity', 'solidity'],",
            "    ['Cell Area (pixels)', 'Circularity', 'Eccentricity', 'Solidity']",
            "):",
            "    ax.hist(df_morph[col].dropna(), bins=50, alpha=0.6, label='CPSAM Morphology')",
            "    ax.hist(df_he[col].dropna(), bins=50, alpha=0.6, label='CPSAM H&E')",
            "    ax.set_xlabel(title)",
            "    ax.set_ylabel('Count')",
            "    ax.legend(fontsize=7)",
            "",
            "plt.tight_layout()",
            "plt.show()",
        ]),
        make_md(["## 5. Comparison Summary"]),
        make_cell([
            "print('=' * 60)",
            "print('SEGMENTATION COMPARISON SUMMARY')",
            "print('=' * 60)",
            "print()",
            "",
            "for method in ['cpsam_morphology', 'cpsam_he']:",
            "    n_cells = len(np.unique(locals()[f'mask_{method.split(\"_\")[1]}'])) - 1",
            "    print(f'{method}: {n_cells} cells')",
            "",
            "print()",
            "print(f'Pairwise Dice coeff: {report[\"pairwise_overlap\"].to_string(index=False)}')",
            "print()",
            "print(f'Matched cells (CPSAM Morph vs HE): {len(comparison[\"matching\"][\"matches\"])}')",
            "print(f'Mean IoU of matched cells: {comparison[\"mean_iou\"]:.3f}')",
        ]),
    ]
    return make_notebook(cells)


# =============================================================================
# Notebook 6: Biomarker Analysis
# =============================================================================

def notebook_06_biomarker_analysis():
    cells = [
        make_md([
            "# Spatial Biomarker Analysis",
            "",
            "This notebook conducts spatial analysis and biomarker discovery",
            "on the Cellpose-SAM segmented Xenium data. It includes spatial",
            "autocorrelation (Moran's I), differential expression analysis,",
            "and biomarker candidate identification.",
        ]),
        make_cell([
            "import sys; sys.path.insert(0, '..')",
            "import numpy as np",
            "import pandas as pd",
            "import matplotlib.pyplot as plt",
            "",
            "from cpsam_xenium_analysis import config",
            "from cpsam_xenium_analysis.data_loader import load_transcripts",
            "from cpsam_xenium_analysis.analysis import SpatialAnalyzer, BiomarkerDiscovery, CellTyper",
            "from cpsam_xenium_analysis.visualization import plots as vis",
            "from cpsam_xenium_analysis.segmentation.post_process import compute_cell_centroids",
            "from scipy.sparse import load_npz",
            "",
            "%matplotlib inline",
        ]),
        make_cell([
            "# Load CPSAM morphology results",
            "gene_names = np.load(config.OUTPUT_DIR / 'expr_cpsam_morphology_gene_names.npy', allow_pickle=True).tolist()",
            "cell_labels = np.load(config.OUTPUT_DIR / 'expr_cpsam_morphology_cell_ids.npy')",
            "expr_mat = load_npz(str(config.OUTPUT_DIR / 'expr_cpsam_morphology_matrix.npz'))",
            "mask_morph = np.load(config.OUTPUT_DIR / 'masks_cpsam_morphology.npy')",
            "transcripts = load_transcripts(min_qv=20)",
            "",
            "print(f'Expression matrix: {expr_mat.shape}')",
            "print(f'Cells: {len(cell_labels)}')",
            "print(f'Genes: {len(gene_names)}')",
        ]),
        make_md(["## 1. Spatial Autocorrelation (Moran's I)"]),
        make_cell([
            "# Get cell centroids",
            "centroids = compute_cell_centroids(mask_morph)",
            "coords = np.array([[centroids[lbl][1], centroids[lbl][0]] ",
            "    for lbl in cell_labels if lbl in centroids])",
            "print(f'Coordinates: {coords.shape}')",
        ]),
        make_cell([
            "spatial = SpatialAnalyzer(k_neighbors=15)",
            "spatial.compute_spatial_weights(coords)",
            "",
            "# Compute Moran's I for all genes",
            "morans_df = spatial.compute_all_morans_i(expr_mat, gene_names)",
            "print(f'Genes with significant spatial autocorrelation: {(morans_df[\"p_value\"] < 0.05).sum()}')",
            "morans_df.head(10)",
        ]),
        make_cell([
            "# Visualize Moran's I",
            "fig = vis.plot_morans_i_scatter(morans_df, ",
            "    highlight_genes=['INS', 'GCG', 'KRT19', 'EPCAM'],",
            "    title=\"Moran's I - Spatial Autocorrelation of Gene Expression\"",
            ")",
        ]),
        make_md(["## 2. Spatially Variable Genes"]),
        make_cell([
            "svg_df = spatial.spatially_variable_genes(expr_mat, gene_names)",
            "print(f'Top 15 spatially variable genes:')",
            "svg_df.head(15)",
        ]),
        make_cell([
            "fig = vis.plot_spatial_variable_genes(svg_df, n_top=20, ",
            "    title='Top 20 Spatially Variable Genes (by Moran\\'s I)'",
            ")",
        ]),
        make_md(["## 3. Spatial Expression of Key Markers"]),
        make_cell([
            "crop = config.CROP_ROI",
            "aligner = __import__('cpsam_xenium_analysis.integration.alignment', fromlist=['CoordinateAligner']).CoordinateAligner()",
            "transcripts_px = aligner.transcripts_to_pixel_coords(transcripts, crop_roi=crop)",
            "",
            "markers = ['INS', 'GCG', 'KRT19', 'PTPRC', 'PECAM1', 'SST']",
            "markers = [g for g in markers if g in gene_names]",
            "",
            "fig, axes = plt.subplots(2, 3, figsize=(15, 10))",
            "axes = axes.flatten()",
            "",
            "for i, gene in enumerate(markers):",
            "    gene_tr = transcripts_px[transcripts_px['feature_name'] == gene]",
            "    axes[i].imshow((mask_morph > 0).astype(np.uint8), cmap='gray', alpha=0.3)",
            "    if len(gene_tr) > 0:",
            "        axes[i].scatter(gene_tr['x_pixel'], gene_tr['y_pixel'], ",
            "            s=0.5, c='red', alpha=0.5)",
            "    axes[i].set_title(f'{gene} ({len(gene_tr)} transcripts)')",
            "    axes[i].axis('off')",
            "    axes[i].invert_yaxis()",
            "",
            "plt.suptitle('Spatial Expression of Key Pancreatic Markers', fontsize=14)",
            "plt.tight_layout()",
            "plt.show()",
        ]),
        make_md(["## 4. Differential Expression Analysis"]),
        make_cell([
            "# First run cell typing to get labels",
            "typer = CellTyper()",
            "cell_types = typer.score_cell_types(expr_mat, gene_names)",
            "",
            "# Create binary labels for DE",
            "type_labels = cell_types['cell_type'].values",
            "binary_labels = np.array([",
            "    'Tumor' if t == 'Tumor_Epithelial' else 'Immune' if t == 'Immune' else 'Other'",
            "    for t in type_labels",
            "])",
        ]),
        make_cell([
            "bio = BiomarkerDiscovery()",
            "",
            "de_results = bio.differential_expression(",
            "    expr_mat, gene_names, binary_labels, ",
            "    group_a='Tumor', group_b='Other',",
            ")",
            "print(f'Significant genes: {(de_results[\"p_value_adj\"] < 0.05).sum()}')",
            "de_results.head(10)",
        ]),
        make_cell([
            "# Volcano plot",
            "fig = vis.plot_biomarker_volcano(de_results,",
            "    title='Differential Expression: Tumor Epithelial vs Other Cells'",
            ")",
        ]),
        make_md(["## 5. Gene-Morphology Correlation"]),
        make_cell([
            "from cpsam_xenium_analysis.segmentation.post_process import compute_morphology_features",
            "",
            "morph_features = compute_morphology_features(mask_morph)",
            "",
            "corr_df = bio.correlate_with_morphology(",
            "    expr_mat, gene_names, morph_features, cell_labels, feature_name='area'",
            ")",
            "print('\\nTop genes correlated with cell area:')",
            "corr_df.head(10)",
        ]),
        make_md(["## 6. Surrogate Biomarker Candidates"]),
        make_cell([
            "candidates = bio.surrogate_biomarkers(",
            "    expr_mat, gene_names, morans_df, de_results",
            ")",
            "print('\\nTop biomarker candidates (combining DE + spatial signals):')",
            "candidates[['gene', 'log2fc', 'p_value_adj', 'morans_i', 'biomarker_score']].head(20)",
        ]),
        make_md(["## 7. Gene Signature Scoring"]),
        make_cell([
            "signatures = {",
            "    'Epithelial': ['EPCAM', 'KRT19', 'KRT7', 'CDH1'],",
            "    'Mesenchymal': ['VIM', 'FN1', 'COL1A1', 'SNAI2'],",
            "    'Immune_activation': ['CD3D', 'CD8A', 'GZMB', 'IFNG'],",
            "    'Angiogenesis': ['VEGFA', 'PECAM1', 'CDH5', 'VWF'],",
            "    'Proliferation': ['MKI67', 'TOP2A', 'PCNA'],",
            "}",
            "# Filter to available genes",
            "signatures = {k: [g for g in v if g in gene_names] for k, v in signatures.items()}",
            "signatures = {k: v for k, v in signatures.items() if len(v) > 0}",
            "print('Available signatures:', list(signatures.keys()))",
        ]),
        make_cell([
            "sig_scores = bio.gene_signature_scoring(expr_mat, gene_names, signatures)",
            "sig_scores.head()",
        ]),
        make_cell([
            "# Map signature scores back to spatial coordinates",
            "valid_idx = [i for i, lbl in enumerate(cell_labels) if lbl in centroids]",
            "valid_coords = np.array([centroids[cell_labels[i]] for i in valid_idx])",
            "",
            "fig, axes = plt.subplots(1, len(signatures), figsize=(4*len(signatures), 4))",
            "if len(signatures) == 1:",
            "    axes = [axes]",
            "",
            "for i, sig in enumerate(signatures.keys()):",
            "    scores = sig_scores[f'{sig}_zscore'].values[valid_idx]",
            "    sc = axes[i].scatter(valid_coords[:, 1], valid_coords[:, 0], ",
            "        c=scores, cmap='RdBu_r', s=3, vmin=-2, vmax=2)",
            "    axes[i].set_title(sig)",
            "    axes[i].axis('off')",
            "    axes[i].invert_yaxis()",
            "    plt.colorbar(sc, ax=axes[i], shrink=0.6)",
            "",
            "plt.suptitle('Spatial Distribution of Gene Signatures', fontsize=14)",
            "plt.tight_layout()",
            "plt.show()",
        ]),
    ]
    return make_notebook(cells)


# =============================================================================
# Generate All Notebooks
# =============================================================================

if __name__ == "__main__":
    notebooks = {
        "01_data_exploration.ipynb": notebook_01_data_exploration(),
        "02_cellpose_sam_segmentation.ipynb": notebook_02_segmentation(),
        "03_transcript_mapping.ipynb": notebook_03_transcript_mapping(),
        "04_cell_typing.ipynb": notebook_04_cell_typing(),
        "05_segmentation_comparison.ipynb": notebook_05_segmentation_comparison(),
        "06_biomarker_analysis.ipynb": notebook_06_biomarker_analysis(),
    }

    for name, notebook in notebooks.items():
        save_notebook(name, notebook)

    print(f"\nAll {len(notebooks)} notebooks created in {NOTEBOOK_DIR}")
