"""
Configuration module for the Cellpose-SAM Xenium analysis pipeline.

All data paths and analysis parameters are centralized here.
"""

import os
from pathlib import Path

# =============================================================================
# Path Configuration
# =============================================================================

# Base data directory - derived from config file location for cross-platform compatibility
# Expected structure:
#   config.py -> cpsam_xenium_analysis/config.py
#   data      -> cpsam_xenium_analysis/../Xenium_V1_human_Pancreas_FFPE_outs/  (relative)
#   OR data   -> ../../../Xenium_V1_human_Pancreas_FFPE_outs/  (from cellpose repo root)
CONFIG_DIR = Path(__file__).parent.resolve()

# Try to locate the Xenium data directory relative to the project
_CANDIDATES = [
    CONFIG_DIR / ".." / "Xenium_V1_human_Pancreas_FFPE_outs",  # sibling dir
    CONFIG_DIR / ".." / ".." / ".." / ".." / "Xenium_V1_human_Pancreas_FFPE_outs",  # from cellpose repo
    CONFIG_DIR / ".." / ".." / "Xenium_V1_human_Pancreas_FFPE_outs",  # from Spatial_analysis/Xenium/Human_pancreas
]

XENIUM_OUTS_DIR = None
for _cand in _CANDIDATES:
    _resolved = _cand.resolve()
    if _resolved.exists():
        XENIUM_OUTS_DIR = _resolved
        break

if XENIUM_OUTS_DIR is None:
    raise FileNotFoundError(
        f"Cannot find Xenium output directory. Tried: {[str(c) for c in _CANDIDATES]}"
    )

DATA_DIR = XENIUM_OUTS_DIR.parent

# Output directory for analysis results
OUTPUT_DIR = CONFIG_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# Image files
MORPHOLOGY_IMAGE_PATH = XENIUM_OUTS_DIR / "morphology.ome.tif"
MORPHOLOGY_FOCUS_DIR = XENIUM_OUTS_DIR / "morphology_focus"
HE_IMAGE_PATH = DATA_DIR / "Xenium_V1_human_Pancreas_FFPE_he_image.ome.tif"

# Xenium data files
TRANSCRIPTS_PATH = XENIUM_OUTS_DIR / "transcripts.parquet"
CELLS_PATH = XENIUM_OUTS_DIR / "cells.csv.gz"
CELL_BOUNDARIES_PATH = XENIUM_OUTS_DIR / "cell_boundaries.parquet"
NUCLEUS_BOUNDARIES_PATH = XENIUM_OUTS_DIR / "nucleus_boundaries.parquet"
EXPRESSION_MATRIX_PATH = XENIUM_OUTS_DIR / "cell_feature_matrix.h5"
EXPERIMENT_PATH = XENIUM_OUTS_DIR / "experiment.xenium"
METRICS_PATH = XENIUM_OUTS_DIR / "metrics_summary.csv"
GENE_PANEL_PATH = XENIUM_OUTS_DIR / "gene_panel.json"

# =============================================================================
# Xenium Data Parameters
# =============================================================================

# Pixel size in micrometers (from experiment.xenium)
PIXEL_SIZE_UM = 0.2125

# =============================================================================
# Cellpose-SAM Segmentation Parameters
# =============================================================================

SEGMENTATION_PARAMS = {
    "morphology": {
        "diameter": 30.0,          # Estimated cell diameter in pixels
        "flow_threshold": 0.4,
        "cellprob_threshold": 0.0,
        "min_size": 15,             # Minimum mask size in pixels
        "max_size_fraction": 0.4,   # Maximum mask size as fraction of image
        "bsize": 256,               # Tile size for processing
        "tile_overlap": 0.1,        # Overlap between tiles
        "augment": False,
        "resample": True,
        "batch_size": 8,            # GPU batch size
    },
    "he": {
        "diameter": 30.0,
        "flow_threshold": 0.4,
        "cellprob_threshold": 0.0,
        "min_size": 15,
        "max_size_fraction": 0.4,
        "bsize": 256,
        "tile_overlap": 0.1,
        "augment": False,
        "resample": True,
        "batch_size": 8,
        "max_tile_size": 2000,      # Max tile dimension for image-level tiling (pixels)
        "tile_min_overlap": 200,    # Overlap between image-level tiles (pixels)
    },
    # ---- Cellpose3 (cyto3) segmentation parameters ----
    "cellpose3_morphology": {
        "diameter": 0,               # auto-estimate
        "channels": [1, 0],          # grayscale
        "bsize": 224,
        "niter": 2000,
        "tile_overlap": 0.5,
        "flow_threshold": 0.4,
        "cellprob_threshold": 0.0,
        "augment": True,
        "min_size": 15,
        "batch_size": 8,
        "max_tile_size": 2000,
        "tile_min_overlap": 200,
    },
    "cellpose3_he": {
        "diameter": 0,
        "channels": [1, 0],
        "bsize": 224,
        "niter": 2000,
        "tile_overlap": 0.5,
        "flow_threshold": 0.4,
        "cellprob_threshold": 0.0,
        "augment": True,
        "min_size": 15,
        "batch_size": 8,
        "max_tile_size": 2000,
        "tile_min_overlap": 200,
    },
    # ---- CellSAM segmentation parameters ----
    "cellsam_morphology": {
        "use_wsi": False,
        "low_contrast_enhancement": False,
        "gauge_cell_size": False,
        "min_size": 15,
        "max_tile_size": 2000,
        "tile_min_overlap": 200,
    },
    "cellsam_he": {
        "use_wsi": False,
        "low_contrast_enhancement": False,
        "gauge_cell_size": False,
        "min_size": 15,
        "max_tile_size": 2000,
        "tile_min_overlap": 200,
    },
    # ---- MicroSAM segmentation parameters ----
    "microsam_morphology": {
        "model_type": "vit_l_lm",
        "min_size": 15,
        "max_tile_size": 2000,
        "tile_min_overlap": 200,
    },
    "microsam_he": {
        "model_type": "vit_l_lm",
        "min_size": 15,
        "max_tile_size": 2000,
        "tile_min_overlap": 200,
    },
}

# GPU device (-1 = auto, or device index)
GPU_DEVICE = -1

# =============================================================================
# ROI Cropping (for fast dev/test)
# =============================================================================

# If set to a dict with x_start, y_start, width, height, only that region
# is used for segmentation. Set to None to process the full image.
CROP_ROI = {
    "x_start": 5000,
    "y_start": 5000,
    "width": 4000,
    "height": 4000,
}
# Set True to enable cropping during development
USE_CROP = False

# =============================================================================
# Analysis Parameters
# =============================================================================

# Cell type markers: {cell_type: [gene_names]}
PANCREAS_CELL_MARKERS = {
    "Alpha": ["GCG"],
    "Beta": ["INS"],
    "Delta": ["SST"],
    "PP": ["PPY"],
    "Acinar": ["PRSS1", "CPA1", "CEL"],
    "Ductal": ["KRT19", "EPCAM", "KRT7"],
    "Stellate": ["PDGFRA", "COL1A1"],
    "Endothelial": ["PECAM1", "CDH5", "VWF"],
    "Immune": ["PTPRC", "CD3D", "CD68"],
    "Tumor_Epithelial": ["KRT19", "EPCAM", "CEACAM5", "ERBB2"],
}

# Spatial analysis parameters
SPATIAL_K_NEIGHBORS = 15  # Number of spatial neighbors for analysis

# Biomarker discovery parameters
DE_PVAL_CUTOFF = 0.05
DE_LOGFC_CUTOFF = 0.5
