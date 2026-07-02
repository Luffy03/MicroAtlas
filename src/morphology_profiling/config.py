"""
config.py
=========
Configuration for BBBC021 Biomarker Discovery pipeline.

Dataset: Human MCF-7 breast cancer cells treated with 113 compounds
         at 8 concentrations, 3-channel fluorescence imaging.
         103 compound-concentrations annotated with 12 MoA classes.
Source:  https://bbbc.broadinstitute.org/BBBC021
"""

import os
from pathlib import Path

# =========================================================================
# Path configuration
# =========================================================================

PROJECT_ROOT = Path(__file__).resolve().parent
CELLPOSE_ROOT = PROJECT_ROOT.parent  # ~/Cellpose

# Data paths
DATA_ROOT = PROJECT_ROOT / "data"
IMAGES_DIR = DATA_ROOT / "images"
METADATA_DIR = DATA_ROOT / "metadata"
# Results base directory
RESULTS_DIR = PROJECT_ROOT / "results"

# Model-aware path functions
def masks_dir(model_name="cellpose4"):
    """Return mask output directory for a given model."""
    return RESULTS_DIR / "masks" / model_name

def features_dir(model_name="cellpose4"):
    """Return feature output directory for a given model."""
    return RESULTS_DIR / "features" / model_name

def aggregation_dir(model_name="cellpose4"):
    """Return aggregation output directory for a given model."""
    return RESULTS_DIR / "aggregation" / model_name

def biomarker_dir(model_name="cellpose4"):
    """Return biomarker output directory for a given model."""
    return RESULTS_DIR / "biomarker" / model_name

def vis_dir(model_name="cellpose4"):
    """Return visualization output directory for a given model."""
    return RESULTS_DIR / "visualization" / model_name

# Legacy aliases (default to cellpose4)
MASKS_DIR = masks_dir("cellpose4")
FEATURES_DIR = features_dir("cellpose4")
BIOMARKER_DIR = biomarker_dir("cellpose4")
VIS_DIR = vis_dir("cellpose4")

# Metadata files
IMAGE_CSV = METADATA_DIR / "BBBC021_v1_image.csv"
COMPOUND_CSV = METADATA_DIR / "BBBC021_v1_compound.csv"
MOA_CSV = METADATA_DIR / "BBBC021_v1_moa.csv"

# =========================================================================
# BBBC021 Channel definitions
# =========================================================================

CHANNELS = {
    "DAPI": {"description": "DNA stain (Hoechst/DAPI)", "organelle": "Nucleus"},
    "Actin": {"description": "F-actin (phalloidin)", "organelle": "Cytoskeleton"},
    "Tubulin": {"description": "beta-tubulin", "organelle": "Microtubules"},
}

CHANNEL_NAMES = list(CHANNELS.keys())  # ["DAPI", "Actin", "Tubulin"]
NUCLEUS_CHANNEL = "DAPI"

# Image properties
IMAGE_DTYPE = "uint16"
IMAGE_BIT_DEPTH = 16

# =========================================================================
# BBBC021 Dataset dimensions
# =========================================================================

N_COMPOUNDS = 113
N_CONCENTRATIONS = 8
N_FIELDS_TOTAL = 13200
N_FILES_TOTAL = 39600  # 13200 fields x 3 channels
N_MOA_CLASSES = 13
N_MOA_COMPOUNDS = 38  # compounds with MoA annotation

# MoA classes (13 known mechanisms, from BBBC021_v1_moa.csv)
MOA_CLASSES = [
    "Actin disruptors",
    "Aurora kinase inhibitors",
    "Cholesterol-lowering",
    "DNA damage",
    "DNA replication",
    "DMSO",
    "Eg5 inhibitors",
    "Epithelial",
    "Kinase inhibitors",
    "Microtubule destabilizers",
    "Microtubule stabilizers",
    "Protein degradation",
    "Protein synthesis",
]

# =========================================================================
# Download URLs
# =========================================================================

BBBC_BASE_URL = "https://data.broadinstitute.org/bbbc/BBBC021"

METADATA_URLS = {
    "image_csv": f"{BBBC_BASE_URL}/BBBC021_v1_image.csv",
    "compound_csv": f"{BBBC_BASE_URL}/BBBC021_v1_compound.csv",
    "moa_csv": f"{BBBC_BASE_URL}/BBBC021_v1_moa.csv",
}

# Plate zip URLs (55 archives, ~800 MB each)
PLATE_ZIP_NAMES = [
    "Week1_22123", "Week1_22141", "Week1_22161", "Week1_22361",
    "Week1_22381", "Week1_22401",
    "Week2_24121", "Week2_24141", "Week2_24161", "Week2_24361",
    "Week2_24381", "Week2_24401",
    "Week3_25421", "Week3_25441", "Week3_25461", "Week3_25681",
    "Week3_25701", "Week3_25721",
    "Week4_27481", "Week4_27521", "Week4_27542", "Week4_27801",
    "Week4_27821", "Week4_27861",
    "Week5_28901", "Week5_28921", "Week5_28961", "Week5_29301",
    "Week5_29321", "Week5_29341",
    "Week6_31641", "Week6_31661", "Week6_31681", "Week6_32061",
    "Week6_32121", "Week6_32161",
    "Week7_34341", "Week7_34381", "Week7_34641", "Week7_34661",
    "Week7_34681",
    "Week8_38203", "Week8_38221", "Week8_38241", "Week8_38341",
    "Week8_38342",
    "Week9_39206", "Week9_39221", "Week9_39222", "Week9_39282",
    "Week9_39283", "Week9_39301",
    "Week10_40111", "Week10_40115", "Week10_40119",
]


def plate_zip_url(zip_name):
    return f"{BBBC_BASE_URL}/BBBC021_v1_images_{zip_name}.zip"


# =========================================================================
# Model definitions
# =========================================================================

MODELS = ["cellpose4", "cellpose3", "microsam", "microatlas", "cellsam"]

MODEL_CONFIGS = {
    "cellpose4": {
        "bsize": 256, "channels": None, "niter": 1000, "batch_size": 256,
    },
    "cellpose3": {
        "bsize": 224, "channels": [1, 0], "niter": 2000,
        "tile_overlap": 0.5, "flow_threshold": 0.4, "augment": True,
    },
    "microsam": {
        "model_type": "vit_l_lm",
    },
    "microatlas": {
        "bsize": 256, "channels": None, "niter": 1000, "batch_size": 256,
        "pretrained_model": "./microatlas/microatlas",
    },
    "cellsam": {
        "use_wsi": False, "low_contrast_enhancement": False, "gauge_cell_size": False,
    },
}

# Default segmentation parameters (shared post-processing)
SEGMENT_DIAMETER = None  # auto-estimate

# Post-processing
MIN_CELL_AREA = 50
MAX_CELL_AREA = 50000

# =========================================================================
# Feature extraction parameters
# =========================================================================

INTENSITY_PERCENTILES = [5, 25, 50, 75, 95]

# Texture (Haralick via mahotas)
GLCM_SCALES = (5, 10)
GLCM_LEVELS = 256
HARALICK_FEATURES = [
    "AngularSecondMoment", "Contrast", "Correlation", "Variance",
    "InverseDifferenceMoment", "SumAverage", "SumVariance", "SumEntropy",
    "Entropy", "DifferenceVariance", "DifferenceEntropy",
    "InfoMeas1", "InfoMeas2",
]

# Radial distribution
RADIAL_N_BINS = 4

# Granularity
GRANULARITY_SPECTRUM_LENGTH = 10

# Zernike moments
ZERNIKE_DEGREE = 9

# =========================================================================
# Aggregation parameters
# =========================================================================

MIN_CELLS_PER_FIELD = 5
AGGREGATION_FUNCS = ["median", "mad"]

# Feature selection
VARIANCE_THRESHOLD = 0.01
CORRELATION_THRESHOLD = 0.95

# =========================================================================
# Biomarker discovery parameters
# =========================================================================

# MoA classification
MOA_MIN_SAMPLES_PER_CLASS = 3  # for leave-one-compound-out CV

# Unsupervised clustering
UMAP_N_COMPONENTS = 2
UMAP_N_NEIGHBORS = 15  # smaller for ~100 compounds
UMAP_MIN_DIST = 0.1
UMAP_METRIC = "cosine"
UMAP_SEED = 42  # UMAP random_state (may need tuning per environment)

# Field-level clustering UMAP (higher dim to preserve structure for treatment aggregation)
FIELD_UMAP_N_COMPONENTS = 5
FIELD_UMAP_N_NEIGHBORS = 5

HDBSCAN_MIN_CLUSTER_SIZE = 5
HDBSCAN_MIN_SAMPLES = 3

# Feature attribution
ATTRIBUTION_ALPHA = 0.05
ATTRIBUTION_TOP_K = 20

# =========================================================================
# Utility
# =========================================================================

def ensure_dirs():
    """Create all necessary output directories."""
    for d in [DATA_ROOT, IMAGES_DIR, METADATA_DIR, RESULTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def ensure_model_dirs(model_name):
    """Create output directories for a specific model."""
    for fn in [masks_dir, features_dir, aggregation_dir, biomarker_dir, vis_dir]:
        fn(model_name).mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    ensure_dirs()
    print("BBBC021 Biomarker Discovery Configuration:")
    print(f"  Project root:  {PROJECT_ROOT}")
    print(f"  Cellpose root: {CELLPOSE_ROOT}")
    print(f"  Data root:     {DATA_ROOT}")
    print(f"  Compounds:     {N_COMPOUNDS}")
    print(f"  MoA classes:   {N_MOA_CLASSES}")
    print(f"  Total fields:  {N_FIELDS_TOTAL:,}")
    print(f"  Plate zips:    {len(PLATE_ZIP_NAMES)}")
