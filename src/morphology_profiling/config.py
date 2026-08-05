"""
config.py
=========
Configuration for the U2OS-Cell-Painting Biomarker Discovery pipeline.

Dataset: Human U2OS osteosarcoma cells treated with 231 compounds spanning 10
         mechanism-of-action (MoA) classes (plus DMSO negative controls), imaged
         with 5-channel Cell Painting fluorescence microscopy. Compounds were
         dosed at 10 uM for 48 h in 384-well plates, replicated 6x and spread
         across 18 microplates. Images are 16-bit, 2160x2160, 5 sites/well.
Source:  figshare record 21378906, DOI 10.17044/scilifelab.21378906 (CC BY 4.0),
         Uppsala University. Each plate is one tar.gz containing FL (5 channels)
         and BF (6 z-planes); this pipeline uses only the FL channels.
Papers:  Gupta/Harrison 2022 (bioRxiv 2022.10.12.511869) and
         Tian/Harrison 2023 (AILS, 10.1016/j.ailsci.2023.100060). BOTH papers do
         SUPERVISED MoA classification (ResNet-50 / EfficientNet-B1). This
         pipeline instead performs UNSUPERVISED morphological clustering, so the
         numbers here are NOT comparable to the papers' supervised scores.

This config feeds the shared downstream
pipeline (segment -> feature_extraction -> feature_aggregation -> biomarker).
Key characteristics of this dataset:
  - Images come from figshare per-plate tar.gz archives (not the S3 gallery)
  - MoA labels + per-channel filenames come from fl_data.csv (no external join)
  - 10 balanced MoA classes -> DEFAULT_N_CLUSTERS = 10
  - 5 fluorescence channels ordered C1..C5 = DNA, Mito, AGP, RNA, ER
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
# Scratch directory for per-plate tar.gz archives (deleted after extraction).
ARCHIVE_DIR = DATA_ROOT / "archives"
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

# Authoritative metadata table (extracted from data_tables.tar.gz). Provides
# plate, well, site, bf_site, compound, C1..C5 filenames and the single MoA
# label per field -- no external platemap/compound join is needed.
FL_DATA_CSV = METADATA_DIR / "fl_data.csv"
BF_DATA_CSV = METADATA_DIR / "bf_data.csv"

# =========================================================================
# Channel definitions (Cell Painting, 5 fluorescence channels)
# =========================================================================
# Channel order matches the C1..C5 columns / _w1.._w5 filename tokens in
# fl_data.csv. The stain assignment is fixed by the paper's acquisition setup
# (excitation order Hoechst, MitoTracker, Phalloidin+WGA, SYTO14, ConA) which
# is identical to the CorrectIllumination module order in the authors'
# CellProfiler pipeline (HOECHST, MITO, PHAandWGA, SYTO, CONC). ImageXpress
# writes the wavelength index (_w1.._w5) in that acquisition order.
# NOTE: the brightfield (BF) z-planes are intentionally ignored.

CHANNELS = {
    "DNA":  {"description": "DNA stain (Hoechst 33342)",                 "organelle": "Nucleus"},
    "Mito": {"description": "Mitochondria (MitoTracker Deep Red)",       "organelle": "Mitochondria"},
    "AGP":  {"description": "Actin/Golgi/plasma membrane (Phalloidin + WGA)", "organelle": "Cytoskeleton / membrane"},
    "RNA":  {"description": "RNA (SYTO 14)",                             "organelle": "Nucleoli / cytoplasmic RNA"},
    "ER":   {"description": "Endoplasmic reticulum (Concanavalin A / Alexa 488)", "organelle": "ER"},
}

CHANNEL_NAMES = list(CHANNELS.keys())  # ["DNA", "Mito", "AGP", "RNA", "ER"]
NUCLEUS_CHANNEL = "DNA"

# fl_data.csv column giving the raw TIFF filename for each channel.
CHANNEL_CSV_COLS = {
    "DNA":  "C1",   # Hoechst      (w1)
    "Mito": "C2",   # MitoTracker  (w2)
    "AGP":  "C3",   # Pha + WGA    (w3)
    "RNA":  "C4",   # SYTO 14      (w4)
    "ER":   "C5",   # ConA         (w5)
}

# Whole-cell segmentation input channels (unified whole-cell instance masks).
#   - cellpose4 / cellpose3 / microatlas / cellsam: 2 markers = cytoplasm + nucleus
#   - microsam:                            single cytoplasm marker
# AGP (Actin/Golgi/plasma membrane) is the best whole-cell boundary marker.
SEG_CYTO_CHANNEL = "AGP"      # cytoplasm / membrane boundary marker
SEG_NUCLEUS_CHANNEL = "DNA"   # nucleus marker (paired with cyto for cellpose family)

# Channels used for Haralick texture features (nucleus + one bright organelle).
TEXTURE_CHANNELS = ["DNA", "RNA"]

# Image properties
IMAGE_DTYPE = "uint16"
IMAGE_BIT_DEPTH = 16

# =========================================================================
# Dataset dimensions
# =========================================================================

N_COMPOUNDS = 231         # perturbagens (excluding DMSO)
N_CONCENTRATIONS = 1      # single concentration (10 uM)
N_MOA_CLASSES = 10        # mechanism-of-action classes (balanced, excl. DMSO)
N_REPLICATE_WELLS = 6     # compound-level replicates across the 18 plates
N_SITES_PER_WELL = 5      # FL fields of view per well (sites s2,s4,s5,s6,s8)
DEFAULT_CONCENTRATION = 10.0  # uM

# FL site identifiers used in fl_data.csv (BF sites 1..5 map to these).
FL_SITES = [2, 4, 5, 6, 8]

# Canonical MoA labels exactly as they appear in fl_data.csv "moa" column,
# with the compound counts reported in the dataset README.
MOA_CLASSES = {
    "ATPase inhibitor":                 18,
    "aurora kinase inhibitor":          20,
    "HDAC inhibitor":                   33,
    "HSP inhibitor":                    24,
    "JAK inhibitor":                    21,
    "PARP inhibitor":                   21,
    "protein synthesis inhibitor":      23,
    "retinoid receptor agonist":        19,
    "topoisomerase inhibitor":          32,
    "tubulin polymerization inhibitor": 20,
}
# The negative-control label in fl_data.csv (excluded from MoA clustering, used
# only as the per-plate normalization baseline).
DMSO_MOA_LABEL = "dmso"

# =========================================================================
# figshare download configuration
# =========================================================================

FIGSHARE_ARTICLE_ID = 21378906
FIGSHARE_DOI = "10.17044/scilifelab.21378906"

# Small metadata / benchmark archives (downloaded and extracted once).
SMALL_FILES = {
    "README.txt":         "https://ndownloader.figshare.com/files/39373556",
    "data_tables.tar.gz": "https://ndownloader.figshare.com/files/37984380",
    "CP_features.tar.gz": "https://ndownloader.figshare.com/files/37984449",
    "grit_scores.tar.gz": "https://ndownloader.figshare.com/files/37985649",
}

# Per-plate raw-image archives: plate -> (download_url, approx_size_GB).
# Ordered small -> large so pilot runs and low-disk pipelines start cheap.
PLATE_TARBALLS = {
    "P015080": ("https://ndownloader.figshare.com/files/37986315",  4.608),
    "P015081": ("https://ndownloader.figshare.com/files/37984389",  4.633),
    "P015082": ("https://ndownloader.figshare.com/files/37986063",  5.724),
    "P015083": ("https://ndownloader.figshare.com/files/37984458",  5.740),
    "P015076": ("https://ndownloader.figshare.com/files/37988091",  7.832),
    "P015077": ("https://ndownloader.figshare.com/files/37983477",  8.521),
    "P015092": ("https://ndownloader.figshare.com/files/37986075", 23.001),
    "P015093": ("https://ndownloader.figshare.com/files/37987962", 23.214),
    "P015096": ("https://ndownloader.figshare.com/files/37984503", 27.986),
    "P015097": ("https://ndownloader.figshare.com/files/37987326", 28.091),
    "P015094": ("https://ndownloader.figshare.com/files/37988391", 29.318),
    "P015095": ("https://ndownloader.figshare.com/files/37983156", 29.775),
    "P015084": ("https://ndownloader.figshare.com/files/37984977", 51.678),
    "P015085": ("https://ndownloader.figshare.com/files/37987770", 51.772),
    "P015090": ("https://ndownloader.figshare.com/files/37983639", 79.484),
    "P015091": ("https://ndownloader.figshare.com/files/37987500", 80.187),
    "P015098": ("https://ndownloader.figshare.com/files/37988268", 85.671),
    "P015099": ("https://ndownloader.figshare.com/files/37985868", 86.447),
}

PLATE_NAMES = list(PLATE_TARBALLS.keys())
DEFAULT_PLATE = "P015080"  # smallest plate; sensible single-plate default

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
        # cellsam is fed an (H, W, 3) uint8 [AGP, DNA, 0] image (same cyto+nucleus
        # signal as the other 4 models). segment.py:
        #   1. Monkey-patches AnchorDETR's nested_tensor_from_tensor_list so a
        #      bare 3D (C,H,W) tensor gets the missing batch dim (cellSAM
        #      0.0.dev1 otherwise raises "ValueError: not supported").
        #   2. Runs the bare cellsam_pipeline(img); cellSAM resizes internally
        #      to its native grid and the mask is resized back to (H, W).
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
MOA_MIN_SAMPLES_PER_CLASS = 2

# Unsupervised clustering
UMAP_N_COMPONENTS = 2
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
UMAP_METRIC = "cosine"
UMAP_SEED = 42

# Field-level clustering UMAP (higher dim to preserve structure for aggregation)
FIELD_UMAP_N_COMPONENTS = 5
FIELD_UMAP_N_NEIGHBORS = 5

HDBSCAN_MIN_CLUSTER_SIZE = 5
HDBSCAN_MIN_SAMPLES = 3

# Number of clusters for treatment-level Agglomerative clustering.
# Defaults to the number of MoA classes (10); override via --n_clusters.
DEFAULT_N_CLUSTERS = N_MOA_CLASSES

# Feature attribution
ATTRIBUTION_ALPHA = 0.05
ATTRIBUTION_TOP_K = 20

# =========================================================================
# Utility
# =========================================================================

def ensure_dirs():
    """Create all necessary output directories."""
    for d in [DATA_ROOT, IMAGES_DIR, METADATA_DIR, ARCHIVE_DIR, RESULTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def ensure_model_dirs(model_name):
    """Create output directories for a specific model."""
    for fn in [masks_dir, features_dir, aggregation_dir, biomarker_dir, vis_dir]:
        fn(model_name).mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    ensure_dirs()
    print("U2OS-Cell-Painting Biomarker Discovery Configuration:")
    print(f"  Project root:  {PROJECT_ROOT}")
    print(f"  Cellpose root: {CELLPOSE_ROOT}")
    print(f"  Data root:     {DATA_ROOT}")
    print(f"  Channels:      {CHANNEL_NAMES} (nucleus={NUCLEUS_CHANNEL}, cyto={SEG_CYTO_CHANNEL})")
    print(f"  Compounds:     {N_COMPOUNDS}")
    print(f"  MoA classes:   {N_MOA_CLASSES} (+ DMSO control)")
    print(f"  Plates:        {len(PLATE_NAMES)} (default {DEFAULT_PLATE})")
    print(f"  figshare:      {FIGSHARE_DOI}")
