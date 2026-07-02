# Counting Evaluation Script Run Guide

> **Working directory**: `d:\code\Cellpose` (all commands executed from this path)

## Overview

Cell counting evaluation pipeline, supports batch running of 5 segmentation models on 6 BBBC datasets:

| Script | Model | Dependency |
|------|------|------|
| `eval_cellpose4_counting.py` | Cellpose4 (cpsam) | cellpose |
| `eval_cellpose3_counting.py` | Cellpose3 (cyto3) | cellpose-cp3 (local) |
| `eval_microatlas_counting.py` | Microatlas | cellpose |
| `eval_cellsam_counting.py` | CellSAM | cellSAM |
| `eval_microsam_counting.py` | MicroSAM | micro-sam (conda) |

Shared logic in `counting_utils.py`, batch entry point `batch_counting.py`.

## Datasets

| Dataset | Images | Format | Description |
|--------|--------|---------|------|
| BBBC001 | 6 | 8-bit TIF | Human colon cancer cells (6 fields, mean of dual manual counts) |
| BBBC002 | 50 | 16-bit TIF | Drosophila Kc167 cells (mean of dual manual counts) |
| BBBC003 | 15 | 8-bit TIF | Mouse embryos (single manual count) |
| BBBC005 | 9,600 | 8-bit TIF, 696x520 | Synthetic cells (w2 nuclear stain only, Bray CSV GT) |
| BBBC006 | 768 | 16-bit TIF, 696x520 | U2OS cells (z=16 best focal plane only, CellProfiler auto count) |
| BBBC041 | 1,328 | 8-bit PNG, 256x256 | Malaria-infected red blood cells (single manual count) |

Data directory structure:
```
counting/BBBC/
├── BBBC001/
│   ├── BBBC001_v1_images_tif/human_ht29_colon_cancer_1_images/
│   └── BBBC001_v1_counts.txt
├── BBBC002/
│   ├── BBBC002_v1_images/drosophila_kc167_1_images/
│   └── BBBC002_v1_counts.txt
├── BBBC003/
│   ├── BBBC003_v1_images/mouse_embryos_dic_images/
│   └── BBBC003_v1_counts.txt
├── BBBC005/
│   ├── BBBC005_v1_images/
│   └── BBBC005_v1_counts.txt    (9,600 entries)
├── BBBC006/
│   ├── BBBC006_v1_images/
│   └── BBBC006_v1_counts.txt    (768 entries)
├── BBBC041/
│   ├── malaria/images/
│   └── BBBC041_v1_counts.txt    (1,328 entries)
```

## Dependency Installation

```bash
# Base dependencies
pip install numpy scipy scikit-learn pandas tqdm openpyxl

# Cellpose4 / Microatlas
pip install cellpose

# Cellpose3 (local cellpose-cp3 package)
pip install -e d:/code/Cellpose/cellpose-cp3

# CellSAM
pip install cellSAM

# MicroSAM (requires conda)
conda install -c conda-forge micro-sam
```

## Run Commands

### Batch Run (Recommended)

```bash
# Default: run all 5 models on all 6 datasets
python counting/batch_counting.py

# Specify dataset(s) + model(s)
python counting/batch_counting.py --datasets BBBC041 --models cellpose4

# Run cellpose4 only
python counting/batch_counting.py --models cellpose4

# View tasks only (dry run, no execution)
python counting/batch_counting.py --dry-run

# CPU mode
python counting/batch_counting.py --cpu

# Only regenerate summary Excel from existing results
python counting/batch_counting.py --excel-only
```

### Single Dataset Run

```bash
# Cellpose4
python counting/eval_cellpose4_counting.py --dataset BBBC041 \
    --image_dir counting/BBBC/BBBC041/malaria/images \
    --counts counting/BBBC/BBBC041/BBBC041_v1_counts.txt \
    --output_dir counting/results

# Cellpose3 (diameter=0 for auto estimation)
python counting/eval_cellpose3_counting.py --dataset BBBC041 \
    --image_dir counting/BBBC/BBBC041/malaria/images \
    --counts counting/BBBC/BBBC041/BBBC041_v1_counts.txt \
    --output_dir counting/results --diameter 0

# MicroSAM (specify model type)
python counting/eval_microsam_counting.py --dataset BBBC041 \
    --image_dir counting/BBBC/BBBC041/malaria/images \
    --counts counting/BBBC/BBBC041/BBBC041_v1_counts.txt \
    --output_dir counting/results --model_type vit_l_lm
```

## Output Structure

```
counting/results/
├── BBBC005_cellpose4_counting/
│   ├── results_per_image/
│   │   └── cellpose4_BBBC005.csv       (per-image GT vs Pred)
│   └── results_metrics/
│       └── cellpose4_BBBC005_metrics.txt (aggregate metrics)
├── BBBC006_cellpose4_counting/
│   └── ...
├── results_summary.xlsx                 (summary Excel)
└── ...
```

## Metrics

| Metric | Description | Unit |
|------|------|------|
| MAE | Mean Absolute Error | cell count |
| RMSE | Root Mean Square Error | cell count |
| R² | Coefficient of Determination | [0, 1] |
| Pearson r | Pearson Correlation Coefficient | [-1, 1] |
| MPE | Mean Percentage Error | % |

## Dataset Preparation

To re-download or add datasets, use the preparation script:

```bash
# Download and process all datasets
python counting/prepare_bbbc_datasets.py

# Specify datasets
python counting/prepare_bbbc_datasets.py --datasets 005,006
```

Note: BBBC044 (mouse hippocampal presynaptic terminals) has been removed due to data characteristics incompatible with the current 2D counting pipeline.
