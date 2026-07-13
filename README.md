<div align="center">
<h1>MicroAtlas: Pushing the Limits of Cell Segmentation with Large-scale Unlabeled Microscopy Images</h1>

<a href='https://huggingface.co/datasets/MicroAtlas/MicroAtlas-2B'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-green' alt='Dataset'></a> <a href='https://huggingface.co/spaces/MicroAtlas/microatlas'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-blue' alt='Demo'></a>
</div>

Precise cell segmentation in microscopy images is essential for biological analysis. Although numerous promising AI models have been developed, their performance remains largely constrained by the scarcity of labeled datasets, which demand substantial and costly human annotation. Here we introduce MicroAtlas, a foundation model for cell segmentation that pioneers the use of large-scale unlabeled microscopy images. To build this model, we curate MicroAtlas-2B, a dataset comprising 5 million microscopy images from 45 diverse sources on the OpenMicroscopy platform. 
By generating pseudo segmentation labels for over 2 billion cells in the MicroAtlas-2B dataset, we propagate supervision from limited labeled data to vast unlabeled data via semi-supervised learning and model distillation, yielding a powerful and generalizable foundation model for cell segmentation. By **scaling training data from million- to billion-level**, MicroAtlas achieves promising performance across diverse cellular imaging data. We demonstrate that MicroAtlas has strong generalization capabilities and can provide a solid foundation for versatile biological analysis tasks, including cell counting, spatial transcriptomics analysis, morphological profiling, and multiplexed imaging cell phenotyping. 

![teaser](assets/fig1.png)


## Installation <a name="Installation"></a>

MicroAtlas is built on top of [**Cellpose**](https://github.com/MouseLand/cellpose). 

### Install with conda

Create a new environment:
   ```bash
   conda create --name microatlas python=3.12
   conda activate microatlas
   ```
Install Cellpose (with or without GUI):
   ```bash
   # with GUI
   python -m pip install cellpose[gui]
   # without GUI
   python -m pip install cellpose
   ```
Install additional dependencies:
   ```bash
   python -m pip install segment-anything pandas openpyxl tqdm pillow
   ```

## Quick start <a name="QuickStart"></a>

You can download our model from [huggingface](https://huggingface.co/MicroAtlas/microatlas-model) and place it at 'src/microatlas/'.

### Inference

#### Python API

```python
from cellpose import models

# Load MicroAtlas model
model = models.CellposeModel(gpu=True, pretrained_model='./microatlas/microatlas')

# Run segmentation
masks = model.eval(img, diameter=None, channels=None, bsize=256)[0]
```

#### Cellpose GUI

MicroAtlas can be used directly in the Cellpose GUI for interactive segmentation:

1. Launch the GUI:
   ```bash
   python -m cellpose --gui
   ```
2. In the menu bar, go to **Models → Add model**.
3. Navigate to `src/microatlas/` and select the `microatlas` model file.
4. The model will appear in the **user-trained models** dropdown. Select it and click **run** to segment the loaded image.

> **Note:** The GUI requires `cellpose[gui]` to be installed (see [Installation](#Installation)).

#### napari

Alternatively, you can use MicroAtlas in [**napari**](https://napari.org/) via the `cellpose-napari` plugin, which provides better multi-dimensional visualization and a richer plugin ecosystem:

```bash
python -m pip install napari[all] cellpose-napari
```

1. Launch napari:
   ```bash
   napari
   ```
2. Open **Plugins → cellpose-napari**.
3. Set **Model type** to `custom` and point to `src/microatlas/microatlas`.
4. Click **Run segmentation**. The result appears as a labels layer that you can edit directly with napari's brush tools.

### Dataset download

The MicroAtlas-2B dataset is available on [Hugging Face](https://huggingface.co/datasets/MicroAtlas/MicroAtlas-2B). It contains two parts:

- **Labeled datasets** — 18 public cell segmentation datasets with human-annotated masks, used for supervised training.
- **Unlabeled IDR datasets** — 45 studies from the [OpenMicroscopy IDR](https://idr.openmicroscopy.org/) platform with pseudo segmentation masks, used for semi-supervised training.

#### Install the Hugging Face CLI

```bash
python -m pip install huggingface_hub
```

#### Download labeled datasets

The 18 labeled datasets are organized under `./src/data/`, each containing `.tif` images with corresponding `_mask.tif` and `_mask_flows.tif` files:

```
data/
├── Cellpose/
│   └── train/
│       ├── img_001.tif
│       └── img_001_mask.tif
├── livecell/
│   └── train/
├── tissuenet/
│   └── train/
├── Deepbacs/
│   └── train/
├── MoNuSAC/
│   └── train/
├── MoNuSeg/
│   └── train/
├── ...                  # 18 datasets in total
└── all/
    └── test/            # evaluation benchmarks
```

Download from Hugging Face:

```bash
huggingface-cli download MicroAtlas/MicroAtlas-2B --repo-type dataset --include "data/*" --local-dir ./
```

#### Download unlabeled IDR datasets

The 45 IDR studies are split into images (`./src/IDR_image/`) and pseudo masks (`./src/IDR_mask/`), organized by study ID:

```
IDR_image/
├── idr0001/
│   ├── img_001.tif
│   └── ...
├── idr0002/
│   └── ...
└── ...                  # 45 IDR studies

IDR_mask/
├── idr0001/
│   ├── img_001_mask.tif
│   └── ...
├── idr0002/
│   └── ...
└── ...
```

Download from Hugging Face:

```bash
huggingface-cli download MicroAtlas/MicroAtlas-2B --repo-type dataset --include "IDR_image/*" --local-dir ./
huggingface-cli download MicroAtlas/MicroAtlas-2B --repo-type dataset --include "IDR_mask/*" --local-dir ./
```

> **Note:** The full dataset is ~46 TB (after unzip). If you only need a subset, use `--include` with specific study paths (e.g., `--include "IDR_image/idr0001/*"`).

### Training

MicroAtlas uses semi-supervised learning with a teacher–student framework and Gram loss. To start training:

```bash
cd src
python microatlas_train.py --root ./microatlas --n_epochs 500 --learning_rate 5e-5 --batch_size 1 --ddp 1 --multi_gpu 1
```

Key arguments:

| Argument | Default        | Description |
|---|----------------|---|
| `--root` | `./microatlas` | Data and model output root directory |
| `--n_epochs` | `500`          | Number of training epochs |
| `--learning_rate` | `1e-4`         | Learning rate (with warm-up + cosine decay) |
| `--batch_size` | `8`            | Batch size per GPU |
| `--gram_weight` | `1.0`          | Weight for Gram consistency loss |
| `--ddp` | `1`            | Use DistributedDataParallel (1) or DataParallel (0) |
| `--multi_gpu` | `1`            | Enable multi-GPU training |
| `--pretrained_model` | `None`         | Path to a pretrained model for resuming training |
| `--add_noise` | `1`            | Apply noise augmentation to unlabeled data |
| `--warmup_epochs` | `3`            | Linear warm-up epochs before cosine decay |

### Segmentation evaluation

Segmentation evaluation scripts for all compared models are provided under `src/eval/`:

```bash
cd src
# Evaluate MicroAtlas
python eval/eval_microatlas.py
# Evaluate Cellpose3
python eval/eval_cellpose3.py
# Evaluate Cellpose4 / Cellpose-SAM
python eval/eval_cellpose4.py
# Evaluate CellSAM
python eval/eval_cellsam.py
# Evaluate MicroSAM
python eval/eval_microsam.py
```

### SPATCH benchmark

Cell segmentation evaluation on 12 spatial transcriptomics datasets (4 platforms × 3 cancer types), measured by AP@0.5. Data is organized under `src/spatch/` with `tile*.png` images and `mask*.json` annotations. The datasets are from [SPATCH](https://spatch.pku-genomics.org). 

```bash
cd src

# Evaluate all models (microatlas, cellpose4, cellpose3, cellsam, microsam)
python spatch/eval_spatch.py

# Evaluate specific models
python spatch/eval_spatch.py --models microatlas cellpose4
```

### Counting evaluation

Cell counting evaluation uses three [BBBC](https://bbbc.broadinstitute.org/) datasets (BBBC001, BBBC039, BBBC041). Dataset zip files are included under `src/counting/BBBC/`:

```bash
cd src/counting

# Batch-run all five models on BBBC001 and BBBC041
python batch_counting.py
# Or run a single model or dataset
python batch_counting.py --datasets BBBC001 --models microatlas
# Available models: cellpose4, cellpose3, microatlas, microsam, cellsam
```

### Spatial transcriptomics evaluation

Spatial transcriptomics evaluation uses three Xenium datasets from [10x Genomics](https://www.10xgenomics.com/datasets). Due to their large size, the data is not included in the repository and must be downloaded separately.

| Dataset | License | Link | Size |
|---|---|---|---|
| Human-pancreas | CC BY 4.0 | [ffpe-human-pancreas-with-xenium-multimodal-cell-segmentation-1-standard](https://www.10xgenomics.com/datasets/ffpe-human-pancreas-with-xenium-multimodal-cell-segmentation-1-standard) | ~6.5 GB |
| Human-lung | CC BY 4.0 | [preview-data-ffpe-human-lung-cancer-with-xenium-multimodal-cell-segmentation-1-standard](https://www.10xgenomics.com/datasets/preview-data-ffpe-human-lung-cancer-with-xenium-multimodal-cell-segmentation-1-standard) | ~19 GB |
| Mouse-colon | CC BY 4.0 | [fresh-frozen-mouse-colon-with-xenium-multimodal-cell-segmentation-1-standard](https://www.10xgenomics.com/datasets/fresh-frozen-mouse-colon-with-xenium-multimodal-cell-segmentation-1-standard) | ~24 GB |

#### Download spatial transcriptomics data

Download the `*_outs` directory from 10x Genomics and place it under the corresponding dataset folder:

```bash
cd src/spatial_analysis/Xenium

# Human-pancreas: ffpe-human-pancreas-with-xenium-multimodal-cell-segmentation-1-standard
#   Extract to: Human_pancreas/Xenium_V1_human_Pancreas_FFPE_outs/

# Human-lung: preview-data-ffpe-human-lung-cancer-with-xenium-multimodal-cell-segmentation-1-standard
#   Extract to: Human_lung/Xenium_V1_humanLung_Cancer_FFPE_outs/

# Mouse-colon: fresh-frozen-mouse-colon-with-xenium-multimodal-cell-segmentation-1-standard
#   Extract to: Mouse_colon/Xenium_V1_mouse_Colon_FF_outs/
```

#### Run segmentation

Each model must be run separately for each dataset. The working directory should be the repository root:

```bash
# MicroAtlas
python -u src/spatial_analysis/Xenium/Human_lung/microatlas_xenium_analysis/run_pipeline.py --model microatlas

# Cellpose4 / Cellpose-SAM (default)
python -u src/spatial_analysis/Xenium/Human_lung/microatlas_xenium_analysis/run_pipeline.py

# Cellpose3 (cyto3)
python -u src/spatial_analysis/Xenium/Human_lung/microatlas_xenium_analysis/run_pipeline.py --model cellpose3

# CellSAM
python -u src/spatial_analysis/Xenium/Human_lung/microatlas_xenium_analysis/run_pipeline.py --model cellsam

# MicroSAM (vit_l_lm)
python -u src/spatial_analysis/Xenium/Human_lung/microatlas_xenium_analysis/run_pipeline.py --model microsam
```

> Each model takes ~1-2h on a full morphology image. Use `--crop` to test on a small ROI first.

| Parameter | Default | Description |
|---|---|---|
| `--model` | `cellpose4` | `microatlas` / `cellpose4` / `cellpose3` / `cellsam` / `microsam` |
| `--crop` | off | Use cropped ROI for quick testing |
| `--gpu` | `0` | GPU device index (`-1` for CPU) |
| `--output` | `output/` | Output directory |

Output per model (saved under `output/`):

| Model | Mask file | Overlay |
|---|---|---|
| `microatlas` | `masks_microatlas_morphology.npy` | `seg_overlay_microatlas_morphology.png` |
| `cellpose4` | `masks_cellpose4_morphology.npy` | `seg_overlay_cellpose4_morphology.png` |
| `cellpose3` | `masks_cellpose3_morphology.npy` | `seg_overlay_cellpose3_morphology.png` |
| `cellsam` | `masks_cellsam_morphology.npy` | `seg_overlay_cellsam_morphology.png` |
| `microsam` | `masks_microsam_morphology.npy` | `seg_overlay_microsam_morphology.png` |

Repeat the above for all three datasets (`Human_lung`, `Human_pancreas`, `Mouse_colon`).

After all models finish segmentation, run `eval.py` to compute the unassigned transcript fraction (spatial bleeding) — the proportion of transcripts that fall outside mask boundaries:

```bash
python spatial_analysis/Xenium/eval.py --project all --models all
```


### Morphological profiling

Morphological profiling evaluation uses the **BBBC021 dataset** from the [Broad Bioimage Benchmark Collection](https://bbbc.broadinstitute.org/BBBC021), a high-throughput fluorescence microscopy screen of MCF-7 breast cancer cells stained with three channels (DAPI, Actin, Tubulin). A library of 113 compounds was assayed at 8 concentrations across 13,200 fields (~44 GB). 38 compounds (103 compound–concentration pairs) are annotated with one of 12 Mechanisms of Action (MoA), plus DMSO as negative control. The pipeline benchmarks five segmentation models for their ability to preserve morphological signal, measured by how well downstream unsupervised clustering separates MoA classes.

#### Download data

Download metadata and plate images from [BBBC](https://bbbc.broadinstitute.org/BBBC021):

```bash
# Download metadata (~4 MB)
python src/morphology_profiling/download.py --metadata_only

# Download 1–2 plates for testing (~800 MB each)
python src/morphology_profiling/download.py --plate Week1_22123

# Download all 55 plates (~44 GB)
python src/morphology_profiling/download.py --all
```

Data is stored under `src/morphology_profiling/data/`: metadata CSVs in `metadata/`, plate images organized as `images/{plate}/DAPI|Actin|Tubulin/*.tif`.

#### Pipeline

All scripts support `--models` to specify one or more models (`cellpose4`, `cellpose3`, `microatlas`, `microsam`, `cellsam`, or `all`).

**Step 1 — Preprocess metadata:** Parse compound/concentration/MoA mappings and build a unified image table.

```bash
python src/morphology_profiling/preprocess.py
```

**Step 2 — Segmentation:** Run instance segmentation for all fields.

```bash
python src/morphology_profiling/segment.py --models microatlas --all
```

Output: `results/masks/{model}/{plate}/{field}_mask.tif`.

**Step 3 — Feature extraction:** Extract ~106 morphological features per cell across 6 categories:

| Category | Description | Features |
|---|---|---|
| AreaShape | regionprops shape descriptors | 13 |
| Intensity | batched ndimage stats + percentiles × 3ch | 38 |
| Texture | Haralick GLCM on DAPI + Actin, 1 scale | 26 |
| Granularity | multi-scale opening on DAPI, 5 scales | 5 |
| RadialDistribution | binned radial intensity × 3ch | 12 |
| Correlation | Pearson + Manders inter-channel | 12 |

```bash
python src/morphology_profiling/feature_extraction.py --models microatlas --all
```

**Step 4 — Aggregation & normalization:** Aggregate single-cell features to field-level (median + MAD), robust z-score against DMSO controls, remove low-variance (< 0.01) and highly correlated (> 0.95) features.

```bash
python src/morphology_profiling/feature_aggregation.py --models microatlas
```

**Step 5 — Unsupervised clustering & evaluation:** Field-level profiles are reduced via PCA (50 dims) then UMAP (5D, cosine distance, `init=pca`), aggregated to treatment-level centroids, and clustered via HDBSCAN (density-based, no preset k). Quality is measured by Hungarian-matched Accuracy — the proportion of 87 valid treatments correctly assigned to their MoA class under optimal cluster-to-MoA mapping.

```bash
python src/morphology_profiling/biomarker/unsupervised.py --models microatlas
```


### Multiplexed imaging cell phenotyping

Cell phenotyping evaluation uses the public **TNBC-MIBI dataset** [Keren et al., 2018](https://www.cell.com/cell/fulltext/S0092-8674(18)31457-2). The evaluation consists of three phases: **(1) Segmentation & AP Assessment** — generating composite RGB images from raw marker TIFs, running model inference; **(2) Expression Extraction & Comparison** — extracting per-cell mean intensity via regionprops, arcsinh transforming, per-patient z-score normalizing, and matching predicted to GT cells via IoU>0.5 for per-marker comparison; **(3) FlowSOM Hierarchical Clustering** — performing three-level immune phenotyping (immune vs non-immune, non-immune subtypes, immune subtypes) and evaluating via Hungarian-matched accuracy.

#### Download data

The dataset is publicly available at [https://www.angelolab.com/mibi-data](https://www.angelolab.com/mibi-data). Download the following and place them under `src/phenotyping/`:

| Data | Path | Description |
|---|---|---|
| Raw marker TIFs | `TNBC/TNBCShareData/Point{pid}/*.tif` | 40 protein markers per patient, 1024×1024 float32 |
| GT segmentation masks | `TNBC_shareCellData/p{pid}_labeledcellData.tiff` | 41 uint16 label maps (~250K cells total) |
| Single-cell expression matrix | `TNBC_shareCellData/cellData.csv` | ~85 MB, arcsinh-transformed, with `Group` (6 classes) and `immuneGroup` (12 classes) labels |
| Patient classification | `TNBC_shareCellData/patient_class.csv` | Molecular subtype of each patient |

#### Generate composite input images

For models requiring RGB input, raw marker TIFs are synthesized into 3-channel composites:

| Channel | Markers summed | Biological meaning |
|---|---|---|
| R | Pan-Keratin + Beta-catenin | Tumor membrane/cytoplasm |
| G | CD45 + HLA-DR | Immune membrane |
| B | dsDNA + H3K27me3 + H3K9ac | Nucleus |

```bash
python src/phenotyping/analysis/gen_composite_preview.py
```

Output: `src/phenotyping/composite_preview/PointXX_composite.tif` (float32) and `.png` (preview).

#### Run evaluation

Each model runs under the same 3-phase framework controlled by `--phase`:

| `--phase` | Stages | Description |
|---|---|---|
| `all` (default) | 1+2+3 | Full pipeline: segment → express → cluster |
| `segment` | 1 only | Segmentation |
| `express` | 2 only | Expression extraction + GT comparison |
| `cluster` | 3 only | FlowSOM hierarchical clustering (requires `pred_expr/*.parquet` from phase 2) |

> **Note:** FlowSOM clustering (phase 3) requires Python 3.10+. If your environment is Python 3.9, run `--phase all` or `--phase express` first, then run `--phase cluster` in a Python 3.10+ environment.

```bash
# MicroAtlas
python src/phenotyping/analysis/eval_microatlas_TNBC.py --gpu --weights ./microatlas/microatlas
```

Output per model saved under `src/phenotyping/eval_results/{model_name}/`:

| Phase | Output files |
|---|---|
| Segment | `ap_results.txt`, `ap_per_image.xlsx`, `masks/p{pid}_pred.tif` |
| Express | `expression_comparison.txt`, `expression_comparison.xlsx`, `pred_expr/p{pid}.parquet` |
| Cluster | `clustering_results.txt`, `clustering_results.xlsx` |


## Acknowledgement <a name="Acknowledgment"></a>

Our dataset is collected from the [**OpenMicroscopy platform**](https://idr.openmicroscopy.org/).
Our model is developed on the prestigious [**Cellpose**](https://github.com/MouseLand/cellpose). We highly appreciate their great efforts. 




