"""
Spatch benchmark: evaluate microsam, cellsam, cellpose3, cellpose4, microatlas
on 12 datasets (4 platforms x 3 cancer types), metric = AP@0.5.

Data structure:
  spatch/{CosMx6K,Xenium5K,VisiumHD,Stereo}_{COAD,HCC,OV}/tile{1-5}.png
  spatch/{CosMx6K,Xenium5K}_{COAD,HCC,OV}/mask{1-5}.json  (labelme polygon)

Usage:
  cd Cellpose/
  python spatch/eval_spatch.py [--models microsam cellsam cellpose3 cellpose4 microatlas]
"""

import argparse
import json
import os
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm, trange
from skimage.draw import polygon as skimage_polygon

from cellpose import metrics as cp_metrics

# ──────────────────────────────────────────────
# 1. Data loading
# ──────────────────────────────────────────────

def labelme_json_to_instance_mask(json_path):
    """Convert a labelme JSON polygon file to a uint16 instance mask.
    Each polygon shape gets a unique integer ID (1, 2, ...).
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    h = data["imageHeight"]
    w = data["imageWidth"]
    mask = np.zeros((h, w), dtype=np.uint16)

    for cell_id, shape in enumerate(data["shapes"], start=1):
        pts = np.array(shape["points"])  # (N, 2)  [x, y]
        rr, cc = skimage_polygon(pts[:, 1], pts[:, 0], shape=(h, w))
        mask[rr, cc] = cell_id

    return mask


def darwin_json_to_instance_mask(json_path):
    """Convert a Darwin JSON v2.0 polygon file to a uint16 instance mask.
    Each annotation polygon gets a unique integer ID (1, 2, ...).
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    slot = data["item"]["slots"][0]
    h = slot["height"]
    w = slot["width"]
    mask = np.zeros((h, w), dtype=np.uint16)

    for cell_id, ann in enumerate(data["annotations"], start=1):
        # Each annotation may have multiple paths (outer + holes)
        poly_paths = ann.get("polygon", {}).get("paths", [])
        if not poly_paths:
            continue
        # Use the first (outer) path for the cell body
        pts = np.array([[p["x"], p["y"]] for p in poly_paths[0]])
        rr, cc = skimage_polygon(pts[:, 1], pts[:, 0], shape=(h, w))
        mask[rr, cc] = cell_id

    return mask


def load_mask_json(json_path):
    """Auto-detect JSON format and load instance mask."""
    with open(json_path, "r") as f:
        data = json.load(f)
    if "imageHeight" in data:
        return labelme_json_to_instance_mask(json_path)
    elif "item" in data:
        return darwin_json_to_instance_mask(json_path)
    else:
        raise ValueError(f"Unknown mask JSON format in {json_path}")


def load_spatch_data(dataset_dir):
    """Load all tile/mask pairs from a spatch sub-directory.

    Returns:
        imgs:      list of np.ndarray (H, W, 3) uint8 RGB
        gt_masks:  list of np.ndarray (H, W) uint16 instance masks
        tile_names: list of str, e.g. ['tile1', ..., 'tile5']
    """
    dataset_dir = Path(dataset_dir)
    tile_files = sorted(dataset_dir.glob("tile*.png"))
    mask_files = sorted(dataset_dir.glob("mask*.json"))
    assert len(tile_files) == len(mask_files), (
        f"Mismatch: {len(tile_files)} tiles vs {len(mask_files)} masks in {dataset_dir}"
    )

    imgs = []
    gt_masks = []
    tile_names = []
    for tf in tile_files:
        img = np.array(Image.open(tf).convert("RGB"))
        imgs.append(img)
        tile_names.append(tf.stem)

    for mf in mask_files:
        gt_masks.append(load_mask_json(mf))

    return imgs, gt_masks, tile_names


# ──────────────────────────────────────────────
# 2. AP computation
# ──────────────────────────────────────────────

def compute_ap(gt_masks, pred_masks):
    """Compute AP@0.5 (per-image mean and global).

    cellpose.metrics.average_precision returns:
        ap: shape (n_images, n_thresholds)
        tp, fp, fn: shape (n_images, n_thresholds)
    """
    threshold = np.arange(0.5, 1.0, 0.05)
    masks_gt = [m.astype("uint16") for m in gt_masks]
    n_images = len(masks_gt)

    # global computation (all images together)
    ap_all, tp_all, fp_all, fn_all = cp_metrics.average_precision(
        masks_gt, pred_masks, threshold=threshold
    )
    # ap_all shape: (n_images, n_thresholds)
    # override with tp/(tp+fp+fn) to match eval convention
    eps = 1e-10
    ap_all = tp_all / (tp_all + fp_all + fn_all + eps)
    # AP@0.5 = threshold index 0, mean across images
    ap_global_05 = float(ap_all[:, 0].mean())

    # per-image AP@0.5
    per_image_ap = []
    for i in range(n_images):
        ap_i, tp_i, fp_i, fn_i = cp_metrics.average_precision(
            [masks_gt[i]], [pred_masks[i]], threshold=threshold
        )
        # ap_i shape: (1, n_thresholds) -> scalar at [0, 0]
        eps_i = 1e-10
        ap_i = tp_i / (tp_i + fp_i + fn_i + eps_i)
        per_image_ap.append(float(ap_i[0, 0]))

    return ap_global_05, np.mean(per_image_ap), per_image_ap


# ──────────────────────────────────────────────
# 3. Model inference functions
# ──────────────────────────────────────────────

# ---------- microsam ----------
def run_microsam(imgs, gt_masks, dataset_name, device="cuda"):
    from micro_sam import util
    from micro_sam.instance_segmentation import (
        InstanceSegmentationWithDecoder,
        get_predictor_and_decoder,
    )

    predictor, decoder = get_predictor_and_decoder(
        model_type="vit_l_lm", checkpoint_path=None
    )

    masks_pred = []
    for i in trange(len(imgs), desc=f"microsam/{dataset_name}"):
        # convert to 3-channel grayscale (H, W, 3) float32
        img = imgs[i]
        if img.ndim == 3:
            gray = img.astype("float32").mean(axis=2)
        else:
            gray = img.astype("float32")
        image = np.stack((gray,) * 3, axis=-1)

        image_embeddings = util.precompute_image_embeddings(
            predictor=predictor, input_=image, ndim=2, verbose=False
        )
        ais = InstanceSegmentationWithDecoder(predictor, decoder)
        ais.initialize(image=image, image_embeddings=image_embeddings)
        prediction = ais.generate(output_mode="instance_segmentation")

        if prediction is not None and len(prediction) > 0:
            masks_pred.append(prediction.astype(np.uint16))
        else:
            masks_pred.append(np.zeros(image.shape[:2], dtype=np.uint16))

    return masks_pred


# ---------- cellsam ----------
def run_cellsam(imgs, gt_masks, dataset_name, device="cuda"):
    from cellSAM import cellsam_pipeline

    masks_pred = []
    for i in trange(len(imgs), desc=f"cellsam/{dataset_name}"):
        img = imgs[i]
        # cellsam accepts (H, W, 3) RGB images directly
        # spatch images are already RGB with signal in blue channel
        # use_wsi=True for large images (faster than processing whole image)
        mask = cellsam_pipeline(
            img,
            use_wsi=True,  # tiling for large images
            block_size=512,  # tile size
            overlap=64,  # overlap for merging
            low_contrast_enhancement=False,
            gauge_cell_size=False,
        )
        masks_pred.append(mask.astype(np.uint16))

    return masks_pred


# ---------- cellpose3 ----------
def run_cellpose3(imgs, gt_masks, dataset_name, device="cuda"):
    # Import cellpose-cp3 (v3) in isolation, then restore cellpose v4
    import sys
    import importlib

    cp3_path = str(Path(__file__).resolve().parent.parent / "cellpose-cp3")

    # Save current v4 modules
    v4_mods = {k: v for k, v in sys.modules.items() if k.startswith("cellpose")}

    # Remove all cellpose.* from cache
    for mod in list(v4_mods.keys()):
        del sys.modules[mod]

    # Insert cp3 path and import
    sys.path.insert(0, cp3_path)
    cp3_models = importlib.import_module("cellpose.models")

    model = cp3_models.Cellpose(gpu=True, model_type="cyto3")
    channels = [0, 0]
    out = model.eval(
        imgs, diameter=0, channels=channels,
        tile_overlap=0.1, flow_threshold=0.4, augment=False,
        batch_size=64, bsize=224, niter=1000,
    )
    if len(out) == 3:
        masks_pred, _, _ = out
    else:
        masks_pred, _, _, _ = out

    # Remove cp3 modules
    cp3_mods = {k: v for k, v in sys.modules.items() if k.startswith("cellpose")}
    for mod in cp3_mods:
        del sys.modules[mod]
    sys.path.remove(cp3_path)

    # Restore v4 modules
    sys.modules.update(v4_mods)

    return [m.astype(np.uint16) for m in masks_pred]


# ---------- cellpose4 ----------
def run_cellpose4(imgs, gt_masks, dataset_name, device="cuda"):
    from cellpose import models

    model = models.CellposeModel(gpu=True)
    # SAM / transformer mode: channels=None, data as (C, H, W)
    test_data = []
    for img in imgs:
        if img.ndim == 2:
            img = np.tile(img[np.newaxis], (3, 1, 1))
            img[1:] = 0
        else:
            img = img.transpose(2, 0, 1)  # (H,W,3) -> (3,H,W)
        test_data.append(img)

    net = model.net
    net.eval()
    masks_pred = model.eval(
        test_data, diameter=30.0, channels=None, niter=1000,
        batch_size=256, bsize=256,
    )[0]
    return [m.astype(np.uint16) for m in masks_pred]


# ---------- microatlas ----------
def run_microatlas(imgs, gt_masks, dataset_name, device="cuda"):
    from cellpose import models

    model = models.CellposeModel(gpu=True, pretrained_model="./microatlas/microatlas")
    test_data = []
    for img in imgs:
        if img.ndim == 2:
            img = np.tile(img[np.newaxis], (3, 1, 1))
            img[1:] = 0
        else:
            img = img.transpose(2, 0, 1)  # (H,W,3) -> (3,H,W)
        test_data.append(img)

    net = model.net
    net.eval()
    masks_pred = model.eval(
        test_data, diameter=30.0, channels=None, niter=1000,
        batch_size=256, bsize=256,
    )[0]
    return [m.astype(np.uint16) for m in masks_pred]


# ──────────────────────────────────────────────
# 4. Mask saving
# ──────────────────────────────────────────────

def save_mask_png(mask, save_path):
    """Save a uint16 instance mask as uint8 PNG (normalized to 0-255)."""
    mask_img = mask.astype(np.uint8)
    if mask_img.max() > 0:
        mask_img = (mask_img / mask_img.max() * 255).astype(np.uint8)
    Image.fromarray(mask_img).save(save_path)


# ──────────────────────────────────────────────
# 5. Main
# ──────────────────────────────────────────────

MODEL_REGISTRY = {
    "microsam":   run_microsam,
    "cellsam":    run_cellsam,
    "cellpose3":  run_cellpose3,
    "cellpose4":  run_cellpose4,
    "microatlas": run_microatlas,
}

PLATFORMS = ["CosMx6K", "Xenium5K", "VisiumHD", "Stereo"]
CANCER_TYPES = ["COAD", "HCC", "OV"]


def parse_dataset_name(name):
    """Parse 'CosMx6K_COAD' -> ('CosMx6K', 'COAD')."""
    for p in PLATFORMS:
        if name.startswith(p):
            ct = name[len(p) + 1:]  # skip underscore
            return p, ct
    return None, name


def main():
    parser = argparse.ArgumentParser(description="Spatch 5-model evaluation")
    parser.add_argument(
        "--models", nargs="+", default=list(MODEL_REGISTRY.keys()),
        choices=list(MODEL_REGISTRY.keys()),
    )
    parser.add_argument("--save_dir", type=str, default="spatch/predicted_masks")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--force", action="store_true", help="Force re-run even if predictions exist")
    args = parser.parse_args()

    # spatch_root is the directory where this script lives
    spatch_root = Path(__file__).resolve().parent
    save_dir = Path(args.save_dir)

    # results: {model: {dataset: ap_global_05}}
    results = {m: {} for m in args.models}
    # per-image results: {model: {dataset: {tile_name: ap50}}}
    results_per_image = {m: {} for m in args.models}

    dataset_dirs = sorted([d for d in spatch_root.iterdir() if d.is_dir() and not d.name.startswith(("predicted_masks", "."))])

    total_tasks = len(dataset_dirs) * len(args.models)
    pbar = tqdm(total=total_tasks, desc="Overall progress", unit="task")

    for dataset_dir in dataset_dirs:
        dataset_name = dataset_dir.name
        platform, cancer_type = parse_dataset_name(dataset_name)
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name}  (platform={platform}, cancer={cancer_type})")
        print(f"{'='*60}")

        imgs, gt_masks, tile_names = load_spatch_data(dataset_dir)
        print(f"Loaded {len(imgs)} tiles, GT cells: {[m.max() for m in gt_masks]}")

        # save GT masks
        gt_save_dir = save_dir / dataset_name / "gt"
        gt_save_dir.mkdir(parents=True, exist_ok=True)
        for idx, tname in enumerate(tile_names):
            save_mask_png(gt_masks[idx], gt_save_dir / f"{tname}_gt.png")
        # also save original tiles
        orig_save_dir = save_dir / dataset_name / "original"
        orig_save_dir.mkdir(parents=True, exist_ok=True)
        for idx, tname in enumerate(tile_names):
            Image.fromarray(imgs[idx]).save(orig_save_dir / f"{tname}.png")

        for model_name in tqdm(args.models, desc=f"{dataset_name}", leave=False):
            # Check if predictions already exist (skip if all tiles saved)
            pred_save_dir = save_dir / dataset_name / model_name
            pred_files = [pred_save_dir / f"{tname}_{model_name}_pred.png" for tname in tile_names]
            if not args.force and all(f.exists() for f in pred_files):
                print(f"{model_name}: predictions exist, skipping (loading from disk)")
                # Load existing predictions and compute AP
                pred_masks = [np.array(Image.open(f)).astype(np.uint16) for f in pred_files]
                ap_global, ap_mean_per_img, per_image_aps = compute_ap(gt_masks, pred_masks)
                results[model_name][dataset_name] = ap_global
                results_per_image[model_name][dataset_name] = {
                    tname: float(per_image_aps[i]) for i, tname in enumerate(tile_names)
                }
                print(f"{model_name} AP@0.5={ap_global:.4f}  per-img={ap_mean_per_img:.4f}  (loaded)")
                pbar.update(1)
                pbar.set_postfix({"dataset": dataset_name, "model": model_name, "AP": f"{ap_global:.4f}"})
                continue

            run_fn = MODEL_REGISTRY[model_name]
            tic = time.time()
            pred_masks = run_fn(imgs, gt_masks, dataset_name, device=args.device)
            elapsed = time.time() - tic

            # compute AP
            ap_global, ap_mean_per_img, per_image_aps = compute_ap(gt_masks, pred_masks)
            results[model_name][dataset_name] = ap_global
            results_per_image[model_name][dataset_name] = {
                tname: float(per_image_aps[i]) for i, tname in enumerate(tile_names)
            }
            print(f"{model_name} AP@0.5={ap_global:.4f}  per-img={ap_mean_per_img:.4f}  time={elapsed:.1f}s")

            # save predicted masks
            pred_save_dir.mkdir(parents=True, exist_ok=True)
            for idx, tname in enumerate(tile_names):
                save_mask_png(pred_masks[idx], pred_save_dir / f"{tname}_{model_name}_pred.png")

            pbar.update(1)
            pbar.set_postfix({"dataset": dataset_name, "model": model_name, "AP": f"{ap_global:.4f}"})

    pbar.close()

    # ── Summary ──
    print(f"\n{'='*60}")
    print("AP@0.5 Summary")
    print(f"{'='*60}")

    # build dataframe
    rows = []
    for dataset_dir in dataset_dirs:
        dn = dataset_dir.name
        row = {"dataset": dn}
        for m in args.models:
            row[m] = results[m].get(dn, np.nan)
        rows.append(row)
    df = pd.DataFrame(rows).set_index("dataset")
    print("\nPer-dataset AP@0.5:")
    print(df.to_string(float_format="%.4f"))

    # aggregate by cancer type (average across 2 platforms)
    print("\n\nBy cancer type (mean across platforms):")
    for ct in CANCER_TYPES:
        cols = [f"{p}_{ct}" for p in PLATFORMS]
        vals = {m: np.nanmean([results[m].get(c, np.nan) for c in cols]) for m in args.models}
        print(f"  {ct}: " + "  ".join(f"{m}={v:.4f}" for m, v in vals.items()))

    # aggregate by platform (average across 3 cancer types)
    print("\nBy platform (mean across cancer types):")
    for pl in PLATFORMS:
        cols = [f"{pl}_{ct}" for ct in CANCER_TYPES]
        vals = {m: np.nanmean([results[m].get(c, np.nan) for c in cols]) for m in args.models}
        print(f"  {pl}: " + "  ".join(f"{m}={v:.4f}" for m, v in vals.items()))

    # save to excel: one file per model
    save_dir.mkdir(parents=True, exist_ok=True)
    for m in args.models:
        excel_path = save_dir / f"spatch_results_{m}.xlsx"
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            # per-dataset global AP
            ds_rows = []
            for dataset_dir in dataset_dirs:
                dn = dataset_dir.name
                ds_rows.append({"dataset": dn, "AP@0.5": results[m].get(dn, np.nan)})
            # Add mean row
            ds_rows.append({"dataset": "mean", "AP@0.5": np.nanmean([r["AP@0.5"] for r in ds_rows])})
            pd.DataFrame(ds_rows).to_excel(writer, sheet_name="per_dataset", index=False)

            # by cancer type
            ct_rows = {}
            for ct in CANCER_TYPES:
                cols = [f"{p}_{ct}" for p in PLATFORMS]
                ct_rows[ct] = {"AP@0.5": np.nanmean([results[m].get(c, np.nan) for c in cols])}
            # Add mean row
            ct_rows["mean"] = {"AP@0.5": np.nanmean([v["AP@0.5"] for v in ct_rows.values()])}
            pd.DataFrame(ct_rows).T.to_excel(writer, sheet_name="by_cancer_type")

            # by platform
            pl_rows = {}
            for pl in PLATFORMS:
                cols = [f"{pl}_{ct}" for ct in CANCER_TYPES]
                pl_rows[pl] = {"AP@0.5": np.nanmean([results[m].get(c, np.nan) for c in cols])}
            # Add mean row
            pl_rows["mean"] = {"AP@0.5": np.nanmean([v["AP@0.5"] for v in pl_rows.values()])}
            pd.DataFrame(pl_rows).T.to_excel(writer, sheet_name="by_platform")

            # per-image AP: one sheet per dataset
            for dataset_dir in dataset_dirs:
                dn = dataset_dir.name
                _, _, tnames = load_spatch_data(dataset_dir)
                rows_img = []
                for tname in tnames:
                    rows_img.append({"tile": tname, "AP@0.5": results_per_image[m].get(dn, {}).get(tname, np.nan)})
                if rows_img:
                    pd.DataFrame(rows_img).to_excel(writer, sheet_name=f"images_{dn}", index=False)

        print(f"Results for {m} saved to: {excel_path}")


if __name__ == "__main__":
    main()
