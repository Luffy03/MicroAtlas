"""
CellSAM cell counting evaluation on BBBC datasets.

Usage:
  cd ~/Cellpose && python counting/eval_cellsam_counting.py --dataset BBBC041 ...
"""
import os
import sys
import time
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm, trange

# Add counting directory for counting_utils
_COUNTING_DIR = Path(__file__).resolve().parent
if str(_COUNTING_DIR) not in sys.path:
    sys.path.insert(0, str(_COUNTING_DIR))

import counting_utils as cu


def _convert_image_cellsam(img):
    """Convert image to CellSAM (H,W,3) format.

    CellSAM expects:
    - (H,W,3) uint8 or float array
    - grayscale -> stack as (H,W,3) with zeros for G/B channels
    """
    if img.ndim == 2:
        img = np.stack([img, np.zeros_like(img), np.zeros_like(img)], axis=-1)
    elif img.ndim == 3 and img.shape[2] == 1:
        img = np.concatenate([img, np.zeros_like(img), np.zeros_like(img)], axis=-1)
    elif img.ndim == 3 and img.shape[2] >= 3:
        img = img[:, :, :3]
    return img


def run_counting_evaluation(dataset_name, image_dir, counts_file,
                            output_dir, use_gpu=True):
    """Run CellSAM counting evaluation on one BBBC dataset."""
    try:
        from cellSAM import cellsam_pipeline
    except ImportError:
        print("  [ERROR] cellSAM not installed. Run: pip install cellSAM")
        return None

    device = "cuda" if use_gpu else "cpu"
    print(f"\n{'='*60}")
    print(f"CellSAM Counting: {dataset_name}")
    print(f"{'='*60}")
    print(f"  Image dir: {image_dir}")
    print(f"  Counts:    {counts_file}")

    # Load ground truth
    gt_dict = cu.load_counts(counts_file)
    print(f"  GT images: {len(gt_dict)}")

    # Locate and load images
    image_dir = Path(image_dir)
    filenames = sorted(gt_dict.keys())
    images = []
    valid_fnames = []
    for fname in tqdm(filenames, desc="  Loading images"):
        img_path = cu.locate_image(image_dir, fname)
        if img_path is None:
            print(f"  [WARN] Image not found: {fname}")
            continue
        img = cu.load_bbbc_image(img_path)
        img_cellsam = _convert_image_cellsam(img)
        images.append(img_cellsam)
        valid_fnames.append(fname)

    print(f"  Loaded images: {len(images)} / {len(filenames)}")

    if len(images) == 0:
        print("  [ERROR] No images loaded, aborting.")
        return None

    # Run inference
    print("  Loading CellSAM model and running inference...")
    t0 = time.time()
    masks_pred = []
    for i, img in enumerate(tqdm(images, desc="  Inference")):
        try:
            mask = cellsam_pipeline(img)
            masks_pred.append(mask)
        except Exception as e:
            print(f"    [WARN] Image {i} failed: {e}")
            masks_pred.append(np.zeros(img.shape[:2], dtype=np.uint16))
    elapsed = time.time() - t0
    print(f"  Inference done: {elapsed:.1f}s ({elapsed/len(images):.2f}s/img)")

    # Count instances
    pred_counts = [cu.count_instances(m) for m in masks_pred]
    gt_counts = [gt_dict[f] for f in valid_fnames]

    # Compute metrics
    metrics = cu.compute_counting_metrics(gt_counts, pred_counts)

    # Print summary
    print(f"\n  Results for {dataset_name}:")
    for key in ["MAE", "RMSE", "R2", "Pearson_r", "MPE", "n_images"]:
        val = metrics.get(key, "N/A")
        if isinstance(val, float):
            if key == "MPE":
                print(f"    {key:12s} = {val:.2f}%")
            else:
                print(f"    {key:12s} = {val:.4f}")
        else:
            print(f"    {key:12s} = {val}")

    # Save
    per_image = [
        {"filename": f, "gt_count": g, "pred_count": p}
        for f, g, p in zip(valid_fnames, gt_counts, pred_counts)
    ]
    cu.save_per_image_csv(output_dir, "cellsam", dataset_name, per_image)
    cu.save_metrics_txt(output_dir, "cellsam", dataset_name, metrics)

    return {
        "dataset": dataset_name,
        "model": "cellsam",
        "per_image": per_image,
        "metrics": metrics,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CellSAM cell counting evaluation")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--counts", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--cpu", action="store_true")

    args = parser.parse_args()
    run_counting_evaluation(
        dataset_name=args.dataset,
        image_dir=args.image_dir,
        counts_file=args.counts,
        output_dir=args.output_dir,
        use_gpu=not args.cpu,
    )
