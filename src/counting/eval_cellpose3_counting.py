"""
Cellpose3 (cyto3) cell counting evaluation on BBBC datasets.

Usage:
  cd ~/Cellpose && python counting/eval_cellpose3_counting.py --dataset BBBC041 ...
"""
import os
import sys
import time
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm, trange

# Add cellpose-cp3 to sys.path (same as tracking/cellpose3_tracking.py)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CP3_PATH = str(_PROJECT_ROOT / "cellpose-cp3")
if _CP3_PATH not in sys.path:
    sys.path.insert(0, _CP3_PATH)

# Add counting directory for counting_utils
_COUNTING_DIR = Path(__file__).resolve().parent
if str(_COUNTING_DIR) not in sys.path:
    sys.path.insert(0, str(_COUNTING_DIR))

import counting_utils as cu


def run_counting_evaluation(dataset_name, image_dir, counts_file,
                            output_dir, use_gpu=True, diameter=0.):
    """Run Cellpose3 counting evaluation on one BBBC dataset.

    Parameters
    ----------
    dataset_name : str
    image_dir : str or Path
    counts_file : str or Path
    output_dir : str or Path
    use_gpu : bool
    diameter : float
        0 means auto-estimate

    Returns
    -------
    dict : evaluation results
    """
    from cellpose import models, io

    io.logger_setup()
    print(f"\n{'='*60}")
    print(f"Cellpose3 Counting: {dataset_name}")
    print(f"{'='*60}")
    print(f"  Image dir: {image_dir}")
    print(f"  Counts:    {counts_file}")

    # Load ground truth
    gt_dict = cu.load_counts(counts_file)
    print(f"  GT images: {len(gt_dict)}")

    # Locate and load images (keep grayscale for cellpose3)
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
        # Cellpose3 works on grayscale (H,W)
        if img.ndim == 3:
            img = img[:, :, 0]  # take first channel
        images.append(img)
        valid_fnames.append(fname)

    print(f"  Loaded images: {len(images)} / {len(filenames)}")

    if len(images) == 0:
        print("  [ERROR] No images loaded, aborting.")
        return None

    # Initialize model
    print("  Loading Cellpose3 model (cyto3)...")
    model = models.Cellpose(gpu=use_gpu, model_type="cyto3")

    # Run inference
    d_str = "auto" if diameter == 0 else str(diameter)
    print(f"  Running inference (diameter={d_str})...")
    t0 = time.time()
    masks_pred = model.eval(images, diameter=diameter, channels=[1, 0],
                            batch_size=64, bsize=224, niter=1000,
                            tile_overlap=0.5, flow_threshold=0.4,
                            augment=True)[0]
    elapsed = time.time() - t0
    print(f"  Inference done: {elapsed:.1f}s ({elapsed/len(images):.2f}s/img)")

    # Count instances
    pred_counts = [cu.count_instances(m) for m in masks_pred]
    gt_counts = [gt_dict[f] for f in valid_fnames]

    # Compute metrics
    metrics = cu.compute_counting_metrics(gt_counts, pred_counts)
    metrics["diameter"] = diameter

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
    cu.save_per_image_csv(output_dir, "cellpose3", dataset_name, per_image)
    cu.save_metrics_txt(output_dir, "cellpose3", dataset_name, metrics)

    return {
        "dataset": dataset_name,
        "model": "cellpose3",
        "per_image": per_image,
        "metrics": metrics,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cellpose3 cell counting evaluation")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--counts", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--diameter", type=float, default=0.,
                        help="Cell diameter (0=auto-estimate)")
    parser.add_argument("--cpu", action="store_true")

    args = parser.parse_args()
    run_counting_evaluation(
        dataset_name=args.dataset,
        image_dir=args.image_dir,
        counts_file=args.counts,
        output_dir=args.output_dir,
        use_gpu=not args.cpu,
        diameter=args.diameter,
    )
