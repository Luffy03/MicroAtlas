"""
Cellpose4 cell counting evaluation on BBBC datasets.

Usage:
  python eval_cellpose4_counting.py --dataset BBBC041 --image_dir .../BBBC041/images \
                                    --counts .../BBBC041/BBBC041_v1_counts.txt --output_dir ./results
"""
import os
import sys
import time
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm, trange

# Add the counting directory to sys.path for counting_utils
_COUNTING_DIR = Path(__file__).resolve().parent
if str(_COUNTING_DIR) not in sys.path:
    sys.path.insert(0, str(_COUNTING_DIR))

import counting_utils as cu


def run_counting_evaluation(dataset_name, image_dir, counts_file,
                            output_dir, use_gpu=True, diameter=30.):
    """Run Cellpose4 counting evaluation on one BBBC dataset.

    Parameters
    ----------
    dataset_name : str
        e.g. "BBBC039"
    image_dir : str or Path
        Directory containing the images
    counts_file : str or Path
        Path to *_v1_counts.txt
    output_dir : str or Path
        Where to save results
    use_gpu : bool
        Whether to use GPU
    diameter : float
        Cell diameter for Cellpose

    Returns
    -------
    dict : evaluation results
    """
    from cellpose import models, io, utils

    io.logger_setup()
    print(f"\n{'='*60}")
    print(f"Cellpose4 Counting: {dataset_name}")
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
        # Convert grayscale to 3-channel for Cellpose4
        if img.ndim == 2:
            img_3c = np.tile(img[np.newaxis, :, :], (3, 1, 1))
            img_3c[1:] = 0
        else:
            # Already 3-channel: transpose from (H,W,3) to (3,H,W)
            img_3c = img.transpose(2, 0, 1)
        images.append(img_3c)
        valid_fnames.append(fname)

    print(f"  Loaded images: {len(images)} / {len(filenames)}")

    if len(images) == 0:
        print("  [ERROR] No images loaded, aborting.")
        return None

    # Initialize model
    print("  Loading Cellpose4 model (cpsam)...")
    model = models.CellposeModel(gpu=use_gpu)
    net = model.net
    net.eval()

    # Run inference
    print(f"  Running inference (diameter={diameter})...")
    t0 = time.time()
    masks_pred = model.eval(images, diameter=diameter, channels=None,
                            niter=1000, batch_size=64, bsize=256)[0]
    elapsed = time.time() - t0
    print(f"  Inference done: {elapsed:.1f}s ({elapsed/len(images):.2f}s/img)")

    # Count instances per image
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

    # Save per-image results
    per_image = [
        {"filename": f, "gt_count": g, "pred_count": p}
        for f, g, p in zip(valid_fnames, gt_counts, pred_counts)
    ]
    cu.save_per_image_csv(output_dir, "cellpose4", dataset_name, per_image)
    cu.save_metrics_txt(output_dir, "cellpose4", dataset_name, metrics)

    return {
        "dataset": dataset_name,
        "model": "cellpose4",
        "per_image": per_image,
        "metrics": metrics,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cellpose4 cell counting evaluation")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g. BBBC039)")
    parser.add_argument("--image_dir", type=str, required=True, help="Image directory")
    parser.add_argument("--counts", type=str, required=True, help="Path to counts.txt")
    parser.add_argument("--output_dir", type=str, default="./results", help="Output directory")
    parser.add_argument("--diameter", type=float, default=30., help="Cell diameter")
    parser.add_argument("--cpu", action="store_true", help="Use CPU")

    args = parser.parse_args()
    run_counting_evaluation(
        dataset_name=args.dataset,
        image_dir=args.image_dir,
        counts_file=args.counts,
        output_dir=args.output_dir,
        use_gpu=not args.cpu,
        diameter=args.diameter,
    )
