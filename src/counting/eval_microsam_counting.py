"""
MicroSAM cell counting evaluation on BBBC datasets.

Usage:
  cd ~/Cellpose && python counting/eval_microsam_counting.py --dataset BBBC041 ...
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


def _convert_image_microsam(img):
    """Convert single image to MicroSAM (H,W,3) float32 format."""
    if img.ndim == 3:
        if np.array(img.shape).argmin() == 2:
            img = img.transpose(2, 0, 1)
        # Average to single channel if RGB, then tile
        if img.shape[0] == 3:
            img = img.mean(axis=0)
        elif img.shape[0] > 3:
            img = img[0]
    # img is now (H,W)
    img = img.astype(np.float32)
    img = (img - img.min()) / max(img.max() - img.min(), 1e-10)
    img = np.stack([img, img, img], axis=-1)  # (H,W,3)
    return img


def run_counting_evaluation(dataset_name, image_dir, counts_file,
                            output_dir, use_gpu=True, model_type="vit_l_lm"):
    """Run MicroSAM counting evaluation on one BBBC dataset."""
    from micro_sam import util
    from micro_sam.instance_segmentation import (
        InstanceSegmentationWithDecoder,
        get_predictor_and_decoder,
    )

    print(f"\n{'='*60}")
    print(f"MicroSAM Counting: {dataset_name}")
    print(f"{'='*60}")
    print(f"  Image dir: {image_dir}")
    print(f"  Counts:    {counts_file}")
    print(f"  Model:     {model_type}")

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
        img_microsam = _convert_image_microsam(img)
        images.append(img_microsam)
        valid_fnames.append(fname)

    print(f"  Loaded images: {len(images)} / {len(filenames)}")

    if len(images) == 0:
        print("  [ERROR] No images loaded, aborting.")
        return None

    # Initialize model (using new micro-sam API)
    device = "cuda" if use_gpu else "cpu"
    print(f"  Loading MicroSAM model ({model_type}) on {device}...")
    predictor, decoder = get_predictor_and_decoder(
        model_type=model_type,
        checkpoint_path=None,
    )

    # Run inference (following eval/eval_microsam.py pattern)
    print("  Running inference (AIS)...")
    t0 = time.time()
    masks_pred = []
    for i, img in enumerate(tqdm(images, desc="  Inference")):
        image_embeddings = util.precompute_image_embeddings(
            predictor=predictor,
            input_=img,
            ndim=2,
            verbose=False,
        )
        ais = InstanceSegmentationWithDecoder(predictor, decoder)
        ais.initialize(
            image=img,
            image_embeddings=image_embeddings,
        )
        prediction = ais.generate(output_mode='instance_segmentation')
        if prediction is not None and len(prediction) > 0:
            masks_pred.append(prediction)
        else:
            masks_pred.append(np.zeros(img.shape[:2], "uint16"))
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
    cu.save_per_image_csv(output_dir, "microsam", dataset_name, per_image)
    cu.save_metrics_txt(output_dir, "microsam", dataset_name, metrics)

    return {
        "dataset": dataset_name,
        "model": "microsam",
        "per_image": per_image,
        "metrics": metrics,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MicroSAM cell counting evaluation")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--counts", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--model_type", type=str, default="vit_l_lm",
                        help="MicroSAM model type")
    parser.add_argument("--cpu", action="store_true")

    args = parser.parse_args()
    run_counting_evaluation(
        dataset_name=args.dataset,
        image_dir=args.image_dir,
        counts_file=args.counts,
        output_dir=args.output_dir,
        use_gpu=not args.cpu,
        model_type=args.model_type,
    )
