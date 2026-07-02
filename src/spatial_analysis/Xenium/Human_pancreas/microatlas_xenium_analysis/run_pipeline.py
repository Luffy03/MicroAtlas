"""
Main entry point for the Cellpose-SAM Xenium segmentation pipeline.

Pipeline steps:
1. Load morphology_focus image
2. Run Cellpose-SAM segmentation
3. Save mask (.npy) and segmentation overlay (.png)

Usage:
    python run_pipeline.py
    python run_pipeline.py --crop --gpu 0
    python run_pipeline.py --model cyto3 --output /path/to/output
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd
from typing import Dict

# Bootstrapping: ensure both cpsam_xenium_analysis and local cellpose package are importable
_PKG_DIR = Path(__file__).resolve().parent  # .../cpsam_xenium_analysis/
_PARENT_DIR = _PKG_DIR.parent  # .../Human_pancreas/
if str(_PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(_PARENT_DIR))

_CELLPOSE_ROOT = _PARENT_DIR.parent.parent.parent  # .../Cellpose/
if str(_CELLPOSE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CELLPOSE_ROOT))

import numpy as np
from skimage.segmentation import mark_boundaries
from skimage.transform import resize
from PIL import Image

from cpsam_xenium_analysis import config
from cpsam_xenium_analysis.data_loader import load_morphology_image
from cpsam_xenium_analysis.segmentation import (
    CPSAMSegmentor, Cellpose3Segmentor, CellSAMSegmentor, MicroSAMSegmentor,
    filter_masks,
)

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Cellpose-SAM Xenium Segmentation Pipeline"
    )
    parser.add_argument(
        "--model",
        choices=["cellpose4", "microatlas", "cellpose3", "cellsam", "microsam"],
        default="cellpose4",
        help="Segmentation model: cellpose4 (default cpsam), microatlas, cellpose3, cellsam, microsam",
    )
    parser.add_argument(
        "--crop", action="store_true",
        help="Use cropped ROI for faster development",
    )
    parser.add_argument(
        "--gpu", type=int, default=0,
        help="GPU device index (default: 0, use -1 for CPU)",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output directory (default: cpsam_xenium_analysis/output)",
    )
    return parser.parse_args()


def get_transcript_hit_labels(
    mask: np.ndarray,
    x_px: np.ndarray,
    y_px: np.ndarray,
) -> set:
    """Return set of mask labels that have at least one transcript assigned.

    Uses efficient batched mask indexing (BATCH_SIZE=500K) to avoid memory
    issues on large full-resolution masks.

    Args:
        mask: Integer label mask (H, W).
        x_px: Transcript x pixel coordinates.
        y_px: Transcript y pixel coordinates.

    Returns:
        Set of integer label values that have at least one transcript.
    """
    h, w = mask.shape[:2]
    valid = (x_px >= 0) & (x_px < w) & (y_px >= 0) & (y_px < h)
    x_v, y_v = x_px[valid], y_px[valid]
    del valid

    n_cells = int(mask.max())
    if n_cells == 0:
        return set()

    hit = np.zeros(n_cells + 1, dtype=bool)
    BATCH_SIZE = 500_000
    n = len(x_v)
    for start in range(0, n, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n)
        vals = mask[y_v[start:end], x_v[start:end]]
        hit[vals] = True
    hit[0] = False

    result = set(np.where(hit)[0].tolist())
    n_empty = n_cells - len(result)
    if n_empty > 0:
        logger.info(f"  Filtered {n_empty}/{n_cells} empty cells from overlay")
    return result


def save_seg_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    save_path: str,
    max_dim: int = 2000,
    outline_color: tuple = (0, 1, 1),  # cyan in RGB
    valid_labels: set = None,
) -> None:
    """Save segmentation overlay as PNG using PIL. No matplotlib, no text.

    Args:
        image: RGB image as uint8 array (H, W, 3).
        mask: Integer label mask (H, W).
        save_path: Output PNG file path.
        max_dim: Downscale if max dimension exceeds this.
        outline_color: RGB tuple in [0,1] for contour color.
        valid_labels: Optional set of label values to draw. Labels not in this
                      set are zeroed out (cells with no transcripts).
    """
    h, w = mask.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        img_small = resize(
            image, (new_h, new_w),
            preserve_range=True, anti_aliasing=True, order=1,
        ).astype(image.dtype)
        mask_small = resize(
            mask, (new_h, new_w),
            preserve_range=True, anti_aliasing=False, order=0,
        ).astype(mask.dtype)
    else:
        img_small = image
        mask_small = mask

    # Filter out cells with no transcripts (after resize for memory safety)
    if valid_labels is not None:
        empty_mask = ~np.isin(mask_small, list(valid_labels))
        mask_small = mask_small.copy()
        mask_small[empty_mask] = 0

    # Normalize image to float [0, 1]
    if img_small.dtype == np.uint8:
        img_float = img_small.astype(np.float32) / 255.0
    else:
        img_float = img_small.astype(np.float32)
        max_val = img_float.max()
        if max_val > 0:
            img_float /= max_val

    # Draw boundaries using skimage
    boundaries = mark_boundaries(
        img_float, mask_small,
        color=outline_color,
        outline_color=None,
        mode='subpixel',
        background_label=0,
    )

    # Save as PNG using PIL - no title, no text, no axes
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    img_uint8 = (np.clip(boundaries, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(img_uint8).save(str(save_path))
    logger.info(f"Saved overlay image to {save_path}")


def run_segmentation(args) -> Dict:
    """Run segmentation on morphology_focus image using the selected model.

    Returns:
        Dict with '{model_name}_morphology' key containing mask and image data.
    """
    results = {}
    crop_roi = config.CROP_ROI if args.crop else None

    # Select segmentor based on model choice
    if args.model in ("cellpose4", "microatlas"):
        if args.model == "microatlas":
            pretrained_model = str(_CELLPOSE_ROOT / "microatlas" / "microatlas")
        else:
            pretrained_model = "cpsam"
        segmentor = CPSAMSegmentor(
            gpu=(args.gpu >= 0),
            pretrained_model=pretrained_model,
            device=f"cuda:{args.gpu}" if args.gpu >= 0 else None,
        )
    elif args.model == "cellpose3":
        segmentor = Cellpose3Segmentor(
            gpu=(args.gpu >= 0),
            device=f"cuda:{args.gpu}" if args.gpu >= 0 else None,
        )
    elif args.model == "cellsam":
        segmentor = CellSAMSegmentor(
            gpu=(args.gpu >= 0),
            device=f"cuda:{args.gpu}" if args.gpu >= 0 else None,
        )
    elif args.model == "microsam":
        segmentor = MicroSAMSegmentor(
            gpu=(args.gpu >= 0),
            device=f"cuda:{args.gpu}" if args.gpu >= 0 else None,
        )
    else:
        raise ValueError(f"Unknown model: {args.model}")

    # Morphology segmentation
    logger.info("=" * 60)
    logger.info("Loading morphology image...")
    morph_img = load_morphology_image(use_focus=True, crop_roi=crop_roi)
    logger.info(f"Morphology image loaded: {morph_img.shape}")

    logger.info(f"Running {args.model} on morphology image...")
    t0 = time.time()
    mask_morph, flows_morph, styles_morph = segmentor.segment_morphology(morph_img)
    t_elapsed = time.time() - t0
    logger.info(f"Morphology segmentation completed in {t_elapsed:.1f}s")

    # Post-process
    mask_morph = filter_masks(mask_morph, min_size=15)

    results[f"{args.model}_morphology"] = {
        "mask": mask_morph,
        "image": morph_img,
        "flows": flows_morph,
        "styles": styles_morph,
    }

    # Save mask
    out_path = config.OUTPUT_DIR / f"masks_{args.model}_morphology.npy"
    np.save(str(out_path), mask_morph)
    logger.info(f"Saved morphology mask to {out_path}")

    return results


def main():
    """Main pipeline execution."""
    args = parse_args()
    setup_logging(args.verbose)

    logger.info("=" * 60)
    logger.info("Cellpose-SAM Xenium Segmentation Pipeline")
    logger.info("=" * 60)

    # Update config
    if args.output:
        config.OUTPUT_DIR = Path(args.output)
        config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    config.USE_CROP = args.crop
    config.GPU_DEVICE = args.gpu

    # Step 1: Segmentation
    seg_results = run_segmentation(args)

    # Step 2: Save segmentation overlay
    logger.info("=" * 60)
    logger.info("Saving segmentation overlay...")

    # Pre-load transcript coordinates once for all methods
    transcript_coords = None
    try:
        logger.info("Loading transcripts for empty cell filtering...")
        tx_df = pd.read_parquet(
            config.TRANSCRIPTS_PATH, columns=["x_location", "y_location"]
        )
        x_all = np.round(
            tx_df["x_location"].values / config.PIXEL_SIZE_UM
        ).astype(np.int32)
        y_all = np.round(
            tx_df["y_location"].values / config.PIXEL_SIZE_UM
        ).astype(np.int32)
        del tx_df
        transcript_coords = (x_all, y_all)
        logger.info(f"  Loaded {len(x_all)} transcripts for filtering")
    except Exception as e:
        logger.warning(
            f"  Failed to load transcripts, skipping empty cell filtering: {e}"
        )

    for method_name, seg_data in seg_results.items():
        if seg_data["image"] is not None:
            valid_labels = None
            if transcript_coords is not None:
                valid_labels = get_transcript_hit_labels(
                    seg_data["mask"],
                    transcript_coords[0],
                    transcript_coords[1],
                )
            save_seg_overlay(
                seg_data["image"],
                seg_data["mask"],
                save_path=config.OUTPUT_DIR / f"seg_overlay_{method_name}.png",
                valid_labels=valid_labels,
            )
        else:
            logger.warning(f"Skipping overlay for {method_name}: image is None")

    logger.info("=" * 60)
    logger.info(f"Pipeline complete! Results saved to {config.OUTPUT_DIR}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
