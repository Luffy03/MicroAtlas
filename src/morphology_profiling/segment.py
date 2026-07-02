"""
segment.py
==========
Cell instance segmentation with 5 models on BBBC021 DAPI channel.

Supported models:
  - cellpose4:   Cellpose4 (cpsam transformer)
  - cellpose3:   Cellpose3 (cyto3, loaded from cellpose-cp3)
  - microsam:    MicroSAM (vit_l_lm)
  - microatlas:  MicroAtlas (CellposeModel, pretrained microatlas weights)
  - cellsam:     CellSAM (pipeline)

Input:  DAPI channel TIFF images from BBBC021
Output: uint32 instance masks per model

Usage:
  python biomarker_discovery/segment.py --models cellpose4 --plate Week1_22123
  python biomarker_discovery/segment.py --models cellpose4 cellpose3 microsam --all
  python biomarker_discovery/segment.py --models all --all
  python biomarker_discovery/segment.py --status
"""

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
from tifffile import imread, imwrite
from skimage.morphology import remove_small_objects
from skimage.segmentation import relabel_sequential

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config


# =========================================================================
# Field key extraction
# =========================================================================

def _get_field_key(dapi_filename):
    """Extract field key from DAPI filename.

    BBBC021 filenames: G10_s1_w1BEDC2073-...tif
    Field key: G10_s1 (well + site, before _wN)
    """
    m = re.match(r'^(.+?)_w\d+', dapi_filename)
    if m:
        return m.group(1)
    return Path(dapi_filename).stem


# =========================================================================
# Model loading
# =========================================================================

def _load_cellpose4(use_gpu=True):
    """Load Cellpose4 model (cpsam)."""
    from cellpose import models, io
    io.logger_setup()
    print(f"  Loading Cellpose4 model (gpu={use_gpu})...")
    model = models.CellposeModel(gpu=use_gpu)
    model.net.eval()
    return model


def _load_cellpose3(use_gpu=True):
    """Load Cellpose3 model from cellpose-cp3."""
    cp3_path = str(config.CELLPOSE_ROOT / 'cellpose-cp3')
    sys.path.insert(0, cp3_path)
    from cellpose import models, io
    io.logger_setup()
    print(f"  Loading Cellpose3 model (gpu={use_gpu})...")
    model = models.Cellpose(gpu=use_gpu, model_type="cyto3")
    return model


def _load_microsam(use_gpu=True):
    """Load MicroSAM model (vit_l_lm)."""
    from micro_sam.instance_segmentation import (
        InstanceSegmentationWithDecoder,
        get_predictor_and_decoder,
    )
    model_type = config.MODEL_CONFIGS["microsam"]["model_type"]
    print(f"  Loading MicroSAM model ({model_type})...")
    predictor, decoder = get_predictor_and_decoder(
        model_type=model_type,
        checkpoint_path=None,
    )
    return {"predictor": predictor, "decoder": decoder}


def _load_microatlas(use_gpu=True):
    """Load MicroAtlas model."""
    from cellpose import models, io
    io.logger_setup()
    pretrained = str(config.CELLPOSE_ROOT /
                     config.MODEL_CONFIGS["microatlas"]["pretrained_model"])
    print(f"  Loading MicroAtlas model ({pretrained})...")
    model = models.CellposeModel(gpu=use_gpu, pretrained_model=pretrained)
    model.net.eval()
    return model


def _load_cellsam(use_gpu=True):
    """CellSAM does not need explicit model loading (managed internally)."""
    print("  CellSAM: model will be loaded per-image via cellsam_pipeline.")
    return None


def load_model(model_name, use_gpu=True):
    """Unified model loading entry point."""
    loaders = {
        "cellpose4": _load_cellpose4,
        "cellpose3": _load_cellpose3,
        "microsam": _load_microsam,
        "microatlas": _load_microatlas,
        "cellsam": _load_cellsam,
    }
    if model_name not in loaders:
        raise ValueError(f"Unknown model: {model_name}. "
                         f"Supported: {list(loaders.keys())}")
    return loaders[model_name](use_gpu=use_gpu)


# =========================================================================
# Segmentation backends
# =========================================================================

def _postprocess_mask(mask):
    """Apply common post-processing: remove small objects + relabel."""
    mask = mask.astype(np.uint32)
    if mask.max() > 0:
        m_clean = remove_small_objects(mask, min_size=config.MIN_CELL_AREA)
        m_clean, _, _ = relabel_sequential(m_clean)
        return m_clean.astype(np.uint32)
    return mask


def _segment_cellpose_family(images, model, model_name):
    """Segment with Cellpose4/Cellpose3/MicroAtlas (shared CellposeModel API)."""
    params = config.MODEL_CONFIGS[model_name].copy()
    bsize = params.pop("bsize", 256)
    channels = params.pop("channels", None)
    niter = params.pop("niter", 1000)
    batch_size = params.pop("batch_size", 64)
    params.pop("pretrained_model", None)  # model loading param, not eval param
    params.pop("model_type", None)        # model loading param, not eval param

    kwargs = {
        "diameter": config.SEGMENT_DIAMETER,
        "channels": channels,
        "niter": niter,
        "batch_size": batch_size,
        "bsize": bsize,
    }
    kwargs.update(params)  # tile_overlap, flow_threshold, augment for cellpose3

    if model_name == "cellpose3":
        # Cellpose3 eval may return 3 or 4 values
        out = model.eval(images, **kwargs)
        masks_pred = out[0]
    else:
        masks_pred = model.eval(images, **kwargs)[0]

    return [_postprocess_mask(m) for m in masks_pred]


def _segment_microsam(images, model_dict):
    """Segment with MicroSAM (per-image inference)."""
    from micro_sam import util
    from micro_sam.instance_segmentation import (
        InstanceSegmentationWithDecoder,
    )

    predictor = model_dict["predictor"]
    decoder = model_dict["decoder"]
    masks_out = []

    for i, img in enumerate(images):
        # Convert to (H, W, 3) float32
        if img.ndim == 2:
            img_rgb = np.stack((img,) * 3, axis=-1).astype(np.float32)
        else:
            if np.array(img.shape).argmin() == 2:
                img = img.transpose(2, 0, 1)
            if np.ptp(img[1]) != 0:
                img = img.astype("float32").mean(axis=0)
            else:
                img = img[0]
            img_rgb = np.stack((img,) * 3, axis=-1).astype(np.float32)

        image_embeddings = util.precompute_image_embeddings(
            predictor=predictor,
            input_=img_rgb,
            ndim=2,
            verbose=False,
        )

        ais = InstanceSegmentationWithDecoder(predictor, decoder)
        ais.initialize(image=img_rgb, image_embeddings=image_embeddings)

        prediction = ais.generate(output_mode='instance_segmentation')
        if prediction is not None and len(prediction) > 0:
            masks_out.append(_postprocess_mask(prediction))
        else:
            masks_out.append(np.zeros(img_rgb.shape[:2], dtype=np.uint32))

        if (i + 1) % 50 == 0:
            print(f"    microsam: {i+1}/{len(images)} images")

    return masks_out


def _segment_cellsam(images):
    """Segment with CellSAM (per-image inference with pad/resize)."""
    import cv2
    from cellSAM import cellsam_pipeline

    masks_out = []

    for i, img in enumerate(images):
        try:
            # Record original size
            if img.ndim == 2:
                Ly, Lx = img.shape
            else:
                Ly, Lx = img.shape[:2]

            # Convert to (3, H, W) float32 for CellSAM
            if img.ndim == 2:
                img_c = np.stack(
                    (np.zeros_like(img), np.zeros_like(img), img), axis=0
                )
            elif img.ndim == 3:
                if np.array(img.shape).argmin() == 2:  # (H,W,C) -> (C,H,W)
                    img_c = img.transpose(2, 0, 1)
                else:
                    img_c = img.copy()
                if img_c.shape[0] < 3:
                    img_c = np.concatenate(
                        (np.zeros((3 - img_c.shape[0], *img_c.shape[1:]),
                                  dtype=img_c.dtype), img_c),
                        axis=0,
                    )
            else:
                img_c = img

            img_c = img_c.astype(np.float32)

            # Per-channel normalization
            for k in range(3):
                if np.ptp(img_c[k]) > 1e-3:
                    img_c[k] = ((img_c[k] - img_c[k].min()) /
                                (img_c[k].max() - img_c[k].min()))

            # Resize if smaller than 512
            Lyr, Lxr = Ly, Lx
            if Ly < 512 and Lx < 512:
                Lxr = int(np.round(512 * (Lx / Ly))) if Ly > Lx else 512
                Lyr = int(np.round(512 * (Ly / Lx))) if Lx >= Ly else 512

            if Lyr != Ly or Lxr != Lx:
                img_c = cv2.resize(
                    img_c.transpose(1, 2, 0), (Lxr, Lyr),
                    interpolation=cv2.INTER_LINEAR
                ).transpose(2, 0, 1)

            # Pad to 512x512
            padyx = [[0, 0], [0, 0]]
            if Lyr < 512:
                padyx[0] = [int(np.floor((512 - Lyr) / 2)),
                            int(np.ceil((512 - Lyr) / 2))]
            if Lxr < 512:
                padyx[1] = [int(np.floor((512 - Lxr) / 2)),
                            int(np.ceil((512 - Lxr) / 2))]

            if any(p[0] > 0 or p[1] > 0 for p in padyx):
                img_c = np.pad(img_c, ((0, 0), padyx[0], padyx[1]),
                               mode='constant')

            params = config.MODEL_CONFIGS["cellsam"]
            mask = cellsam_pipeline(
                img_c,
                use_wsi=params["use_wsi"],
                low_contrast_enhancement=params["low_contrast_enhancement"],
                gauge_cell_size=params["gauge_cell_size"],
            )

            # Remove padding
            if any(p[0] > 0 or p[1] > 0 for p in padyx):
                mask = mask[padyx[0][0]: mask.shape[0] - padyx[0][1],
                            padyx[1][0]: mask.shape[1] - padyx[1][1]]

            # Resize back to original size
            if Ly != mask.shape[0] or Lx != mask.shape[1]:
                mask = cv2.resize(mask, (Lx, Ly),
                                  interpolation=cv2.INTER_NEAREST)

            masks_out.append(_postprocess_mask(mask))

        except Exception as e:
            print(f"    [ERROR] cellsam image {i}: {e}")
            if img.ndim == 2:
                Ly, Lx = img.shape
            else:
                Ly, Lx = img.shape[:2]
            masks_out.append(np.zeros((Ly, Lx), dtype=np.uint32))

        if (i + 1) % 50 == 0:
            print(f"    cellsam: {i+1}/{len(images)} images")

    return masks_out


def segment_batch(model_name, model, images):
    """Unified segmentation entry point.

    Args:
        model_name: one of ["cellpose4", "cellpose3", "microsam",
                            "microatlas", "cellsam"]
        model: loaded model object (or None for cellsam)
        images: list of 2D numpy arrays (uint16), one per field

    Returns:
        list of uint32 masks
    """
    if len(images) == 0:
        return []

    if model_name in ("cellpose4", "cellpose3", "microatlas"):
        return _segment_cellpose_family(images, model, model_name)
    elif model_name == "microsam":
        return _segment_microsam(images, model)
    elif model_name == "cellsam":
        return _segment_cellsam(images)
    else:
        raise ValueError(f"Unknown model: {model_name}")


# =========================================================================
# Plate-level segmentation
# =========================================================================

def get_plate_fields(plate_name):
    """Get list of DAPI image paths for a plate.

    Returns:
        list of (dapi_path, field_key) tuples
    """
    plate_dir = config.IMAGES_DIR / plate_name / config.NUCLEUS_CHANNEL
    if not plate_dir.exists():
        print(f"  [WARN] Plate directory not found: {plate_dir}")
        return []

    dapi_files = sorted(plate_dir.glob("*.tif"))
    fields = []
    for dapi_path in dapi_files:
        field_key = _get_field_key(dapi_path.name)
        fields.append((dapi_path, field_key))

    return fields


def segment_plate(plate_name, model_name, model=None,
                  diameter=None, force=False):
    """Segment all fields in a plate using the specified model.

    Args:
        plate_name: plate identifier
        model_name: model name
        model: pre-loaded model (or None to load)
        diameter: cell diameter
        force: re-segment even if mask exists
    """
    fields = get_plate_fields(plate_name)
    if not fields:
        return

    mask_dir = config.masks_dir(model_name) / plate_name
    mask_dir.mkdir(parents=True, exist_ok=True)

    # Build todo list
    todo = []
    done = 0
    for dapi_path, field_key in fields:
        mask_path = mask_dir / f"{field_key}_mask.tif"
        if force or not mask_path.exists():
            todo.append((dapi_path, field_key, mask_path))
        else:
            done += 1

    if done > 0 and not force:
        print(f"  Skipping {done} already segmented")
    if not todo:
        print(f"  All {len(fields)} fields already segmented.")
        return

    # Load model if needed
    if model is None:
        model = load_model(model_name)

    # Process in batches (for cellpose-family models)
    # For per-image models (microsam, cellsam), batch_size=1
    if model_name in ("microsam", "cellsam"):
        BATCH = 1
    else:
        BATCH = config.MODEL_CONFIGS[model_name].get("batch_size", 64)

    n_processed = 0
    t_total = time.time()

    for i in range(0, len(todo), max(BATCH, 1)):
        batch_items = todo[i:i + BATCH]

        # Load DAPI images
        images = []
        for dapi_path, _, _ in batch_items:
            img = imread(str(dapi_path))
            if img.ndim > 2:
                img = img[..., 0]
            images.append(img)

        # Segment
        masks = segment_batch(model_name, model, images)

        # Save masks
        for mask, (_, _, mask_path) in zip(masks, batch_items):
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            imwrite(str(mask_path), mask.astype(np.uint32))

        n_processed += len(masks)

        if (i // max(BATCH, 1)) % 2 == 0 or i + BATCH >= len(todo):
            elapsed = time.time() - t_total
            rate = n_processed / elapsed if elapsed > 0 else 0
            print(f"    [{model_name}] {n_processed}/{len(todo)} fields "
                  f"({rate:.1f} fields/s)")

    elapsed = time.time() - t_total
    print(f"  Plate {plate_name} [{model_name}]: {n_processed} fields "
          f"in {elapsed:.1f}s ({n_processed/elapsed:.1f} fields/s)")


# =========================================================================
# Status
# =========================================================================

def list_status():
    """Show segmentation status per model per plate."""
    print("\n=== Segmentation Status ===")
    if not config.IMAGES_DIR.exists():
        print("  No images downloaded yet.")
        return

    for model_name in config.MODELS:
        model_mask_dir = config.masks_dir(model_name)
        total_images = 0
        total_masks = 0

        has_any = False
        for plate_dir in sorted(config.IMAGES_DIR.iterdir()):
            if not plate_dir.is_dir():
                continue
            ch_dir = plate_dir / config.NUCLEUS_CHANNEL
            if not ch_dir.exists():
                continue
            n_images = len(list(ch_dir.glob("*.tif")))
            total_images += n_images

            plate_mask_dir = model_mask_dir / plate_dir.name
            n_masks = 0
            if plate_mask_dir.exists():
                n_masks = len(list(plate_mask_dir.glob("*.tif")))

            total_masks += n_masks
            if n_masks > 0:
                has_any = True

        if has_any or total_images > 0:
            status = "OK" if total_masks >= total_images else "INCOMPLETE"
            print(f"  {model_name}: {total_masks}/{total_images} "
                  f"masks [{status}]")


# =========================================================================
# CLI
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Segment BBBC021 images with multiple models")
    parser.add_argument("--models", type=str, nargs="+", default=None,
                        help="Models to use (e.g. cellpose4 cellpose3). "
                             "Use 'all' for all 5 models.")
    parser.add_argument("--plate", type=str, default=None,
                        help="Segment a single plate")
    parser.add_argument("--plates", type=str, nargs="+", default=None,
                        help="Segment multiple plates")
    parser.add_argument("--all", action="store_true",
                        help="Segment all downloaded plates")
    parser.add_argument("--diameter", type=float, default=None,
                        help="Expected cell diameter (default: auto)")
    parser.add_argument("--no_gpu", action="store_true",
                        help="Disable GPU")
    parser.add_argument("--force", action="store_true",
                        help="Re-segment even if mask exists")
    parser.add_argument("--status", action="store_true",
                        help="Show segmentation status")
    args = parser.parse_args()

    config.ensure_dirs()

    if args.status:
        list_status()
        return

    # Determine models
    if args.models is None:
        models_to_run = ["cellpose4"]  # default
    elif "all" in args.models:
        models_to_run = config.MODELS
    else:
        models_to_run = args.models

    # Validate
    for m in models_to_run:
        if m not in config.MODELS:
            print(f"Unknown model: {m}. Supported: {config.MODELS}")
            return

    # Determine plates
    if args.plate:
        plates = [args.plate]
    elif args.plates:
        plates = args.plates
    elif args.all:
        if not config.IMAGES_DIR.exists():
            print("No images found. Run download.py first.")
            return
        plates = []
        for plate_dir in sorted(config.IMAGES_DIR.iterdir()):
            if plate_dir.is_dir() and \
               (plate_dir / config.NUCLEUS_CHANNEL).exists():
                plates.append(plate_dir.name)
    else:
        print("No action. Use --plate, --plates, --all, or --status")
        return

    use_gpu = not args.no_gpu

    for model_name in models_to_run:
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")

        config.ensure_model_dirs(model_name)
        model = load_model(model_name, use_gpu=use_gpu)

        for plate in plates:
            segment_plate(plate, model_name, model,
                          diameter=args.diameter, force=args.force)

        # Restore sys.path for cellpose3 (avoid conflict with cellpose4)
        if model_name == "cellpose3":
            cp3_path = str(config.CELLPOSE_ROOT / 'cellpose-cp3')
            if cp3_path in sys.path:
                sys.path.remove(cp3_path)


if __name__ == "__main__":
    main()
