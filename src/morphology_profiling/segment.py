"""
segment.py
==========
Whole-cell instance segmentation with 5 models on U2OS-Cell-Painting images.

Supported models:
  - cellpose4:   Cellpose4 (cpsam transformer)
  - cellpose3:   Cellpose3 (cyto3, loaded from cellpose-cp3)
  - microsam:    MicroSAM (vit_l_lm)
  - microatlas:  MicroAtlas (CellposeModel, pretrained microatlas weights)
  - cellsam:     CellSAM (pipeline)

Segmentation input channels (all models produce whole-cell masks):
  All five models receive the same AGP (cyto) + DNA (nucleus) signal:
        [AGP (cyto), DNA (nucleus), 0]
  - cellpose4 / microatlas: 3-channel (3, H, W) uint16 stack (channels=None, cpsam)
  - cellpose3:               3-channel (3, H, W) uint16 stack, uses AGP via channels=[1, 0]
  - microsam:                per-image reduces to single channel + tiles to (H, W, 3)
  - cellsam:                 percentile-clamps AGP + DNA to [0, 255] and stacks
        into (H, W, 3) uint8 = [AGP_uint8, DNA_uint8, 0] entirely in
        memory (no on-disk cache). The bare `cellsam_pipeline(img)` call
        mirrors the canonical counting-pipeline invocation in
        counting/eval_cellsam_counting.py:86 -- the only call style
        verified to run end-to-end on cellSAM 0.0.dev1. The percentile
        endpoints (--lo-pct / --hi-pct) default to 1.0 / 99.9.

Input:  AGP (config.SEG_CYTO_CHANNEL) + DNA (config.SEG_NUCLEUS_CHANNEL) TIFFs
Output: uint16 whole-cell instance masks per model (deflate-compressed TIFF)

Usage:
  python src/morphology_profiling/segment.py --models cellpose4 --plate P015080
  python src/morphology_profiling/segment.py --models cellpose4 cellpose3 microsam --all
  python src/morphology_profiling/segment.py --models all --all
  python src/morphology_profiling/segment.py --status
"""

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
from tifffile import TiffFileError, imread, imwrite
from skimage.morphology import remove_small_objects
from skimage.segmentation import relabel_sequential

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config


# =========================================================================
# Field key extraction
# =========================================================================

def _get_field_key(filename):
    """Extract field key from a normalized channel filename.

    Filenames: A01_s1_w1.tif
    Field key: A01_s1 (well + site, before _wN)
    """
    m = re.match(r'^(.+?)_w\d+', filename)
    if m:
        return m.group(1)
    return Path(filename).stem


def _channel_path(plate_name, field_key, channel):
    """Absolute path to a channel TIFF for a given field.

    Channel index follows config.CHANNEL_NAMES order (1-based, e.g. DNA=w1).
    """
    idx = config.CHANNEL_NAMES.index(channel) + 1
    return config.IMAGES_DIR / plate_name / channel / f"{field_key}_w{idx}.tif"


def _field_ready(plate_name, field_key, model_name):
    """Check whether all channel TIFFs required by ``model_name`` are on disk.

    Required channels (all 5 models): AGP (cyto) + DNA (nucleus).

    Returns:
        (is_ready, missing_channels): ``is_ready`` is True iff every required
        channel TIFF exists with non-zero size; otherwise ``missing_channels``
        lists the names of the channels that are not yet fully downloaded.

    Rationale: this lets ``download.py`` and ``segment.py`` run in parallel.
    When the downloader is mid-way through a plate, AGP may exist for a
    field but DNA may still be in flight. Instead of crashing with a
    FileNotFoundError deep inside the segmentation batch loop, segment.py
    just defers those fields to the next run via the per-field resume check.
    """
    required = [config.SEG_CYTO_CHANNEL, config.SEG_NUCLEUS_CHANNEL]
    missing = []
    for ch in required:
        p = _channel_path(plate_name, field_key, ch)
        if not p.exists() or p.stat().st_size == 0:
            missing.append(ch)
    return (len(missing) == 0, missing)


def _load_2d(path):
    """Read a TIFF and collapse to a single 2D plane."""
    img = imread(str(path))
    if img.ndim > 2:
        img = img[..., 0]
    return img


# Exceptions tifffile raises on a truncated / half-written TIFF.
_BAD_TIFF_ERRORS = (TiffFileError, ValueError, OSError, EOFError, IndexError)


def _tiff_readable(path):
    """True iff ``path`` can be decoded end-to-end as a 2D plane."""
    try:
        _load_2d(path)
        return True
    except _BAD_TIFF_ERRORS:
        return False


def _quarantine_broken_channels(plate_name, field_key):
    """Rename a field's undecodable channel TIFFs to ``*.corrupt``.

    A truncated TIFF (extraction killed mid-write, or read while download.py
    was still writing it) exists with non-zero size, so both ``_field_ready``
    here and ``_count_extracted`` in download.py treat it as complete and the
    field stays permanently broken. Moving it aside makes the next
    ``download.py`` run re-extract it, and keeps the bad bytes around for
    inspection instead of deleting them.

    Returns the list of channel names that were quarantined.
    """
    broken = []
    for ch in (config.SEG_CYTO_CHANNEL, config.SEG_NUCLEUS_CHANNEL):
        p = _channel_path(plate_name, field_key, ch)
        if p.exists() and not _tiff_readable(p):
            p.replace(p.with_name(p.name + ".corrupt"))
            broken.append(ch)
    return broken


def _percentile_to_uint8(img, lo_pct=1.0, hi_pct=99.9):
    """Percentile-clip then linearly rescale a 2D image to uint8 [0, 255].

    Used to build cellsam's uint8 RGB input entirely in-memory (no PNG
    round-trip): the 1st/99.9th percentiles define the display range so a few
    saturated pixels do not crush the dynamic range.
    """
    if img.size == 0:
        return np.zeros(img.shape, dtype=np.uint8)
    arr = img.astype(np.float32, copy=False)
    if float(arr.max() - arr.min()) < 1e-3:
        return np.full(arr.shape, 128, dtype=np.uint8)
    low, high = np.percentile(arr, [lo_pct, hi_pct])
    if not (high > low):
        return np.full(arr.shape, 128, dtype=np.uint8)
    out = np.clip((arr - low) / (high - low) * 255.0, 0, 255).astype(np.uint8)
    return out


def _agp_dna_to_rgb_uint8(agp_2d, dna_2d, lo_pct=1.0, hi_pct=99.9):
    """Build a (H, W, 3) uint8 [AGP, DNA, 0] image from 2D AGP + DNA maps.

    Cellsam contract: (H, W, 3) channel-last, uint8 -- the dtype and channel
    layout cellSAM's internal `normalize_image` (cv2 CLAHE) expects, so the
    bare `cellsam_pipeline(img)` call runs end-to-end on cellSAM 0.0.dev1.
    The channel layout [AGP (cyto), DNA (nucleus), 0] matches the other four
    models' (3, H, W) stack, so cellsam receives the same cytoplasm + nucleus
    signal and produces whole-cell (not nuclear-only) masks. Each channel is
    percentile-clipped to [0, 255] independently.
    """
    if agp_2d.ndim == 3:
        agp_2d = agp_2d[..., 0]
    if dna_2d.ndim == 3:
        dna_2d = dna_2d[..., 0]
    agp_u8 = _percentile_to_uint8(agp_2d, lo_pct=lo_pct, hi_pct=hi_pct)
    dna_u8 = _percentile_to_uint8(dna_2d, lo_pct=lo_pct, hi_pct=hi_pct)
    zeros = np.zeros_like(agp_u8)
    return np.stack([agp_u8, dna_u8, zeros], axis=-1)


def _load_field_image(plate_name, field_key, model_name, lo_pct=1.0, hi_pct=99.9):
    """Load the per-model segmentation input image for a field.

    Per-model input shape:
      - cellpose4 / cellpose3 / microatlas / microsam: (3, H, W) uint16 =
            [AGP (cyto), DNA (nucleus), 0]
      - cellsam: (H, W, 3) uint8 = [AGP, DNA, 0], built in-memory from the
            raw uint16 AGP + DNA TIFFs by percentile-clipping each to
            [0, 255] and right-padding zeros in ch2. No on-disk cache is
            required -- the conversion runs in the segmentation loop.

    Notes:
      * cellpose3 selects channels via MODEL_CONFIGS channels=[1, 0] (AGP only).
      * microsam converts (3, H, W) to (H, W, 3) for the predictor.
      * cellsam is fed the same AGP (cyto) + DNA (nucleus) signal as the other
        four models, so it produces whole-cell masks (not nuclear-only). The
        uint8 (H, W, 3) layout is required only for cellSAM 0.0.dev1's internal
        normalize_image/CLAHE; the channel order [AGP, DNA, 0] matches the
        (3, H, W) stack the other models receive.
    """
    if model_name == "cellsam":
        dna = _load_2d(_channel_path(plate_name, field_key,
                                     config.SEG_NUCLEUS_CHANNEL))
        agp = _load_2d(_channel_path(plate_name, field_key,
                                     config.SEG_CYTO_CHANNEL))
        return _agp_dna_to_rgb_uint8(agp, dna, lo_pct=lo_pct, hi_pct=hi_pct)
    dna = _load_2d(_channel_path(plate_name, field_key, config.SEG_NUCLEUS_CHANNEL))
    agp = _load_2d(_channel_path(plate_name, field_key, config.SEG_CYTO_CHANNEL))
    zero = np.zeros_like(agp)
    return np.stack([agp, dna, zero], axis=0)  # (3, H, W) channel-first


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
    """Segment with MicroSAM (per-image inference).

    Each input is the (3, H, W) = [AGP, DNA, 0] stack from _load_field_image.
    Mirrors `eval/eval_microsam.py::convert_images_microsam`:
      - if DNA channel is non-zero, reduce to the mean over all channels;
      - otherwise use the AGP channel.
    The single-channel result is then tiled to (H, W, 3) float32 and fed
    to the predictor.
    """
    from micro_sam import util
    from micro_sam.instance_segmentation import (
        InstanceSegmentationWithDecoder,
    )

    predictor = model_dict["predictor"]
    decoder = model_dict["decoder"]
    masks_out = []

    for i, img in enumerate(images):
        # img: (3, H, W) = [AGP, DNA, 0]
        if np.ptp(img[1]) != 0:  # DNA present -> mean across channels
            single = img.astype("float32").mean(axis=0)
        else:  # no DNA -> use cyto channel (AGP)
            single = img[0]
        img_rgb = np.stack((single,) * 3, axis=-1).astype(np.float32)

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


def _patch_cellsam_nested_tensor():
    """Work around a cellSAM 0.0.dev1 bug in AnchorDETR inference.

    cellSAM's ``generate_bounding_boxes`` feeds a tensor of various ranks
    (3D ``(C, H, W)``, 4D ``(B, C, H, W)``, or 5D ``(1, 3, 3, H, W)`` after
    the WSI pre-processing) into ``self.decode_head`` (AnchorDETR.forward),
    which calls ``nested_tensor_from_tensor_list(samples)``. That helper
    only accepts a *list* of 3D tensors -- given a bare 3D tensor it
    iterates over the channel axis, sees ``samples[0].ndim == 2`` and
    raises ``ValueError('not supported')``. This happens regardless of
    image size or use_wsi.

    We patch the ``nested_tensor_from_tensor_list`` name *in the module
    that calls it* (cellSAM/AnchorDETR/models/anchor_detr.py) so that a
    bare 3D / 4D / 5D tensor is coerced into a list of 3D tensors before
    the original helper runs. Lists and correct-shape inputs pass through
    unchanged, so this is a no-op for every correct call.
    """
    try:
        import torch
        import cellSAM.AnchorDETR.models.anchor_detr as ad
    except Exception as e:  # pragma: no cover - defensive
        print(f"    [WARN] cellsam AnchorDETR patch skipped: {e}")
        return

    if getattr(ad, "_u2os_nested_patched", False):
        return

    _orig = ad.nested_tensor_from_tensor_list

    def _patched(tensor_list):
        # Best-effort coercion so tensor_list[0].ndim == 3 (what _orig needs).
        if torch.is_tensor(tensor_list):
            if tensor_list.ndim == 3:
                # (C,H,W) -> [ (C,H,W) ] : a list with one 3D tensor.
                tensor_list = [tensor_list]
            elif tensor_list.ndim == 4:
                # (B,C,H,W) -> list of B (C,H,W) tensors.
                tensor_list = [t for t in tensor_list]
            elif tensor_list.ndim == 2:
                # (H,W) -> [ (1,H,W) ] : add channel dim, wrap in list.
                tensor_list = [tensor_list.unsqueeze(0)]
            elif tensor_list.ndim == 5:
                # Observed: (1, 3, 3, H, W). cellSAM's sam_bbox_preprocessing
                # stacks a per-image tensor that already carries a spurious
                # duplicated dim, giving (batch=1, dup=3, C=3, H, W).
                # forward_inference computes orig_target_sizes from the batch
                # dim (len == 1), so we must keep exactly ONE (C, H, W) image
                # -- collapsing to a batch of 3 makes len(out_logits)=3 !=
                # len(target_sizes)=1 and trips anchor_detr.py's
                # `assert len(out_logits) == len(target_sizes)`.
                tensor_list = [tensor_list[0, 0]]  # (C, H, W), batch stays 1
        return _orig(tensor_list)

    ad.nested_tensor_from_tensor_list = _patched
    ad._u2os_nested_patched = True


def _segment_cellsam(images):
    """Segment with CellSAM (per-image inference) on in-memory RGB arrays.

    Each input is a (H, W, 3) uint8 numpy array with AGP (cyto) in channel 0,
    DNA (nucleus) in channel 1, and zeros in channel 2 (built in-memory by
    `_agp_dna_to_rgb_uint8`). The bare `cellsam_pipeline(img)` call style is
    the same one used successfully in
    `counting/eval_cellsam_counting.py:86`, which is the only invocation
    verified to run end-to-end on cellSAM 0.0.dev1.

    Two things are required for cellSAM 0.0.dev1 to run on U2OS-Cell-Painting:

      1. **AnchorDETR patch** (``_patch_cellsam_nested_tensor``): cellSAM's
         inference path passes a bare 3D / 4D / 5D tensor into
         ``nested_tensor_from_tensor_list``, which raises
         ``ValueError('not supported')``. The patch coerces it to a list
         of 3D tensors. This is the actual root cause and is independent
         of image size or use_wsi -- so it is still needed even though we
         no longer pre-resize to 1024x1024.

      2. **Pre-processed uint8 RGB input**: the input must already be
         (H, W, 3) uint8 with AGP in channel 0 and DNA in channel 1 -- that
         is the format cellSAM's internal `normalize_image` + cv2 CLAHE
         expects (uint16 raw channels would be rejected with "not
         supported"). We do the dtype conversion + stacking in
         `_agp_dna_to_rgb_uint8` so the inner loop here has zero
         image-processing overhead.

    cellSAM may return a mask at a different size than the input (it
    resizes internally to its native 512x512 grid). We resize the
    predicted labels back to the original (H, W) with nearest-neighbour
    interpolation to preserve integer instance IDs.
    """
    import cv2
    from cellSAM import cellsam_pipeline

    _patch_cellsam_nested_tensor()

    masks_out = []

    for i, img in enumerate(images):
        H0, W0 = img.shape[:2]
        try:
            # Bare call -- mirrors counting/eval_cellsam_counting.py:86.
            # cellSAM's defaults handle dtype / range / resize internally.
            mask = np.asarray(cellsam_pipeline(img))

            # cellSAM may return a mask at a different resolution than the
            # input (depends on internal tiling). Resize back to (H0, W0)
            # with nearest-neighbour so integer instance IDs are preserved.
            if mask.shape != (H0, W0):
                mask = cv2.resize(
                    mask.astype(np.int32), (W0, H0),
                    interpolation=cv2.INTER_NEAREST,
                )
            masks_out.append(_postprocess_mask(mask))
        except Exception as e:
            # Print full traceback for the FIRST failure so we can see
            # exactly where in cellsam the error originates.
            import traceback
            if sum(1 for m in masks_out if m.max() == 0) <= 1:
                print(f"    [ERROR] cellsam image {i}: {e}")
                print(f"    img shape={img.shape} dtype={img.dtype} "
                      f"range=[{img.min()}, {img.max()}]")
                traceback.print_exc()
            else:
                print(f"    [ERROR] cellsam image {i}: {e}")
            masks_out.append(np.zeros((H0, W0), dtype=np.uint32))

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
        list of uint16 masks (converted from internal uint32 at save time)
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
    """List field keys for a plate.

    Fields are enumerated from the cyto-channel (AGP) directory, since AGP is
    required by every model. cellpose-family models additionally load DNA.

    Returns:
        list of field_key strings (e.g. "A01_s1")
    """
    plate_dir = config.IMAGES_DIR / plate_name / config.SEG_CYTO_CHANNEL
    if not plate_dir.exists():
        print(f"  [WARN] Plate directory not found: {plate_dir}")
        return []

    files = sorted(plate_dir.glob("*.tif"))
    return [_get_field_key(f.name) for f in files]


def segment_plate(plate_name, model_name, model=None,
                  diameter=None, force=False,
                  lo_pct=1.0, hi_pct=99.9):
    """Segment all fields in a plate using the specified model.

    Args:
        plate_name: plate identifier
        model_name: model name
        model: pre-loaded model (or None to load)
        diameter: cell diameter
        force: re-segment even if mask exists
        lo_pct, hi_pct: percentile endpoints for the cellsam in-memory
            uint8 conversion (ignored for the other 4 models).
    """
    fields = get_plate_fields(plate_name)
    if not fields:
        return

    mask_dir = config.masks_dir(model_name) / plate_name
    mask_dir.mkdir(parents=True, exist_ok=True)

    # Build todo list
    todo = []
    done = 0
    skipped_incomplete = 0
    for field_key in fields:
        mask_path = mask_dir / f"{field_key}_mask.tif"
        if not force and mask_path.exists():
            done += 1
            continue
        # Pre-check: skip fields whose required channel TIFFs are not yet
        # fully on disk. This makes `download.py` and `segment.py` safe to
        # run in parallel -- an in-flight download simply defers the field
        # to the next segment.py invocation rather than crashing the batch.
        is_ready, missing = _field_ready(plate_name, field_key, model_name)
        if not is_ready:
            skipped_incomplete += 1
            continue
        todo.append((field_key, mask_path))

    if done > 0 and not force:
        print(f"  Skipping {done} already segmented")
    if skipped_incomplete > 0:
        print(f"  Skipping {skipped_incomplete} fields with missing channels "
              f"(likely still downloading; will retry on next run)")
    if not todo:
        if skipped_incomplete == 0:
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
    n_corrupt = 0
    t_total = time.time()

    for i in range(0, len(todo), max(BATCH, 1)):
        batch_items = todo[i:i + BATCH]

        # Load per-model segmentation input images (AGP+DNA or AGP only).
        # A field whose TIFFs cannot be decoded is quarantined and dropped
        # from the batch instead of killing the whole run.
        images = []
        loaded_items = []
        for field_key, mask_path in batch_items:
            try:
                img = _load_field_image(plate_name, field_key, model_name,
                                        lo_pct=lo_pct, hi_pct=hi_pct)
            except _BAD_TIFF_ERRORS as exc:
                broken = _quarantine_broken_channels(plate_name, field_key)
                n_corrupt += 1
                print(f"    [WARN] {plate_name}/{field_key}: undecodable TIFF "
                      f"({type(exc).__name__}: {exc}); quarantined "
                      f"{broken or ['<none>']} as *.corrupt -- re-run "
                      f"download.py for this plate to re-extract. Skipping.")
                continue
            images.append(img)
            loaded_items.append((field_key, mask_path))

        if not images:
            continue

        # Segment
        masks = segment_batch(model_name, model, images)

        # Save masks
        for mask, (_, mask_path) in zip(masks, loaded_items):
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            imwrite(str(mask_path), mask.astype(np.uint16), compression="deflate")

        n_processed += len(masks)

        if (i // max(BATCH, 1)) % 2 == 0 or i + BATCH >= len(todo):
            elapsed = time.time() - t_total
            rate = n_processed / elapsed if elapsed > 0 else 0
            print(f"    [{model_name}] {n_processed}/{len(todo)} fields "
                  f"({rate:.1f} fields/s)")

    elapsed = time.time() - t_total
    rate = n_processed / elapsed if elapsed > 0 else 0.0
    print(f"  Plate {plate_name} [{model_name}]: {n_processed} fields "
          f"in {elapsed:.1f}s ({rate:.1f} fields/s)")
    if n_corrupt > 0:
        print(f"  Plate {plate_name} [{model_name}]: {n_corrupt} fields "
              f"skipped with undecodable TIFFs (quarantined as *.corrupt)")


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
            ch_dir = plate_dir / config.SEG_CYTO_CHANNEL
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
        description="Segment U2OS-Cell-Painting images (whole-cell) with multiple models")
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
    parser.add_argument("--lo-pct", type=float, default=1.0,
                        help="Lower percentile for cellsam in-memory "
                             "uint8 conversion (default 1.0)")
    parser.add_argument("--hi-pct", type=float, default=99.9,
                        help="Upper percentile for cellsam in-memory "
                             "uint8 conversion (default 99.9)")
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
               (plate_dir / config.SEG_CYTO_CHANNEL).exists():
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
                          diameter=args.diameter, force=args.force,
                          lo_pct=args.lo_pct, hi_pct=args.hi_pct)

        # Restore sys.path for cellpose3 (avoid conflict with cellpose4)
        if model_name == "cellpose3":
            cp3_path = str(config.CELLPOSE_ROOT / 'cellpose-cp3')
            if cp3_path in sys.path:
                sys.path.remove(cp3_path)


if __name__ == "__main__":
    main()
