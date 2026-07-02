"""
Cellpose-SAM segmentation module for Xenium images.

Handles large-image tiled segmentation using the Cellpose-SAM model.
Supports both morphology and H&E image segmentation modes.
"""

import logging
import time
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from tqdm import trange

from .. import config

logger = logging.getLogger(__name__)


class CPSAMSegmentor:
    """Segment large Xenium images using Cellpose-SAM.

    Handles tiled processing for large images to fit GPU memory.
    Supports both morphology (DAPI/nuclear) and H&E staining modes.

    Attributes:
        model: CellposeModel instance.
        gpu: Whether GPU is being used.
        params: Segmentation parameters for current mode.
    """

    def __init__(
        self,
        gpu: bool = True,
        pretrained_model: str = "cpsam",
        device: Optional[str] = None,
        use_bfloat16: bool = True,
    ):
        """Initialize the Cellpose-SAM segmentor.

        Args:
            gpu: Whether to use GPU.
            pretrained_model: Model name or path ("cpsam" for default).
            device: Specific torch device (e.g., "cuda:0", "cpu").
            use_bfloat16: Use half precision for model weights.
        """
        import torch
        from cellpose.models import CellposeModel

        # Convert string device to torch.device if needed
        if isinstance(device, str):
            device = torch.device(device)

        self.model = CellposeModel(
            gpu=gpu,
            pretrained_model=pretrained_model,
            device=device,
            use_bfloat16=use_bfloat16,
        )
        self.gpu = gpu
        self.params = {}

        logger.info(
            f"Initialized CPSAMSegmentor with device={self.model.device}, "
            f"pretrained_model={pretrained_model}"
        )

    def segment_morphology(
        self,
        image: np.ndarray,
        **kwargs,
    ) -> Tuple[np.ndarray, List, np.ndarray]:
        """Segment a morphology (nuclear stain) image.

        Args:
            image: RGB image as uint8 array of shape (H, W, 3).
            **kwargs: Override default segmentation parameters (see config).

        Returns:
            Tuple of (masks, flows, styles):
                masks: uint32 label array of shape (H, W), 0=background.
                flows: Flow predictions for visualization.
                styles: Style vectors (empty for CP4 compatibility).
        """
        params = {**config.SEGMENTATION_PARAMS["morphology"], **kwargs}
        self.params = params
        return self._segment(image, params)

    def segment_he(
        self,
        image: np.ndarray,
        **kwargs,
    ) -> Tuple[np.ndarray, List, np.ndarray]:
        """Segment an H&E stained image.

        Args:
            image: RGB image as uint8 array of shape (H, W, 3).
            **kwargs: Override default segmentation parameters (see config).

        Returns:
            Tuple of (masks, flows, styles).
        """
        params = {**config.SEGMENTATION_PARAMS["he"], **kwargs}
        self.params = params
        return self._segment(image, params)

    def _segment(
        self,
        image: np.ndarray,
        params: Dict,
    ) -> Tuple[np.ndarray, List, np.ndarray]:
        """Run cellpose-sam segmentation on an image.

        Handles large images by tiling internally. For smaller images
        that fit in GPU memory, runs directly.

        Args:
            image: RGB uint8 image (H, W, 3).
            params: Segmentation parameter dictionary.

        Returns:
            Tuple of (masks, flows, styles).
        """
        h, w = image.shape[:2]
        logger.info(
            f"Segmenting image of size {h}x{w} "
            f"with params: diameter={params['diameter']}, "
            f"flow_threshold={params['flow_threshold']}, "
            f"cellprob_threshold={params['cellprob_threshold']}"
        )

        # Cellpose expects float inputs in [0, 1] range
        if image.dtype == np.uint8:
            img_float = image.astype(np.float32) / 255.0
        elif image.dtype == np.uint16:
            img_float = image.astype(np.float32) / 65535.0
        else:
            img_float = image.astype(np.float32)

        # Check if image-level tiling is needed to avoid GPU OOM
        # The network inference is tiled internally, but post-processing
        # (compute_masks) operates on full-resolution tensors on GPU.
        max_tile_size = params.get("max_tile_size", 2000)
        if h > max_tile_size or w > max_tile_size:
            return self._segment_tiled(img_float, params)

        tic = time.time()
        masks, flows, styles = self.model.eval(
            img_float,
            batch_size=params.get("batch_size", 8),
            channel_axis=-1,
            normalize=True,
            diameter=params.get("diameter"),
            flow_threshold=params.get("flow_threshold", 0.4),
            cellprob_threshold=params.get("cellprob_threshold", 0.0),
            min_size=params.get("min_size", 15),
            max_size_fraction=params.get("max_size_fraction", 0.4),
            tile_overlap=params.get("tile_overlap", 0.1),
            bsize=params.get("bsize", 256),
            augment=params.get("augment", False),
            resample=params.get("resample", True),
        )
        elapsed = time.time() - tic

        n_cells = len(np.unique(masks)) - 1  # exclude background (0)
        logger.info(
            f"Segmentation complete: {n_cells} cells found in {elapsed:.1f}s"
        )

        return masks, flows, styles

    def _segment_tiled(
        self,
        img_float: np.ndarray,
        params: Dict,
    ) -> Tuple[np.ndarray, List, np.ndarray]:
        """Segment a large image by splitting into overlapping tiles.

        Each tile is processed independently through model.eval() to avoid
        GPU OOM during post-processing. Tiles overlap by tile_min_overlap
        pixels; overlap regions are split equally between adjacent tiles
        for seamless stitching.

        Args:
            img_float: Float image (H, W, C) in [0, 1] range.
            params: Segmentation parameter dictionary.

        Returns:
            Tuple of (masks, flows, styles).
        """
        h, w = img_float.shape[:2]
        tile_size = params.get("max_tile_size", 2000)
        overlap = params.get("tile_min_overlap", max(200, tile_size // 10))
        step = max(1, tile_size - overlap)

        # Compute tile grid
        y_starts = list(range(0, h, step))
        x_starts = list(range(0, w, step))
        n_tiles = len(y_starts) * len(x_starts)

        logger.info(
            f"Image too large ({h}x{w}), using tiled segmentation "
            f"({len(y_starts)}x{len(x_starts)} tiles, tile_size={tile_size}, "
            f"overlap={overlap})"
        )

        # Initialize full-size outputs
        full_mask = np.zeros((h, w), dtype=np.uint32)
        full_dP = np.zeros((2, h, w), dtype=np.float32)
        full_cellprob = np.zeros((h, w), dtype=np.float32)
        full_weight = np.zeros((h, w), dtype=np.float32)

        next_label = 1
        styles_result = None

        for yi, y1 in enumerate(y_starts):
            for xi, x1 in enumerate(x_starts):
                y2 = min(y1 + tile_size, h)
                x2 = min(x1 + tile_size, w)

                tile = img_float[y1:y2, x1:x2]

                tile_idx = yi * len(x_starts) + xi + 1
                logger.info(
                    f"  Tile [{tile_idx}/{n_tiles}]: "
                    f"pos=({y1}:{y2}, {x1}:{x2}), size={tile.shape[:2]}"
                )

                tic_tile = time.time()
                mask_tile, flows_tile, styles_tile = self.model.eval(
                    tile,
                    batch_size=params.get("batch_size", 8),
                    channel_axis=-1,
                    normalize=True,
                    diameter=params.get("diameter"),
                    flow_threshold=params.get("flow_threshold", 0.4),
                    cellprob_threshold=params.get("cellprob_threshold", 0.0),
                    min_size=params.get("min_size", 15),
                    max_size_fraction=params.get("max_size_fraction", 0.4),
                    tile_overlap=params.get("tile_overlap", 0.1),
                    bsize=params.get("bsize", 256),
                    augment=params.get("augment", False),
                    resample=params.get("resample", True),
                )

                # Keep last tile's style as approximation
                styles_result = styles_tile

                ty, tx = mask_tile.shape

                # Determine valid (center) region of this tile
                # For interior edges, crop half the overlap; for image borders keep full
                crop_top = overlap // 2 if y1 > 0 else 0
                crop_bottom = (overlap - overlap // 2) if y2 < h else 0
                crop_left = overlap // 2 if x1 > 0 else 0
                crop_right = (overlap - overlap // 2) if x2 < w else 0

                valid_top = crop_top
                valid_bottom = max(ty - crop_bottom, valid_top + 1)
                valid_left = crop_left
                valid_right = max(tx - crop_right, valid_left + 1)

                py1 = y1 + valid_top
                py2 = y1 + valid_bottom
                px1 = x1 + valid_left
                px2 = x1 + valid_right

                # Stitch mask: relabel and place into full mask
                valid_mask = mask_tile[valid_top:valid_bottom, valid_left:valid_right]
                mask_nonzero = valid_mask > 0
                if mask_nonzero.any():
                    valid_mask = valid_mask.astype(np.uint32)
                    valid_mask[mask_nonzero] += next_label - 1
                    next_label = int(valid_mask.max()) + 1

                # Use np.where to avoid overwriting existing labels in overlap
                full_mask[py1:py2, px1:px2] = np.where(
                    full_mask[py1:py2, px1:px2] == 0,
                    valid_mask,
                    full_mask[py1:py2, px1:px2],
                )

                # Stitch flows (dP and cellprob) by accumulation
                dP_tile = flows_tile[1]      # (2, ty, tx)
                cellprob_tile = flows_tile[2] # (ty, tx)

                full_dP[:, py1:py2, px1:px2] += dP_tile[:, valid_top:valid_bottom, valid_left:valid_right]
                full_cellprob[py1:py2, px1:px2] += cellprob_tile[valid_top:valid_bottom, valid_left:valid_right]
                full_weight[py1:py2, px1:px2] += 1.0

                tile_elapsed = time.time() - tic_tile
                n_in_tile = len(np.unique(mask_tile)) - 1
                logger.info(
                    f"    Tile done: {n_in_tile} cells in {tile_elapsed:.1f}s"
                )

        # Average overlapping flow regions
        weight_mask = full_weight > 0
        full_dP[:, weight_mask] /= full_weight[weight_mask]
        full_cellprob[weight_mask] /= full_weight[weight_mask]

        # Build flows return value matching model.eval format
        from cellpose import plot
        flows_display = plot.dx_to_circ(full_dP)
        flows = [flows_display, full_dP, full_cellprob]

        n_cells = len(np.unique(full_mask)) - 1
        logger.info(f"Tiled segmentation complete: {n_cells} cells total")

        return full_mask, flows, styles_result


# ============================================================================
# Cellpose3 Segmentor
# ============================================================================


class Cellpose3Segmentor:
    """Segment large Xenium images using Cellpose3 (cyto3).

    Uses the Cellpose3 model from cellpose-cp3 package.
    Handles tiled processing for large images to fit GPU memory.

    Attributes:
        model: Cellpose model instance (model_type="cyto3").
        gpu: Whether GPU is being used.
        params: Segmentation parameters for current mode.
    """

    def __init__(
        self,
        gpu: bool = True,
        device: Optional[str] = None,
    ):
        """Initialize the Cellpose3 segmentor.

        Args:
            gpu: Whether to use GPU.
            device: Specific torch device (e.g., "cuda:0", "cpu").
        """
        import importlib
        import sys
        from pathlib import Path

        # Insert cellpose-cp3 at the front of sys.path so that
        # "import cellpose" resolves to cellpose-cp3 instead of the
        # system-installed cellpose4.
        # Path: segmentor.py -> segmentation/ -> cpsam_xenium_analysis/
        #       -> Human_lung/ -> Xenium/ -> Spatial_analysis/ -> Cellpose/
        _project_root = Path(__file__).resolve().parents[5]
        _cp3_dir = str(_project_root / "cellpose-cp3")
        logger.info(f"Cellpose3: project_root={_project_root}, cp3_dir={_cp3_dir}")
        if _cp3_dir not in sys.path:
            sys.path.insert(0, _cp3_dir)

        # Save and remove any previously cached cellpose modules so that the
        # import below picks up cellpose-cp3.  We restore them afterwards so
        # that other code holding references to cellpose4 is not affected.
        _saved = {
            k: sys.modules.pop(k)
            for k in list(sys.modules)
            if k == "cellpose" or k.startswith("cellpose.")
        }
        logger.info(
            f"Cellpose3: removed {len(_saved)} cached cellpose modules from sys.modules"
        )

        try:
            from cellpose import models as _cp3_models
            # Use getattr to avoid Python name-mangling of dunder attrs
            # inside a class method (__file -> _Cellpose3Segmentor__file)
            _models_path = getattr(_cp3_models, "__file__", "<unknown>")
            logger.info(
                f"Cellpose3: loaded models from {_models_path}, "
                f"has Cellpose={hasattr(_cp3_models, 'Cellpose')}"
            )
            if not hasattr(_cp3_models, "Cellpose"):
                raise RuntimeError(
                    f"cellpose.models loaded from {_models_path} does not "
                    f"have 'Cellpose' class. This likely means cellpose-cp3 was not "
                    f"found and the system-installed cellpose4 was loaded instead. "
                    f"cp3_dir={_cp3_dir}, sys.path[:3]={sys.path[:3]}"
                )
            _Cellpose = _cp3_models.Cellpose
        finally:
            # Restore previously cached cellpose4 modules
            sys.modules.update(_saved)

        if isinstance(device, str):
            import torch
            device = torch.device(device)

        self.model = _Cellpose(gpu=gpu, model_type="cyto3")
        self.gpu = gpu
        self.params = {}

        logger.info(
            f"Initialized Cellpose3Segmentor with model_type=cyto3, gpu={gpu}"
        )

    def segment_morphology(
        self,
        image: np.ndarray,
        **kwargs,
    ) -> Tuple[np.ndarray, List, np.ndarray]:
        """Segment a morphology (nuclear stain) image.

        Args:
            image: RGB image as uint8 array of shape (H, W, 3).
            **kwargs: Override default segmentation parameters.

        Returns:
            Tuple of (masks, flows, styles).
        """
        params = {**config.SEGMENTATION_PARAMS["cellpose3_morphology"], **kwargs}
        self.params = params
        return self._segment(image, params)

    def segment_he(
        self,
        image: np.ndarray,
        **kwargs,
    ) -> Tuple[np.ndarray, List, np.ndarray]:
        """Segment an H&E stained image.

        Args:
            image: RGB image as uint8 array of shape (H, W, 3).
            **kwargs: Override default segmentation parameters.

        Returns:
            Tuple of (masks, flows, styles).
        """
        params = {**config.SEGMENTATION_PARAMS["cellpose3_he"], **kwargs}
        self.params = params
        return self._segment(image, params)

    def _segment(
        self,
        image: np.ndarray,
        params: Dict,
    ) -> Tuple[np.ndarray, List, np.ndarray]:
        """Run Cellpose3 segmentation on an image.

        Handles large images by tiling internally.

        Args:
            image: RGB uint8 image (H, W, 3).
            params: Segmentation parameter dictionary.

        Returns:
            Tuple of (masks, flows, styles).
        """
        h, w = image.shape[:2]
        logger.info(
            f"Segmenting image of size {h}x{w} with Cellpose3 "
            f"(diameter={params.get('diameter', 0)}, "
            f"bsize={params.get('bsize', 224)})"
        )

        # Convert to grayscale for Cellpose3
        if image.ndim == 3 and image.shape[2] in (3, 4):
            img_gray = image[:, :, 0]  # use first channel (DAPI for morphology)
        else:
            img_gray = image

        # Cellpose3 expects float inputs
        if img_gray.dtype == np.uint8:
            img_float = img_gray.astype(np.float32) / 255.0
        elif img_gray.dtype == np.uint16:
            img_float = img_gray.astype(np.float32) / 65535.0
        else:
            img_float = img_gray.astype(np.float32)

        # Check if image-level tiling is needed
        max_tile_size = params.get("max_tile_size", 2000)
        if h > max_tile_size or w > max_tile_size:
            return self._segment_tiled(img_float, params)

        tic = time.time()
        # Cellpose.eval() returns (masks, flows, styles, diams) — 4 values
        masks, flows, styles, _diams = self.model.eval(
            img_float,
            diameter=params.get("diameter", 0),
            channels=params.get("channels", [1, 0]),
            batch_size=params.get("batch_size", 8),
            bsize=params.get("bsize", 224),
            niter=params.get("niter", 2000),
            tile_overlap=params.get("tile_overlap", 0.5),
            flow_threshold=params.get("flow_threshold", 0.4),
            cellprob_threshold=params.get("cellprob_threshold", 0.0),
            augment=params.get("augment", True),
            min_size=params.get("min_size", 15),
        )
        elapsed = time.time() - tic

        n_cells = len(np.unique(masks)) - 1
        logger.info(
            f"Cellpose3 segmentation complete: {n_cells} cells found in {elapsed:.1f}s"
        )

        return masks, flows, styles

    def _segment_tiled(
        self,
        img_float: np.ndarray,
        params: Dict,
    ) -> Tuple[np.ndarray, List, np.ndarray]:
        """Segment a large image by splitting into overlapping tiles.

        Args:
            img_float: Float image (H, W) in [0, 1] range.
            params: Segmentation parameter dictionary.

        Returns:
            Tuple of (masks, flows, styles).
        """
        h, w = img_float.shape[:2]
        tile_size = params.get("max_tile_size", 2000)
        overlap = params.get("tile_min_overlap", max(200, tile_size // 10))
        step = max(1, tile_size - overlap)

        y_starts = list(range(0, h, step))
        x_starts = list(range(0, w, step))
        n_tiles = len(y_starts) * len(x_starts)

        logger.info(
            f"Image too large ({h}x{w}), using tiled segmentation "
            f"({len(y_starts)}x{len(x_starts)} tiles, tile_size={tile_size}, "
            f"overlap={overlap})"
        )

        full_mask = np.zeros((h, w), dtype=np.uint32)
        next_label = 1
        styles_result = None

        for yi, y1 in enumerate(y_starts):
            for xi, x1 in enumerate(x_starts):
                y2 = min(y1 + tile_size, h)
                x2 = min(x1 + tile_size, w)

                tile = img_float[y1:y2, x1:x2]

                tile_idx = yi * len(x_starts) + xi + 1
                logger.info(
                    f"  Tile [{tile_idx}/{n_tiles}]: "
                    f"pos=({y1}:{y2}, {x1}:{x2}), size={tile.shape[:2]}"
                )

                tic_tile = time.time()
                # Cellpose.eval() returns (masks, flows, styles, diams) — 4 values
                mask_tile, flows_tile, styles_tile, _diams = self.model.eval(
                    tile,
                    diameter=params.get("diameter", 0),
                    channels=params.get("channels", [1, 0]),
                    batch_size=params.get("batch_size", 8),
                    bsize=params.get("bsize", 224),
                    niter=params.get("niter", 2000),
                    tile_overlap=params.get("tile_overlap", 0.5),
                    flow_threshold=params.get("flow_threshold", 0.4),
                    cellprob_threshold=params.get("cellprob_threshold", 0.0),
                    augment=params.get("augment", True),
                    min_size=params.get("min_size", 15),
                )

                styles_result = styles_tile

                ty, tx = mask_tile.shape

                crop_top = overlap // 2 if y1 > 0 else 0
                crop_bottom = (overlap - overlap // 2) if y2 < h else 0
                crop_left = overlap // 2 if x1 > 0 else 0
                crop_right = (overlap - overlap // 2) if x2 < w else 0

                valid_top = crop_top
                valid_bottom = max(ty - crop_bottom, valid_top + 1)
                valid_left = crop_left
                valid_right = max(tx - crop_right, valid_left + 1)

                py1 = y1 + valid_top
                py2 = y1 + valid_bottom
                px1 = x1 + valid_left
                px2 = x1 + valid_right

                valid_mask = mask_tile[valid_top:valid_bottom, valid_left:valid_right]
                mask_nonzero = valid_mask > 0
                if mask_nonzero.any():
                    valid_mask = valid_mask.astype(np.uint32)
                    valid_mask[mask_nonzero] += next_label - 1
                    next_label = int(valid_mask.max()) + 1

                full_mask[py1:py2, px1:px2] = np.where(
                    full_mask[py1:py2, px1:px2] == 0,
                    valid_mask,
                    full_mask[py1:py2, px1:px2],
                )

                tile_elapsed = time.time() - tic_tile
                n_in_tile = len(np.unique(mask_tile)) - 1
                logger.info(
                    f"    Tile done: {n_in_tile} cells in {tile_elapsed:.1f}s"
                )

        n_cells = len(np.unique(full_mask)) - 1
        logger.info(f"Tiled Cellpose3 segmentation complete: {n_cells} cells total")

        return full_mask, [], styles_result


# ============================================================================
# CellSAM Segmentor
# ============================================================================


class CellSAMSegmentor:
    """Segment large Xenium images using CellSAM.

    Uses the CellSAM pipeline for instance segmentation.
    Handles tiled processing for large images to fit GPU memory.

    Attributes:
        gpu: Whether GPU is being used.
        device: Device string (e.g., "cuda", "cpu").
        params: Segmentation parameters for current mode.
    """

    def __init__(
        self,
        gpu: bool = True,
        device: Optional[str] = None,
    ):
        """Initialize the CellSAM segmentor.

        Args:
            gpu: Whether to use GPU.
            device: Device string (e.g., "cuda:0", "cpu").
        """
        try:
            from cellSAM import cellsam_pipeline
        except ImportError:
            raise ImportError(
                "cellSAM is not installed. Please run: pip install cellSAM"
            )

        self.gpu = gpu
        self.device = device or ("cuda" if gpu else "cpu")
        self.params = {}

        logger.info(
            f"Initialized CellSAMSegmentor with device={self.device}"
        )

    def segment_morphology(
        self,
        image: np.ndarray,
        **kwargs,
    ) -> Tuple[np.ndarray, List, np.ndarray]:
        """Segment a morphology (nuclear stain) image.

        Args:
            image: RGB image as uint8 array of shape (H, W, 3).
            **kwargs: Override default segmentation parameters.

        Returns:
            Tuple of (masks, flows, styles). flows and styles are empty.
        """
        params = {**config.SEGMENTATION_PARAMS["cellsam_morphology"], **kwargs}
        self.params = params
        return self._segment(image, params)

    def segment_he(
        self,
        image: np.ndarray,
        **kwargs,
    ) -> Tuple[np.ndarray, List, np.ndarray]:
        """Segment an H&E stained image.

        Args:
            image: RGB image as uint8 array of shape (H, W, 3).
            **kwargs: Override default segmentation parameters.

        Returns:
            Tuple of (masks, flows, styles).
        """
        params = {**config.SEGMENTATION_PARAMS["cellsam_he"], **kwargs}
        self.params = params
        return self._segment(image, params)

    @staticmethod
    def _convert_image_cellsam(img):
        """Convert image to CellSAM (H,W,3) format."""
        if img.ndim == 2:
            img = np.stack([img, np.zeros_like(img), np.zeros_like(img)], axis=-1)
        elif img.ndim == 3 and img.shape[2] == 1:
            img = np.concatenate(
                [img, np.zeros_like(img), np.zeros_like(img)], axis=-1
            )
        elif img.ndim == 3 and img.shape[2] >= 3:
            img = img[:, :, :3]
        return img

    def _segment(
        self,
        image: np.ndarray,
        params: Dict,
    ) -> Tuple[np.ndarray, List, np.ndarray]:
        """Run CellSAM segmentation on an image.

        Handles large images by tiling internally.

        Args:
            image: RGB uint8 image (H, W, 3).
            params: Segmentation parameter dictionary.

        Returns:
            Tuple of (masks, flows, styles).
        """
        from cellSAM import cellsam_pipeline

        h, w = image.shape[:2]
        logger.info(f"Segmenting image of size {h}x{w} with CellSAM")

        # Check if image-level tiling is needed
        max_tile_size = params.get("max_tile_size", 2000)
        if h > max_tile_size or w > max_tile_size:
            return self._segment_tiled(image, params)

        img_cellsam = self._convert_image_cellsam(image)

        tic = time.time()
        try:
            mask = cellsam_pipeline(
                img_cellsam,
                use_wsi=params.get("use_wsi", False),
                low_contrast_enhancement=params.get("low_contrast_enhancement", False),
                gauge_cell_size=params.get("gauge_cell_size", False),
            )
        except Exception as e:
            logger.error(f"CellSAM inference failed: {e}")
            mask = np.zeros((h, w), dtype=np.uint16)

        elapsed = time.time() - tic
        n_cells = len(np.unique(mask)) - 1
        logger.info(
            f"CellSAM segmentation complete: {n_cells} cells found in {elapsed:.1f}s"
        )

        return mask, [], None

    def _segment_tiled(
        self,
        image: np.ndarray,
        params: Dict,
    ) -> Tuple[np.ndarray, List, np.ndarray]:
        """Segment a large image by splitting into overlapping tiles.

        Args:
            image: RGB uint8 image (H, W, 3).
            params: Segmentation parameter dictionary.

        Returns:
            Tuple of (masks, flows, styles).
        """
        from cellSAM import cellsam_pipeline

        h, w = image.shape[:2]
        tile_size = params.get("max_tile_size", 2000)
        overlap = params.get("tile_min_overlap", max(200, tile_size // 10))
        step = max(1, tile_size - overlap)

        y_starts = list(range(0, h, step))
        x_starts = list(range(0, w, step))
        n_tiles = len(y_starts) * len(x_starts)

        logger.info(
            f"Image too large ({h}x{w}), using tiled CellSAM segmentation "
            f"({len(y_starts)}x{len(x_starts)} tiles, tile_size={tile_size}, "
            f"overlap={overlap})"
        )

        full_mask = np.zeros((h, w), dtype=np.uint32)
        next_label = 1

        for yi, y1 in enumerate(y_starts):
            for xi, x1 in enumerate(x_starts):
                y2 = min(y1 + tile_size, h)
                x2 = min(x1 + tile_size, w)

                tile = image[y1:y2, x1:x2]
                tile_cellsam = self._convert_image_cellsam(tile)

                tile_idx = yi * len(x_starts) + xi + 1
                logger.info(
                    f"  Tile [{tile_idx}/{n_tiles}]: "
                    f"pos=({y1}:{y2}, {x1}:{x2}), size={tile.shape[:2]}"
                )

                tic_tile = time.time()
                try:
                    mask_tile = cellsam_pipeline(
                        tile_cellsam,
                        use_wsi=params.get("use_wsi", False),
                        low_contrast_enhancement=params.get("low_contrast_enhancement", False),
                        gauge_cell_size=params.get("gauge_cell_size", False),
                    )
                except Exception as e:
                    logger.error(f"  CellSAM tile inference failed: {e}")
                    mask_tile = np.zeros(tile.shape[:2], dtype=np.uint16)

                ty, tx = mask_tile.shape

                crop_top = overlap // 2 if y1 > 0 else 0
                crop_bottom = (overlap - overlap // 2) if y2 < h else 0
                crop_left = overlap // 2 if x1 > 0 else 0
                crop_right = (overlap - overlap // 2) if x2 < w else 0

                valid_top = crop_top
                valid_bottom = max(ty - crop_bottom, valid_top + 1)
                valid_left = crop_left
                valid_right = max(tx - crop_right, valid_left + 1)

                py1 = y1 + valid_top
                py2 = y1 + valid_bottom
                px1 = x1 + valid_left
                px2 = x1 + valid_right

                valid_mask = mask_tile[valid_top:valid_bottom, valid_left:valid_right]
                mask_nonzero = valid_mask > 0
                if mask_nonzero.any():
                    valid_mask = valid_mask.astype(np.uint32)
                    valid_mask[mask_nonzero] += next_label - 1
                    next_label = int(valid_mask.max()) + 1

                full_mask[py1:py2, px1:px2] = np.where(
                    full_mask[py1:py2, px1:px2] == 0,
                    valid_mask,
                    full_mask[py1:py2, px1:px2],
                )

                tile_elapsed = time.time() - tic_tile
                n_in_tile = len(np.unique(mask_tile)) - 1
                logger.info(
                    f"    Tile done: {n_in_tile} cells in {tile_elapsed:.1f}s"
                )

        n_cells = len(np.unique(full_mask)) - 1
        logger.info(f"Tiled CellSAM segmentation complete: {n_cells} cells total")

        return full_mask, [], None


# ============================================================================
# MicroSAM Segmentor
# ============================================================================


class MicroSAMSegmentor:
    """Segment large Xenium images using MicroSAM.

    Uses the MicroSAM InstanceSegmentationWithDecoder pipeline.
    Handles tiled processing for large images to fit GPU memory.

    Attributes:
        model_type: MicroSAM model type (e.g., "vit_l_lm").
        gpu: Whether GPU is being used.
        device: Device string (e.g., "cuda", "cpu").
        predictor: MicroSAM predictor.
        decoder: MicroSAM decoder.
        params: Segmentation parameters for current mode.
    """

    def __init__(
        self,
        gpu: bool = True,
        device: Optional[str] = None,
        model_type: str = "vit_l_lm",
    ):
        """Initialize the MicroSAM segmentor.

        Args:
            gpu: Whether to use GPU.
            device: Device string (e.g., "cuda:0", "cpu").
            model_type: MicroSAM model type, default "vit_l_lm".
        """
        try:
            from micro_sam.instance_segmentation import (
                InstanceSegmentationWithDecoder,
                get_predictor_and_decoder,
            )
        except ImportError:
            raise ImportError(
                "micro_sam is not installed. "
                "Please run: conda install -c conda-forge microsam"
            )

        self.model_type = model_type
        self.gpu = gpu
        self.device = device or ("cuda" if gpu else "cpu")
        self.params = {}

        # Check CUDA availability
        if self.device.startswith("cuda"):
            import torch
            if not torch.cuda.is_available():
                logger.warning("CUDA not available, falling back to CPU")
                self.device = "cpu"

        logger.info(
            f"Loading MicroSAM model (model_type={model_type}) on {self.device}..."
        )
        self.predictor, self.decoder = get_predictor_and_decoder(
            model_type=model_type,
            checkpoint_path=None,
        )
        logger.info(
            f"Initialized MicroSAMSegmentor with model_type={model_type}, "
            f"device={self.device}"
        )

    @staticmethod
    def _convert_image_microsam(img):
        """Convert image to MicroSAM (H,W,3) float32 format."""
        if img.ndim == 3:
            if np.array(img.shape).argmin() == 2:
                img = img.transpose(2, 0, 1)
            if np.ptp(img[1]) != 0:
                img = img.astype("float32").mean(axis=0)
            else:
                img = img[0]
        # img is now (H,W)
        img = np.stack((img,) * 3, axis=-1)  # (H,W,3)
        img = img.astype(np.float32)
        return img

    def segment_morphology(
        self,
        image: np.ndarray,
        **kwargs,
    ) -> Tuple[np.ndarray, List, np.ndarray]:
        """Segment a morphology (nuclear stain) image.

        Args:
            image: RGB image as uint8 array of shape (H, W, 3).
            **kwargs: Override default segmentation parameters.

        Returns:
            Tuple of (masks, flows, styles). flows and styles are empty.
        """
        params = {**config.SEGMENTATION_PARAMS["microsam_morphology"], **kwargs}
        self.params = params
        return self._segment(image, params)

    def segment_he(
        self,
        image: np.ndarray,
        **kwargs,
    ) -> Tuple[np.ndarray, List, np.ndarray]:
        """Segment an H&E stained image.

        Args:
            image: RGB image as uint8 array of shape (H, W, 3).
            **kwargs: Override default segmentation parameters.

        Returns:
            Tuple of (masks, flows, styles).
        """
        params = {**config.SEGMENTATION_PARAMS["microsam_he"], **kwargs}
        self.params = params
        return self._segment(image, params)

    def _segment(
        self,
        image: np.ndarray,
        params: Dict,
    ) -> Tuple[np.ndarray, List, np.ndarray]:
        """Run MicroSAM segmentation on an image.

        Handles large images by tiling internally.

        Args:
            image: RGB uint8 image (H, W, 3).
            params: Segmentation parameter dictionary.

        Returns:
            Tuple of (masks, flows, styles).
        """
        from micro_sam import util
        from micro_sam.instance_segmentation import InstanceSegmentationWithDecoder

        h, w = image.shape[:2]
        logger.info(f"Segmenting image of size {h}x{w} with MicroSAM")

        # Check if image-level tiling is needed
        max_tile_size = params.get("max_tile_size", 2000)
        if h > max_tile_size or w > max_tile_size:
            return self._segment_tiled(image, params)

        image_ms = self._convert_image_microsam(image)

        tic = time.time()
        image_embeddings = util.precompute_image_embeddings(
            predictor=self.predictor,
            input_=image_ms,
            ndim=2,
            verbose=False,
        )

        ais = InstanceSegmentationWithDecoder(self.predictor, self.decoder)
        ais.initialize(
            image=image_ms,
            image_embeddings=image_embeddings,
        )

        mask = ais.generate(output_mode="instance_segmentation")
        elapsed = time.time() - tic

        if mask is None or len(mask) == 0:
            mask = np.zeros((h, w), dtype=np.uint16)

        n_cells = len(np.unique(mask)) - 1
        logger.info(
            f"MicroSAM segmentation complete: {n_cells} cells found in {elapsed:.1f}s"
        )

        return mask, [], None

    def _segment_tiled(
        self,
        image: np.ndarray,
        params: Dict,
    ) -> Tuple[np.ndarray, List, np.ndarray]:
        """Segment a large image by splitting into overlapping tiles.

        Args:
            image: RGB uint8 image (H, W, 3).
            params: Segmentation parameter dictionary.

        Returns:
            Tuple of (masks, flows, styles).
        """
        from micro_sam import util
        from micro_sam.instance_segmentation import InstanceSegmentationWithDecoder

        h, w = image.shape[:2]
        tile_size = params.get("max_tile_size", 2000)
        overlap = params.get("tile_min_overlap", max(200, tile_size // 10))
        step = max(1, tile_size - overlap)

        y_starts = list(range(0, h, step))
        x_starts = list(range(0, w, step))
        n_tiles = len(y_starts) * len(x_starts)

        logger.info(
            f"Image too large ({h}x{w}), using tiled MicroSAM segmentation "
            f"({len(y_starts)}x{len(x_starts)} tiles, tile_size={tile_size}, "
            f"overlap={overlap})"
        )

        full_mask = np.zeros((h, w), dtype=np.uint32)
        next_label = 1

        for yi, y1 in enumerate(y_starts):
            for xi, x1 in enumerate(x_starts):
                y2 = min(y1 + tile_size, h)
                x2 = min(x1 + tile_size, w)

                tile = image[y1:y2, x1:x2]
                tile_ms = self._convert_image_microsam(tile)

                tile_idx = yi * len(x_starts) + xi + 1
                logger.info(
                    f"  Tile [{tile_idx}/{n_tiles}]: "
                    f"pos=({y1}:{y2}, {x1}:{x2}), size={tile.shape[:2]}"
                )

                tic_tile = time.time()
                try:
                    image_embeddings = util.precompute_image_embeddings(
                        predictor=self.predictor,
                        input_=tile_ms,
                        ndim=2,
                        verbose=False,
                    )

                    ais = InstanceSegmentationWithDecoder(
                        self.predictor, self.decoder
                    )
                    ais.initialize(
                        image=tile_ms,
                        image_embeddings=image_embeddings,
                    )

                    mask_tile = ais.generate(
                        output_mode="instance_segmentation"
                    )

                    if mask_tile is None or len(mask_tile) == 0:
                        mask_tile = np.zeros(tile.shape[:2], dtype=np.uint16)

                except Exception as e:
                    logger.error(f"  MicroSAM tile inference failed: {e}")
                    mask_tile = np.zeros(tile.shape[:2], dtype=np.uint16)

                ty, tx = mask_tile.shape

                crop_top = overlap // 2 if y1 > 0 else 0
                crop_bottom = (overlap - overlap // 2) if y2 < h else 0
                crop_left = overlap // 2 if x1 > 0 else 0
                crop_right = (overlap - overlap // 2) if x2 < w else 0

                valid_top = crop_top
                valid_bottom = max(ty - crop_bottom, valid_top + 1)
                valid_left = crop_left
                valid_right = max(tx - crop_right, valid_left + 1)

                py1 = y1 + valid_top
                py2 = y1 + valid_bottom
                px1 = x1 + valid_left
                px2 = x1 + valid_right

                valid_mask = mask_tile[valid_top:valid_bottom, valid_left:valid_right]
                mask_nonzero = valid_mask > 0
                if mask_nonzero.any():
                    valid_mask = valid_mask.astype(np.uint32)
                    valid_mask[mask_nonzero] += next_label - 1
                    next_label = int(valid_mask.max()) + 1

                full_mask[py1:py2, px1:px2] = np.where(
                    full_mask[py1:py2, px1:px2] == 0,
                    valid_mask,
                    full_mask[py1:py2, px1:px2],
                )

                tile_elapsed = time.time() - tic_tile
                n_in_tile = len(np.unique(mask_tile)) - 1
                logger.info(
                    f"    Tile done: {n_in_tile} cells in {tile_elapsed:.1f}s"
                )

        n_cells = len(np.unique(full_mask)) - 1
        logger.info(f"Tiled MicroSAM segmentation complete: {n_cells} cells total")

        return full_mask, [], None
