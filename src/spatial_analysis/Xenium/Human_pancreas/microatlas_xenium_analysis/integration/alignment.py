"""
Coordinate alignment module for Xenium data.

Handles conversion between:
- Xenium transcript coordinates (micrometers, global)
- Image pixel coordinates (local to the cropped/acquired image)

The Xenium morphology image and transcript coordinates use the same
coordinate system (micrometers from the origin), but the image was
acquired at 0.2125 um/pixel resolution.
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .. import config

logger = logging.getLogger(__name__)


class CoordinateAligner:
    """Align Xenium transcript coordinates with image pixel coordinates.

    The Xenium data has two coordinate systems:
    1. Transcript coordinates (micrometers) - global, from experiment
    2. Image coordinates (pixels) - 0.2125 um/pixel resolution

    For the morphology and H&E images, we need to convert transcript
    positions to pixel positions to assign transcripts to cell masks.
    """

    def __init__(self):
        # Pixel size in micrometers
        self.pixel_size_um = config.PIXEL_SIZE_UM

        # Image offset (if the image is a cropped region)
        # Set during alignment
        self.x_offset_px = 0
        self.y_offset_px = 0

        # Whether the image is cropped from a larger field
        self.is_cropped = False

    def set_crop_offset(
        self, x_start: int, y_start: int
    ) -> None:
        """Set the crop offset for alignment.

        When working with a cropped image region, transcript coordinates
        need to be shifted by the crop offset.

        Args:
            x_start: X start of crop in pixels.
            y_start: Y start of crop in pixels.
        """
        self.x_offset_px = x_start
        self.y_offset_px = y_start
        self.is_cropped = True
        logger.info(
            f"Set crop offset: x={x_start}px, y={y_start}px"
        )

    def transcripts_to_pixel_coords(
        self,
        transcripts_df: pd.DataFrame,
        crop_roi: Optional[Dict[str, int]] = None,
    ) -> pd.DataFrame:
        """Convert transcript (x, y) micrometers to pixel coordinates.

        Adds columns: x_pixel, y_pixel to the DataFrame.

        Args:
            transcripts_df: DataFrame with 'x_location' and 'y_location' columns.
            crop_roi: Optional crop dict with x_start, y_start, width, height.
                      If provided, only transcripts within the crop region
                      are kept, and coordinates are adjusted.

        Returns:
            DataFrame with added x_pixel, y_pixel columns, and optionally
            filtered to the crop region.
        """
        df = transcripts_df.copy()

        # Convert micrometers to pixels
        df["x_pixel"] = (df["x_location"] / self.pixel_size_um).astype(np.int32)
        df["y_pixel"] = (df["y_location"] / self.pixel_size_um).astype(np.int32)

        # Apply crop filtering
        if crop_roi is not None:
            xs = crop_roi["x_start"]
            ys = crop_roi["y_start"]
            xe = xs + crop_roi["width"]
            ye = ys + crop_roi["height"]

            n_before = len(df)
            df = df[
                (df["x_pixel"] >= xs)
                & (df["x_pixel"] < xe)
                & (df["y_pixel"] >= ys)
                & (df["y_pixel"] < ye)
            ].copy()

            # Shift to local crop coordinates
            df["x_pixel"] -= xs
            df["y_pixel"] -= ys

            logger.info(
                f"Filtered transcripts to crop region: {n_before} -> {len(df)}"
            )

        return df

    def cell_centroids_to_pixels(
        self,
        cells_df: pd.DataFrame,
        crop_roi: Optional[Dict[str, int]] = None,
    ) -> pd.DataFrame:
        """Convert Xenium cell centroid coordinates to pixels.

        Args:
            cells_df: DataFrame with 'x_centroid' and 'y_centroid' columns.
            crop_roi: Optional crop dict.

        Returns:
            DataFrame with added x_pixel, y_pixel columns.
        """
        df = cells_df.copy()
        df["x_pixel"] = (df["x_centroid"] / self.pixel_size_um).astype(np.int32)
        df["y_pixel"] = (df["y_centroid"] / self.pixel_size_um).astype(np.int32)

        if crop_roi is not None:
            xs = crop_roi["x_start"]
            ys = crop_roi["y_start"]
            xe = xs + crop_roi["width"]
            ye = ys + crop_roi["height"]

            n_before = len(df)
            df = df[
                (df["x_pixel"] >= xs)
                & (df["x_pixel"] < xe)
                & (df["y_pixel"] >= ys)
                & (df["y_pixel"] < ye)
            ].copy()

            df["x_pixel"] -= xs
            df["y_pixel"] -= ys

            logger.info(
                f"Filtered cells to crop region: {n_before} -> {len(df)}"
            )

        return df

    def build_cell_id_to_mask_map(
        self,
        xenium_cells_df: pd.DataFrame,
        cell_mask: np.ndarray,
        crop_roi: Optional[Dict[str, int]] = None,
    ) -> Dict[str, int]:
        """Build a mapping from Xenium cell_ids to mask labels.

        For each Xenium cell, find which cellpose-sam mask cell it
        overlaps with (based on centroid position).

        Args:
            xenium_cells_df: DataFrame with cell_id, x_centroid, y_centroid.
            cell_mask: Cellpose-SAM mask array (H, W).
            crop_roi: Optional crop dict.

        Returns:
            Dict mapping xenium_cell_id -> mask_label (or 0 if no match).
        """
        cells_px = self.cell_centroids_to_pixels(xenium_cells_df, crop_roi)

        mapping = {}
        matched = 0
        for _, row in cells_px.iterrows():
            y, x = int(row["y_pixel"]), int(row["x_pixel"])
            if 0 <= y < cell_mask.shape[0] and 0 <= x < cell_mask.shape[1]:
                mask_label = cell_mask[y, x]
                mapping[row["cell_id"]] = int(mask_label)
                if mask_label > 0:
                    matched += 1
            else:
                mapping[row["cell_id"]] = 0

        logger.info(
            f"Built cell-to-mask mapping: {matched}/{len(mapping)} "
            f"Xenium cells matched to cellpose-sam cells"
        )
        return mapping
