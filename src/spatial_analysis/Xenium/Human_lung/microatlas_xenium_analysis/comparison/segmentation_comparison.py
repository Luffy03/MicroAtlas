"""
Segmentation comparison module for systematic evaluation of
Xenium vs Cellpose-SAM (morphology) vs Cellpose-SAM (H&E) segmentation.

Integrates all metrics from the metrics module and produces
a unified comparison report.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .metrics import (
    cell_matching,
    compare_cell_counts,
    compare_area_distribution,
    compare_gene_detection,
    dice_coefficient,
    iou_score,
    hausdorff_distance,
)
from ..segmentation.post_process import masks_to_polygons

logger = logging.getLogger(__name__)


class SegmentationComparator:
    """Systematic comparison of multiple segmentation methods.

    Compares:
    - Cell count and density
    - Cell area distribution
    - Cell morphology (shape metrics)
    - Per-cell overlap metrics (Dice, IoU) for matched cells
    - Boundary distance (Hausdorff)
    - Gene detection sensitivity
    - Cell type composition
    """

    def __init__(self):
        self.masks = {}
        self.expression_matrices = {}
        self.gene_names = []
        self.cell_types = {}

    def add_segmentation(
        self,
        name: str,
        mask: np.ndarray,
        expression_matrix=None,
        cell_types: Optional[pd.DataFrame] = None,
    ) -> None:
        """Add a segmentation result for comparison.

        Args:
            name: Method name (e.g., "xenium", "cpsam_morphology", "cpsam_he").
            mask: Label array.
            expression_matrix: Optional expression matrix (n_cells, n_genes).
            cell_types: Optional DataFrame with cell type assignments.
        """
        self.masks[name] = mask
        if expression_matrix is not None:
            self.expression_matrices[name] = expression_matrix
        if cell_types is not None:
            self.cell_types[name] = cell_types

        n_cells = len(np.unique(mask)) - 1
        logger.info(f"Added segmentation '{name}': {n_cells} cells")

    def summary_report(self) -> Dict[str, pd.DataFrame]:
        """Generate a comprehensive comparison report.

        Returns:
            Dict with:
                - "cell_counts": DataFrame
                - "area_distribution": DataFrame
                - "pairwise_overlap": DataFrame (Dice/IoU between methods)
                - "gene_detection": DataFrame (if expression matrices available)
        """
        report = {}

        # 1. Cell counts
        report["cell_counts"] = compare_cell_counts(self.masks)

        # 2. Area distribution
        report["area_distribution"] = compare_area_distribution(self.masks)

        # 3. Pairwise overlap (by binary mask comparison)
        pairwise = []
        methods = list(self.masks.keys())
        for i, m1 in enumerate(methods):
            for j, m2 in enumerate(methods):
                if i >= j:
                    continue
                dice = dice_coefficient(self.masks[m1], self.masks[m2])
                iou = iou_score(self.masks[m1], self.masks[m2])
                pairwise.append({
                    "method_1": m1,
                    "method_2": m2,
                    "dice": round(dice, 4),
                    "iou": round(iou, 4),
                })
        report["pairwise_overlap"] = pd.DataFrame(pairwise)

        # 4. Gene detection sensitivity
        if len(self.expression_matrices) > 0:
            report["gene_detection"] = compare_gene_detection(
                self.expression_matrices, self.gene_names
            )

        return report

    def per_cell_comparison(
        self,
        method_a: str,
        method_b: str,
        iou_threshold: float = 0.5,
    ) -> Dict:
        """Detailed per-cell comparison between two methods.

        Args:
            method_a: Name of first method.
            method_b: Name of second method.
            iou_threshold: Minimum IoU for cell matching.

        Returns:
            Dict with matching statistics and matched cell metrics.
        """
        if method_a not in self.masks or method_b not in self.masks:
            raise ValueError(f"Methods must be added first: {method_a}, {method_b}")

        mask_a = self.masks[method_a]
        mask_b = self.masks[method_b]

        # Cell matching
        match_result = cell_matching(mask_a, mask_b, iou_threshold)

        # Boundary distance (only for small masks, O(N_cells * H*W))
        n_a = len(np.unique(mask_a)) - 1
        n_b = len(np.unique(mask_b)) - 1
        max_cells_for_boundary = 5000

        if n_a <= max_cells_for_boundary and n_b <= max_cells_for_boundary:
            polygons_a = masks_to_polygons(mask_a)
            polygons_b = masks_to_polygons(mask_b)

            boundary_distances = []
            for label_a, label_b, iou_val in match_result["matches"]:
                if label_a in polygons_a and label_b in polygons_b:
                    dist = hausdorff_distance(
                        polygons_a[label_a], polygons_b[label_b]
                    )
                    boundary_distances.append(dist)
            mean_boundary_dist = (
                np.mean(boundary_distances) if boundary_distances else 0
            )
            logger.info(
                f"Boundary distances computed for {len(boundary_distances)} matched cells"
            )
        else:
            mean_boundary_dist = -1.0
            logger.warning(
                f"Skipping boundary distance: masks have {n_a}/{n_b} cells "
                f"(max {max_cells_for_boundary})"
            )

        iou_values = [m[2] for m in match_result["matches"]]

        result = {
            "matching": match_result,
            "mean_iou": np.mean(iou_values) if iou_values else 0,
            "median_iou": np.median(iou_values) if iou_values else 0,
            "mean_boundary_distance": mean_boundary_dist,
        }

        logger.info(
            f"Per-cell comparison {method_a} vs {method_b}: "
            f"{len(match_result['matches'])} matched cells, "
            f"mean IoU={result['mean_iou']:.3f}, "
            f"mean boundary dist={result['mean_boundary_distance']:.1f}px"
        )

        return result
