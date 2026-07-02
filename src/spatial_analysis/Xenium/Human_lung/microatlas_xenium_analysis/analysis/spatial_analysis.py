"""
Spatial analysis module for Xenium spatial transcriptomics data.

Computes spatial autocorrelation (Moran's I), hotspot detection (Getis-Ord Gi*),
spatially variable gene identification, and region-based analysis.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.spatial import KDTree
from scipy.stats import norm

from .. import config
from tqdm import tqdm

logger = logging.getLogger(__name__)


class SpatialAnalyzer:
    """Analyze spatial patterns of gene expression in tissue.

    Provides methods for:
    - Spatial autocorrelation (Moran's I)
    - Hotspot/coldspot detection (Getis-Ord Gi*)
    - Spatially variable gene identification
    - Regional differential expression
    """

    def __init__(self, k_neighbors: int = 15):
        """Initialize the spatial analyzer.

        Args:
            k_neighbors: Number of nearest neighbors for spatial weights.
        """
        self.k_neighbors = k_neighbors
        self.spatial_weights = None  # Will be computed from coordinates
        self.coords = None  # (n_cells, 2) array of cell centroid coordinates

    def compute_spatial_weights(
        self, coords: np.ndarray
    ) -> np.ndarray:
        """Compute k-nearest neighbor spatial weight matrix.

        Args:
            coords: (n_cells, 2) array of (x, y) cell coordinates.

        Returns:
            Spatial weights matrix (n_cells, n_cells) as binary CSR matrix.
        """
        from scipy.sparse import csr_matrix as csr

        n_cells = coords.shape[0]
        k = min(self.k_neighbors, n_cells - 1)

        # Build KDTree for efficient nearest neighbor search
        tree = KDTree(coords)
        distances, indices = tree.query(coords, k=k + 1)  # +1 because self is included

        # Exclude self (first neighbor is always self with distance 0)
        indices = indices[:, 1:]  # (n_cells, k)
        distances = distances[:, 1:]

        # Build sparse weights matrix (binary: 1 for neighbors)
        rows = np.repeat(np.arange(n_cells), k)
        cols = indices.flatten()
        data = np.ones(n_cells * k, dtype=np.float32)

        weights = csr((data, (rows, cols)), shape=(n_cells, n_cells))

        self.spatial_weights = weights
        self.coords = coords
        logger.info(f"Computed spatial weights: {n_cells} cells, k={k}")

        return weights

    def morans_i(
        self,
        expression_vector: np.ndarray,
    ) -> Dict[str, float]:
        """Compute Moran's I for a single gene's expression.

        Moran's I measures spatial autocorrelation:
        I > 0: clustering (similar values near each other)
        I < 0: dispersion (dissimilar values near each other)
        I = 0: random spatial distribution

        Args:
            expression_vector: 1D array of gene expression values (n_cells,).

        Returns:
            Dict with keys: morans_i, expected_i, variance, z_score, p_value.
        """
        if self.spatial_weights is None:
            raise ValueError(
                "Spatial weights not computed. Call compute_spatial_weights() first."
            )

        n = len(expression_vector)
        expr = expression_vector.astype(np.float64)
        expr_mean = expr.mean()

        # Deviations from mean
        z = expr - expr_mean

        # Sum of spatial weights
        w_sum = self.spatial_weights.sum()

        # Cross-product of spatial weights and deviations
        # numerator: n * sum_i sum_j w_ij * z_i * z_j
        wz = self.spatial_weights @ z
        numerator = n * np.dot(z, wz)

        # Denominator: w_sum * sum_i z_i^2
        denominator = w_sum * np.sum(z**2)

        if denominator == 0:
            return {
                "morans_i": 0.0,
                "expected_i": -1.0 / (n - 1),
                "z_score": 0.0,
                "p_value": 1.0,
            }

        morans_i = numerator / denominator

        # Expected value under randomization
        expected_i = -1.0 / (n - 1)

        # Variance (under randomization assumption)
        # Simplified variance calculation
        s1 = 0.5 * (self.spatial_weights + self.spatial_weights.T).sum()
        row_sums = np.asarray(self.spatial_weights.sum(axis=1)).ravel()
        s2 = (row_sums ** 2).sum()
        s3d = (row_sums ** 3).sum()

        # Fourth moment
        z4 = np.sum(z**4) / n
        z2 = np.sum(z**2) / n
        b2 = z4 / (z2**2) if z2 > 0 else 1.0

        # Variance formula (Cliff & Ord, 1981)
        var_i = (
            n * ((n**2 - 3 * n + 3) * s1 - n * s2 + 3 * w_sum**2)
            - b2 * ((n**2 - n) * s1 - 2 * n * s2 + 6 * w_sum**2)
        ) / ((n - 1) * (n - 2) * (n - 3) * w_sum**2) - expected_i**2

        if var_i <= 0:
            z_score = 0.0
        else:
            z_score = (morans_i - expected_i) / np.sqrt(var_i)

        # Two-sided p-value
        p_value = 2 * (1 - norm.cdf(abs(z_score)))

        return {
            "morans_i": float(morans_i),
            "expected_i": float(expected_i),
            "variance": float(var_i),
            "z_score": float(z_score),
            "p_value": float(p_value),
        }

    def compute_all_morans_i(
        self,
        expression_matrix: csr_matrix,
        gene_names: List[str],
    ) -> pd.DataFrame:
        """Compute Moran's I for all genes.

        Args:
            expression_matrix: CSR matrix (n_cells, n_genes).
            gene_names: List of gene names.

        Returns:
            DataFrame with columns: gene, morans_i, z_score, p_value.
        """
        results = []
        for i, gene in tqdm(
            enumerate(gene_names),
            desc="Moran's I", unit="gene",
            total=len(gene_names), disable=len(gene_names) < 50,
        ):
            expr = expression_matrix[:, i].toarray().flatten()
            if expr.sum() == 0:
                continue
            stats = self.morans_i(expr)
            stats["gene"] = gene
            results.append(stats)

        df = pd.DataFrame(results)
        df = df.sort_values("morans_i", ascending=False)
        logger.info(
            f"Computed Moran's I for {len(df)} genes: "
            f"top={df['gene'].iloc[0]} (I={df['morans_i'].iloc[0]:.4f}), "
            f"bottom={df['gene'].iloc[-1]} (I={df['morans_i'].iloc[-1]:.4f})"
        )
        return df

    def getis_ord_gi(
        self,
        expression_vector: np.ndarray,
    ) -> np.ndarray:
        """Compute Getis-Ord Gi* statistic for each cell.

        Gi* measures local spatial clustering:
        Gi* > 0: hot spot (high values cluster)
        Gi* < 0: cold spot (low values cluster)

        Args:
            expression_vector: 1D array of expression values (n_cells,).

        Returns:
            Array of Gi* z-scores (n_cells,).
        """
        if self.spatial_weights is None:
            raise ValueError("Spatial weights not computed.")

        n = len(expression_vector)
        expr = expression_vector.astype(np.float64)
        x_mean = expr.mean()
        x_std = expr.std(ddof=1)

        if x_std == 0:
            return np.zeros(n)

        # Sum of expression in neighborhood (including self)
        # For Gi*, include self in the neighborhood
        w_self = self.spatial_weights.copy()
        w_self.setdiag(1)

        # Weighted sum
        local_sum = np.array(w_self @ expr).flatten()

        # Sum of weights per location (including self)
        wi_sum = np.array(w_self.sum(axis=1)).flatten()

        # Expected value
        expected = x_mean * wi_sum

        # Variance
        s2 = np.sum(expr**2) / n - x_mean**2
        var_gi = s2 * ((n * wi_sum - wi_sum**2) / (n - 1))

        var_gi = np.where(var_gi <= 0, 1e-10, var_gi)

        # Gi* z-score
        gi_star = (local_sum - expected) / np.sqrt(var_gi)

        return gi_star

    def spatially_variable_genes(
        self,
        expression_matrix: csr_matrix,
        gene_names: List[str],
        n_top: int = 50,
    ) -> pd.DataFrame:
        """Identify spatially variable genes.

        Ranks genes by Moran's I and identifies those with
        significant spatial autocorrelation.

        Args:
            expression_matrix: CSR matrix (n_cells, n_genes).
            gene_names: List of gene names.
            n_top: Number of top SVGs to return in summary.

        Returns:
            DataFrame of all genes with spatial statistics, sorted by
            significance then Moran's I.
        """
        morans_df = self.compute_all_morans_i(expression_matrix, gene_names)

        # Filter significant genes and sort
        svg_df = morans_df[morans_df["p_value"] < 0.05].copy()
        svg_df = svg_df.sort_values(["p_value", "morans_i"], ascending=[True, False])

        logger.info(
            f"Identified {len(svg_df)} spatially variable genes "
            f"(p < 0.05) out of {len(morans_df)} total"
        )
        return svg_df

    def regional_analysis(
        self,
        expression_matrix: csr_matrix,
        gene_names: List[str],
        region_labels: np.ndarray,
    ) -> pd.DataFrame:
        """Perform differential expression analysis between regions.

        Args:
            expression_matrix: CSR matrix (n_cells, n_genes).
            gene_names: List of gene names.
            region_labels: 1D array of region labels per cell (e.g., cell types).

        Returns:
            DataFrame with per-gene-per-region mean expression.
        """
        unique_regions = np.unique(region_labels)
        results = []

        for region in unique_regions:
            mask = region_labels == region
            if mask.sum() < 5:  # Skip regions with too few cells
                continue

            region_expr = expression_matrix[mask].toarray()
            mean_expr = region_expr.mean(axis=0)
            pct_expressed = (region_expr > 0).mean(axis=0)

            for i, gene in enumerate(gene_names):
                results.append({
                    "region": region,
                    "gene": gene,
                    "mean_expression": float(mean_expr[i]),
                    "pct_expressed": float(pct_expressed[i]),
                })

        df = pd.DataFrame(results)
        logger.info(
            f"Regional analysis: {len(unique_regions)} regions, "
            f"{len(gene_names)} genes"
        )
        return df
