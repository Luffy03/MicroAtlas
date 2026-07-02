"""
Cell type identification module using known lung cell markers.

Assigns cell types to segmented cells based on gene expression
signatures of known lung cell types.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import StandardScaler

from .. import config

logger = logging.getLogger(__name__)


class CellTyper:
    """Assign cell types to cells based on marker gene expression.

    Uses a panel of known lung cell type markers to compute
    per-cell type scores and assign the most likely cell type.
    """

    def __init__(self, cell_markers: Optional[Dict[str, List[str]]] = None):
        """Initialize with marker gene definitions.

        Args:
            cell_markers: Dict mapping cell_type_name -> list of marker gene names.
                          Defaults to config.LUNG_CELL_MARKERS.
        """
        self.markers = cell_markers or config.LUNG_CELL_MARKERS
        self.cell_types = list(self.markers.keys())
        self._validate_markers()

    def _validate_markers(self):
        """Log marker gene information."""
        for ct, genes in self.markers.items():
            logger.debug(f"Cell type '{ct}': {len(genes)} markers: {genes}")

    def score_cell_types(
        self,
        expression_matrix: csr_matrix,
        gene_names: List[str],
    ) -> pd.DataFrame:
        """Score each cell for each cell type based on marker expression.

        For each cell type, computes the mean expression of its marker
        genes, normalized as Z-scores across all cell types.

        Args:
            expression_matrix: CSR matrix (n_cells, n_genes).
            gene_names: List of gene names matching matrix columns.

        Returns:
            DataFrame with columns: cell_id, cell_type, and score columns
            for each cell type.
        """
        gene_to_idx = {g: i for i, g in enumerate(gene_names)}
        n_cells = expression_matrix.shape[0]

        # Compute mean expression per cell type for each cell
        scores = np.zeros((n_cells, len(self.cell_types)), dtype=np.float32)

        for j, (ct_name, marker_genes) in enumerate(self.markers.items()):
            # Find marker gene indices in the expression matrix
            valid_indices = []
            for mg in marker_genes:
                if mg in gene_to_idx:
                    valid_indices.append(gene_to_idx[mg])
                else:
                    logger.debug(f"Marker gene '{mg}' not found in expression matrix")

            if not valid_indices:
                logger.warning(f"No valid markers found for cell type '{ct_name}'")
                continue

            # Extract marker expression sub-matrix (n_cells, n_markers)
            marker_expr = expression_matrix[:, valid_indices].toarray()

            # Compute mean expression per cell (if multiple markers)
            if marker_expr.shape[1] > 0:
                scores[:, j] = marker_expr.mean(axis=1)

        # Z-score normalization across cell types for each cell
        row_means = scores.mean(axis=1, keepdims=True)
        row_stds = scores.std(axis=1, keepdims=True)
        row_stds = np.where(row_stds == 0, 1.0, row_stds)  # avoid division by zero
        scores_z = (scores - row_means) / row_stds

        # Assign each cell to the cell type with the highest Z-score
        assigned_indices = np.argmax(scores_z, axis=1)
        assigned_types = [self.cell_types[idx] for idx in assigned_indices]

        # Build result DataFrame
        result = pd.DataFrame({
            "cell_id": np.arange(n_cells),
            "cell_type": assigned_types,
            "max_zscore": scores_z.max(axis=1),
        })

        # Add individual scores as columns
        for j, ct_name in enumerate(self.cell_types):
            result[f"score_{ct_name}"] = scores[:, j]
            result[f"zscore_{ct_name}"] = scores_z[:, j]

        # Summary statistics
        type_counts = result["cell_type"].value_counts()
        logger.info("Cell type assignment summary:")
        for ct, count in type_counts.items():
            logger.info(f"  {ct}: {count} cells ({100*count/n_cells:.1f}%)")

        return result

    def cluster_and_annotate(
        self,
        expression_matrix: csr_matrix,
        gene_names: List[str],
        n_clusters: int = 15,
        random_state: int = 42,
    ) -> pd.DataFrame:
        """Cluster cells by gene expression and annotate clusters
        using marker gene enrichment.

        Uses Leiden clustering via UMAP preprocessing.

        Args:
            expression_matrix: CSR matrix (n_cells, n_genes).
            gene_names: List of gene names.
            n_clusters: Number of clusters (approximate).
            random_state: Random seed.

        Returns:
            DataFrame with cell_id, cluster, and marker-based cell type.
        """
        try:
            import scanpy as sc
        except ImportError:
            logger.error(
                "scanpy is required for clustering. "
                "Install with: pip install scanpy"
            )
            raise

        # Create AnnData object
        adata = sc.AnnData(X=expression_matrix)
        adata.var_names = gene_names

        # Preprocessing
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=200)
        sc.pp.pca(adata, n_comps=50)
        sc.pp.neighbors(adata)
        sc.tl.leiden(adata, resolution=1.0)

        # Get cluster labels
        clusters = adata.obs["leiden"].astype(int).values

        # For each cluster, find top marker genes
        sc.tl.rank_genes_groups(adata, "leiden", method="wilcoxon")

        # For each cluster, find the best matching cell type
        cluster_types = []
        for cluster_id in range(clusters.max() + 1):
            # Get top genes for this cluster
            top_genes = sc.get.rank_genes_groups_df(adata, group=str(cluster_id))
            top_genes = top_genes[top_genes["pvals_adj"] < 0.05].head(10)

            # Score each cell type based on how many marker genes are in top genes
            top_gene_set = set(top_genes["names"].values)
            best_type = "Unknown"
            best_score = 0

            for ct, marker_genes in self.markers.items():
                overlap = len(set(marker_genes) & top_gene_set)
                if overlap > best_score:
                    best_score = overlap
                    best_type = ct

            cluster_types.append(best_type)

        # Build result
        result = pd.DataFrame({
            "cell_id": np.arange(expression_matrix.shape[0]),
            "cluster": clusters,
            "cell_type": [cluster_types[c] for c in clusters],
        })

        logger.info(
            f"Clustering and annotation complete: {len(np.unique(clusters))} clusters"
        )
        return result
