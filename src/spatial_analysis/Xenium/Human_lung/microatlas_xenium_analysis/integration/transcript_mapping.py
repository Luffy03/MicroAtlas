"""
Transcript-to-cell assignment and expression matrix construction.

After cellpose-sam segmentation, this module assigns each transcript
to the cell whose mask it falls within, then builds a gene expression
matrix (cells x genes) for downstream analysis.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz

from .. import config

logger = logging.getLogger(__name__)


class TranscriptMapper:
    """Assign transcripts to segmented cells and build expression matrices.

    For each transcript at pixel coordinates (x, y), checks which
    cell mask it falls into and increments the gene count for that cell.
    """

    def __init__(self, gene_names: List[str]):
        """Initialize the mapper with the gene panel.

        Args:
            gene_names: Ordered list of gene names.
        """
        self.gene_names = gene_names
        self.gene_to_idx = {g: i for i, g in enumerate(gene_names)}
        self.n_genes = len(gene_names)

    def assign_transcripts(
        self,
        transcripts_df: pd.DataFrame,
        cell_mask: np.ndarray,
    ) -> pd.DataFrame:
        """Assign each transcript to a cell based on mask overlap.

        Args:
            transcripts_df: DataFrame with columns including
                'x_pixel', 'y_pixel', 'feature_name' (gene).
            cell_mask: uint16 label array (H, W).

        Returns:
            DataFrame with added 'assigned_cell' column.
            Unassigned transcripts get assigned_cell = 0.
        """
        df = transcripts_df.copy()

        # Vectorized lookup: for each transcript, get mask value at (y, x)
        xs = df["x_pixel"].values
        ys = df["y_pixel"].values

        # Clip coordinates to mask bounds
        h, w = cell_mask.shape
        xs = np.clip(xs, 0, w - 1)
        ys = np.clip(ys, 0, h - 1)

        # Lookup cell_id in mask for each transcript
        assigned = cell_mask[ys.astype(np.int32), xs.astype(np.int32)]
        df["assigned_cell"] = assigned

        n_assigned = (assigned > 0).sum()
        n_total = len(df)
        logger.info(
            f"Transcript assignment: {n_assigned}/{n_total} "
            f"({100 * n_assigned / max(n_total, 1):.1f}%) assigned to cells"
        )

        return df

    def build_expression_matrix(
        self,
        assigned_df: pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray, csr_matrix]:
        """Build a cells x genes expression matrix from assigned transcripts.

        Args:
            assigned_df: DataFrame from assign_transcripts() with
                'assigned_cell' and 'feature_name' columns.

        Returns:
            Tuple of (cell_ids, gene_names, expression_matrix):
                cell_ids: 1D array of cell label IDs present.
                gene_names: List of gene names.
                expression_matrix: csr_matrix of shape (n_cells, n_genes).
        """
        # Filter to assigned transcripts
        assigned = assigned_df[assigned_df["assigned_cell"] > 0].copy()

        if len(assigned) == 0:
            logger.warning("No transcripts assigned to any cells!")
            return np.array([], dtype=np.int32), self.gene_names, csr_matrix((0, self.n_genes))

        # Only keep transcripts with known genes
        known_genes = assigned["feature_name"].isin(self.gene_names)
        assigned = assigned[known_genes]

        if len(assigned) == 0:
            logger.warning("No transcripts matched known genes!")
            return np.array([], dtype=np.int32), self.gene_names, csr_matrix((0, self.n_genes))

        # Map cell labels to consecutive indices
        cell_labels = np.sort(assigned["assigned_cell"].unique())
        cell_to_idx = {lbl: i for i, lbl in enumerate(cell_labels)}

        # Map genes to indices
        gene_idx = assigned["feature_name"].map(self.gene_to_idx).values

        # Build sparse matrix
        n_cells = len(cell_labels)
        row = np.array([cell_to_idx[lbl] for lbl in assigned["assigned_cell"].values])
        col = gene_idx
        data = np.ones(len(assigned), dtype=np.float32)

        matrix = csr_matrix(
            (data, (row, col)),
            shape=(n_cells, self.n_genes),
            dtype=np.float32,
        )

        # Remove zero-expression cells (empty rows)
        nonzero_rows = np.array(matrix.getnnz(axis=1) > 0).flatten()
        matrix = matrix[nonzero_rows]
        cell_labels = cell_labels[nonzero_rows]

        logger.info(
            f"Built expression matrix: {matrix.shape} "
            f"({len(cell_labels)} cells x {self.n_genes} genes), "
            f"{matrix.nnz} non-zero entries"
        )

        return cell_labels, self.gene_names, matrix

    def build_expression_from_xenium_matrix(
        self,
        xenium_cell_ids: np.ndarray,
        xenium_matrix,
        cell_to_mask_map: Dict[str, int],
    ) -> Tuple[np.ndarray, List[str], csr_matrix]:
        """Build expression matrix by aggregating Xenium expression data
        per cellpose-sam cell.

        For each cellpose-sam cell, combine the expression of all Xenium
        cells whose centroids fall within it.

        Args:
            xenium_cell_ids: Array of Xenium cell ID strings.
            xenium_matrix: CSR matrix (n_xenium_cells, n_genes).
            cell_to_mask_map: Dict {xenium_cell_id: mask_label}.

        Returns:
            Tuple of (mask_labels, gene_names, expression_matrix).
        """
        # Build mapping: mask_label -> list of Xenium cell indices
        mask_to_cells: Dict[int, List[int]] = {}
        for i, cell_id in enumerate(xenium_cell_ids):
            mask_label = cell_to_mask_map.get(cell_id, 0)
            if mask_label > 0:
                if mask_label not in mask_to_cells:
                    mask_to_cells[mask_label] = []
                mask_to_cells[mask_label].append(i)

        # Aggregate expression per mask cell
        mask_labels = sorted(mask_to_cells.keys())
        n_cells = len(mask_labels)
        aggregated = np.zeros((n_cells, self.n_genes), dtype=np.float32)

        for i, mask_label in enumerate(mask_labels):
            xenium_indices = mask_to_cells[mask_label]
            if len(xenium_indices) == 1:
                aggregated[i] = xenium_matrix[xenium_indices[0]].toarray().flatten()
            else:
                # Sum expression from all Xenium cells in this mask
                aggregated[i] = (
                    xenium_matrix[xenium_indices].sum(axis=0).A1
                )

        matrix = csr_matrix(aggregated)

        logger.info(
            f"Aggregated Xenium expression by cellpose-sam cells: "
            f"{matrix.shape}, {matrix.nnz} non-zero"
        )

        return np.array(mask_labels), self.gene_names, matrix


def build_expression_matrix(
    assigned_transcripts: pd.DataFrame,
    gene_names: List[str],
) -> Tuple[np.ndarray, List[str], csr_matrix]:
    """Standalone function to build expression matrix.

    Args:
        assigned_transcripts: DataFrame with 'assigned_cell' and 'feature_name'.
        gene_names: List of gene names.

    Returns:
        Tuple of (cell_ids, gene_names, expression_matrix).
    """
    mapper = TranscriptMapper(gene_names)
    return mapper.build_expression_matrix(assigned_transcripts)


def save_expression_h5(
    cell_ids: np.ndarray,
    gene_names: List[str],
    matrix: csr_matrix,
    output_path: Path,
) -> None:
    """Save expression matrix in Xenium-compatible H5 format.

    Args:
        cell_ids: Array of cell label IDs.
        gene_names: List of gene names.
        matrix: CSR matrix (n_cells, n_genes).
        output_path: Path to save the H5 file.
    """
    from scipy.sparse import csc_matrix

    # Convert to CSC for storage (matching Xenium format)
    csc = matrix.T.tocsc()

    with h5py.File(str(output_path), "w") as f:
        grp = f.create_group("matrix")
        grp.create_dataset("barcodes", data=[f"cell_{i}".encode() for i in cell_ids])
        grp.create_dataset("data", data=csc.data)
        grp.create_dataset("indices", data=csc.indices)
        grp.create_dataset("indptr", data=csc.indptr)
        grp.create_dataset("shape", data=csc.shape)

        feat_grp = grp.create_group("features")
        feat_grp.create_dataset("id", data=[f"gene_{i}".encode() for i in range(len(gene_names))])
        feat_grp.create_dataset("name", data=[g.encode() for g in gene_names])
        feat_grp.create_dataset("feature_type", data=[b"Gene Expression"] * len(gene_names))
        feat_grp.create_dataset("genome", data=[b"unknown"] * len(gene_names))

    logger.info(f"Saved expression matrix to {output_path}")


def save_expression_npz(
    matrix: csr_matrix,
    cell_ids: np.ndarray,
    gene_names: List[str],
    output_prefix: Path,
) -> None:
    """Save expression matrix in NPZ format for fast loading.

    Args:
        matrix: CSR matrix.
        cell_ids: Array of cell label IDs.
        gene_names: List of gene names.
        output_prefix: Prefix for output files.
    """
    save_npz(str(output_prefix) + "_matrix.npz", matrix)
    np.save(str(output_prefix) + "_cell_ids.npy", cell_ids)
    np.save(str(output_prefix) + "_gene_names.npy", np.array(gene_names, dtype=object))
    logger.info(f"Saved expression matrix to {output_prefix}_*.npz")
