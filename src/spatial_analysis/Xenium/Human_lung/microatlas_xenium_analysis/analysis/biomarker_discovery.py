"""
Biomarker discovery module for spatial transcriptomics.

Methods for differential expression analysis, correlation of gene
expression with morphological features, spatial pattern-based
biomarker discovery, and gene signature scoring.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.stats import mannwhitneyu, pearsonr, spearmanr

from .. import config
from tqdm import tqdm

logger = logging.getLogger(__name__)


class BiomarkerDiscovery:
    """Discover potential biomarkers from spatial transcriptomics data.

    Provides:
    - Differential expression analysis (tumor vs normal regions)
    - Correlation of gene expression with morphological features
    - Spatially-informed biomarker identification
    - Gene signature scoring for known pathways
    """

    def __init__(self):
        self.de_results = None

    def differential_expression(
        self,
        expression_matrix: csr_matrix,
        gene_names: List[str],
        group_labels: np.ndarray,
        group_a: str,
        group_b: str,
        method: str = "wilcoxon",
    ) -> pd.DataFrame:
        """Perform differential expression analysis between two groups.

        Args:
            expression_matrix: CSR matrix (n_cells, n_genes).
            gene_names: List of gene names.
            group_labels: 1D array of group labels per cell.
            group_a: Label for the "test" group (e.g., "tumor").
            group_b: Label for the "reference" group (e.g., "normal").
            method: Statistical test ("wilcoxon" for Mann-Whitney U, or "ttest").

        Returns:
            DataFrame with columns: gene, log2fc, p_value, p_value_adj,
            mean_a, mean_b, pct_a, pct_b.
        """
        mask_a = group_labels == group_a
        mask_b = group_labels == group_b

        n_a = mask_a.sum()
        n_b = mask_b.sum()

        if n_a < 3 or n_b < 3:
            raise ValueError(
                f"Need at least 3 cells per group: group_a={n_a}, group_b={n_b}"
            )

        logger.info(
            f"DE analysis: {group_a} ({n_a} cells) vs {group_b} ({n_b} cells)"
        )

        expr_a = expression_matrix[mask_a].toarray()
        expr_b = expression_matrix[mask_b].toarray()

        results = []
        for i, gene in tqdm(
            enumerate(gene_names),
            desc="DE analysis", unit="gene",
            total=len(gene_names), disable=len(gene_names) < 50,
        ):
            vals_a = expr_a[:, i]
            vals_b = expr_b[:, i]

            mean_a = float(vals_a.mean())
            mean_b = float(vals_b.mean())
            pct_a = float((vals_a > 0).mean())
            pct_b = float((vals_b > 0).mean())

            # Log2 fold change (with pseudocount)
            log2fc = np.log2((mean_a + 1e-6) / (mean_b + 1e-6))

            # Statistical test
            if method == "wilcoxon":
                stat, p_val = mannwhitneyu(
                    vals_a, vals_b, alternative="two-sided"
                )
            else:
                from scipy.stats import ttest_ind
                stat, p_val = ttest_ind(vals_a, vals_b)

            results.append({
                "gene": gene,
                "log2fc": log2fc,
                "p_value": p_val,
                "mean_a": mean_a,
                "mean_b": mean_b,
                "pct_a": pct_a,
                "pct_b": pct_b,
                "group_a": group_a,
                "group_b": group_b,
            })

        df = pd.DataFrame(results)

        # Multiple testing correction (Benjamini-Hochberg)
        df = df.sort_values("p_value")
        n_tests = len(df)
        df["p_value_adj"] = np.minimum(
            df["p_value"].values * n_tests
            / (np.arange(n_tests) + 1),
            1.0,
        )
        df = df.sort_values("p_value_adj")

        # Identify significant biomarkers
        n_sig = (df["p_value_adj"] < config.DE_PVAL_CUTOFF).sum()
        n_up = ((df["p_value_adj"] < config.DE_PVAL_CUTOFF)
                & (df["log2fc"] > config.DE_LOGFC_CUTOFF)).sum()
        n_down = ((df["p_value_adj"] < config.DE_PVAL_CUTOFF)
                  & (df["log2fc"] < -config.DE_LOGFC_CUTOFF)).sum()

        logger.info(
            f"DE results: {n_sig} significant genes "
            f"({n_up} up, {n_down} down in {group_a})"
        )

        self.de_results = df
        return df

    def correlate_with_morphology(
        self,
        expression_matrix: csr_matrix,
        gene_names: List[str],
        morphology_features: Dict[int, Dict[str, float]],
        mask_labels: np.ndarray,
        feature_name: str = "area",
    ) -> pd.DataFrame:
        """Correlate gene expression with cell morphological features.

        Args:
            expression_matrix: CSR matrix (n_cells, n_genes).
            gene_names: List of gene names.
            morphology_features: Dict of {mask_label: {feat_name: value}}.
            mask_labels: Array of mask labels matching expression_matrix rows.
            feature_name: Morphology feature to correlate with.

        Returns:
            DataFrame with columns: gene, correlation, p_value.
        """
        # Build feature vector aligned with expression matrix
        feature_values = []
        valid_indices = []

        for i, label in enumerate(mask_labels):
            if label in morphology_features and feature_name in morphology_features[label]:
                feature_values.append(morphology_features[label][feature_name])
                valid_indices.append(i)

        if len(valid_indices) < 10:
            logger.warning(
                f"Too few cells with feature '{feature_name}': {len(valid_indices)}"
            )
            return pd.DataFrame()

        feature_values = np.array(feature_values)
        expr_subset = expression_matrix[valid_indices]

        results = []
        for i, gene in enumerate(gene_names):
            expr = expr_subset[:, i].toarray().flatten()

            # Skip zero-expression genes
            if expr.sum() == 0:
                continue

            try:
                corr, p_val = spearmanr(expr, feature_values)
                if np.isnan(corr):
                    continue
                results.append({
                    "gene": gene,
                    "correlation": float(corr),
                    "p_value": float(p_val),
                    "feature": feature_name,
                })
            except Exception:
                continue

        df = pd.DataFrame(results)
        df = df.sort_values("p_value")

        if len(df) > 0:
            logger.info(
                f"Correlation with '{feature_name}': "
                f"top positive={df['gene'].iloc[0]} (r={df['correlation'].iloc[0]:.3f}), "
                f"top negative={df['gene'].iloc[-1]} (r={df['correlation'].iloc[-1]:.3f})"
            )

        return df

    def surrogate_biomarkers(
        self,
        expression_matrix: csr_matrix,
        gene_names: List[str],
        morans_i_df: pd.DataFrame,
        de_results_df: pd.DataFrame,
        n_top: int = 30,
    ) -> pd.DataFrame:
        """Identify surrogate biomarkers that combine spatial clustering
        and differential expression signals.

        Args:
            expression_matrix: CSR matrix (n_cells, n_genes).
            gene_names: List of gene names.
            morans_i_df: DataFrame from SpatialAnalyzer.compute_all_morans_i().
            de_results_df: DataFrame from differential_expression().
            n_top: Number of top candidates to return.

        Returns:
            DataFrame of top biomarker candidates, ranked by combined score.
        """
        if de_results_df is None or len(de_results_df) == 0:
            logger.warning("No DE results available for biomarker ranking")
            return pd.DataFrame()

        # Merge spatial and DE results
        combined = de_results_df.merge(
            morans_i_df[["gene", "morans_i", "z_score"]],
            on="gene",
            how="left",
        )
        combined["morans_i"] = combined["morans_i"].fillna(0)

        # Composite score: combine DE significance and spatial clustering
        # -log10(p_value) * abs(log2fc) * (1 + abs(morans_i))
        combined["significance_score"] = (
            -np.log10(combined["p_value_adj"] + 1e-100)
            * np.abs(combined["log2fc"])
            * (1 + np.abs(combined["morans_i"]))
        )

        # Adjust for direction: positive for up in group_a, negative for down
        combined["biomarker_score"] = (
            combined["significance_score"] * np.sign(combined["log2fc"])
        )

        combined = combined.sort_values(
            "significance_score", ascending=False
        ).head(n_top)

        logger.info(f"Identified top {len(combined)} surrogate biomarkers")
        return combined

    def gene_signature_scoring(
        self,
        expression_matrix: csr_matrix,
        gene_names: List[str],
        signatures: Dict[str, List[str]],
    ) -> pd.DataFrame:
        """Score each cell for known gene signatures/pathways.

        Args:
            expression_matrix: CSR matrix (n_cells, n_genes).
            gene_names: List of gene names.
            signatures: Dict mapping signature_name -> [gene_list].

        Returns:
            DataFrame with per-cell signature scores.
        """
        gene_to_idx = {g: i for i, g in enumerate(gene_names)}
        n_cells = expression_matrix.shape[0]
        sig_names = list(signatures.keys())

        scores = np.zeros((n_cells, len(sig_names)), dtype=np.float32)

        for j, (sig_name, sig_genes) in enumerate(signatures.items()):
            valid_idx = [gene_to_idx[g] for g in sig_genes if g in gene_to_idx]
            if len(valid_idx) == 0:
                logger.warning(f"No valid genes for signature '{sig_name}'")
                continue

            sig_expr = expression_matrix[:, valid_idx].toarray()
            # Mean expression per cell
            scores[:, j] = sig_expr.mean(axis=1)

        # Z-score per signature
        result = pd.DataFrame(scores, columns=sig_names)
        for col in result.columns:
            mean_val = result[col].mean()
            std_val = result[col].std()
            if std_val > 0:
                result[f"{col}_zscore"] = (result[col] - mean_val) / std_val
            else:
                result[f"{col}_zscore"] = 0.0

        logger.info(
            f"Computed signature scores for {len(signatures)} signatures "
            f"across {n_cells} cells"
        )
        return result
