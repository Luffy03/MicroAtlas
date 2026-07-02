"""
biomarker/unsupervised.py
=========================
Unsupervised clustering and dimensionality reduction for biomarker discovery.

Methods:
  - UMAP: nonlinear embedding for visualization + distance-based analysis
  - HDBSCAN: density-based clustering (no preset k)
  - Evaluation: NMI against known MoA labels, silhouette score

Usage:
  python biomarker_discovery/biomarker/unsupervised.py --models cellpose4
  python biomarker_discovery/biomarker/unsupervised.py --models all
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


def get_field_feature_columns(df):
    """Get numeric feature columns from field-level data."""
    meta_cols = {'plate', 'field', 'compound', 'concentration', 'moa',
                 'is_dmso', 'well', 'n_cells'}
    return [c for c in df.columns
            if c not in meta_cols and df[c].dtype in (np.float64, np.int64, float, int)]


def run_pca(X, n_components=50):
    """Run PCA dimensionality reduction (used for high-dim feature spaces)."""
    from sklearn.decomposition import PCA
    print(f"  PCA: n_components={n_components}")
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X)
    var_ratio = pca.explained_variance_ratio_.sum()
    print(f"  Explained variance ratio: {var_ratio:.4f}")
    return X_pca


def run_umap(X, n_components=None, n_neighbors=None, min_dist=None, metric=None, seed=None, umap_init=None, exact_knn=False):
    """Run UMAP dimensionality reduction.

    Args:
        X: (n_samples, n_features) normalized feature matrix
        n_components: output dimensions (default from config)
        n_neighbors: UMAP n_neighbors (default from config)
        min_dist: UMAP min_dist (default from config)
        metric: distance metric (default from config)
        seed: random_state for UMAP (default from config.UMAP_SEED)
        umap_init: UMAP initialization ('spectral', 'random', 'pca', etc.).
                   Default (None) uses UMAP's default ('spectral').
        exact_knn: if True, use sklearn brute-force kNN (cross-platform deterministic)

    Returns:
        embedding: (n_samples, n_components) array
    """
    import umap

    n_components = n_components or config.UMAP_N_COMPONENTS
    n_neighbors = n_neighbors or config.UMAP_N_NEIGHBORS
    min_dist = min_dist or config.UMAP_MIN_DIST
    metric = metric or config.UMAP_METRIC
    if seed is None:
        seed = config.UMAP_SEED

    init_name = umap_init if umap_init else "spectral (default)"
    print(f"  UMAP: n_components={n_components}, n_neighbors={n_neighbors}, "
          f"min_dist={min_dist}, metric={metric}, init={init_name}")

    umap_kwargs = dict(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=seed,
    )
    if umap_init is not None:
        umap_kwargs["init"] = umap_init

    if exact_knn:
        from sklearn.neighbors import NearestNeighbors
        print(f"  Exact kNN (sklearn brute-force)")
        nn = NearestNeighbors(n_neighbors=n_neighbors, metric=metric, algorithm="brute")
        nn.fit(X)
        knn_dists, knn_indices = nn.kneighbors(return_distance=True)
        umap_kwargs["precomputed_knn"] = (knn_indices, knn_dists, None)

    reducer = umap.UMAP(**umap_kwargs)
    embedding = reducer.fit_transform(X)
    return embedding


def run_hdbscan(X, min_cluster_size=None, min_samples=None):
    """Run HDBSCAN density-based clustering.

    Args:
        X: (n_samples, n_features) feature matrix
        min_cluster_size: minimum cluster size (default from config)
        min_samples: minimum samples (default from config)

    Returns:
        labels: cluster assignments (-1 = noise)
        probabilities: membership probabilities
    """
    import hdbscan

    min_cluster_size = min_cluster_size or config.HDBSCAN_MIN_CLUSTER_SIZE
    min_samples = min_samples or config.HDBSCAN_MIN_SAMPLES

    print(f"  HDBSCAN: min_cluster_size={min_cluster_size}, "
          f"min_samples={min_samples}")

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method='eom',
    )
    labels = clusterer.fit_predict(X)
    probabilities = clusterer.probabilities_

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    print(f"  Clusters: {n_clusters}, Noise points: {n_noise}")

    return labels, probabilities



def _hungarian_accuracy(pred_labels, true_labels):
    """Compute Hungarian matching accuracy between predicted clusters and true labels.

    Args:
        pred_labels: array of predicted cluster assignments (noise should be pre-filtered)
        true_labels: array of ground truth labels

    Returns:
        accuracy: optimal matching accuracy via Hungarian algorithm
    """
    from scipy.optimize import linear_sum_assignment

    cluster_ids = np.unique(pred_labels)
    true_ids = np.unique(true_labels)
    n_clusters = len(cluster_ids)
    n_classes = len(true_ids)

    # Map labels to 0-indexed
    pred_mapped = np.searchsorted(cluster_ids, pred_labels)
    true_mapped = np.searchsorted(true_ids, true_labels)

    # Cost matrix: cost_matrix[i, j] = -count(cluster=i AND true=j)
    cost = np.zeros((n_clusters, n_classes), dtype=int)
    for p, t in zip(pred_mapped, true_mapped):
        cost[p, t] += 1
    cost = -cost  # negate for minimization (linear_sum_assignment minimizes)

    row_ind, col_ind = linear_sum_assignment(cost)
    hungarian_correct = -cost[row_ind, col_ind].sum()
    accuracy = hungarian_correct / len(pred_labels)
    return accuracy


def bootstrap_hungarian_ci(pred_labels, true_labels, n_bootstrap=1000, ci_percent=95):
    """Bootstrap confidence interval for Hungarian matching accuracy.

    Uses stratified resampling by MoA class (preserves class proportions
    in every iteration) to produce symmetric CIs even when rare classes
    are present.

    Args:
        pred_labels: array of predicted cluster assignments
        true_labels: array of ground truth labels
        n_bootstrap: number of bootstrap iterations (default 1000)
        ci_percent: confidence level (default 95)

    Returns:
        ci: dict with 'Accuracy_CI_low' and 'Accuracy_CI_high'
        boot_accs: array of bootstrap accuracy values
    """
    from scipy.stats import norm
    rng = np.random.default_rng(42)
    n = len(pred_labels)
    boot_accs = np.zeros(n_bootstrap, dtype=float)

    # Group indices by MoA class for stratified resampling
    unique_classes = np.unique(true_labels)
    class_indices = {c: np.where(true_labels == c)[0] for c in unique_classes}
    class_sizes = {c: len(idx) for c, idx in class_indices.items()}
    print(f"    [DEBUG] bootstrap_hungarian_ci: stratified={len(unique_classes)} classes, normal-approx CI")

    for i in range(n_bootstrap):
        idx = np.concatenate([
            rng.choice(class_indices[c], size=class_sizes[c], replace=True)
            for c in unique_classes
        ])
        boot_accs[i] = _hungarian_accuracy(pred_labels[idx], true_labels[idx])

    # Normal-approximation CI using bootstrap SE
    pt_estimate = _hungarian_accuracy(pred_labels, true_labels)
    se = np.std(boot_accs, ddof=1)
    z = norm.ppf(0.5 + ci_percent / 200)
    lo = max(0.0, pt_estimate - z * se)
    hi = min(1.0, pt_estimate + z * se)

    ci = {
        'Accuracy_CI_low': lo,
        'Accuracy_CI_high': hi,
    }
    return ci, boot_accs


def evaluate_clustering(labels, true_labels):
    """Evaluate clustering quality against known MoA labels.

    Metrics:
      - NMI (Normalized Mutual Information)
      - ARI (Adjusted Rand Index)
      - Accuracy (Hungarian matching: optimal cluster→MoA mapping)
      - Purity (majority-vote per cluster)
      - Silhouette score (if enough clusters)

    Only uses compounds with known MoA labels.
    """
    from sklearn.metrics import (
        normalized_mutual_info_score,
        adjusted_rand_score,
    )

    # Filter to non-noise, non-unknown compounds
    valid = (labels != -1) & (true_labels != 'unknown')
    if valid.sum() < 3:
        print("  Too few valid points for evaluation.")
        return {}

    pred = labels[valid]
    true = true_labels[valid]

    nmi = normalized_mutual_info_score(true, pred)
    ari = adjusted_rand_score(true, pred)

    print(f"  NMI: {nmi:.3f}")
    print(f"  ARI: {ari:.3f}")

    results = {'NMI': nmi, 'ARI': ari}

    # --- Hungarian matching accuracy (reuse standalone function) ---
    accuracy = _hungarian_accuracy(pred, true)
    results['Accuracy'] = accuracy
    print(f"  Accuracy (Hungarian): {accuracy:.3f}")

    # --- Purity ---
    cluster_ids = np.unique(pred)
    true_ids = np.unique(true)
    n_clusters = len(cluster_ids)
    n_classes = len(true_ids)
    pred_mapped = np.searchsorted(cluster_ids, pred)
    true_mapped = np.searchsorted(true_ids, true)

    purity_correct = 0
    for cid in range(n_clusters):
        mask = pred_mapped == cid
        if mask.sum() == 0:
            continue
        # majority class in this cluster
        counts = np.bincount(true_mapped[mask], minlength=n_classes)
        purity_correct += counts.max()
    purity = purity_correct / len(pred)
    results['Purity'] = purity
    print(f"  Purity: {purity:.3f}")

    return results


def find_moa_enriched_clusters(labels, moa_labels):
    """For each cluster, find the dominant MoA class and enrichment p-value.

    Uses Fisher's exact test for enrichment.

    Returns:
        DataFrame: cluster_id, dominant_moa, n_members, enrichment_p
    """
    from scipy.stats import fisher_exact

    valid = labels != -1
    cluster_ids = sorted(set(labels[valid]))
    moa_classes = sorted(set(moa_labels[valid]))

    records = []
    for cid in cluster_ids:
        in_cluster = labels == cid
        n_members = in_cluster.sum()

        best_moa = None
        best_p = 1.0
        best_count = 0

        for moa in moa_classes:
            if moa == 'unknown':
                continue
            in_moa = moa_labels == moa

            # 2x2 contingency table
            a = (in_cluster & in_moa).sum()  # in cluster AND in moa
            b = (in_cluster & ~in_moa).sum()  # in cluster AND not in moa
            c = (~in_cluster & in_moa).sum()  # not in cluster AND in moa
            d = (~in_cluster & ~in_moa).sum()  # not in cluster AND not in moa

            if a == 0:
                continue

            _, p = fisher_exact([[a, b], [c, d]])

            if a > best_count or (a == best_count and p < best_p):
                best_moa = moa
                best_p = p
                best_count = a

        records.append({
            'cluster_id': cid,
            'n_members': n_members,
            'dominant_moa': best_moa,
            'dominant_count': best_count,
            'enrichment_p': best_p,
        })

    return pd.DataFrame(records)





def run_field_level_clustering(model_name="cellpose4",
                                n_umap=None, n_neighbors=None,
                                umap_init=None, exact_knn=False):
    """Field-level UMAP \u2192 treatment-level HDBSCAN.

    Procedure: field PCA \u2192 field UMAP \u2192 aggregate to treatment (mean) \u2192 HDBSCAN.

    n_umap: UMAP output dimensions for field-level embedding (default: config.FIELD_UMAP_N_COMPONENTS).
    n_neighbors: UMAP n_neighbors for field-level embedding (default: config.FIELD_UMAP_N_NEIGHBORS).

    Uses field_profiles_normalized.csv (all features, field-level).
    """
    tag = " [Treat HDBSCAN]"
    print("=" * 60)
    print(f"Field-Level Clustering [{model_name}]{tag}")
    print("=" * 60)

    # 1. Load field profiles (keep metadata for later grouping)
    path = config.aggregation_dir(model_name) / "field_profiles_normalized.csv"
    if not path.exists():
        raise FileNotFoundError(f"Field profiles not found: {path}")
    df = pd.read_csv(path)
    feat_cols = get_field_feature_columns(df)
    print(f"Loaded field profiles: {len(df)} fields, {len(feat_cols)} features")

    # Store metadata for grouping
    group_keys = ['compound', 'concentration', 'moa', 'is_dmso']

    # 2. Prepare data (field-level)
    X = df[feat_cols].values.astype(np.float64)
    X = np.nan_to_num(X, nan=0.0)

    # Exclude DMSO fields
    non_dmso = df['is_dmso'] != True
    X_nd = X[non_dmso]
    field_meta = df[non_dmso][group_keys].reset_index(drop=True)
    print(f"  Non-DMSO fields for clustering: {len(X_nd)}")

    # 3. PCA
    print(f"\n--- PCA ---")
    X_nd = run_pca(X_nd, n_components=50)

    # 4. UMAP (on all fields)
    # Use higher-dimensional UMAP to preserve field-level structure for treatment clustering
    if n_umap is None:
        n_umap = config.FIELD_UMAP_N_COMPONENTS
    if n_neighbors is None:
        n_neighbors = config.FIELD_UMAP_N_NEIGHBORS
    print(f"\n--- UMAP (n_components={n_umap}) ---")
    embedding = run_umap(X_nd, n_components=n_umap, n_neighbors=n_neighbors, umap_init=umap_init, exact_knn=exact_knn)

    # ---- Path B: treatment-level clustering (on field UMAP mean) ----
    # Aggregate field UMAP embedding to treatment level
    print(f"\n--- Aggregating field UMAP to treatments ---")
    embed_dict = {
        'compound': field_meta['compound'].values,
        'concentration': field_meta['concentration'].values,
    }
    for i in range(n_umap):
        embed_dict[f'UMAP_{i+1}'] = embedding[:, i]
    embed_df = pd.DataFrame(embed_dict)
    treat_embed = embed_df.groupby(['compound', 'concentration'], sort=False).mean().values

    # Treatment-level metadata
    treat_meta = field_meta.groupby(group_keys, sort=False).first().reset_index()
    print(f"  Aggregated to {len(treat_meta)} treatments")

    # Clustering on treatment-level UMAP
    print(f"\n--- HDBSCAN (treatment-level) ---")
    cluster_labels, probabilities = run_hdbscan(treat_embed)

    # Evaluate
    print(f"\n--- Clustering Evaluation ---")
    moa_labels = treat_meta['moa'].values
    metrics = evaluate_clustering(cluster_labels, moa_labels)

    # Silhouette on treatment-level UMAP
    from sklearn.metrics import silhouette_score
    valid = cluster_labels != -1
    if valid.sum() > 2 and len(set(cluster_labels[valid])) >= 2:
        sil = silhouette_score(treat_embed[valid], cluster_labels[valid])
        metrics['Silhouette'] = sil
        print(f"  Silhouette (treatment-level): {sil:.3f}")

    # MoA enrichment
    print(f"\n--- MoA Enrichment ---")
    enrichment_df = find_moa_enriched_clusters(cluster_labels, moa_labels)
    print(enrichment_df.to_string(index=False))

    # Build treatment-level cluster results
    treatment_labels = treat_meta.copy()
    treatment_labels['cluster'] = cluster_labels
    treatment_labels['probability'] = probabilities

    # Save
    output_dir = config.biomarker_dir(model_name) / "unsupervised"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Field-level UMAP embedding (all n_umap components)
    field_dict = {
        'compound': field_meta['compound'].values,
        'concentration': field_meta['concentration'].values,
        'moa': field_meta['moa'].values,
    }
    for i in range(n_umap):
        field_dict[f'UMAP_{i+1}'] = embedding[:, i]
    field_umap_df = pd.DataFrame(field_dict)
    field_umap_df.to_csv(output_dir / "field_umap_embedding.csv", index=False)
    print(f"\n  Saved: {output_dir / 'field_umap_embedding.csv'}")

    cluster_df = treatment_labels
    cluster_df.to_csv(output_dir / "cluster_assignments.csv", index=False)
    print(f"  Saved: {output_dir / 'cluster_assignments.csv'}")

    enrichment_df.to_csv(output_dir / "cluster_moa_enrichment.csv", index=False)
    print(f"  Saved: {output_dir / 'cluster_moa_enrichment.csv'}")

    pd.DataFrame([metrics]).to_csv(output_dir / "clustering_metrics.csv", index=False)
    print(f"  Saved: {output_dir / 'clustering_metrics.csv'}")

    return field_umap_df, cluster_df, enrichment_df, metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, nargs="+", default=None,
                        help="Models. Use 'all' for all 5 models.")
    parser.add_argument("--n_umap", type=int, default=None,
                        help="UMAP dimensions for field-level embedding (default: config.FIELD_UMAP_N_COMPONENTS).")
    parser.add_argument("--n_neighbors", type=int, default=None,
                        help="UMAP n_neighbors for field-level embedding (default: config.FIELD_UMAP_N_NEIGHBORS).")
    parser.add_argument("--umap_init", type=str, default="pca",
                        choices=["spectral", "random", "pca"],
                        help="UMAP initialization (default: pca). 'pca' gives high accuracy with stable ranking. "
                             "Use 'random' for guaranteed cross-platform ranking consistency.")
    parser.add_argument("--exact_knn", action="store_true",
                        help="Use sklearn brute-force kNN (deterministic across platforms) instead of pynndescent.")
    args = parser.parse_args()

    if args.models is None:
        models_to_run = ["cellpose4"]
    elif "all" in args.models:
        models_to_run = config.MODELS
    else:
        models_to_run = args.models

    for m in models_to_run:
        run_field_level_clustering(model_name=m,
                                   n_umap=args.n_umap, n_neighbors=args.n_neighbors,
                                   umap_init=args.umap_init, exact_knn=args.exact_knn)
