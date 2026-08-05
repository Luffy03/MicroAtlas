"""
biomarker/unsupervised.py
=========================
Unsupervised clustering and dimensionality reduction for biomarker discovery.

Methods:
  - UMAP: nonlinear embedding for visualization + distance-based analysis
  - Agglomerative (ward, k=N_MOA_CLASSES): clustering with a preset number
    of clusters.  Two modes:
      * --mode field   (default): cluster all field profiles directly, then
        reduce to compound level via majority vote for evaluation.
      * --mode treatment: aggregate field UMAP to treatment means first,
        then cluster the 90 treatment embeddings.
  - Evaluation: NMI against known MoA labels, silhouette score

Usage:
  python src/morphology_profiling/biomarker/unsupervised.py --models cellpose4
  python src/morphology_profiling/biomarker/unsupervised.py --models all --mode field
  python src/morphology_profiling/biomarker/unsupervised.py --models all --mode treatment
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


def run_agglomerative(X, n_clusters=None, linkage="ward"):
    """Run Agglomerative (hierarchical) clustering with a preset number of clusters.

    Unlike HDBSCAN, this forces exactly ``n_clusters`` groups and assigns every
    sample to a cluster (no noise). This is required for MoA recovery where
    the number of MoA classes is known a priori (10 balanced classes for
    U2OS-Cell-Painting): density clustering can collapse to a few blobs,
    capping Hungarian accuracy at min(n_clusters_found, n_classes).
    Forcing k=N_MOA_CLASSES opens that ceiling.

    Args:
        X: (n_samples, n_features) feature matrix
        n_clusters: number of clusters (default config.DEFAULT_N_CLUSTERS = N_MOA_CLASSES)
        linkage: linkage criterion (default 'ward', minimizes within-cluster variance)

    Returns:
        labels: cluster assignments (0..n_clusters-1, no noise)
        probabilities: array of ones (Agglomerative has no soft membership)
    """
    from sklearn.cluster import AgglomerativeClustering

    if n_clusters is None:
        n_clusters = config.DEFAULT_N_CLUSTERS
    # Cannot request more clusters than samples
    n_clusters = min(n_clusters, X.shape[0])

    print(f"  Agglomerative: n_clusters={n_clusters}, linkage={linkage}")

    clusterer = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = clusterer.fit_predict(X)
    probabilities = np.ones(len(labels), dtype=float)

    n_found = len(set(labels))
    print(f"  Clusters: {n_found}, Noise points: 0")

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


def bootstrap_nmi_ci(pred_labels, true_labels, n_bootstrap=1000, ci_percent=95):
    """Bootstrap confidence interval for NMI.

    Mirrors bootstrap_hungarian_ci: stratified resampling by MoA class and a
    normal-approximation interval around the point estimate.

    Args:
        pred_labels: array of predicted cluster assignments
        true_labels: array of ground truth labels
        n_bootstrap: number of bootstrap iterations (default 1000)
        ci_percent: confidence level (default 95)

    Returns:
        ci: dict with 'NMI_CI_low' and 'NMI_CI_high'
        boot_nmis: array of bootstrap NMI values
    """
    from scipy.stats import norm
    from sklearn.metrics import normalized_mutual_info_score
    rng = np.random.default_rng(42)
    pred_labels = np.asarray(pred_labels)
    true_labels = np.asarray(true_labels)
    boot_nmis = np.zeros(n_bootstrap, dtype=float)

    unique_classes = np.unique(true_labels)
    class_indices = {c: np.where(true_labels == c)[0] for c in unique_classes}
    class_sizes = {c: len(idx) for c, idx in class_indices.items()}
    print(f"    [DEBUG] bootstrap_nmi_ci: stratified={len(unique_classes)} classes, normal-approx CI")

    for i in range(n_bootstrap):
        idx = np.concatenate([
            rng.choice(class_indices[c], size=class_sizes[c], replace=True)
            for c in unique_classes
        ])
        boot_nmis[i] = normalized_mutual_info_score(true_labels[idx], pred_labels[idx])

    pt_estimate = normalized_mutual_info_score(true_labels, pred_labels)
    se = np.std(boot_nmis, ddof=1)
    z = norm.ppf(0.5 + ci_percent / 200)
    lo = max(0.0, pt_estimate - z * se)
    hi = min(1.0, pt_estimate + z * se)

    ci = {
        'NMI_CI_low': lo,
        'NMI_CI_high': hi,
    }
    return ci, boot_nmis


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





def compound_majority_vote(field_meta, cluster_labels):
    """Reduce field-level cluster labels to compound level via majority vote.

    For each (compound, concentration) group, the cluster assigned to the
    largest number of fields becomes the compound's cluster.  Ties are
    broken by the first cluster encountered.

    Args:
        field_meta: DataFrame with at least 'compound', 'concentration', 'moa'
        cluster_labels: 1-D array of cluster labels aligned with field_meta rows

    Returns:
        compound_labels: (n_compounds,) int array of cluster assignments
        compound_moa: (n_compounds,) array of MoA labels
        compound_meta: DataFrame with compound/concentration/moa/cluster columns
    """
    meta = field_meta.copy()
    meta['cluster'] = cluster_labels

    records = []
    for (cpd, conc), grp in meta.groupby(['compound', 'concentration'], sort=False):
        # Majority vote: mode of cluster labels
        vote_counts = grp['cluster'].value_counts()
        winner = vote_counts.index[0]
        records.append({
            'compound': cpd,
            'concentration': conc,
            'moa': grp['moa'].iloc[0],
            'cluster': winner,
            'n_fields': len(grp),
            'n_agree': int(vote_counts.iloc[0]),
        })

    result = pd.DataFrame(records)
    compound_labels = result['cluster'].values
    compound_moa = result['moa'].values
    return compound_labels, compound_moa, result


def otsu_threshold(values):
    """Exact Otsu split of a 1-D sample, with no free parameter.

    Every midpoint between two consecutive distinct values is tried and the one
    minimising the within-group variance is returned.  Unlike the usual
    histogram implementation this needs no bin count, so the result depends
    only on the data.

    Intended for the perturbation-score distribution, which is bimodal: one
    mode of compounds that barely differ from DMSO and one of compounds with a
    clear phenotype.  Otsu recovers the valley between them.
    """
    x = np.sort(np.asarray(values, dtype=np.float64))
    if len(x) < 2 or x[0] == x[-1]:
        raise ValueError("Otsu needs at least two distinct values.")
    best_var, best_thr = np.inf, None
    for i in range(1, len(x)):
        if x[i] == x[i - 1]:
            continue
        lo, hi = x[:i], x[i:]
        within = len(lo) * lo.var() + len(hi) * hi.var()
        if within < best_var:
            best_var, best_thr = within, 0.5 * (x[i - 1] + x[i])
    return float(best_thr)


def load_active_treatments(model_name, min_pscore):
    """Treatments whose perturbation score exceeds ``min_pscore``.

    The score is the fraction of features that differ significantly from DMSO
    after Bonferroni correction, as measured on this model's own profiles, so
    the surviving set is model specific and not comparable across models as a
    fixed sample.

    ``min_pscore`` is either a number or the string 'otsu', in which case the
    cut is placed at the Otsu split of this model's own score distribution and
    no threshold has to be chosen by hand.

    Returns ``(keys, threshold)`` where keys is a set of
    (compound, concentration) tuples and threshold is the value actually used.
    """
    path = config.biomarker_dir(model_name) / "perturbation" / \
        "treatment_perturbation_stats.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Perturbation stats not found: {path}\n"
            f"Run perturbation.py --models {model_name} before clustering with "
            f"--min_perturbation_score.")
    stats = pd.read_csv(path)
    if isinstance(min_pscore, str):
        if min_pscore.lower() != "otsu":
            raise ValueError(
                f"min_pscore must be a number or 'otsu', got {min_pscore!r}")
        thr = otsu_threshold(stats['perturbation_score'].values)
        print(f"  Otsu threshold on {len(stats)} treatment scores: {thr:.4f}")
    else:
        thr = float(min_pscore)
    keep = stats[stats['perturbation_score'] > thr]
    print(f"  Active treatments (perturbation_score > {thr:.4f}): "
          f"{len(keep)} / {len(stats)}")
    return set(zip(keep['compound'], keep['concentration'])), thr


def run_field_level_clustering(model_name="cellpose4",
                                mode="field",
                                n_umap=None, n_neighbors=None,
                                umap_init=None, exact_knn=False,
                                no_vote=False,
                                min_pscore=None):
    """Field PCA → field UMAP → clustering → evaluation.

    Two modes:
      mode='field'      : cluster all field embeddings directly (k=10),
                          then reduce to compound level via majority vote
                          for evaluation.  Reports compound-level metrics.
                          If ``no_vote=True``, skip the majority vote and
                          evaluate directly at the field level (each field
                          is scored against its own MoA label).
      mode='treatment'  : aggregate field UMAP to treatment means, then
                          cluster the treatment embeddings (k=10).
                          Reports treatment-level metrics.

    n_umap: UMAP output dimensions for field-level embedding (default: config.FIELD_UMAP_N_COMPONENTS).
    n_neighbors: UMAP n_neighbors for field-level embedding (default: config.FIELD_UMAP_N_NEIGHBORS).

    min_pscore: if set, restrict the analysis to treatments whose perturbation
        score exceeds this value, either a number or 'otsu' to let the Otsu
        split of the model's own score distribution place the cut (see
        ``load_active_treatments``).  Inactive treatments are dropped whole,
        never field by field, and they are dropped before PCA so that
        PCA/UMAP/clustering all see active treatments only.  This keeps
        k=N_MOA_CLASSES applying to exactly the treatments being scored.
        Leaving ``min_pscore`` at None reproduces the unfiltered pipeline.

    Uses field_profiles_normalized.csv (all features, field-level).
    """
    tag = f" [Field Agglomerative k={config.DEFAULT_N_CLUSTERS}]" if mode == "field" \
        else " [Treat Agglomerative]"
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

    # 2b. Restrict to active treatments, before anything is fitted
    active_thr = None
    if min_pscore is not None:
        print(f"\n--- Active-treatment filter ---")
        active, active_thr = load_active_treatments(model_name, min_pscore)
        active_mask = np.array(
            [(c, k) in active for c, k in zip(field_meta['compound'],
                                              field_meta['concentration'])])
        if active_mask.sum() == 0:
            raise ValueError(
                f"No fields survive perturbation_score > {min_pscore}.")
        X_nd = X_nd[active_mask]
        field_meta = field_meta[active_mask].reset_index(drop=True)
        print(f"  Fields kept: {len(X_nd)}")

    # 3. PCA
    print(f"\n--- PCA ---")
    X_nd = run_pca(X_nd, n_components=50)

    # 4. UMAP (on all fields)
    if n_umap is None:
        n_umap = config.FIELD_UMAP_N_COMPONENTS
    if n_neighbors is None:
        n_neighbors = config.FIELD_UMAP_N_NEIGHBORS
    print(f"\n--- UMAP (n_components={n_umap}) ---")
    embedding = run_umap(X_nd, n_components=n_umap, n_neighbors=n_neighbors, umap_init=umap_init, exact_knn=exact_knn)

    # Build field-level UMAP DataFrame (saved later in both modes)
    field_dict = {
        'compound': field_meta['compound'].values,
        'concentration': field_meta['concentration'].values,
        'moa': field_meta['moa'].values,
    }
    for i in range(n_umap):
        field_dict[f'UMAP_{i+1}'] = embedding[:, i]
    field_umap_df = pd.DataFrame(field_dict)

    # ------------------------------------------------------------------
    # Mode-specific clustering and evaluation
    # ------------------------------------------------------------------
    if mode == "field":
        # --- Cluster all field embeddings directly ---
        print(f"\n--- Agglomerative (field-level, {len(embedding)} points) ---")
        cluster_labels, probabilities = run_agglomerative(embedding)

        if no_vote:
            # --- Field-level evaluation (no majority vote) ---
            eval_labels = cluster_labels
            eval_moa = field_meta['moa'].values
            print(f"\n--- Clustering Evaluation (field-level, no vote) ---")
            metrics = evaluate_clustering(eval_labels, eval_moa)
        else:
            # --- Compound-level evaluation via majority vote ---
            print(f"\n--- Compound Majority Vote ---")
            comp_labels, comp_moa, comp_df = compound_majority_vote(field_meta, cluster_labels)
            print(f"  Reduced to {len(comp_df)} compounds")
            eval_labels, eval_moa = comp_labels, comp_moa

            print(f"\n--- Clustering Evaluation (compound-level) ---")
            metrics = evaluate_clustering(eval_labels, eval_moa)

        # Bootstrap CI on the evaluated labels
        try:
            ci, _ = bootstrap_hungarian_ci(eval_labels, eval_moa)
            metrics.update(ci)
            print(f"  Accuracy 95% CI: [{ci['Accuracy_CI_low']:.3f}, {ci['Accuracy_CI_high']:.3f}]")
        except Exception as e:
            print(f"  [WARN] bootstrap CI failed: {e}")

        try:
            ci, _ = bootstrap_nmi_ci(eval_labels, eval_moa)
            metrics.update(ci)
            print(f"  NMI 95% CI: [{ci['NMI_CI_low']:.3f}, {ci['NMI_CI_high']:.3f}]")
        except Exception as e:
            print(f"  [WARN] NMI bootstrap CI failed: {e}")

        # Silhouette on field-level UMAP (subsample for speed if >5000)
        from sklearn.metrics import silhouette_score
        valid = cluster_labels != -1
        if valid.sum() > 2 and len(set(cluster_labels[valid])) >= 2:
            sil_X = embedding[valid]
            sil_l = cluster_labels[valid]
            if len(sil_X) > 5000:
                rng = np.random.default_rng(42)
                idx = rng.choice(len(sil_X), 5000, replace=False)
                sil_X, sil_l = sil_X[idx], sil_l[idx]
            sil = silhouette_score(sil_X, sil_l)
            metrics['Silhouette'] = sil
            print(f"  Silhouette (field-level, subsampled): {sil:.3f}")

        # MoA enrichment (on field-level labels)
        print(f"\n--- MoA Enrichment ---")
        enrichment_df = find_moa_enriched_clusters(cluster_labels,
                                                   field_meta['moa'].values)
        print(enrichment_df.to_string(index=False))

        # Add cluster column to field UMAP DataFrame
        field_umap_df['cluster'] = cluster_labels

        # Save
        output_dir = config.biomarker_dir(model_name) / "unsupervised"
        output_dir.mkdir(parents=True, exist_ok=True)

        field_umap_df.to_csv(output_dir / "field_umap_embedding.csv", index=False)
        print(f"\n  Saved: {output_dir / 'field_umap_embedding.csv'}")

        # cluster_assignments: field-level
        cluster_df = field_meta.copy()
        cluster_df['cluster'] = cluster_labels
        cluster_df['probability'] = probabilities
        cluster_df.to_csv(output_dir / "cluster_assignments.csv", index=False)
        print(f"  Saved: {output_dir / 'cluster_assignments.csv'} ({len(cluster_df)} rows, field-level)")

        enrichment_df.to_csv(output_dir / "cluster_moa_enrichment.csv", index=False)
        print(f"  Saved: {output_dir / 'cluster_moa_enrichment.csv'}")

        level_tag = "field-level" if no_vote else "compound-level"
        if active_thr is not None:
            # Record which cut was used, since 'otsu' resolves to a different
            # value per model.
            metrics['active_min_perturbation_score'] = active_thr
            metrics['n_evaluated'] = len(eval_labels)
        pd.DataFrame([metrics]).to_csv(output_dir / "clustering_metrics.csv", index=False)
        print(f"  Saved: {output_dir / 'clustering_metrics.csv'} ({level_tag} metrics)")

        return field_umap_df, cluster_df, enrichment_df, metrics

    else:
        # --- mode == 'treatment': original logic ---
        print(f"\n--- Aggregating field UMAP to treatments ---")
        embed_dict_agg = {
            'compound': field_meta['compound'].values,
            'concentration': field_meta['concentration'].values,
        }
        for i in range(n_umap):
            embed_dict_agg[f'UMAP_{i+1}'] = embedding[:, i]
        embed_df = pd.DataFrame(embed_dict_agg)
        treat_embed = embed_df.groupby(['compound', 'concentration'], sort=False).mean().values

        # Treatment-level metadata
        treat_meta = field_meta.groupby(group_keys, sort=False).first().reset_index()
        print(f"  Aggregated to {len(treat_meta)} treatments")

        # Clustering on treatment-level UMAP
        print(f"\n--- Agglomerative (treatment-level) ---")
        cluster_labels, probabilities = run_agglomerative(treat_embed)

        # Evaluate
        print(f"\n--- Clustering Evaluation ---")
        moa_labels = treat_meta['moa'].values
        metrics = evaluate_clustering(cluster_labels, moa_labels)

        # Bootstrap CI
        try:
            ci, _ = bootstrap_hungarian_ci(cluster_labels, moa_labels)
            metrics.update(ci)
            print(f"  Accuracy 95% CI: [{ci['Accuracy_CI_low']:.3f}, {ci['Accuracy_CI_high']:.3f}]")
        except Exception as e:
            print(f"  [WARN] bootstrap CI failed: {e}")

        try:
            ci, _ = bootstrap_nmi_ci(cluster_labels, moa_labels)
            metrics.update(ci)
            print(f"  NMI 95% CI: [{ci['NMI_CI_low']:.3f}, {ci['NMI_CI_high']:.3f}]")
        except Exception as e:
            print(f"  [WARN] NMI bootstrap CI failed: {e}")

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

        field_umap_df.to_csv(output_dir / "field_umap_embedding.csv", index=False)
        print(f"\n  Saved: {output_dir / 'field_umap_embedding.csv'}")

        cluster_df = treatment_labels
        cluster_df.to_csv(output_dir / "cluster_assignments.csv", index=False)
        print(f"  Saved: {output_dir / 'cluster_assignments.csv'} ({len(cluster_df)} rows, treatment-level)")

        enrichment_df.to_csv(output_dir / "cluster_moa_enrichment.csv", index=False)
        print(f"  Saved: {output_dir / 'cluster_moa_enrichment.csv'}")

        if active_thr is not None:
            metrics['active_min_perturbation_score'] = active_thr
            metrics['n_evaluated'] = len(cluster_labels)
        pd.DataFrame([metrics]).to_csv(output_dir / "clustering_metrics.csv", index=False)
        print(f"  Saved: {output_dir / 'clustering_metrics.csv'}")

        return field_umap_df, cluster_df, enrichment_df, metrics


def run_per_plate_field_clustering(model_name="cellpose4",
                                   n_umap=None, n_neighbors=None,
                                   umap_init=None, exact_knn=False):
    """Cluster each plate independently (field mode), then average metrics.

    Each plate is treated as an independent replicate: for every plate we run
    the full field-mode pipeline (exclude DMSO -> PCA -> field UMAP ->
    Agglomerative k=10 -> compound majority vote -> compound-level evaluation).
    Metrics are reported per plate and averaged across plates. This sidesteps
    the plate-to-plate batch effect that appears when all plates are pooled
    into a single embedding.

    (Bootstrap CI / silhouette are intentionally skipped here.)
    """
    print("=" * 60)
    print(f"Per-Plate Field Clustering [{model_name}] "
          f"[Agglomerative k={config.DEFAULT_N_CLUSTERS}]")
    print("=" * 60)

    path = config.aggregation_dir(model_name) / "field_profiles_normalized.csv"
    if not path.exists():
        raise FileNotFoundError(f"Field profiles not found: {path}")
    df = pd.read_csv(path)
    if 'plate' not in df.columns:
        raise ValueError("field_profiles_normalized.csv has no 'plate' column; "
                         "cannot run per-plate clustering.")
    feat_cols = get_field_feature_columns(df)
    plates = sorted(df['plate'].unique())
    print(f"Loaded field profiles: {len(df)} fields, {len(feat_cols)} features, "
          f"{len(plates)} plates")

    if n_umap is None:
        n_umap = config.FIELD_UMAP_N_COMPONENTS
    if n_neighbors is None:
        n_neighbors = config.FIELD_UMAP_N_NEIGHBORS

    group_keys = ['compound', 'concentration', 'moa', 'is_dmso']

    per_plate_rows = []
    all_assignments = []
    for plate in plates:
        print("\n" + "-" * 60)
        print(f"Plate {plate}")
        print("-" * 60)
        df_p = df[df['plate'] == plate]

        X = df_p[feat_cols].values.astype(np.float64)
        X = np.nan_to_num(X, nan=0.0)
        non_dmso = (df_p['is_dmso'] != True).values
        X_nd = X[non_dmso]
        field_meta = df_p[non_dmso][group_keys].reset_index(drop=True)
        print(f"  Non-DMSO fields: {len(X_nd)}")
        if len(X_nd) < config.DEFAULT_N_CLUSTERS:
            print(f"  [WARN] Plate {plate}: only {len(X_nd)} fields "
                  f"(< k={config.DEFAULT_N_CLUSTERS}); skipping.")
            continue

        X_nd = run_pca(X_nd, n_components=50)
        embedding = run_umap(X_nd, n_components=n_umap, n_neighbors=n_neighbors,
                             umap_init=umap_init, exact_knn=exact_knn)
        cluster_labels, _ = run_agglomerative(embedding)

        comp_labels, comp_moa, comp_df = compound_majority_vote(field_meta, cluster_labels)
        metrics = evaluate_clustering(comp_labels, comp_moa)

        row = {'plate': plate, 'n_compounds': len(comp_df),
               'n_fields': int(len(X_nd))}
        row.update(metrics)
        per_plate_rows.append(row)

        assign = field_meta.copy()
        assign.insert(0, 'plate', plate)
        assign['cluster'] = cluster_labels
        all_assignments.append(assign)

    if not per_plate_rows:
        raise RuntimeError("No plate produced valid clustering results.")

    per_plate_df = pd.DataFrame(per_plate_rows)

    # Aggregate across plates: mean (and std) of each numeric metric column
    metric_cols = [c for c in per_plate_df.columns
                   if c not in ('plate', 'n_compounds', 'n_fields')]
    mean_row = {'plate': 'MEAN', 'n_plates': len(per_plate_df)}
    for c in metric_cols:
        mean_row[c] = per_plate_df[c].mean()
        mean_row[f'{c}_std'] = per_plate_df[c].std(ddof=0)

    # Save
    output_dir = config.biomarker_dir(model_name) / "unsupervised"
    output_dir.mkdir(parents=True, exist_ok=True)

    per_plate_df.to_csv(output_dir / "per_plate_metrics.csv", index=False)
    print(f"\n  Saved: {output_dir / 'per_plate_metrics.csv'} ({len(per_plate_df)} plates)")

    pd.DataFrame([mean_row]).to_csv(output_dir / "clustering_metrics.csv", index=False)
    print(f"  Saved: {output_dir / 'clustering_metrics.csv'} (across-plate mean)")

    pd.concat(all_assignments, ignore_index=True).to_csv(
        output_dir / "cluster_assignments.csv", index=False)
    print(f"  Saved: {output_dir / 'cluster_assignments.csv'} (field-level, per plate)")

    # Console summary
    print("\n" + "=" * 60)
    print("Per-plate summary")
    print("=" * 60)
    show_cols = ['plate'] + [c for c in ['Accuracy', 'NMI', 'ARI', 'Purity']
                             if c in per_plate_df.columns]
    print(per_plate_df[show_cols].to_string(index=False))
    if 'Accuracy' in per_plate_df.columns:
        print(f"\n  Mean Accuracy across {len(per_plate_df)} plates: "
              f"{per_plate_df['Accuracy'].mean():.3f} "
              f"\u00b1 {per_plate_df['Accuracy'].std(ddof=0):.3f}")

    return per_plate_df, mean_row


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, nargs="+", default=None,
                        help="Models. Use 'all' for all 5 models.")
    parser.add_argument("--mode", type=str, default="field",
                        choices=["field", "treatment"],
                        help="Clustering granularity: 'field' clusters all field "
                             "profiles directly (default); 'treatment' aggregates to "
                             "treatment means first.")
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
    parser.add_argument("--per_plate", action="store_true",
                        help="Cluster each plate independently (field mode) and "
                             "average metrics across plates. Sidesteps batch "
                             "effect; skips bootstrap CI.")
    parser.add_argument("--no_vote", action="store_true",
                        help="Field mode only: skip compound majority vote and "
                             "evaluate accuracy directly at the field level (each "
                             "field scored against its own MoA label).")
    parser.add_argument("--min_perturbation_score", type=str, default="otsu",
                        help="Restrict the analysis to treatments whose "
                             "perturbation score exceeds this value, using each "
                             "model's own score. Defaults to 'otsu', which "
                             "places the cut at the Otsu split of the model's "
                             "own score distribution so no threshold is chosen "
                             "by hand. Pass a number to override the cut, or "
                             "'none' to reproduce the unfiltered pipeline. "
                             "Requires perturbation.py to have been run first.")
    args = parser.parse_args()

    min_pscore = args.min_perturbation_score
    if min_pscore.lower() in ("none", "off"):
        min_pscore = None
    elif min_pscore.lower() != "otsu":
        try:
            min_pscore = float(min_pscore)
        except ValueError:
            parser.error("--min_perturbation_score expects a number, 'otsu' or "
                         f"'none', got {min_pscore!r}")

    if args.models is None:
        models_to_run = ["cellpose4"]
    elif "all" in args.models:
        models_to_run = config.MODELS
    else:
        models_to_run = args.models

    for m in models_to_run:
        if args.per_plate:
            run_per_plate_field_clustering(model_name=m,
                                           n_umap=args.n_umap, n_neighbors=args.n_neighbors,
                                           umap_init=args.umap_init, exact_knn=args.exact_knn)
            continue
        run_field_level_clustering(model_name=m,
                                   mode=args.mode,
                                   n_umap=args.n_umap, n_neighbors=args.n_neighbors,
                                   umap_init=args.umap_init, exact_knn=args.exact_knn,
                                   no_vote=args.no_vote,
                                   min_pscore=min_pscore)
