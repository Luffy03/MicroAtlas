"""
biomarker/feature_attribution.py
=================================
Statistical biomarker identification via feature-level testing.

For each MoA class vs all others:
  - Welch's t-test per feature
  - Cohen's d effect size
  - Bonferroni-corrected significance
  - Rank features by effect size

For each MoA class vs DMSO:
  - Which features are most perturbed
  - Feature importance ranking

No supervised classifiers or SHAP. Pure statistical testing.

Usage:
  python biomarker_discovery/biomarker/feature_attribution.py --models cellpose4
  python biomarker_discovery/biomarker/feature_attribution.py --models all
"""

import multiprocessing
import sys
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


def load_treatment_profiles(model_name="cellpose4"):
    """Load treatment-level profiles."""
    path = config.aggregation_dir(model_name) / "treatment_profiles.csv"
    if not path.exists():
        raise FileNotFoundError(f"Treatment profiles not found: {path}")
    df = pd.read_csv(path)
    print(f"Loaded treatment profiles: {len(df)} treatments")
    return df


def get_feature_columns(df):
    """Get numeric feature columns."""
    meta_cols = {'plate', 'field', 'compound', 'concentration', 'moa',
                 'is_dmso', 'well', 'n_cells', 'n_cells_total',
                 'n_fields', 'cell_id', 'smiles', 'replicate'}
    return [c for c in df.columns
            if c not in meta_cols and df[c].dtype in (np.float64, np.int64, float, int)]


def moa_vs_rest(df, moa_class, feat_cols, alpha=0.05):
    """Test each feature: MoA class vs all other MoA classes.

    Args:
        df: treatment profiles (non-DMSO, with known MoA)
        moa_class: target MoA class name
        feat_cols: feature column names
        alpha: significance level

    Returns:
        DataFrame: feature, t_stat, p_value, cohens_d, significant, direction
    """
    mask_in = df['moa'] == moa_class
    mask_out = (df['moa'] != moa_class) & (df['moa'] != 'unknown')

    vals_in = df.loc[mask_in, feat_cols].values.astype(np.float64)
    vals_out = df.loc[mask_out, feat_cols].values.astype(np.float64)

    n_tests = len(feat_cols)
    records = []

    for i, feat in enumerate(feat_cols):
        a = vals_in[:, i]
        b = vals_out[:, i]
        a = a[np.isfinite(a)]
        b = b[np.isfinite(b)]

        if len(a) < 2 or len(b) < 2:
            records.append({
                'feature': feat, 't_stat': 0.0, 'p_value': 1.0,
                'cohens_d': 0.0, 'significant': False, 'direction': 0.0,
            })
            continue

        t_stat, p_value = stats.ttest_ind(a, b, equal_var=False)
        t_stat = float(t_stat) if np.isfinite(t_stat) else 0.0
        p_value = float(p_value) if np.isfinite(p_value) else 1.0

        # Cohen's d
        mean_diff = np.mean(a) - np.mean(b)
        pooled_var = (
            (len(a) - 1) * np.var(a, ddof=1) + (len(b) - 1) * np.var(b, ddof=1)
        ) / (len(a) + len(b) - 2)
        pooled_std = np.sqrt(pooled_var)
        cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0.0

        records.append({
            'feature': feat,
            't_stat': t_stat,
            'p_value': p_value,
            'cohens_d': float(cohens_d),
            'significant': False,  # set after correction
            'direction': float(np.sign(cohens_d)),
        })

    result = pd.DataFrame(records)

    # Bonferroni correction
    corrected_alpha = alpha / n_tests
    result['significant'] = result['p_value'] < corrected_alpha
    result['corrected_p'] = np.minimum(result['p_value'] * n_tests, 1.0)

    # Sort by absolute Cohen's d
    result = result.sort_values('cohens_d', key=abs, ascending=False)
    return result


def moa_vs_dmso(df, dmso_df, moa_class, feat_cols, alpha=0.05):
    """Test each feature: MoA class vs DMSO controls.

    Returns:
        DataFrame: feature, t_stat, p_value, cohens_d, significant
    """
    mask = df['moa'] == moa_class
    vals_moa = df.loc[mask, feat_cols].values.astype(np.float64)
    vals_dmso = dmso_df[feat_cols].values.astype(np.float64)

    n_tests = len(feat_cols)
    records = []

    for i, feat in enumerate(feat_cols):
        a = vals_moa[:, i]
        b = vals_dmso[:, i]
        a = a[np.isfinite(a)]
        b = b[np.isfinite(b)]

        if len(a) < 2 or len(b) < 2:
            records.append({
                'feature': feat, 't_stat': 0.0, 'p_value': 1.0,
                'cohens_d': 0.0, 'significant': False,
            })
            continue

        t_stat, p_value = stats.ttest_ind(a, b, equal_var=False)
        t_stat = float(t_stat) if np.isfinite(t_stat) else 0.0
        p_value = float(p_value) if np.isfinite(p_value) else 1.0

        mean_diff = np.mean(a) - np.mean(b)
        pooled_var = (
            (len(a) - 1) * np.var(a, ddof=1) + (len(b) - 1) * np.var(b, ddof=1)
        ) / (len(a) + len(b) - 2)
        pooled_std = np.sqrt(pooled_var)
        cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0.0

        records.append({
            'feature': feat,
            't_stat': t_stat,
            'p_value': p_value,
            'cohens_d': float(cohens_d),
            'significant': False,
        })

    result = pd.DataFrame(records)
    corrected_alpha = alpha / n_tests
    result['significant'] = result['p_value'] < corrected_alpha
    result['corrected_p'] = np.minimum(result['p_value'] * n_tests, 1.0)
    result = result.sort_values('cohens_d', key=abs, ascending=False)
    return result


def _worker_moa_attribution(moa, df, dmso_df, feat_cols, alpha, top_k):
    """Worker: biomarker identification for one MoA class."""
    n = (df['moa'] == moa).sum()

    # vs rest
    vs_rest = moa_vs_rest(df, moa, feat_cols, alpha=alpha)
    n_sig_rest = vs_rest['significant'].sum()
    top_rest = vs_rest.head(top_k)

    # vs DMSO
    combined = None
    if dmso_df is not None:
        vs_dmso = moa_vs_dmso(df, dmso_df, moa, feat_cols, alpha=alpha)
        n_sig_dmso = vs_dmso['significant'].sum()

        sig_rest = set(vs_rest[vs_rest['significant']]['feature'])
        sig_dmso = set(vs_dmso[vs_dmso['significant']]['feature'])
        intersection = sig_rest & sig_dmso

        combined = pd.DataFrame({
            'feature': feat_cols,
            'cohens_d_vs_rest': vs_rest.set_index('feature').loc[feat_cols, 'cohens_d'].values,
            'p_vs_rest': vs_rest.set_index('feature').loc[feat_cols, 'corrected_p'].values,
            'cohens_d_vs_dmso': vs_dmso.set_index('feature').loc[feat_cols, 'cohens_d'].values,
            'p_vs_dmso': vs_dmso.set_index('feature').loc[feat_cols, 'corrected_p'].values,
        })
        combined['biomarker'] = combined['feature'].isin(intersection)
        combined = combined.sort_values(
            'cohens_d_vs_rest', key=abs, ascending=False)
    else:
        combined = top_rest.copy()
        combined['biomarker'] = combined['significant']
        n_sig_dmso = 0

    combined['moa'] = moa

    return {
        'moa': moa,
        'n_treatments': n,
        'n_sig_rest': int(n_sig_rest),
        'top_feature_rest': top_rest['feature'].iloc[0] if len(top_rest) > 0 else '',
        'top_cohens_d_rest': float(top_rest['cohens_d'].iloc[0]) if len(top_rest) > 0 else 0.0,
        'combined': combined,
    }


def identify_biomarker_features(df, model_name="cellpose4", top_k=None,
                                 n_workers=8):
    """Identify biomarker features for each MoA class.

    For each MoA:
      1. vs rest: find features with largest Cohen's d
      2. vs DMSO: find most perturbed features
      3. Intersection: features significant in both comparisons

    Args:
        df: treatment profiles
        top_k: number of top features to report (default from config)
        n_workers: number of parallel workers (default 8)

    Returns:
        biomarkers: dict {moa_class: DataFrame of top features}
    """
    if top_k is None:
        top_k = config.ATTRIBUTION_TOP_K

    feat_cols = get_feature_columns(df)
    moa_classes = sorted(df[df['moa'] != 'unknown']['moa'].unique())

    # Filter MoA classes with enough samples
    min_samples = config.MOA_MIN_SAMPLES_PER_CLASS
    valid_moas = []
    for moa in moa_classes:
        n = (df['moa'] == moa).sum()
        if n >= min_samples:
            valid_moas.append(moa)
        else:
            print(f"  [SKIP] {moa}: only {n} treatments (need {min_samples})")

    print(f"\n  Analyzing {len(valid_moas)} MoA classes ({n_workers} workers)...")

    # Load DMSO
    dmso_path = config.aggregation_dir(model_name) / "dmso_profiles.csv"
    dmso_df = pd.read_csv(dmso_path) if dmso_path.exists() else None

    # Parallel execution
    worker_fn = partial(
        _worker_moa_attribution,
        df=df,
        dmso_df=dmso_df,
        feat_cols=feat_cols,
        alpha=0.05,
        top_k=top_k,
    )

    biomarkers = {}
    all_results = []

    if n_workers > 1 and len(valid_moas) > 1:
        with multiprocessing.Pool(processes=min(n_workers, len(valid_moas))) as pool:
            for result in tqdm(
                pool.imap_unordered(worker_fn, valid_moas),
                total=len(valid_moas),
                desc="  MoA attribution",
                unit="class",
            ):
                moa = result['moa']
                biomarkers[moa] = result['combined']
                all_results.append(result['combined'])
                print(f"    {moa}: {result['n_treatments']} treatments, "
                      f"{result['n_sig_rest']} sig vs rest")
    else:
        for moa in tqdm(valid_moas, desc="  MoA attribution", unit="class"):
            result = worker_fn(moa)
            biomarkers[result['moa']] = result['combined']
            all_results.append(result['combined'])
            print(f"    {result['moa']}: {result['n_treatments']} treatments, "
                  f"{result['n_sig_rest']} sig vs rest")

    return biomarkers, pd.concat(all_results, ignore_index=True)


def feature_category_summary(biomarkers_df):
    """Summarize biomarker features by category.

    Group features into: AreaShape, Intensity, Texture, Granularity,
    RadialDistribution, Correlation.

    Returns:
        DataFrame with category-level counts per MoA
    """
    def get_category(feat_name):
        for cat in ['AreaShape', 'Intensity', 'Texture', 'Granularity',
                     'RadialDistribution', 'Correlation', 'Location']:
            if cat in feat_name:
                return cat
        return 'Other'

    df = biomarkers_df.copy()
    df['category'] = df['feature'].apply(get_category)

    # Category counts per MoA (only biomarker features)
    bm = df[df['biomarker']]
    summary = bm.groupby(['moa', 'category']).size().unstack(fill_value=0)
    return summary


def run_feature_attribution(model_name="cellpose4", n_workers=8):
    """Run full feature attribution pipeline."""
    print("=" * 60)
    print(f"Feature Attribution (Biomarker Identification) [{model_name}]")
    print("=" * 60)

    df = load_treatment_profiles(model_name=model_name)

    # Identify biomarkers
    biomarkers, all_results = identify_biomarker_features(
        df, model_name=model_name, n_workers=n_workers)

    # Category summary
    print("\n\n=== Biomarker Features by Category ===")
    cat_summary = feature_category_summary(all_results)
    print(cat_summary.to_string())

    # Save
    output_dir = config.biomarker_dir(model_name) / "attribution"
    output_dir.mkdir(parents=True, exist_ok=True)

    # All results
    all_path = output_dir / "feature_attribution_all.csv"
    all_results.to_csv(all_path, index=False)
    print(f"\n  Saved: {all_path}")

    # Biomarker-only
    bm = all_results[all_results['biomarker']]
    bm_path = output_dir / "biomarker_features.csv"
    bm.to_csv(bm_path, index=False)
    print(f"  Saved: {bm_path} ({len(bm)} biomarker features)")

    # Category summary
    cat_path = output_dir / "category_summary.csv"
    cat_summary.to_csv(cat_path)
    print(f"  Saved: {cat_path}")

    # Per-MoA top features (for visualization)
    for moa, moa_df in biomarkers.items():
        moa_safe = moa.replace(' ', '_').replace('/', '_')
        moa_path = output_dir / f"top_features_{moa_safe}.csv"
        moa_df.head(config.ATTRIBUTION_TOP_K).to_csv(moa_path, index=False)

    # Print top biomarkers per MoA
    print("\n\n=== Top Biomarker Features per MoA ===")
    for moa in sorted(biomarkers.keys()):
        bm_moa = biomarkers[moa][biomarkers[moa]['biomarker']]
        if len(bm_moa) == 0:
            bm_moa = biomarkers[moa].head(3)
        print(f"\n  {moa}:")
        for _, row in bm_moa.head(5).iterrows():
            d_rest = row.get('cohens_d_vs_rest', row.get('cohens_d', 0))
            print(f"    {row['feature']}: d={d_rest:.2f}")

    return biomarkers, all_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, nargs="+", default=None,
                        help="Models. Use 'all' for all 5 models.")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of parallel workers (default: 8)")
    args = parser.parse_args()

    if args.models is None:
        models_to_run = ["cellpose4"]
    elif "all" in args.models:
        models_to_run = config.MODELS
    else:
        models_to_run = args.models

    for m in models_to_run:
        run_feature_attribution(model_name=m, n_workers=args.workers)
