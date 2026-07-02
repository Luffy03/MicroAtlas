"""
biomarker/perturbation.py
=========================
Compound perturbation analysis vs DMSO controls.

For each treatment (compound + concentration):
  - Welch's t-test per feature: treatment vs DMSO
  - Cohen's d effect size per feature
  - Perturbation score: fraction of significantly changed features
  - Dose-response analysis: monotonic trend across concentrations
  - MoA consistency validation: perturbation profile correlation vs MoA labels

No supervised classifiers used. Pure statistical testing.

Usage:
  python biomarker_discovery/biomarker/perturbation.py --models cellpose4
  python biomarker_discovery/biomarker/perturbation.py --models all
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


def load_profiles(model_name="cellpose4"):
    """Load normalized field-level profiles + DMSO controls."""
    agg_dir = config.aggregation_dir(model_name)
    field_path = agg_dir / "field_profiles_normalized.csv"
    dmso_path = agg_dir / "dmso_profiles.csv"

    if not field_path.exists():
        raise FileNotFoundError(
            f"Field profiles not found: {field_path}\n"
            f"Run feature_aggregation.py first."
        )

    field_df = pd.read_csv(field_path)
    dmso_df = pd.read_csv(dmso_path)

    print(f"Loaded: {len(field_df)} field profiles, {len(dmso_df)} DMSO controls")
    return field_df, dmso_df


def get_feature_columns(df):
    """Get numeric feature columns."""
    meta_cols = {'plate', 'field', 'compound', 'concentration', 'moa',
                 'is_dmso', 'well', 'n_cells', 'cell_id', 'smiles',
                 'replicate'}
    return [c for c in df.columns
            if c not in meta_cols and df[c].dtype in (np.float64, np.int64, float, int)]


def welch_ttest_per_feature(treatment_vals, dmso_vals, feature_names):
    """Welch's t-test for each feature.

    Returns:
        DataFrame with columns: feature, t_stat, p_value, significant
    """
    results = []
    for i, feat in enumerate(feature_names):
        t_vals = treatment_vals[:, i]
        d_vals = dmso_vals[:, i]

        # Remove NaNs
        t_vals = t_vals[np.isfinite(t_vals)]
        d_vals = d_vals[np.isfinite(d_vals)]

        if len(t_vals) < 2 or len(d_vals) < 2:
            results.append({
                'feature': feat,
                't_stat': np.nan,
                'p_value': 1.0,
                'significant': False,
            })
            continue

        t_stat, p_value = stats.ttest_ind(t_vals, d_vals, equal_var=False)

        results.append({
            'feature': feat,
            't_stat': float(t_stat) if np.isfinite(t_stat) else 0.0,
            'p_value': float(p_value) if np.isfinite(p_value) else 1.0,
            'significant': False,  # set after Bonferroni
        })

    return pd.DataFrame(results)


def cohens_d(treatment_vals, dmso_vals, feature_names):
    """Compute Cohen's d effect size for each feature.

    d = (mean_treatment - mean_dmso) / pooled_std
    """
    results = []
    for i, feat in enumerate(feature_names):
        t_vals = treatment_vals[:, i]
        d_vals = dmso_vals[:, i]
        t_vals = t_vals[np.isfinite(t_vals)]
        d_vals = d_vals[np.isfinite(d_vals)]

        if len(t_vals) < 2 or len(d_vals) < 2:
            results.append({'feature': feat, 'cohens_d': 0.0})
            continue

        mean_diff = np.mean(t_vals) - np.mean(d_vals)
        pooled_var = (
            (len(t_vals) - 1) * np.var(t_vals, ddof=1)
            + (len(d_vals) - 1) * np.var(d_vals, ddof=1)
        ) / (len(t_vals) + len(d_vals) - 2)
        pooled_std = np.sqrt(pooled_var)

        d = mean_diff / pooled_std if pooled_std > 0 else 0.0
        results.append({'feature': feat, 'cohens_d': float(d)})

    return pd.DataFrame(results)


def compute_perturbation_score(ttest_df, alpha=0.05):
    """Compute perturbation score for a treatment.

    Perturbation score = fraction of features significantly different from DMSO
    (after Bonferroni correction).
    """
    n_features = len(ttest_df)
    if n_features == 0:
        return 0.0, 0

    # Bonferroni correction
    corrected_alpha = alpha / n_features
    ttest_df = ttest_df.copy()
    ttest_df['significant'] = ttest_df['p_value'] < corrected_alpha

    n_sig = ttest_df['significant'].sum()
    score = n_sig / n_features
    return float(score), int(n_sig)


def _worker_perturbation_treatment(args, dmso_vals, feat_cols, alpha):
    """Worker: run perturbation analysis for one treatment group."""
    (cpd, conc), treatment_vals, n_group, moa = args

    if n_group < 3:
        return None, None

    # Welch's t-test
    ttest_df = welch_ttest_per_feature(treatment_vals, dmso_vals, feat_cols)
    # Cohen's d
    cd_df = cohens_d(treatment_vals, dmso_vals, feat_cols)
    # Perturbation score
    score, n_sig = compute_perturbation_score(ttest_df, alpha)

    # Merge detail
    detail = ttest_df.merge(cd_df, on='feature')
    detail['compound'] = cpd
    detail['concentration'] = conc
    detail['moa'] = moa

    # Treatment-level summary
    treatment_rec = {
        'compound': cpd,
        'concentration': conc,
        'moa': moa,
        'n_fields': n_group,
        'perturbation_score': score,
        'n_significant_features': n_sig,
        'n_total_features': len(feat_cols),
        'max_abs_cohens_d': float(cd_df['cohens_d'].abs().max()),
        'mean_abs_cohens_d': float(cd_df['cohens_d'].abs().mean()),
    }

    return treatment_rec, detail


def perturbation_analysis(field_df, dmso_df, alpha=0.05, n_workers=8):
    """Run perturbation analysis for all treatments.

    For each unique (compound, concentration) pair:
      1. Welch's t-test vs DMSO
      2. Cohen's d
      3. Perturbation score

    Returns:
        treatment_stats: per-treatment summary
        per_feature_stats: per-treatment-per-feature details
    """
    feat_cols = get_feature_columns(field_df)
    print(f"\nRunning perturbation analysis ({len(feat_cols)} features, "
          f"{n_workers} workers)...")

    dmso_vals = dmso_df[feat_cols].values.astype(np.float64)

    # Guard: no DMSO controls
    if len(dmso_df) == 0:
        print("  [ERROR] No DMSO controls found. Run preprocess.py first.")
        return pd.DataFrame(), pd.DataFrame()

    # Prepare task list
    non_dmso = field_df[~field_df['is_dmso']]
    groups = non_dmso.groupby(['compound', 'concentration'])

    tasks = []
    for (cpd, conc), group in groups:
        treatment_vals = group[feat_cols].values.astype(np.float64)
        moa = group['moa'].iloc[0]
        tasks.append(((cpd, conc), treatment_vals, len(group), moa))

    # Parallel execution
    worker_fn = partial(
        _worker_perturbation_treatment,
        dmso_vals=dmso_vals,
        feat_cols=feat_cols,
        alpha=alpha,
    )

    treatment_records = []
    all_detail_dfs = []

    if n_workers > 1 and len(tasks) > 1:
        with multiprocessing.Pool(processes=n_workers) as pool:
            for rec, detail in tqdm(
                pool.imap_unordered(worker_fn, tasks),
                total=len(tasks),
                desc="  Perturbation",
                unit="treatment",
            ):
                if rec is not None:
                    treatment_records.append(rec)
                    all_detail_dfs.append(detail)
    else:
        for task in tqdm(tasks, desc="  Perturbation", unit="treatment"):
            rec, detail = worker_fn(task)
            if rec is not None:
                treatment_records.append(rec)
                all_detail_dfs.append(detail)

    treatment_stats = pd.DataFrame(treatment_records)
    if len(all_detail_dfs) > 0:
        per_feature_stats = pd.concat(all_detail_dfs, ignore_index=True)
    else:
        per_feature_stats = pd.DataFrame()
        print("  [WARN] No treatments analyzed (no valid compound/concentration groups).")

    print(f"  Analyzed {len(treatment_stats)} treatments")
    if len(treatment_stats) > 0:
        n_active = (treatment_stats['perturbation_score'] > 0.05).sum()
        print(f"  Active treatments (score > 0.05): {n_active}")

    return treatment_stats, per_feature_stats


def _worker_dose_response_compound(args, feat_cols, alpha):
    """Worker: dose-response analysis for one compound."""
    cpd, cpd_data, concentrations = args
    moa = cpd_data['moa'].iloc[0]
    records = []

    for feat in feat_cols:
        conc_medians = []
        for conc in concentrations:
            vals = cpd_data[cpd_data['concentration'] == conc][feat].values
            vals = vals[np.isfinite(vals)]
            conc_medians.append(np.median(vals) if len(vals) > 0 else np.nan)

        conc_medians = np.array(conc_medians)
        valid = np.isfinite(conc_medians)
        if valid.sum() < 3:
            continue

        conc_arr = np.array(concentrations)[valid]
        med_arr = conc_medians[valid]
        r, p = stats.spearmanr(conc_arr, med_arr)

        if np.isfinite(r) and np.isfinite(p):
            records.append({
                'compound': cpd, 'moa': moa, 'feature': feat,
                'spearman_r': float(r), 'p_value': float(p),
                'monotonic': abs(float(r)) > 0.7 and p < alpha,
            })

    return records


def dose_response_analysis(field_df, alpha=0.05, n_workers=8):
    """Analyze dose-response: which features change monotonically with concentration.

    For each compound, test if features show monotonic trend across concentrations.
    Uses Spearman rank correlation between concentration and feature median.

    Returns:
        DataFrame with compound, feature, spearman_r, p_value, monotonic
    """
    feat_cols = get_feature_columns(field_df)
    non_dmso = field_df[~field_df['is_dmso']]

    print(f"\nDose-response analysis ({len(feat_cols)} features, "
          f"{n_workers} workers)...")

    # Prepare per-compound tasks
    tasks = []
    for cpd in non_dmso['compound'].unique():
        cpd_data = non_dmso[non_dmso['compound'] == cpd]
        concentrations = sorted(cpd_data['concentration'].dropna().unique())
        if len(concentrations) >= 3:
            tasks.append((cpd, cpd_data, concentrations))

    worker_fn = partial(
        _worker_dose_response_compound,
        feat_cols=feat_cols,
        alpha=alpha,
    )

    all_records = []
    if n_workers > 1 and len(tasks) > 1:
        with multiprocessing.Pool(processes=n_workers) as pool:
            for recs in tqdm(
                pool.imap_unordered(worker_fn, tasks),
                total=len(tasks),
                desc="  Dose-response",
                unit="compound",
            ):
                all_records.extend(recs)
    else:
        for task in tqdm(tasks, desc="  Dose-response", unit="compound"):
            recs = worker_fn(task)
            all_records.extend(recs)

    dose_df = pd.DataFrame(all_records)
    n_monotonic = dose_df['monotonic'].sum()
    print(f"  Total tests: {len(dose_df)}")
    print(f"  Monotonic (|r|>0.7, p<0.05): {n_monotonic}")

    return dose_df


# =========================================================================
# MoA Consistency Validation
# =========================================================================

def moa_consistency_validation(per_feature_stats, treatment_stats):
    """Validate perturbation results against known MoA labels.

    Computes pairwise Pearson correlation of Cohen's d profiles between
    all treatments, then compares intra-MoA vs inter-MoA correlations.

    A good perturbation analysis should yield:
      - High intra-MoA correlation (same MoA -> similar perturbation profile)
      - Lower inter-MoA correlation (different MoA -> different profiles)

    Returns:
        dict with validation scores, or empty dict if insufficient data
    """
    print(f"\nMoA Consistency Validation...")

    # Build per-treatment Cohen's d profile matrix
    pivot = per_feature_stats.pivot_table(
        index=['compound', 'concentration'],
        columns='feature',
        values='cohens_d',
        aggfunc='first',
    ).fillna(0)

    if len(pivot) < 5:
        print("  Too few treatments for validation")
        return {}

    # Merge MoA labels
    moa_map = treatment_stats.set_index(
        ['compound', 'concentration'])['moa'].to_dict()
    pivot_moa = [moa_map.get(idx, 'unknown') for idx in pivot.index]

    # Filter to known MoA
    known_mask = np.array([m != 'unknown' for m in pivot_moa])
    if known_mask.sum() < 5:
        print("  Too few MoA-annotated treatments")
        return {}

    X = pivot.values[known_mask]
    labels = np.array(pivot_moa)[known_mask]

    n = len(X)
    moa_classes = sorted(set(labels))
    n_classes = len(moa_classes)

    if n_classes < 2:
        print("  Need >= 2 MoA classes")
        return {}

    print(f"  {n} treatments, {n_classes} MoA classes")

    # Standardize rows then compute Pearson correlation matrix
    X_std = X.copy()
    for i in range(n):
        row_std = X_std[i].std()
        if row_std > 0:
            X_std[i] = (X_std[i] - X_std[i].mean()) / row_std
        else:
            X_std[i] = 0

    corr_matrix = X_std @ X_std.T / X_std.shape[1]
    np.fill_diagonal(corr_matrix, 1.0)

    # Classify pairs as intra-MoA or inter-MoA
    intra_corrs = []
    inter_corrs = []
    per_class_intra = {c: [] for c in moa_classes}

    for i in range(n):
        for j in range(i + 1, n):
            r = corr_matrix[i, j]
            if labels[i] == labels[j]:
                intra_corrs.append(r)
                per_class_intra[labels[i]].append(r)
            else:
                inter_corrs.append(r)

    intra_corrs = np.array(intra_corrs)
    inter_corrs = np.array(inter_corrs)

    if len(intra_corrs) == 0 or len(inter_corrs) == 0:
        print("  Cannot compute intra/inter (too few pairs)")
        return {}

    intra_mean = float(np.mean(intra_corrs))
    intra_std = float(np.std(intra_corrs))
    inter_mean = float(np.mean(inter_corrs))
    inter_std = float(np.std(inter_corrs))

    separation = intra_mean - inter_mean
    pooled_std = np.sqrt((intra_std**2 + inter_std**2) / 2)
    separation_d = separation / pooled_std if pooled_std > 0 else 0.0

    # Wilcoxon rank-sum: is intra significantly > inter?
    try:
        w_stat, w_pvalue = stats.ranksums(intra_corrs, inter_corrs,
                                           alternative='greater')
    except Exception:
        w_stat, w_pvalue = 0.0, 1.0

    # Per-class summary
    class_records = []
    for c in moa_classes:
        vals = per_class_intra[c]
        if vals:
            class_records.append({
                'moa': c,
                'n_pairs': len(vals),
                'mean_intra_corr': float(np.mean(vals)),
                'std_intra_corr': float(np.std(vals)),
            })
    class_df = pd.DataFrame(class_records).sort_values(
        'mean_intra_corr', ascending=False)

    result = {
        'n_treatments': n,
        'n_moa_classes': n_classes,
        'n_intra_pairs': len(intra_corrs),
        'n_inter_pairs': len(inter_corrs),
        'intra_mean_corr': intra_mean,
        'intra_std_corr': intra_std,
        'inter_mean_corr': inter_mean,
        'inter_std_corr': inter_std,
        'separation': separation,
        'separation_cohens_d': float(separation_d),
        'wilcoxon_stat': float(w_stat),
        'wilcoxon_pvalue': float(w_pvalue),
        'class_summary': class_df,
        'corr_matrix': corr_matrix,
        'corr_labels': labels,
    }

    # Print
    print(f"  Intra-MoA correlation: {intra_mean:.4f} +/- {intra_std:.4f} "
          f"({len(intra_corrs)} pairs)")
    print(f"  Inter-MoA correlation: {inter_mean:.4f} +/- {inter_std:.4f} "
          f"({len(inter_corrs)} pairs)")
    print(f"  Separation (intra - inter): {separation:.4f}")
    print(f"  Separation Cohen's d: {separation_d:.3f}")
    print(f"  Wilcoxon p-value (intra > inter): {w_pvalue:.4e}")

    if separation > 0 and w_pvalue < 0.05:
        print(f"  -> VALID: perturbation profiles are consistent with MoA")
    elif separation > 0:
        print(f"  -> WEAK: trend toward consistency, but not significant")
    else:
        print(f"  -> NOT CONSISTENT: intra-MoA not higher than inter-MoA")

    print(f"\n  Per-class intra-MoA correlation:")
    for _, row in class_df.iterrows():
        print(f"    {row['moa']:<35s}  r={row['mean_intra_corr']:.4f} "
              f"+/- {row['std_intra_corr']:.4f}  ({row['n_pairs']} pairs)")

    return result


def save_results(treatment_stats, per_feature_stats, dose_response_df,
                 model_name="cellpose4", moa_validation=None):
    """Save all perturbation analysis results."""
    output_dir = config.biomarker_dir(model_name) / "perturbation"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Treatment summary
    path1 = output_dir / "treatment_perturbation_stats.csv"
    treatment_stats.to_csv(path1, index=False)
    print(f"\n  Saved: {path1} ({len(treatment_stats)} treatments)")

    # Per-feature detail
    path2 = output_dir / "per_feature_tests.csv"
    per_feature_stats.to_csv(path2, index=False)
    print(f"  Saved: {path2} ({len(per_feature_stats)} tests)")

    # Dose-response
    path3 = output_dir / "dose_response.csv"
    dose_response_df.to_csv(path3, index=False)
    print(f"  Saved: {path3} ({len(dose_response_df)} compound-feature pairs)")

    # Top perturbed treatments
    top = treatment_stats.nlargest(20, 'perturbation_score')
    path4 = output_dir / "top_perturbed_treatments.csv"
    top.to_csv(path4, index=False)
    print(f"  Saved: {path4}")

    # Monotonic dose-response features
    mono = dose_response_df[dose_response_df['monotonic']]
    if len(mono) > 0:
        path5 = output_dir / "monotonic_features.csv"
        mono.to_csv(path5, index=False)
        print(f"  Saved: {path5} ({len(mono)} monotonic features)")

    # MoA consistency validation
    if moa_validation:
        summary_rec = {
            'n_treatments': moa_validation.get('n_treatments', 0),
            'n_moa_classes': moa_validation.get('n_moa_classes', 0),
            'intra_mean_corr': moa_validation.get('intra_mean_corr', 0),
            'inter_mean_corr': moa_validation.get('inter_mean_corr', 0),
            'separation': moa_validation.get('separation', 0),
            'separation_cohens_d': moa_validation.get('separation_cohens_d', 0),
            'wilcoxon_pvalue': moa_validation.get('wilcoxon_pvalue', 1.0),
        }
        path6 = output_dir / "moa_validation_summary.csv"
        pd.DataFrame([summary_rec]).to_csv(path6, index=False)
        print(f"  Saved: {path6}")

        if 'class_summary' in moa_validation and len(moa_validation['class_summary']) > 0:
            path7 = output_dir / "moa_validation_per_class.csv"
            moa_validation['class_summary'].to_csv(path7, index=False)
            print(f"  Saved: {path7}")

        corr = moa_validation.get('corr_matrix')
        if corr is not None and corr.shape[0] <= 2000:
            labels = moa_validation.get('corr_labels', [])
            corr_df = pd.DataFrame(corr)
            corr_df.insert(0, 'moa', labels)
            path8 = output_dir / "treatment_correlation_matrix.csv"
            corr_df.to_csv(path8, index=False)
            print(f"  Saved: {path8} ({corr.shape[0]}x{corr.shape[1]})")


def run_perturbation_analysis(model_name="cellpose4", n_workers=8):
    """Run full perturbation analysis pipeline."""
    print("=" * 60)
    print(f"Perturbation Analysis [{model_name}]")
    print("=" * 60)

    field_df, dmso_df = load_profiles(model_name=model_name)

    # Guard: no DMSO controls
    if len(dmso_df) == 0:
        print("[ERROR] No DMSO controls. Run preprocess.py first, then re-run.")
        return

    # 1. Perturbation analysis
    treatment_stats, per_feature_stats = perturbation_analysis(
        field_df, dmso_df, n_workers=n_workers)

    if len(treatment_stats) == 0:
        print("[WARN] No treatments found. Skipping dose-response and MoA validation.")
        return

    # 2. Dose-response
    dose_response_df = dose_response_analysis(
        field_df, n_workers=n_workers)

    # 3. MoA consistency validation
    moa_val = moa_consistency_validation(per_feature_stats, treatment_stats)

    # 4. Save
    save_results(treatment_stats, per_feature_stats, dose_response_df,
                 model_name=model_name, moa_validation=moa_val)

    # 5. Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    active = treatment_stats[treatment_stats['perturbation_score'] > 0.05]
    print(f"  Total treatments: {len(treatment_stats)}")
    print(f"  Active (perturbation > 5%): {len(active)}")
    if len(active) > 0:
        print(f"\n  Active treatments by MoA:")
        moa_counts = active.groupby('moa').agg(
            n_treatments=('compound', 'count'),
            mean_score=('perturbation_score', 'mean'),
        ).sort_values('mean_score', ascending=False)
        print(moa_counts.to_string())

    return treatment_stats, per_feature_stats, dose_response_df, moa_val


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
        run_perturbation_analysis(model_name=m, n_workers=args.workers)
