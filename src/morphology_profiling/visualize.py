"""
visualize.py
============
Visualization outputs for the U2OS-Cell-Painting morphological profiling
pipeline.

Generates:
  1. UMAP scatter plot (field embeddings colored by MoA class / cluster)
  2. Heatmap: MoA classes x top biomarker features (mean z-score)
  3. Compound feature deviations vs DMSO (single 10 uM dose)
  4. Segmentation overlay examples (DNA channel + mask contours)
  5. Perturbation score distribution

Usage:
  python src/morphology_profiling/visualize.py --all
  python src/morphology_profiling/visualize.py --umap
  python src/morphology_profiling/visualize.py --heatmap
  python src/morphology_profiling/visualize.py --compound_deviation --compound Y-39983
  python src/morphology_profiling/visualize.py --overlay --plate P015080
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

# Color palette for MoA classes (config.MOA_CLASSES is a {label: n_compounds}
# dict on this dataset; DMSO is the negative control, not a MoA class).
MOA_COLORS = dict(zip(
    list(config.MOA_CLASSES.keys()),
    list(mcolors.TABLEAU_COLORS.values())[:len(config.MOA_CLASSES)]
))


def ensure_vis_dir(model_name="cellpose4"):
    d = config.vis_dir(model_name)
    d.mkdir(parents=True, exist_ok=True)
    return d


# =========================================================================
# 1. UMAP scatter plot
# =========================================================================

def plot_umap_moa(model_name="cellpose4"):
    """Field-level UMAP scatter plot colored by MoA class."""
    path = config.biomarker_dir(model_name) / "unsupervised" / "field_umap_embedding.csv"
    if not path.exists():
        print("  [SKIP] Field UMAP embedding not found. Run unsupervised.py first.")
        return

    df = pd.read_csv(path)
    out_dir = ensure_vis_dir(model_name)

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    moa_classes = sorted(df['moa'].unique())
    for moa in moa_classes:
        if moa == 'unknown':
            continue
        mask = df['moa'] == moa
        color = MOA_COLORS.get(moa, '#888888')
        ax.scatter(
            df.loc[mask, 'UMAP_1'], df.loc[mask, 'UMAP_2'],
            c=color, label=moa, s=20, alpha=0.6, edgecolors='white', linewidth=0.2,
        )

    # Unknown/other
    if 'unknown' in df['moa'].values:
        mask = df['moa'] == 'unknown'
        ax.scatter(
            df.loc[mask, 'UMAP_1'], df.loc[mask, 'UMAP_2'],
            c='lightgray', label='Unknown', s=10, alpha=0.4,
        )

    ax.set_xlabel('UMAP 1', fontsize=12)
    ax.set_ylabel('UMAP 2', fontsize=12)
    ax.set_title('U2OS-Cell-Painting Field Profiles (UMAP)', fontsize=14)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),
              fontsize=8, framealpha=0.8)
    plt.tight_layout()

    out_path = out_dir / "umap_moa.png"
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")

    # Also plot colored by cluster (cluster column added by unsupervised.py)
    if 'cluster' not in df.columns:
        return
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    cluster_ids = sorted(df['cluster'].unique())
    cmap = plt.cm.Set3
    for cid in cluster_ids:
        mask = df['cluster'] == cid
        label = f'Cluster {cid}' if cid >= 0 else 'Noise'
        color = cmap(cid / max(1, max(cluster_ids))) if cid >= 0 else 'gray'
        ax.scatter(
            df.loc[mask, 'UMAP_1'], df.loc[mask, 'UMAP_2'],
            c=[color], label=label, s=20, alpha=0.6, edgecolors='white', linewidth=0.2,
        )

    ax.set_xlabel('UMAP 1', fontsize=12)
    ax.set_ylabel('UMAP 2', fontsize=12)
    ax.set_title('U2OS-Cell-Painting Field Profiles '
                 f'(Agglomerative k={config.DEFAULT_N_CLUSTERS})', fontsize=14)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),
              fontsize=8, framealpha=0.8)
    plt.tight_layout()

    out_path = out_dir / "umap_clusters.png"
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")


# =========================================================================
# 2. Heatmap: MoA x features
# =========================================================================

def plot_moa_heatmap(model_name="cellpose4", top_k=20):
    """Heatmap of top biomarker features across MoA classes."""
    bm_path = config.biomarker_dir(model_name) / "attribution" / "biomarker_features.csv"
    treat_path = config.aggregation_dir(model_name) / "treatment_profiles.csv"

    if not bm_path.exists() or not treat_path.exists():
        print("  [SKIP] Biomarker features not found. Run feature_attribution.py first.")
        return

    bm_df = pd.read_csv(bm_path)
    treat_df = pd.read_csv(treat_path)

    # Get top features (union across MoA classes)
    all_features = bm_df['feature'].unique()[:top_k]

    if len(all_features) == 0:
        print("  [SKIP] No biomarker features found.")
        return

    # Build matrix: MoA classes x features (mean z-score)
    moa_classes = sorted(treat_df[treat_df['moa'] != 'unknown']['moa'].unique())

    # Filter to valid feature columns
    valid_feats = [f for f in all_features if f in treat_df.columns]

    matrix = []
    row_labels = []
    for moa in moa_classes:
        mask = treat_df['moa'] == moa
        if mask.sum() < 2:
            continue
        vals = treat_df.loc[mask, valid_feats].mean()
        matrix.append(vals.values)
        row_labels.append(moa)

    if len(matrix) == 0:
        print("  [SKIP] Not enough MoA classes.")
        return

    matrix = np.array(matrix)

    # Plot
    out_dir = ensure_vis_dir(model_name)
    fig_height = max(6, len(row_labels) * 0.4)
    fig_width = max(8, len(valid_feats) * 0.5)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    sns.heatmap(
        matrix,
        xticklabels=[f.split('_')[-1] if len(f) > 30 else f for f in valid_feats],
        yticklabels=row_labels,
        cmap='RdBu_r',
        center=0,
        ax=ax,
        cbar_kws={'label': 'Mean Z-score', 'shrink': 0.8},
        linewidths=0.5,
    )
    ax.set_title(f'Biomarker Feature Heatmap ({len(row_labels)} MoA x {len(valid_feats)} features)',
                 fontsize=12)
    ax.set_xlabel('Features', fontsize=10)
    ax.set_ylabel('MoA Class', fontsize=10)
    plt.xticks(rotation=45, ha='right', fontsize=7)
    plt.yticks(fontsize=9)
    plt.tight_layout()

    out_path = out_dir / "moa_heatmap.png"
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")


# =========================================================================
# 3. Compound feature deviations vs DMSO (single 10 uM dose)
# =========================================================================

def plot_compound_deviation(model_name="cellpose4", compound=None):
    """Bar plot of compound feature deviations against the DMSO baseline.

    U2OS-Cell-Painting is dosed at a single concentration (10 uM), so no
    dose-response curve exists; instead we show the strongest per-compound
    deviations (z-scored treatment profile vs DMSO).

    If compound is given: its top-4 deviating features.
    Otherwise: the 6 most perturbed compounds, one subplot each.
    """
    treat_path = config.aggregation_dir(model_name) / "treatment_profiles.csv"
    if not treat_path.exists():
        print("  [SKIP] Treatment profiles not found.")
        return

    treat_df = pd.read_csv(treat_path)

    # Get feature columns
    meta_cols = {'compound', 'concentration', 'moa', 'is_dmso', 'well',
                 'plate', 'n_cells', 'n_cells_total', 'n_fields'}
    feat_cols = [c for c in treat_df.columns if c not in meta_cols]

    out_dir = ensure_vis_dir(model_name)

    if compound:
        cpd_data = treat_df[treat_df['compound'].str.lower() == compound.lower()]
        if len(cpd_data) == 0:
            print(f"  Compound '{compound}' not found.")
            return

        # Deviation of the compound's mean profile from the DMSO mean
        dmso_mean = treat_df[treat_df['is_dmso']][feat_cols].mean()
        dev = cpd_data[feat_cols].mean() - dmso_mean
        top_feats = dev.abs().nlargest(4).index.tolist()

        fig, axes = plt.subplots(1, len(top_feats),
                                 figsize=(4 * len(top_feats), 4), squeeze=False)
        for ax, feat in zip(axes[0], top_feats):
            vals = cpd_data[feat].values
            ax.bar(range(len(vals)), vals, color='steelblue')
            ax.axhline(dmso_mean[feat], color='gray', linestyle='--', linewidth=1)
            ax.set_ylabel(feat, fontsize=8)
            ax.set_title(feat, fontsize=9)

        moa = cpd_data['moa'].iloc[0]
        fig.suptitle(f'{compound} (MoA: {moa}) vs DMSO baseline', fontsize=13)
        plt.tight_layout()

        cpd_safe = compound.replace(' ', '_')
        out_path = out_dir / f"compound_deviation_{cpd_safe}.png"
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {out_path}")
    else:
        # Most perturbed compounds (proxy: mean |deviation| from DMSO mean)
        non_dmso = treat_df[~treat_df['is_dmso']]
        dmso_mean = treat_df[treat_df['is_dmso']][feat_cols].mean()
        cpd_dev = (non_dmso.groupby('compound')[feat_cols].mean()
                   .sub(dmso_mean).abs().mean(axis=1))
        top_compounds = cpd_dev.nlargest(6).index.tolist()

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        for ax, cpd in zip(axes.flat, top_compounds):
            cpd_data = non_dmso[non_dmso['compound'] == cpd]
            if len(cpd_data) == 0:
                continue

            dev = cpd_data[feat_cols].mean() - dmso_mean
            best_feat = dev.abs().idxmax()

            vals = cpd_data[best_feat].values
            ax.bar(range(len(vals)), vals, color='steelblue')
            ax.axhline(dmso_mean[best_feat], color='gray', linestyle='--',
                       linewidth=1)
            moa = cpd_data['moa'].iloc[0]
            ax.set_title(f'{cpd}\n(MoA: {moa})', fontsize=9)
            ax.set_ylabel(best_feat[:40], fontsize=7)

        fig.suptitle('Top Perturbed Compounds vs DMSO (10 uM)', fontsize=14)
        plt.tight_layout()

        out_path = out_dir / "compound_deviation_top.png"
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {out_path}")


# =========================================================================
# 4. Segmentation overlay
# =========================================================================

def plot_segmentation_overlay(model_name="cellpose4", plate_name=None, n_examples=4):
    """Plot DNA-channel images with segmentation mask contours overlaid."""
    from tifffile import imread
    from skimage.segmentation import find_boundaries

    if plate_name is None:
        # Use first available plate
        if not config.IMAGES_DIR.exists():
            print("  [SKIP] No images found.")
            return
        plates = [d.name for d in config.IMAGES_DIR.iterdir() if d.is_dir()]
        if not plates:
            print("  [SKIP] No plates found.")
            return
        plate_name = sorted(plates)[0]

    nuc_dir = config.IMAGES_DIR / plate_name / config.NUCLEUS_CHANNEL
    mask_dir = config.masks_dir(model_name) / plate_name

    if not nuc_dir.exists() or not mask_dir.exists():
        print(f"  [SKIP] Images/masks not found for {plate_name}")
        return

    nuc_files = sorted(nuc_dir.glob("*.tif"))[:n_examples]
    if not nuc_files:
        print(f"  [SKIP] No {config.NUCLEUS_CHANNEL} images found.")
        return

    out_dir = ensure_vis_dir(model_name)
    fig, axes = plt.subplots(2, n_examples, figsize=(4 * n_examples, 8))

    for idx, nuc_path in enumerate(nuc_files):
        stem = nuc_path.stem
        mask_path = mask_dir / f"{stem}_mask.tif"

        # Load
        nuc = imread(str(nuc_path))
        if nuc.ndim > 2:
            nuc = nuc[..., 0]

        if mask_path.exists():
            mask = imread(str(mask_path))
            boundaries = find_boundaries(mask, mode='outer')
        else:
            boundaries = None

        # DNA only
        ax = axes[0, idx]
        ax.imshow(nuc, cmap='gray')
        ax.set_title(f'{config.NUCLEUS_CHANNEL}: {stem}', fontsize=8)
        ax.axis('off')

        # DNA + overlay
        ax = axes[1, idx]
        ax.imshow(nuc, cmap='gray')
        if boundaries is not None:
            overlay = np.zeros((*nuc.shape, 3), dtype=np.float64)
            overlay[boundaries] = [1, 0, 0]  # red contours
            ax.imshow(overlay, alpha=0.6)
            n_cells = mask.max()
            ax.set_title(f'Segmentation ({n_cells} cells)', fontsize=8)
        else:
            ax.set_title('No mask', fontsize=8)
        ax.axis('off')

    fig.suptitle(f'Segmentation Overlay: {plate_name}', fontsize=14)
    plt.tight_layout()

    out_path = out_dir / f"overlay_{plate_name}.png"
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")


# =========================================================================
# 5. Perturbation score distribution
# =========================================================================

def plot_perturbation_distribution(model_name="cellpose4"):
    """Plot distribution of perturbation scores across compounds and MoA classes."""
    pert_path = config.biomarker_dir(model_name) / "perturbation" / "treatment_perturbation_stats.csv"
    if not pert_path.exists():
        print("  [SKIP] Perturbation stats not found.")
        return

    df = pd.read_csv(pert_path)
    out_dir = ensure_vis_dir(model_name)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram by MoA
    ax = axes[0]
    moa_classes = sorted(df['moa'].unique())
    for moa in moa_classes:
        mask = df['moa'] == moa
        color = MOA_COLORS.get(moa, '#888888')
        ax.hist(df.loc[mask, 'perturbation_score'], bins=20,
                alpha=0.5, label=moa, color=color, density=True)
    ax.set_xlabel('Perturbation Score')
    ax.set_ylabel('Density')
    ax.set_title('Perturbation Score Distribution by MoA')
    ax.legend(fontsize=7, loc='upper right')

    # Box plot by MoA
    ax = axes[1]
    moa_order = df.groupby('moa')['perturbation_score'].median().sort_values(ascending=False).index.tolist()
    data_by_moa = [df[df['moa'] == moa]['perturbation_score'].values for moa in moa_order]

    bp = ax.boxplot(data_by_moa, tick_labels=[m[:18] for m in moa_order],
                    patch_artist=True, showfliers=True)
    for i, moa in enumerate(moa_order):
        color = MOA_COLORS.get(moa, '#888888')
        bp['boxes'][i].set_facecolor(color)
        bp['boxes'][i].set_alpha(0.6)

    ax.set_ylabel('Perturbation Score')
    ax.set_title('Perturbation Score by MoA (median-sorted)')
    plt.xticks(rotation=45, ha='right', fontsize=8)

    plt.tight_layout()
    out_path = out_dir / "perturbation_distribution.png"
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate visualization outputs")
    parser.add_argument("--models", type=str, nargs="+", default=None,
                        help="Models (e.g. cellpose4 cellpose3). "
                             "Use 'all' for all 5 models.")
    parser.add_argument("--all", action="store_true", help="Generate all plots")
    parser.add_argument("--umap", action="store_true", help="UMAP scatter plots")
    parser.add_argument("--heatmap", action="store_true", help="MoA heatmap")
    parser.add_argument("--compound_deviation", action="store_true",
                        help="Compound deviation vs DMSO plots")
    parser.add_argument("--compound", type=str, default=None,
                        help="Specific compound for deviation plots")
    parser.add_argument("--overlay", action="store_true",
                        help="Segmentation overlay")
    parser.add_argument("--plate", type=str, default=None,
                        help="Plate for overlay")
    parser.add_argument("--perturbation", action="store_true",
                        help="Perturbation distribution")
    args = parser.parse_args()

    config.ensure_dirs()

    # Determine models
    if args.models is None:
        models_to_run = ["cellpose4"]
    elif "all" in args.models:
        models_to_run = config.MODELS
    else:
        models_to_run = args.models

    for model_name in models_to_run:
        print(f"\n{'='*60}")
        print(f"Visualizations for: {model_name}")
        print(f"{'='*60}")

        if args.all:
            plot_umap_moa(model_name=model_name)
            plot_moa_heatmap(model_name=model_name)
            plot_compound_deviation(model_name=model_name, compound=args.compound)
            plot_segmentation_overlay(model_name=model_name, plate_name=args.plate)
            plot_perturbation_distribution(model_name=model_name)
            continue

        if args.umap:
            plot_umap_moa(model_name=model_name)
        if args.heatmap:
            plot_moa_heatmap(model_name=model_name)
        if args.compound_deviation:
            plot_compound_deviation(model_name=model_name, compound=args.compound)
        if args.overlay:
            plot_segmentation_overlay(model_name=model_name, plate_name=args.plate)
        if args.perturbation:
            plot_perturbation_distribution(model_name=model_name)

    if not any([args.umap, args.heatmap, args.compound_deviation, args.overlay,
                args.perturbation, args.all]):
        print("No action specified. Use --all or specific flags.")


if __name__ == "__main__":
    main()
