"""
TNBC-MIBI CellSAM Evaluation Script
====================================
Segmentation → AP evaluation (AP@0.5/0.75/0.9) → Expression extraction + comparison → FlowSOM clustering
Usage: python TNBC-MIBI/analysis/eval_cellsam_TNBC.py --gpu
"""
import sys, argparse, numpy as np, pandas as pd, tifffile, time, traceback
from pathlib import Path

_script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_script_dir.parent.parent))
from tnbc_eval_common import (
    DATA, MASK_DIR, CELLDATA_CSV, COMPOSITE_DIR,
    get_valid_pids, format_for_sam,
    compute_ap_single, extract_expression_from_mask,
    compare_expression, run_clustering, compute_gt_profiles,
    save_txt, save_excel
)


def segment_with_cellsam(image, device='cuda'):
    """
    Segment a single image with CellSAM.
    image: (H, W, 3) uint8 — output of format_for_sam
    Returns pred_mask: ndarray (H, W) uint16
    """
    try:
        from cellSAM import cellsam_pipeline
    except ImportError:
        print("Error: CellSAM is not installed. Please run pip install cellSAM")
        return None

    try:
        mask = cellsam_pipeline(image, use_wsi=False,
                                low_contrast_enhancement=False,
                                gauge_cell_size=False)
        if mask is None:
            return np.zeros(image.shape[:2], dtype=np.uint16)
        return np.asarray(mask, dtype=np.uint16)
    except Exception as e:
        print(f"    CellSAM failed: {e}")
        traceback.print_exc()
        return np.zeros(image.shape[:2], dtype=np.uint16)


def main():
    parser = argparse.ArgumentParser(description='TNBC-MIBI CellSAM Evaluation')
    parser.add_argument('--gpu', action='store_true', default=True)
    parser.add_argument('--patients', nargs='+', type=int, default=None)
    parser.add_argument('--phase', type=str, default='all',
                        choices=['all','segment','express','cluster'],
                        help="Pipeline phase: all=all, segment=seg only, express=expression only, cluster=cluster only")
    parser.add_argument('--model_name', type=str, default='cellsam')
    parser.add_argument('--iou_threshold', type=float, default=0.3,
                        help='IoU matching threshold (default 0.3)')
    args = parser.parse_args()
    model_name = args.model_name

    out_dir = DATA / 'eval_results' / model_name
    mask_dir = out_dir / 'masks'
    mask_dir.mkdir(parents=True, exist_ok=True)

    cell_df = pd.read_csv(CELLDATA_CSV)
    valid_pids = get_valid_pids()
    if args.patients:
        valid_pids = [p for p in args.patients if p in valid_pids]
    print(f"CellSAM Evaluation: {len(valid_pids)} patients, GPU={args.gpu}")
    print(f"Output directory: {out_dir}")

    # ---- 1. Segmentation ----
    if args.phase in ('all', 'segment'):
        print(f"\n{'='*60}")
        print("Phase 1: Segmentation")
        print(f"{'='*60}")

        ap_records = []
        for pid in valid_pids:
            t_start = time.time()
            print(f"\n  Point {pid}: ", end='', flush=True)

            # Composite RGB: [R=cancer mem, G=immune mem, B=nucleus] → nuclear signal moved to G channel (CellSAM expected)
            composite = tifffile.imread(str(COMPOSITE_DIR / f'Point{pid:02d}_composite.tif'))
            img = composite.copy()
            img[:,:,1] = composite[:,:,2]  # G ← nuclear
            img[:,:,2] = composite[:,:,1]  # B ← immune membrane
            img_sam = format_for_sam(img)
            pred_mask = segment_with_cellsam(img_sam, device='cuda' if args.gpu else 'cpu')

            gt_path = MASK_DIR / f'p{pid}_labeledcellData.tiff'
            gt_mask = tifffile.imread(str(gt_path)).astype(np.uint16)

            if pred_mask is None:
                print("  CellSAM loading failed, skipping rest")
                break

            out_path = mask_dir / f'p{pid}_pred.tif'
            tifffile.imwrite(str(out_path), pred_mask, compression='deflate')

            ap05, ap075, ap09 = compute_ap_single(gt_mask, pred_mask)
            elapsed = time.time() - t_start
            print(f"AP@0.5={ap05:.4f}  AP@0.75={ap075:.4f}  AP@0.9={ap09:.4f}  ({elapsed:.0f}s)")
            ap_records.append({'Patient': pid, 'AP@0.5': ap05,
                               'AP@0.75': ap075, 'AP@0.9': ap09})

        if ap_records:
            df_ap = pd.DataFrame(ap_records)
            df_ap.loc['mean'] = df_ap.mean(numeric_only=True)
            df_ap.loc['std'] = df_ap.std(numeric_only=True)
            df_ap.loc['mean', 'Patient'] = 'Overall Mean'
            df_ap.loc['std', 'Patient'] = 'Overall Std'

            lines = ["TNBC-MIBI CellSAM - AP Results",
                     f"{'Patient':>8}  {'AP@0.5':>8}  {'AP@0.75':>8}  {'AP@0.9':>8}",
                     '-' * 40]
            for _, r in df_ap.iterrows():
                if r['Patient'] in ('Overall Mean', 'Overall Std'):
                    lines.append(f"{r['Patient']:>8}  {r['AP@0.5']:>8.4f}  {r['AP@0.75']:>8.4f}  {r['AP@0.9']:>8.4f}")
                else:
                    lines.append(f"{int(r['Patient']):>8d}  {r['AP@0.5']:>8.4f}  {r['AP@0.75']:>8.4f}  {r['AP@0.9']:>8.4f}")
            save_txt(out_dir / 'ap_results.txt', lines)
            save_excel(out_dir / 'ap_per_image.xlsx', {'AP_Scores': df_ap})
    else:
        print("Skipping segmentation phase")
    
    # ---- 2. Expression extraction (express) ----
    expr_dir = out_dir / 'pred_expr'
    all_pred_expr = {}
    all_matched = {}
    
    if args.phase in ('all', 'express'):
        print(f"\n{'='*60}")
        print("Phase 2: Expression Extraction and Comparison")
        print(f"{'='*60}")
    
        all_comp_records = []
        for pid in valid_pids:
            print(f"\n  Point {pid}")
            gt_path = MASK_DIR / f'p{pid}_labeledcellData.tiff'
            pred_path = mask_dir / f'p{pid}_pred.tif'
            if not pred_path.exists():
                continue
    
            df_gt, n_gt = extract_expression_from_mask(gt_path, pid)
            df_pred, n_pred = extract_expression_from_mask(pred_path, pid)
            all_pred_expr[pid] = df_pred
            print(f"    GT: {n_gt}, Pred: {n_pred}", flush=True)
    
            # Save to disk (for --phase cluster use)
            expr_dir.mkdir(parents=True, exist_ok=True)
            df_pred.to_parquet(expr_dir / f'p{pid}.parquet')
    
            print(f"    Running compare_expression...", flush=True)
    
            comp, matched_df = compare_expression(gt_path, pred_path, pid, cell_df, iou_thresh=args.iou_threshold)
            all_matched[pid] = matched_df
            matched_df.to_parquet(expr_dir / f'p{pid}_matched.parquet')
            n_matched = len(matched_df)
            print(f"    Matched (IoU>{args.iou_threshold}): {n_matched}")
            if n_matched > 0:
                diffs = [v['median_abs_diff'] for v in comp.values()
                         if not np.isnan(v['median_abs_diff'])]
                if diffs:
                    print(f"    Median diff: {np.median(diffs):.4f}")
    
            all_comp_records.append({
                'Patient': pid, 'n_gt_cells': n_gt, 'n_pred_cells': n_pred,
                'n_matched': n_matched,
            })
    
        if all_comp_records:
            df_comp = pd.DataFrame(all_comp_records)
            save_txt(out_dir / 'expression_comparison.txt',
                     [f"Patient {r['Patient']}: GT={r['n_gt_cells']}, "
                      f"Pred={r['n_pred_cells']}, Matched={r['n_matched']}"
                      for r in all_comp_records])
            save_excel(out_dir / 'expression_comparison.xlsx',
                       {'Cell_Counts': df_comp})
    
    # ---- 3. FlowSOM clustering (cluster) ----
    if args.phase in ('all', 'cluster'):
        print(f"\n{'='*60}")
        print("Phase 3: FlowSOM Clustering Analysis (abundance-level)")
        print(f"{'='*60}")

        gt_profiles, gt_counts = compute_gt_profiles(cell_df)

        all_expr = []
        for pid in valid_pids:
            if pid in all_pred_expr:
                all_expr.append(all_pred_expr[pid])
            else:
                pf = expr_dir / f'p{pid}.parquet'
                if pf.exists():
                    all_expr.append(pd.read_parquet(pf))

        if all_expr:
            expr_concat = pd.concat(all_expr, ignore_index=True)
            result_df, per_patient_df = run_clustering(
                expr_concat, cell_df, gt_profiles, gt_counts, 'CellSAM', out_dir, iou_thresh=args.iou_threshold)

            lines = ["TNBC-MIBI CellSAM - FlowSOM Cell-level Evaluation",
                     f"{'Level':6s} {'N_GT':>6s} {'N_Matched':>10s} {'N_Valid':>8s} {'N_Correct':>10s} {'Accuracy':>9s} {'Macro_F1':>9s}",
                     '-' * 62]
            for _, r in result_df.iterrows():
                def _fmt(v, w):
                    if isinstance(v, str):
                        return ' ' * w
                    return f"{v:>{w}.4f}" if isinstance(v, float) else f"{int(v):>{w}d}"
                lines.append(f"{str(r['level']):6s} {_fmt(r['n_gt'], 6)} {_fmt(r['n_matched'], 10)} {_fmt(r['n_valid'], 8)} {_fmt(r['n_correct'], 10)} {_fmt(r['accuracy'], 9)} {_fmt(r['macro_f1'], 9)}")
            save_txt(out_dir / 'clustering_results.txt', lines)

            save_excel(out_dir / 'clustering_results.xlsx',
                       {'Clustering': result_df, 'PerPatient': per_patient_df})

    print(f"\nDone! Results saved to {out_dir}")


if __name__ == '__main__':
    main()
