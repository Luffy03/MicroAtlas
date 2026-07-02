"""
Batch Counting: Run cell counting evaluation on all datasets under counting/BBBC
================================================================================

Usage examples:
  # Default: run all five models (cellpose4 + microatlas + cellpose3 + microsam + cellsam) on all datasets
  python counting/batch_counting.py

  # Run cellpose4 only on specified datasets
  python counting/batch_counting.py --datasets BBBC041 --models cellpose4

  # Run cellpose3 only
  python counting/batch_counting.py --models cellpose3

  # CPU mode + dry run only
  python counting/batch_counting.py --cpu --dry-run
"""

import os
import sys
import csv
import time
import tempfile
import argparse
import numpy as np
from pathlib import Path

# ============================================================================
# Add counting directory to sys.path to ensure eval_*_counting modules can be imported
# ============================================================================
COUNTING_DIR = Path(__file__).resolve().parent  # d:\code\Cellpose\counting
if str(COUNTING_DIR) not in sys.path:
    sys.path.insert(0, str(COUNTING_DIR))

import counting_utils as cu


# ============================================================================
# Dataset definitions
# ============================================================================
BBBC_ROOT = COUNTING_DIR / "BBBC"

DATASET_CONFIGS = {
    "BBBC001": {
        "image_dir": str(BBBC_ROOT / "BBBC001" / "BBBC001_v1_images_tif" / "human_ht29_colon_cancer_1_images"),
        "counts_file": str(BBBC_ROOT / "BBBC001" / "BBBC001_v1_counts.txt"),
        "n_images": 6,
    },
    "BBBC039": {
        "image_dir": str(BBBC_ROOT / "BBBC039" / "images"),
        "counts_file": str(BBBC_ROOT / "BBBC039" / "BBBC039_v1_counts.txt"),
        "n_images": 200,
    },
    "BBBC041": {
        "image_dir": str(BBBC_ROOT / "BBBC041" / "malaria" / "images"),
        "counts_file": str(BBBC_ROOT / "BBBC041" / "BBBC041_v1_counts.txt"),
        "n_images": 1328,
    },
}


def run_single_task(model, dataset_name, output_root, **kwargs):
    """Execute a single (model, dataset) counting evaluation task."""
    cfg = DATASET_CONFIGS[dataset_name]
    image_dir = cfg["image_dir"]
    counts_file = cfg["counts_file"]
    output_dir = str(Path(output_root) / f"{dataset_name}_{model}_counting")

    if not os.path.isdir(image_dir):
        print(f"  [Skip] Image directory not found: {image_dir}")
        return False, 0
    if not os.path.isfile(counts_file):
        print(f"  [Skip] Counts file not found: {counts_file}")
        return False, 0

    os.makedirs(output_dir, exist_ok=True)

    start_t = time.time()
    result_data = None
    try:
        if model == "cellpose4":
            import eval_cellpose4_counting
            result_data = eval_cellpose4_counting.run_counting_evaluation(
                dataset_name=dataset_name,
                image_dir=image_dir,
                counts_file=counts_file,
                output_dir=output_dir,
                use_gpu=kwargs.get("use_gpu", True),
                diameter=kwargs.get("diameter", 30.),
            )
        elif model == "cellpose3":
            import eval_cellpose3_counting
            result_data = eval_cellpose3_counting.run_counting_evaluation(
                dataset_name=dataset_name,
                image_dir=image_dir,
                counts_file=counts_file,
                output_dir=output_dir,
                use_gpu=kwargs.get("use_gpu", True),
                diameter=kwargs.get("diameter", 0.),
            )
        elif model == "microatlas":
            import eval_microatlas_counting
            result_data = eval_microatlas_counting.run_counting_evaluation(
                dataset_name=dataset_name,
                image_dir=image_dir,
                counts_file=counts_file,
                output_dir=output_dir,
                use_gpu=kwargs.get("use_gpu", True),
                diameter=kwargs.get("diameter", 30.),
            )
        elif model == "microsam":
            import eval_microsam_counting
            result_data = eval_microsam_counting.run_counting_evaluation(
                dataset_name=dataset_name,
                image_dir=image_dir,
                counts_file=counts_file,
                output_dir=output_dir,
                use_gpu=kwargs.get("use_gpu", True),
                model_type=kwargs.get("model_type", "vit_l_lm"),
            )
        elif model == "cellsam":
            import eval_cellsam_counting
            result_data = eval_cellsam_counting.run_counting_evaluation(
                dataset_name=dataset_name,
                image_dir=image_dir,
                counts_file=counts_file,
                output_dir=output_dir,
                use_gpu=kwargs.get("use_gpu", True),
            )
        else:
            print(f"  [Error] Unknown model: {model}")
            return False, 0, None
    except Exception as e:
        import traceback
        print(f"  [Error] Task execution failed: {e}")
        traceback.print_exc()
        return False, time.time() - start_t, None

    elapsed = time.time() - start_t
    if result_data is None:
        print(f"  [Error] {model} on {dataset_name}: no results returned")
        return False, elapsed, {}
    metrics = result_data.get("metrics", {})
    return True, elapsed, metrics


def _read_per_image_csv(output_root, model_name, dataset_name):
    """Read gt and pred counts from per-image CSV.

    Returns
    -------
    (np.ndarray, np.ndarray) or (None, None)
        (gt_counts, pred_counts) or (None, None) if file does not exist
    """
    csv_path = (Path(output_root) /
                f"{dataset_name}_{model_name}_counting" /
                "results_per_image" /
                f"{model_name}_{dataset_name}.csv")
    if not csv_path.exists():
        return None, None
    gt_list, pred_list = [], []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            gt_list.append(float(row['gt_count']))
            pred_list.append(float(row['pred_count']))
    return np.array(gt_list), np.array(pred_list)


def save_excel_summary(results, output_root):
    """Save summary results to an Excel file.

    Format: one sheet per model + Summary sheet
    - Per model sheet: Dataset | MAE | MAE_Std | MAE_CI95 | RMSE | R² | Pearson r | MPE(%) | MPE_Std | MPE_CI95 | N
    - Summary sheet: Metric × Model cross table (main metrics only)
    """
    try:
        import openpyxl
        from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
        from openpyxl.utils import get_column_letter
    except ImportError:
        print("  [Skip Excel] openpyxl not installed (pip install openpyxl)")
        return

    # Group results by model
    model_order = ["cellpose4", "cellpose3", "microatlas", "microsam", "cellsam"]
    model_data = {m: {} for m in model_order}  # model -> {dataset: metrics_dict}
    dataset_order = ["BBBC001", "BBBC039", "BBBC041"]

    for r in results:
        m = r["model"]
        d = r["dataset"]
        if m in model_data:
            model_data[m][d] = r["metrics"]

    # Extended metrics: read from per-image CSV and compute Std/CI95
    for m in model_order:
        for d in list(model_data[m].keys()):
            gt_arr, pred_arr = _read_per_image_csv(output_root, m, d)
            if gt_arr is not None:
                try:
                    std_ci95 = cu.compute_std_ci95(gt_arr, pred_arr)
                    model_data[m][d].update(std_ci95)
                except Exception:
                    pass  # Silently skip calculation failures

    # Style definitions
    header_fill = PatternFill(start_color="D9E1F2", end_color="D9E1F2", fill_type="solid")
    header_font = Font(bold=True, size=11)
    best_fill = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")
    overall_fill = PatternFill(start_color="E2EFDA", end_color="E2EFDA", fill_type="solid")
    center_align = Alignment(horizontal="center", vertical="center")
    thin_border = Border(
        left=Side(style="thin"),
        right=Side(style="thin"),
        top=Side(style="thin"),
        bottom=Side(style="thin"),
    )

    xlsx_path = Path(output_root) / "results_summary.xlsx"
    wb = openpyxl.Workbook()
    # Remove default sheet
    wb.remove(wb.active)

    # ---- Metric columns for each model sheet (including Std/CI95) ----
    metric_keys = ["MAE", "MAE_std", "MAE_ci95", "RMSE", "R2", "Pearson_r", "MPE", "MPE_std", "MPE_ci95"]
    metric_labels = ["MAE", "MAE_Std", "MAE_CI95", "RMSE", "R²", "Pearson r", "MPE(%)", "MPE_Std", "MPE_CI95"]

    # ---- Summary sheet with main metrics and Std/CI95 ----
    summary_metric_keys = ["MAE", "MAE_std", "MAE_ci95", "RMSE", "R2", "Pearson_r", "MPE", "MPE_std", "MPE_ci95"]
    summary_metric_labels = ["MAE", "MAE_Std", "MAE_CI95", "RMSE", "R²", "Pearson r", "MPE(%)", "MPE_Std", "MPE_CI95"]

    # ---- One sheet per model ----
    for model_name in model_order:
        m_data = model_data[model_name]
        ds_names = [d for d in dataset_order if d in m_data]

        ws = wb.create_sheet(title=model_name)

        # Header
        headers = ["Dataset"] + metric_labels + ["N"]
        for col_idx, h in enumerate(headers, 1):
            cell = ws.cell(row=1, column=col_idx, value=h)
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = center_align
            cell.border = thin_border

        # Data rows
        for row_idx, ds_name in enumerate(ds_names, 2):
            met = m_data[ds_name]
            n = met.get("n_images", "N/A")
            values = [met.get(k, "N/A") for k in metric_keys]

            # Dataset name
            cell = ws.cell(row=row_idx, column=1, value=ds_name)
            cell.border = thin_border
            cell.alignment = center_align

            for col_idx, (k, v) in enumerate(zip(metric_keys, values), 2):
                cell = ws.cell(row=row_idx, column=col_idx)
                if isinstance(v, float):
                    # MPE-related columns use 2 decimal places, others use 4
                    if "MPE" in k:
                        cell.value = round(v, 2)
                        cell.number_format = "0.00"
                    else:
                        cell.value = round(v, 4)
                        cell.number_format = "0.0000"
                else:
                    cell.value = v
                cell.border = thin_border
                cell.alignment = center_align

            # N count
            cell = ws.cell(row=row_idx, column=len(headers), value=n)
            cell.border = thin_border
            cell.alignment = center_align

        # ---- Overall row ----
        last_row = len(ds_names) + 2
        overall_values = {}
        for k_idx, k in enumerate(metric_keys):
            if k in ("MAE_std", "MAE_ci95", "MPE_std", "MPE_ci95"):
                # Std/CI95: pooled per-image data across all images
                all_gt, all_pred = [], []
                for ds_name in ds_names:
                    gt_arr, pred_arr = _read_per_image_csv(output_root, model_name, ds_name)
                    if gt_arr is not None:
                        all_gt.extend(gt_arr.tolist())
                        all_pred.extend(pred_arr.tolist())
                if all_gt:
                    try:
                        overall_std = cu.compute_std_ci95(np.array(all_gt), np.array(all_pred))
                        overall_values[k] = overall_std.get(k, "N/A")
                    except Exception:
                        overall_values[k] = "N/A"
                else:
                    overall_values[k] = "N/A"
            else:
                # Main metrics: weighted average
                weighted_sum = 0.0
                total_weight = 0
                for ds_name in ds_names:
                    v = m_data[ds_name].get(k, None)
                    w = m_data[ds_name].get("n_images", 1)
                    if isinstance(v, (int, float)) and isinstance(w, (int, float)) and w > 0:
                        weighted_sum += v * w
                        total_weight += w
                overall_values[k] = weighted_sum / total_weight if total_weight > 0 else "N/A"

        cell = ws.cell(row=last_row, column=1, value="Overall")
        cell.fill = overall_fill
        cell.font = Font(bold=True)
        cell.border = thin_border
        cell.alignment = center_align

        for col_idx, (k, v) in enumerate(zip(metric_keys, [overall_values[k] for k in metric_keys]), 2):
            cell = ws.cell(row=last_row, column=col_idx)
            cell.fill = overall_fill
            cell.border = thin_border
            cell.alignment = center_align
            if isinstance(v, float):
                if "MPE" in k:
                    cell.value = round(v, 2)
                    cell.number_format = "0.00"
                else:
                    cell.value = round(v, 4)
                    cell.number_format = "0.0000"
            else:
                cell.value = v

        # N count for Overall
        tot_n = sum(m_data[ds].get("n_images", 0) for ds in ds_names)
        cell = ws.cell(row=last_row, column=len(headers), value=tot_n)
        cell.fill = overall_fill
        cell.border = thin_border
        cell.alignment = center_align

        # Mark best values for each main metric (skip Std/CI95 columns)
        no_highlight_keys = {"MAE_std", "MAE_ci95", "MPE_std", "MPE_ci95"}
        for k_idx, k in enumerate(metric_keys, 2):
            if k in no_highlight_keys:
                continue
            col_vals = []
            col_rows = []
            for row_idx, ds_name in enumerate(ds_names, 2):
                v = m_data[ds_name].get(k, None)
                if isinstance(v, (int, float)):
                    col_vals.append(v)
                    col_rows.append(row_idx)

            if len(col_vals) == 0:
                continue

            # MAE, RMSE, MPE -> lower is better (min)
            # R2, Pearson_r -> higher is better (max)
            if k in ("MAE", "RMSE", "MPE"):
                best_val = min(col_vals)
            else:
                best_val = max(col_vals)

            for row_idx, v in zip(col_rows, col_vals):
                if abs(v - best_val) < 1e-10:
                    cell = ws.cell(row=row_idx, column=k_idx + 1)
                    cell.fill = best_fill

        # Freeze header row
        ws.freeze_panes = "A2"

        # Auto-fit column widths
        for col_idx in range(1, len(headers) + 1):
            col_letter = get_column_letter(col_idx)
            max_len = len(str(headers[col_idx - 1]))
            for row_idx in range(2, last_row + 1):
                val = ws.cell(row=row_idx, column=col_idx).value
                if val is not None:
                    max_len = max(max_len, len(str(val)))
            ws.column_dimensions[col_letter].width = max_len + 4

    # ---- Summary sheet (cross-model comparison, including Std/CI95) ----
    ws_summary = wb.create_sheet(title="Summary", index=0)
    no_highlight_summary = {"MAE_std", "MAE_ci95", "MPE_std", "MPE_ci95"}

    for m_idx, metric_label in enumerate(summary_metric_labels):
        metric_key = summary_metric_keys[m_idx]
        is_std_ci95 = metric_key in no_highlight_summary
        start_row = m_idx * (len(dataset_order) + 3) + 1

        # Metric title
        cell = ws_summary.cell(row=start_row, column=1, value=metric_label)
        cell.font = Font(bold=True, size=12)

        # Header: Dataset | model1 | model2 | ...
        headers_sum = ["Dataset"] + model_order
        for col_idx, h in enumerate(headers_sum, 1):
            cell = ws_summary.cell(row=start_row + 1, column=col_idx, value=h)
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = center_align
            cell.border = thin_border

        # Data
        for ds_idx, ds_name in enumerate(dataset_order):
            row = start_row + 2 + ds_idx
            cell = ws_summary.cell(row=row, column=1, value=ds_name)
            cell.border = thin_border
            cell.alignment = center_align

            vals_this_row = []
            for col_idx, model_name in enumerate(model_order, 2):
                met = model_data[model_name].get(ds_name, {})
                v = met.get(metric_key, "N/A")
                cell = ws_summary.cell(row=row, column=col_idx)
                cell.border = thin_border
                cell.alignment = center_align
                if isinstance(v, float):
                    if "MPE" in metric_key:
                        cell.value = round(v, 2)
                        cell.number_format = "0.00"
                    else:
                        cell.value = round(v, 4)
                        cell.number_format = "0.0000"
                else:
                    cell.value = v
                vals_this_row.append(v)

            # Mark best (main metrics only, skip Std/CI95)
            if not is_std_ci95:
                num_vals = [(i, v) for i, v in enumerate(vals_this_row) if isinstance(v, (int, float))]
                if len(num_vals) > 0:
                    if metric_key in ("MAE", "RMSE", "MPE"):
                        best_idx, _ = min(num_vals, key=lambda x: x[1])
                    else:
                        best_idx, _ = max(num_vals, key=lambda x: x[1])
                    ws_summary.cell(row=row, column=best_idx + 2).fill = best_fill

        # ---- Average row ----
        avg_row = start_row + 2 + len(dataset_order)
        cell = ws_summary.cell(row=avg_row, column=1, value="Average")
        cell.fill = overall_fill
        cell.font = Font(bold=True)
        cell.border = thin_border
        cell.alignment = center_align

        avg_vals = []
        for col_idx, model_name in enumerate(model_order, 2):
            if is_std_ci95:
                # Std/CI95: pooled per-image data across all images
                all_gt, all_pred = [], []
                for ds_name in dataset_order:
                    gt_arr, pred_arr = _read_per_image_csv(output_root, model_name, ds_name)
                    if gt_arr is not None:
                        all_gt.extend(gt_arr.tolist())
                        all_pred.extend(pred_arr.tolist())
                if all_gt:
                    try:
                        pooled = cu.compute_std_ci95(np.array(all_gt), np.array(all_pred))
                        avg_v = pooled.get(metric_key, "N/A")
                    except Exception:
                        avg_v = "N/A"
                else:
                    avg_v = "N/A"
            else:
                # Main metrics: weighted average
                weighted_sum = 0.0
                total_weight = 0
                for ds_name in dataset_order:
                    v = model_data[model_name].get(ds_name, {}).get(metric_key, None)
                    w = model_data[model_name].get(ds_name, {}).get("n_images", 1)
                    if isinstance(v, (int, float)) and isinstance(w, (int, float)) and w > 0:
                        weighted_sum += v * w
                        total_weight += w
                avg_v = weighted_sum / total_weight if total_weight > 0 else "N/A"
            avg_vals.append(avg_v)

            cell = ws_summary.cell(row=avg_row, column=col_idx)
            cell.fill = overall_fill
            cell.border = thin_border
            cell.alignment = center_align
            if isinstance(avg_v, float):
                if "MPE" in metric_key:
                    cell.value = round(avg_v, 2)
                    cell.number_format = "0.00"
                else:
                    cell.value = round(avg_v, 4)
                    cell.number_format = "0.0000"
            else:
                cell.value = avg_v

        # Mark best in Average row (main metrics only)
        if not is_std_ci95:
            num_avg = [(i, v) for i, v in enumerate(avg_vals) if isinstance(v, (int, float))]
            if len(num_avg) > 0:
                if metric_key in ("MAE", "RMSE", "MPE"):
                    best_idx_avg, _ = min(num_avg, key=lambda x: x[1])
                else:
                    best_idx_avg, _ = max(num_avg, key=lambda x: x[1])
                ws_summary.cell(row=avg_row, column=best_idx_avg + 2).fill = best_fill

        # Freeze header row
        ws_summary.freeze_panes = "A2"

    # Column widths
    ws_summary.column_dimensions["A"].width = 12
    for i in range(len(model_order)):
        ws_summary.column_dimensions[get_column_letter(i + 2)].width = 14

    # Use temp file to avoid Windows file locking issues
    temp_path = xlsx_path.with_suffix(".xlsx.tmp")
    wb.save(str(temp_path))
    try:
        if xlsx_path.exists():
            xlsx_path.unlink()
        temp_path.rename(xlsx_path)
    except PermissionError:
        # File locked by external process, keep temp file
        print(f"  [Note] Original file locked, results saved to: {temp_path}")
    print(f"\n  Excel summary saved: {xlsx_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch Counting: Run Cellpose4 / Microatlas / Cellpose3 / MicroSAM / CellSAM cell counting evaluation in batch",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: run all five models on all datasets
  python counting/batch_counting.py

  # Specify datasets + model
  python counting/batch_counting.py --datasets BBBC041 --models cellpose4

  # CPU + dry run only
  python counting/batch_counting.py --cpu --dry-run
        """,
    )
    parser.add_argument(
        "--datasets", "-d", type=str, default=None,
        help="Comma-separated dataset names, e.g. 'BBBC001,BBBC041'. Run all if not specified."
    )
    parser.add_argument(
        "--models", "-m", type=str, default="both",
        choices=["cellpose4", "cellpose3", "microatlas", "microsam", "cellsam", "both"],
        help="Model(s) to run (default both: all 5 models)"
    )
    parser.add_argument(
        "--output_dir", "-o", type=str, default=str(COUNTING_DIR / "results"),
        help="Results output root directory (default: counting/results)"
    )
    parser.add_argument(
        "--cpu", action="store_true",
        help="Use CPU (default GPU)"
    )
    parser.add_argument(
        "--diameter", type=float, default=None,
        help="Override cell diameter (pixels), uses model default if not specified"
    )
    parser.add_argument(
        "--model_type", type=str, default="vit_l_lm",
        help="MicroSAM model type (default vit_l_lm)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Only list tasks, do not execute"
    )
    parser.add_argument(
        "--excel-only", action="store_true",
        help="Only regenerate summary Excel from existing results, skip counting evaluation"
    )

    args = parser.parse_args()

    # ---- Excel-only mode: scan existing results and regenerate summary ----
    if args.excel_only:
        print("=" * 70)
        print("Excel-only mode: regenerating summary from existing results...")
        print("=" * 70)
        results = []
        output_root = Path(args.output_dir)
        if not output_root.is_dir():
            print(f"[Error] Results directory not found: {output_root}")
            sys.exit(1)

        # Scan all dataset_model_counting directories
        for ds_dir in sorted(output_root.iterdir()):
            if not ds_dir.is_dir():
                continue
            dir_name = ds_dir.name
            # Parse directory name: BBBC001_cellpose4_counting
            parts = dir_name.rsplit("_", 2)
            if len(parts) != 3 or parts[2] != "counting":
                continue
            dataset_name = parts[0]
            model_name = parts[1]

            # Skip removed datasets
            if dataset_name not in DATASET_CONFIGS:
                continue

            metrics_dir = ds_dir / "results_metrics"
            if not metrics_dir.is_dir():
                continue

            # Find metrics file
            metrics_files = sorted(metrics_dir.glob(f"{model_name}_{dataset_name}_metrics.txt"))
            if not metrics_files:
                # Try wildcard
                metrics_files = sorted(metrics_dir.glob("*_metrics.txt"))
            if not metrics_files:
                continue

            metrics_path = metrics_files[0]
            metrics = {}
            with open(metrics_path) as f:
                for line in f:
                    line = line.strip()
                    if "=" not in line:
                        continue
                    key, val_str = line.split("=", 1)
                    key = key.strip()
                    val_str = val_str.strip().rstrip("%")
                    if key == "n_images":
                        metrics[key] = int(float(val_str))
                    elif key in ("MAE", "RMSE", "R2", "Pearson_r", "MPE"):
                        try:
                            metrics[key] = float(val_str)
                        except ValueError:
                            metrics[key] = val_str

            if not metrics:
                continue

            results.append({
                "dataset": dataset_name,
                "model": model_name,
                "metrics": metrics,
            })

        if not results:
            print(f"[Error] No result data found under {output_root}")
            sys.exit(1)

        print(f"  Found {len(results)} valid results")
        save_excel_summary(results, args.output_dir)
        print(f"\n  Summary Excel saved to: {output_root / 'results_summary.xlsx'}")
        return

    # ---- Parse datasets ----
    if args.datasets:
        datasets = [d.strip() for d in args.datasets.split(",")]
        for ds in datasets:
            if ds not in DATASET_CONFIGS:
                print(f"[Error] Unknown dataset: {ds}")
                print(f"  Available datasets: {', '.join(DATASET_CONFIGS.keys())}")
                sys.exit(1)
    else:
        datasets = sorted(DATASET_CONFIGS.keys())

    # ---- Parse models ----
    if args.models == "both":
        models = ["cellpose4", "cellpose3", "microatlas", "microsam", "cellsam"]
    else:
        models = [args.models]

    # ---- Build task list ----
    tasks = []
    for ds in datasets:
        for model in models:
            cfg = DATASET_CONFIGS[ds]
            tasks.append({
                "dataset": ds,
                "model": model,
                "n_images": cfg["n_images"],
                "output_dir": str(Path(args.output_dir) / f"{ds}_{model}_counting"),
            })

    if len(tasks) == 0:
        print("[Info] No executable tasks found.")
        sys.exit(0)

    # ---- Print task summary ----
    print("=" * 70)
    print("Batch Counting Task Summary")
    print("=" * 70)
    print(f"  GPU mode: {'No (CPU)' if args.cpu else 'Yes'}")
    print(f"  Datasets: {', '.join(datasets)}")
    print(f"  Models: {', '.join(models)}")
    print(f"  Total tasks: {len(tasks)}")
    print("-" * 70)
    for i, t in enumerate(tasks, 1):
        print(f"  [{i:2d}] {t['dataset']:10s} | {t['model']:12s} | {t['n_images']:5d} images")
    print("-" * 70)

    if args.dry_run:
        print("[dry-run] Tasks listed only, not executing.")
        sys.exit(0)

    # ---- Confirm execution ----
    total_images = sum(t["n_images"] for t in tasks)
    print(f"  Estimated total images: {total_images}")
    print("=" * 70)
    print()

    # ---- Batch execution ----
    results = []
    total_start = time.time()

    for i, t in enumerate(tasks, 1):
        header = f"  Task [{i}/{len(tasks)}] {t['dataset']} - {t['model']}"
        print("=" * 70)
        print(header)
        print("=" * 70)

        kwargs = {
            "use_gpu": not args.cpu,
            "diameter": args.diameter,
            "model_type": args.model_type,
        }

        success, elapsed, metrics = run_single_task(
            model=t["model"],
            dataset_name=t["dataset"],
            output_root=args.output_dir,
            **kwargs,
        )

        status = "OK" if success else "FAIL"
        results.append({
            "dataset": t["dataset"],
            "model": t["model"],
            "n_images": t["n_images"],
            "success": success,
            "elapsed": elapsed,
            "metrics": metrics,
        })
        elapsed_str = f"{elapsed / 60:.1f}min" if elapsed > 120 else f"{elapsed:.0f}s"
        print(f"  [{status}] {header} elapsed: {elapsed_str}")
        print()

    # ---- Generate Excel summary ----
    print("=" * 70)
    print("Generating summary Excel...")
    print("=" * 70)
    save_excel_summary(results, args.output_dir)

    # ---- Final summary ----
    total_elapsed = time.time() - total_start
    n_ok = sum(1 for r in results if r["success"])
    n_fail = len(results) - n_ok

    print()
    print("=" * 70)
    print("Batch Counting Complete Summary")
    print("=" * 70)
    print(f"  Total elapsed: {total_elapsed / 60:.1f}min ({total_elapsed:.0f}s)")
    print(f"  Success: {n_ok} / {len(results)}")
    if n_fail > 0:
        print(f"  Failed: {n_fail}")
    print()
    print(f"  {'Dataset':10s} {'Model':12s} {'Status':6s} {'Elapsed':10s}")
    print(f"  {'-'*10} {'-'*12} {'-'*6} {'-'*10}")
    for r in results:
        elapsed_str = f"{r['elapsed'] / 60:.1f}min" if r['elapsed'] > 120 else f"{r['elapsed']:.0f}s"
        status = "OK" if r["success"] else "FAIL"
        print(f"  {r['dataset']:10s} {r['model']:12s} {status:6s} {elapsed_str:>10s}")
    print("=" * 70)

    # Summary Excel path hint
    print(f"\n  Summary Excel saved to: {Path(args.output_dir) / 'results_summary.xlsx'}")
    print(f"  Per-image CSV saved to: {Path(args.output_dir) / 'results_per_image'}")


if __name__ == "__main__":
    main()
