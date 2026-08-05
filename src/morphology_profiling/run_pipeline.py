"""
run_pipeline.py
===============
Disk-frugal per-plate driver for the U2OS-Cell-Painting pipeline.

The full figshare record is ~634 GB, but only ~200 GB of free disk is needed if
each plate is processed to completion before the next one is fetched. This
driver orchestrates that loop by shelling out to the existing stage scripts:

    metadata + full label table (once)
        download.py --metadata_only
        preprocess.py                      # builds data/results/preprocess/image_table.csv

    for plate in plates (small -> large):
        download.py  --plate P [--stream]  # fetch tar.gz, extract 5 FL channels, delete archive
        segment.py   --models ... --plate P
        feature_extraction.py --models ... --plate P
        (cleanup)    delete data/images/P  # keep only the field-level feature CSVs

    for model in models (once, after all plates):
        feature_aggregation.py --models model
        biomarker/unsupervised.py --models model

Feature CSVs are written per (model, plate) by feature_extraction.py and are
tiny compared with the images, so the aggregation + clustering steps run after
every plate has been featurized and its images deleted.

Usage:
  python src/morphology_profiling/run_pipeline.py --pilot            # 4 smallest plates
  python src/morphology_profiling/run_pipeline.py --all --stream     # all 18 plates, no archive on disk
  python src/morphology_profiling/run_pipeline.py --plates P015080 P015081 --keep_images
  python src/morphology_profiling/run_pipeline.py --cluster_only     # just aggregate + cluster
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

HERE = Path(__file__).resolve().parent
PILOT_PLATES = ["P015080", "P015081", "P015082", "P015083"]  # 4 smallest (~20 GB total)


def _run(script_rel, *cli_args):
    """Run a stage script with the current interpreter; abort on failure."""
    cmd = [sys.executable, "-u", str(HERE / script_rel), *map(str, cli_args)]
    print(f"\n$ {' '.join(cmd)}", flush=True)
    res = subprocess.run(cmd)
    if res.returncode != 0:
        raise SystemExit(f"[ABORT] {script_rel} exited with {res.returncode}")


def _cleanup_plate_images(plate_name):
    plate_dir = config.IMAGES_DIR / plate_name
    if not plate_dir.exists():
        return
    shutil.rmtree(plate_dir, ignore_errors=True)
    if plate_dir.exists():
        # A partially deleted tree is worse than an intact one: a later
        # extraction run would featurize only the surviving fields and save
        # that as the plate's final CSV.
        left = sum(1 for _ in plate_dir.rglob("*.tif"))
        print(f"  [ERROR] {plate_dir} not fully removed ({left} TIFFs left). "
              f"Delete it manually, or re-run download.py --plate "
              f"{plate_name} before extracting features again.")
    else:
        print(f"  [cleanup] removed {plate_dir}")


def process_plate(plate_name, models, stream=False, keep_images=False,
                  keep_archive=False):
    print(f"\n{'#' * 64}\n# PLATE {plate_name}\n{'#' * 64}")
    dl_args = ["--plate", plate_name]
    if stream:
        dl_args.append("--stream")
    if keep_archive:
        dl_args.append("--keep_archive")
    _run("download.py", *dl_args)

    _run("segment.py", "--models", *models, "--plate", plate_name)
    _run("feature_extraction.py", "--models", *models, "--plate", plate_name)

    if not keep_images:
        _cleanup_plate_images(plate_name)


def aggregate_and_cluster(models):
    for model in models:
        print(f"\n{'#' * 64}\n# AGGREGATE + CLUSTER: {model}\n{'#' * 64}")
        _run("feature_aggregation.py", "--models", model)
        _run("biomarker/unsupervised.py", "--models", model)


def main():
    parser = argparse.ArgumentParser(
        description="Disk-frugal per-plate driver for the U2OS-Cell-Painting pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--models", type=str, nargs="+", default=config.MODELS,
                        help=f"Models to run (default: {config.MODELS})")
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument("--pilot", action="store_true",
                     help="Only the 4 smallest plates (pilot / sanity run)")
    grp.add_argument("--all", action="store_true",
                     help="All 18 plates (small -> large)")
    grp.add_argument("--plates", type=str, nargs="+", default=None,
                     help="Explicit plate list")
    parser.add_argument("--stream", action="store_true",
                        help="Stream-extract FL without saving the tar.gz (no resume)")
    parser.add_argument("--keep_archive", action="store_true",
                        help="Keep each downloaded tar.gz")
    parser.add_argument("--keep_images", action="store_true",
                        help="Do NOT delete extracted images after featurizing a plate")
    parser.add_argument("--cluster_only", action="store_true",
                        help="Skip download/segment; only aggregate + cluster")
    args = parser.parse_args()

    models = ["cellpose4", "cellpose3", "microsam", "microatlas", "cellsam"] \
        if args.models == ["all"] else args.models

    config.ensure_dirs()

    if args.cluster_only:
        aggregate_and_cluster(models)
        return

    if args.pilot:
        plates = PILOT_PLATES
    elif args.plates:
        plates = args.plates
    else:  # default and --all both mean the full ordered list
        plates = config.PLATE_NAMES

    # 1. Metadata + full label table (once).
    _run("download.py", "--metadata_only")
    _run("preprocess.py")

    # 2. Per-plate disk-frugal loop.
    for plate in plates:
        process_plate(plate, models, stream=args.stream,
                      keep_images=args.keep_images, keep_archive=args.keep_archive)

    # 3. Aggregate + cluster per model (once, after all images processed).
    aggregate_and_cluster(models)

    print("\n=== Pipeline complete ===")


if __name__ == "__main__":
    main()
