"""
TNBC-MIBI Nuclear Channel Image Generation
==========================================
dsDNA + H3K27me3 + H3K9ac sum → nuclear channel, save as single-channel TIF

Usage: python TNBC-MIBI/analysis/gen_composite_preview.py
"""
import numpy as np
import tifffile
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent
MARKER_DIR = DATA / 'TNBC' / 'TNBCShareData'
MASK_DIR = DATA / 'TNBC_shareCellData'
OUT_DIR = DATA / 'composite_preview'
OUT_DIR.mkdir(parents=True, exist_ok=True)

MARKER_NUCLEAR = ['dsDNA', 'H3K27me3', 'H3K9ac']


def _percentile_clip(img, low=2, high=99.5):
    if img.max() == img.min():
        return np.zeros_like(img, dtype=np.float32)
    lo, hi = np.percentile(img, [low, high])
    clipped = np.clip(img.astype(np.float64), lo, hi)
    clipped = (clipped - lo) / (hi - lo + 1e-10)
    return clipped.astype(np.float32)


def load_nuclear_channel(pid):
    """Load and normalize nuclear channel: dsDNA + H3K27me3 + H3K9ac"""
    pid_dir = MARKER_DIR / f'Point{pid}'
    s = None
    for m in MARKER_NUCLEAR:
        path = pid_dir / f'{m}.tif'
        if path.exists():
            img = tifffile.imread(path).astype(np.float64)
            s = img if s is None else s + img
    if s is None:
        return np.zeros((1024, 1024), dtype=np.float32)
    return _percentile_clip(s, 2, 99.5)


def main():
    valid_pids = []
    for pid in range(1, 45):
        if (MASK_DIR / f'p{pid}_labeledcellData.tiff').exists() and \
           (MARKER_DIR / f'Point{pid}').exists():
            valid_pids.append(pid)

    print(f"Generating nuclear channel images for {len(valid_pids)} patients...")
    print(f"Output directory: {OUT_DIR}")
    print(f"  Nuclear channel = dsDNA + H3K27me3 + H3K9ac (2%~99.5% clipping normalization)")
    print()

    for pid in valid_pids:
        nuclear = load_nuclear_channel(pid)  # (H, W) float32 [0,1]

        tif_path = OUT_DIR / f'Point{pid:02d}_nuclear.tif'
        tifffile.imwrite(str(tif_path), nuclear.astype(np.float32))

        print(f"  Point {pid:2d}: saved {nuclear.shape[1]}x{nuclear.shape[0]}")

    print(f"\nDone! Generated {len(valid_pids)} nuclear channel TIF files")


if __name__ == '__main__':
    main()
