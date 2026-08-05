"""
download.py
===========
Download the U2OS-Cell-Painting dataset (Uppsala, figshare record 21378906).

The raw images are distributed as ONE tar.gz per microplate, each containing
both the fluorescence (FL, 5 channels) and brightfield (BF, 6 z-planes) TIFFs.
The full record is ~634 GB. This pipeline uses only the FL channels, so we run
a disk-frugal per-plate pipeline designed to fit in ~200 GB of free space:

    for each plate (small -> large):
        1. download it as resumable Range chunks into
           data/archives/{plate}.tar.gz.parts/chunk_NNNN  (--stream to avoid disk)
        2. extract ONLY the 5 FL channels    (BF is skipped) into
           data/images/{plate}/{CHANNEL}/{well}_s{site}_w{idx}.tif
        3. delete the chunk parts             (--keep_archive assembles a tar.gz)

The FL TIFFs are read straight out of the chunk parts, so the 51.8 GB archive is
never assembled: peak disk is one archive, not two, and the parts survive until
the extraction succeeds, so any failure is resumable. Only one download per plate
may run at a time (enforced by a PID lock in data/archives/.{plate}.lock);
concurrent runs would append to the same chunk files and interleave bytes.

The figshare download URL 302-redirects to a short-lived (10s) presigned
KTH-S3 URL that honours HTTP Range (206). A single TCP stream is BDP-limited
on the HK->EU link (~3.5 MB/s), so by default the disk mode pulls each archive
with `--workers` parallel Range chunks (each chunk re-resolves its own fresh
presigned URL, so the 10s expiry never bites). Set `--workers 1` for the old
single-stream behaviour.

The normalized filenames follow the shared layout so the shared field-key
convention `{well}_s{site}` keeps working for every downstream stage. After a
plate is segmented + featurized you can delete data/images/{plate} before
moving on, keeping only the feature CSVs.

Metadata (small, downloaded/extracted once into data/metadata/):
  - fl_data.csv / bf_data.csv   plate/well/site/compound/MoA + channel filenames
  - CP_features_cells.csv       authors' CellProfiler benchmark features
  - grit_scores.csv             grit scores / nuclear counts

Usage:
  python src/morphology_profiling/download.py --metadata_only
  python src/morphology_profiling/download.py --plate P015080
  python src/morphology_profiling/download.py --plate P015099 --workers 8
  python src/morphology_profiling/download.py --plate P015099 --stream
  python src/morphology_profiling/download.py --all
  python src/morphology_profiling/download.py --status
"""

import argparse
import io
import os
import shutil
import sys
import tarfile
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config


# =========================================================================
# HTTP helpers
# =========================================================================

_USER_AGENT = "Mozilla/5.0 (u2os-biomarker-pipeline)"


def _open_url(url, offset=0):
    """Open a URL for streaming, optionally resuming from a byte offset."""
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    if offset:
        req.add_header("Range", f"bytes={offset}-")
    return urllib.request.urlopen(req)


def _download_resumable(url, dest, desc=None):
    """Download url -> dest with resume support (via HTTP Range on a .part file)."""
    dest = Path(dest)
    part = dest.with_suffix(dest.suffix + ".part")
    offset = part.stat().st_size if part.exists() else 0

    resp = _open_url(url, offset=offset)
    # If the server ignored the Range request, restart from scratch.
    if offset and resp.status != 206:
        offset = 0
        part.unlink(missing_ok=True)
        resp = _open_url(url)

    total = offset + int(resp.headers.get("Content-Length", 0))
    mode = "ab" if offset else "wb"
    bar = tqdm(total=total or None, initial=offset, unit="B", unit_scale=True,
               desc=desc or dest.name)
    with open(part, mode) as fh:
        while True:
            chunk = resp.read(1 << 20)  # 1 MiB
            if not chunk:
                break
            fh.write(chunk)
            bar.update(len(chunk))
    bar.close()
    part.replace(dest)
    return dest


# =========================================================================
# Parallel chunked download (figshare 302 -> KTH S3, Range supported)
# =========================================================================

def _open_url_range(url, start, end=None, timeout=60):
    """Open ``url`` requesting byte range [start, end] (end inclusive).

    urllib follows figshare's 302 to a *fresh* presigned KTH-S3 URL and
    re-sends the Range header (host is the only signed header, so adding Range
    is free). Because every call re-resolves its own short-lived presigned URL,
    the 10-second ``X-Amz-Expires`` window never expires mid-request.
    """
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    rng = f"bytes={start}-" if end is None else f"bytes={start}-{end}"
    req.add_header("Range", rng)
    return urllib.request.urlopen(req, timeout=timeout)


def _get_total_size(url):
    """Return the total file size in bytes via a 1-byte Range probe."""
    resp = _open_url_range(url, 0, 0)
    cr = resp.headers.get("Content-Range")
    resp.close()
    if cr and "/" in cr:
        return int(cr.rsplit("/", 1)[-1])
    raise RuntimeError(f"server did not return Content-Range for {url}")


def _download_chunk(url, start, end, part_path, bar, lock, stall_retries=8):
    """Download byte range [start, end] into ``part_path`` with resume + retry.

    Each attempt re-resolves a fresh presigned URL, so a dropped connection or
    a transient 502 from figshare's redirector is retried with exponential
    backoff, resuming from the bytes already on disk. The retry budget is spent
    only on *stalled* attempts: any attempt that appends at least one byte
    refills it, so a chunk that keeps trickling in is never abandoned. A 206
    that closes with an empty body is a stall, not a success, and is reported
    as such instead of surfacing as ``last_err: None``.
    """
    expected = end - start + 1
    got = part_path.stat().st_size if part_path.exists() else 0
    if got > expected:
        # A previous run hit a server that ignored Range and served from byte 0,
        # so the on-disk bytes are misaligned garbage -> refetch the chunk.
        part_path.unlink(missing_ok=True)
        with lock:
            bar.update(-expected)  # the initial bar credited min(size, expected)
        got = 0
    elif got == expected:
        return

    last_err = None
    stalls = 0
    while stalls < stall_retries:
        cur = start + got
        try:
            resp = _open_url_range(url, cur, end)
            status = getattr(resp, "status", None) or resp.getcode()
            if status != 206:
                # Body would start at byte 0, not at ``cur`` -> unusable here.
                resp.close()
                raise RuntimeError(f"server ignored Range (HTTP {status})")
            mode = "ab" if got else "wb"
            with open(part_path, mode) as fh:
                while True:
                    buf = resp.read(1 << 20)
                    if not buf:
                        break
                    fh.write(buf)
                    with lock:
                        bar.update(len(buf))
            resp.close()
        except Exception as e:  # 502 / timeout / reset -> back off and resume
            last_err = e

        now = part_path.stat().st_size if part_path.exists() else 0
        if now >= expected:
            return
        if now > got:  # progress made -> the stall budget is refilled
            got, stalls, last_err = now, 0, None
            continue
        got = now
        stalls += 1
        time.sleep(min(2 ** stalls, 60))

    raise RuntimeError(
        f"chunk [{start}-{end}] stalled at {got}/{expected} B after "
        f"{stall_retries} attempts with no progress: "
        f"{last_err or 'server returned HTTP 206 with an empty body'}")


class _PartsReader(io.RawIOBase):
    """Sequential read-only view over an ordered list of chunk files.

    Lets ``tarfile`` stream straight out of ``{archive}.parts/chunk_NNNN`` so the
    51.8 GB archive never has to be materialized on disk.
    """

    def __init__(self, paths):
        self._paths = list(paths)
        self._i = 0
        self._fh = None

    def readable(self):
        return True

    def readinto(self, b):
        while True:
            if self._fh is None:
                if self._i >= len(self._paths):
                    return 0
                self._fh = open(self._paths[self._i], "rb")
            n = self._fh.readinto(b)
            if n:
                return n
            self._fh.close()
            self._fh = None
            self._i += 1

    def close(self):
        if self._fh is not None:
            self._fh.close()
            self._fh = None
        super().close()


def _fetch_parts(url, dest, num_workers=6, chunk_size=256 * 1024 * 1024,
                 desc=None, rounds=5):
    """Fetch ``url`` as ordered Range chunks; return (parts_dir, [part paths]).

    The KTH S3 backend honours Range (HTTP 206) but a single TCP stream is
    bandwidth-delay-product limited on the HK->EU link (~3.5 MB/s). Splitting
    into large chunks pulled by a handful of workers fills the pipe. Concurrency
    is kept modest (default 6) so figshare's redirector is not hammered into
    502s the way ``aria2c -x16`` was.

    Each chunk is stored as ``{dest}.parts/chunk_NNNN`` so an interrupted run
    resumes per-chunk. Chunks that exhaust their stall budget are re-attempted
    in further rounds with halved concurrency and a growing pause (the
    redirector starves late chunks when many workers hold connections open);
    the run aborts only after ``rounds`` rounds still leave a chunk short.

    Nothing is deleted here: the caller consumes the parts (streaming them into
    tarfile, or concatenating them) and removes them only once that succeeded,
    so any failure downstream is still fully resumable.
    """
    dest = Path(dest)
    total = _get_total_size(url)
    chunk_size = int(chunk_size)  # a float would emit a bad "bytes=0-4095.0"
    parts_dir = dest.parent / (dest.name + ".parts")
    parts_dir.mkdir(parents=True, exist_ok=True)

    chunks = []
    start = idx = 0
    while start < total:
        end = min(start + chunk_size - 1, total - 1)
        chunks.append((idx, start, end, parts_dir / f"chunk_{idx:04d}"))
        start = end + 1
        idx += 1

    def _on_disk():
        n = 0
        for _, s, e, pp in chunks:
            if pp.exists():
                n += min(pp.stat().st_size, e - s + 1)
        return n

    bar = tqdm(total=total, initial=_on_disk(), unit="B", unit_scale=True,
               desc=desc or dest.name)
    lock = threading.Lock()
    pending = chunks
    workers = num_workers

    for rnd in range(1, rounds + 1):
        failed = []

        def _task(item):
            _, s, e, pp = item
            try:
                _download_chunk(url, s, e, pp, bar, lock)
            except Exception as exc:
                with lock:
                    failed.append((item, exc))

        before = _on_disk()
        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(_task, pending))
        if not failed:
            break

        gained = _on_disk() - before
        if rnd == rounds:
            bar.close()
            raise failed[0][1]
        # A zero-progress round is not fatal: the redirector can stay cold for
        # minutes, so wait longer each time rather than abandoning the archive.
        pause = 30 * rnd
        bar.write(f"  [retry {rnd}/{rounds - 1}] {len(failed)} chunk(s) stalled, "
                  f"{gained / 1e6:.1f} MB gained this round; retrying them in "
                  f"{pause}s with {max(1, workers // 2)} worker(s)")
        pending = [item for item, _ in failed]
        workers = max(1, workers // 2)
        time.sleep(pause)

    bar.close()

    # Verify every part up front, so a short/missing chunk is reported here
    # rather than as a gzip CRC error thousands of TIFFs later.
    paths = []
    for _, s, e, pp in chunks:
        size = pp.stat().st_size if pp.exists() else -1
        if size != e - s + 1:
            raise RuntimeError(
                f"chunk {pp.name} is {size} B, expected {e - s + 1} B; "
                f"delete it and re-run to refetch just that chunk")
        paths.append(pp)
    return parts_dir, paths


def _assemble_parts(paths, dest):
    """Concatenate verified parts into ``dest`` (only needed for --keep_archive).

    Written to a ``.assembling`` scratch name and renamed on success, so a crash
    never leaves a truncated file that looks like a finished archive. Parts are
    kept until the rename succeeds, so this needs room for archive + parts.
    """
    dest = Path(dest)
    tmp = dest.parent / (dest.name + ".assembling")
    with open(tmp, "wb") as out:
        for pp in tqdm(paths, unit="chunk", desc=f"  assemble {dest.name}"):
            with open(pp, "rb") as fh:
                shutil.copyfileobj(fh, out, length=1 << 20)
    tmp.replace(dest)
    return dest


# =========================================================================
# Metadata
# =========================================================================

def download_metadata():
    """Download + extract the small metadata / benchmark archives."""
    print("\n=== Downloading U2OS-Cell-Painting metadata (figshare 21378906) ===")
    config.METADATA_DIR.mkdir(parents=True, exist_ok=True)

    for name, url in config.SMALL_FILES.items():
        dest = config.METADATA_DIR / name
        if dest.exists() and dest.stat().st_size > 0:
            print(f"  [SKIP] {name} ({dest.stat().st_size/1024:.1f} KB)")
        else:
            print(f"  Downloading: {name}")
            _download_resumable(url, dest)
        if name.endswith(".tar.gz"):
            _extract_flat(dest, config.METADATA_DIR)

    _validate_metadata()
    return True


def _extract_flat(tar_path, out_dir):
    """Extract every *.csv in a tar.gz into out_dir (flattened, no subdirs)."""
    out_dir = Path(out_dir)
    with tarfile.open(tar_path, "r:gz") as tf:
        for m in tf.getmembers():
            if not m.isfile() or not m.name.endswith(".csv"):
                continue
            target = out_dir / Path(m.name).name
            if target.exists() and target.stat().st_size > 0:
                continue
            src = tf.extractfile(m)
            if src is None:
                continue
            with open(target, "wb") as fh:
                fh.write(src.read())
            print(f"    extracted {target.name}")


def _validate_metadata():
    if not config.FL_DATA_CSV.exists():
        print("  [WARN] fl_data.csv not found after extraction")
        return
    df = pd.read_csv(config.FL_DATA_CSV, dtype=str)
    n_moa = df["moa"].nunique()
    n_cpd = df["compound"].nunique()
    print(f"  fl_data.csv: {len(df)} FL fields, {n_cpd} compounds, "
          f"{n_moa} MoA labels (incl. DMSO), {df['plate'].nunique()} plates")


# =========================================================================
# Per-plate FL extraction
# =========================================================================

def _load_fl_table():
    if not config.FL_DATA_CSV.exists():
        raise FileNotFoundError(
            f"{config.FL_DATA_CSV} not found. Run: python download.py --metadata_only"
        )
    return pd.read_csv(config.FL_DATA_CSV, dtype=str)


def _plate_filemap(fl_df, plate_name):
    """Map every FL TIFF basename of a plate -> normalized (channel, out_name).

    out_name follows the shared layout: {well}_s{site}_w{idx}.tif
    where idx is the 1-based position of the channel in config.CHANNEL_NAMES
    (== the C-number), so downstream path building stays consistent across datasets.
    """
    sub = fl_df[fl_df["plate"] == plate_name]
    channel_idx = {ch: i + 1 for i, ch in enumerate(config.CHANNEL_NAMES)}
    fmap = {}
    for _, row in sub.iterrows():
        well = str(row["well"]).strip()
        site = int(str(row["site"]).lstrip("s"))
        for ch in config.CHANNEL_NAMES:
            raw = str(row[config.CHANNEL_CSV_COLS[ch]]).strip()
            if not raw or raw.lower() == "nan":
                continue
            out_name = f"{well}_s{site}_w{channel_idx[ch]}.tif"
            fmap[raw] = (ch, out_name)
    return fmap


def _count_extracted(plate_name, fmap):
    """Count how many of a plate's mapped FL TIFFs are already on disk."""
    present = 0
    for channel, out_name in fmap.values():
        dest = config.IMAGES_DIR / plate_name / channel / out_name
        if dest.exists() and dest.stat().st_size > 0:
            present += 1
    return present


def _write_member(fileobj, plate_name, channel, out_name):
    """Extract one tar member to its final path, atomically.

    The bytes land in a sibling ``{out_name}.part`` file that is renamed into
    place only after the whole member has been written. Two reasons:

      * ``segment.py`` is meant to run in parallel with this downloader, and a
        reader that opens a half-written TIFF dies with
        ``TiffFileError: corrupted tag list``.
      * A killed extraction leaves a ``.part`` scrap (which no downstream glob
        picks up and which the next run overwrites) instead of a truncated
        ``.tif`` that passes the "exists and non-empty" check forever.
    """
    dest_dir = config.IMAGES_DIR / plate_name / channel
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / out_name
    if dest.exists() and dest.stat().st_size > 0:
        return False
    tmp = dest.with_name(dest.name + ".part")
    with open(tmp, "wb") as fh:
        shutil.copyfileobj(fileobj, fh, length=1 << 20)
    tmp.replace(dest)
    return True


def _extract_fl_from_tar(tf, fmap, plate_name):
    """Iterate tar members and extract only the FL channels named in fmap."""
    written = skipped = 0
    bar = tqdm(total=len(fmap), unit="img", desc=f"  extract {plate_name}")
    for m in tf:
        if not m.isfile():
            continue
        base = Path(m.name).name
        hit = fmap.get(base)
        if hit is None:
            continue  # BF plane or anything not in the FL map -> skip
        channel, out_name = hit
        src = tf.extractfile(m)
        if src is None:
            continue
        if _write_member(src, plate_name, channel, out_name):
            written += 1
        else:
            skipped += 1
        bar.update(1)
    bar.close()
    return written, skipped


@contextmanager
def _plate_lock(plate_name):
    """Refuse to run two downloads of the same plate at once.

    Two concurrent runs share the same ``chunk_NNNN`` files and append to them
    in parallel, which interleaves bytes and destroys the archive. A PID lock
    file makes the second run fail immediately instead. Locks left behind by a
    killed process are detected via ``os.kill(pid, 0)`` and reclaimed.
    """
    config.ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    lock_path = config.ARCHIVE_DIR / f".{plate_name}.lock"
    if lock_path.exists():
        try:
            owner = int(lock_path.read_text().strip())
        except (ValueError, OSError):
            owner = None
        alive = False
        if owner is not None:
            try:
                os.kill(owner, 0)
                alive = True
            except OSError:
                alive = False
        if alive:
            raise RuntimeError(
                f"{plate_name} is already being downloaded by PID {owner} "
                f"(lock: {lock_path}). Running two downloads of one plate "
                f"corrupts the chunk parts. Wait for it, or kill it and delete "
                f"the lock file.")
        print(f"  [lock] reclaiming stale lock from PID {owner}")
        lock_path.unlink(missing_ok=True)

    fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
    except BaseException:
        os.close(fd)
        lock_path.unlink(missing_ok=True)
        raise
    try:
        yield
    finally:
        lock_path.unlink(missing_ok=True)


def download_plate(plate_name, stream=False, keep_archive=False,
                   workers=6, chunk_mb=256):
    """Download one plate, extract its 5 FL channels, then delete the archive."""
    if plate_name not in config.PLATE_TARBALLS:
        print(f"  [ERROR] Unknown plate {plate_name}")
        return False
    url, size_gb = config.PLATE_TARBALLS[plate_name]
    fl_df = _load_fl_table()
    fmap = _plate_filemap(fl_df, plate_name)
    if not fmap:
        print(f"  [ERROR] No FL rows for {plate_name} in fl_data.csv")
        return False

    print(f"\n=== Plate {plate_name} ({size_gb:.1f} GB archive, "
          f"{len(fmap)} FL TIFFs) ===")

    # Archives are deleted after extraction, so guard on the extracted images
    # (not the tar.gz) to make --all resumable: skip a plate whose FL TIFFs
    # are all already on disk instead of re-downloading the whole archive.
    present = _count_extracted(plate_name, fmap)
    if present >= len(fmap):
        print(f"  [SKIP] {plate_name}: all {len(fmap)} FL TIFFs already extracted")
        return True
    if present:
        print(f"  {present}/{len(fmap)} FL TIFFs already present, "
              f"fetching the remaining {len(fmap) - present}")

    if stream:
        # Stream the tar.gz straight from HTTP; the archive never lands on disk.
        # Trade-off: no resume if the connection drops mid-plate.
        print("  [stream] extracting FL channels without saving the archive")
        resp = _open_url(url)
        with tarfile.open(mode="r|gz", fileobj=resp) as tf:
            written, skipped = _extract_fl_from_tar(tf, fmap, plate_name)
    else:
        with _plate_lock(plate_name):
            written, skipped = _fetch_and_extract(
                url, plate_name, fmap, keep_archive=keep_archive,
                workers=workers, chunk_mb=chunk_mb)

    print(f"  Plate {plate_name}: {written} extracted, {skipped} already present")
    return True


def _fetch_and_extract(url, plate_name, fmap, keep_archive=False,
                       workers=6, chunk_mb=256):
    """Download a plate to disk and extract its FL channels.

    By default the FL TIFFs are read straight out of the downloaded chunk parts,
    so the 51.8 GB archive is never assembled: peak disk is one archive instead
    of two, and there is no assembly step to crash or corrupt. The parts are
    removed only after the extraction succeeded, keeping every failure mode
    resumable. ``--keep_archive`` still concatenates a real tar.gz first.
    """
    config.ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    archive = config.ARCHIVE_DIR / f"{plate_name}.tar.gz"

    if archive.exists() and archive.stat().st_size > 0:
        with tarfile.open(archive, "r:gz") as tf:
            written, skipped = _extract_fl_from_tar(tf, fmap, plate_name)
    elif workers > 1:
        parts_dir, paths = _fetch_parts(url, archive, num_workers=workers,
                                        chunk_size=chunk_mb * 1024 * 1024,
                                        desc=f"  dl {plate_name}")
        if keep_archive:
            _assemble_parts(paths, archive)
            shutil.rmtree(parts_dir, ignore_errors=True)
            with tarfile.open(archive, "r:gz") as tf:
                written, skipped = _extract_fl_from_tar(tf, fmap, plate_name)
        else:
            print(f"  extracting FL channels directly from {len(paths)} "
                  f"chunk parts (archive is never assembled)")
            with _PartsReader(paths) as raw:
                with tarfile.open(mode="r|gz",
                                  fileobj=io.BufferedReader(raw, 1 << 22)) as tf:
                    written, skipped = _extract_fl_from_tar(tf, fmap, plate_name)
            shutil.rmtree(parts_dir, ignore_errors=True)
            print(f"  deleted {parts_dir.name}")
            return written, skipped
    else:
        _download_resumable(url, archive, desc=f"  dl {plate_name}")
        with tarfile.open(archive, "r:gz") as tf:
            written, skipped = _extract_fl_from_tar(tf, fmap, plate_name)

    if not keep_archive:
        archive.unlink(missing_ok=True)
        print(f"  deleted {archive.name}")
    return written, skipped


def download_plates(plate_names, stream=False, keep_archive=False,
                    workers=6, chunk_mb=256):
    print(f"\n=== Downloading {len(plate_names)} plate(s) ===")
    ok = fail = 0
    for name in plate_names:
        if download_plate(name, stream=stream, keep_archive=keep_archive,
                          workers=workers, chunk_mb=chunk_mb):
            ok += 1
        else:
            fail += 1
    print(f"\n  Done: {ok} ok, {fail} failed")
    return fail == 0


# =========================================================================
# Status
# =========================================================================

def list_status():
    print("\n=== U2OS-Cell-Painting Download Status ===")
    for name in config.SMALL_FILES:
        p = config.METADATA_DIR / name
        state = f"OK ({p.stat().st_size/1024:.1f} KB)" if p.exists() else "MISSING"
        print(f"  {name}: {state}")

    if config.IMAGES_DIR.exists():
        total = 0
        for plate_dir in sorted(config.IMAGES_DIR.iterdir()):
            if not plate_dir.is_dir():
                continue
            n = sum(len(list((plate_dir / ch).glob("*.tif")))
                    for ch in config.CHANNEL_NAMES
                    if (plate_dir / ch).exists())
            total += n
            print(f"  {plate_dir.name}: {n} FL images")
        print(f"\n  Total FL images: {total:,} "
              f"(expect {config.N_SITES_PER_WELL * len(config.CHANNEL_NAMES)} per well)")
    else:
        print("  Images: NONE")


# =========================================================================
# CLI
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Download U2OS-Cell-Painting dataset (figshare 21378906)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--metadata_only", action="store_true",
                        help="Download only the metadata / benchmark tables")
    parser.add_argument("--plate", type=str, default=None,
                        help=f"Download a single plate (default {config.DEFAULT_PLATE})")
    parser.add_argument("--plates", type=str, nargs="+", default=None,
                        help="Download multiple plates")
    parser.add_argument("--all", action="store_true",
                        help=f"Download all {len(config.PLATE_NAMES)} plates (small->large)")
    parser.add_argument("--stream", action="store_true",
                        help="Stream-extract FL without saving the tar.gz (no resume)")
    parser.add_argument("--keep_archive", action="store_true",
                        help="Keep the downloaded tar.gz instead of deleting it")
    parser.add_argument("--workers", type=int, default=6,
                        help="Parallel Range connections for the disk mode "
                             "(default 6). Set 1 to fall back to single-stream.")
    parser.add_argument("--chunk_mb", type=int, default=256,
                        help="Chunk size in MB for parallel download (default 256)")
    parser.add_argument("--status", action="store_true",
                        help="Show download status")
    args = parser.parse_args()

    config.ensure_dirs()

    if args.status:
        list_status()
        return

    # Always ensure metadata first (needed to know which TIFFs are FL).
    download_metadata()
    if args.metadata_only:
        return

    if args.plate:
        download_plate(args.plate, stream=args.stream, keep_archive=args.keep_archive,
                       workers=args.workers, chunk_mb=args.chunk_mb)
    elif args.plates:
        download_plates(args.plates, stream=args.stream, keep_archive=args.keep_archive,
                        workers=args.workers, chunk_mb=args.chunk_mb)
    elif args.all:
        download_plates(config.PLATE_NAMES, stream=args.stream,
                        keep_archive=args.keep_archive,
                        workers=args.workers, chunk_mb=args.chunk_mb)
    else:
        download_plate(config.DEFAULT_PLATE, stream=args.stream,
                       keep_archive=args.keep_archive,
                       workers=args.workers, chunk_mb=args.chunk_mb)

    print("\n=== Status ===")
    list_status()


if __name__ == "__main__":
    main()
