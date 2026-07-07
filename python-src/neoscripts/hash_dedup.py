#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Simple content-hash based file deduplication.

Recursively walks an input directory, computes the BLAKE3 hash of every
regular file, and removes duplicates keeping either the oldest or newest
copy by mtime. Supports a dry-run mode to preview actions without
deleting anything.

Hashing can be parallelised across multiple I/O threads (``-t`` /
``--io-threads``). The threaded pipeline is tuned for SSD / ZFS pools
holding many fragmented files: it splits work into ``N-1`` small-file
worker threads plus a single large-file worker behind a dispatch thread.
While a large file is being hashed, dispatch of *new* small files is
frozen so the big sequential read is not interleaved with small random
reads; the already-queued small files keep draining in parallel.

Usage:
    hash_dedup.py [-n] [--keep oldest|newest] [-t N] [--threshold M] INPUTDIR
"""

import enum
import os
import queue
import stat as stat_mod
import threading
from typing import Dict, List, Optional, Tuple

import click
from tqdm import tqdm

try:
    import blake3 as _blake3
except ImportError:  # pragma: no cover
    raise SystemExit("ERROR: blake3 package is not installed. " "Install it with `uv pip install blake3` or `pip install blake3`.")

# Distinct sentinel objects so they can never collide with a real path/None.
_SMALL_SENTINEL = object()
_BIG_SENTINEL = object()


class KeepPolicy(enum.Enum):
    OLDEST = "oldest"
    NEWEST = "newest"


def _compute_blake3(path: str, chunk_size: int = 1 << 20) -> str:
    """Return the hex digest of the BLAKE3 hash of the file at ``path``."""
    h = _blake3.blake3()
    try:
        with open(path, "rb") as fp:
            while True:
                chunk = fp.read(chunk_size)
                if not chunk:
                    break
                h.update(chunk)
    except OSError as exc:
        raise OSError(f"Cannot read file '{path}': {exc}") from exc
    return h.hexdigest()


def _walk_tasks(root: str) -> List[Tuple[str, int]]:
    """Return ``(path, size)`` for every regular file under ``root``.

    Order is the natural ``os.walk`` order (not sorted). Symlinks are
    skipped (``lstat`` so we never follow links or double-count).
    """
    tasks: List[Tuple[str, int]] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            path = os.path.join(dirpath, name)
            try:
                st = os.stat(path, follow_symlinks=False)
            except OSError:
                continue
            if not stat_mod.S_ISREG(st.st_mode):
                # S_ISLNK and other non-regular types are skipped.
                continue
            tasks.append((path, st.st_size))
    return tasks


def _keep_key(path: str, policy: KeepPolicy) -> float:
    """Return an mtime sort key; the *smallest* value is the one to keep."""
    mtime = os.path.getmtime(path)
    if policy is KeepPolicy.NEWEST:
        # For "keep newest" we want the largest mtime to sort first,
        # so negate it so it becomes the smallest key.
        return -mtime
    return mtime


def _group_dupes(
    path_to_digest: Dict[str, Optional[str]],
    policy: KeepPolicy,
) -> Dict[str, List[str]]:
    """Group paths by their non-null digest, returning only duplicate sets.

    Within each set the paths are sorted so that the first entry is the
    one to keep (oldest or newest, depending on ``policy``).
    """
    by_hash: Dict[str, List[str]] = {}
    for path, digest in path_to_digest.items():
        if digest is None:
            continue
        by_hash.setdefault(digest, []).append(path)
    dupe_sets: Dict[str, List[str]] = {}
    for digest, paths in by_hash.items():
        if len(paths) < 2:
            continue
        dupe_sets[digest] = sorted(paths, key=lambda p: _keep_key(p, policy))
    return dupe_sets


def _hash_files_serial(tasks: List[Tuple[str, int]]) -> Dict[str, Optional[str]]:
    """Hash every file on the current thread, returning {path: digest-or-None}."""
    results: Dict[str, Optional[str]] = {}
    for path, _size in tqdm(tasks, desc="Hashing", unit="file"):
        try:
            results[path] = _compute_blake3(path)
        except OSError as exc:
            tqdm.write(f"ERROR hashing {path}: {exc}")
            results[path] = None
    return results


def _small_worker(small_q: "queue.Queue", result_q: "queue.Queue") -> None:
    """Pull small files off ``small_q``, hash them, push results to ``result_q``."""
    while True:
        item = small_q.get()
        if item is _SMALL_SENTINEL:
            break
        path = item
        try:
            digest = _compute_blake3(path)
        except OSError as exc:
            result_q.put((path, None, str(exc)))
        else:
            result_q.put((path, digest, None))


def _big_worker(big_q: "queue.Queue", result_q: "queue.Queue", big_idle: threading.Event) -> None:
    """Pull large files off ``big_q``, hash them, push results to ``result_q``.

    ``big_idle`` is *set* only when, after finishing a file, the big queue
    is empty -- i.e. no further large file has been committed by the
    dispatch thread. This avoids a race window between consecutive large
    files where ``big_idle`` would otherwise briefly rise and let a small
    file be dispatched just before the next big file starts. The dispatch
    thread is responsible for *clearing* ``big_idle`` before it hands a
    large file to this worker.
    """
    while True:
        item = big_q.get()
        if item is _BIG_SENTINEL:
            big_idle.set()
            break
        path = item
        try:
            digest = _compute_blake3(path)
        except OSError as exc:
            result_q.put((path, None, str(exc)))
        else:
            result_q.put((path, digest, None))
        # If another big file is already queued, keep big_idle cleared (the
        # dispatch thread will have cleared it before q.put()). Otherwise we
        # are idle, so lift the freeze and let small files flow again.
        if big_q.empty():
            big_idle.set()


def _dispatch(
    tasks: List[Tuple[str, int]],
    threshold_bytes: int,
    small_q: "queue.Queue",
    big_q: "queue.Queue",
    big_idle: threading.Event,
    num_small: int,
) -> None:
    """Hand files out to the small/big workers in walk order.

    Rules:
    * large file (size >= threshold): clear ``big_idle`` first, then
      ``big_q.put`` (maxsize 1 ensures at most one large file in flight
      and blocks the dispatch thread until the big worker is ready).
    * small file: if a large file is in flight (``big_idle`` cleared),
      wait for it to be set again before putting onto ``small_q``; the
      bounded ``small_q`` provides natural backpressure otherwise.
    After all files are dispatched, per-worker sentinels are sent so the
    workers can exit.
    """
    for path, size in tasks:
        if size >= threshold_bytes:
            big_idle.clear()
            big_q.put(path)
        else:
            # Freeze: do not feed small workers while a big file is running.
            big_idle.wait()
            small_q.put(path)

    # Tell every worker to stop.
    for _ in range(num_small):
        small_q.put(_SMALL_SENTINEL)
    big_q.put(_BIG_SENTINEL)


def _hash_files_threaded(
    tasks: List[Tuple[str, int]],
    n_workers: int,
    threshold_bytes: int,
) -> Dict[str, Optional[str]]:
    """Hash files concurrently with N-1 small + 1 big worker behind a dispatcher.

    Returns ``{path: digest-or-None}``. The dispatch thread walks the
    task list in order; small files go to a bounded small-file queue
    shared by ``N-1`` small workers, large files go to a single-slot
    big-file queue served by one big worker. While a large file is in
    flight, new small files are not dispatched (already-queued small
    files keep draining).
    """
    num_small = n_workers - 1
    small_q: "queue.Queue" = queue.Queue(maxsize=2 * num_small)
    big_q: "queue.Queue" = queue.Queue(maxsize=1)
    result_q: "queue.Queue" = queue.Queue()
    big_idle = threading.Event()
    big_idle.set()  # initially idle: small files may be dispatched immediately

    dispatch_thread = threading.Thread(
        target=_dispatch,
        args=(tasks, threshold_bytes, small_q, big_q, big_idle, num_small),
        name="dispatch",
    )
    small_threads = [threading.Thread(target=_small_worker, args=(small_q, result_q), name=f"small-{i}") for i in range(num_small)]
    threads = [dispatch_thread, *small_threads]
    threads.append(threading.Thread(target=_big_worker, args=(big_q, result_q, big_idle), name="big"))
    for t in threads:
        t.start()

    # Main thread collects results via a single progress bar.
    total = len(tasks)
    results: Dict[str, Optional[str]] = {}
    with tqdm(total=total, desc="Hashing", unit="file") as bar:
        for _ in range(total):
            path, digest, err = result_q.get()
            if err is not None:
                bar.write(f"ERROR hashing {path}: {err}")
            results[path] = digest
            bar.update(1)

    for t in threads:
        t.join()
    return results


@click.command()
@click.option(
    "-n",
    "--dry-run",
    is_flag=True,
    help="Do not actually delete files, just print what would be done.",
)
@click.option(
    "--keep",
    type=click.Choice([KeepPolicy.OLDEST.value, KeepPolicy.NEWEST.value], case_sensitive=False),
    default=KeepPolicy.OLDEST.value,
    show_default=True,
    help="Which duplicate copy to keep based on mtime.",
)
@click.option(
    "-t",
    "--io-threads",
    type=click.IntRange(min=1),
    default=1,
    show_default=True,
    help=(
        "Number of worker threads for parallel hashing. 1 = single-threaded. "
        "N>=2 spawns N-1 small-file workers + 1 large-file worker + 1 dispatch thread. "
        "Tuned for SSD / ZFS pools with many fragmented files."
    ),
)
@click.option(
    "--threshold",
    type=click.IntRange(min=0),
    default=4,
    show_default=True,
    help="File size threshold (MiB); files >= this are handled by the single large-file worker.",
)
@click.argument("inputdir", type=click.Path(exists=True, file_okay=False, dir_okay=True))
def cli(dry_run: bool, keep: str, io_threads: int, threshold: int, inputdir: str):
    """Remove duplicate files under INPUTDIR by BLAKE3 content hash.

    Recursively walks INPUTDIR, hashes every regular file, and deletes
    duplicates. By default the oldest copy (by mtime) is kept; use
    ``--keep newest`` to keep the newest copy instead. Parallel hashing
    is opt-in via ``-t`` / ``--io-threads``.
    """
    inputdir = os.path.abspath(inputdir)
    policy = KeepPolicy(keep.lower())
    threshold_bytes = threshold * 1024 * 1024

    click.echo(f"Scanning: {inputdir}")
    click.echo(f"Policy: keep {policy.value}")
    click.echo(f"I/O threads: {io_threads}" + (f" (threshold: {threshold} MiB)" if io_threads >= 2 else ""))
    if dry_run:
        click.echo("[DRY-RUN] No files will be deleted.")

    tasks = _walk_tasks(inputdir)
    click.echo(f"Found {len(tasks)} regular files.")

    if not tasks:
        click.echo("No files to process. Nothing to do.")
        return

    if io_threads >= 2:
        path_to_digest = _hash_files_threaded(tasks, io_threads, threshold_bytes)
    else:
        path_to_digest = _hash_files_serial(tasks)

    dupe_sets = _group_dupes(path_to_digest, policy)
    error_count = sum(1 for v in path_to_digest.values() if v is None)

    if not dupe_sets:
        click.echo("No duplicates found.")
        if error_count:
            click.echo(f"  ({error_count} file(s) could not be hashed)")
        return

    total_deleted = 0
    n_dups = sum(len(paths) - 1 for paths in dupe_sets.values())
    with tqdm(total=n_dups, desc="Deleting", unit="file") as bar:
        for _digest, paths in sorted(dupe_sets.items()):
            keep_path = paths[0]
            delete_paths = paths[1:]
            bar.write(f"Duplicate set [keep={keep_path}]:")
            for d in delete_paths:
                if dry_run:
                    bar.write(f"  [DRY-RUN] would delete: {d}")
                else:
                    try:
                        os.remove(d)
                        bar.write(f"  deleted: {d}")
                    except OSError as exc:
                        bar.write(f"  ERROR deleting {d}: {exc}")
                        continue
                total_deleted += 1
                bar.update(1)

    click.echo("")
    click.echo("Summary:")
    click.echo(f"  Duplicate sets: {len(dupe_sets)}")
    click.echo(f"  Files {'would be ' if dry_run else ''}deleted: {total_deleted}")
    if error_count:
        click.echo(f"  Files failed to hash: {error_count}")
    if dry_run:
        click.echo("  (dry run: no files were actually deleted)")


if __name__ == "__main__":
    cli()
