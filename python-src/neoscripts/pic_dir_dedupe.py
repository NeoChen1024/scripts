#!/usr/bin/env python
# -*- coding: utf-8 -*-
# A simple script to help deduplicate my pic directory
# 4 different categories:
# * "Portrait" -- images with subjects that are primarily vertical in orientation
# * "Landscape" -- images with subjects that are primarily horizontal in orientation
# * "Wallpaper" -- images that are hand-picked wallpapers, inspected manually to be of high quality
# * "VWallpaper" -- same as above but vertical (for phones)
#
# The script will scan through the 4 directories, look for duplicates based on file name.
# "Portrait" and "VWallpaper" are processed together, as are "Landscape" and "Wallpaper".
# If a file has duplicates, only one copy (preferably in Wallpaper or VWallpaper) is kept, and the rest are moved to a separate directory for manual inspection.
# If duplicates are found, it will move them to a "duplicates" directory for manual inspection
# Finally, it will print out a summary of the duplicates found.
# Use SHA512 hash for better collision resistance.
# -n / --dry-run : do not actually move files, just print what would be done
# Usage:
# python pic_dir_dedupe.py [-n] /path/to/pic_directory /path/to/dupe_directory


import os
import sys
from typing import Dict, List, Tuple

from rich.console import Console
from rich.text import Text
import click

PORTRAIT_DIRNAME = "Portrait"
LANDSCAPE_DIRNAME = "Landscape"
WALLPAPER_DIRNAME = "Wallpaper"
VWALLPAPER_DIRNAME = "VWallpaper"

console = Console()


def _iter_category_files(root: str, category: str) -> List[str]:
    """Return a sorted list of file paths under the given category directory.

    Only regular files are returned; subdirectories are ignored.
    If the category directory does not exist, an empty list is returned.
    """

    category_dir = os.path.join(root, category)
    if not os.path.isdir(category_dir):
        return []

    entries: List[str] = []
    for name in os.listdir(category_dir):
        path = os.path.join(category_dir, name)
        if os.path.isfile(path):
            entries.append(path)
    entries.sort()
    return entries


def _group_by_name(files: List[str]) -> Dict[str, List[str]]:
    """Group file paths by their basename."""

    groups: Dict[str, List[str]] = {}
    for path in files:
        name = os.path.basename(path)
        groups.setdefault(name, []).append(path)
    return groups


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _move_file(src: str, dst_dir: str, dry_run: bool) -> None:
    _ensure_dir(dst_dir)
    dst = os.path.join(dst_dir, os.path.basename(src))
    if dry_run:
        console.print(f"[DRY-RUN] mv '{src}' -> '{dst}'", style="yellow")
        return
    console.print(f"mv '{src}' -> '{dst}'", style="green")
    os.rename(src, dst)


def _find_and_handle_dupes(files: List[str], dupedir: str, dry_run: bool) -> Tuple[int, int]:
    """Given a list of files, find duplicates and move them.

    Deduplication stages:
    1) Group by file name.
    2) For each name group with more than one file, keep a *preferred*
       copy (Wallpaper/VWallpaper first, then others) and move the
       rest into ``dupedir``.

    Returns (num_duplicate_sets, num_files_moved).
    """

    name_groups = _group_by_name(files)

    duplicate_sets = 0
    moved_files = 0

    def pref_key(path: str) -> Tuple[int, str]:
        """Return a sort key where Wallpaper/VWallpaper are preferred.

        Lower tuple sorts first, so preferred items get smaller scores.
        """

        dirname = os.path.basename(os.path.dirname(path))
        if dirname in (WALLPAPER_DIRNAME, VWALLPAPER_DIRNAME):
            score = 0
        else:
            score = 1
        return (score, path)

    for name, paths in sorted(name_groups.items()):
        if len(paths) < 2:
            continue

        # We have duplicates purely by filename.
        duplicate_sets += 1

        # Choose the best file to keep based on preference key.
        sorted_paths = sorted(paths, key=pref_key)
        keep = sorted_paths[0]
        dupes = sorted_paths[1:]

        console.print(f"Duplicate set (name='{name}'):", style="bold magenta")
        console.print(f"  Keeping: {keep}", style="bold green")
        for d in dupes:
            console.print(f"  Duplicate: {d}", style="red")
            _move_file(d, dupedir, dry_run)
            moved_files += 1

    return duplicate_sets, moved_files


@click.command()
@click.option("-n", "--dry-run", is_flag=True, help="Do not actually move files, just print what would be done")
@click.argument("inputdir", type=click.Path(exists=True, file_okay=False))
@click.argument("dupedir", type=click.Path(writable=True, file_okay=False))
def __main__(dry_run, inputdir, dupedir):
    """Deduplicate picture directories under ``inputdir``.

    ``inputdir`` is expected to contain ``Portrait``, ``Landscape``,
    ``Wallpaper`` and ``VWallpaper`` directories.  Deduplication is
    performed in two groups: Portrait+VWallpaper and
    Landscape+Wallpaper.

    For files that have the same name across the group, only one copy
    is kept, preferring those located in ``Wallpaper``/``VWallpaper``.
    All other copies are moved to ``dupedir`` for manual inspection.
    With ``--dry-run`` nothing is moved, only actions are printed.
    """

    inputdir = os.path.abspath(inputdir)
    dupedir = os.path.abspath(dupedir)

    portrait_files = _iter_category_files(inputdir, PORTRAIT_DIRNAME)
    landscape_files = _iter_category_files(inputdir, LANDSCAPE_DIRNAME)
    wallpaper_files = _iter_category_files(inputdir, WALLPAPER_DIRNAME)
    vwallpaper_files = _iter_category_files(inputdir, VWALLPAPER_DIRNAME)

    console.print(f"Scanning directories under: {inputdir}", style="bold cyan")
    console.print(f"  {PORTRAIT_DIRNAME}: {len(portrait_files)} files")
    console.print(f"  {VWALLPAPER_DIRNAME}: {len(vwallpaper_files)} files")
    console.print(f"  {LANDSCAPE_DIRNAME}: {len(landscape_files)} files")
    console.print(f"  {WALLPAPER_DIRNAME}: {len(wallpaper_files)} files")

    # Process Portrait + VWallpaper together, and Landscape + Wallpaper
    # together, as described in the header comments.
    total_sets = 0
    total_moved = 0

    console.print("\nProcessing Portrait + VWallpaper group...", style="bold blue")
    s, m = _find_and_handle_dupes(portrait_files + vwallpaper_files, dupedir, dry_run)
    total_sets += s
    total_moved += m

    console.print("\nProcessing Landscape + Wallpaper group...", style="bold blue")
    s, m = _find_and_handle_dupes(landscape_files + wallpaper_files, dupedir, dry_run)
    total_sets += s
    total_moved += m

    console.print("\nSummary:", style="bold underline")
    console.print(f"  Duplicate sets found: {total_sets}")
    console.print(f"  Files moved: {total_moved}")
    if dry_run:
        console.print("  (dry run: no files were actually moved)", style="yellow")


if __name__ == "__main__":
    __main__()
