#!/usr/bin/env python
# -*- coding: utf-8 -*-
# A simple script to help deduplicate the importing dir with my pic directory
# Usage:
# python pic_dir_dedupe.py [-n] /path/to/pic_directory /path/to/import_directory /path/to/dupe_directory

import os
import sys
from typing import Dict, List, Tuple

from neoscripts.pic_dir_dedupe import _iter_category_files, _find_and_handle_dupes
from rich.console import Console
import rich.traceback
import click

ANIME_DIRNAME = "Anime"
WALLPAPER_DIRNAME = "Wallpaper"
VWALLPAPER_DIRNAME = "VWallpaper"

console = Console()
rich.traceback.install(console=console)

@click.command()
@click.option(
    "-n",
    "--dry-run",
    is_flag=True,
    help="Perform a trial run with no changes made",
)
@click.argument("inputdir", type=click.Path(exists=True, file_okay=False))
@click.argument("importdir", type=click.Path(exists=True, file_okay=False))
@click.argument("dupedir", type=click.Path())
def __main__(dry_run, inputdir, importdir, dupedir):
    """Remove duplicates from the IMPORTDIR against the picture directories under
    ``inputdir``.

    ``inputdir`` is expected to contain ``Anime``, ``Wallpaper`` and
    ``VWallpaper`` directories.  Deduplication is performed in 3 groups:
    Anime+importdir, Wallpaper+importdir, and VWallpaper+importdir."""

    inputdir = os.path.abspath(inputdir)
    importdir = os.path.abspath(importdir)
    dupedir = os.path.abspath(dupedir)

    anime_files = _iter_category_files(inputdir, ANIME_DIRNAME)
    wallpaper_files = _iter_category_files(inputdir, WALLPAPER_DIRNAME)
    vwallpaper_files = _iter_category_files(inputdir, VWALLPAPER_DIRNAME)
    import_files = _iter_category_files(importdir, "")

    console.print(f"Scanning directories under: {inputdir}", style="bold cyan")
    console.print(f"  {ANIME_DIRNAME}: {len(anime_files)} files")
    console.print(f"  {VWALLPAPER_DIRNAME}: {len(vwallpaper_files)} files")
    console.print(f"  {WALLPAPER_DIRNAME}: {len(wallpaper_files)} files")
    console.print(f"Import directory: {importdir}: {len(import_files)} files")

    total_sets = 0
    total_moved = 0

    console.print("\nProcessing Anime + ImportDir group...", style="bold blue")
    s, m = _find_and_handle_dupes(anime_files + import_files, dupedir, dry_run)
    total_sets += s
    total_moved += m

    console.print("\nProcessing Wallpaper + ImportDir group...", style="bold blue")
    s, m = _find_and_handle_dupes(wallpaper_files + import_files, dupedir, dry_run)
    total_sets += s
    total_moved += m

    console.print("\nProcessing VWallpaper + ImportDir group...", style="bold blue")
    s, m = _find_and_handle_dupes(vwallpaper_files + import_files, dupedir, dry_run)
    total_sets += s
    total_moved += m

    console.print("\nSummary:", style="bold underline")
    console.print(f"  Duplicate sets found: {total_sets}")
    console.print(f"  Files moved: {total_moved}")
    if dry_run:
        console.print("  (dry run: no files were actually moved)", style="yellow")
        
if __name__ == "__main__":
    __main__()