#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Flatten a dataset's nested directory structure into a single output directory.

Example:
    data/dir1/dir2/123.png --becomes---> output/dir1_dir2_123.png

Usage:
    python flatten_dir.py INPUT_DIR OUTPUT_DIR [-s STRIP] [-m MODE] [-d DELIMITER]
"""

import os
from shutil import copy2, move
from typing import Callable, Dict

import click
from reflink import reflink

# Mapping of operation mode names to their corresponding file-operation functions
MODES: Dict[str, Callable] = {
    "copy": copy2,
    "move": move,
    "symlink": os.symlink,
    "hardlink": os.link,
    "reflink": reflink,
}


def filename_flatten(path: str, strip_level: int = 0, delimiter: str = "_") -> str:
    """Convert a file path into a flattened filename.

    Replaces directory separators in *path* with *delimiter*, skipping the
    first *strip_level* leading components.

    Args:
        path: Original file path.
        strip_level: Number of leading path components to discard.
        delimiter: String to insert between directory names.

    Returns:
        The flattened filename.

    Raises:
        ValueError: If *strip_level* is out of range.
    """
    parts = os.path.normpath(path).split(os.sep)

    if strip_level < 0 or strip_level >= len(parts):
        raise ValueError(
            f"Invalid strip level {strip_level} for path {path!r} "
            f"(path has {len(parts)} components, valid range: 0..{len(parts) - 1})"
        )

    directories = delimiter.join(parts[strip_level:-1])
    filename = parts[-1]
    return f"{directories}_{filename}"


def traverse_dir(
    input_dir: str,
    output_dir: str,
    strip_level: int,
    file_op: Callable,
    delimiter: str,
) -> None:
    """Walk *input_dir* and apply *file_op* to flatten every file into *output_dir*.

    Args:
        input_dir: Source directory to walk recursively.
        output_dir: Target directory for flattened files.
        strip_level: Number of leading path components to strip from names.
        file_op: Operation to apply on each (source, dest) pair.
        delimiter: Separator used between directory names in the output.
    """
    for dirpath, _dirnames, filenames in os.walk(input_dir):
        for filename in filenames:
            src = os.path.join(dirpath, filename)
            new_name = filename_flatten(src, strip_level, delimiter)
            dst = os.path.join(output_dir, new_name)
            click.echo(f"\t{src} ==> {dst}")
            file_op(src, dst)


@click.command()
@click.argument("input_dir", type=click.Path(exists=True, file_okay=False))
@click.argument("output_dir", type=click.Path())
@click.option(
    "-s",
    "--strip",
    type=int,
    default=None,
    help="How many levels of directories to strip in the output file name. "
    "(default: determined automatically from input_dir depth)",
)
@click.option(
    "-m",
    "--mode",
    type=click.Choice(list(MODES.keys())),
    default="copy",
    help="Choose the operation mode. (default: copy)",
)
@click.option(
    "-d",
    "--delimiter",
    type=str,
    default="_",
    help="Delimiter for joining directory names. (default: _)",
)
def cli(input_dir: str, output_dir: str, strip: int | None, mode: str, delimiter: str) -> None:
    """Flatten a nested directory tree into a single flat output directory."""
    os.makedirs(output_dir, exist_ok=True)

    strip_level: int = strip if strip is not None else len(os.path.normpath(input_dir).split(os.sep))

    click.echo(f"Flattening {input_dir} --> {output_dir} " f"and stripping {strip_level} levels of directories")

    traverse_dir(input_dir, output_dir, strip_level, MODES[mode], delimiter)


if __name__ == "__main__":
    cli()
