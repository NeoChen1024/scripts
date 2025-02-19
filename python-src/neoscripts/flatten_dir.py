#!/usr/bin/env python
# -*- coding: utf-8 -*-
# A simple script to flatten a dataset's directories
# For example:
# data/dir1/dir2/123.png --becomes---> output/dir1_dir2_123.png
# args: input_dir flattened_output_dir


import argparse
import os
import shutil
import sys
from shutil import copy2, move
from typing import Callable

import click
from reflink import reflink

modes = {
    "copy": copy2,
    "move": move,
    "symlink": os.symlink,
    "hardlink": os.link,
    "reflink": reflink,
}


def filename_flatten(path: str, strip_level: int = 0, delimiter: str = "_") -> str:
    path = os.path.normpath(path).split(os.sep)
    if strip_level < 0 or strip_level >= len(path):
        print(f"Invalid strip level {strip_level} for path {path}")
    dirs = path[strip_level:-1]  # get directories
    directories = delimiter.join(dirs)
    filename = path[-1]
    return f"{directories}_{filename}"


def traverse_dir(inputdir: str, outputdir: str, strip_level: int, function: Callable, delimiter: str) -> None:
    for dirpath, dirnames, filenames in os.walk(inputdir):
        files = [os.path.join(dirpath, file) for file in filenames]
        for file in files:
            new_name = filename_flatten(file, strip_level, delimiter)
            new_path = os.path.join(outputdir, new_name)
            print(f"\t{file} ==> {new_path}")
            function(file, new_path)


@click.command()
@click.argument("inputdir", type=click.Path(exists=True, file_okay=False))
@click.argument("outputdir", type=click.Path())
@click.option(
    "-s",
    "--strip",
    type=int,
    default=None,
    help="How many levels of directories to strip in the output file name. (default: dynamic)",
)
@click.option(
    "-m",
    "--mode",
    type=click.Choice(list(modes.keys())),
    default="copy",
    help="Choose between modes. (default: copy)",
)
@click.option("-d", "--delimiter", type=str, default="_", help="Delimiter for directory names")
def __main__(inputdir, outputdir, strip, mode, delimiter):
    os.makedirs(outputdir, exist_ok=True)

    # calculate how many levels of directories to strip
    strip_level = strip
    if strip_level is None:
        strip_level = len(os.path.normpath(inputdir).split(os.path.sep))

    click.echo(f"Flattening {inputdir} --> {outputdir} and strip {strip_level} levels of directories")

    func = modes[mode]
    traverse_dir(inputdir, outputdir, strip_level, func, delimiter)


if __name__ == "__main__":
    __main__()
