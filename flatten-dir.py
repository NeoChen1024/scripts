#!/usr/bin/env python
# -*- coding: utf-8 -*-
# A simple script to flatten a dataset's directories
# For example:
# data/dir1/dir2/123.png --becomes---> output/dir1_dir2_123.png
# args: input_dir flattened_output_dir


import os
from os import walk
import sys
import argparse
import shutil
from openai import files
from rich import print
from rich.console import Console
from rich.traceback import install

install()
console = Console()


def filename_flatten(path, strip_level=0):
    path = os.path.normpath(path).split(os.sep)
    if strip_level < 0 or strip_level >= len(path):
        console.log(f"Invalid strip level {strip_level} for path {path}")
    dirs = path[strip_level:-1]  # get directories
    directories = "_".join(dirs)
    filename = path[-1]
    return f"{directories}_{filename}"


def traverse_dir(inputdir, outputdir, strip_level, function=shutil.copy2):
    for dirpath, dirnames, filenames in walk(inputdir):
        files = [os.path.join(dirpath, file) for file in filenames]
        for file in files:
            new_name = filename_flatten(file, strip_level)
            new_path = os.path.join(outputdir, new_name)
            console.log(f"{file} ==> {new_path}")
            function(file, new_path)


def parsearg():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input directory to flatten")
    parser.add_argument("output", help="Output directory")
    parser.add_argument(
        "-s",
        "--strip",
        type=int,
        help="How many levels of directories to strip in the output file name. (default: dynamic)",
    )
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        default="copy",
        choices=["copy", "move", "symlink", "hardlink"],
        help="Choose between modes. (default: copy)",
    )
    return parser.parse_args()


def main():
    args = parsearg()
    inputdir = args.input
    outputdir = args.output
    strip_level = args.strip
    # Modes: 'copy', 'move', 'symlink', 'hardlink'
    if args.mode == "copy":
        func = shutil.copy2
    elif args.mode == "move":
        func = shutil.move
    elif args.mode == "symlink":
        func = os.symlink
    elif args.mode == "hardlink":
        func = os.link
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")

    os.makedirs(outputdir, exist_ok=True)

    # calculate how many levels of directories to strip
    if strip_level is None:
        strip_level = len(os.path.normpath(inputdir).split(os.path.sep))

    console.log(
        f"Flattening {inputdir} --> {outputdir} and strip {strip_level} levels of directories"
    )
    traverse_dir(inputdir, outputdir, strip_level, func)


if __name__ == "__main__":
    main()
