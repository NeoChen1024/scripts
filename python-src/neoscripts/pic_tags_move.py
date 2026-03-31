#!/usr/bin/env python

# This script moves images from the input directory to the corresponding output directory based on matched tags.
# For each image, there is a corresponding .txt file with the same name (but .txt extension) that contains tags.
# Tags in the .txt file are expected to be comma-separated, e.g. "tag1, tag2, tag3".
# The script moves the images and their corresponding .txt files to the output directory when any specified tag or regex matches.
# It uses multiprocessing and reports real completion progress.

import os
import re
import shutil
from collections import Counter
from multiprocessing import Pool

import click
import tqdm

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff", ".avif"}

USE_REGEX = False
EXACT_TAGS = set()
REGEX_PATTERNS = []
OUTPUT_DIR = ""
DRY_RUN = False


def init_worker(tags, regex, output_dir, dry_run):
    global USE_REGEX, EXACT_TAGS, REGEX_PATTERNS, OUTPUT_DIR, DRY_RUN
    USE_REGEX = regex
    OUTPUT_DIR = output_dir
    DRY_RUN = dry_run
    if regex:
        REGEX_PATTERNS = [re.compile(tag) for tag in tags]
        EXACT_TAGS = set()
    else:
        EXACT_TAGS = set(tags)
        REGEX_PATTERNS = []


def process_image(image_path):
    try:
        txt_path = os.path.splitext(image_path)[0] + ".txt"
        if not os.path.exists(txt_path):
            return "missing_txt"

        with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
            image_tags = [tag.strip() for tag in f.read().strip().split(",") if tag.strip()]

        if USE_REGEX:
            match_found = any(pattern.search(image_tag) for pattern in REGEX_PATTERNS for image_tag in image_tags)
        else:
            match_found = bool(EXACT_TAGS.intersection(image_tags))

        if not match_found:
            return "no_match"

        dst_path = os.path.join(OUTPUT_DIR, os.path.basename(image_path))
        if os.path.exists(dst_path):
            return "destination_exists"

        if DRY_RUN:
            return "would_move"

        shutil.move(image_path, dst_path)
        shutil.move(txt_path, os.path.join(OUTPUT_DIR, os.path.basename(txt_path)))
        return "moved"
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return "error"


@click.command(help="Move images based on matched tags")
@click.option(
    "--input-dir", "-i", required=True, help="Directory containing input images", type=click.Path(exists=True, file_okay=False)
)
@click.option(
    "--output-dir", "-o", required=True, help="Directory to save moved images", type=click.Path(file_okay=False, writable=True)
)
@click.option("--tags", "-t", required=True, help="List of tags to match (e.g. 'tag1' 'tag2' 'tag3')", type=str, multiple=True)
@click.option("--regex", "-r", is_flag=True, help="Interpret tags as regular expressions")
@click.option("--dry-run", "-n", is_flag=True, help="Preview matched files without moving them")
def main(input_dir, output_dir, tags, regex, dry_run):
    os.makedirs(output_dir, exist_ok=True)

    image_paths = [
        entry.path for entry in os.scandir(input_dir) if entry.is_file() and os.path.splitext(entry.name)[1].lower() in IMAGE_EXTS
    ]

    stats = Counter()
    with Pool(initializer=init_worker, initargs=(tags, regex, output_dir, dry_run)) as pool:
        for status in tqdm.tqdm(
            pool.imap_unordered(process_image, image_paths),
            total=len(image_paths),
            desc="Processing",
        ):
            stats[status] += 1

    click.echo("Summary:")
    click.echo(f"  Total images scanned: {len(image_paths)}")
    click.echo(f"  Missing .txt: {stats['missing_txt']}")
    click.echo(f"  Tag not matched: {stats['no_match']}")
    click.echo(f"  Destination exists: {stats['destination_exists']}")
    click.echo(f"  Errors: {stats['error']}")
    if dry_run:
        click.echo(f"  Would move: {stats['would_move']}")
    else:
        click.echo(f"  Moved: {stats['moved']}")


if __name__ == "__main__":
    main()
