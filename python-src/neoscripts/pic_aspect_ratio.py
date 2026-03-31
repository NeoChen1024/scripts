#!/usr/bin/env python

# This script moves images from the input directory to the corresponding output directory based on the aspect ratio of the images.
# The aspect ratio is calculated as width divided by height. The script categorizes images into three groups.
# It supports dry-run mode, conflict handling, and summary statistics.
# Multiprocessing is used with a worker initializer to avoid repeatedly passing shared arguments.

import os
import shutil
from collections import Counter
from multiprocessing import Pool

from PIL import Image
import click
import tqdm


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff", ".avif"}

OUTPUT_DIR = ""
DRY_RUN = False
SQUARE_THRESHOLD = 0.02
CONFLICT_POLICY = "skip"


def init_worker(output_dir, dry_run, square_threshold, conflict_policy):
    global OUTPUT_DIR, DRY_RUN, SQUARE_THRESHOLD, CONFLICT_POLICY
    OUTPUT_DIR = output_dir
    DRY_RUN = dry_run
    SQUARE_THRESHOLD = square_threshold
    CONFLICT_POLICY = conflict_policy


def process_image(image_path):
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            if height == 0:
                return "error", None
            aspect_ratio = width / height

            if abs(aspect_ratio - 1.0) <= SQUARE_THRESHOLD:
                category = "square"
            elif aspect_ratio > 1.0:
                category = "landscape"
            else:
                category = "portrait"

            category_dir = os.path.join(OUTPUT_DIR, category)
            os.makedirs(category_dir, exist_ok=True)
            dst_path = os.path.join(category_dir, os.path.basename(image_path))

            if os.path.exists(dst_path):
                if CONFLICT_POLICY == "skip":
                    return "destination_exists", category
                if CONFLICT_POLICY == "overwrite":
                    if DRY_RUN:
                        return "overwritten", category
                    if os.path.isdir(dst_path):
                        return "error", category
                    os.remove(dst_path)
                    shutil.move(image_path, dst_path)
                    return "overwritten", category
                return "error", category

            if DRY_RUN:
                return "moved", category

            shutil.move(image_path, dst_path)
            return "moved", category
    except (OSError, ValueError):
        return "invalid_image", None
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return "error", None


@click.command(help="Categorize images based on aspect ratio")
@click.option("--input-dir", required=True, help="Directory containing input images", type=click.Path(exists=True, file_okay=False))
@click.option(
    "--output-dir", required=True, help="Directory to save categorized images", type=click.Path(file_okay=False, writable=True)
)
@click.option("--dry-run", is_flag=True, help="Preview categorized results without moving files")
@click.option(
    "--square-threshold",
    default=0.02,
    show_default=True,
    type=click.FloatRange(min=0.0),
    help="Treat |aspect_ratio-1| within this value as square",
)
@click.option(
    "--conflict-policy",
    default="skip",
    show_default=True,
    type=click.Choice(["skip", "overwrite"], case_sensitive=False),
    help="Action when destination file already exists",
)
def main(input_dir, output_dir, dry_run, square_threshold, conflict_policy):
    os.makedirs(output_dir, exist_ok=True)
    for category in ("landscape", "portrait", "square"):
        os.makedirs(os.path.join(output_dir, category), exist_ok=True)

    entries = [entry for entry in os.scandir(input_dir) if entry.is_file()]
    image_paths = [entry.path for entry in entries if os.path.splitext(entry.name)[1].lower() in IMAGE_EXTS]
    skipped_non_image = len(entries) - len(image_paths)

    stats = Counter()
    moved_by_category = Counter()
    with Pool(
        initializer=init_worker,
        initargs=(output_dir, dry_run, square_threshold, conflict_policy.lower()),
    ) as pool:
        for status, category in tqdm.tqdm(
            pool.imap_unordered(process_image, image_paths),
            total=len(image_paths),
            desc="Processing",
        ):
            stats[status] += 1
            if status == "moved" and category:
                moved_by_category[category] += 1

    click.echo("Summary:")
    click.echo(f"  Total files scanned: {len(entries)}")
    click.echo(f"  Eligible images: {len(image_paths)}")
    click.echo(f"  Skipped non-image files: {skipped_non_image}")
    click.echo(f"  Destination exists (skip): {stats['destination_exists']}")
    click.echo(f"  Invalid image files: {stats['invalid_image']}")
    click.echo(f"  Errors: {stats['error']}")
    click.echo(f"  Moved: {stats['moved']}")
    click.echo(f"    - Moved to square: {moved_by_category['square']}")
    click.echo(f"    - Moved to landscape: {moved_by_category['landscape']}")
    click.echo(f"    - Moved to portrait: {moved_by_category['portrait']}")
    click.echo(f"  Overwritten: {stats['overwritten']}")
    click.echo(f"  Renamed: {stats['renamed']}")


if __name__ == "__main__":
    main()
