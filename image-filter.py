#!/usr/bin/env python
import os
from pprint import pprint as pp
import shutil
import multiprocessing as mp

import click
import filetype
from PIL import Image
from torch import mul
from transformers import pipeline


def load_image(file_path):
    i = Image.open(file_path)
    i.convert("RGB")  # force it to load into memory
    # resize to fit into 1024x1024, greatly speeds up processing
    if i.width > 1024 or i.height > 1024:
        i.thumbnail((1024, 1024))
    return i


@click.command(help="Filter images using a pre-trained ViT model")
@click.option(
    "--input-dir",
    "-i",
    required=True,
    type=click.Path(dir_okay=True, file_okay=False, readable=True),
    help="Input directory containing images",
)
@click.option(
    "--output-dir",
    "-o",
    required=True,
    type=click.Path(dir_okay=True, file_okay=False, writable=True),
    help="Output directory to save filtered images",
)
@click.option(
    "--rejected-dir",
    "-r",
    type=click.Path(dir_okay=True, file_okay=False, writable=True),
    required=True,
    help="Output directory to save rejected images",
)
@click.option(
    "--threshold",
    "-t",
    default=0.7,
    type=click.FloatRange(0, 1, clamp=True),
    help="Threshold for filtering images (0-1)",
)
@click.option(
    "--batch-size", "-b", default=8, type=int, help="Batch size for filtering images"
)
@click.option(
    "--model", "-m", default="NeoChen1024/aesthetic-shadow-v2-backup", help="Model name"
)
@click.option("--device", "-d", default="cuda", help="Device to use for inference")
@click.option("--noop", "-n", is_flag=True, help="Dry run without copying images")
@click.option(
    "--name-sort",
    "-s",
    is_flag=True,
    help="Include HQ likeliness in the output filename",
)
def filter_images(
    input_dir,
    output_dir,
    rejected_dir,
    threshold=0.5,
    batch_size=8,
    model="NeoChen1024/aesthetic-shadow-v2-backup",
    device="cuda",
    noop=False,
    name_sort=False,
):
    pipe = pipeline(
        "image-classification",
        use_fast=True,
        model=model,
        device=device,
    )
    if not noop:
        # Create the output directories if they don't exist
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(rejected_dir, exist_ok=True)

    if not (threshold >= 0 and threshold <= 1):
        raise ValueError("Threshold must be between 0 and 1")

    # Get the list of image files in the input directory
    image_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            file_path = os.path.join(root, file)
            if filetype.is_image(file_path):
                image_files.append(file_path)

    # Process images in batches
    for i in range(0, len(image_files), batch_size):
        batch = image_files[i : i + batch_size]
        image_batch = []
        # use multiprocessing to load images in parallel
        with mp.Pool() as pool:
            image_batch = pool.map(load_image, batch)

        # Perform classification for the batch
        results = pipe(images=image_batch)

        for idx, result in enumerate(results):
            # Extract the prediction scores and labels
            predictions = result
            print(f">> {batch[idx]}")
            pp(predictions)

            # Determine the destination folder based on the prediction and threshold
            hq_score = [p for p in predictions if p["label"] == "hq"][0]["score"]
            destination_folder = output_dir if hq_score >= threshold else rejected_dir

            # Copy the image to the appropriate folder
            if not noop:
                filename = os.path.basename(batch[idx])
                if name_sort:
                    filename = f"{hq_score * 1000:4.0f}_{filename}"
                shutil.copy(
                    batch[idx],
                    os.path.join(destination_folder, filename),
                )
                print(f"Image {batch[idx]} copied to {destination_folder}")

    print("All images processed")


if __name__ == "__main__":
    filter_images()
