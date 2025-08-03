#!/usr/bin/env python
import multiprocessing as mp
import os
from shutil import copy2, move
from typing import List

import click
import filetype
import torch
from accelerate import Accelerator
from PIL import Image, ImageOps
from reflink import reflink
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from transformers import AutoImageProcessor, ViTForImageClassification, pipeline

modes = {
    "copy": copy2,
    "move": move,
    "symlink": os.symlink,
    "hardlink": os.link,
    "reflink": reflink,
}


class ImageDatasetLoader(Dataset):
    image_paths: List[str]

    def __init__(self, image_paths: List[str]):
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image_path = self.image_paths[idx]
        try:
            img = Image.open(image_path)
            corr = ImageOps.exif_transpose(img)
            if corr is not None:
                img = corr
            img.convert("RGB")
            img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
            # Pad to ensure the image is 1024x1024
            img = ImageOps.pad(img, (1024, 1024), color="black")
            return (img, image_path)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None


def collate_fn(batch):
    batch = [x for x in batch if x is not None]
    return batch


# Process Image
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
    default=0.75,
    type=click.FloatRange(0, 1, clamp=True),
    help="Threshold for filtering images (0-1)",
)
@click.option("--batch-size", "-b", default=16, type=int, help="Batch size for filtering images")
@click.option(
    "--model-name",
    "-m",
    default="NeoChen1024/aesthetic-shadow-v2-backup",
    help="Model name",
)
@click.option("--device", "-d", help="Device to use for inference")
@click.option("--noop", "-n", is_flag=True, help="Dry run without copying images")
@click.option(
    "--name-sort",
    "-s",
    is_flag=True,
    help="Include HQ likeliness in the output filename",
)
@click.option(
    "-m",
    "--mode",
    type=click.Choice(list(modes.keys())),
    default="copy",
    help="Choose between modes. (default: copy)",
)
@click.option(
    "--buffer-multiplier",
    default=2,
    type=int,
    help="Multiplier for the image buffer size",
)
def __main__(
    input_dir,
    output_dir,
    rejected_dir,
    threshold=0.75,
    batch_size=16,
    model_name="NeoChen1024/aesthetic-shadow-v2-backup",
    device=None,
    noop=False,
    name_sort=False,
    mode="copy",
    buffer_multiplier=2,
):
    try:
        accelerator = Accelerator()
        actual_device = accelerator.device
        if device is not None:
            actual_device = device
        model = ViTForImageClassification.from_pretrained(
            model_name,
            attn_implementation="sdpa",
        )
        processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
        pipe = pipeline(
            "image-classification",
            use_fast=True,
            model=model,
            image_processor=processor,
            device=actual_device,
            batch_size=batch_size,
        )
        if not noop:
            # Create the output directories if they don't exist
            os.makedirs(output_dir, exist_ok=True)
            os.makedirs(rejected_dir, exist_ok=True)

        if not (threshold >= 0 and threshold <= 1):
            raise ValueError("Threshold must be between 0 and 1")

        file_func = modes[mode]

        num_processes = min(mp.cpu_count(), batch_size)

        image_file_paths = []

        # Load all image file paths
        for root, _, files in os.walk(input_dir):
            for file in files:
                file_path = os.path.join(root, file)
                if filetype.is_image(file_path):
                    image_file_paths.append(file_path)

        data = torch.utils.data.DataLoader(
            ImageDatasetLoader(image_file_paths),
            batch_size=batch_size,
            collate_fn=collate_fn,
            num_workers=num_processes,
            prefetch_factor=buffer_multiplier,
        )

        # Process images in batches
        t = tqdm(data, total=len(data), smoothing=0)
        for batch in t:
            batch_images, original_image_path = [], []
            for imgs, paths in batch:
                batch_images.append(imgs)
                original_image_path.append(paths)
            # TODO: find a way to silence this warning by pipeline:
            # "You seem to be using the pipelines sequentially on GPU. In order to maximize efficiency please use a dataset"
            # NOTE: It's still using DataLoader internally, there's no difference.
            # Then there's no way to pass a DataLoader to the pipeline, nor a collate_fn.
            # Letting it load data by itself is a bad idea too, as there can be errors in loading images, and
            # we need a reliable way to preserve the input image paths.
            result = pipe(batch_images)
            if result is None:
                t.write("No result from pipeline")
                continue
            for result, image_path in zip(result, original_image_path):
                # Get the path of image relative to the input directory
                relative_filepath = os.path.relpath(image_path, input_dir)
                # separate the path and the filename
                relative_dirpath = os.path.join(*relative_filepath.split(os.path.sep)[:-1])
                bn = os.path.basename(image_path)
                # Extract the prediction scores and labels
                predictions = result

                # Determine the destination dir based on the prediction and threshold
                hq_score = [p for p in predictions if p["label"] == "hq"][0]["score"]
                destination_dir = output_dir if hq_score >= threshold else rejected_dir
                destination_dir = os.path.join(destination_dir, relative_dirpath)

                t.write(f"[{hq_score:0.3f}] {bn}")
                if not noop:
                    if name_sort:
                        bn = f"{hq_score * 1000:04.0f}_{bn}"
                    os.makedirs(destination_dir, exist_ok=True)
                    file_func(
                        image_path,
                        os.path.join(destination_dir, bn),
                    )
                    t.write(f"{mode} {image_path} -> {destination_dir}")
        return
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    __main__()
