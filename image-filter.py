#!/usr/bin/env python
import multiprocessing as mp
from multiprocessing import JoinableQueue, Queue, Process
import os
from pprint import pprint as pp
from shutil import copy2, move

import click
import filetype
from PIL import Image
from reflink import reflink
from transformers import pipeline

modes = {
    "copy": copy2,
    "move": move,
    "symlink": os.symlink,
    "hardlink": os.link,
    "reflink": reflink,
}


# Process Image
def load_image(
    file_path_queue: JoinableQueue,
    image_return_queue: Queue,
):
    print(f"Image Loader {os.getpid()} started")
    while True:
        file_path = file_path_queue.get()
        if file_path is None:  # exit signal
            break
        img = Image.open(file_path)
        img.convert("RGB")  # force it to load into memory
        # resize to fit into 1024x1024, greatly speeds up processing
        if img.width > 1024 or img.height > 1024:
            img.thumbnail((1024, 1024))
        # will block if the return queue is full
        image_return_queue.put((file_path, img))
        # TODO: figure out whether this will cause deadlock or not
        file_path_queue.task_done()
    print(f"Image Loader {os.getpid()} exiting")


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
@click.option(
    "--batch-size", "-b", default=16, type=int, help="Batch size for filtering images"
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
@click.option(
    "-m",
    "--mode",
    type=click.Choice(list(modes.keys())),
    default="copy",
    help="Choose between modes. (default: copy)",
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
    mode="copy",
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

    file_func = modes[mode]

    file_path_queue = JoinableQueue()
    image_return_queue = Queue(maxsize=batch_size * 2)
    # Launch the image loading processes
    num_processes = min(mp.cpu_count(), batch_size)
    processes = []
    for i in range(num_processes):
        p = Process(
            name=f"image_loader_{i}",
            daemon=True,
            target=load_image,
            args=(file_path_queue, image_return_queue),
        )
        p.start()
        processes.append(p)

    # Load all image file paths
    for root, _, files in os.walk(input_dir):
        for file in files:
            file_path = os.path.join(root, file)
            if filetype.is_image(file_path):
                file_path_queue.put(file_path)

    # Process images in batches
    while not image_return_queue.empty() or not file_path_queue.empty():
        batch = []
        return_image_file_paths = []
        for _ in range(batch_size):
            if not image_return_queue.empty():
                (file_path, img) = image_return_queue.get()
                batch.append(img)
                return_image_file_paths.append(file_path)
            elif not file_path_queue.empty():
                continue
            else:
                break

        # Perform classification for the batch
        results = pipe(images=batch)

        for idx, result in enumerate(results):
            # Extract the prediction scores and labels
            predictions = result
            print(f">> {return_image_file_paths[idx]}")
            pp(predictions)

            # Determine the destination folder based on the prediction and threshold
            hq_score = [p for p in predictions if p["label"] == "hq"][0]["score"]
            destination_folder = output_dir if hq_score >= threshold else rejected_dir

            # Copy the image to the appropriate folder
            if not noop:
                filename = os.path.basename(return_image_file_paths[idx])
                if name_sort:
                    filename = f"{hq_score * 1000:4.0f}_{filename}"
                file_func(
                    batch[idx],
                    os.path.join(destination_folder, filename),
                )
                print(f"{mode} {batch[idx]} to {destination_folder}")

    # Send exit signal to the image loading processes
    for _ in range(num_processes * 2):
        file_path_queue.put(None)
    for p in processes:
        p.join()
    print("All images processed")


if __name__ == "__main__":
    filter_images()
