#!/usr/bin/env python3

# Simple script to visualize the value distribution of tensors in a safetensors file

from math import floor, ceil
from typing import Optional, Tuple

import click
import safetensors
import torch
from tqdm import tqdm
from torch import Tensor
import matplotlib

# Use a fast, non-interactive backend to speed up rendering and avoid GUI overhead
matplotlib.use("pdf")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator


def read_model(file, dtype=None) -> dict:
    tensors = {}
    with safetensors.safe_open(file, framework="pt", device="cpu") as f:
        for key in f.keys():
            if dtype:
                tensors[key] = f.get_tensor(key).to(dtype)
            else:
                tensors[key] = f.get_tensor(key)
    return tensors


def bin_count(min_val: int, max_val: int) -> int:
    bins = max_val - min_val
    if min_val == max_val:
        bins += 1
    return bins


safe_dtypes = [
    torch.float32,
    torch.float64,
    torch.float,
    torch.double,
]


def tensor_value_histogram_autorange(tensor: Tensor, bins: int = 100) -> Tuple[Tensor, Tensor]:
    return torch.histogram(tensor, bins=int(bins))


def tensor_exponent_histogram_autorange(
    tensor: Tensor,
) -> Tuple[Tensor, Tensor]:
    # Disable autograd for math ops
    with torch.inference_mode():
        _, exponent = torch.frexp(tensor)  # mantissa not used
        min_val = floor(exponent.min().item())
        max_val = ceil(exponent.max().item())
        bins = bin_count(min_val, max_val)

        # convert to float32 for histogram
        exponent = exponent.to(torch.float32)
        return torch.histogram(exponent, bins=bins, range=(min_val, max_val))


@click.command()
@click.argument(
    "input_file",
    type=click.Path(exists=True, dir_okay=False, readable=True),
    required=True,
    nargs=1,
)
@click.option(
    "-o",
    "--output",
    type=click.Path(dir_okay=False, writable=True),
    required=True,
    help="Output PDF file",
)
@click.option(
    "--bins",
    type=int,
    default=100,
    show_default=True,
    help="Number of bins for value histograms.",
)
@click.option(
    "--log-scale/--no-log-scale",
    default=True,
    show_default=True,
    help="Use log scale for y-axis in histograms.",
)
def __main__(
    input_file,
    output,
    bins,
    log_scale,
):
    if not output.endswith(".pdf"):
        output += ".pdf"

    with PdfPages(output) as pdf:
        with safetensors.safe_open(input_file, framework="pt", device="cpu") as f:
            keys = list(f.keys())
            print(f"Found {len(keys)} tensors in {input_file}")

            t = tqdm(keys, desc="Processing tensors", unit="tensor")
            for key in t:
                t.display(f"{key}", pos=1)
                # Get tensor lazily to keep memory low
                tensor = f.get_tensor(key)
                tensor = tensor.flatten()
                # set all NaN to zero to avoid issues in histogram
                tensor = torch.nan_to_num(tensor, nan=0.0)
                if tensor.numel() == 0:
                    t.write(f"Skipping empty tensor {key}")
                    continue
                if tensor.dtype not in safe_dtypes:
                    tensor = tensor.to(torch.float32)
                # Always plot both exponent and value histograms (2 subplots)
                num_subplots = 2

                plt.figure()

                with torch.inference_mode():
                    # Exponent histogram
                    plt.subplot(num_subplots, 1, 1)
                    exponent_histogram, exponent_bins = tensor_exponent_histogram_autorange(tensor)
                    plt.bar(exponent_bins[:-1], exponent_histogram, color="green")
                    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
                    plt.xlabel("Exponent")
                    plt.yscale("log" if log_scale else "linear")
                    plt.ylabel("Density")
                    plt.title(f"Tensor {key} Exponent Histogram")

                    # Value histogram
                    plt.subplot(num_subplots, 1, 2)
                    value_histogram, value_bins = tensor_value_histogram_autorange(tensor, bins=bins)
                    plt.bar(value_bins[:-1], value_histogram, color="blue")
                    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=False))
                    plt.xlabel("Value")
                    plt.yscale("log" if log_scale else "linear")
                    plt.ylabel("Density")
                    plt.title(f"Tensor {key} Value Histogram")

                plt.tight_layout()
                pdf.savefig()
                plt.close()

    print(f"Saved histogram to {output}")


if __name__ == "__main__":
    __main__()
