#!/usr/bin/env python3

from math import ceil, floor
from typing import Tuple

import click
import matplotlib
import safetensors
import torch
from torch import Tensor
from tqdm import tqdm

matplotlib.use("pdf")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator

_SAMPLE_SIZE = 1_000_000

SAFE_DTYPES = {
    torch.float32,
    torch.float64,
    torch.float,
    torch.double,
}


def _sample_tensor(tensor: Tensor, max_samples: int | None) -> Tensor:
    if max_samples is None:
        return tensor
    n = tensor.numel()
    if n <= max_samples:
        return tensor
    idx = torch.randint(0, n, (max_samples,))
    return tensor[idx]


def _bin_count(min_val: int, max_val: int) -> int:
    span = max_val - min_val
    if min_val == max_val:
        span += 1
    return span


def tensor_value_histogram_autorange(
    tensor: Tensor, bins: int = 100, max_samples: int | None = _SAMPLE_SIZE
) -> Tuple[Tensor, Tensor]:
    with torch.inference_mode():
        sampled = _sample_tensor(tensor, max_samples)
        return torch.histogram(sampled, bins=int(bins))


def tensor_exponent_histogram_autorange(tensor: Tensor, max_samples: int | None = _SAMPLE_SIZE) -> Tuple[Tensor, Tensor]:
    with torch.inference_mode():
        sampled = _sample_tensor(tensor, max_samples)
        _, exponent = torch.frexp(sampled)
        min_val = floor(exponent.min().item())
        max_val = ceil(exponent.max().item())
        exp_bins = _bin_count(min_val, max_val)
        exponent = exponent.to(torch.float32)
        return torch.histogram(exponent, bins=exp_bins, range=(min_val, max_val))


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
@click.option(
    "--sample/--no-sample",
    default=True,
    show_default=True,
    help="Use random sampling for large tensors to speed up histogram computation.",
)
def __main__(
    input_file,
    output,
    bins,
    log_scale,
    sample,
):
    if not output.endswith(".pdf"):
        output += ".pdf"

    max_samples = _SAMPLE_SIZE if sample else None

    with PdfPages(output) as pdf:
        with safetensors.safe_open(input_file, framework="pt", device="cpu") as f:
            keys = list(f.keys())
            print(f"Found {len(keys)} tensors in {input_file}")

            t = tqdm(keys, desc="Processing tensors", unit="tensor")
            for key in t:
                t.display(f"{key}", pos=1)
                tensor = f.get_tensor(key)
                tensor = tensor.flatten()
                if tensor.numel() == 0:
                    t.write(f"Skipping empty tensor {key}")
                    continue
                if tensor.dtype not in SAFE_DTYPES:
                    tensor = tensor.to(torch.float32)
                tensor = torch.nan_to_num(tensor, nan=0.0)

                plt.figure()

                with torch.inference_mode():
                    exponent_histogram, exponent_bins = tensor_exponent_histogram_autorange(tensor, max_samples=max_samples)
                    plt.subplot(2, 1, 1)
                    plt.bar(exponent_bins[:-1], exponent_histogram, color="green")
                    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
                    plt.xlabel("Exponent")
                    plt.yscale("log" if log_scale else "linear")
                    plt.ylabel("Density")
                    plt.title(f"Tensor {key} Exponent Histogram")

                    value_histogram, value_bins = tensor_value_histogram_autorange(tensor, bins=bins, max_samples=max_samples)
                    plt.subplot(2, 1, 2)
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
