#!/usr/bin/env python3

# Simple script to visualize the value distribution of tensors in a safetensors file

import sys
from math import ceil, floor
from typing import Optional, Tuple

import click
import safetensors
import torch
from tqdm import tqdm
from rich.console import Console
from rich.traceback import install
from torch import Tensor
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

console = Console()
install(console=console)


def read_model(file, dtype=None) -> dict:
    tensors = {}
    with safetensors.safe_open(file, framework="pt", device="cpu") as f:
        for key in f.keys():
            if dtype:
                tensors[key] = f.get_tensor(key).to(dtype)
            else:
                tensors[key] = f.get_tensor(key)
    return tensors


def error_exit(message):
    console.log(message)
    sys.exit(1)


def bin_count(min_val: int, max_val: int) -> int:
    bins = max_val - min_val
    if min_val == max_val:
        bins += 1
    return bins


safe_dtypes = [
    torch.float32,
    torch.float16,
    torch.bfloat16,
    torch.float64,
    torch.float,
    torch.half,
    torch.double,
]


def tensor_value_histogram_autorange(tensor: Tensor, bins: Optional[int] = 100) -> Tuple[Tensor, Tensor]:
    if tensor.dtype not in safe_dtypes:
        tensor = tensor.to(torch.float32)
    return torch.histogram(tensor, bins=bins, density=True)


def tensor_exponent_histogram_autorange(tensor: Tensor) -> Tuple[Tensor, Tensor]:
    tensor = tensor.flatten()
    if tensor.dtype not in safe_dtypes:
        tensor = tensor.to(torch.float32)
    mantissa, exponent = torch.frexp(tensor)  # mantissa not used
    min_val = floor(exponent.min().item())
    max_val = ceil(exponent.max().item())
    bins = bin_count(min_val, max_val)

    exponent = exponent.to(torch.float32)
    return torch.histogram(exponent, bins=bins, range=(min_val, max_val), density=True)


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
def __main__(
    input_file,
    output,
):
    if not output.endswith(".pdf"):
        output += ".pdf"

    tensors = read_model(input_file)
    console.log(f"Loaded {len(tensors)} tensors from {input_file}")
    
    with PdfPages(output) as pdf: 
        t = tqdm(tensors.items(), desc="Processing tensors", unit="tensor")
        for key, tensor in t:
            histogram, bins = tensor_exponent_histogram_autorange(tensor)
            t.write(f"\t{key}")
            
            bins = bins[:-1]
            plt.figure()
            plt.bar(bins, histogram, align='center', color=['forestgreen']) 
            plt.xlabel('Bins') 
            plt.ylabel('Frequency') 
            plt.title(f'Tensor {key} Histogram') 
            pdf.savefig() 
            plt.close()

    console.log(f"Saved histogram to {output}")


if __name__ == "__main__":
    __main__()
