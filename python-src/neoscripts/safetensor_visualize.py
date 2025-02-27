#!/usr/bin/env python3

# Simple script to visualize the value distribution of tensors in a safetensors file

import sys
from math import ceil, floor
from typing import Optional, Tuple

import click
import safetensors
import torch
from tqdm import tqdm
from torch import Tensor
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
    torch.float16,
#    torch.bfloat16, # not supported by torch.histogram
    torch.float64,
    torch.float,
    torch.half,
    torch.double,
]


def tensor_value_histogram_autorange(tensor: Tensor, bins: Optional[int] = 100) -> Tuple[Tensor, Tensor]:
    if tensor.dtype not in safe_dtypes:
        tensor = tensor.to(torch.float32)
    return torch.histogram(tensor, bins=bins)


def tensor_exponent_histogram_autorange(tensor: Tensor) -> Tuple[Tensor, Tensor]:
    tensor = tensor.flatten()
    if tensor.dtype not in safe_dtypes:
        tensor = tensor.to(torch.float32)
    mantissa, exponent = torch.frexp(tensor)  # mantissa not used
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
def __main__(
    input_file,
    output,
):
    if not output.endswith(".pdf"):
        output += ".pdf"

    tensors = read_model(input_file)
    print(f"Loaded {len(tensors)} tensors from {input_file}")
    
    with PdfPages(output) as pdf: 
        t = tqdm(tensors.items(), desc="Processing tensors", unit="tensor")
        for key, tensor in t:
            t.write(f"\t{key}")
            # subplot with 2 rows and 1 column
            plt.figure()
            
            plt.subplot(2, 1, 1)
            exponent_histogram, exponent_bins = tensor_exponent_histogram_autorange(tensor)
            plt.bar(exponent_bins[:-1], exponent_histogram, color='green') 
            # integer bins
            plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.xlabel('Exponent') 
            plt.yscale('log')
            plt.ylabel('Density') 
            plt.title(f'Tensor {key} Exponent Histogram') 
            
            plt.subplot(2, 1, 2)
            value_histogram, value_bins = tensor_value_histogram_autorange(tensor)
            plt.bar(value_bins[:-1], value_histogram, color='blue')
            plt.gca().xaxis.set_major_locator(MaxNLocator(integer=False))
            plt.xlabel('Value') 
            plt.yscale('log')
            plt.ylabel('Density') 
            plt.title(f'Tensor {key} Value Histogram') 

            plt.tight_layout()
            
            pdf.savefig() 
            plt.close()

    print(f"Saved histogram to {output}")


if __name__ == "__main__":
    __main__()
