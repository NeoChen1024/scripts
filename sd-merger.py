#!/usr/bin/env python3

# Script to merge SD1.5/SDXL models into a single model
# TODO: add support for loading VAE and selectively merge/skip VAE/CLIP-{L,G} weights

import os
import sys
import json
import argparse
import numpy
import torch
import safetensors
from safetensors.torch import save_file
from rich.console import Console
from rich.table import Table
from rich.traceback import install
from rich.progress import Progress

def arg_parser():
    parser = argparse.ArgumentParser(description="Merge safetensors models into a single model")
    parser.add_argument("-i", "--input", type=str, help="Path to the safetensors file", nargs="+", action="append")
    parser.add_argument("-w", "--weight", type=float, help="Weight for each model", nargs="+", action="append")
    parser.add_argument("-o", "--output", type=str, help="Output file", required=True)
    parser.add_argument("-of", "--output-format", type=str, help="Output dtype (default: bf16)", default="bf16")
    return parser.parse_args()

dtype_map = {
    "fp32": torch.float,
    "fp16": torch.float16,
    "bf16": torch.bfloat16
}

def read_model(file, dtype) -> dict:
    tensors = {}
    with safetensors.safe_open(file, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key).to(dtype)
    return tensors

def main():
    install()
    console = Console()
    args = arg_parser()
    if not args.input:
        console.log("No input files specified")
        sys.exit(1)
    if not args.output:
        console.log("No output file specified")
        sys.exit(1)
    if not args.weight:
        console.log("No weight specified")
        sys.exit(1)
    if len(args.input) != len(args.weight):
        console.log("Number of input files and weights should be the same")
        sys.exit(1)
    if args.output_format not in dtype_map:
        console.log(f"Invalid output dtype: {args.output_format}")
        sys.exit(1)
    
    # print input and weight
    console.log(f"Input and weight: ({len(args.input)} model(s))")
    for i in range(len(args.input)):
        console.log(f"\tInput {i}: {args.input[i][0]}")
        console.log(f"\tWeight {i}: {args.weight[i][0]}")

    models = [None] * len(args.input)
    # read first model and cast to fp32 for accumulation precision
    models[0] = read_model(args.input[0][0], torch.float)
    console.log(f"Reading first model: {args.input[0][0]}")
    # scale first model by weight[0]
    console.log(f"Scaling first model by weight[0]: {args.weight[0][0]}")
    for key in models[0].keys():
        models[0][key] *= args.weight[0][0]

    console.log("Starting to merge models...")
    for i in range(1, len(args.input), 1): # skip first model
        input_file = args.input[i][0]
        weight = args.weight[i][0]
        merge_model = read_model(input_file, torch.float)
        console.log(f"Merging model {i} ({input_file}) with weight {weight} ...")
        for key in models[0].keys():
            models[0][key] += weight * merge_model[key]
        del merge_model # free memory, memory is precious

    console.log(f"Casting to {args.output_format}...")
    # cast to output dtype
    for key in models[0].keys():
        models[0][key] = models[0][key].to(dtype_map[args.output_format])

    console.log(f"Saving to {args.output}...")
    save_file(models[0], args.output)

    console.log("Done!")
    return

if __name__ == "__main__":
    install()
    main()
