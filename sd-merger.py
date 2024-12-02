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
    # skip CLIP-{L,G} and VAE weights
    parser.add_argument("-sc", "--skip-clip", action="store_true", help="Skip merging CLIP-{L,G} weights")
    parser.add_argument("-sv", "--skip-vae", action="store_true", help="Skip merging VAE weights")
    # load VAE (no support for CLIP-{L,G}, because they are always present in the models)
    parser.add_argument("-lv", "--vae-input", type=str, help="Path to the VAE safetensors file", default=None)
    parser.add_argument("--vae-dtype", type=str, help="Cast VAE weights to dtype (default: None)", default=None)
    return parser.parse_args()

dtype_map = {
    # safetensors
    "F32": torch.float,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    # pytorch
    "float": torch.float,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    # sane name
    "fp32": torch.float
}

def read_model(file, dtype=None) -> dict:
    tensors = {}
    with safetensors.safe_open(file, framework="pt", device="cpu") as f:
        for key in f.keys():
            if dtype:
                tensors[key] = f.get_tensor(key).to(dtype)
            else:
                tensors[key] = f.get_tensor(key)
    return tensors

def error_exit(console, message):
    console.log(message)
    sys.exit(1)

def main():
    install()
    console = Console()
    args = arg_parser()
    if not args.input:
        error_exit(console, "No input files specified")
    if not args.output:
        error_exit(console, "No output file specified")
    if not args.weight:
        error_exit(console, "No weight specified")
    if len(args.input) != len(args.weight):
        error_exit(console, "Number of input files and weights should be the same")
        sys.exit(1)
    if args.output_format not in dtype_map:
        error_exit(console, f"Invalid output dtype: {args.output_format}")
    if args.vae_dtype not in dtype_map:
        error_exit(console, f"Invalid VAE dtype: {args.vae_dtype}")

    input_files = {}
    model_weights = {}
    for i in range(len(args.input)):
        input_files[i] = args.input[i][0]
        model_weights[input_files[i]] = args.weight[i][0]
    # check if all weights add up to 1
    total_weight = sum(model_weights.values())
    if total_weight != 1:
        console.log(f"Warning: weights do not add up to 1: {total_weight}")

    # print input and weight
    console.log(f"Input and weight: ({len(args.input)} model(s))")
    for i in range(len(args.input)):
        console.log(f"\tInput {i}: {input_files[i]}")
        console.log(f"\tWeight {i}: {model_weights[input_files[i]]}")

    # read first model and cast to fp32 for accumulation precision
    output_model = read_model(args.input[0][0], torch.float)
    console.log(f"Reading first model: {args.input[0][0]}")
    # scale first model by weight[0]
    console.log(f"Scaling first model by weight[0]: {model_weights[input_files[0]]}")
    for key in output_model.keys():
        if key.startswith("first_stage_model.") and args.skip_vae:
            continue
        if (key.startswith("cond_stage_model.") or key.startswith("conditioner.")) and args.skip_clip:
            continue
        output_model[key] *= model_weights[input_files[0]]

    console.log("Starting to merge models...")
    for i in range(1, len(input_files), 1): # skip first model
        input_file = input_files[i]
        weight = model_weights[input_file]
        merge_model = read_model(input_file) # no need for casting to float
        console.log(f"Merging model {i} ({input_file}) with weight {weight} ...")
        for key in output_model.keys():
            if key.startswith("first_stage_model.") and args.skip_vae:
                continue
            if (key.startswith("cond_stage_model.") or key.startswith("conditioner.")) and args.skip_clip:
                continue
            output_model[key] += weight * merge_model[key]
        del merge_model # free memory, memory is precious

    console.log(f"Casting to {args.output_format}...")
    # cast to output dtype
    for key in output_model.keys():
        output_model[key] = output_model[key].to(dtype_map[args.output_format])

    # override VAE weights if specified, the key name prefix is the same in SD1.5/SDXL
    if args.vae_input:
        console.log(f"Overriding VAE weights with {args.vae_input}...")
        vae_model = read_model(args.vae_input, dtype_map[args.vae_dtype])
        for key in vae_model.keys():
            output_model[f"first_stage_model.{key}"] = vae_model[key]

    console.log(f"Saving to {args.output}...")
    save_file(output_model, args.output)

    console.log("Done!")
    return

if __name__ == "__main__":
    install()
    main()
