#!/usr/bin/env python3

# Simple script to merge multiple SD1.5/SDXL models into a single model

import os
import sys
import gc
from datetime import datetime

import click
import torch
import safetensors
from safetensors.torch import save_file
from rich.console import Console
from rich.traceback import install

console = Console()
install(console=console)

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
    "fp32": torch.float,
}

dtype_name = {torch.float: "FP32", torch.float16: "FP16", torch.bfloat16: "BF16"}


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


@torch.no_grad()
def check_tensor_rounding_to_zero(tensor, output_dtype) -> bool:
    tensor_test = tensor.detach().clone()
    rtz_limit = torch.finfo(output_dtype).tiny
    tensor_test[tensor_test == 0] = 1  # avoid zero values
    if tensor_test.abs().min() < rtz_limit:
        return True
    return False


@torch.no_grad()
def all_checking(model, dtype):
    console.log("Checking for overflow / rounding to zero / NaN / Inf ...")
    count_overflow = 0
    count_rtz = 0
    count_nan = 0
    count_inf = 0
    max_val = torch.finfo(dtype).max
    for key in model.keys():
        if model[key].abs().max() > max_val:
            count_overflow += 1
        if check_tensor_rounding_to_zero(model[key], dtype):
            count_rtz += 1
        if torch.isnan(model[key]).any():
            count_nan += 1
        if torch.isinf(model[key]).any():
            count_inf += 1
    if count_overflow:
        console.log(
            f"[bright_red]Warning:[/bright_red] {count_overflow} tensor(s) will have overflow in {dtype_name[dtype]}"
        )
    if count_rtz:
        console.log(
            f"[bright_red]Warning:[/bright_red] {count_rtz} tensor(s) will have rounding to zero in {dtype_name[dtype]}"
        )
    if count_nan:
        console.log(
            f"[bright_red]Warning:[/bright_red] {count_nan} tensor(s) contain NaN"
        )
    if count_inf:
        console.log(
            f"[bright_red]Warning:[/bright_red] {count_inf} tensor(s) contain Inf"
        )


@click.command()
@click.option(
    "-i",
    "--input",
    "input_files",
    type=click.Path(exists=True, dir_okay=False, readable=True),
    required=True,
    multiple=True,
    help="Path to the safetensors file. Can be used multiple times.",
)
@click.option(
    "-w",
    "--weight",
    "weights",
    type=click.FloatRange(0, 1, clamp=True),
    required=True,
    multiple=True,
    help="Weight for each model. Must be used as many times as --input.",
)
@click.option(
    "-af",
    "--accumulate-format",
    default="fp32",
    type=click.Choice(list(dtype_map.keys())),
    help="Accumulate dtype (default: fp32)",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(dir_okay=False, writable=True),
    required=True,
    help="Output file",
)
@click.option(
    "-of",
    "--output-format",
    default="bf16",
    type=click.Choice(list(dtype_map.keys())),
    help="Output dtype (default: bf16)",
)
@click.option(
    "-sc",
    "--skip-clip",
    is_flag=True,
    help="Skip merging CLIP-{L,G} weights, leave them as is from the first model",
)
@click.option(
    "-sv",
    "--skip-vae",
    is_flag=True,
    help="Skip merging VAE weights, leave them as is from the first model",
)
@click.option(
    "-lv",
    "--vae-input",
    type=click.Path(dir_okay=False, readable=True, exists=True),
    default=None,
    help="Path to the VAE safetensors file",
)
@click.option(
    "--vae-dtype",
    type=click.Choice(list(dtype_map.keys())),
    default=None,
    help="Cast VAE weights to dtype (default: None)",
)
@click.option("--title", default=None, help="Title of the merged model")
@click.option("--description", default=None, help="Description of the merged model")
@click.option("--author", default=None, help="Author of the merged model")
@torch.no_grad()
def __main__(
    input_files,
    weights,
    accumulate_format,
    output,
    output_format,
    skip_clip,
    skip_vae,
    vae_input,
    vae_dtype,
    title,
    description,
    author,
):
    if len(input_files) != len(weights):
        error_exit("Number of input files and weights should be the same")

    accumulate_dtype = dtype_map[accumulate_format]
    output_dtype = dtype_map[output_format]

    # Informative console logging
    console.log(f"Input and weight: {len(input_files)} model(s)")
    for i, file in enumerate(input_files):
        console.log(f"\tInput {i}: {file} with weight {weights[i]}")

    total_weight = sum(weights)
    if total_weight != 1:
        console.log(
            f"[bright_red]Warning:[/bright_red] weights do not add up to 1: {total_weight}"
        )

    # Load the first model and apply scaling
    output_model = read_model(input_files[0], accumulate_dtype)
    console.log(f"Reading first model: {input_files[0]}")
    console.log(f"Scaling first model by weight: {weights[0]}")
    for key in output_model.keys():
        if key.startswith("first_stage_model.") and skip_vae:
            continue
        if (
            key.startswith("cond_stage_model.") or key.startswith("conditioner.")
        ) and skip_clip:
            continue
        output_model[key] *= weights[0]

    console.log("Starting to merge models...")
    for i in range(1, len(input_files)):
        input_file = input_files[i]
        weight = weights[i]
        merge_model = read_model(input_file)
        console.log(f"Merging model {i} ({input_file}) with weight {weight} ...")
        for key in output_model.keys():
            if key.startswith("first_stage_model.") and skip_vae:
                continue
            if (
                key.startswith("cond_stage_model.") or key.startswith("conditioner.")
            ) and skip_clip:
                continue
            output_model[key] += weight * merge_model[key].to(accumulate_dtype)
        del merge_model
        gc.collect()

    all_checking(output_model, output_dtype)

    console.log(f"Casting to {dtype_name[output_dtype]}...")
    for key in output_model.keys():
        output_model[key] = output_model[key].to(output_dtype)

    if vae_input:
        console.log(f"Overriding VAE weights with {vae_input}...")
        vae_model = read_model(vae_input, dtype_map[vae_dtype])
        for key in vae_model.keys():
            output_model[f"first_stage_model.{key}"] = vae_model[key]
        del vae_model

    formula = " + ".join(
        [
            f"{os.path.basename(input_files[i])} * {weights[i]}"
            for i in range(len(input_files))
        ]
    )
    model_name = title if title else os.path.basename(output)
    description = description if description else formula
    author = author if author else "SD-Merger"
    console.log(f"Saving to {output}...")
    metadata = {
        "modelspec.title": model_name,
        "modelspec.date": datetime.now().strftime("%Y-%m-%d"),
        "modelspec.description": description,
        "modelspec.author": author,
    }
    console.log(f"Metadata: {metadata}")
    save_file(output_model, output, metadata=metadata)

    console.log("Done!")


if __name__ == "__main__":
    __main__()
