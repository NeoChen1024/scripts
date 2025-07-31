#!/usr/bin/env python
# Simple script to scale LoRa alpha values

import click
from click import echo
import safetensors
from safetensors.torch import save_file


@click.command()
@click.option("--scale-by", "-s", type=float)
@click.option("--set-alpha", "-a", type=float)
@click.option("--list-alpha-values", "-l", is_flag=True)
@click.argument("input-file", type=click.Path(exists=True, dir_okay=False))
@click.argument("output-file", type=click.Path(dir_okay=False, writable=True), required=False)
def scale_lora(input_file, output_file, scale_by, set_alpha, list_alpha_values):
    lora_tensors = {}
    metadata = {}
    with safetensors.safe_open(input_file, framework="pt", device="cpu") as t:
        for key in t.keys():
            lora_tensors[key] = t.get_tensor(key).clone().to("cpu")
        metadata = t.metadata()

    for key in lora_tensors.keys():
        if not key.startswith("lora"):
            raise ValueError(f"Unexpected key: {key}")

    if list_alpha_values:
        for key in lora_tensors.keys():
            if key.endswith(".alpha"):
                echo(f"{key}: {lora_tensors[key].item()}")

    if scale_by:
        for key in lora_tensors.keys():
            if key.endswith(".alpha"):
                lora_tensors[key] *= scale_by
    if set_alpha:
        for key in lora_tensors.keys():
            if key.endswith(".alpha"):
                lora_tensors[key] = set_alpha

    if output_file:
        save_file(lora_tensors, output_file, metadata=metadata)
        echo(f"Saved to {output_file}")


if __name__ == "__main__":
    scale_lora()
