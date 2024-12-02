#!/usr/bin/env python
# view safetensors metadata without safetensors library

import sys
import struct
import json
import argparse
import math
from rich.console import Console
from rich.table import Table
from rich.traceback import install
import humanize

def arg_parser():
    parser = argparse.ArgumentParser(description="View safetensors metadata without safetensors library")
    parser.add_argument("file", type=str, help="Path to the safetensors file", nargs="+")
    parser.add_argument("--json", action="store_true", help="Output metadata in JSON format")
    return parser.parse_args()

def read_metadata(file):
    with open(file, "rb") as f:
        # read first 8 bytes as uint64
        metadata_len = struct.unpack("<Q", f.read(8))[0]
        # read metadata
        metadata = f.read(metadata_len)
        return json.loads(metadata.decode("utf-8"))

# print metadata in table format
# reference format:
# {
#    "__metadata__": {}, // optional
#    "conditioner.embedders.0.transformer.text_model.embeddings.position_embedding.weight": {
#        "data_offsets": [
#            0,
#            118272
#        ],
#        "dtype": "F16",
#        "shape": [
#            77,
#            768
#        ]
#    },
#    "conditioner.embedders.0.transformer.text_model.embeddings.token_embedding.weight": {
#        "data_offsets": [
#            118272,
#            76008960
#        ],
#        "dtype": "F16",
#        "shape": [
#            49408,
#            768
#        ]
#    },
#    "conditioner.embedders.0.transformer.text_model.encoder.layers.0.layer_norm1.bias": {
#        "data_offsets": [
#            76008960,
#            76010496
#        ],
#        "dtype": "F16",
#        "shape": [
#            768
#        ]
#    },
# .....

def shape_to_str(shape):
    return " x ".join(map(str, shape))

def parameter_count(shape):
    return math.prod(shape)

def print_metadata(metadata, console):
    table = Table(safe_box=True, highlight=True)
    table.title = "Metadata"
    table.add_column("Key")
    table.add_column("Value")

    if "__metadata__" in metadata:
        for key, value in metadata["__metadata__"].items():
            table.add_row(key, value)
        console.print(table)
    else:
        console.print("[yellow]No metadata...[/yellow]")

    return

def print_tensors(metadata, console):
    total_parameters = 0
    total_size = 0

    table = Table(safe_box=True, highlight=True)
    table.title = "Tensors"
    table.add_column("Name")
    table.add_column("Shape")
    table.add_column("Parameters")
    table.add_column("Dtype")
    table.add_column("Size")
    if len(metadata) > 1:
        for key, value in metadata.items():
            if key == "__metadata__":
                continue
            total_parameters += parameter_count(value["shape"])
            total_size += value["data_offsets"][1] - value["data_offsets"][0]
            table.add_row(key,
                shape_to_str(value["shape"]),
                str(parameter_count(value["shape"])),
                value["dtype"],
                humanize.naturalsize(value["data_offsets"][1] - value["data_offsets"][0]))
    else:
        console.print("[yellow]No tensors...[/yellow]")

    console.print(table)

    console.print(f"[b]Total parameters:[/b] [bright_cyan]{total_parameters}[/bright_cyan]")
    console.print(f"[b]Total size:[/b] [bright_cyan]{humanize.naturalsize(total_size)}[/bright_cyan]")

    return

def main():
    install(show_locals=True)

    console = Console()

    args = arg_parser()
    for file in args.file:
        metadata = read_metadata(file)
        if args.json:
            console.print(json.dumps(metadata, indent=4, sort_keys=True))
        else:
            print_metadata(metadata, console)
            print_tensors(metadata, console)
    sys.exit(0)

if __name__ == "__main__":
    main()
