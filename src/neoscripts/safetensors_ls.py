#!/usr/bin/env python
# view safetensors metadata without safetensors library

import argparse
import json
import math
import struct
import sys

import click
import humanize
from rich.console import Console
from rich.table import Table
from rich.traceback import install


def read_header(file):
    with open(file, "rb") as f:
        file_size = f.seek(0, 2)
        f.seek(0)
        # read first 8 bytes as uint64
        metadata_len = struct.unpack("<Q", f.read(8))[0]
        if metadata_len <= 2 or metadata_len > file_size:  # at least 2 bytes for "{}"
            print("Invalid metadata length")
            sys.exit(1)
        # read metadata
        metadata = f.read(metadata_len)
        return json.loads(metadata.decode("utf-8"))


# safetensors JSON format:
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
    if len(shape) == 0:
        return "Scalar"
    else:
        return " x ".join(map(str, shape))


def shape_to_parameter_count(shape):
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
        console.print("[r]No metadata[/r]\n")

    return


def strip_metadata(metadata):
    if "__metadata__" in metadata:
        del metadata["__metadata__"]
    return metadata


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
    if len(metadata) > 0:
        for key, value in metadata.items():
            shape = value["shape"]
            parameters = shape_to_parameter_count(shape)
            size = value["data_offsets"][1] - value["data_offsets"][0]
            dtype = value["dtype"]

            total_parameters += parameters
            total_size += size
            table.add_row(
                key,
                shape_to_str(shape),
                str(parameters),
                dtype,
                humanize.naturalsize(size, binary=True),
            )
    else:
        console.print("[yellow]No tensors...[/yellow]")

    console.print(table)

    return


def summary(metadata, console):
    total_parameters = 0
    total_size = 0
    dtype_histogram = {}
    # shape_histogram -> {(shape, dtype), count}
    shape_histogram = {}
    if len(metadata) > 0:
        for key, value in metadata.items():
            shape = value["shape"]
            parameters = shape_to_parameter_count(shape)
            size = value["data_offsets"][1] - value["data_offsets"][0]
            dtype = value["dtype"]

            total_parameters += parameters
            total_size += size
            dtype_histogram[dtype] = dtype_histogram.get(dtype, 0) + 1
            shape_histogram[(tuple(shape), dtype)] = (
                shape_histogram.get((tuple(shape), dtype), 0) + 1
            )
    # sort by name
    dtype_histogram = sorted(dtype_histogram.items(), key=lambda x: x[0])
    # natural sort by shape
    shape_histogram = sorted(shape_histogram.items(), key=lambda x: x[0])

    table = Table(safe_box=True, highlight=True)
    table.title = "Dtype histogram"
    table.add_column("Dtype")
    table.add_column("Count")
    for dtype, count in dtype_histogram:
        table.add_row(dtype, str(count))
    console.print(table)

    table = Table(safe_box=True, highlight=True)
    table.title = "Shape histogram"
    table.add_column("Shape")
    table.add_column("Dtype")
    table.add_column("Count")
    for (shape, dtype), count in shape_histogram:
        table.add_row(shape_to_str(shape), dtype, str(count))
    console.print(table)

    console.print(
        f"[b]Total parameters:[/b] [bright_cyan]{humanize.intword(total_parameters)}[/bright_cyan] ({total_parameters})"
    )
    console.print(
        f"[b]Total size:[/b] [bright_cyan]{humanize.naturalsize(total_size, binary=True)}[/bright_cyan]"
    )
    return


@click.command()
@click.argument("file", nargs=-1, required=True)
@click.option(
    "-j", "--json", "dump_json", is_flag=True, help="Dump all metadata in JSON format"
)
@click.option("-l", "--list", "list_tensors", is_flag=True, help="List all tensors")
def __main__(file, dump_json, list_tensors):
    install(show_locals=True)

    console = Console()

    for f in file:
        console.print(f"Reading [bright_magenta]{f}[/bright_magenta]")
        metadata = read_header(f)
        if dump_json:
            console.print(json.dumps(metadata, indent=4, sort_keys=False))
        else:
            print_metadata(metadata, console)
            metadata = strip_metadata(metadata)
            if list_tensors:
                print_tensors(metadata, console)
            summary(metadata, console)
    sys.exit(0)


if __name__ == "__main__":
    __main__()
