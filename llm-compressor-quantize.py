#!/usr/bin/env python
import argparse
import gc
from genericpath import exists
import os
import sys

import accelerate
import click
import torch
import xformers
from datasets import load_dataset
from llmcompressor.modifiers.quantization import GPTQModifier, QuantizationModifier
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
from llmcompressor.transformers import oneshot
from llmcompressor.transformers.compression.helpers import calculate_offload_device_map
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.functional import scaled_dot_product_attention
from transformers import AutoModelForCausalLM, AutoTokenizer

# Require latest transformers and llmcompressor
# TODO: find a better way to intergrate UMA allocator


@click.command()
@click.option("--model_id", required=True, help="Model ID to quantize.")
@click.option(
    "--dataset_id",
    default="neuralmagic/LLM_compression_calibration",
    help="Dataset ID to use for calibration.",
)
@click.option(
    "--dataset_split", default="train", help="Dataset split to use for calibration."
)
@click.option(
    "--num_calibration_samples",
    default=512,
    type=int,
    help="Number of samples to use for calibration.",
)
@click.option(
    "--max_sequence_length",
    default=2048,
    type=int,
    help="Maximum sequence length to use for calibration.",
)
@click.option(
    "--scheme",
    default="W8A8",
    type=click.Choice(["W8A8", "W4A16", "W4A8", "FP8", "FP8_DYNAMIC"]),
    help="Quantization scheme to use. (default: W8A8)",
)
@click.option(
    "--output_dir",
    help="Output directory to save quantized model. (default: <model_id>-W8A8)",
)
@click.option(
    "--ignore",
    multiple=True,
    default=["lm_head"],
    help="Tensors to ignore during quantization.",
)
@click.option(
    "--smoothing_strength",
    default=0.8,
    type=float,
    help="SmoothQuant smoothing strength.",
)
@click.option(
    "--dampening_frac",
    default=0.1,
    type=float,
    help="Dampening fraction for GPTQModifier.",
)
@click.option(
    "--sample_prompt",
    default="Hello my name is",
    help="Prompt to use for sample generation.",
)
@click.option(
    "--trust_remote_code",
    is_flag=True,
    help="Trust remote code for loading model and tokenizer.",
)
@click.option(
    "--dtype",
    default="auto",
    type=click.Choice(["auto", "half", "bfloat16"]),
    help="Data type to use for model weights. (Must use half on Turing GPUs)",
)
@click.option("--torch_compile", is_flag=True, help="Enable torch.compile.")
@click.option(
    "--attention_backend",
    default="FLASH_ATTENTION",
    type=click.Choice(
        ["EFFICIENT_ATTENTION", "FLASH_ATTENTION", "CUDNN_ATTENTION", "MATH"]
    ),
    help="Attention backend to use.",
)
@click.option("--deepspeed", is_flag=True, help="Use DeepSpeed")
@click.option(
    "--uma_allocator",
    type=click.Path(exists=True),
    help="Use custom UMA allocator .so file",
)
@click.option(
    "--cpu_offload",
    is_flag=True,
    help="Offload to CPU",
)
def main(
    model_id,
    dataset_id,
    dataset_split,
    num_calibration_samples,
    max_sequence_length,
    scheme,
    output_dir,
    ignore,
    smoothing_strength,
    dampening_frac,
    sample_prompt,
    trust_remote_code,
    dtype,
    torch_compile,
    attention_backend,
    deepspeed,
    uma_allocator,
    cpu_offload,
):
    # use xformers for efficient attention
    if attention_backend == "EFFICIENT_ATTENTION":
        xformers.config.set_memory_efficient(True)

    # Set up attention backend.
    attention = getattr(SDPBackend, attention_backend)

    if uma_allocator is not None:
        uma_alloc = torch.cuda.memory.CUDAPluggableAllocator(
            uma_allocator, "uma_malloc", "uma_free"
        )
        torch.cuda.memory.change_current_allocator(uma_alloc)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=dtype,
        trust_remote_code=trust_remote_code,
    )
    
    # Load dataset and preprocess.
    ds = load_dataset(dataset_id, split=dataset_split)
    ds = ds.shuffle(seed=42).select(range(num_calibration_samples))

    def preprocess_fn(example):
        return {
            "text": tokenizer.apply_chat_template(example["messages"], tokenize=False)
        }

    ignore_list = ignore
    recipe = []
    if scheme.startswith("FP8"):
        recipe = QuantizationModifier(
            targets="Linear", scheme=scheme, ignore=ignore_list
        )
    elif scheme.startswith("W"):
        recipe = [
            SmoothQuantModifier(smoothing_strength=smoothing_strength),
            GPTQModifier(
                targets="Linear",
                scheme=scheme,
                ignore=ignore_list,
                dampening_frac=dampening_frac,
            ),
        ]

    # Save to disk compressed.
    save_dir = model_id + "-" + scheme
    if output_dir is not None:
        save_dir = output_dir

    use_bf16 = None
    use_fp16 = None
    if dtype == "bfloat16":
        use_bf16 = True
        use_fp16 = False
    elif dtype == "half":
        use_bf16 = False
        use_fp16 = True

    use_torch_compile = True if torch_compile else None

    with sdpa_kernel(attention):
        gc.collect()
        torch.cuda.empty_cache()
        oneshot(
            deepspeed=deepspeed,
            bf16=use_bf16,
            fp16=use_fp16,
            torch_compile=use_torch_compile,
            torch_compile_backend="inductor",
            model=model,
            dataset=ds,
            recipe=recipe,
            max_seq_length=max_sequence_length,
            num_calibration_samples=num_calibration_samples,
            output_dir=save_dir,
            tokenizer=tokenizer,
            save_compressed=True,
        )

    # Confirm generations of the quantized model look sane.
    print("\n\n")
    print("========== SAMPLE GENERATION ==============")
    input_ids = tokenizer(sample_prompt, return_tensors="pt").input_ids.to("cuda")
    output = model.generate(input_ids, max_new_tokens=100)
    print(tokenizer.decode(output[0]))
    print("==========================================\n\n")

    return


if __name__ == "__main__":
    main()
