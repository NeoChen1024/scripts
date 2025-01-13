#!/usr/bin/env python
import argparse
import gc
import os
import sys

import torch
from datasets import load_dataset
from llmcompressor.modifiers.quantization import GPTQModifier, QuantizationModifier
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
from llmcompressor.transformers import oneshot
from llmcompressor.transformers.compression.helpers import calculate_offload_device_map
from transformers import AutoTokenizer, AutoModelForCausalLM

# Require latest transformers and llmcompressor
# TODO: figure out how to correctly offload tensors to CPU

# Parse arguments.
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id", type=str, required=True, help="Model ID to quantize."
    )
    parser.add_argument(
        "--dataset_id",
        type=str,
        default="neuralmagic/LLM_compression_calibration",
        help="Dataset ID to use for calibration.",
    )
    parser.add_argument(
        "--dataset_split",
        type=str,
        default="train",
        help="Dataset split to use for calibration.",
    )
    parser.add_argument(
        "--num_calibration_samples",
        type=int,
        default=1024,
        help="Number of samples to use for calibration.",
    )
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=4096,
        help="Maximum sequence length to use for calibration.",
    )
    parser.add_argument(
        "--scheme",
        type=str,
        default="W8A8",
        choices=["W8A8", "W4A16", "W4A8", "FP8", "FP8_DYNAMIC"],
        help="Quantization scheme to use. (default: W8A8)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Output directory to save quantized model. (default: <model_id>-W8A8)",
    )
    parser.add_argument(
        "--ignore",
        type=str,
        nargs="*",
        default=["lm_head"],
        help="Tensors to ignore during quantization.",
    )
    parser.add_argument(
        "--smoothing_strength",
        type=float,
        default=0.8,
        help="SmoothQuant smoothing strength.",
    )
    parser.add_argument(
        "--dampening_frac",
        type=float,
        default=0.1,
        help="Dampening fraction for GPTQModifier.",
    )
    parser.add_argument(
        "--sample_prompt",
        type=str,
        default="Hello my name is",
        help="Prompt to use for sample generation.",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Number of GPUs to use for quantization",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Trust remote code for loading model and tokenizer.",
    )
    parser.add_argument(
        "--device_map",
        type=str,
        default="auto",
        help="Device map for offloading tensors.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model_id = args.model_id
    scheme = args.scheme
    max_sequence_length = args.max_sequence_length
    num_calibration_samples = args.num_calibration_samples

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=args.device_map,
        low_cpu_mem_usage=True,
        torch_dtype="auto",
        trust_remote_code=args.trust_remote_code,
    )

    # Load dataset and preprocess.
    ds = load_dataset(args.dataset_id, split=args.dataset_split)
    ds = ds.shuffle(seed=42).select(range(num_calibration_samples))

    def preprocess_fn(example):
        return {
            "text": tokenizer.apply_chat_template(example["messages"], tokenize=False)
        }

    ignore_list = args.ignore
    recipe = []
    if scheme.startswith("FP8"):
        recipe = QuantizationModifier(
            targets="Linear", scheme=scheme, ignore=ignore_list
        )
    elif scheme.startswith("W"):
        recipe = [
            SmoothQuantModifier(smoothing_strength=args.smoothing_strength),
            GPTQModifier(
                targets="Linear",
                scheme=scheme,
                ignore=ignore_list,
                dampening_frac=args.dampening_frac,
            ),
        ]

    # Save to disk compressed.
    save_dir = model_id + '-' + scheme
    if args.output_dir is not None:
        save_dir = args.output_dir
    
    oneshot(
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
    input_ids = tokenizer(args.sample_prompt, return_tensors="pt").input_ids.to("cuda")
    output = model.generate(input_ids, max_new_tokens=100)
    print(tokenizer.decode(output[0]))
    print("==========================================\n\n")

    return


if __name__ == "__main__":
    main()
