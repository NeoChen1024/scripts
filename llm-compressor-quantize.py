#!/usr/bin/env python
import argparse
import gc
import os
import sys

import torch
from datasets import load_dataset
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
from llmcompressor.transformers import SparseAutoModelForCausalLM, oneshot
from transformers import AutoTokenizer


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
        "--targets",
        type=str,
        default="Linear",
        help="Target layer type to quantize. (default: Linear)",
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
    return parser.parse_args()


def main():
    args = parse_args()
    model_id = args.model_id
    max_sequence_length = args.max_sequence_length
    num_calibration_samples = args.num_calibration_samples

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = SparseAutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype="auto",
    )

    # Load dataset and preprocess.
    ds = load_dataset(args.dataset_id, split=args.dataset_split)
    ds = ds.shuffle(seed=42).select(range(num_calibration_samples))

    def preprocess_fn(example):
        return {
            "text": tokenizer.apply_chat_template(example["messages"], tokenize=False)
        }

    ignore_list = args.ignore

    recipe = [
        SmoothQuantModifier(smoothing_strength=args.smoothing_strength),
        GPTQModifier(
            targets=args.targets,
            scheme=args.scheme,
            ignore=ignore_list,
            dampening_frac=args.dampening_frac,
        ),
    ]

    # Save to disk compressed.
    save_dir = model_id + '-' + args.scheme
    if args.output_dir is not None:
        save_dir = args.output_dir
    
    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=max_sequence_length,
        num_calibration_samples=num_calibration_samples,
        output_dir=save_dir,
        save_compressed=True,
    )

    # Confirm generations of the quantized model look sane.
    print("\n\n")
    print("========== SAMPLE GENERATION ==============")
    input_ids = tokenizer(args.sample_prompt, return_tensors="pt").input_ids.to("cuda")
    output = model.generate(input_ids, max_new_tokens=100)
    print(tokenizer.decode(output[0]))
    print("==========================================\n\n")

    tokenizer.save_pretrained(save_dir)

    return


if __name__ == "__main__":
    main()
