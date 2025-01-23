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
from torch.nn.functional import scaled_dot_product_attention
from torch.nn.attention import SDPBackend, sdpa_kernel


# Require latest transformers and llmcompressor
# TODO: figure out how to correctly offload tensors to CPU
# TODO: make attention backend configurable


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
        default=512,
        help="Number of samples to use for calibration.",
    )
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=2048,
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
        "--trust_remote_code",
        action="store_true",
        help="Trust remote code for loading model and tokenizer.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "half", "bfloat16"],
        help="Data type to use for model weights. (Must use half on Turing GPUs)",
    )
    parser.add_argument(
        "--torch_compile",
        action="store_true",
        help="Enable torch.compile.",
    )
    parser.add_argument(
        "--attention_backend",
        type=str,
        default="FLASH_ATTENTION",
        choices=["EFFICIENT_ATTENTION", "FLASH_ATTENTION", "CUDNN_ATTENTION", "MATH"],
        help="Attention backend to use.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model_id = args.model_id
    scheme = args.scheme
    max_sequence_length = args.max_sequence_length
    num_calibration_samples = args.num_calibration_samples
    dtype = args.dtype

    # Set up attention backend.
    attention = getattr(SDPBackend, args.attention_backend)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        low_cpu_mem_usage=True,
        torch_dtype=dtype,
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
    save_dir = model_id + "-" + scheme
    if args.output_dir is not None:
        save_dir = args.output_dir

    use_bf16 = None
    use_fp16 = None
    if dtype == "bfloat16":
        use_bf16 = True
        use_fp16 = False
    elif dtype == "half":
        use_bf16 = False
        use_fp16 = True

    use_torch_compile = True if args.torch_compile else None

    with sdpa_kernel(attention):
        gc.collect()
        torch.cuda.empty_cache()
        oneshot(
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
    input_ids = tokenizer(args.sample_prompt, return_tensors="pt").input_ids.to("cuda")
    output = model.generate(input_ids, max_new_tokens=100)
    print(tokenizer.decode(output[0]))
    print("==========================================\n\n")

    return


if __name__ == "__main__":
    main()
