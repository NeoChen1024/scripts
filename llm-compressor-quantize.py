#!/usr/bin/env python
import gc

import click
import torch
from datasets import load_dataset
from llmcompressor.modifiers.quantization import GPTQModifier, QuantizationModifier
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
from llmcompressor.transformers import oneshot
from llmcompressor.transformers.compression.helpers import calculate_offload_device_map
from compressed_tensors.quantization.quant_scheme import PRESET_SCHEMES
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Require latest transformers and llmcompressor


@click.command()
@click.option("--model-id", required=True, help="Model ID to quantize.")
@click.option(
    "--dataset-id",
    default="neuralmagic/LLM_compression_calibration",
    show_default=True,
    help="Dataset ID to use for calibration.",
)
@click.option(
    "--dataset-split", default="train", show_default=True, help="Dataset split to use for calibration."
)
@click.option(
    "--num-calibration-samples",
    default=512,
    type=int,
    show_default=True,
    help="Number of samples to use for calibration.",
)
@click.option(
    "--max-sequence-length",
    default=2048,
    type=int,
    show_default=True,
    help="Maximum sequence length to use for calibration.",
)
@click.option(
    "--apply-chat-template",
    type=bool,
    default=True,
    show_default=True,
    help="Apply chat template to dataset messages.",
)
@click.option(
    "--scheme",
    default="W8A8",
    type=click.Choice(list(PRESET_SCHEMES.keys())),
    show_default=True,
    help="Quantization scheme to use. (default: W8A8)",
)
@click.option(
    "--output-dir",
    help="Output directory to save quantized model. (default: <model_id>-W8A8)",
)
@click.option(
    "--ignore",
    multiple=True,
    default=["lm_head"],
    show_default=True,
    help="Tensors to ignore during quantization.",
)
@click.option(
    "--smoothing-strength",
    default=0.8,
    type=float,
    show_default=True,
    help="SmoothQuant smoothing strength.",
)
@click.option(
    "--dampening-frac",
    default=0.1,
    type=float,
    show_default=True,
    help="Dampening fraction for GPTQModifier.",
)
@click.option(
    "--sample-prompt",
    default="Hello my name is",
    show_default=True,
    help="Prompt to use for sample generation.",
)
@click.option(
    "--trust-remote-code",
    is_flag=True,
    help="Trust remote code for loading model and tokenizer.",
)
@click.option(
    "--dtype",
    default="auto",
    type=click.Choice(["auto", "half", "bfloat16"]),
    show_default=True,
    help="Data type to use for model weights. (Must use half on Turing GPUs)",
)
@click.option("--torch-compile", is_flag=True, help="Enable torch.compile.")
@click.option(
    "--attention-backend",
    default="EFFICIENT_ATTENTION",
    type=click.Choice(
        ["EFFICIENT_ATTENTION", "FLASH_ATTENTION", "CUDNN_ATTENTION", "MATH"]
    ),
    show_default=True,
    help="Attention backend to use.",
)
@click.option("--deepspeed", is_flag=True, help="Use DeepSpeed")
@click.option(
    "--num-gpu",
    type=click.IntRange(min=1),
    default=1,
    show_default=True,
    help="Number of GPUs to use for compression",
)
@click.option(
    "--offload-hessians",
    is_flag=True,
    default=False,
    help="Offload hessians to CPU during compression",
)
def main(
    model_id,
    dataset_id,
    dataset_split,
    num_calibration_samples,
    max_sequence_length,
    apply_chat_template,
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
    num_gpu,
    offload_hessians,
):
    # Set up attention backend.
    attention = getattr(SDPBackend, attention_backend)

    device_map = calculate_offload_device_map(
        model_id, reserve_for_hessians=True, num_gpus=num_gpu, torch_dtype="auto"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=device_map,
        torch_dtype=dtype,
        trust_remote_code=trust_remote_code,
    )

    # Load dataset and preprocess.
    ds = load_dataset(dataset_id, split=dataset_split)
    ds = ds.shuffle(seed=42).select(range(num_calibration_samples))

    if apply_chat_template:

        def preprocess_fn(example):
            return {
                "text": tokenizer.apply_chat_template(
                    example["messages"], tokenize=False
                )
            }

    else:

        def preprocess_fn(example):
            string = " ".join(example.values())
            return {"text": string}

    ds = ds.map(preprocess_fn)

    ignore_list = ignore
    recipe = []
    if scheme.startswith("FP8"):
        recipe = QuantizationModifier(
            targets="Linear", scheme=scheme, ignore=ignore_list
        )
    elif scheme == "W4A16": # Pure GPTQ
        recipe = [
            GPTQModifier(
                targets="Linear",
                scheme=scheme,
                ignore=ignore_list,
                dampening_frac=dampening_frac,
                offload_hessians=offload_hessians,
            ),
        ]
    elif scheme.startswith("W"): # W4A4, W8A8
        recipe = [
            SmoothQuantModifier(smoothing_strength=smoothing_strength),
            GPTQModifier(
                targets="Linear",
                scheme=scheme,
                ignore=ignore_list,
                dampening_frac=dampening_frac,
                offload_hessians=offload_hessians,
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
