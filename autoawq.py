#!/usr/bin/env python

# Require latest autoawq

import click
from datasets import load_dataset
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer


@click.command()
@click.option("--model-id", "-m", required=True, type=str, help="Path to the model")
@click.option(
    "--output-path",
    "-q",
    type=str,
    help="Path to save the quantized model",
)
@click.option(
    "--attn-implementation",
    default="sdpa",
    type=click.Choice(["sdpa", "flash_attention_2", "eager"]),
    help="Attention backend to use.",
)
@click.option(
    "--dataset-id",
    default="neuralmagic/LLM_compression_calibration",
    help="Dataset ID to use for calibration.",
)
@click.option(
    "--num-calibration-samples",
    default=256,
    type=int,
    help="Number of samples to use for calibration.",
)
@click.option(
    "--max-sequence-length",
    default=1024,
    type=int,
    help="Maximum sequence length to use for calibration.",
)
@click.option(
    "--num-parallel-calib-samples",
    default=32,
    type=int,
    help="Number of parallel calibration samples to process at a time.",
)
@click.option(
    "--sample-prompt",
    default="Hello my name is",
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
    help="Data type to use for model weights. (Must use half on Turing GPUs)",
)
def _main(
    model_id,
    output_path,
    attn_implementation,
    dataset_id,
    num_calibration_samples,
    max_sequence_length,
    num_parallel_calib_samples,
    sample_prompt,
    trust_remote_code,
    dtype,
):
    quant_config = {
        "zero_point": True,
        "q_group_size": 128,
        "w_bit": 4,
        "version": "GEMM",
    }

    # Load model
    model = AutoAWQForCausalLM.from_pretrained(
        model_id,
        attn_implementation=attn_implementation,
        torch_dtype=dtype,
        trust_remote_code=trust_remote_code,
        low_cpu_mem_usage=True,
        use_cache=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    # Quantize
    model.quantize(
        tokenizer,
        quant_config=quant_config,
        calib_data=dataset_id,
        max_calib_samples=num_calibration_samples,
        max_calib_seq_len=max_sequence_length,
        n_parallel_calib_samples=num_parallel_calib_samples,
    )

    save_dir = model_id + "-" + "AWQ"
    if output_path is not None:
        save_dir = output_path
    # Save quantized model
    model.save_quantized(save_dir)
    tokenizer.save_pretrained(save_dir)
    click.echo(f'Model is quantized and saved at "{save_dir}"')

    click.echo("\n\n")
    click.echo("========== SAMPLE GENERATION ==============")
    input_ids = tokenizer(sample_prompt, return_tensors="pt").input_ids.to("cuda")
    output = model.generate(input_ids, max_new_tokens=100)
    click.echo(tokenizer.decode(output[0]))
    click.echo("==========================================\n\n")


if __name__ == "__main__":
    _main()
