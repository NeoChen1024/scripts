#!/usr/bin/env python
# Simple Python script to translate text using OpenAI API
# Designed to be used with GNU parallel to achieve batch inference
# Example:
# find . -type f \( -iname "*.txt" \) -print0 | \
#     parallel --bar -0 -j 16 ~/vcs/scripts/llm-chain-translation.py -i '{}' -o 'out/{/}' -ex skip

import math
import os
import sys

import click
import humanize
from openai import OpenAI
from rich import print
from rich.console import Console
from rich.padding import Padding
from rich.progress import Progress


def get_translation_from_text(
    client,
    model_name,
    system_prompt,
    text,
    temperature,
    top_p,
    max_tokens,
    last_round_original,
    last_round_translated,
):

    messages = []
    # Add system prompt
    messages.append({"role": "system", "content": system_prompt})
    if last_round_original is not None and last_round_translated is not None:
        messages.append({"role": "user", "content": last_round_original})
        messages.append({"role": "assistant", "content": last_round_translated})

    messages.append({"role": "user", "content": text})

    chat_response = client.chat.completions.create(
        model=model_name,
        stream=False,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        messages=messages,
    )
    return chat_response.choices[0].message.content


@click.command()
@click.option(
    "-i",
    "--input-file",
    "--input_file",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to the input text file",
)
@click.option(
    "-o",
    "--output-file",
    "--output_file",
    required=True,
    type=click.Path(dir_okay=False),
    help="Path to save the translated output",
)
@click.option(
    "-l",
    "--language",
    default="English",
    show_default=True,
    help="Language to translate the text to; replaces all %LANG% in the system prompt",
)
@click.option(
    "-n",
    "--lines",
    type=click.IntRange(min=1),
    default=20,
    show_default=True,
    help="Number of lines to process in one round",
)
@click.option(
    "-p",
    "--prompt-file",
    "--prompt_file",
    type=click.Path(exists=True, dir_okay=False),
    help="System prompt file for translation (default: built-in)",
)
@click.option(
    "-ap",
    "--additional-prompt",
    "--additional_prompt",
    help='Additional prompt for translation, replaces "%%" in the prompt, " %%" will be removed if it\'s not set (default: None)',
)
@click.option(
    "-m",
    "--model",
    help="Model to use for translation (default: mistral-large-latest if LLM_MODEL_NAME environment variable is not set)",
)
@click.option(
    "-k",
    "--api-key",
    "--api_key",
    help="Override LLM API key (default: content of LLM_API_KEY environment variable)",
)
@click.option("-u", "--base-url", "--base_url", help="Override LLM base URL (default: Mistral's)")
@click.option("-t", "--temperature", type=float, default=0.7, show_default=True, help="Temperature for translation")
@click.option("-tp", "--top-p", "--top_p", type=float, default=0.9, show_default=True, help="Top P for translation")
@click.option(
    "-mt",
    "--max-tokens",
    "--max_tokens",
    type=click.IntRange(min=1),
    default=500,
    show_default=True,
    help="Maximum number of tokens in the translation",
)
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
@click.option(
    "-ex",
    "--existing-translation",
    "--existing_translation",
    default="overwrite",
    show_default=True,
    type=click.Choice(["overwrite", "skip"]),
    help="How to handle an existing translation file",
)
def __main__(
    input_file,
    output_file,
    language,
    lines,
    prompt_file,
    additional_prompt,
    model,
    api_key,
    base_url,
    temperature,
    top_p,
    max_tokens,
    verbose,
    existing_translation,
):
    """Translate text using an OpenAI-compatible API."""

    console = Console()
    lines_per_round = lines

    if os.path.exists(output_file) and existing_translation == "skip":
        console.log("Translation file already exists, skipping...")
        return

    # Check if the API key is set
    resolved_api_key = "xxxx"  # placeholder, it's fine for it to be any string for local LLM
    if (key := os.environ.get("LLM_API_KEY")) is not None:
        resolved_api_key = key
    # Override if provided by command line argument
    if api_key is not None:
        resolved_api_key = api_key

    # Get model name from environment variable if it exists
    model_name = "mistral-large-latest"  # default model from Mistral
    if (name := os.environ.get("LLM_MODEL_NAME")) is not None:
        model_name = name
    # Override if provided by command line argument
    if model is not None:
        model_name = model

    resolved_base_url = "https://api.mistral.ai/v1"
    if (url := os.environ.get("LLM_BASE_URL")) is not None:
        resolved_base_url = url
    # Override if provided by command line argument
    if base_url is not None:
        resolved_base_url = base_url

    system_prompt = (
        "You are a linguistic expert capable of identifying and translating text from "
        + "various languages into %LANG%. Given the following text, determine the original "
        + "language and provide an accurate %LANG% translation. Ensure the translation "
        + "maintains the context and meaning of the original text according to the provided context. "
        + "Output only the %LANG% translation without any additional information or context. "
        + "Focus on delivering an accurate and contextually relevant translation."
        + " %%"
    )

    if prompt_file is not None:
        with open(prompt_file, "r", encoding="utf-8") as f:
            system_prompt = f.read()

    # Replace all %LANG% with the language
    system_prompt = system_prompt.replace("%LANG%", language)

    # Save system prompt for log colorization
    system_prompt_for_colored_log = system_prompt
    if additional_prompt is not None:
        system_prompt_for_colored_log = system_prompt_for_colored_log.replace(" %%", "[bright_cyan] %%[/bright_cyan]", 1)

        system_prompt = system_prompt.replace("%%", additional_prompt, 1)
        system_prompt_for_colored_log = system_prompt_for_colored_log.replace("%%", additional_prompt, 1)
    else:
        system_prompt = system_prompt.replace(" %%", "", 1)  # Remove the extra space
        system_prompt_for_colored_log = system_prompt_for_colored_log.replace(" %%", "", 1)

    # Print all information in a banner
    if verbose:
        print("================================================")
        print("Model:\t\t[cyan]" + model_name + "[/cyan]")
        print("Base URL:\t[cyan]" + resolved_base_url + "[/cyan]")
        print("Temperature:\t[cyan]" + str(temperature) + "[/cyan]")
        print("Top P:\t\t[cyan]" + str(top_p) + "[/cyan]")
        print("Max tokens:\t[cyan]" + str(max_tokens) + "[/cyan]")
        print("Prompt:")
        print(
            Padding(
                "[bright_magenta]" + system_prompt_for_colored_log + "[/bright_magenta]",
                (0, 0, 0, 4),
            )
        )
        print("================================================")

    # Initialize OpenAI client
    client = OpenAI(api_key=resolved_api_key, base_url=resolved_base_url)

    print("\n[white on blue]>>[/white on blue] [yellow]" + input_file + "[/yellow] => [yellow]" + output_file + "[/yellow]")

    # Chained translation
    current_round_original = ""
    current_round_translated = ""
    last_round_original = None
    last_round_translated = None

    with open(input_file, "r", encoding="utf-8") as f:
        text = f.read()

    translation_file = open(output_file, "w", encoding="utf-8")
    # truncate the file
    translation_file.truncate()

    lines = text.splitlines()

    round_number = 1
    total_rounds = math.ceil(len(lines) / lines_per_round)

    while len(lines) > 0:
        # take the first lines_per_round lines
        current_round_original = "\n".join(lines[:lines_per_round])
        lines = lines[lines_per_round:]

        console.log("================================================")
        console.log("Round [bright_white]" + str(round_number) + "/" + str(total_rounds) + "[/bright_white] Original:")
        console.log(
            Padding(
                "[bright_yellow]" + current_round_original + "[/bright_yellow]",
                (0, 0, 0, 4),
            )
        )

        current_round_translated = get_translation_from_text(
            client,
            model_name,
            system_prompt,
            current_round_original,
            temperature,
            top_p,
            max_tokens,
            last_round_original,
            last_round_translated,
        )

        console.log("================================================")
        console.log("Translated:")
        console.log(
            Padding(
                "[bright_green]" + current_round_translated + "[/bright_green]",
                (0, 0, 0, 4),
            )
        )
        # Update last round
        last_round_original = current_round_original
        last_round_translated = current_round_translated
        round_number += 1

        # Write to translation file
        current_round_translated += "\n"
        translation_file.write(current_round_translated)
        translation_file.flush()

    console.log(
        "Translation saved to [yellow]"
        + output_file
        + "[/yellow] ([bright_yellow]"
        + humanize.naturalsize(translation_file.tell(), binary=True)
        + "[/bright_yellow])"
    )
    translation_file.close()


if __name__ == "__main__":
    try:
        __main__()
    except Exception as e:
        print("[red]Error:\t" + str(e) + "[/red]")
        sys.exit(1)
