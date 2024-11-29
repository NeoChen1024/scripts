#!/usr/bin/env python
# Simple Python script to translate text using OpenAI API
# Designed to be used with GNU parallel to achieve batch inference
# Example:
# find . -type f \( -iname "*.txt" \) -print0 | \
#     parallel --bar -0 -j 16 ~/vcs/scripts/llm-chain-translation.py -i '{}' -o 'out/{/}' -ex skip

import os
import sys
from openai import OpenAI
import argparse
from rich import print
from rich.padding import Padding
import humanize

def none_or_str(value):
    if value is None:
        return "None"
    return str(value)

def parse_args():
    parser = argparse.ArgumentParser(
            description='Translate text using OpenAI-compatible API')
    parser.add_argument('-i', '--input_file', required=True,
            help='Path to the input text file')
    parser.add_argument('-o', '--output_file', required=True,
            help='Path to save the translated output')
    parser.add_argument('-l', '--language', required=False, default="English",
            help='Language to translate the text to, that the LLM can understand (default: English)')
    parser.add_argument('-n', '--lines', type=int, default=20,
            help='Number of lines to process in one round (default: 20)')
    parser.add_argument('-p', '--prompt_file',
            help='Prompt file for translation (default: built-in)')
    parser.add_argument('-ap', '--additional_prompt',
            help='Additional prompt for translation, replaces "%%" in the prompt, " %%" will be removed if it\'s not set (default: None)')
    parser.add_argument('-m', '--model',
            help='Model to use for translation (default: mistral-large-latest if LLM_MODEL_NAME environment variable is not set)')
    parser.add_argument('-k', '--api_key',
            help='Override LLM API key (default: content of LLM_API_KEY environment variable)')
    parser.add_argument('-u', '--base_url',
            help='Override LLM base URL (default: Mistral\'s)')
    parser.add_argument('-t', '--temperature', type=float, default=0.7,
            help='Temperature for translation (default: 0.7)')
    parser.add_argument('-tp', '--top_p', type=float, default=0.9,
            help='Top P for translation (default: 0.9)')
    parser.add_argument('-mt', '--max_tokens', type=int, default=500,
            help='Maximum number of tokens in the translation (default: 500)')
    parser.add_argument('-v', '--verbose', default=False, action='store_true',
            help='Verbose output (default: False)')
    parser.add_argument('-ex', '--existing_translation', default="overwrite", choices=["overwrite", "skip"],
            help='"overwrite" or "skip" existing translation file (default: overwrite)')
    return parser.parse_args()

def panic(e):
    print("[red]Error:\t" + str(e) + "[/red]")
    sys.exit(1)

def get_translation_from_text(client, model_name, system_prompt, text, temperature, top_p, max_tokens,
                              last_round_original, last_round_translated):

    messages = []
    # Add system prompt
    messages.append({
        "role": "system",
        "content": system_prompt
    })
    if last_round_original is not None and last_round_translated is not None:
        messages.append({
            "role": "user",
            "content": last_round_original
        })
        messages.append({
            "role": "assistant",
            "content": last_round_translated
        })

    messages.append({
        "role": "user",
        "content": text
    })

    chat_response = client.chat.completions.create(
        model=model_name,
        stream=False,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        messages=messages
    )
    return chat_response.choices[0].message.content

def __main__():
    args = parse_args()
    input_file = args.input_file
    output_file = args.output_file
    language = args.language
    lines_per_round = args.lines
    prompt_file = args.prompt_file
    additional_prompt = args.additional_prompt
    verbose = args.verbose
    temperature = args.temperature
    top_p = args.top_p
    max_tokens = args.max_tokens

    # Argument validation
    if not os.path.exists(input_file):
        panic("Input file does not exist: " + input_file)
    
    if os.path.exists(output_file) and args.existing_translation == "skip":
        print("Translation file already exists, skipping...")
        sys.exit(0)

    # Check if the API key is set
    api_key = "xxxx" # placeholder, it's fine for it to be any string for local LLM
    if (key := os.environ.get("LLM_API_KEY")) is not None:
        api_key = key
    # Override if provided by command line argument
    if args.api_key is not None:
        api_key = args.api_key

    # Get model name from environment variable if it exists
    model_name = "mistral-large-latest" # default model from Mistral
    if (name := os.environ.get("LLM_MODEL_NAME")) is not None:
        model_name = name
    # Override if provided by command line argument
    if args.model is not None:
        model_name = args.model

    base_url = "https://api.mistral.ai/v1"
    if (url := os.environ.get("LLM_BASE_URL")) is not None:
        base_url = url
    # Override if provided by command line argument
    if args.base_url is not None:
        base_url = args.base_url

    system_prompt = \
          "You are a linguistic expert capable of identifying and translating text from " \
        + "various languages into " + language + ". Given the following text, determine the original " \
        + "language and provide an accurate " + language + " translation. Ensure the translation " \
        + "maintains the context and meaning of the original text according to the provided context. " \
        + "Output only the " + language + " translation without any additional information or context. " \
        + "Focus on delivering an accurate and contextually relevant translation." \
        + " %%"

    if prompt_file is not None:
        with open(prompt_file, "r", encoding="utf-8") as f:
            system_prompt = f.read()

    # Save system prompt for log colorization
    system_prompt_for_colored_log = system_prompt
    if additional_prompt is not None:
        system_prompt_for_colored_log = system_prompt_for_colored_log.replace(" %%", "[bright_cyan] %%[/bright_cyan]", 1)

        system_prompt = system_prompt.replace("%%", additional_prompt, 1)
        system_prompt_for_colored_log = system_prompt_for_colored_log.replace("%%", additional_prompt, 1)
    else:
        system_prompt = system_prompt.replace(" %%", "", 1) # Remove the extra space
        system_prompt_for_colored_log = system_prompt_for_colored_log.replace(" %%", "", 1)

    # Print all information in a banner
    if verbose:
        print("================================================")
        print("Model:\t\t[cyan]" + model_name + "[/cyan]")
        print("Base URL:\t[cyan]" + base_url + "[/cyan]")
        print("Temperature:\t[cyan]" + str(temperature) + "[/cyan]")    
        print("Top P:\t\t[cyan]" + str(top_p) + "[/cyan]")
        print("Max tokens:\t[cyan]" + str(max_tokens) + "[/cyan]")
        print("Prompt:")
        print(Padding("[bright_magenta]" + system_prompt_for_colored_log + "[/bright_magenta]", (0, 0, 0, 4)))
        print("================================================")

    # Initialize OpenAI client
    client = OpenAI(
        api_key=api_key,
        base_url=base_url
    )

    print("\n[white on blue]>>[/white on blue] [yellow]" + input_file + "[/yellow] => [yellow]" + output_file + "[/yellow]")

    # Chained translation
    current_round_original = ""
    current_round_translated = ""
    last_round_original= None
    last_round_translated= None
    all_translations = ""
    round_number = 0

    with open(input_file, "r", encoding="utf-8") as f:
        text = f.read()
    
    translation_file = open(output_file, "w", encoding="utf-8")
    # truncate the file
    translation_file.truncate()
    translation_file_size = 0

    # split the text into chunks, lines_per_round lines each
    lines = text.splitlines()
    while len(lines) > 0:
        # take the first lines_per_round lines
        current_round_original = "\n".join(lines[:lines_per_round])
        lines = lines[lines_per_round:]

        print("================================================")
        print("Round [bright_white]" + str(round_number) + "[/bright_white] Input:")
        print(Padding("[bright_yellow]" + current_round_original + "[/bright_yellow]", (0, 0, 0, 4)))

        current_round_translated = get_translation_from_text(
            client, model_name, system_prompt, current_round_original, temperature, top_p, max_tokens,
            last_round_original, last_round_translated)

        print("================================================")
        print("Translation:")
        print(Padding("[bright_green]" + current_round_translated + "[/bright_green]", (0, 0, 0, 4)))
        # Update last round
        last_round_original = current_round_original
        last_round_translated = current_round_translated
        round_number += 1

        # Write to translation file
        current_round_translated += "\n"
        translation_file_size += len(current_round_translated)
        translation_file.write(current_round_translated)
        translation_file.flush()

    print("Translation saved to [yellow]" + output_file + "[/yellow] ([bright_yellow]" \
        + humanize.naturalsize(translation_file_size) + "[/bright_yellow])")

if __name__ == "__main__":
    try:
        __main__()
    except Exception as e:
        panic(str(e))
