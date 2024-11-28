#!/usr/bin/env python
# Simple Python script to generate image captions using OpenAI API
# Designed to be used with GNU parallel to achieve batch inference
# Example:
# find . -type f \( -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" \) -print0 | \
#     parallel --bar -0 -j 16 ~/vcs/scripts/llm-image-captioning.py '{}' -ex skip

import os
import sys
import base64
from PIL import Image
from io import BytesIO
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
            description='Generate image captions using OpenAI-compatible API')
    parser.add_argument('file_paths', help='Path(s) to the input image file', nargs='+')
    parser.add_argument('-o', '--caption_output',
            help='Path to save the caption output, can\'t be used with multiple input images (default: same as image file with .txt extension)')
    parser.add_argument('-vp', '--vision_prompt_file',
            help='Vision prompt file for caption generation (default: built-in)')
    parser.add_argument('-ap', '--additional_prompt',
            help='Additional prompt for caption generation, replaces "%%" in the prompt, " %%" will be removed if it\'s not set (default: None)')
    parser.add_argument('-m', '--model',
            help='Model to use for caption generation (default: pixtral-large-latest if LLM_MODEL_NAME environment variable is not set)')
    parser.add_argument('-k', '--api_key',
            help='Override LLM API key (default: content of LLM_API_KEY environment variable)')
    parser.add_argument('-u', '--base_url',
            help='Override LLM base URL (default: Mistral\'s)')
    parser.add_argument('-t', '--temperature', type=float, default=0.7,
            help='Temperature for caption generation (default: 0.7)')
    parser.add_argument('-tp', '--top_p', type=float, default=0.9,
            help='Top P for caption generation (default: 0.9)')
    parser.add_argument('-mt', '--max_tokens', type=int, default=500,
            help='Maximum number of tokens in the caption (default: 500)')
    parser.add_argument('-v', '--verbose', default=False, action='store_true',
            help='Verbose output (default: False)')
    parser.add_argument('-ex', '--existing_caption', default="overwrite", choices=["overwrite", "skip"],
            help='"overwrite" or "skip" existing caption file (default: overwrite)')
    parser.add_argument('-ce', '--caption_extension', default="txt",
            help='Caption file extension, will be ignored if --caption_output is provided (default: txt)')
    return parser.parse_args()

def panic(e):
    print("[red]Error:\t" + str(e) + "[/red]")
    sys.exit(1)

def get_image_base64(image_path):
    # Load the image
    image = Image.open(image_path)

    if image.mode != "RGB":
        image = image.convert("RGB")

    image_jpeg = BytesIO()

    # Resize the image to fit within 2048x2048 pixels
    width, height = image.size
    if width > 2048 or height > 2048:
        scale_factor = max(width, height) / 2048
        image = image.resize((int(width / scale_factor), int(height / scale_factor)))

    image.save(image_jpeg, format="JPEG", quality=90) # Use JPEG quality 90 to reduce file size
    del image

    # Convert the image to a base64 string
    return base64.b64encode(image_jpeg.getvalue()).decode("utf-8")

def get_caption_from_image(client, model_name, vision_prompt, image_base64, temperature, top_p, max_tokens):
    chat_response = client.chat.completions.create(
        model=model_name,
        stream=False,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": vision_prompt
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                }
            ]
        }]
    )
    return chat_response.choices[0].message.content

def __main__():
    args = parse_args()
    file_paths = args.file_paths
    additional_prompt = args.additional_prompt
    verbose = args.verbose
    temperature = args.temperature
    caption_extension = args.caption_extension
    top_p = args.top_p
    max_tokens = args.max_tokens

    # Argument validation
    if len(file_paths) > 1 and args.caption_output is not None:
        print("Error: Can't use --caption_output with multiple input images")
        sys.exit(1)
    
    if args.caption_output is not None and not args.caption_output.endswith("." + caption_extension):
        print("[bright_yellow]INFO:[/bright_yellow] Caption extension will be ignored if -o/--caption_output is provided")

    # Check if the API key is set
    api_key = "xxxx" # placeholder, it's fine for it to be any string for local LLM
    if (key := os.environ.get("LLM_API_KEY")) is not None:
        api_key = key
    # Override if provided by command line argument
    if args.api_key is not None:
        api_key = args.api_key

    # Get model name from environment variable if it exists
    model_name = "pixtral-large-latest" # default model from Mistral
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

    vision_prompt = \
        "Describe this image, which will be used to train an image generator in natural language." \
        + " The caption should be as descriptive, detailed, and specific as possible," \
        + " including people's ethnicity, gender, face, eye color, hair color," \
        + " clothing, accessories, objects, actions, and context, camera angle, etc." \
        + " The caption should not start with phrases like \"An image of\" or \"A photo of\". %%"

    if args.vision_prompt_file is not None:
        with open(args.vision_prompt_file, "r", encoding="utf-8") as f:
            vision_prompt = f.read()

    # Save vision prompt for log colorization
    vision_prompt_for_colored_log = vision_prompt
    if additional_prompt is not None:
        vision_prompt_for_colored_log = vision_prompt.replace(" %%", "[bright_cyan] %%[/bright_cyan]", 1)

        vision_prompt = vision_prompt.replace("%%", additional_prompt, 1)
        vision_prompt_for_colored_log = vision_prompt_for_colored_log.replace("%%", additional_prompt, 1)
    else:
        vision_prompt = vision_prompt.replace(" %%", "", 1) # Remove the extra space
        vision_prompt_for_colored_log = vision_prompt_for_colored_log.replace(" %%", "", 1)

    # Print all information in a banner
    if verbose:
        print("================================================")
        print("Model:\t\t[cyan]" + model_name + "[/cyan]")
        print("Base URL:\t[cyan]" + base_url + "[/cyan]")
        print("Temperature:\t[cyan]" + str(temperature) + "[/cyan]")    
        print("Top P:\t\t[cyan]" + str(top_p) + "[/cyan]")
        print("Max tokens:\t[cyan]" + str(max_tokens) + "[/cyan]")
        print("Full vision prompt:")
        print(Padding("[bright_magenta]" + vision_prompt_for_colored_log + "[/bright_magenta]", (0, 0, 0, 4)))
        print("================================================")

    # Initialize OpenAI client
    client = OpenAI(
        api_key=api_key,
        base_url=base_url
    )

    for file_path in file_paths:
        try: # catch all exceptions and continue by default
            caption_output = args.caption_output
            if caption_output is None:
                print("\n[white on blue]>>[/white on blue] [yellow]" + file_path + "[/yellow] => [yellow] *." + caption_extension)
                caption_output = os.path.splitext(file_path)[0] + "." + caption_extension
            else:
                print("\n[white on blue]>>[/white on blue] [yellow]" + file_path + "[/yellow] => [yellow]" + caption_output + "[/yellow]")

            if os.path.exists(caption_output) and args.existing_caption == "skip":
                print("Caption file already exists, skipping...")
                continue
            # Open caption file before proceeding, so we can catch errors early
            caption_file = open(caption_output, "w", encoding="utf-8")

            image_base64 = get_image_base64(file_path)

            if verbose:
                # print payload size in human readable format
                print("Payload size: [bright_yellow]" + humanize.naturalsize(len(image_base64)) + "[/bright_yellow]")
                print("Caption:")

            try:
                caption_response = get_caption_from_image(client, model_name, vision_prompt, image_base64, temperature, top_p, max_tokens)
            except Exception as e:
                panic(str(e)) # Exit on API error

            print(Padding("[green]" + caption_response + "[/green]", (0, 0, 0, 4)))

            caption_response += "\n" # Add a newline at the end of the caption
            # Save the caption to a file
            try:
                caption_file.write(caption_response)
                caption_file.close()
            except Exception as e:
                panic(str(e)) # Exit on I/O error, so we won't waste tokens

            if verbose:
                print("Caption saved to [yellow]" + caption_output \
                    + "[/yellow] ([bright_yellow]" + humanize.naturalsize(len(caption_response)) \
                    + "[/bright_yellow])")

        except Exception as e:
            print(f"[red]Error: {e}[/red]")
            pass # Continue on all other errors
    # End of Loop


if __name__ == "__main__":
    __main__()
