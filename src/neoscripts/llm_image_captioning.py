#!/usr/bin/env python
# Generate image captions using an OpenAI-compatible API.
# Designed for GNU parallel batch.
# Example:
#     find . -type f \( -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" \) -print0 | \
#         parallel --bar -0 -j 16 -n 16 --lb ~/vcs/scripts/llm-image-captioning.py '{}' --existing-caption skip

import os
import sys
import base64
from io import BytesIO

import click
import humanize
from openai import OpenAI
from PIL import Image
from rich import print
from rich.console import Console
from rich.padding import Padding

console = Console()


def get_image_base64(
    image_path: str, downscale: bool = True, size: tuple[int, int] = (2048, 2048)
) -> str:
    image = Image.open(image_path)
    if image.mode != "RGB":
        image = image.convert("RGB")
    image_jpeg = BytesIO()

    if downscale:
        image.thumbnail(size)

    image.save(image_jpeg, format="JPEG", quality=90)
    del image
    return base64.b64encode(image_jpeg.getvalue()).decode("utf-8")


def get_caption_for_image(
    client: OpenAI,
    model_name: str,
    vision_prompt: str,
    image_base64: str,
    temperature: float | None,
    max_tokens: int | None,
) -> str:
    chat_response = client.chat.completions.create(
        model=model_name,
        stream=False,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": vision_prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                    },
                ],
            },
        ],
    )
    return chat_response.choices[0].message.content


def panic(e: str):
    print("[red]Error:\t" + e + "[/red]")
    sys.exit(1)


@click.command()
@click.argument("file_paths", nargs=-1, required=True)
@click.option(
    "-o",
    "--caption-output",
    default=None,
    help="Path to save the caption output. Can't be used with multiple images.",
)
@click.option(
    "-vp",
    "--vision-prompt-file",
    default=None,
    help="Vision prompt file for caption generation (default: built-in).",
)
@click.option(
    "-ap",
    "--additional-prompt",
    default=None,
    help='Additional prompt for caption generation. Replaces "%%" in the prompt (default: None).',
)
@click.option(
    "-m",
    "--model",
    default=None,
    help="Model to use for caption generation (default: pixtral-large-latest if LLM_MODEL_NAME is not set).",
)
@click.option(
    "-k",
    "--api-key",
    default=None,
    help="Override LLM API key (default: from LLM_API_KEY environment variable).",
)
@click.option(
    "-u",
    "--base-url",
    default=None,
    help="Override LLM base URL (default: from LLM_BASE_URL environment variable).",
)
@click.option(
    "-t",
    "--temperature",
    type=float,
    default=None,
    help="Temperature for caption generation.",
)
@click.option(
    "-mt",
    "--max_tokens",
    type=int,
    default=None,
    help="Maximum number of tokens in the caption.",
)
@click.option(
    "-v", "--verbose", is_flag=True, default=False, help="Enable verbose output."
)
@click.option(
    "-ex",
    "--existing-caption",
    default="overwrite",
    type=click.Choice(["overwrite", "skip"]),
    help='"overwrite" or "skip" existing caption file.',
)
@click.option(
    "-ce",
    "--caption-extension",
    default="txt",
    help="Caption file extension (default: txt).",
)
@click.option(
    "--no-downscale",
    "-nd",
    is_flag=True,
    help="Disable image downscaling before uploading.",
)
def __main__(
    file_paths,
    caption_output,
    vision_prompt_file,
    additional_prompt,
    model,
    api_key,
    base_url,
    temperature,
    max_tokens,
    verbose,
    existing_caption,
    caption_extension,
    no_downscale,
):
    """
    Generate image captions for one or more image files.
    """
    # Validate arguments
    if len(file_paths) > 1 and caption_output is not None:
        console.log("Error: Can't use --caption_output with multiple input images")
        sys.exit(1)

    if caption_output is not None and not caption_output.endswith(
        "." + caption_extension
    ):
        console.log(
            "[bright_yellow]INFO:[/bright_yellow] Caption extension will be ignored if --caption_output is provided"
        )

    # API key resolution
    final_api_key = "xxxx"  # default placeholder
    if os.environ.get("LLM_API_KEY") is not None:
        final_api_key = os.environ.get("LLM_API_KEY")
    if api_key is not None:
        final_api_key = api_key

    # Model resolution
    model_name = "pixtral-large-latest"
    if os.environ.get("LLM_MODEL_NAME") is not None:
        model_name = os.environ.get("LLM_MODEL_NAME")
    if model is not None:
        model_name = model

    # Base URL resolution
    final_base_url = "https://api.mistral.ai/v1"
    if os.environ.get("LLM_BASE_URL") is not None:
        final_base_url = os.environ.get("LLM_BASE_URL")
    if base_url is not None:
        final_base_url = base_url

    # Set default vision prompt
    vision_prompt = (
        "Describe this image, which will be used to train an image generator in natural language."
        " The caption should be as descriptive, detailed, and specific as possible,"
        " including people's ethnicity, gender, face, eye color, hair color,"
        " clothing, accessories, objects, actions, context and camera angle, etc."
        " Do not mention the image itself, the caption should not start with phrases"
        ' like "An image of", "This image", "The image" or "A photo of". %%'
    )

    if vision_prompt_file is not None:
        try:
            with open(vision_prompt_file, "r", encoding="utf-8") as f:
                vision_prompt = f.read()
        except Exception as e:
            panic(str(e))

    # Process additional prompt
    vision_prompt_for_colored_log = vision_prompt
    if additional_prompt is not None:
        vision_prompt_for_colored_log = vision_prompt.replace(
            " %%", "[bright_cyan] %%[/bright_cyan]", 1
        )
        vision_prompt = vision_prompt.replace("%%", additional_prompt, 1)
        vision_prompt_for_colored_log = vision_prompt_for_colored_log.replace(
            "%%", additional_prompt, 1
        )
    else:
        vision_prompt = vision_prompt.replace(" %%", "", 1)
        vision_prompt_for_colored_log = vision_prompt_for_colored_log.replace(
            " %%", "", 1
        )

    if verbose:
        print("================================================")
        print("Model:\t\t[cyan]" + model_name + "[/cyan]")
        print("Base URL:\t[cyan]" + final_base_url + "[/cyan]")
        print("Temperature:\t[cyan]" + str(temperature) + "[/cyan]")
        print("Max tokens:\t[cyan]" + str(max_tokens) + "[/cyan]")
        print("Full vision prompt:")
        print(
            Padding(
                "[bright_magenta]"
                + vision_prompt_for_colored_log
                + "[/bright_magenta]",
                (0, 0, 0, 4),
            )
        )
        print("================================================")

    # Initialize OpenAI client
    client = OpenAI(api_key=final_api_key, base_url=final_base_url)

    for file_path in file_paths:
        try:
            this_caption_output = caption_output
            filename_info_line = ""
            if this_caption_output is None:
                filename_info_line = (
                    "[white on blue]>>[/white on blue] [yellow]"
                    + file_path
                    + "[/yellow] => [yellow]*."
                    + caption_extension
                )
                this_caption_output = (
                    os.path.splitext(file_path)[0] + "." + caption_extension
                )
            else:
                filename_info_line = (
                    "[white on blue]>>[/white on blue] [yellow]"
                    + file_path
                    + "[/yellow] => [yellow]"
                    + this_caption_output
                    + "[/yellow]"
                )

            if verbose:
                console.log(filename_info_line)

            if os.path.exists(this_caption_output) and existing_caption == "skip":
                console.log("Caption file already exists, skipping...")
                continue

            try:
                caption_file = open(this_caption_output, "w", encoding="utf-8")
            except Exception as e:
                panic(str(e))

            image_base64 = get_image_base64(file_path, not no_downscale)

            if verbose:
                console.log(
                    "Payload size: [bright_yellow]"
                    + humanize.naturalsize(len(image_base64), binary=True)
                    + "[/bright_yellow]"
                )
                console.log("Caption:")

            try:
                caption_response = get_caption_for_image(
                    client,
                    model_name,
                    vision_prompt,
                    image_base64,
                    temperature,
                    max_tokens,
                )
            except Exception as e:
                panic(str(e))

            padded_response = Padding(
                "[green]" + caption_response + "[/green]", (0, 0, 0, 4)
            )
            if verbose:
                console.log(padded_response)
            else:
                print(filename_info_line)
                print(padded_response)

            caption_response += "\n"
            try:
                caption_file.write(caption_response)
                caption_file.flush()
                if verbose:
                    console.log(
                        "Caption saved to [yellow]"
                        + this_caption_output
                        + "[/yellow] ([bright_yellow]"
                        + humanize.naturalsize(caption_file.tell(), binary=True)
                        + "[/bright_yellow])"
                    )
                caption_file.close()
            except Exception as e:
                panic(str(e))

        except Exception as e:
            console.log(f"[red]Error: {e}[/red]")
            continue  # Continue on error for each file

    sys.exit(0)


if __name__ == "__main__":
    try:
        __main__()
    except Exception as e:
        panic(str(e))
