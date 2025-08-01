#!/usr/bin/env python
# Generate image captions using an OpenAI-compatible API.

import base64
import os
import sys
from io import BytesIO
from typing import Dict, Optional, Tuple, List, Union, Any
import threading

import click
import humanize
from openai import OpenAI
from openai.types.chat import ChatCompletionUserMessageParam, ChatCompletionSystemMessageParam, ChatCompletionAssistantMessageParam
from PIL import Image
from rich import print
from rich.traceback import install as install_traceback
from rich.console import Console
from rich.padding import Padding

console = Console()
install_traceback(show_locals=True)


# fallback: default -> environment variable -> override
def _fallback_environment(default: str, env_var: str, override: Optional[str] = None) -> str:
    val = default
    if (env_val := os.environ.get(env_var)) is not None:
        val = env_val
    if override is not None:
        val = override
    return val


def get_image_base64(image_or_path: Union[str, Image.Image], downscale: bool = True, size: Tuple[int, int] = (2048, 2048)) -> str:
    """
    Convert an image (PIL.Image or file path) to a base64-encoded JPEG string.
    Downscale if requested. Uses efficient in-memory operations.
    """
    # Load image if path is given
    img = Image.open(image_or_path) if isinstance(image_or_path, str) else image_or_path
    img = img.convert("RGB")
    if downscale:
        img.thumbnail(size, Image.Resampling.LANCZOS)
    with BytesIO() as output:
        img.save(output, format="JPEG", quality=90, optimize=True)
        return base64.b64encode(output.getvalue()).decode("utf-8")


def load_examples_from_dir(
    examples_dir: str, downscale: bool = True, size: tuple[int, int] = (2048, 2048)
) -> Tuple[int, List[Union[ChatCompletionUserMessageParam, ChatCompletionAssistantMessageParam]]]:
    examples = []
    total_payload_size = 0
    for example_file in os.listdir(examples_dir):
        # test if the file is an image with PIL
        try:
            if example_file.lower().endswith(".txt"):
                continue
            print(f"Loading example: [yellow]{example_file}[/yellow]")
            image_path = os.path.join(examples_dir, example_file)
            image = Image.open(image_path)
            image.verify()  # PITFALL: verify() closes the file handle
            image = Image.open(image_path)
            # find a text file with the same name
            caption_file = os.path.splitext(example_file)[0] + ".txt"
            caption_path = os.path.join(examples_dir, caption_file)
            if not os.path.exists(caption_path):
                console.log(f"[red]Warning:[/red] Caption file {caption_file} not found for example {example_file}. Skipping.")
                continue
            with open(caption_path, "r", encoding="utf-8") as f:
                caption = f.read().strip()
            if not caption:
                console.log(f"[red]Warning:[/red] Caption file {caption_file} is empty for example {example_file}. Skipping.")
                continue
            image_base64 = get_image_base64(image, downscale, size)
            total_payload_size += len(image_base64)
            examples.extend(
                [
                    ChatCompletionUserMessageParam(
                        role="user",
                        content=[
                            {"type": "text", "text": "Generate the caption for the following image."},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                            },
                        ],
                    ),
                    ChatCompletionAssistantMessageParam(
                        role="assistant",
                        content=caption,
                    ),
                ]
            )
        except (IOError, SyntaxError) as e:
            continue
    return total_payload_size, examples


def get_caption_for_image(
    client: OpenAI,
    model_name: str,
    vision_prompt: str,
    image_base64: str,
    examples: Optional[List[Union[ChatCompletionUserMessageParam, ChatCompletionAssistantMessageParam]]] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> str:
    kwarg = {}
    if temperature is not None:
        kwarg["temperature"] = temperature
    if max_tokens is not None:
        kwarg["max_tokens"] = max_tokens
    messages: List[Union[ChatCompletionUserMessageParam, ChatCompletionSystemMessageParam, ChatCompletionAssistantMessageParam]] = [
        ChatCompletionSystemMessageParam(
            role="system",
            content=vision_prompt,
        )
    ]
    if examples is not None:
        messages.extend(examples)
    messages.append(
        ChatCompletionUserMessageParam(
            role="user",
            content=[
                {"type": "text", "text": "Generate the caption for the following image."},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                },
            ],
        )
    )
    chat_response = client.chat.completions.create(
        model=model_name,
        stream=False,
        messages=messages,
        **kwarg,
    )
    r = chat_response.choices[0].message.content
    if r is None:
        raise ValueError("Caption generation failed, no content returned.")
    return r.strip() + "\n"


def panic(e: str):
    print("[red]Error:\t" + e + "[/red]")
    sys.exit(1)


@click.command()
@click.argument("file_paths", nargs=-1, required=True)
@click.option(
    "-e",
    "--examples-dir",
    default=None,
    help="Directory with example image & caption pairs for few-shot prompting (default: None).",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
)
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
@click.option("-v", "--verbose", is_flag=True, default=False, help="Enable verbose output.")
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
    model,
    api_key,
    base_url,
    temperature,
    max_tokens,
    verbose,
    existing_caption,
    caption_extension,
    no_downscale,
    examples_dir,
):
    """
    Generate image captions for one or more image files.
    """
    # Validate arguments
    if len(file_paths) > 1 and caption_output is not None:
        console.log("Error: Can't use --caption_output with multiple input images")
        sys.exit(1)

    if caption_output is not None and not caption_output.endswith("." + caption_extension):
        console.log("[bright_yellow]INFO:[/bright_yellow] Caption extension will be ignored if --caption_output is provided")

    # API key resolution
    final_api_key = _fallback_environment("xxxx", "LLM_API_KEY", api_key)

    # Model resolution
    model_name = _fallback_environment("pixtral-large-latest", "LLM_MODEL_NAME", model)

    # Base URL resolution
    final_base_url = _fallback_environment("https://api.mistral.ai/v1", "LLM_BASE_URL", base_url)

    # Set default vision prompt
    vision_prompt = """
You are an expert image analyst. Your purpose is to generate highly detailed, objective, and literal descriptions of images.
These descriptions will be used as high-quality training data for a text-to-image AI generator.

Your task is to create a single, cohesive paragraph of natural language that describes the provided image.

**Key requirements for the description:**
- **Be hyper-descriptive and specific:** The goal is to capture every visual detail possible.
- **Include diverse attributes:** Mention elements like ethnicity, gender, age, facial expressions, eye color, hair color and style, clothing, and accessories for any people present.
- **Describe the environment:** Detail the setting, objects, background, and overall context.
- **Capture action and mood:** Describe what is happening, the interactions between subjects, and the overall atmosphere (e.g., joyful, serene, chaotic).
- **Specify visual composition:** Mention camera angle (e.g., eye-level shot, low-angle shot, aerial view), lighting (e.g., soft morning light, harsh noon sun, neon glow), and focus (e.g., shallow depth of field with a blurry background).

**Strict constraints (What NOT to do):**
- **DO NOT** refer to the image as an "image", "photo", "picture", or any similar term.
- **DO NOT** start the caption with phrases like "An image of...", "This is a photo of...", "The image shows...".
- **DO NOT** interpret or add information that is not visually present. Be objective and literal.
    """

    if vision_prompt_file is not None:
        try:
            with open(vision_prompt_file, "r", encoding="utf-8") as f:
                vision_prompt = f.read()
        except Exception as e:
            panic(str(e))

    # Process examples directory if provided
    examples = []
    total_examples_payload_size = 0
    if examples_dir is not None:
        if verbose:
            console.log(f"Loading examples from directory: [yellow]{examples_dir}[/yellow]")
        total_examples_payload_size, examples = load_examples_from_dir(examples_dir, not no_downscale)
        if not examples:
            panic("No valid examples found in the specified directory.")

    if verbose:
        print("================================================")
        print("Model:\t\t[cyan]" + model_name + "[/cyan]")
        print("Base URL:\t[cyan]" + final_base_url + "[/cyan]")
        print("Temperature:\t[cyan]" + str(temperature) + "[/cyan]")
        print("Max tokens:\t[cyan]" + str(max_tokens) + "[/cyan]")
        print("Examples:\t\t[cyan]" + str(len(examples) // 2) + "[/cyan]")
        print("================================================")

    # Initialize OpenAI client
    client = OpenAI(api_key=final_api_key, base_url=final_base_url)

    try:
        for file_path in file_paths:
            this_caption_output = caption_output
            filename_info_line = ""
            if this_caption_output is None:
                filename_info_line = (
                    "[white on blue]>>[/white on blue] [yellow]" + file_path + "[/yellow] => [yellow]*." + caption_extension
                )
                this_caption_output = os.path.splitext(file_path)[0] + "." + caption_extension
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

            caption_file = open(this_caption_output, "w", encoding="utf-8")

            try:
                image_base64 = get_image_base64(file_path, not no_downscale)
            except Exception as e:
                console.log(f"[red]Error processing image {file_path}: {e}[/red]")
                caption_file.close()
                continue

            if verbose:
                console.log(
                    "Payload size: [bright_yellow]"
                    + humanize.naturalsize(len(image_base64) + total_examples_payload_size, binary=True)
                    + "[/bright_yellow]"
                )
                console.log("Caption:")

                caption_response = get_caption_for_image(
                    client,
                    model_name,
                    vision_prompt,
                    image_base64,
                    examples,
                    temperature,
                    max_tokens,
                )
                padded_response = Padding("[green]" + caption_response + "[/green]", (0, 0, 0, 4))
                if verbose:
                    console.log(padded_response)
                else:
                    print(filename_info_line)
                    print(padded_response)

                caption_response += "\n"
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
        console.log(f"[red]Error: {e}[/red]")

    sys.exit(0)


if __name__ == "__main__":
    __main__()
