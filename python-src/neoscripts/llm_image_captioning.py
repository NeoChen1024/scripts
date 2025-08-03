#!/usr/bin/env python
# Generate image captions using an OpenAI-compatible API.

import base64
import os
import sys
import threading
from io import BytesIO
from queue import Queue
from typing import Any, Dict, List, Optional, Tuple, Union

import click
import humanize
from openai import OpenAI
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)
from PIL import Image
from rich.console import Console
from rich.padding import Padding
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.traceback import install as install_traceback

console = Console()
install_traceback(show_locals=True)

print = console.print  # For convenience, use console.print instead of print


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
    examples_dir: str, vision_prompt: str, downscale: bool = True, size: tuple[int, int] = (2048, 2048)
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
                            {"type": "text", "text": vision_prompt},
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
    system_prompt: str,
    vision_prompt: str,
    image_base64: str,
    examples: Optional[List[Union[ChatCompletionUserMessageParam, ChatCompletionAssistantMessageParam]]] = None,
    api_args: Optional[Dict[str, Any]] = None,
) -> str:
    kwarg = {}
    if api_args is not None:
        kwarg.update(api_args)
    messages: List[Union[ChatCompletionUserMessageParam, ChatCompletionSystemMessageParam, ChatCompletionAssistantMessageParam]] = [
        ChatCompletionSystemMessageParam(
            role="system",
            content=system_prompt,
        )
    ]
    if examples is not None:
        messages.extend(examples)
    messages.append(
        ChatCompletionUserMessageParam(
            role="user",
            content=[
                {"type": "text", "text": vision_prompt},
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


def process_captions_from_queue(
    verbose: bool,
    caption_extension: str,
    existing_caption: str,
    total_examples_payload_size: int,
    no_downscale: bool,
    queue: Queue,
    client: OpenAI,
    model_name: str,
    system_prompt: str,
    vision_prompt: str,
    examples: Optional[List[Union[ChatCompletionUserMessageParam, ChatCompletionAssistantMessageParam]]] = None,
    api_args: Optional[Dict[str, Any]] = None,
):
    """
    Process captions from a queue using the OpenAI client.
    This function is designed to be run in a separate thread.
    """
    while True:
        item = queue.get()
        if item is None:  # None is used as a sentinel value to stop the thread
            break
        image_path: str = item
        try:
            caption_output_path = os.path.splitext(image_path)[0] + "." + caption_extension

            if os.path.exists(caption_output_path) and existing_caption == "skip":
                print("Caption file already exists, skipping...")
                continue

            caption_file = open(caption_output_path, "w", encoding="utf-8")

            try:
                image_base64 = get_image_base64(image_path, not no_downscale)
            except Exception as e:
                print(f"[red]Error processing image {image_path}: {e}[/red]")
                caption_file.close()
                continue

            file_info_line = (
                "[white on blue]>>[/white on blue] [yellow]"
                + image_path
                + "[/yellow] => [yellow]*."
                + caption_extension
                + "[/yellow], payload size: [bright_yellow]"
                + humanize.naturalsize(len(image_base64) + total_examples_payload_size, binary=True)
                + "[/bright_yellow]"
            )

            if api_args is None:
                api_args = {}
            caption_response = get_caption_for_image(
                client,
                model_name,
                system_prompt,
                vision_prompt,
                image_base64,
                examples,
                api_args
            )
            caption_response = caption_response.strip()
            # log file info + padded caption response
            print(file_info_line, Padding("[green]" + caption_response + "[/green]", (0, 0, 0, 4)))

            caption_response += "\n"
            caption_file.write(caption_response)
            caption_file.flush()
            if verbose:
                print(
                    "Caption saved to [yellow]"
                    + caption_output_path
                    + "[/yellow] ([bright_yellow]"
                    + humanize.naturalsize(caption_file.tell(), binary=True)
                    + "[/bright_yellow])"
                    + "\n\n"
                )
            caption_file.close()
            queue.task_done()
        except Exception as e:
            print(f"[red]Error processing item from queue: {e}[/red]")
            queue.task_done()


def panic(e: str):
    print("[red]Error:\t" + e + "[/red]")
    sys.exit(1)


def scan_files(paths: List[str], caption_extension: str) -> List[str]:
    result = []
    for path in paths:
        if os.path.isdir(path):
            for root, _, files in os.walk(path):
                for file in files:
                    # try checking if the file is an image
                    try:
                        if os.path.exists(os.path.join(root, file)) and not file.lower().endswith(("." + caption_extension)):
                            # check if the file is a valid image
                            img = Image.open(os.path.join(root, file))
                            img.verify()
                            result.append(os.path.join(root, file))
                    except Exception:
                        continue
        elif os.path.isfile(path):
            result.append(path)
    return result


default_system_prompt = """\
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

default_vision_prompt = "Generate the caption for the following image."


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
    "-sp",
    "--system-prompt",
    default=None,
    help="System prompt or path to the file for caption generation (default: built-in).",
)
@click.option(
    "-vp",
    "--vision-prompt",
    default=None,
    help="Vision prompt or path to the file for caption generation (default: built-in).",
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
    "-a",
    "--api-args",
    default=None,
    help="Additional API arguments as a JSON string or file path (default: None).",
    type=click.STRING,
    show_default=True,
)
@click.option("-v", "--verbose", is_flag=True, default=False, help="Enable verbose output.")
@click.option(
    "-ex",
    "--existing-caption",
    default="overwrite",
    type=click.Choice(["overwrite", "skip"]),
    help='"overwrite" or "skip" existing caption file.',
    show_default=True,
)
@click.option(
    "-ce",
    "--caption-extension",
    default="txt",
    help="Caption file extension",
    show_default=True,
)
@click.option(
    "--no-downscale",
    "-nd",
    is_flag=True,
    help="Disable image downscaling before uploading.",
)
@click.option(
    "-t",
    "--threads",
    default=1,
    type=int,
    help="Number of threads to use for calling VLM API.",
    show_default=True,
)
def __main__(
    file_paths,
    caption_output,
    system_prompt,
    vision_prompt,
    model,
    api_key,
    base_url,
    api_args,
    verbose,
    existing_caption,
    caption_extension,
    no_downscale,
    examples_dir,
    threads,
):
    """
    Generate image captions for one or more image files.
    """
    # Scan the file paths
    if not file_paths:
        console.log("[red]Error: No input files provided.[/red]")
        sys.exit(1)
    # recursively resolve file paths
    file_paths = scan_files(file_paths, caption_extension)
    print(f"[bright_yellow]INFO:[/bright_yellow] Found {len(file_paths)} image(s) to process.")
    if not file_paths:
        console.log("[red]Error: No valid image files found.[/red]")
        sys.exit(1)
    # Validate arguments
    if len(file_paths) > 1 and caption_output is not None:
        print("Error: Can't use --caption_output with multiple input images")
        sys.exit(1)

    if caption_output is not None and not caption_output.endswith("." + caption_extension):
        print("[bright_yellow]INFO:[/bright_yellow] Caption extension will be ignored if --caption_output is provided")

    # API key resolution
    final_api_key = _fallback_environment("xxxx", "LLM_API_KEY", api_key)

    # Model resolution
    model_name = _fallback_environment("pixtral-large-latest", "LLM_MODEL_NAME", model)

    # Base URL resolution
    final_base_url = _fallback_environment("https://api.mistral.ai/v1", "LLM_BASE_URL", base_url)

    if system_prompt is not None:
        try:
            with open(system_prompt, "r", encoding="utf-8") as f:
                system_prompt = f.read()
        except Exception as e:
            pass
    else:
        system_prompt = default_system_prompt
    if vision_prompt is not None:
        try:
            with open(vision_prompt, "r", encoding="utf-8") as f:
                vision_prompt = f.read()
        except Exception as e:
            pass
    else:
        vision_prompt = default_vision_prompt
    
    final_api_args: Optional[Dict[str, Any]] = None
    if api_args is not None:
        try:
            if os.path.exists(api_args):
                with open(api_args, "r", encoding="utf-8") as f:
                    api_args = f.read()
            final_api_args = eval(api_args)  # Convert JSON string to dict
            if not isinstance(final_api_args, dict):
                raise ValueError("API arguments must be a JSON object.")
        except Exception as e:
            panic(f"Failed to parse API arguments: {e}")
    else:
        final_api_args = {}

    # Process examples directory if provided
    examples = []
    total_examples_payload_size = 0
    if examples_dir is not None:
        if verbose:
            console.log(f"Loading examples from directory: [yellow]{examples_dir}[/yellow]")
        total_examples_payload_size, examples = load_examples_from_dir(examples_dir, vision_prompt, not no_downscale)
        if not examples:
            panic("No valid examples found in the specified directory.")

    if verbose:
        print("================================================")
        print("Base URL:\t[cyan]" + final_base_url + "[/cyan]")
        print("Model:\t\t[cyan]" + model_name + "[/cyan]")
        if final_api_args is not None and isinstance(final_api_args, dict):
            print("API arguments:\t[cyan]" + str(final_api_args) + "[/cyan]")
        print(
            "Examples:\t[cyan]"
            + str(len(examples) // 2)
            + "[/cyan]"
            + ", size: [cyan]"
            + humanize.naturalsize(total_examples_payload_size, binary=True)
            + "[/cyan]"
        )
        print("================================================")
        print("System prompt:\n")
        print(Padding("[green]" + system_prompt + "[/green]", (0, 0, 0, 4)))
        print("\nVision prompt:\n")
        print(Padding("[green]" + vision_prompt + "[/green]", (0, 0, 0, 4)))
        print("================================================")

    # Initialize OpenAI client
    client = OpenAI(api_key=final_api_key, base_url=final_base_url)

    if len(file_paths) == 1:
        # single image
        image_path = file_paths[0]
        caption_output_path = caption_output
        filename_info_line = ""
        if caption_output_path is None:
            filename_info_line = (
                "[white on blue]>>[/white on blue] [yellow]" + image_path + "[/yellow] => [yellow]*." + caption_extension
            )
            caption_output_path = os.path.splitext(image_path)[0] + "." + caption_extension
        else:
            filename_info_line = (
                "[white on blue]>>[/white on blue] [yellow]"
                + image_path
                + "[/yellow] => [yellow]"
                + caption_output_path
                + "[/yellow]"
            )
        print(filename_info_line)
        caption_file = open(caption_output_path, "w", encoding="utf-8")

        image_base64 = get_image_base64(image_path, not no_downscale)

        if verbose:
            print(
                "Payload size: [bright_yellow]"
                + humanize.naturalsize(len(image_base64) + total_examples_payload_size, binary=True)
                + "[/bright_yellow]"
            )
            print("Caption:")

        caption_response = get_caption_for_image(
            client,
            model_name,
            system_prompt,
            vision_prompt,
            image_base64,
            examples,
            final_api_args
        )
        caption_response = caption_response.strip()
        # log padded caption response
        print(Padding("[green]" + caption_response + "[/green]", (0, 0, 0, 4)))

        caption_response += "\n"
        caption_file.write(caption_response)
        caption_file.flush()
        if verbose:
            print(
                "Caption saved to [yellow]"
                + caption_output_path
                + "[/yellow] ([bright_yellow]"
                + humanize.naturalsize(caption_file.tell(), binary=True)
                + "[/bright_yellow])"
                + "\n\n"
            )
        sys.exit(0)

    max_threads = min(threads, len(file_paths))
    image_queue = Queue(maxsize=max_threads)  # stores the image paths to process
    # Start worker threads
    workers = []
    for _ in range(max_threads):
        worker = threading.Thread(
            target=process_captions_from_queue,
            args=(
                verbose,
                caption_extension,
                existing_caption,
                total_examples_payload_size,
                no_downscale,
                image_queue,
                client,
                model_name,
                system_prompt,
                vision_prompt,
                examples,
                api_args,
            ),
        )
        worker.daemon = True
        worker.start()
        workers.append(worker)

    try:
        with Progress(
            SpinnerColumn(),
            MofNCompleteColumn(),
            BarColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            "{task.description}",
            console=console,
        ) as progress:
            task = progress.add_task("captioning", total=len(file_paths))
            for file_path in file_paths:
                image_queue.put(file_path)
                progress.update(task, advance=1, description=f"[bright_yellow]Queued: {os.path.basename(file_path)} [/bright_yellow]")

            progress.update(task, completed=len(file_paths), description="[green]All images queued![/green]")
        # wait for all tasks to be done
        print("[bright_yellow]INFO:[/bright_yellow] All images queued, waiting for processing to finish...")
        image_queue.join()
        print("[bright_yellow]INFO:[/bright_yellow] All images processed.")
        for _ in range(max_threads):
            image_queue.put(None)

    except KeyboardInterrupt:
        print("[red]Interrupted by user, draining queue...[/red]")
        while True:
            try:
                image_queue.get(timeout=0.1)
            except Exception:
                break
        print("[red]Queue drained, stopping workers...[/red]")
        for _ in range(max_threads):
            image_queue.put(None)
        for worker in workers:
            worker.join()
        print("[red]Processing stopped.[/red]")
        sys.exit(1)

    except Exception as e:
        console.log(f"[red]Error: {e}[/red]")

    sys.exit(0)


if __name__ == "__main__":
    __main__()
