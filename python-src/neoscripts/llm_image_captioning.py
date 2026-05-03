#!/usr/bin/env python
# Generate image captions using an OpenAI-compatible API.

import base64
import json
import os
import sys
import threading
from dataclasses import dataclass
from io import BytesIO
from queue import Queue
from typing import Any, Dict, List, Literal, Optional, Tuple, Union, overload

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
install_traceback(show_locals=True, console=console)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

ExampleList = List[Union[ChatCompletionUserMessageParam, ChatCompletionAssistantMessageParam]]

MessageListType = List[
    Union[
        ChatCompletionUserMessageParam,
        ChatCompletionSystemMessageParam,
        ChatCompletionAssistantMessageParam,
    ]
]


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------


@dataclass
class CaptionConfig:
    """Immutable-ish configuration bundle for caption generation."""

    verbose: bool
    caption_extension: str
    total_examples_payload_size: int
    downscale: bool
    lossless: bool
    resolution: Tuple[int, int]
    client: OpenAI
    model_name: str
    system_prompt: Optional[str]
    vision_prompt: str
    examples: Optional[ExampleList]
    api_args: Optional[Dict[str, Any]]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _fallback_environment(default: str, env_var: str, override: Optional[str] = None) -> str:
    """Resolve a setting with priority: override > env-var > default."""
    val = default
    if (env_val := os.environ.get(env_var)) is not None:
        val = env_val
    if override is not None:
        val = override
    return val


def get_image_base64(
    image_or_path: Union[str, Image.Image],
    downscale: bool = True,
    lossless: bool = False,
    size: Tuple[int, int] = (2048, 2048),
) -> str:
    """Convert an image (PIL.Image or file path) to a base64-encoded data URL."""
    img = Image.open(image_or_path) if isinstance(image_or_path, str) else image_or_path
    img = img.convert("RGB")
    if downscale:
        img.thumbnail(size, Image.Resampling.LANCZOS)

    with BytesIO() as output:
        if lossless:
            img.save(output, format="PNG", optimize=True)
            prefix = "data:image/png;base64,"
        else:
            img.save(output, format="JPEG", quality=90, optimize=True)
            prefix = "data:image/jpeg;base64,"
        base64_str = base64.b64encode(output.getvalue()).decode("utf-8")

    return f"{prefix}{base64_str}"


def load_examples_from_dir(
    examples_dir: str,
    vision_prompt: str,
    downscale: bool = True,
    lossless: bool = False,
    size: Tuple[int, int] = (2048, 2048),
) -> Tuple[int, ExampleList]:
    """Load few-shot image+caption examples from a directory.

    Returns:
        (total_payload_size, examples_list)
    """
    examples: List = []
    total_payload_size = 0

    for entry in os.listdir(examples_dir):
        if entry.lower().endswith(".txt"):
            continue
        try:
            console.print(f"Loading example: [yellow]{entry}[/yellow]")
            image_path = os.path.join(examples_dir, entry)
            image = Image.open(image_path)
            image.load()  # force-load to verify integrity

            caption_filename = os.path.splitext(entry)[0] + ".txt"
            caption_path = os.path.join(examples_dir, caption_filename)
            if not os.path.exists(caption_path):
                console.log(f"[red]Warning:[/red] Caption file {caption_filename} not found for example {entry}. Skipping.")
                continue

            with open(caption_path, "r", encoding="utf-8") as f:
                caption = f.read().strip()
            if not caption:
                console.log(f"[red]Warning:[/red] Caption file {caption_filename} is empty for example {entry}. Skipping.")
                continue

            image_base64 = get_image_base64(image, downscale, lossless, size)
            total_payload_size += len(image_base64)
            examples.extend(
                [
                    ChatCompletionUserMessageParam(
                        role="user",
                        content=[
                            {"type": "text", "text": vision_prompt},
                            {"type": "image_url", "image_url": {"url": image_base64}},
                        ],
                    ),
                    ChatCompletionAssistantMessageParam(
                        role="assistant",
                        content=caption,
                    ),
                ]
            )
        except (IOError, SyntaxError):
            continue

    return total_payload_size, examples


def get_caption_for_image(
    client: OpenAI,
    model_name: str,
    system_prompt: Union[str, None],
    vision_prompt: str,
    image_base64: str,
    examples: Optional[ExampleList] = None,
    api_args: Optional[Dict[str, Any]] = None,
) -> Tuple[Tuple[int, int], str]:
    """Call the VLM API and return (prompt_tokens, completion_tokens) and caption text."""
    extra_kwargs: Dict[str, Any] = {}
    if isinstance(api_args, dict):
        extra_kwargs.update(api_args)

    messages: MessageListType = []
    if system_prompt is not None:
        messages.append(ChatCompletionSystemMessageParam(role="system", content=system_prompt))
    if examples is not None:
        messages.extend(examples)
    messages.append(
        ChatCompletionUserMessageParam(
            role="user",
            content=[
                {"type": "text", "text": vision_prompt},
                {"type": "image_url", "image_url": {"url": image_base64}},
            ],
        )
    )

    chat_response = client.chat.completions.create(
        model=model_name,
        stream=False,
        messages=messages,
        **extra_kwargs,
    )

    generated_caption = chat_response.choices[0].message.content
    if generated_caption is None:
        raise ValueError("Caption generation failed, no content returned.")
    generated_caption = generated_caption.strip() + "\n"

    if chat_response.usage:
        token_count = (
            chat_response.usage.prompt_tokens,
            chat_response.usage.completion_tokens,
        )
    else:
        token_count = (0, 0)

    return token_count, generated_caption


def get_caption_output_path(image_path: str, caption_extension: str) -> str:
    """Derive the caption file path from the image path."""
    filename, _ = os.path.splitext(image_path)
    return f"{filename}.{caption_extension}"


@overload
def resolve_prompt(
    prompt_input: Optional[str],
    default: str,
    allow_none: Literal[False] = False,
) -> str: ...


@overload
def resolve_prompt(
    prompt_input: Optional[str],
    default: str,
    allow_none: Literal[True],
) -> Optional[str]: ...


def resolve_prompt(
    prompt_input: Optional[str],
    default: str,
    allow_none: bool = False,
) -> str | Optional[str]:
    """Resolve a prompt from file path, literal string, or fallback default.

    - ``None``               → *default*
    - ``"NONE"`` (if allowed)→ ``None``
    - valid file path        → file contents
    - anything else          → literal string
    """
    if prompt_input is None:
        return default
    if allow_none and prompt_input == "NONE":
        return None
    try:
        with open(prompt_input, "r", encoding="utf-8") as f:
            return f.read()
    except (OSError, IOError):
        return prompt_input  # treat as a literal prompt string


def resolve_api_args(api_args_input: Optional[str]) -> Dict[str, Any]:
    """Parse API arguments from a JSON string or a JSON file path."""
    if api_args_input is None:
        return {}
    try:
        if os.path.exists(api_args_input):
            with open(api_args_input, "r", encoding="utf-8") as f:
                api_args_input = f.read()
        result = json.loads(api_args_input, parse_int=int, parse_float=float, parse_constant=str)
        if not isinstance(result, dict):
            raise ValueError("API arguments must be a JSON object.")
        return result
    except Exception as e:
        panic(f"Failed to parse API arguments: {e}")
        return {}  # unreachable, but keeps type-checkers happy


def print_verbose_config(
    base_url: str,
    model_name: str,
    api_args: Dict[str, Any],
    examples: ExampleList,
    total_examples_payload_size: int,
    system_prompt: Optional[str],
    vision_prompt: str,
) -> None:
    """Print the current configuration in verbose mode."""
    console.print("=" * 48)
    console.print(f"Base URL:\t[cyan]{base_url}[/cyan]")
    console.print(f"Model:\t\t[cyan]{model_name}[/cyan]")
    if api_args:
        console.print(f"API arguments:\t[cyan]{api_args}[/cyan]")
    console.print(
        f"Examples:\t[cyan]{len(examples) // 2}[/cyan]"
        f", size: [cyan]{humanize.naturalsize(total_examples_payload_size, binary=True)}[/cyan]"
    )
    console.print("=" * 48)
    console.print("System prompt:\n")
    console.print(Padding(f"[green]{system_prompt}[/green]", (0, 0, 0, 4)))
    console.print("\nVision prompt:\n")
    console.print(Padding(f"[green]{vision_prompt}[/green]", (0, 0, 0, 4)))
    console.print("=" * 48)


def panic(message: str) -> None:
    """Print an error message and exit."""
    console.log(f"[red]Error:\t{message}[/red]")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------


def process_single_image(
    config: CaptionConfig,
    image_path: str,
    caption_output_path: Optional[str],
    existing_caption: str,
) -> None:
    """Process a single image and write its caption."""
    if caption_output_path is None:
        caption_output_path = get_caption_output_path(image_path, config.caption_extension)
        console.print(f"[white on blue]>>[/white on blue] [yellow]{image_path}[/yellow] => [yellow]*.{config.caption_extension}")
    else:
        console.print(f"[white on blue]>>[/white on blue] [yellow]{image_path}[/yellow] => [yellow]{caption_output_path}[/yellow]")

    if existing_caption == "skip" and os.path.exists(caption_output_path):
        console.print("[bright_yellow]INFO:[/bright_yellow] Caption file already exists, skipping...")
        sys.exit(0)

    image_base64 = get_image_base64(image_path, config.downscale, config.lossless, config.resolution)

    if config.verbose:
        payload_size = len(image_base64) + config.total_examples_payload_size
        console.print(f"Payload size: [bright_cyan]" f"{humanize.naturalsize(payload_size, binary=True)}[/bright_cyan]")
        console.print("Caption:")

    token_count, caption_response = get_caption_for_image(
        config.client,
        config.model_name,
        config.system_prompt,
        config.vision_prompt,
        image_base64,
        config.examples,
        config.api_args,
    )

    console.print(Padding(f"[green]{caption_response}[/green]", (0, 0, 0, 4)))

    with open(caption_output_path, "w", encoding="utf-8") as f:
        f.write(caption_response)

    if config.verbose:
        console.print(
            f"Caption saved to [yellow]{caption_output_path}[/yellow] "
            f"(Input [bright_yellow]{humanize.metric(token_count[0], 'T')}[/bright_yellow] "
            f"Output [bright_yellow]{humanize.metric(token_count[1], 'T')}[/bright_yellow])\n\n"
        )


def process_captions_from_queue(config: CaptionConfig, queue: Queue) -> None:
    """Worker thread: pull image paths from *queue* and generate captions."""
    while True:
        item = queue.get()
        if item is None:  # sentinel – stop the worker
            queue.task_done()
            break

        image_path: str = item
        try:
            caption_output_path = get_caption_output_path(image_path, config.caption_extension)
            image_base64 = get_image_base64(image_path, config.downscale, config.lossless, config.resolution)

            token_cost, caption_response = get_caption_for_image(
                config.client,
                config.model_name,
                config.system_prompt,
                config.vision_prompt,
                image_base64,
                config.examples,
                config.api_args,
            )

            if config.verbose:
                payload_size = len(image_base64) + config.total_examples_payload_size
                console.print(
                    f"[white on blue]>>[/white on blue] [yellow]{image_path}[/yellow] "
                    f"=> [yellow]*.{config.caption_extension}[/yellow], "
                    f"payload size: [bright_cyan]"
                    f"{humanize.naturalsize(payload_size, binary=True)}[/bright_cyan]"
                    f" (Input [bright_yellow]{humanize.metric(token_cost[0], 'T')}[/bright_yellow], "
                    f"Output [bright_yellow]{humanize.metric(token_cost[1], 'T')}[/bright_yellow])",
                    Padding(f"[green]{caption_response}[/green]", (0, 0, 0, 4)),
                )

            with open(caption_output_path, "w", encoding="utf-8") as f:
                f.write(caption_response)

            queue.task_done()

        except Exception as e:
            queue.task_done()
            console.log(f"[red]Error processing item from queue: {e}[/red]")


def scan_files(paths: List[str], caption_extension: str) -> List[str]:
    """Recursively scan directories for valid image files, excluding caption files."""
    result: List[str] = []
    for path in paths:
        if os.path.isdir(path):
            for root, _, files in os.walk(path):
                for file in files:
                    if file.lower().endswith(f".{caption_extension}"):
                        continue
                    file_path = os.path.join(root, file)
                    try:
                        img = Image.open(file_path)
                        img.verify()
                        result.append(file_path)
                    except Exception:
                        continue
        elif os.path.isfile(path):
            result.append(path)
    return result


# ---------------------------------------------------------------------------
# Default prompts
# ---------------------------------------------------------------------------

DEFAULT_SYSTEM_PROMPT = """\
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

DEFAULT_VISION_PROMPT = "Generate the caption for the following image."


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


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
    help='System prompt or path to the file for caption generation (default: built-in, "NONE" for none).',
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
    "-ll",
    "--lossless",
    is_flag=True,
    help="Use PNG compression for images (default: JPEG with quality=90).",
)
@click.option(
    "-res",
    "--resolution",
    default=(2048, 2048),
    type=(int, int),
    help="Downscale to resolution (width, height) before uploading.",
    show_default=True,
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
    lossless,
    resolution,
    examples_dir,
    threads,
):
    """Generate image captions for one or more image files."""
    # ---- Scan & validate input files ---------------------------------------
    if not file_paths:
        console.log("[red]Error: No input files provided.[/red]")
        sys.exit(1)

    file_paths = scan_files(file_paths, caption_extension)
    console.print(f"[bright_yellow]INFO:[/bright_yellow] Found {len(file_paths)} image(s) to process.")

    if not file_paths:
        console.log("[red]Error: No valid image files found.[/red]")
        sys.exit(1)

    if len(file_paths) > 1 and caption_output is not None:
        console.log("[red]Error: Can't use --caption_output with multiple input images[/red]")
        sys.exit(1)

    if caption_output is not None and not caption_output.endswith(f".{caption_extension}"):
        console.print("[bright_yellow]INFO:[/bright_yellow] " "Caption extension will be ignored if --caption_output is provided")

    # ---- Resolve API settings ----------------------------------------------
    final_api_key = _fallback_environment("xxxx", "LLM_API_KEY", api_key)
    model_name = _fallback_environment("pixtral-large-latest", "LLM_MODEL_NAME", model)
    final_base_url = _fallback_environment("https://api.mistral.ai/v1", "LLM_BASE_URL", base_url)

    # ---- Resolve prompts ---------------------------------------------------
    resolved_system_prompt = resolve_prompt(system_prompt, DEFAULT_SYSTEM_PROMPT, allow_none=True)
    resolved_vision_prompt = resolve_prompt(vision_prompt, DEFAULT_VISION_PROMPT)

    # ---- Resolve API extra arguments ---------------------------------------
    final_api_args = resolve_api_args(api_args)

    # ---- Load few-shot examples --------------------------------------------
    examples: List = []
    total_examples_payload_size = 0
    if examples_dir is not None:
        if verbose:
            console.log(f"Loading examples from directory: [yellow]{examples_dir}[/yellow]")
        total_examples_payload_size, examples = load_examples_from_dir(
            examples_dir,
            resolved_vision_prompt,
            not no_downscale,
            lossless,
            resolution,
        )
        if not examples:
            panic("No valid examples found in the specified directory.")

    # ---- Verbose output ----------------------------------------------------
    if verbose:
        print_verbose_config(
            final_base_url,
            model_name,
            final_api_args,
            examples,
            total_examples_payload_size,
            resolved_system_prompt,
            resolved_vision_prompt,
        )

    # ---- Build shared configuration ----------------------------------------
    client = OpenAI(api_key=final_api_key, base_url=final_base_url)
    config = CaptionConfig(
        verbose=verbose,
        caption_extension=caption_extension,
        total_examples_payload_size=total_examples_payload_size,
        downscale=not no_downscale,
        lossless=lossless,
        resolution=resolution,
        client=client,
        model_name=model_name,
        system_prompt=resolved_system_prompt,
        vision_prompt=resolved_vision_prompt,
        examples=examples if examples else None,
        api_args=final_api_args if final_api_args else None,
    )

    # ---- Single-image path -------------------------------------------------
    if len(file_paths) == 1:
        process_single_image(config, file_paths[0], caption_output, existing_caption)
        sys.exit(0)

    # ---- Multi-image (threaded) path ---------------------------------------
    max_threads = min(threads, len(file_paths))
    image_queue: Queue = Queue(maxsize=max_threads)

    workers = []
    for _ in range(max_threads):
        worker = threading.Thread(
            target=process_captions_from_queue,
            args=(config, image_queue),
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
                if existing_caption == "skip" and os.path.exists(get_caption_output_path(file_path, caption_extension)):
                    console.print(
                        f"[bright_yellow]INFO:[/bright_yellow] " f"Caption file already exists for {file_path}, skipping..."
                    )
                else:
                    image_queue.put(file_path)
                progress.update(
                    task,
                    advance=1,
                    description=f"[bright_yellow]Queued: {file_path} [/bright_yellow]",
                )

            progress.update(
                task,
                completed=len(file_paths),
                description="[green]All images queued![/green]",
            )

        console.print("[bright_yellow]INFO:[/bright_yellow] All images queued, waiting for processing to finish...")
        image_queue.join()
        console.print("[bright_yellow]INFO:[/bright_yellow] All images processed.")

        # Send stop signals and wait for workers
        for _ in range(max_threads):
            image_queue.put(None)
        for worker in workers:
            worker.join()
        console.print("[green]Processing stopped.[/green]")

    except KeyboardInterrupt:
        console.log("[red]Interrupted by user, draining queue...[/red]")
        drained_items = 0
        while True:
            try:
                image_queue.get(timeout=0.1)
                drained_items += 1
            except Exception:
                break
        console.log(f"[red]Queue drained ({drained_items} items), stopping workers...[/red]")
        for _ in range(max_threads):
            image_queue.put(None)
        for worker in workers:
            worker.join()
        console.log("[red]Processing stopped.[/red]")
        sys.exit(1)

    except Exception as e:
        console.log(f"[red]Error: {e}[/red]")

    sys.exit(0)


if __name__ == "__main__":
    __main__()
