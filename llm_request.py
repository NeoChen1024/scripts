#!/usr/bin/env python
# Wrapper for OpenAI API Client

from email.mime import base
import os
from typing import Callable, Optional, Tuple, List, overload
import base64
from io import BytesIO

from PIL import Image
import click
from openai import OpenAI
from pydantic import BaseModel


def init_llm_api(
    key: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
) -> Tuple[OpenAI, str]:
    # Check if the API key is set
    api_key = "xxxx"  # placeholder, it's fine for it to be any string for local LLM
    if (env_key := os.environ.get("LLM_API_KEY")) is not None:
        api_key = env_key
    # Override if provided by command line argument
    if key is not None:
        api_key = key

    # Get model name from environment variable if it exists
    model_name = "mistral-large-latest"  # default model from Mistral
    if (env_model := os.environ.get("LLM_MODEL_NAME")) is not None:
        model_name = env_model
    # Override if provided by command line argument
    if model is not None:
        model_name = model

    use_base_url = "https://api.mistral.ai/v1"
    if (env_url := os.environ.get("LLM_BASE_URL")) is not None:
        use_base_url = env_url
    # Override if provided by command line argument
    if base_url is not None:
        use_base_url = base_url

    # Initialize OpenAI client
    client = OpenAI(api_key=api_key, base_url=use_base_url)

    return client, model_name


def image_to_jpeg_base64(
    input_image: Image, downscale: bool = True, size: tuple[int, int] = (2048, 2048)
) -> str:
    image = input_image.copy()
    if image.mode != "RGB":
        image = image.convert("RGB")
    image_jpeg = BytesIO()

    if downscale:
        image.thumbnail(size)

    image.save(image_jpeg, format="JPEG", quality=90)
    del image
    return base64.b64encode(image_jpeg.getvalue()).decode("utf-8")


# for text-only completions
# NOTE: modifies the messages list in place
@overload
def add_user_message(messages: List[dict], message_input: str) -> List[dict]:
    messages.append({"role": "user", "content": message_input})
    return messages


@overload
def add_user_message(
    messages: List[dict], message_input: str, image: Image, **kwargs
) -> List[dict]:
    new_message = {
        "role": "user",
        "content": [
            {"type": "text", "text": message_input},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_to_jpeg_base64(image, **kwargs)}"
                },
            },
        ],
    }
    messages.append(new_message)
    return messages


# NOTE: Some APIs (like Gemini) doesn't like image-only messages
@overload
def add_user_message(messages: List[dict], image: Image, **kwargs) -> List[dict]:
    new_message = {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_to_jpeg_base64(image, **kwargs)}"
                },
            },
        ],
    }
    messages.append(new_message)
    return messages


def add_system_message(messages: List[dict], message_input: str) -> List[dict]:
    messages.append({"role": "system", "content": message_input})
    return messages


# NOTE: modifies the messages list in place
def _append_assistant(messages: List[dict], response: str) -> List[dict]:
    messages.append({"role": "assistant", "content": response})
    return messages


# TODO: consider using overload to separate the return types
def llm_query(
    client: OpenAI,
    model_name: str,
    text: Optional[str] = None,
    system_prompt: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    history: Optional[List[dict]] = None,
    format: Optional[BaseModel] = None,
) -> Tuple[str | BaseModel, list]:
    messages = []
    if history is not None:
        messages = history
    # Add system prompt
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})
    if text is not None:
        messages.append({"role": "user", "content": text})

    kwargs = {}
    if (max_tokens is not None) and (format is None):
        kwargs["max_tokens"] = max_tokens
    if temperature is not None:
        kwargs["temperature"] = temperature

    if format is None:
        chat_response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            **kwargs,
        )
        response_text = chat_response.choices[0].message.content
        return response_text, _append_assistant(messages, response_text)
    else:
        # no max_tokens, because we want to get all the messages
        completion = client.beta.chat.completions.parse(
            model=model_name,
            messages=messages,
            response_format=format,
            **kwargs,
        )
        response = completion.choices[0].message.parsed
        response_json = response.model_dump_json()
        return response, _append_assistant(messages, response_json)


# TODO: introduce interactive mode


@click.command()
@click.option(
    "-k",
    "--api_key",
    help="Override LLM API key (default: content of LLM_API_KEY environment variable)",
)
@click.option(
    "-m",
    "--model",
    help="Model to use for translation (default: mistral-large-latest if LLM_MODEL_NAME environment variable is not set)",
)
@click.option(
    "-u",
    "--base_url",
    help="Override LLM base URL (default: Mistral's)",
)
@click.option(
    "-mt",
    "--max_tokens",
    type=int,
    show_default=True,
    help="Maximum number of tokens to generate in the response",
)
@click.option(
    "-t",
    "--temperature",
    type=float,
    show_default=True,
    help="Override temperature for response generation",
)
@click.option("-i", "--interactive", is_flag=True, help="Run in interactive mode")
@click.argument(
    "text",
    nargs=1,
    required=False,
)
def _main(api_key, model, base_url, max_tokens, temperature, interactive, text) -> None:
    client, model_name = init_llm_api(api_key, model, base_url)
    click.echo(f"Using API {base_url} with model {model_name}")

    if not interactive and text is not None:
        response, history = llm_query(
            client, model_name, text, temperature=temperature, max_tokens=max_tokens
        )
        click.echo(response)

    if interactive:
        click.echo("Running in interactive mode...")
        click.echo("Type '/exit' or '/quit' to quit")
        click.echo("Type '/system to add a system message")
        click.echo("Type '/image to add an image message")
        click.echo("Type '/history to view past messages")
        click.echo("Type '/reset to clear the chat history")

        history = []
        while True:
            user_input = click.prompt("> ", prompt_suffix="")
            if user_input == "/exit" or user_input == "/quit":
                break
            elif user_input == "/reset":
                history = []
                click.echo("Chat history cleared")
            elif user_input == "/system":
                system_message = click.prompt("Enter system message: ")
                history.append({"role": "system", "content": system_message})
                click.echo("System message added")
            elif user_input == "/image":
                image_path = click.prompt("Enter image path: ")
                user_input = click.prompt("Enter optional message: ")
                image = Image.open(image_path)
                if user_input:
                    history = add_user_message(history, str(user_input), image)
                else:
                    history = add_user_message(history, image)
                click.echo("Image message added")
            else:
                response, history = llm_query(
                    client,
                    model_name,
                    user_input,
                    history=history,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                click.echo(response)

    return

if __name__ == "__main__":
    _main()
