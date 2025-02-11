#!/usr/bin/env python
# Wrapper for OpenAI API Client

from email.mime import base
import os
from typing import Optional, Tuple, List
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


# NOTE: modifies the messages list in place
def add_user_message(
    messages: List[dict], message_input: Optional[str] = None, image: Optional[Image.Image] = None, **kwargs
) -> List[dict]:
    content_list = []
    
    if message_input is not None:
        content_list.append({"type": "text", "text": message_input})
    
    if image is not None:
        content_list.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{image_to_jpeg_base64(image, **kwargs)}"
            },
        })
    new_message = {
        "role": "user",
        "content": content_list,
    }
    messages.append(new_message)
    return messages


# NOTE: modifies the messages list in place
def del_last_message(messages: List[dict]) -> List[dict]:
    if messages:
        messages.pop()
    return messages


# NOTE: modifies the messages list in place
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
        refusal = chat_response.choices[0].message.refusal
        if refusal is not None:
            return refusal, history
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
        # if refused, return the error message
        refusal = completion.choices[0].message.get("refusal")
        if refusal is not None:
            return refusal, history

        response = completion.choices[0].message.parsed
        response_json = response.model_dump_json()
        return response, _append_assistant(messages, response_json)


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
    # Import rich for better console output
    from rich.console import Console
    from rich.pretty import pprint
    console = Console()
    print = console.print  # override print function
    prompt = console.input  # override input function

    client, model_name = init_llm_api(api_key, model, base_url)
    actual_url = client.base_url
    print(f"Using API {actual_url} with model {model_name}")

    if not interactive and text is not None:
        response, history = llm_query(
            client, model_name, text, temperature=temperature, max_tokens=max_tokens
        )
        print(response)

    if interactive:
        print("Running in interactive mode...")
        print("Type '/exit' or '/quit' to quit")
        print("Type '/system to add a system message")
        print("Type '/image to add an image message")
        print("Type '/history to view past messages")
        print("Type '/reset to clear the chat history")

        history = []
        while True:
            try:
                user_input = prompt("[white on blue]>>[/] ")
                if user_input == "/exit" or user_input == "/quit":
                    break
                elif user_input == "/reset":
                    history = []
                    print("Chat history cleared")
                    continue
                elif user_input == "/system":
                    system_message = prompt("Enter system message: ")
                    history.append({"role": "system", "content": system_message})
                    print("System message added")
                    continue
                elif user_input == "/history":
                    pprint(history, max_string=256)  # pretty print the history
                    continue
                elif user_input == "/image":
                    image_path = prompt("Enter image path: ")
                    user_input = prompt("Enter optional message: ")
                    image = Image.open(image_path)
                    history = add_user_message(history, str(user_input), image)
                    print("Image message added")
                    user_input = None # ugly hack to prevent the text from being sent twice
                    # There's no continue here because we want to send the image / text now

                with console.status("[bold green]Waiting for response...", spinner="line"):
                    response, history = llm_query(
                        client,
                        model_name,
                        user_input,
                        history=history,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                print(response, style="green")
            except Exception as e:
                print(f"Error: {e}")
                pass
    return


if __name__ == "__main__":
    _main()
