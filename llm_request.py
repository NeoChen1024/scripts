#!/usr/bin/env python
# Wrapper for OpenAI API Client

import base64
import os
from io import BytesIO
from typing import List, Optional
from collections import UserList

import click
from openai import OpenAI
from PIL import Image
from pydantic import BaseModel
from rich import print
from rich.console import Console
from rich.pretty import pprint

console = Console()


def _fallback_parameters(*args) -> any:
    for arg in args:
        if arg is not None:
            return arg
    return None


class LLM_Config:
    client: OpenAI
    model_name: str
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None

    def __init__(
        self,
        client: OpenAI,
        model_name: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> None:
        self.client = client
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens


def init_llm_api(
    key: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> LLM_Config:
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
    return LLM_Config(
        client=client,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
    )


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


# For DeepSeek R1-like models
# Their responses are in the format of:
# <thinking>
# Thinking process..
# </thinking>
# Actual response
def strip_thinking(response: str, delimiter: Optional[str] = "</thinking>") -> str:
    if delimiter in response:
        response = response.split(delimiter)[1]
        # strip newlines and spaces
        response = response.strip()
        return response
    else:
        return response


class LLM_History(UserList):
    data: List[dict]

    def __init__(self, history: Optional[List[dict]] = None) -> None:
        if history is not None:
            self.data = history
        else:
            self.data = []

    def __del__(self):
        del self.data

    def del_last_message(self) -> None:
        if self.data:
            self.data.pop()

    def get_last_message(self) -> dict:
        if self.data:
            return self.data[-1]
        else:
            return {}

    def add_system(self, message_input: str) -> None:
        self.data.append({"role": "system", "content": message_input})

    def add_assistant(
        self, response: str, _strip_thinking: Optional[bool] = False
    ) -> None:
        if _strip_thinking:
            response = strip_thinking(response)
        self.data.append({"role": "assistant", "content": response})

    def add_user(
        self,
        message_input: Optional[str] = None,
        image: Optional[Image.Image] = None,
        **kwargs,
    ) -> List[dict]:
        content_list = []

        if message_input is not None:
            content_list.append({"type": "text", "text": message_input})

        if image is not None:
            content_list.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_to_jpeg_base64(image, **kwargs)}"
                    },
                }
            )

        if content_list:
            new_message = {
                "role": "user",
                "content": content_list,
            }
            self.data.append(new_message)


# TODO: consider using overload to separate the return types
def llm_query(
    llm_config: LLM_Config,
    text: Optional[str] = None,
    system_prompt: Optional[str] = None,
    history: Optional[LLM_History] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    format: Optional[BaseModel] = None,
    refusal_is_error: Optional[bool] = False,
) -> str | BaseModel:
    client = llm_config.client
    model_name = llm_config.model_name
    temperature = _fallback_parameters(temperature, llm_config.temperature)
    max_tokens = _fallback_parameters(max_tokens, llm_config.max_tokens)

    messages = []
    if history is not None:
        messages = history.data
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
        return response_text
    else:
        # no max_tokens, because we want to get all the messages
        completion = client.beta.chat.completions.parse(
            model=model_name,
            messages=messages,
            response_format=format,
            **kwargs,
        )
        # if refused, return the error message
        message = completion.choices[0].message
        refusal = message.refusal
        if refusal is not None:
            if refusal_is_error:
                raise ValueError("Refusal occurred: " + refusal)
            console.log("Refusal occurred: " + refusal)
            console.log("History: ", history)
            return refusal
        return message.parsed


# Client implementation for reference
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
@click.option(
    "--strip-thinking",
    "-s",
    is_flag=True,
    help="Strip the thinking process from the history (for DeepSeek R1-like models)",
)
def _main(
    api_key, model, base_url, max_tokens, temperature, interactive, text, strip_thinking
) -> None:
    print = console.print  # override print function
    prompt = console.input  # override input function

    llm_config = init_llm_api(api_key, model, base_url, temperature, max_tokens)
    actual_url = llm_config.client.base_url
    print(f"Using API {actual_url} with model {llm_config.model_name}")

    if not interactive and text is not None:
        response = llm_query(llm_config, text)
        print(response)

    if interactive:
        print("Running in interactive mode...")
        print("Type '/exit' or '/quit' to quit")
        print("Type '/system' to add a system message")
        print("Type '/image' to add an image message")
        print("Type '/history' to view past messages")
        print("Type '/regen' to regenerate the last message")
        print("Type '/reset' to clear the chat history")

        history = LLM_History()
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
                elif user_input == "/regen":
                    history.del_last_message()
                    user_input = None
                    # There's no continue here!! We want to regenerate the last message
                elif user_input == "/image":
                    try:
                        image_path = prompt("Enter image path: ")
                        user_input = prompt("Enter optional message: ")
                        image = Image.open(str(image_path))
                        history.add_user(history, str(user_input), image)
                        print("Image message added")
                        user_input = (
                            None  # ugly hack to prevent the text from being sent twice
                        )
                    # There's no continue here because we want to send the image / text now
                    except Exception as e:
                        print(f"Error: {e}")
                        continue  # continue to the command line

                with console.status(
                    "[bold green]Waiting for response...", spinner="line"
                ):
                    response = llm_query(
                        llm_config,
                        user_input,
                        history=history,
                    )
                history.add_assistant(response, strip_thinking)
                if strip_thinking:
                    # colorize the thinking process in grey
                    if "</think>" in response:
                        response = response.replace(
                            "</think>", "[/bright_black][green]"
                        )
                        if "<think>" in response:
                            response = response.replace("<think>", "[bright_black]")
                        else:
                            response = "[bright_black]" + response

                print(response)
            except Exception as e:
                print(f"Error: {e}")
                continue
    return


if __name__ == "__main__":
    _main()
