#!/usr/bin/env python
"""Wrapper for OpenAI API Client.

Provides ``LLM_Config``, ``LLM_History``, and ``llm_query()`` to simplify
interactions with any OpenAI-compatible chat endpoint.

Usage (as a library)::

    from neoscripts.llm_request import LLM_Config, llm_query

    cfg = LLM_Config()
    resp = llm_query(cfg, text="Hello!")
    print(resp)

A minimal interactive / one-shot CLI is also available::

    python -m neoscripts.llm_request "Tell me a joke"
    python -m neoscripts.llm_request -i
"""

import base64
import os
from collections import UserList
from copy import deepcopy
from io import BytesIO
from typing import Any, List, Literal, Optional, TypeVar, cast, overload

import click
from openai import OpenAI
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionContentPartImageParam,
    ChatCompletionContentPartTextParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)
from PIL import Image
from pydantic import BaseModel
from rich import print
from rich.console import Console
from rich.pretty import pprint

console = Console()
T = TypeVar("T", bound=BaseModel)


# ---------------------------------------------------------------------------
# Helper: fallback resolution
# ---------------------------------------------------------------------------


def _fallback_environment(default: str, env_var: str, override: Optional[str] = None) -> str:
    """Resolve a configuration value with override > environment > default priority."""
    val = default
    if (env_val := os.environ.get(env_var)) is not None:
        val = env_val
    if override is not None:
        val = override
    return val


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class LLM_Config:
    """Holds connection parameters and defaults for an OpenAI-compatible API.

    Every field can be overridden at construction time, falling back to an
    environment variable, then to a hard-coded default.

    Parameters
    ----------
    api_key:
        API key.  Falls back to ``LLM_API_KEY`` env-var, then ``"xxxx"``.
    model_name:
        Model identifier.  Falls back to ``LLM_MODEL_NAME`` env-var, then
        ``"mistral-large-latest"``.
    base_url:
        API base URL.  Falls back to ``LLM_BASE_URL`` env-var, then
        ``"https://api.mistral.ai/v1"``.
    temperature:
        Default sampling temperature (``None`` = leave unset for the API).
    max_tokens:
        Default max tokens for *non-structured* queries (``None`` = unset).
    strip_thinking:
        Automatically remove the ``<thinking>``…``</thinking>`` block from
        responses?  Useful for DeepSeek R1-like models.
    thinking_delimiter:
        The delimiter used to locate the end of the thinking block.
        Default ``"</thinking>"``.
    **kwargs:
        Any extra keyword arguments are forwarded to every
        ``client.chat.completions.create`` / ``.parse()`` call (e.g.
        ``top_p``, ``stop``, …).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        strip_thinking: bool = False,
        thinking_delimiter: str = "</thinking>",
        **kwargs: Any,
    ) -> None:
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.model_name = _fallback_environment("mistral-large-latest", "LLM_MODEL_NAME", model_name)
        self.strip_thinking = strip_thinking
        self.thinking_delimiter = thinking_delimiter
        self.kwargs = kwargs

        resolved_key = _fallback_environment("xxxx", "LLM_API_KEY", api_key)
        resolved_url = _fallback_environment("https://api.mistral.ai/v1", "LLM_BASE_URL", base_url)
        self.client = OpenAI(api_key=resolved_key, base_url=resolved_url)


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------


def image_to_jpeg_base64(
    input_image: Image.Image,
    downscale: bool = True,
    size: tuple[int, int] = (2048, 2048),
) -> str:
    """Convert a PIL ``Image`` to a JPEG base64-encoded string.

    Parameters
    ----------
    input_image:
        The image to encode.  A copy is made so the original is untouched.
    downscale:
        When ``True`` (default), shrink the image to fit inside *size*
        while preserving aspect ratio (via ``Image.thumbnail``).
    size:
        Maximum (width, height) used when *downscale* is enabled.
    """
    image = input_image.copy()
    if image.mode != "RGB":
        image = image.convert("RGB")

    if downscale:
        image.thumbnail(size)

    with BytesIO() as output:
        image.save(output, format="JPEG", quality=90)
        return base64.b64encode(output.getvalue()).decode("utf-8")


# ---------------------------------------------------------------------------
# Thinking / reasoning stripping
# ---------------------------------------------------------------------------


def strip_thinking(response: str, delimiter: str = "</thinking>") -> str:
    """Remove the reasoning block from a model response.

    For DeepSeek R1-like models the output looks like::

        <thinking>…</thinking>
        Actual answer

    This function returns everything after *delimiter* (stripped), or the
    original response when the delimiter is not found.

    Parameters
    ----------
    response:
        Raw model response text.
    delimiter:
        The marker that signals the end of the thinking block.
    """
    if delimiter and delimiter in response:
        return response.split(delimiter, 1)[1].strip()
    return response


# ---------------------------------------------------------------------------
# Chat history
# ---------------------------------------------------------------------------


class LLM_History(UserList):
    """A mutable list of chat messages with convenience helpers.

    Each item is a ``ChatCompletionMessageParam`` (TypedDict), following the
    OpenAI message format.
    """

    def __init__(self, history: Optional[List[ChatCompletionMessageParam]] = None) -> None:
        super().__init__()
        if history is not None:
            self.data: list[ChatCompletionMessageParam] = list(history)
        else:
            self.data: list[ChatCompletionMessageParam] = []

    # -- message-level operations -------------------------------------------

    def del_last_message(self) -> None:
        """Remove the most recent message."""
        if self.data:
            self.data.pop()

    def get_last_message(self) -> Optional[ChatCompletionMessageParam]:
        """Return the most recent message, or ``None`` if history is empty."""
        if self.data:
            return self.data[-1]
        return None

    def del_last_messages(self, n: int = 1) -> None:
        """Remove the last *n* messages."""
        for _ in range(n):
            if self.data:
                self.data.pop()

    def del_first_n_non_system_messages(self, n: int = 1) -> None:
        """Delete the first *n* non-system messages and everything before them.

        .. note::

            Messages with ``role == "system"`` are preserved.
        """
        count = 0
        idx = 0
        while idx < len(self.data):
            if self.data[idx].get("role") != "system":
                count += 1
                if count >= n:
                    break
            idx += 1
        # Keep everything from after the first non-system message we want to keep
        # (or all messages if we never hit n non-system messages)
        self.data = self.data[idx + 1 :] if count >= n else []

    def del_all_system_messages(self) -> None:
        """Remove every system message from the history."""
        self.data = [msg for msg in self.data if msg.get("role") != "system"]

    # -- message builders ---------------------------------------------------

    def add_system(self, message_input: str) -> None:
        """Append a ``system`` message."""
        self.data.append(ChatCompletionSystemMessageParam(role="system", content=message_input))

    def add_assistant(self, response: str, strip_thinking_in_response: bool = False) -> None:
        """Append an ``assistant`` message.

        Parameters
        ----------
        response:
            The raw assistant response text.
        strip_thinking_in_response:
            When ``True``, the ``<thinking>`` / ``</thinking>`` blocks are
            removed before storing the message.
        """
        if strip_thinking_in_response:
            response = strip_thinking(response)
        self.data.append(ChatCompletionAssistantMessageParam(role="assistant", content=response))

    def add_user(
        self,
        message_input: Optional[str] = None,
        image: Optional[Image.Image] = None,
        **kwargs: Any,
    ) -> None:
        """Append a ``user`` message, optionally with an attached image.

        Parameters
        ----------
        message_input:
            Plain-text part of the message.
        image:
            A PIL ``Image`` to include (encoded as JPEG base64).
        **kwargs:
            Forwarded to :func:`image_to_jpeg_base64` when *image* is given.
        """
        content_parts: list[ChatCompletionContentPartTextParam | ChatCompletionContentPartImageParam] = []

        if message_input is not None:
            content_parts.append(ChatCompletionContentPartTextParam(type="text", text=message_input))

        if image is not None:
            content_parts.append(
                ChatCompletionContentPartImageParam(
                    type="image_url",
                    image_url={"url": f"data:image/jpeg;base64,{image_to_jpeg_base64(image, **kwargs)}"},
                )
            )

        if content_parts:
            self.data.append(ChatCompletionUserMessageParam(role="user", content=content_parts))


# ---------------------------------------------------------------------------
# Core query function
# ---------------------------------------------------------------------------


@overload
def llm_query(
    llm_config: LLM_Config,
    format: None = None,
    text: Optional[str] = None,
    system_prompt: Optional[str] = None,
    history: Optional[LLM_History] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    refusal_is_error: bool = False,
) -> str: ...
@overload
def llm_query(
    llm_config: LLM_Config,
    format: type[T],
    text: Optional[str] = None,
    system_prompt: Optional[str] = None,
    history: Optional[LLM_History] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    refusal_is_error: Literal[True] = True,
) -> T: ...
@overload
def llm_query(
    llm_config: LLM_Config,
    format: type[T],
    text: Optional[str] = None,
    system_prompt: Optional[str] = None,
    history: Optional[LLM_History] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    refusal_is_error: Literal[False] = False,
) -> T | str: ...


def llm_query(
    llm_config: LLM_Config,
    format: Optional[type[T]] = None,
    text: Optional[str] = None,
    system_prompt: Optional[str] = None,
    history: Optional[LLM_History] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    refusal_is_error: bool = False,
) -> str | T:
    """Send a chat completion request and return the response.

    Parameters
    ----------
    llm_config:
        Pre-built :class:`LLM_Config` instance.
    format:
        When provided (a ``pydantic.BaseModel`` subclass), the response is
        parsed into that model via ``client.beta.chat.completions.parse()``.
        When ``None`` (default), plain text is returned.
    text:
        The user message to send.
    system_prompt:
        An optional system prompt prepended to the message list.
    history:
        Previous conversation history.  When given, its messages are
        deep-copied into the request.
    temperature:
        Override the default temperature from *llm_config*.
    max_tokens:
        Override the default max_tokens from *llm_config*.
        *Only used when format is ``None``* (plain-text mode).
    refusal_is_error:
        When ``True`` and the model refuses the request (structured mode),
        a ``ValueError`` is raised.  When ``False`` the refusal text is
        returned as-is.
    """
    client = llm_config.client
    model_name = llm_config.model_name

    # Resolve per-call overrides
    temperature = temperature if temperature is not None else llm_config.temperature
    max_tokens = max_tokens if max_tokens is not None else llm_config.max_tokens

    # Build message list
    messages: list[ChatCompletionMessageParam] = []
    if history is not None:
        messages = deepcopy(history.data)
    if system_prompt is not None:
        messages.append(ChatCompletionSystemMessageParam(role="system", content=system_prompt))
    if text is not None:
        messages.append(ChatCompletionUserMessageParam(role="user", content=text))

    # Merge default kwargs with per-call overrides
    kwargs: dict[str, Any] = deepcopy(llm_config.kwargs)
    if (max_tokens is not None) and (format is None):
        kwargs["max_tokens"] = max_tokens
    if temperature is not None:
        kwargs["temperature"] = temperature

    if format is None:
        # -- plain-text completion ------------------------------------------
        chat_response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            **kwargs,
        )
        response_text = chat_response.choices[0].message.content
        if llm_config.strip_thinking:
            response_text = strip_thinking(response_text, llm_config.thinking_delimiter)
        return response_text

    # -- structured (parsed) completion -------------------------------------
    completion = client.beta.chat.completions.parse(
        model=model_name,
        messages=messages,
        response_format=format,
        **kwargs,
    )
    message = completion.choices[0].message
    refusal = message.refusal
    if refusal is not None:
        if refusal_is_error:
            raise ValueError("Refusal occurred: " + refusal)
        console.log("Refusal occurred: " + refusal)
        console.log("History: ", history)
        return refusal
    return cast(T, message.parsed)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


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
    help="Strip the thinking process from the response (for DeepSeek R1-like models)",
)
def __main__(
    api_key: Optional[str],
    model: Optional[str],
    base_url: Optional[str],
    max_tokens: Optional[int],
    temperature: Optional[float],
    interactive: bool,
    text: Optional[str],
    strip_thinking: bool,
) -> None:
    """Interact with an OpenAI-compatible LLM from the command line."""
    print = console.print  # noqa: A001
    prompt = console.input  # noqa: A001

    llm_config = LLM_Config(
        api_key=api_key,
        model_name=model,
        base_url=base_url,
        temperature=temperature,
        max_tokens=max_tokens,
        strip_thinking=strip_thinking,
    )
    actual_url = llm_config.client.base_url
    print(f"Using API {actual_url} with model {llm_config.model_name}")

    # -- One-shot mode ------------------------------------------------------
    if not interactive and text is not None:
        response = llm_query(llm_config, text=text)
        print(response)

    # -- Interactive mode ----------------------------------------------------
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
                if user_input in ("/exit", "/quit"):
                    break
                if user_input == "/reset":
                    history.clear()
                    print("Chat history cleared")
                    continue
                if user_input == "/system":
                    system_message = prompt("Enter system message: ")
                    history.add_system(system_message)
                    print("System message added")
                    continue
                if user_input == "/history":
                    pprint(history, max_string=256)
                    continue
                if user_input == "/regen":
                    history.del_last_message()
                    user_input = None  # re-send previous user message
                if user_input == "/image":
                    try:
                        image_path = prompt("Enter image path: ")
                        message_text = prompt("Enter optional message: ")
                        image = Image.open(str(image_path))
                        history.add_user(message_text, image)
                        print("Image message added")
                        user_input = None  # prevent sending the text twice
                    except Exception as e:
                        print(f"Error: {e}")
                        continue

                with console.status("[bold green]Waiting for response...", spinner="line"):
                    response = llm_query(
                        llm_config,
                        text=user_input,
                        history=history,
                    )
                history.add_user(user_input)  # add the user message after the response is received
                history.add_assistant(response, strip_thinking_in_response=strip_thinking)

                # Colorize the thinking block when stripping is enabled
                if strip_thinking:
                    delim = llm_config.thinking_delimiter
                    if delim in response:
                        response = response.replace(delim, "[/bright_black][green]")
                        if delim.replace("/", "") in response:
                            response = response.replace(delim.replace("</", "<"), "[bright_black]")
                        else:
                            response = "[bright_black]" + response

                print(response)
            except KeyboardInterrupt:
                print("\nExiting interactive mode.")
                break
            except SystemExit:
                break
            except Exception as e:
                print(f"Error: {e}")
                continue
    return


if __name__ == "__main__":
    __main__()
