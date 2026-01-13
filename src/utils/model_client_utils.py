"""Utility functions for getting model clients."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Mapping, Optional, Sequence

import anthropic
import openai
from autogen_core.models import (
    ChatCompletionClient,
    ModelInfo,
    SystemMessage,
    UserMessage,
)
from autogen_ext.models.anthropic import AnthropicChatCompletionClient
from autogen_ext.models.openai import OpenAIChatCompletionClient


MAX_TOKENS = 1024 * 30

logger = logging.getLogger(__name__)

GEMINI_STUDIO_BASE = "https://generativelanguage.googleapis.com/v1beta/openai/"


def get_standard_model_client(
    model_name: str,
    *,
    seed: Optional[int] = None,
    **kwargs: Any,
) -> ChatCompletionClient:
    """Build a plain client for use with `async_call_model`."""
    n = model_name.lower()

    # OpenAI GPT / o-series models
    if n.startswith(("gpt-", "o1-", "o3-", "gpt-5", "o4-")):
        # Convert max_tokens to max_completion_tokens for OpenAI
        if "max_tokens" in kwargs:
            kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")

        # o-series and gpt-5 models don't support custom temperature
        # Remove temperature if it's set for these models
        if (
            any(key in n for key in ("o1-", "o3-", "o4-", "gpt-5"))
            and "temperature" in kwargs
        ):
            logger.debug(
                "Removing 'temperature' parameter for model '%s' - not supported",
                model_name,
            )
            kwargs.pop("temperature")

        return OpenAIChatCompletionClient(model=model_name, seed=seed, **kwargs)

    # Anthropic Claude models
    if "claude" in n:
        kwargs.setdefault("timeout", None)
        return AnthropicChatCompletionClient(model=model_name, **kwargs)

    # Gemini via OpenAI-compatible AI Studio endpoint
    if "gemini" in n:
        api_key = kwargs.pop("api_key", os.getenv("GOOGLE_API_KEY"))
        if not api_key:
            raise ValueError("Set GOOGLE_API_KEY for Gemini (AI Studio).")

        model_info = kwargs.pop(
            "model_info",
            ModelInfo(
                vision=True,
                function_calling=True,
                json_output=True,
                structured_output=True,
                family="unknown",
            ),
        )

        return OpenAIChatCompletionClient(
            model=model_name,
            base_url=GEMINI_STUDIO_BASE,
            api_key=api_key,
            model_info=model_info,
            **kwargs,
        )

    raise ValueError(f"Unsupported model '{model_name}'.")


class ModelCallError(RuntimeError):
    """Error raised when a standardized model call fails."""


class ModelCallMode:
    """Output modes for `async_call_model`."""

    TEXT = "text"
    JSON_PARSE = "json_parse"
    STRUCTURED = "structured"


async def async_call_model(
    model_client: ChatCompletionClient,
    *,
    system_prompt: Optional[str] = None,
    user_prompt: Optional[str] = None,
    messages: Optional[Sequence[Any]] = None,
    mode: str = ModelCallMode.TEXT,
    max_attempts: int = 3,
    extra_kwargs: Optional[Mapping[str, Any]] = None,
) -> Any:
    """Perform a standard async model call with output modes and retry logic.

    - Builds messages from prompts if `messages` is None.
    - `mode`:
      - TEXT: return `str` content.
      - JSON_PARSE: parse JSON and return `dict`.
      - STRUCTURED: return the raw provider response.
    - Retries on transient API errors (rate limits, timeouts, server errors),
      empty content, and JSON parse failures.

    Note: temperature, max_tokens, seed should be passed to get_standard_model_client()
    when creating the client, not here.
    """
    if messages is None:
        if user_prompt is None and system_prompt is None:
            raise ValueError(
                "Either 'messages' or at least one of 'system_prompt' / 'user_prompt' "
                "must be provided."
            )

        built_messages: list[Any] = []
        if system_prompt:
            built_messages.append(SystemMessage(content=system_prompt))
        if user_prompt:
            built_messages.append(UserMessage(content=user_prompt, source="user"))
        messages = built_messages

    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")

    last_error: Exception | None = None

    # Define retryable exceptions
    retryable_exceptions = (
        openai.RateLimitError,
        openai.APITimeoutError,
        openai.InternalServerError,
        anthropic.RateLimitError,
        anthropic.APITimeoutError,
        anthropic.InternalServerError,
    )

    for attempt in range(1, max_attempts + 1):
        request_kwargs: Dict[str, Any] = {}

        # Output / structured config
        if mode in (ModelCallMode.JSON_PARSE, ModelCallMode.STRUCTURED):
            # Many clients support json_output / structured_output flags.
            # Some may ignore these silently; others might raise if unsupported.
            request_kwargs.setdefault("json_output", True)
            if mode == ModelCallMode.STRUCTURED:
                request_kwargs.setdefault("structured_output", True)

        # Extra kwargs always win
        if extra_kwargs:
            request_kwargs.update(extra_kwargs)

        try:
            response = await model_client.create(
                messages=list(messages),  # type: ignore[arg-type]
                **request_kwargs,
            )
        except retryable_exceptions as exc:
            # Retry on transient API errors
            last_error = exc
            if attempt < max_attempts:
                wait_time = min(2**attempt, 10)  # Exponential backoff, max 10s
                logger.warning(
                    "Transient API error on attempt %d/%d: %s. Retrying in %ds...",
                    attempt,
                    max_attempts,
                    exc,
                    wait_time,
                )
                import asyncio

                await asyncio.sleep(wait_time)
                continue
            logger.error("Max retries reached for transient API error: %s", exc)
            break
        except Exception as exc:
            # Non-retryable error - fail immediately
            logger.error("Model call failed with non-retryable error: %s", exc)
            last_error = exc
            break

        # Extract content in a provider-agnostic way.
        content = getattr(response, "content", None)
        if content is None:
            last_error = ModelCallError("Model returned empty response content")
            logger.warning(
                "Empty response content on attempt %d/%d", attempt, max_attempts
            )
            if attempt < max_attempts:
                continue
            break

        # Normalize to string for text / JSON modes.
        if isinstance(content, (list, tuple)):
            content_str = "\n".join(str(part) for part in content)
        else:
            content_str = str(content)

        content_str = content_str.strip()
        if not content_str:
            last_error = ModelCallError("Model returned empty response content")
            logger.warning(
                "Blank response content on attempt %d/%d", attempt, max_attempts
            )
            if attempt < max_attempts:
                continue
            break

        if mode == ModelCallMode.TEXT:
            return content_str

        if mode == ModelCallMode.JSON_PARSE:
            import json

            try:
                return json.loads(content_str)
            except Exception as exc:  # pragma: no cover - JSON edge cases
                last_error = ModelCallError(
                    f"Failed to parse JSON from model response: {exc}"
                )
                logger.warning(
                    "JSON parsing failed on attempt %d/%d: %s",
                    attempt,
                    max_attempts,
                    exc,
                )
                if attempt < max_attempts:
                    continue
                break

        # STRUCTURED mode: return provider object as-is to the caller.
        return response

    # If we get here, all attempts failed.
    if last_error is None:
        raise ModelCallError("Model call failed for unknown reasons")
    if isinstance(last_error, ModelCallError):
        raise last_error
    raise ModelCallError(f"Model call failed: {last_error}") from last_error
