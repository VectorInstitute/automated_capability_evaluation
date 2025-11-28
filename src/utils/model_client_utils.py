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
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)


MAX_TOKENS = 1024 * 30

logger = logging.getLogger(__name__)

GEMINI_STUDIO_BASE = "https://generativelanguage.googleapis.com/v1beta/openai/"


class RetryableModelClient:
    """Wrap a client and retry `create` on transient API errors."""

    def __init__(self, client: Any, max_retries: int = 3):
        self.client = client
        self.max_retries = max_retries

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(
            (
                openai.RateLimitError,
                openai.APITimeoutError,
                openai.InternalServerError,
                anthropic.RateLimitError,
                anthropic.APITimeoutError,
                anthropic.InternalServerError,
            )
        ),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    async def create(self, *args: Any, **kwargs: Any) -> Any:
        """Create with retry logic for transient errors."""
        return await self.client.create(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        """Delegate all other attributes to the wrapped client."""
        return getattr(self.client, name)


def get_model_client(model_name: str, seed: Optional[int] = None, **kwargs: Any) -> Any:
    """Legacy factory: return a retry-wrapped client for `model_name`."""
    n = model_name.lower()

    if n.startswith(("gpt-", "o1-", "o3-", "gpt-5")):
        kwargs.setdefault("max_completion_tokens", MAX_TOKENS)
        openai_client = OpenAIChatCompletionClient(
            model=model_name, seed=seed, **kwargs
        )
        return RetryableModelClient(openai_client)

    if "claude" in n:
        kwargs.setdefault("max_tokens", MAX_TOKENS)
        kwargs.setdefault("timeout", None)
        anthropic_client = AnthropicChatCompletionClient(model=model_name, **kwargs)
        return RetryableModelClient(anthropic_client)

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

        kwargs.setdefault("max_completion_tokens", MAX_TOKENS)

        client = OpenAIChatCompletionClient(
            model=model_name,
            base_url=GEMINI_STUDIO_BASE,
            api_key=api_key,
            model_info=model_info,
            **kwargs,
        )
        return RetryableModelClient(client)

    raise ValueError(f"Unsupported model '{model_name}'.")


def get_standard_model_client(
    model_name: str,
    *,
    seed: Optional[int] = None,
    **kwargs: Any,
) -> ChatCompletionClient:
    """Build a plain client for use with `async_call_model`."""
    n = model_name.lower()

    # OpenAI GPT / o-series models
    if n.startswith(("gpt-", "o1-", "o3-", "gpt-5")):
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
    model_name: Optional[str] = None,
    system_prompt: Optional[str] = None,
    user_prompt: Optional[str] = None,
    messages: Optional[Sequence[Any]] = None,
    mode: str = ModelCallMode.TEXT,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    top_p: Optional[float] = None,
    seed: Optional[int] = None,
    max_attempts: int = 3,
    extra_kwargs: Optional[Mapping[str, Any]] = None,
) -> Any:
    """Perform a standard async model call with provider-aware args and output modes.

    - Builds messages from prompts if `messages` is None.
    - Maps `temperature`, `max_tokens`, `top_p`, `seed` to the right provider kwargs.
    - `mode`:
      - TEXT: return `str` content.
      - JSON_PARSE: parse JSON and return `dict`.
      - STRUCTURED: return the raw provider response.
    - Retries only for empty content / JSON parse failures; other errors raise
      `ModelCallError` immediately.
    """
    # Try to infer model name if not provided explicitly.
    resolved_model_name: Optional[str] = model_name
    if resolved_model_name is None:
        underlying = getattr(model_client, "client", model_client)
        resolved_model_name = getattr(underlying, "model", None)

    # Identify provider family from the model name.
    provider: Optional[str] = None
    lowered_name = (
        resolved_model_name.lower() if isinstance(resolved_model_name, str) else ""
    )
    if lowered_name.startswith(("gpt-", "o1-", "o3-", "gpt-5")):
        provider = "openai"
    elif "claude" in lowered_name:
        provider = "anthropic"
    elif "gemini" in lowered_name:
        provider = "gemini"

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
    drop_temperature_for_model = False

    for attempt in range(1, max_attempts + 1):
        request_kwargs: Dict[str, Any] = {}

        if temperature is not None and not drop_temperature_for_model:
            if provider == "openai" and lowered_name:
                # "o1" models: special handling, often ignore temperature.
                # "o3-mini", "o3", "o4-mini": temperature is not always supported.
                if any(
                    key in lowered_name for key in ("o1", "o3-mini", "o3", "o4-mini")
                ):
                    logger.debug(
                        "Not sending 'temperature' for model '%s' due to known "
                        "limitations.",
                        resolved_model_name,
                    )
                else:
                    request_kwargs["temperature"] = temperature
            elif provider in {"anthropic", "gemini", None}:
                # Anthropic Claude and Gemini generally support temperature;
                # for unknown providers we optimistically pass it through.
                request_kwargs["temperature"] = temperature

        # Map unified `max_tokens` to provider-specific kwarg.
        if max_tokens is not None:
            if provider in {"openai", "gemini"}:
                request_kwargs["max_completion_tokens"] = max_tokens
            elif provider == "anthropic":
                request_kwargs["max_tokens"] = max_tokens
            else:
                request_kwargs["max_tokens"] = max_tokens

        # `top_p` only for OpenAI-style providers.
        if top_p is not None and provider in {"openai", "gemini", None}:
            request_kwargs["top_p"] = top_p
        if seed is not None:
            request_kwargs["seed"] = seed

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
        except TypeError as exc:
            # Some models (e.g., certain reasoning or o-series models) do not
            # support temperature or other generation parameters. If the error
            # message clearly points to 'temperature', drop it and retry once.
            if (
                "temperature" in str(exc)
                and "temperature" in request_kwargs
                and not drop_temperature_for_model
            ):
                logger.warning(
                    "Model rejected 'temperature' parameter; retrying without it. "
                    "Error was: %s",
                    exc,
                )
                drop_temperature_for_model = True
                last_error = exc
                continue
            last_error = exc
            logger.error("Model call failed with TypeError: %s", exc)
            break
        except Exception as exc:  # pragma: no cover - network/SDK errors
            # Let lower-level client / infrastructure handle any network or
            # transient retries. At this layer we convert to ModelCallError
            # without additional retry loops to avoid duplicating behaviour.
            logger.error("Model call failed with unexpected error: %s", exc)
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
