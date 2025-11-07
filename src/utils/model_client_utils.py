"""Utility functions for getting model clients."""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

import anthropic
import openai
from autogen_core.models import ModelInfo
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
    """Wrapper that adds retry logic to any model client."""

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
    """Get a model client for the given model name."""
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
