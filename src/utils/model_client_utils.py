"""Utility functions for getting model clients."""

from __future__ import annotations

import os
from typing import Any, Optional

from autogen_core.models import ModelInfo
from autogen_ext.models.anthropic import AnthropicChatCompletionClient
from autogen_ext.models.openai import OpenAIChatCompletionClient


GEMINI_STUDIO_BASE = "https://generativelanguage.googleapis.com/v1beta/openai/"


def get_model_client(model_name: str, seed: Optional[int] = None, **kwargs) -> Any:
    """Return a model client for the given model name."""
    n = model_name.lower()

    if n.startswith(("gpt-", "o1-", "o3-")):
        return OpenAIChatCompletionClient(model=model_name, seed=seed, **kwargs)

    if "claude" in n:
        return AnthropicChatCompletionClient(model=model_name, **kwargs)

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
