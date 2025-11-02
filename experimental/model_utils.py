"""Utilities for LLM API calls."""

import logging

from openai import OpenAI


logger = logging.getLogger(__name__)


def call_model(
    client: OpenAI,
    system_prompt: str,
    user_prompt: str,
    model_name: str,
    temperature: float,
    max_tokens: int,
    response_format: dict = None,
) -> str:
    """Call LLM API with given prompts."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        kwargs = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            kwargs["response_format"] = response_format

        response = client.chat.completions.create(**kwargs)
        return response.choices[0].message.content

    except Exception as e:
        logger.error(f"Model API call failed: {e}")
        raise
