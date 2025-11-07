"""Utilities for LLM API calls."""

import logging
import time

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
    max_retries: int = 3,
    retry_delay: float = 2.0,
) -> str:
    """Call LLM API with given prompts and automatic retries."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    kwargs = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if response_format:
        kwargs["response_format"] = response_format

    last_error = None
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content

            if content is None:
                raise ValueError("Model returned empty response")

            return content

        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2**attempt)
                logger.warning(
                    f"API call failed (attempt {attempt + 1}/{max_retries}): {e}"
                )
                logger.info(f"Retrying in {wait_time:.1f} seconds...")
                time.sleep(wait_time)
            else:
                logger.error(f"API call failed after {max_retries} attempts: {e}")

    raise last_error
