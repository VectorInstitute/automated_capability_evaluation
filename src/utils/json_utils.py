"""JSON utility functions for parsing and cleaning LLM responses."""

import json
import logging
import re
from typing import Any, Dict, Union


log = logging.getLogger("utils.json_utils")


def extract_json_from_markdown(content: str) -> str:
    """Extract JSON from markdown if present and clean control characters."""
    content = content.strip()

    if content.startswith("```json") and content.endswith("```"):
        content = content[7:-3].strip()
    elif content.startswith("```") and content.endswith("```"):
        content = content[3:-3].strip()

    return re.sub(r"[\x00-\x1f\x7f-\x9f]", "", content)


def parse_llm_json_response(raw_content: Union[str, Any]) -> Dict[str, Any]:
    """Parse LLM JSON response."""
    try:
        # Ensure content is a string
        if not isinstance(raw_content, str):
            raw_content = str(raw_content)

        # Clean the content first
        cleaned_content = extract_json_from_markdown(raw_content)

        # Parse the JSON
        return json.loads(cleaned_content)

    except json.JSONDecodeError as e:
        log.error(f"Failed to parse JSON response: {e}")
        log.error(f"Raw content: {raw_content}")
        raise
    except Exception as e:
        log.error(f"Unexpected error parsing JSON: {e}")
        log.error(f"Raw content: {raw_content}")
        raise
