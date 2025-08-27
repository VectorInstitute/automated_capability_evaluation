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
        log.error(f"Content length: {len(cleaned_content)} characters")

        # Try to fix common JSON issues
        try:
            # Attempt to fix unterminated strings by finding the last complete entry
            if "Unterminated string" in str(e):
                # Find the last complete capability entry
                last_complete = cleaned_content.rfind('"},')
                if last_complete > 0:
                    # Truncate to last complete entry and close the JSON
                    fixed_content = cleaned_content[: last_complete + 2] + "\n  }\n}"
                    log.warning(
                        "Attempting to fix unterminated JSON by truncating to last complete entry"
                    )
                    return json.loads(fixed_content)
        except Exception as fix_error:
            log.error(f"Failed to fix JSON: {fix_error}")

        # If we can't fix it, log more details and re-raise
        log.error(f"Raw content (last 500 chars): {raw_content[-500:]}")
        raise
    except Exception as e:
        log.error(f"Unexpected error parsing JSON: {e}")
        log.error(f"Raw content: {raw_content}")
        raise
