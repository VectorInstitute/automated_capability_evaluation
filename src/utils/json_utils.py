"""JSON utility functions for parsing and cleaning LLM responses."""

import json
import logging
import re
from typing import Any, Dict, Union


log = logging.getLogger("utils.json_utils")


def extract_json_from_markdown(content: str) -> str:
    """Extract JSON from markdown if present and clean control characters."""
    content = content.strip()

    # Handle Gemini's format: "```json\n...\n```"
    if content.startswith('"```json') and content.endswith('```"'):
        content = content[8:-4].strip()
    elif content.startswith('"```') and content.endswith('```"'):
        content = content[4:-4].strip()
    # Handle standard markdown format: ```json\n...\n```
    elif content.startswith("```json") and content.endswith("```"):
        content = content[7:-3].strip()
    elif content.startswith("```") and content.endswith("```"):
        content = content[3:-3].strip()

    content = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", content)

    if content and not content.lstrip().startswith(("{", "[")):
        brace_start = content.find("{")
        brace_end = content.rfind("}")
        bracket_start = content.find("[")
        bracket_end = content.rfind("]")

        if brace_start != -1 and brace_end > brace_start:
            content = content[brace_start : brace_end + 1].strip()
        elif bracket_start != -1 and bracket_end > bracket_start:
            content = content[bracket_start : bracket_end + 1].strip()

    return content


def fix_common_json_errors(content: str) -> str:
    """Fix common JSON syntax errors."""
    content = re.sub(r':\s*=\s*"', ':"', content)
    content = re.sub(r'(\w+):\s*"', r'"\1":"', content)
    content = re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", content)
    return re.sub(r",(\s*[}\]])", r"\1", content)


def parse_llm_json_response(raw_content: Union[str, Any]) -> Dict[str, Any]:
    """Parse LLM JSON response."""
    try:
        if not isinstance(raw_content, str):
            raw_content = str(raw_content)

        cleaned_content = extract_json_from_markdown(raw_content)
        cleaned_content = fix_common_json_errors(cleaned_content)

        if not cleaned_content:
            raise json.JSONDecodeError("Empty JSON content", cleaned_content or "", 0)

        result = json.loads(cleaned_content)
        return result if isinstance(result, dict) else {}

    except json.JSONDecodeError as e:
        log.error(f"Failed to parse JSON response: {e}")
        log.error(f"Content length: {len(cleaned_content)} characters")

        try:
            if "Unterminated string" in str(e):
                last_complete = cleaned_content.rfind('"},')
                if last_complete > 0:
                    fixed_content = cleaned_content[: last_complete + 2] + "\n  }\n}"
                    log.warning(
                        "Attempting to fix unterminated JSON by truncating to last complete entry"
                    )
                    result = json.loads(fixed_content)
                    return result if isinstance(result, dict) else {}
        except Exception as fix_error:
            log.error(f"Failed to fix JSON: {fix_error}")

        log.error(f"Raw content (last 500 chars): {raw_content[-500:]}")
        raise

    except Exception as e:
        log.error(f"Unexpected error parsing JSON: {e}")
        log.error(f"Raw content: {raw_content}")
        raise
