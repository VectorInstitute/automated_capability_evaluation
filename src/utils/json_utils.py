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
    
    # Fix LaTeX backslashes: escape backslashes that are not part of valid JSON escape sequences
    # Valid JSON escapes: \", \\, \/, \b, \f, \n, \r, \t, \uXXXX
    # LaTeX uses \(, \), \[, \], \epsilon, \delta, etc. which need to be escaped as \\(, \\), etc.
    
    # Process the content character by character to properly handle string boundaries
    result = []
    i = 0
    in_string = False
    
    while i < len(content):
        char = content[i]
        
        # Track when we enter/exit string values
        if char == '"':
            # Check if this quote is escaped by counting backslashes before it
            # An escaped quote has an odd number of backslashes before it
            backslash_count = 0
            j = i - 1
            while j >= 0 and content[j] == '\\':
                backslash_count += 1
                j -= 1
            
            # If odd number of backslashes, the quote is escaped (part of string content)
            if backslash_count % 2 == 1:
                result.append(char)
                i += 1
                continue
            
            # This is a real quote - toggle string state
            in_string = not in_string
            result.append(char)
            i += 1
            continue
        
        # Handle backslashes inside string values
        if char == '\\' and in_string:
            if i + 1 < len(content):
                next_char = content[i + 1]
                # Check for valid JSON escape sequences
                # For single-character escapes: ", \, /, b, f, n, r, t
                # We need to ensure they're not part of a longer sequence (e.g., \to should not match \t)
                if next_char in '"\\/':
                    # Always valid single-char escapes
                    result.append(char)
                    result.append(next_char)
                    i += 2
                    continue
                elif next_char in 'bfnrt':
                    # Check if this is a complete escape (not part of longer sequence)
                    # Valid if followed by non-alphanumeric or end of string
                    if i + 2 >= len(content) or not (content[i + 2].isalnum() or content[i + 2] == '_'):
                        # Complete escape sequence (e.g., \t, \n, \r, \b, \f)
                        result.append(char)
                        result.append(next_char)
                        i += 2
                        continue
                    # Otherwise it's part of a longer sequence (e.g., \to, \lim) - escape it
                    result.append('\\\\')
                    result.append(next_char)
                    i += 2
                    continue
                elif next_char == 'u' and i + 5 < len(content):
                    # Check for \uXXXX pattern
                    hex_part = content[i + 2:i + 6]
                    if all(c in '0123456789abcdefABCDEF' for c in hex_part):
                        # Valid \uXXXX escape
                        result.append(char)
                        result.append(next_char)
                        result.append(hex_part)
                        i += 6
                        continue
                
                # Invalid escape sequence (like LaTeX \(, \), \[, \], \epsilon, \to, etc.)
                # Double-escape the backslash
                result.append('\\\\')
                result.append(next_char)
                i += 2
                continue
            else:
                # Backslash at end of content - escape it
                result.append('\\\\')
                i += 1
                continue
        
        # Regular character
        result.append(char)
        i += 1
    
    content = ''.join(result)
    
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

        # Try to fix LaTeX backslash issues if the error is about invalid escape
        if "Invalid \\escape" in str(e) or "Invalid escape" in str(e):
            try:
                log.warning("Attempting to fix LaTeX backslash escape issues")
                # Apply more aggressive LaTeX backslash fixing
                # Escape all backslashes in string values that aren't valid JSON escapes
                fixed_content = cleaned_content
                
                # Use regex to find and fix invalid escapes in string values
                # This pattern finds backslashes in string values and escapes invalid ones
                def fix_escapes_in_string(match):
                    """Fix escapes within a JSON string value."""
                    string_content = match.group(1)
                    # Escape backslashes not followed by valid JSON escape chars
                    fixed = re.sub(r'\\(?!["\\/bfnrtu]|u[0-9a-fA-F]{4})', r'\\\\', string_content)
                    return f'"{fixed}"'
                
                # Find string values and fix them
                # Pattern: "([^"\\]|\\.)*" - matches JSON string values
                fixed_content = re.sub(r'"((?:[^"\\]|\\.)*)"', fix_escapes_in_string, fixed_content)
                
                result = json.loads(fixed_content)
                log.info("Successfully fixed LaTeX backslash issues")
                return result if isinstance(result, dict) else {}
            except Exception as fix_error:
                log.error(f"Failed to fix LaTeX escape issues: {fix_error}")

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
