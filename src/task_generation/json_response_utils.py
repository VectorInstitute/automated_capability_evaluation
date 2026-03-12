"""Shared helpers for normalizing model replies and parsing JSON-like text."""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Optional

from src.schemas.task_gen_io_utils import (
    escape_invalid_json_backslashes,
    strip_agent_terminator,
)


def stringify_payload(payload: Any) -> str:
    """Serialize dict/list payloads for prompts, otherwise coerce to string."""
    if isinstance(payload, (dict, list)):
        return json.dumps(payload, indent=2, ensure_ascii=False)
    return str(payload)


def normalize_reply_to_text(reply: Any) -> str:  # noqa: PLR0911
    """Best-effort normalization of provider replies into a plain text string."""
    if reply is None:
        return ""

    if isinstance(reply, str):
        return reply.strip()

    if isinstance(reply, list) and reply:
        return normalize_reply_to_text(reply[-1])

    if isinstance(reply, dict):
        if "content" in reply:
            return str(reply["content"]).strip()
        if "message" in reply:
            return normalize_reply_to_text(reply["message"])
        if "choices" in reply and isinstance(reply["choices"], list) and reply["choices"]:
            return normalize_reply_to_text(reply["choices"][-1])

    try:
        return str(reply).strip()
    except Exception:
        return ""


def parse_json_like(
    content: str,
    *,
    on_repair: Optional[Callable[[str], None]] = None,
) -> Optional[Any]:  # noqa: PLR0911
    """Parse JSON from raw/fenced/braced text with invalid-escape repair."""
    content = strip_agent_terminator((content or "").strip())
    if not content:
        return None

    def _loads_with_repair(candidate: str, repair_context: str) -> Optional[Any]:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as exc:
            repaired = escape_invalid_json_backslashes(candidate)
            if repaired != candidate:
                try:
                    parsed = json.loads(repaired)
                    if on_repair is not None:
                        on_repair(
                            f"Recovered JSON after escape repair ({repair_context}). "
                            f"Original error at line {exc.lineno} col {exc.colno}: {exc.msg}"
                        )
                    return parsed
                except json.JSONDecodeError:
                    pass
            return None

    blocks = re.findall(r"```json\s*(.*?)\s*```", content, flags=re.DOTALL | re.IGNORECASE)
    if blocks:
        for block in blocks:
            obj = _loads_with_repair(strip_agent_terminator(block.strip()), "fenced block")
            if obj is not None:
                return obj
    else:
        obj = _loads_with_repair(content, "raw content")
        if obj is not None:
            return obj

    start, end = content.find("{"), content.rfind("}")
    if start != -1 and end != -1 and end > start:
        obj = _loads_with_repair(
            strip_agent_terminator(content[start : end + 1]),
            "brace slice",
        )
        if obj is not None:
            return obj

    for match in re.finditer(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", content):
        candidate = strip_agent_terminator(match.group(0).strip())
        obj = _loads_with_repair(candidate, "nested brace candidate")
        if obj is not None:
            return obj

    return None
