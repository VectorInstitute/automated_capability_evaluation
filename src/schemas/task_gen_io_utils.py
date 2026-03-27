"""Shared I/O-oriented utilities for task-generation agents."""

from __future__ import annotations

import re
from typing import Any, Sequence


try:
    from autogen_agentchat.messages import TextMessage as _TextMessage
except ModuleNotFoundError:  # pragma: no cover - optional runtime dependency
    _TextMessage = None


def strip_agent_terminator(text: str) -> str:
    """Remove trailing agent terminator tokens from model output."""
    if not text:
        return ""
    return re.sub(r"\s*TERMINATE\s*$", "", text, flags=re.IGNORECASE).strip()


def escape_invalid_json_backslashes(s: str) -> str:
    """
    Escape backslashes that are invalid in JSON escape sequences.

    Example repaired patterns:
    - "\\dot" -> "\\\\dot"
    - "\\frac" -> "\\\\frac"

    Valid JSON escapes (\\", \\\\, \\/, \\b, \\f, \\n, \\r, \\t, \\uXXXX) are preserved.
    For \\u, preservation only applies when followed by exactly 4 hex digits.

    Args:
        s: The input string to process.

    Returns
    -------
        A new string with invalid JSON backslashes escaped.
    """  # noqa: D301
    out: list[str] = []
    i = 0
    n = len(s)
    while i < n:
        ch = s[i]
        if ch != "\\":
            out.append(ch)
            i += 1
            continue

        if i + 1 >= n:
            out.append("\\\\")
            i += 1
            continue

        nxt = s[i + 1]
        if nxt in ['"', "\\", "/", "b", "f", "n", "r", "t"]:
            out.append("\\")
            out.append(nxt)
            i += 2
            continue

        if nxt == "u":
            if i + 5 < n and re.fullmatch(r"[0-9a-fA-F]{4}", s[i + 2 : i + 6]):
                out.append("\\u")
                out.append(s[i + 2 : i + 6])
                i += 6
                continue
            out.append("\\\\u")
            i += 2
            continue

        out.append("\\\\")
        out.append(nxt)
        i += 2

    return "".join(out)


def extract_last_message_text(
    messages: Sequence[Any],
    *,
    include_text_attr: bool,
    strip_content: bool,
) -> str:
    """Extract the last message text from AgentChat result messages."""
    if not messages:
        return ""

    for message in reversed(messages):
        if _TextMessage is not None and isinstance(message, _TextMessage):
            text_content = message.content or ""
            return text_content.strip() if strip_content else text_content

        raw_content = getattr(message, "content", None)
        if isinstance(raw_content, str):
            return raw_content.strip() if strip_content else raw_content

        if include_text_attr:
            raw_text = getattr(message, "text", None)
            if isinstance(raw_text, str):
                return raw_text.strip() if strip_content else raw_text

    return ""
