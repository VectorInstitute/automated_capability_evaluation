"""Shared helpers for the self-contrast runner scripts.

Keeps the per-variant runners (V1-V5) focused on their pipeline wiring by
extracting the pieces that are genuinely identical: the batch-file lookup
and the JSON-response parser shared between the two single-agent runners.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional, Tuple


def resolve_batch_file(batch_file: str, dataset_dir: Path) -> Path:
    """Resolve *batch_file* against the current directory or *dataset_dir*."""
    candidate = Path(batch_file)
    if candidate.exists():
        return candidate
    candidate = dataset_dir / batch_file
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        f"Batch file not found: {batch_file} (searched {dataset_dir})"
    )


def parse_single_agent_response(
    content: str, task_type: str
) -> Tuple[Optional[str], str]:
    """Parse a single-agent JSON response into ``(prediction, reasoning)``.

    Used by both V2 (``run_single_agent``) and V5 (``run_single_agent_tools``)
    which expect the model to reply with ``{"answer": ..., "reasoning": ...}``.
    """
    del task_type  # reserved for future task-type-aware parsing
    prediction: Optional[str] = None
    reasoning = "No reasoning provided"

    json_match = re.search(r"\{.*\}", content, re.DOTALL)
    if json_match:
        try:
            json_str = json_match.group(0)
            json_str = re.sub(r'"\s*\n\s*"', '", "', json_str)
            parsed = json.loads(json_str)
            prediction = parsed.get("answer")
            reasoning = parsed.get("reasoning", reasoning)
        except Exception:
            pass

    if prediction is None:
        ans_match = re.search(
            r'"answer":\s*["\']?(.*?)["\']?[\s,}]', content, re.IGNORECASE
        )
        if ans_match:
            prediction = ans_match.group(1).strip()

    if prediction is None and not json_match:
        prediction = content.strip()
        reasoning = "Raw model response"

    return prediction, reasoning
