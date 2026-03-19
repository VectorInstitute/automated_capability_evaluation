"""Adapter for a local finance task JSON export.

This adapter ingests a local JSON file (e.g. `finance_tasks.json`) that follows
the repo's task-generation export shape:

- Top-level keys: `metadata`, `tasks`
- Each task contains:
  - `task_id` (str)
  - `task_statement` (str) — includes options for multiple-choice tasks
  - `generation_metadata.correct_answer` (str) — e.g. "A", "B", ...

We map:
- input: task_statement
- target: correct_answer (fallbacks supported)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.eval_stages.prompts import DEFAULT_EVAL_PROMPT_TEMPLATE
from src.eval_stages.static_benchmarks.specs import StaticBenchmarkSpec
from src.schemas.eval_schemas import EvalDataset


def _sanitize_text(text: str) -> str:
    """Sanitize text so it is safe to JSON-encode and send to APIs."""
    # Remove null bytes (can break downstream tooling / transports).
    text = text.replace("\x00", "")
    # Replace any invalid unicode sequences (e.g., unpaired surrogates) deterministically.
    return text.encode("utf-8", errors="replace").decode("utf-8", errors="replace")


def _resolve_json_path(benchmark_id: str) -> Path:
    raw = benchmark_id.strip()
    if raw.startswith("file://"):
        raw = raw[len("file://") :]
    candidate = Path(raw)
    if candidate.exists():
        return candidate

    # Default: assume a repo-root file name was given as benchmark_id.
    default = Path("finance_tasks.json")
    if default.exists():
        return default

    # Fall back to relative path from CWD.
    return candidate


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}, got {type(data).__name__}")
    return data


def _extract_target(task: Dict[str, Any]) -> str:
    gen_md = task.get("generation_metadata")
    if isinstance(gen_md, dict):
        val = gen_md.get("correct_answer")
        if val is not None:
            return str(val).strip()

    # Fallbacks for other possible exports.
    for key in ("correct_answer", "answer", "target", "label"):
        if key in task and task[key] is not None:
            return str(task[key]).strip()

    return ""


def build_eval_datasets_from_finance_tasks(spec: StaticBenchmarkSpec) -> List[EvalDataset]:
    """Convert a local finance_tasks JSON file into a single EvalDataset."""
    json_path = _resolve_json_path(spec.benchmark_id)
    payload = _read_json(json_path)
    raw_tasks = payload.get("tasks", [])
    if not isinstance(raw_tasks, list):
        raise ValueError(
            f"Expected `tasks` to be a list in {json_path}, got {type(raw_tasks).__name__}"
        )

    tasks: List[Dict[str, str]] = []
    limit: Optional[int] = spec.limit
    offset: int = max(0, int(spec.offset or 0))

    for idx, row in enumerate(raw_tasks[offset:]):
        if not isinstance(row, dict):
            continue

        task_id = str(row.get("task_id", "")).strip()
        statement = _sanitize_text(str(row.get("task_statement", "")).strip())
        target = _extract_target(row)

        if not task_id:
            global_idx = offset + len(tasks)
            task_id = f"finance_tasks_{global_idx:05d}"
        if not statement or not target:
            continue

        tasks.append({"id": task_id, "input": statement, "target": target})
        if limit is not None and len(tasks) >= limit:
            break

    if not tasks:
        return []

    dataset = EvalDataset(
        area_id=spec.area_id,
        capability_id=str(spec.capability_id or "finance_tasks"),
        capability_name=str(spec.capability_name or "Finance Tasks"),
        domain=str(spec.domain or "finance"),
        tasks=tasks,
        num_tasks=len(tasks),
        prompt_template=DEFAULT_EVAL_PROMPT_TEMPLATE,
    )
    return [dataset]

