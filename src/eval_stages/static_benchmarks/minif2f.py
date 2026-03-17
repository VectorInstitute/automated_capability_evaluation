"""Adapter for the Tonic/MiniF2F static benchmark.

Dataset card: https://huggingface.co/datasets/Tonic/MiniF2F

MiniF2F contains mathematical problems with informal statements (LaTeX) and
formal Lean statements. Single split: train (488 rows).

Columns: name, split, informal_prefix, formal_statement, goal, header.
We use informal_prefix as input and formal_statement as target (autoformalization).
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List

from datasets import load_dataset

from src.eval_stages.prompts import DEFAULT_EVAL_PROMPT_TEMPLATE
from src.eval_stages.static_benchmarks.specs import StaticBenchmarkSpec
from src.schemas.eval_schemas import EvalDataset


def _slugify(text: str) -> str:
    """Convert arbitrary strings into safe directory-friendly IDs."""
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip()).strip("_").lower()
    return cleaned or "unknown"


def _iter_minif2f_samples(
    split: str,
    limit: int | None,
) -> Iterable[Dict[str, Any]]:
    """Yield rows from Tonic/MiniF2F in order."""
    ds = load_dataset("Tonic/MiniF2F", split=split)
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))
    yield from ds


def build_eval_datasets_from_minif2f(spec: StaticBenchmarkSpec) -> List[EvalDataset]:
    """Convert MiniF2F into a single EvalDataset.

    - input: informal_prefix (informal mathematical statement in LaTeX)
    - target: formal_statement (formal theorem in Lean)
    - domain: math
    - capability_id: minif2f
    Rows with empty informal_prefix or formal_statement are skipped.
    """
    tasks: List[Dict[str, str]] = []
    id_counts: Dict[str, int] = {}

    for idx, row in enumerate(_iter_minif2f_samples(spec.split, spec.limit)):
        informal = str(row.get("informal_prefix", "")).strip()
        formal = str(row.get("formal_statement", "")).strip()
        if not informal or not formal:
            continue

        raw_id = row.get("name")
        base_id = _slugify(str(raw_id).strip()) if raw_id else f"minif2f_{idx:04d}"
        cnt = id_counts.get(base_id, 0)
        id_counts[base_id] = cnt + 1
        task_id = base_id if cnt == 0 else f"{base_id}_{cnt}"

        tasks.append({"id": task_id, "input": informal, "target": formal})

    if not tasks:
        return []

    dataset = EvalDataset(
        area_id=spec.area_id,
        capability_id="minif2f",
        capability_name="MiniF2F",
        domain="math",
        tasks=tasks,
        num_tasks=len(tasks),
        prompt_template=DEFAULT_EVAL_PROMPT_TEMPLATE,
    )
    return [dataset]
