"""Adapter for the StatEval Statistical Research static benchmark.

Dataset: https://huggingface.co/datasets/0v01111/StatEval-Statistical-Research

Research-level, proof-based tasks from papers. Exposed as a single
capability "statistical_research" for the unified StatEval benchmark.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

from datasets import load_dataset

from src.eval_stages.prompts import DEFAULT_EVAL_PROMPT_TEMPLATE
from src.eval_stages.static_benchmarks.specs import StaticBenchmarkSpec
from src.schemas.eval_schemas import EvalDataset


def _iter_stateval_research_samples(
    split: str,
    limit: int | None,
) -> Iterable[Dict[str, Any]]:
    """Yield rows from StatEval-Statistical-Research in stable order."""
    ds = load_dataset("0v01111/StatEval-Statistical-Research", split=split)
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))
    yield from ds


def _extract_input(row: Dict[str, Any]) -> str:
    """Build question/prompt text from a research row."""
    stem = (
        str(
            row.get("question")
            or row.get("prompt")
            or row.get("task")
            or row.get("problem")
            or row.get("context")
            or ""
        ).strip()
    )
    return stem or ""


def _extract_answer(row: Dict[str, Any]) -> str:
    """Extract target answer from a research row."""
    answer = row.get("answer")
    if isinstance(answer, dict):
        for key in ("label", "text", "final", "value", "solution"):
            if key in answer and answer[key] is not None:
                return str(answer[key]).strip()
        return str(answer).strip()
    if answer is None:
        return ""
    return str(answer).strip()


def build_eval_datasets_from_stateval_research(
    spec: StaticBenchmarkSpec,
) -> List[EvalDataset]:
    """Convert StatEval Statistical Research into a single EvalDataset.

    All tasks under one capability "statistical_research".
    Uses spec.area_id and spec.domain (e.g. area_id=stateval, domain=math).
    """
    tasks: List[Dict[str, str]] = []

    for idx, row in enumerate(
        _iter_stateval_research_samples(spec.split, spec.limit)
    ):
        input_text = _extract_input(row)
        target = _extract_answer(row)
        if not input_text or not target:
            continue

        task_id = (
            str(row.get("id") or row.get("ID") or "").strip()
            or f"stateval_research_{idx:05d}"
        )
        tasks.append({"id": task_id, "input": input_text, "target": target})

    if not tasks:
        return []

    domain = (spec.domain or "math").strip().lower() or "math"
    dataset = EvalDataset(
        area_id=spec.area_id,
        capability_id="statistical_research",
        capability_name="Statistical Research",
        domain=domain,
        tasks=tasks,
        num_tasks=len(tasks),
        prompt_template=DEFAULT_EVAL_PROMPT_TEMPLATE,
    )
    return [dataset]
