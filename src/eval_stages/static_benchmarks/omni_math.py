"""Adapter for the KbsdJames/Omni-MATH static benchmark.

Dataset card: https://huggingface.co/datasets/KbsdJames/Omni-MATH

Omni-MATH is an Olympiad-level math benchmark (~4.4k problems). Single split: test.

Columns: domain, difficulty, problem, solution, answer, source.
We use problem as input and answer as target.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

from datasets import load_dataset

from src.eval_stages.prompts import DEFAULT_EVAL_PROMPT_TEMPLATE
from src.eval_stages.static_benchmarks.specs import StaticBenchmarkSpec
from src.schemas.eval_schemas import EvalDataset


def _normalize_answer(val: Any) -> str:
    """Normalize answer to string (dataset may have mixed types)."""
    if val is None:
        return ""
    if isinstance(val, dict):
        for key in ("label", "text", "value"):
            if key in val and val[key] is not None:
                return str(val[key]).strip()
        return str(val).strip()
    return str(val).strip()


def _iter_omni_math_samples(
    split: str,
    limit: int | None,
) -> Iterable[Dict[str, Any]]:
    """Yield rows from KbsdJames/Omni-MATH in order."""
    ds = load_dataset("KbsdJames/Omni-MATH", split=split)
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))
    yield from ds


def build_eval_datasets_from_omni_math(spec: StaticBenchmarkSpec) -> List[EvalDataset]:
    """Convert Omni-MATH into a single EvalDataset.

    - input: problem (Olympiad-level math problem text)
    - target: answer (normalized to string)
    - domain: math
    - capability_id: omni_math
    Rows with empty problem or answer are skipped.
    """
    tasks: List[Dict[str, str]] = []

    for idx, row in enumerate(_iter_omni_math_samples(spec.split, spec.limit)):
        problem = str(row.get("problem", "")).strip()
        raw_answer = row.get("answer")
        answer = _normalize_answer(raw_answer)
        if not problem or not answer:
            continue

        task_id = f"omni_math_{idx:05d}"
        tasks.append({"id": task_id, "input": problem, "target": answer})

    if not tasks:
        return []

    dataset = EvalDataset(
        area_id=spec.area_id,
        capability_id="omni_math",
        capability_name="Omni-MATH",
        domain="math",
        tasks=tasks,
        num_tasks=len(tasks),
        prompt_template=DEFAULT_EVAL_PROMPT_TEMPLATE,
    )
    return [dataset]
