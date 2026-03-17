"""Adapter for the microsoft/orca-math-word-problems-200k static benchmark.

Dataset card: https://huggingface.co/datasets/microsoft/orca-math-word-problems-200k

Columns: question (math word problem), answer (step-by-step solution).
Single split: train. Use +static_benchmark_cfg.split=train when running Stage 0.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

from datasets import load_dataset

from src.eval_stages.prompts import DEFAULT_EVAL_PROMPT_TEMPLATE
from src.eval_stages.static_benchmarks.specs import StaticBenchmarkSpec
from src.schemas.eval_schemas import EvalDataset


def _iter_orca_math_samples(
    split: str,
    limit: int | None,
) -> Iterable[Dict[str, Any]]:
    """Yield rows from microsoft/orca-math-word-problems-200k in order."""
    ds = load_dataset("microsoft/orca-math-word-problems-200k", split=split)
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))
    yield from ds


def build_eval_datasets_from_orca_math(spec: StaticBenchmarkSpec) -> List[EvalDataset]:
    """Convert Orca Math Word Problems into a single EvalDataset.

    - input: question (math word problem text)
    - target: answer (step-by-step solution from the dataset)
    - domain: math
    - capability_id: orca_math_word_problems
    """
    tasks: List[Dict[str, str]] = []

    for idx, row in enumerate(_iter_orca_math_samples(spec.split, spec.limit)):
        question = str(row.get("question", "")).strip()
        answer = str(row.get("answer", "")).strip()
        if not question or not answer:
            continue

        task_id = f"orca_math_{idx:06d}"
        tasks.append({"id": task_id, "input": question, "target": answer})

    if not tasks:
        return []

    dataset = EvalDataset(
        area_id=spec.area_id,
        capability_id="orca_math_word_problems",
        capability_name="Orca Math Word Problems",
        domain="math",
        tasks=tasks,
        num_tasks=len(tasks),
        prompt_template=DEFAULT_EVAL_PROMPT_TEMPLATE,
    )
    return [dataset]
