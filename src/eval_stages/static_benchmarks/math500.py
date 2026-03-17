"""Adapter for the HuggingFaceH4/MATH-500 static benchmark.

Dataset card: https://huggingface.co/datasets/HuggingFaceH4/MATH-500
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List

from datasets import load_dataset

from src.eval_stages.prompts import DEFAULT_EVAL_PROMPT_TEMPLATE
from src.schemas.eval_schemas import EvalDataset
from src.eval_stages.static_benchmarks.specs import StaticBenchmarkSpec


def _slugify(text: str) -> str:
    """Convert arbitrary strings into safe directory-friendly IDs."""
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip()).strip("_").lower()
    return cleaned or "unknown"


def _iter_math500_samples(
    split: str,
    limit: int | None,
) -> Iterable[Dict[str, Any]]:
    """Yield rows from HuggingFaceH4/MATH-500 in a stable order."""
    ds = load_dataset("HuggingFaceH4/MATH-500", split=split)
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))
    yield from ds


def build_eval_datasets_from_math500(spec: StaticBenchmarkSpec) -> List[EvalDataset]:
    """Convert HF MATH-500 into one EvalDataset per subject.

    We treat:
    - domain: always "math"
    - area_id: taken from spec.area_id (e.g. "math" or "static_benchmarks")
    - capability_id / capability_name: derived from the dataset "subject" column
      (Prealgebra, Algebra, Geometry, ...).
    """
    by_subject: Dict[str, List[Dict[str, str]]] = {}

    for idx, row in enumerate(_iter_math500_samples(spec.split, spec.limit)):
        problem = str(row.get("problem", "")).strip()
        answer = str(row.get("answer", "")).strip()
        unique_id = row.get("unique_id")
        task_id = str(unique_id).strip() if unique_id else f"math500_{idx:04d}"
        subject = str(row.get("subject", "")).strip() or "unknown"

        if not problem:
            continue

        by_subject.setdefault(subject, []).append(
            {"id": task_id, "input": problem, "target": answer}
        )

    datasets: List[EvalDataset] = []
    for subject, tasks in sorted(by_subject.items()):
        capability_id = _slugify(subject)
        capability_name = subject

        dataset = EvalDataset(
            area_id=spec.area_id,
            capability_id=capability_id,
            capability_name=capability_name,
            domain="math",
            tasks=tasks,
            num_tasks=len(tasks),
            prompt_template=DEFAULT_EVAL_PROMPT_TEMPLATE,
        )
        datasets.append(dataset)

    return datasets


