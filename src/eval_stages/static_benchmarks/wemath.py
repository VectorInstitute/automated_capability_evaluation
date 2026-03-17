"""Adapter for the We-Math/We-Math static benchmark.

Dataset card: https://huggingface.co/datasets/We-Math/We-Math
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


def _iter_wemath_samples(
    split: str,
    limit: int | None,
) -> Iterable[Dict[str, Any]]:
    """Yield rows from We-Math/We-Math in a stable order."""
    # The public config exposes a "testmini" split; callers should pass
    # static_benchmark_cfg.split=testmini in Hydra.
    ds = load_dataset("We-Math/We-Math", split=split)
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))
    yield from ds


def build_eval_datasets_from_wemath(spec: StaticBenchmarkSpec) -> List[EvalDataset]:
    """Convert We-Math into EvalDatasets grouped by knowledge concept.

    We treat:
    - domain: always "math"
    - area_id: taken from spec.area_id (e.g. "math" or "static_benchmarks")
    - capability_id / capability_name: derived from the "knowledge_concept"
      column (e.g., "Properties and Understanding of Squares").

    Each task:
    - input: question text plus options as a single string
    - target: the correct option letter from the "answer" column
    """
    by_concept: Dict[str, List[Dict[str, str]]] = {}
    id_counts: Dict[str, int] = {}

    for idx, row in enumerate(_iter_wemath_samples(spec.split, spec.limit)):
        # Skip questions that have an image; this pipeline is text-only and does not pass images.
        if row.get("image") is not None:
            continue

        question = str(row.get("question", "")).strip()
        options = str(row.get("option", "")).strip()
        answer = str(row.get("answer", "")).strip()
        if not question or not options or not answer:
            continue

        concept = str(row.get("knowledge_concept", "")).strip() or "unknown"
        base_id = str(row.get("ID", "")).strip() or f"wemath_{idx:04d}"
        # Ensure uniqueness of task ids since Inspect requires unique ids.
        cnt = id_counts.get(base_id, 0)
        id_counts[base_id] = cnt + 1
        task_id = base_id if cnt == 0 else f"{base_id}_{cnt}"

        # Pack question and options into a single prompt input.
        input_text = f"{question}\n\nOptions: {options}"

        by_concept.setdefault(concept, []).append(
            {"id": task_id, "input": input_text, "target": answer}
        )

    datasets: List[EvalDataset] = []
    for concept, tasks in sorted(by_concept.items()):
        capability_id = _slugify(concept)
        capability_name = concept

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

