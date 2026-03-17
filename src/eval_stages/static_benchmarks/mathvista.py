"""Adapter for the AI4Math/MathVista static benchmark.

Dataset card: https://huggingface.co/datasets/AI4Math/MathVista

This adapter focuses on the labeled ``testmini`` split, which provides
answers for 1,000 examples. The \"test\" split does not expose labels.
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


def _iter_mathvista_samples(
    split: str,
    limit: int | None,
) -> Iterable[Dict[str, Any]]:
    """Yield rows from AI4Math/MathVista in a stable order."""
    ds = load_dataset("AI4Math/MathVista", split=split)
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))
    yield from ds


def build_eval_datasets_from_mathvista(spec: StaticBenchmarkSpec) -> List[EvalDataset]:
    """Convert MathVista into a single EvalDataset.

    We treat:
    - domain: always "math"
    - area_id: taken from spec.area_id (e.g. "math" or "static_benchmarks")
    - capability_id / capability_name: a single capability "mathvista"
      covering all tasks in the chosen split (typically ``testmini``).
    """
    tasks: List[Dict[str, str]] = []

    for idx, row in enumerate(_iter_mathvista_samples(spec.split, spec.limit)):
        # Prefer the curated query prompt if present.
        query = str(row.get("query", "")).strip()
        question = str(row.get("question", "")).strip()
        image_path = str(row.get("image", "")).strip()
        choices = row.get("choices")

        if query:
            input_text = query
        else:
            parts: List[str] = []
            if question:
                parts.append(question)
            if isinstance(choices, list) and choices:
                labeled: List[str] = []
                for i, opt in enumerate(choices):
                    label = chr(ord("A") + i)
                    labeled.append(f"{label}. {str(opt).strip()}")
                parts.append("Options: " + " ".join(labeled))
            input_text = "\n\n".join(parts).strip()

        if image_path:
            input_text = f"{input_text}\n\n[Image path: {image_path}]".strip()

        answer = str(row.get("answer", "")).strip()

        if not input_text or not answer:
            continue

        pid = str(row.get("pid", "")).strip()
        task_id = pid or f"mathvista_{idx:04d}"

        tasks.append({"id": task_id, "input": input_text, "target": answer})

    if not tasks:
        return []

    capability_id = "mathvista"
    capability_name = "MathVista"

    dataset = EvalDataset(
        area_id=spec.area_id,
        capability_id=capability_id,
        capability_name=capability_name,
        domain="math",
        tasks=tasks,
        num_tasks=len(tasks),
        prompt_template=DEFAULT_EVAL_PROMPT_TEMPLATE,
    )
    return [dataset]

