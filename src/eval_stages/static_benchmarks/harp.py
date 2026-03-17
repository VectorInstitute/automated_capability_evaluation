"""Adapter for the HARP static benchmark.

Repository: https://github.com/aadityasingh/HARP

We use the main split HARP.jsonl (short-answer questions).

Fields (from README):
- problem: problem text
- answer: ground truth answer
We ignore other metadata fields.
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
        for key in ("label", "text", "value", "answer"):
            if key in val and val[key] is not None:
                return str(val[key]).strip()
        return str(val).strip()
    return str(val).strip()


def _iter_harp_samples(
    split: str,
    limit: int | None,
) -> Iterable[Dict[str, Any]]:
    """Yield rows from HARP.jsonl (single split).

    The dataset is hosted in the GitHub repo as a JSONL file in a zip archive.
    We load it via datasets.load_dataset with a remote URL.
    """
    # Main short-answer split; datasets can read compressed JSONL directly.
    data_files = "https://github.com/aadityasingh/HARP/raw/main/HARP.jsonl.zip"
    ds = load_dataset("json", data_files=data_files, split="train")

    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))
    yield from ds


def build_eval_datasets_from_harp(spec: StaticBenchmarkSpec) -> List[EvalDataset]:
    """Convert HARP into a single EvalDataset.

    - input: problem (competition math problem text)
    - target: answer (normalized to string)
    - domain: math
    - capability_id: harp
    Rows with empty problem or answer are skipped.
    """
    tasks: List[Dict[str, str]] = []

    for idx, row in enumerate(_iter_harp_samples(spec.split, spec.limit)):
        problem = str(row.get("problem", "")).strip()
        raw_answer = row.get("answer")
        answer = _normalize_answer(raw_answer)
        if not problem or not answer:
            continue

        task_id = f"harp_{idx:05d}"
        tasks.append({"id": task_id, "input": problem, "target": answer})

    if not tasks:
        return []

    dataset = EvalDataset(
        area_id=spec.area_id,
        capability_id="harp",
        capability_name="HARP",
        domain="math",
        tasks=tasks,
        num_tasks=len(tasks),
        prompt_template=DEFAULT_EVAL_PROMPT_TEMPLATE,
    )
    return [dataset]

