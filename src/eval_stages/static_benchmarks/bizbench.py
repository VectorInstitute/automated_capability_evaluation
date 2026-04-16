"""Adapter for the kensho/bizbench static benchmark.

Dataset card: https://huggingface.co/datasets/kensho/bizbench

Splits: train, test

Columns (per dataset viewer):
- question (str)
- answer (str)
- task (str)
- context (str | None)
- context_type (str)
- options (list)
- program (str | None)

We use:
- input: context (if present) + question + options (if present)
- target: answer
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

from datasets import load_dataset

from src.eval_stages.prompts import DEFAULT_EVAL_PROMPT_TEMPLATE
from src.eval_stages.static_benchmarks.specs import StaticBenchmarkSpec
from src.schemas.eval_schemas import EvalDataset


def _normalize_answer(val: Any) -> str:
    if val is None:
        return ""
    if isinstance(val, dict):
        for key in ("answer", "label", "text", "value"):
            if key in val and val[key] is not None:
                return str(val[key]).strip()
        return str(val).strip()
    return str(val).strip()


def _format_options(options: Any) -> str:
    if not options:
        return ""
    if isinstance(options, list):
        cleaned = [str(o).strip() for o in options if str(o).strip()]
        if not cleaned:
            return ""
        return "\n".join(f"- {o}" for o in cleaned)
    opt = str(options).strip()
    return f"- {opt}" if opt else ""


def _build_input(question: str, context: Any, options: Any) -> str:
    question = question.strip()
    parts: List[str] = []

    ctx = "" if context is None else str(context).strip()
    if ctx:
        parts.append(f"Context:\n{ctx}")

    parts.append(f"Question:\n{question}")

    opts = _format_options(options)
    if opts:
        parts.append(f"Options:\n{opts}")

    return "\n\n".join(parts).strip()


def _iter_bizbench_samples(
    split: str,
    offset: int | None,
    limit: int | None,
    *,
    finknow_only: bool,
) -> Iterable[Dict[str, Any]]:
    ds = load_dataset("kensho/bizbench", split=split)
    n = len(ds)

    start = 0 if offset is None else max(0, int(offset))
    end = None if limit is None else start + int(limit)

    # Apply filtering first, then slice by (offset, limit) over the filtered stream.
    # This keeps the sharding logic consistent with the adapter's filtering.
    kept_rank = 0
    for dataset_idx, row in enumerate(ds):
        if finknow_only:
            task_val = str(row.get("task", "") or "")
            if "finknow" not in task_val.lower():
                continue

        question = str(row.get("question", "")).strip()
        answer_norm = _normalize_answer(row.get("answer"))
        if not question or not answer_norm:
            continue

        if kept_rank < start:
            kept_rank += 1
            continue
        if end is not None and kept_rank >= end:
            break

        row = dict(row)
        row["_global_idx"] = dataset_idx
        yield row
        kept_rank += 1


def build_eval_datasets_from_bizbench(spec: StaticBenchmarkSpec) -> List[EvalDataset]:
    """Convert BizBench into a single EvalDataset."""
    tasks: List[Dict[str, str]] = []

    for local_idx, row in enumerate(
        _iter_bizbench_samples(
            spec.split,
            spec.offset,
            spec.limit,
            finknow_only=spec.finknow_only,
        )
    ):
        question = str(row.get("question", "")).strip()
        raw_answer = row.get("answer")
        answer = _normalize_answer(raw_answer)

        if not question or not answer:
            continue

        inp = _build_input(question, row.get("context"), row.get("options"))
        global_idx = int(row.get("_global_idx", local_idx))
        task_id = f"bizbench_{global_idx:05d}"
        tasks.append({"id": task_id, "input": inp, "target": answer})

    if not tasks:
        return []

    dataset = EvalDataset(
        area_id=spec.area_id,
        capability_id="bizbench",
        capability_name="BizBench",
        domain="finance",
        tasks=tasks,
        num_tasks=len(tasks),
        prompt_template=DEFAULT_EVAL_PROMPT_TEMPLATE,
    )
    return [dataset]

