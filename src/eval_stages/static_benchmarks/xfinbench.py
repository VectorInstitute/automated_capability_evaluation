"""Adapter for the Zhihan/XFinBench static benchmark.

Dataset card: https://huggingface.co/datasets/Zhihan/XFinBench

We load from the CSV files inside the repo:
- validation_set.csv
- test_set.csv

Columns (per inspection):
- id: string (e.g. "vali_0", "test_0")
- task: string, question type ("calcu", "mcq", etc.)
- question: str, problem text (often includes tables/LaTeX)
- choice: optional str, options text for MCQs (may contain newlines)
- ground_truth: label/answer; for MCQ it's a letter like "A", for others numeric
- figure: optional, ignored
- fin_capability: capability tag, ignored here
- gold_fin_term_id: term id, ignored here

We use:
- input: question (+ choices if present)
- target: ground_truth normalized to string
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from datasets import load_dataset

from src.eval_stages.prompts import DEFAULT_EVAL_PROMPT_TEMPLATE
from src.eval_stages.static_benchmarks.specs import StaticBenchmarkSpec
from src.schemas.eval_schemas import EvalDataset


def _normalize_target(task: str, val: Any) -> str:
    """Normalize ground_truth to a string target."""
    if val is None:
        return ""

    # Boolean validity questions: map to Yes / No.
    if task == "bool":
        # Accept numeric, bool, and string encodings.
        if isinstance(val, (int, float)):
            return "Yes" if float(val) != 0.0 else "No"
        s = str(val).strip().lower()
        if s in {"1", "1.0", "true", "yes"}:
            return "Yes"
        if s in {"0", "0.0", "false", "no"}:
            return "No"
        # Fallback: pass through.
        return str(val).strip()

    # For MCQ, dataset uses option letter like "A".
    if task == "mcq":
        return str(val).strip()

    # For numeric / calculation tasks, just stringify (preserving decimals).
    if isinstance(val, dict):
        for key in ("ground_truth", "value", "answer"):
            if key in val and val[key] is not None:
                return str(val[key]).strip()
        return str(val).strip()

    return str(val).strip()


def _build_input(task: str, question: str, choice: Any) -> str:
    """Build model input from question plus optional choices."""
    question = str(question or "").strip()

    # For boolean tasks, explicitly request Yes/No.
    if task == "bool":
        if not question:
            return ""
        return f"Answer only 'Yes' or 'No'.\n\nStatement:\n{question}"

    if not choice:
        return question

    choice_text = str(choice).strip()
    if not choice_text:
        return question

    return f"{question}\n\nOptions:\n{choice_text}"


def _iter_xfinbench_samples(
    split: str,
    offset: Optional[int],
    limit: Optional[int],
) -> Iterable[Dict[str, Any]]:
    """Yield rows from Zhihan/XFinBench (CSV-backed) with offset/limit."""
    ds_dict = load_dataset(
        "Zhihan/XFinBench",
        data_files={"validation": "validation_set.csv", "test": "test_set.csv"},
    )
    if split not in ds_dict:
        raise ValueError(f"Unknown XFinBench split: {split}")

    ds = ds_dict[split]
    n = len(ds)

    start = 0 if offset is None else max(0, int(offset))
    if start >= n:
        return iter(())

    if limit is None:
        end = n
    else:
        end = min(start + int(limit), n)

    if start == 0 and end == n:
        yield from ds
        return

    yield from ds.select(range(start, end))


def build_eval_datasets_from_xfinbench(spec: StaticBenchmarkSpec) -> List[EvalDataset]:
    """Convert XFinBench into a single EvalDataset."""
    tasks: List[Dict[str, str]] = []
    offset: int = max(0, int(spec.offset or 0))

    for local_idx, row in enumerate(
        _iter_xfinbench_samples(spec.split, spec.offset, spec.limit)
    ):
        question = str(row.get("question", "")).strip()
        task_type = str(row.get("task", "")).strip()
        choice = row.get("choice")
        figure = row.get("figure")
        raw_gt = row.get("ground_truth")
        target = _normalize_target(task_type, raw_gt)

        # Skip image-based questions (figure present).
        if figure is not None:
            continue

        if not question or not target:
            continue

        inp = _build_input(task_type, question, choice)
        global_idx = offset + local_idx
        task_id = f"xfinbench_{global_idx:05d}"
        tasks.append({"id": task_id, "input": inp, "target": target})

    if not tasks:
        return []

    dataset = EvalDataset(
        area_id=spec.area_id,
        capability_id="xfinbench",
        capability_name="XFinBench",
        domain="finance",
        tasks=tasks,
        num_tasks=len(tasks),
        prompt_template=DEFAULT_EVAL_PROMPT_TEMPLATE,
    )
    return [dataset]

