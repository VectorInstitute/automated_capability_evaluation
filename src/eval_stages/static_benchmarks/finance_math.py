"""Adapter for the yale-nlp/FinanceMath static benchmark.

Dataset card: https://huggingface.co/datasets/yale-nlp/FinanceMath

FinanceMath is a finance-domain math reasoning benchmark with two splits:
- validation: 200 examples with answers
- test: 1000 examples (answers not publicly released)

We expect to use the validation split for evaluation.

Fields:
- question_id: string
- question: problem text
- tables: list of markdown tables (strings)
- python_solution: expert solution code (ignored here)
- ground_truth: float, executed result rounded to 3 decimals
- topic: financial area (ignored here)

We use (tables + question) as input and ground_truth (string) as target.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

from datasets import load_dataset

from src.eval_stages.prompts import DEFAULT_EVAL_PROMPT_TEMPLATE
from src.eval_stages.static_benchmarks.specs import StaticBenchmarkSpec
from src.schemas.eval_schemas import EvalDataset


def _normalize_answer(val: Any) -> str:
    """Normalize answer to string."""
    if val is None:
        return ""
    if isinstance(val, dict):
        for key in ("ground_truth", "value", "answer"):
            if key in val and val[key] is not None:
                return str(val[key]).strip()
        return str(val).strip()
    # For floats, preserve the dataset's rounding (usually 3 decimals).
    return str(val).strip()


def _build_input(question: str, tables: Any) -> str:
    """Construct model input from question plus optional markdown tables."""
    question = question.strip()
    if not tables:
        return question

    # tables is a list of markdown strings according to the dataset card.
    if isinstance(tables, list):
        tables_str = "\n\n".join(str(t).strip() for t in tables if str(t).strip())
    else:
        tables_str = str(tables).strip()

    if not tables_str:
        return question

    return f"Tables:\n{tables_str}\n\nQuestion:\n{question}"


def _iter_finance_math_samples(
    split: str,
    offset: int | None,
    limit: int | None,
) -> Iterable[Dict[str, Any]]:
    """Yield rows from yale-nlp/FinanceMath in order."""
    ds = load_dataset("yale-nlp/FinanceMath", split=split)
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


def build_eval_datasets_from_finance_math(
    spec: StaticBenchmarkSpec,
) -> List[EvalDataset]:
    """Convert FinanceMath into a single EvalDataset.

    - input: tables (markdown) + question text
    - target: ground_truth (normalized to string)
    - domain: math
    - capability_id: finance_math
    Rows with missing question or ground_truth are skipped.
    """
    tasks: List[Dict[str, str]] = []

    for local_idx, row in enumerate(
        _iter_finance_math_samples(spec.split, spec.offset, spec.limit)
    ):
        question = str(row.get("question", "")).strip()
        tables = row.get("tables")
        raw_answer = row.get("ground_truth")
        answer = _normalize_answer(raw_answer)

        # Skip table-based questions entirely.
        # The dataset uses `tables` as a list; we only keep rows where it's empty.
        if isinstance(tables, list) and len(tables) > 0:
            continue
        if tables not in (None, [], ""):
            # Defensive: if tables is any non-empty structure/string, skip.
            if str(tables).strip():
                continue

        if not question or not answer:
            continue

        inp = question
        global_idx = (spec.offset or 0) + local_idx
        task_id = f"finance_math_{global_idx:05d}"
        tasks.append({"id": task_id, "input": inp, "target": answer})

    if not tasks:
        return []

    dataset = EvalDataset(
        area_id=spec.area_id,
        capability_id="finance_math",
        capability_name="FinanceMath",
        domain="math",
        tasks=tasks,
        num_tasks=len(tasks),
        prompt_template=DEFAULT_EVAL_PROMPT_TEMPLATE,
    )
    return [dataset]

