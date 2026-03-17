"""Adapter for the HARDMath static benchmark.

Source JSON:
- https://github.com/sarahmart/HARDMath/blob/main/data/HARDMath.json
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List
from urllib.request import urlopen

from src.eval_stages.prompts import DEFAULT_EVAL_PROMPT_TEMPLATE
from src.eval_stages.static_benchmarks.specs import StaticBenchmarkSpec
from src.schemas.eval_schemas import EvalDataset


HARDMATH_URL = (
    "https://raw.githubusercontent.com/sarahmart/HARDMath/main/data/HARDMath.json"
)


def _slugify(text: str) -> str:
    """Convert arbitrary strings into safe directory-friendly IDs."""
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip()).strip("_").lower()
    return cleaned or "unknown"


def _iter_hardmath_samples(limit: int | None) -> Iterable[Dict[str, Any]]:
    """Yield rows from HARDMath JSON in a stable order."""
    with urlopen(HARDMATH_URL) as f:
        data = json.load(f)

    # The JSON is a dict keyed by string indices ("0", "1", ...).
    # We iterate in key-sorted order for reproducibility.
    items = [data[k] for k in sorted(data.keys(), key=lambda x: int(x))]

    if limit is not None:
        items = items[: max(0, min(limit, len(items)))]

    yield from items


def build_eval_datasets_from_hardmath(spec: StaticBenchmarkSpec) -> List[EvalDataset]:
    """Convert HARDMath into one EvalDataset per question_type.

    We treat:
    - domain: always "math"
    - area_id: taken from spec.area_id (e.g. "math" or "static_benchmarks")
    - capability_id / capability_name: derived from the "question_type" field
      (e.g., "integral", "ODE", "polynomial_roots_corrections", ...).
    """
    by_qtype: Dict[str, List[Dict[str, str]]] = {}

    for idx, row in enumerate(_iter_hardmath_samples(spec.limit)):
        question = str(row.get("question", "")).strip()
        # Use the curated LaTeX-like final answer field.
        answer = str(row.get("answer_val", "")).strip()
        if not question or not answer:
            continue

        qtype = str(row.get("question_type", "")).strip() or "unknown"
        answer_type = str(row.get("answer_type", "")).strip()
        precision = row.get("precision")

        # Enrich the input with answer-type instructions so the subject model
        # knows what form is expected.
        extra_lines = []
        if answer_type:
            if answer_type == "list":
                extra_lines.append(
                    "Answer format: provide a Python-style list of expressions or numbers, in the order requested."
                )
            elif answer_type in {"integer", "float"}:
                extra_lines.append(
                    f"Answer format: a single {answer_type} value (no explanation in the final line)."
                )
            elif answer_type == "math_expression":
                extra_lines.append(
                    "Answer format: a single closed-form mathematical expression."
                )
            else:
                extra_lines.append(f"Answer format: {answer_type}.")

        if isinstance(precision, (int, float)):
            extra_lines.append(
                f"If the answer is numeric, round to {int(precision)} decimal places."
            )

        if extra_lines:
            input_text = question + "\n\n" + "\n".join(extra_lines)
        else:
            input_text = question

        task_id = f"hardmath_{idx:04d}"

        by_qtype.setdefault(qtype, []).append(
            {"id": task_id, "input": input_text, "target": answer}
        )

    datasets: List[EvalDataset] = []
    for qtype, tasks in sorted(by_qtype.items()):
        capability_id = _slugify(qtype)
        capability_name = qtype

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

