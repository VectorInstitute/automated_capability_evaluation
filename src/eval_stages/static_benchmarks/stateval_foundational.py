"""Adapter for the StatEval Foundational Knowledge static benchmark.

Primary dataset:
- https://huggingface.co/datasets/0v01111/StatEval-Foundational-knowledge

The upstream JSON has mixed types for the "answer" column (string vs object),
which breaks the default HuggingFace loader. We try load_dataset first, then
fall back to downloading raw JSON and parsing with per-row normalization.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

from datasets import load_dataset
from datasets.exceptions import DatasetGenerationError
from huggingface_hub import hf_hub_download, list_repo_files

from src.eval_stages.prompts import DEFAULT_EVAL_PROMPT_TEMPLATE
from src.eval_stages.static_benchmarks.specs import StaticBenchmarkSpec
from src.schemas.eval_schemas import EvalDataset

REPO_ID = "0v01111/StatEval-Foundational-knowledge"


def _slugify(text: str) -> str:
    """Convert arbitrary strings into safe directory-friendly IDs."""
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip()).strip("_").lower()
    return cleaned or "unknown"


def _normalize_answer_in_row(row: Dict[str, Any]) -> None:
    """In-place: ensure row['answer'] is a string (upstream has mixed types)."""
    a = row.get("answer")
    if a is None:
        row["answer"] = ""
        return
    if isinstance(a, dict):
        for key in ("label", "text", "final", "value"):
            if key in a and a[key] is not None:
                row["answer"] = str(a[key]).strip()
                return
        row["answer"] = str(a).strip()
        return
    row["answer"] = str(a).strip()


def _load_foundational_raw(split: str, limit: int | None) -> Iterable[Dict[str, Any]]:
    """Load repo JSON manually and yield rows with normalized 'answer'.

    Used when load_dataset fails due to mixed column types in the upstream data.
    """
    files = list_repo_files(REPO_ID, repo_type="dataset")
    # Prefer the canonical Foundational-knowledge.jsonl if present.
    filename = None
    for cand in files:
        if "Foundational-knowledge" in cand:
            filename = cand
            break
    if filename is None:
        # Fallback: any JSON/JSONL file that mentions the split, else any JSON/JSONL.
        json_files = [
            f for f in files if (f.endswith(".json") or f.endswith(".jsonl")) and split in f
        ]
        if not json_files:
            json_files = [f for f in files if f.endswith(".json") or f.endswith(".jsonl")]
        if not json_files:
            return
        filename = json_files[0]

    path = hf_hub_download(
        repo_id=REPO_ID,
        filename=filename,
        repo_type="dataset",
    )
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    rows: List[Dict[str, Any]] = []
    try:
        data = json.loads(text)
        if isinstance(data, list):
            rows = data
        elif isinstance(data, dict) and "data" in data:
            rows = data["data"]
        elif isinstance(data, dict) and data:
            keys = sorted(
                data.keys(),
                key=lambda k: int(k) if isinstance(k, str) and k.isdigit() else k,
            )
            rows = [data[k] for k in keys if isinstance(data[k], dict)]
    except (json.JSONDecodeError, TypeError, ValueError):
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    rows.append(obj)
                elif isinstance(obj, list):
                    rows.extend(obj)
            except json.JSONDecodeError:
                continue
    for i, row in enumerate(rows):
        if limit is not None and i >= limit:
            break
        if not isinstance(row, dict):
            continue
        _normalize_answer_in_row(row)
        yield row


def _iter_stateval_foundational_samples(
    split: str,
    limit: int | None,
) -> Iterable[Dict[str, Any]]:
    """Yield rows from StatEval-Foundational-knowledge in a stable order.

    Always loads raw JSON from the Hub and normalizes 'answer', avoiding
    mixed-type issues in the official JSONL that break load_dataset.
    """
    yield from _load_foundational_raw(split, limit)


def _extract_question_and_options(row: Dict[str, Any]) -> str:
    """Build a human-readable question string from a StatEval row.

    We try several likely field names for the question stem and options to
    make the adapter robust to minor schema variations.
    """
    stem = (
        str(
            row.get("question")
            or row.get("prompt")
            or row.get("task")
            or row.get("problem")
            or ""
        ).strip()
    )

    options = row.get("options") or row.get("choices") or row.get("mc_options")
    options_text = ""
    if isinstance(options, list) and options:
        labeled = []
        for i, opt in enumerate(options):
            label = chr(ord("A") + i)
            labeled.append(f"{label}. {str(opt).strip()}")
        options_text = " ".join(labeled)
    elif isinstance(options, str) and options.strip():
        options_text = options.strip()

    if options_text:
        return f"{stem}\n\nOptions: {options_text}"
    return stem


def _extract_answer(row: Dict[str, Any]) -> str:
    """Extract a compact target answer string from a StatEval row.

    The 'answer' field can be heterogeneous (string or object). We normalize
    it into a single string that can be used as target text.
    """
    answer = row.get("answer")

    # If answer is a mapping, look for common keys first.
    if isinstance(answer, dict):
        for key in ("label", "text", "final", "value"):
            if key in answer and answer[key] is not None:
                return str(answer[key]).strip()
        # Fallback: stringify the whole object.
        return str(answer).strip()

    if answer is None:
        return ""

    return str(answer).strip()


def build_eval_datasets_from_stateval_foundational(
    spec: StaticBenchmarkSpec,
) -> List[EvalDataset]:
    """Convert StatEval Foundational Knowledge into a single EvalDataset.

    All tasks are grouped under one capability "foundational_knowledge".
    Uses spec.area_id and spec.domain (caller sets e.g. area_id=stateval, domain=math).
    """
    tasks: List[Dict[str, str]] = []

    for idx, row in enumerate(
        _iter_stateval_foundational_samples(spec.split, spec.limit)
    ):
        input_text = _extract_question_and_options(row)
        target = _extract_answer(row)
        if not input_text or not target:
            continue

        task_id = (
            str(row.get("id") or row.get("ID") or "").strip()
            or f"stateval_fk_{idx:05d}"
        )
        tasks.append({"id": task_id, "input": input_text, "target": target})

    if not tasks:
        return []

    domain = (spec.domain or "math").strip().lower() or "math"
    dataset = EvalDataset(
        area_id=spec.area_id,
        capability_id="foundational_knowledge",
        capability_name="Foundational Knowledge",
        domain=domain,
        tasks=tasks,
        num_tasks=len(tasks),
        prompt_template=DEFAULT_EVAL_PROMPT_TEMPLATE,
    )
    return [dataset]

