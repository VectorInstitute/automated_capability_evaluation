"""Adapter for the hoskinson-center/proofnet static benchmark.

Dataset card: https://huggingface.co/datasets/hoskinson-center/proofnet

ProofNet is a benchmark for autoformalization and formal proving of undergraduate
mathematics. Uses the "plain_text" config. Splits: validation (185), test (186).

Columns: id, nl_statement (natural language theorem), nl_proof (natural language
proof in LaTeX), formal_statement (Lean 3), src_header.
We use nl_statement as input and nl_proof as target.
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


def _iter_proofnet_samples(
    split: str,
    limit: int | None,
) -> Iterable[Dict[str, Any]]:
    """Yield rows from hoskinson-center/proofnet plain_text in order."""
    ds = load_dataset(
        "hoskinson-center/proofnet",
        "plain_text",
        split=split,
    )
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))
    yield from ds


def build_eval_datasets_from_proofnet(spec: StaticBenchmarkSpec) -> List[EvalDataset]:
    """Convert ProofNet into a single EvalDataset.

    - input: nl_statement (natural language theorem statement)
    - target: nl_proof (natural language proof)
    - domain: math
    - capability_id: proofnet
    Rows with empty nl_proof are skipped.
    """
    tasks: List[Dict[str, str]] = []
    id_counts: Dict[str, int] = {}

    for idx, row in enumerate(_iter_proofnet_samples(spec.split, spec.limit)):
        nl_statement = str(row.get("nl_statement", "")).strip()
        nl_proof = str(row.get("nl_proof", "")).strip()
        if not nl_statement or not nl_proof:
            continue

        raw_id = row.get("id")
        base_id = _slugify(str(raw_id).strip()) if raw_id else f"proofnet_{idx:04d}"
        cnt = id_counts.get(base_id, 0)
        id_counts[base_id] = cnt + 1
        task_id = base_id if cnt == 0 else f"{base_id}_{cnt}"

        tasks.append(
            {"id": task_id, "input": nl_statement, "target": nl_proof}
        )

    if not tasks:
        return []

    dataset = EvalDataset(
        area_id=spec.area_id,
        capability_id="proofnet",
        capability_name="ProofNet",
        domain="math",
        tasks=tasks,
        num_tasks=len(tasks),
        prompt_template=DEFAULT_EVAL_PROMPT_TEMPLATE,
    )
    return [dataset]
