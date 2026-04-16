"""Shared specs and types for static benchmark adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class StaticBenchmarkSpec:
    """Minimal specification for ingesting a static benchmark.

    Attributes
    ----------
    benchmark_id
        Identifier used to select the adapter (e.g. "HuggingFaceH4/MATH-500").
    split
        Data split to load (e.g. "train", "test", "validation").
    limit
        Optional maximum number of rows to load (for smoke tests).
    area_id
        Area identifier used in EvalDataset (groups capabilities).
    capability_id
        Capability identifier in EvalDataset; if omitted, adapters may derive it.
    capability_name
        Human-readable capability name; if omitted, adapters may derive it.
    domain
        Domain label for EvalDataset (e.g. "math", "external").
    +    offset
    +        Optional starting index for slicing the underlying dataset. Used
    +        together with `limit` to support chunked / array-style evaluation.
    +    """

    benchmark_id: str
    split: str = "test"
    limit: Optional[int] = None
    offset: Optional[int] = None
    area_id: str = "static_benchmarks"
    capability_id: Optional[str] = None
    capability_name: Optional[str] = None
    domain: str = "external"

    # Controls the filtering behavior for local "finance tasks" JSON adapters
    # (e.g. finance_tasks.json / seed_tasks.json).
    # When true, removes Bloom level "Create - Combine elements..." tasks.
    exclude_bloom_create: bool = True

    # Controls whether BizBench ingestion should keep only the FinKnow subset.
    # When true, filters rows where the dataset's `task` field indicates FinKnow.
    finknow_only: bool = True

