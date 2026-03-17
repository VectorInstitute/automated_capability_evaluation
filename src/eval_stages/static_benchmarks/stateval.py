"""Unified StatEval benchmark: one benchmark, domain math, two areas.

StatEval has two subsets:
- Foundational Knowledge Dataset (0v01111/StatEval-Foundational-knowledge)
- Statistical Research Dataset (0v01111/StatEval-Statistical-Research)

This module exposes a single benchmark_id (StatEval / stateval) that loads
both and produces two capabilities under one area "stateval" with domain "math":
- foundational_knowledge
- statistical_research
"""

from __future__ import annotations

from typing import List

from src.eval_stages.static_benchmarks.specs import StaticBenchmarkSpec
from src.eval_stages.static_benchmarks.stateval_foundational import (
    build_eval_datasets_from_stateval_foundational,
)
from src.eval_stages.static_benchmarks.stateval_research import (
    build_eval_datasets_from_stateval_research,
)
from src.schemas.eval_schemas import EvalDataset

STATEVAL_AREA_ID = "stateval"
STATEVAL_DOMAIN = "math"


def build_eval_datasets_from_stateval(spec: StaticBenchmarkSpec) -> List[EvalDataset]:
    """Build EvalDatasets for both StatEval subsets with domain=math, area=stateval.

    Returns two datasets: Foundational Knowledge and Statistical Research.
    Uses the same split and limit from spec for each subset.
    """
    stateval_spec = StaticBenchmarkSpec(
        benchmark_id=spec.benchmark_id,
        split=spec.split,
        limit=spec.limit,
        area_id=STATEVAL_AREA_ID,
        capability_id=spec.capability_id,
        capability_name=spec.capability_name,
        domain=STATEVAL_DOMAIN,
    )

    foundational = build_eval_datasets_from_stateval_foundational(stateval_spec)
    research = build_eval_datasets_from_stateval_research(stateval_spec)

    return foundational + research
