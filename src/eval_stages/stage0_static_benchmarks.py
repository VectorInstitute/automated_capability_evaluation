"""Eval Stage 0_static: Static benchmark ingestion.

This stage lets you reuse Eval Stages 1 and 2 on external/static benchmarks
(e.g., Hugging Face datasets) that do not originate from this repo's
generation/validation pipeline.

It converts a benchmark-specific schema into the pipeline's EvalDataset JSON
format and writes outputs under:

    <output_dir>/<exp_id>/eval/datasets/<validation_tag>/

so that Stage 1 can run unchanged (it only needs eval_config.json plus one or
more dataset.json files).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List

from omegaconf import DictConfig, OmegaConf

from src.eval_stages.static_benchmarks.mathvista import (
    build_eval_datasets_from_mathvista,
)
from src.eval_stages.static_benchmarks.finance_math import (
    build_eval_datasets_from_finance_math,
)
from src.eval_stages.static_benchmarks.bizbench import (
    build_eval_datasets_from_bizbench,
)
from src.eval_stages.static_benchmarks.finance_tasks import (
    build_eval_datasets_from_finance_tasks,
)
from src.eval_stages.static_benchmarks.xfinbench import (
    build_eval_datasets_from_xfinbench,
)
from src.eval_stages.static_benchmarks.specs import StaticBenchmarkSpec
from src.schemas.eval_io_utils import save_eval_config, save_eval_dataset
from src.schemas.eval_schemas import EvalConfig, EvalDataset
from src.schemas.metadata_schemas import PipelineMetadata
from src.utils.timestamp_utils import iso_timestamp


logger = logging.getLogger(__name__)


def _slugify(text: str) -> str:
    """Convert arbitrary strings into safe directory-friendly IDs."""
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip()).strip("_").lower()
    return cleaned or "unknown"


def _build_datasets_from_spec(spec: StaticBenchmarkSpec) -> List[EvalDataset]:
    """Dispatch to the appropriate adapter based on benchmark_id.

    Returns a list of EvalDataset objects so that one static benchmark can
    produce multiple capabilities if desired.
    """
    bid = spec.benchmark_id.strip()
    if bid in {"AI4Math/MathVista", "MathVista", "mathvista"}:
        return build_eval_datasets_from_mathvista(spec)
    if bid in {"yale-nlp/FinanceMath", "FinanceMath", "finance_math"}:
        return build_eval_datasets_from_finance_math(spec)
    if bid in {"kensho/bizbench", "BizBench", "bizbench"}:
        return build_eval_datasets_from_bizbench(spec)
    if bid in {"Zhihan/XFinBench", "XFinBench", "xfinbench"}:
        return build_eval_datasets_from_xfinbench(spec)
    if bid in {
        "finance_tasks",
        "FinanceTasks",
        "finance_tasks.json",
        "local_finance_tasks",
    } or bid.endswith(".json"):
        # If a user points benchmark_id to a local JSON path, we ingest it here.
        # This is intentionally permissive for local workflows.
        return build_eval_datasets_from_finance_tasks(spec)
    raise ValueError(f"Unknown static benchmark_id: {spec.benchmark_id}")


def run_eval_stage0_static(cfg: DictConfig, validation_tag: str) -> None:
    """Prepare eval datasets/config from a static benchmark."""
    exp_id = cfg.exp_cfg.exp_id
    output_base_dir = Path(cfg.global_cfg.output_dir)
    experiment_dir = output_base_dir / exp_id
    eval_cfg: Dict[str, Any] = cfg.get("eval_cfg", {})

    static_cfg: Dict[str, Any] = cfg.get("static_benchmark_cfg", {})
    benchmark_id = static_cfg.get("benchmark_id")
    if not benchmark_id:
        raise ValueError(
            "static_benchmark_cfg.benchmark_id is required for stage=0_static "
            "(e.g. static_benchmark_cfg.benchmark_id=HuggingFaceH4/MATH-500)"
        )

    spec = StaticBenchmarkSpec(
        benchmark_id=str(benchmark_id),
        split=str(static_cfg.get("split", "test")),
        limit=static_cfg.get("limit"),
        offset=static_cfg.get("offset"),
        area_id=str(static_cfg.get("area_id", StaticBenchmarkSpec.area_id)),
        capability_id=static_cfg.get("capability_id"),
        capability_name=static_cfg.get("capability_name"),
        domain=str(static_cfg.get("domain", StaticBenchmarkSpec.domain)),
        exclude_bloom_create=static_cfg.get("exclude_bloom_create", True),
    )

    logger.info(
        "Eval Stage 0_static: exp_id=%s | benchmark_id=%s | split=%s | limit=%s | validation_tag=%s",
        exp_id,
        spec.benchmark_id,
        spec.split,
        spec.limit,
        validation_tag,
    )

    datasets = _build_datasets_from_spec(spec)
    total_tasks = sum(d.num_tasks for d in datasets)
    if total_tasks == 0:
        raise ValueError(f"No tasks created for benchmark: {spec.benchmark_id}")

    datasets_dir = experiment_dir / "eval" / "datasets" / validation_tag
    for dataset in datasets:
        dataset_path = (
            datasets_dir / dataset.area_id / dataset.capability_id / "dataset.json"
        )
        save_eval_dataset(dataset, dataset_path)
        logger.info(
            "Wrote dataset.json with %d tasks to %s",
            dataset.num_tasks,
            dataset_path,
        )

    # Convert Hydra containers to plain Python types for JSON serialization.
    subject_llms_cfg = eval_cfg.get("subject_llms")
    judge_llm_cfg = eval_cfg.get("judge_llm")

    subject_llms = OmegaConf.to_container(subject_llms_cfg, resolve=True) if subject_llms_cfg is not None else []
    judge_llm = OmegaConf.to_container(judge_llm_cfg, resolve=True) if judge_llm_cfg is not None else {}

    eval_config = EvalConfig(
        experiment_id=exp_id,
        eval_tag="",
        subject_llms=subject_llms,
        judge_llm=judge_llm,
        validation_tag=validation_tag,
    )
    metadata = PipelineMetadata(
        experiment_id=exp_id,
        output_base_dir=str(output_base_dir),
        timestamp=iso_timestamp(),
        input_stage_tag=spec.benchmark_id,
        output_stage_tag=None,
        resume=False,
    )
    eval_config_path = datasets_dir / "eval_config.json"
    save_eval_config(eval_config, metadata, eval_config_path)
    logger.info("Wrote eval_config.json to %s", eval_config_path)

