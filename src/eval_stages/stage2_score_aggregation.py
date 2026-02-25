"""Eval Stage 2: Score Aggregation.

This stage computes final capability scores from raw Inspect results.
No LLM calls, just aggregation of results from Stage 1.

See: https://inspect.aisi.org.uk/
"""

import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from inspect_ai.log import read_eval_log
from omegaconf import DictConfig

from src.schemas.eval_io_utils import (
    load_eval_config,
    load_eval_dataset,
    save_capability_scores,
)
from src.schemas.eval_schemas import CapabilityScore


logger = logging.getLogger(__name__)


def _find_result_dirs(results_dir: Path, subject_llm: str) -> List[Path]:
    """Return capability result directories for one subject model."""
    llm_results_dir = results_dir / subject_llm
    if not llm_results_dir.exists():
        return []

    # Find all directories with structure: <area_id>/<capability_id>/
    result_dirs = []
    for area_dir in sorted(llm_results_dir.iterdir()):
        if area_dir.is_dir():
            for cap_dir in sorted(area_dir.iterdir()):
                if cap_dir.is_dir():
                    result_dirs.append(cap_dir)
    return result_dirs


def _find_inspect_logs(result_dir: Path) -> List[Path]:
    """Find Inspect JSON log files for a capability result directory."""
    return sorted(result_dir.glob("*.json"))


def _compute_stats(scores: List[float]) -> Dict[str, Any]:
    """Compute mean, standard error, and sample count."""
    if not scores:
        return {"mean": 0.0, "std_err": 0.0, "num_tasks": 0}

    n = len(scores)
    mean = sum(scores) / n

    if n > 1:
        variance = sum((s - mean) ** 2 for s in scores) / (n - 1)
        std_dev = math.sqrt(variance)
        std_err = std_dev / math.sqrt(n)
    else:
        std_err = 0.0

    return {"mean": mean, "std_err": std_err, "num_tasks": n}


def _score_value_to_float(value: object) -> Optional[float]:
    """Convert a score value to float when possible."""
    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, str):
        upper = value.strip().upper()
        if upper == "C":
            return 1.0
        if upper == "I":
            return 0.0
        try:
            return float(value)
        except ValueError:
            return None

    return None


def _extract_scores_from_log(log_file: Path) -> Dict[str, float]:
    """Extract one score per sample ID from a single Inspect log file."""
    scores: Dict[str, float] = {}
    log = read_eval_log(str(log_file))

    if not log.samples:
        return scores

    for sample in log.samples:
        sample_id = str(getattr(sample, "id", ""))
        if not sample_id or not sample.scores:
            continue

        # Count at most one score per sample to avoid duplicating across scorers.
        for score_obj in sample.scores.values():
            score_value = _score_value_to_float(getattr(score_obj, "value", None))
            if score_value is not None:
                scores[sample_id] = score_value
                break

    return scores


def _parse_inspect_logs(
    result_dir: Path, expected_task_ids: Set[str]
) -> Dict[str, Any]:
    """Parse logs and return stats for the best-matching retry log."""
    # Find Inspect log files (.json)
    log_files = _find_inspect_logs(result_dir)

    if not log_files:
        logger.warning("No log files found in %s", result_dir)
        return {"mean": 0.0, "std_err": 0.0, "num_tasks": 0}

    log_scores: List[Tuple[Path, List[float], Set[str]]] = []
    for log_file in log_files:
        try:
            scored_by_id = _extract_scores_from_log(log_file)
            scored_ids = set(scored_by_id.keys())
            matched_scores = [
                scored_by_id[task_id]
                for task_id in expected_task_ids
                if task_id in scored_by_id
            ]
            log_scores.append((log_file, matched_scores, scored_ids))
        except Exception as e:
            logger.warning("Failed to parse log %s: %s", log_file, e)
            continue

    if not log_scores:
        return {"mean": 0.0, "std_err": 0.0, "num_tasks": 0, "exact_match": False}

    # If multiple logs exist, prefer exact task-id match, then best coverage.
    # This avoids double-counting retries in the same capability directory.
    selected_log, selected_scores, selected_ids = max(
        log_scores,
        key=lambda x: (
            x[2] == expected_task_ids,
            len(x[1]),
            x[0].stat().st_mtime,
            x[0].name,
        ),
    )

    if len(log_scores) > 1:
        logger.info(
            "Multiple logs found in %s; selected %s with %d scored samples",
            result_dir,
            selected_log.name,
            len(selected_scores),
        )

    stats = _compute_stats(selected_scores)
    stats["exact_match"] = selected_ids == expected_task_ids
    return stats


def run_eval_stage2(
    cfg: DictConfig,
    eval_tag: str,
) -> str:
    """Run Stage 2 score aggregation and return eval_tag."""
    # Derive paths from config
    exp_id = cfg.exp_cfg.exp_id
    output_base_dir = Path(cfg.global_cfg.output_dir)
    experiment_dir = output_base_dir / exp_id
    results_dir = experiment_dir / "eval" / "results" / eval_tag

    # Load eval config from Stage 1
    eval_config_path = results_dir / "eval_config.json"
    if not eval_config_path.exists():
        raise ValueError(
            f"eval_config.json not found at {eval_config_path}. Run Stage 1 first."
        )
    eval_config, _ = load_eval_config(eval_config_path)

    logger.info("Eval Stage 2: Aggregating scores (eval_tag=%s)", eval_tag)

    # Find datasets (saved under validation_tag)
    validation_tag = eval_config.validation_tag
    datasets_dir = experiment_dir / "eval" / "datasets" / validation_tag

    scores_dir = experiment_dir / "eval" / "scores" / eval_tag

    # Load datasets for capability info
    dataset_map = {}  # (area_id, cap_id) -> EvalDataset
    for dataset_path in sorted(datasets_dir.rglob("dataset.json")):
        dataset = load_eval_dataset(dataset_path)
        dataset_map[(dataset.area_id, dataset.capability_id)] = dataset

    if not dataset_map:
        raise ValueError(
            f"No datasets found in {datasets_dir}. Run Eval Stage 0 first."
        )

    num_llms_processed = 0

    for llm_config in eval_config.subject_llms:
        llm_name = llm_config["name"]
        logger.info("  Processing results for %s", llm_name)

        # Find all result directories for this LLM
        result_dirs = _find_result_dirs(results_dir, llm_name)

        if not result_dirs:
            logger.warning("  No results found for %s", llm_name)
            continue

        capability_scores = []

        for result_dir in result_dirs:
            # Extract area_id and capability_id from path
            cap_id = result_dir.name
            area_id = result_dir.parent.name

            # Get capability info from dataset
            cap_dataset = dataset_map.get((area_id, cap_id))
            if cap_dataset is None:
                logger.warning(
                    "  No dataset found for %s/%s, skipping",
                    area_id,
                    cap_id,
                )
                continue

            expected_task_ids = {str(task["id"]) for task in cap_dataset.tasks}

            # Parse Inspect logs
            parsed = _parse_inspect_logs(result_dir, expected_task_ids)

            if parsed["num_tasks"] < cap_dataset.num_tasks:
                logger.warning(
                    "  Incomplete scoring for %s/%s with %s: %d/%d tasks scored",
                    area_id,
                    cap_id,
                    llm_name,
                    parsed["num_tasks"],
                    cap_dataset.num_tasks,
                )
            elif not parsed.get("exact_match", False):
                logger.warning(
                    "  Task ID mismatch for %s/%s with %s "
                    "(scored task IDs differ from dataset task IDs)",
                    area_id,
                    cap_id,
                    llm_name,
                )

            # Create CapabilityScore
            score = CapabilityScore(
                area_id=area_id,
                capability_id=cap_id,
                capability_name=cap_dataset.capability_name,
                subject_llm=llm_name,
                mean=parsed["mean"],
                std_err=parsed["std_err"],
                num_tasks=parsed["num_tasks"],
            )
            capability_scores.append(score)

        capability_scores.sort(key=lambda s: (s.area_id, s.capability_id))

        # Save scores for this LLM
        if capability_scores:
            scores_path = scores_dir / llm_name / "capability_scores.json"
            save_capability_scores(capability_scores, scores_path)
            logger.info(
                "  Saved %d capability scores for %s",
                len(capability_scores),
                llm_name,
            )
            num_llms_processed += 1

    logger.info(
        "Eval Stage 2: Aggregated scores for %d LLMs",
        num_llms_processed,
    )

    return eval_tag
