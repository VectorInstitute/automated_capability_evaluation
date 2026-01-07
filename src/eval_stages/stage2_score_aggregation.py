"""Eval Stage 2: Score Aggregation.

This stage computes final capability scores from raw Inspect results.
No LLM calls, just aggregation of results from Stage 1.

See: https://inspect.aisi.org.uk/
"""

import logging
import math
from pathlib import Path
from typing import Dict, List

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
    """Find all result directories for a subject LLM.

    Args:
        results_dir: Path to results directory
        subject_llm: Subject LLM name

    Returns
    -------
        List of paths to capability result directories
    """
    llm_results_dir = results_dir / subject_llm
    if not llm_results_dir.exists():
        return []

    # Find all directories with structure: <area_id>/<capability_id>/
    result_dirs = []
    for area_dir in llm_results_dir.iterdir():
        if area_dir.is_dir():
            for cap_dir in area_dir.iterdir():
                if cap_dir.is_dir():
                    result_dirs.append(cap_dir)
    return result_dirs


def _compute_stats(scores: List[float]) -> Dict:
    """Compute mean and standard error from scores.

    Args:
        scores: List of score values (0.0 to 1.0)

    Returns
    -------
        Dict with 'mean', 'std_err', 'num_tasks'
    """
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


def _parse_inspect_logs(result_dir: Path) -> Dict:
    """Parse Inspect logs to extract scores.

    Args:
        result_dir: Path to capability result directory

    Returns
    -------
        Dict with 'mean', 'std_err', 'num_tasks'
    """
    # Find Inspect log files (they have .json extension)
    log_files = list(result_dir.glob("*.json"))

    if not log_files:
        logger.warning("No log files found in %s", result_dir)
        return {"mean": 0.0, "std_err": 0.0, "num_tasks": 0}

    scores = []

    for log_file in log_files:
        try:
            log = read_eval_log(str(log_file))

            # Extract scores from samples
            # In Inspect AI 0.3.159+, sample.scores is dict[str, Score] | None
            if log.samples:
                for sample in log.samples:
                    if sample.scores:
                        # Iterate over all scorers (usually just one)
                        for _scorer_name, score_obj in sample.scores.items():
                            if score_obj.value is not None:
                                # Score value can be numeric or string
                                score_val = score_obj.value
                                if isinstance(score_val, (int, float)):
                                    scores.append(float(score_val))
                                elif score_val == "C":  # Correct
                                    scores.append(1.0)
                                elif score_val == "I":  # Incorrect
                                    scores.append(0.0)

        except Exception as e:
            logger.warning("Failed to parse log %s: %s", log_file, e)
            continue

    return _compute_stats(scores)


def run_eval_stage2(
    cfg: DictConfig,
    eval_tag: str,
) -> str:
    """Eval Stage 2: Score Aggregation.

    Computes final capability scores from raw Inspect results.

    Args:
        cfg: Configuration object
        eval_tag: Tag from Eval Stage 1

    Returns
    -------
        The eval_tag (same as input, for chaining)
    """
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
    for dataset_path in datasets_dir.rglob("dataset.json"):
        dataset = load_eval_dataset(dataset_path)
        dataset_map[(dataset.area_id, dataset.capability_id)] = dataset

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
            dataset = dataset_map.get((area_id, cap_id))
            if not dataset:
                logger.warning(
                    "  No dataset found for %s/%s, skipping",
                    area_id,
                    cap_id,
                )
                continue

            # Parse Inspect logs
            parsed = _parse_inspect_logs(result_dir)

            # Create CapabilityScore
            score = CapabilityScore(
                area_id=area_id,
                capability_id=cap_id,
                capability_name=dataset.capability_name,
                subject_llm=llm_name,
                mean=parsed["mean"],
                std_err=parsed["std_err"],
                num_tasks=parsed["num_tasks"],
            )
            capability_scores.append(score)

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
