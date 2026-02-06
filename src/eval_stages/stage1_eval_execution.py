"""Eval Stage 1: Evaluation Execution.

This stage runs Inspect AI evaluation for each capability with each subject LLM.
Creates a new eval_tag by default, or reuses a provided eval_tag in resume mode.

See: https://inspect.aisi.org.uk/
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Set

from inspect_ai import Task
from inspect_ai import eval as inspect_eval
from inspect_ai import eval_retry as inspect_eval_retry
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.log import read_eval_log
from inspect_ai.scorer import model_graded_fact
from inspect_ai.solver import generate
from omegaconf import DictConfig

from src.schemas.eval_io_utils import (
    load_eval_config,
    load_eval_dataset,
    save_eval_config,
)
from src.schemas.eval_schemas import EvalDataset
from src.schemas.metadata_schemas import PipelineMetadata
from src.utils.timestamp_utils import iso_timestamp, timestamp_tag


logger = logging.getLogger(__name__)


def _find_datasets(datasets_dir: Path) -> List[Path]:
    """Return all Stage 0 dataset files."""
    if not datasets_dir.exists():
        return []
    return sorted(datasets_dir.rglob("dataset.json"))


def _find_inspect_logs(result_dir: Path) -> List[Path]:
    """Find Inspect JSON log files for a capability result directory."""
    return sorted(result_dir.glob("*.json"))


def _score_value_to_float(value: object) -> Optional[float]:
    """Convert an Inspect score value to float when possible."""
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


def _scored_sample_ids_from_log(log: object) -> Set[str]:
    """Return scored sample IDs from a parsed Inspect log object."""
    samples = getattr(log, "samples", None)
    if not samples:
        return set()

    scored_ids: Set[str] = set()
    for sample in samples:
        sample_id = str(getattr(sample, "id", ""))
        sample_scores = getattr(sample, "scores", None)
        if not sample_id or not sample_scores:
            continue

        for score_obj in sample_scores.values():
            if _score_value_to_float(getattr(score_obj, "value", None)) is not None:
                scored_ids.add(sample_id)
                break

    return scored_ids


def _scored_sample_ids(log_file: Path) -> Set[str]:
    """Return sample IDs with at least one interpretable score."""
    try:
        log = read_eval_log(str(log_file))
    except Exception:
        return set()
    return _scored_sample_ids_from_log(log)


def _check_eval_completed(
    results_dir: Path,
    subject_llm: str,
    area_id: str,
    capability_id: str,
    expected_task_ids: Set[str],
) -> bool:
    """Return True if scored task IDs exactly match expected task IDs."""
    if not expected_task_ids:
        return False

    result_dir = results_dir / subject_llm / area_id / capability_id
    if result_dir.exists():
        for log_file in _find_inspect_logs(result_dir):
            if _scored_sample_ids(log_file) == expected_task_ids:
                return True
    return False


def _find_retry_log(
    result_dir: Path,
    expected_task_ids: Set[str],
) -> Optional[Path]:
    """Find the best failed/incomplete log to resume with Inspect eval_retry."""
    if not result_dir.exists():
        return None

    candidates: List[tuple[Path, int]] = []
    for log_file in _find_inspect_logs(result_dir):
        try:
            log = read_eval_log(str(log_file))
        except Exception:
            continue

        scored_ids = _scored_sample_ids_from_log(log)
        if scored_ids == expected_task_ids:
            continue

        status = str(getattr(log, "status", "")).lower()
        invalidated = bool(getattr(log, "invalidated", False))
        is_retryable = invalidated or status in {"started", "error", "cancelled"}
        if is_retryable:
            matched_expected = len(scored_ids & expected_task_ids)
            candidates.append((log_file, matched_expected))

    if not candidates:
        return None

    best_log, _ = max(
        candidates,
        key=lambda item: (item[1], item[0].stat().st_mtime, item[0].name),
    )
    return best_log


def _create_inspect_task(
    dataset: EvalDataset,
    judge_model: str,
) -> "Task":
    """Build an Inspect task for one capability dataset."""
    # Create Inspect samples from our dataset
    samples = [
        Sample(
            input=task["input"],
            target=task["target"],
            id=task["id"],
        )
        for task in dataset.tasks
    ]

    # Create memory dataset
    inspect_dataset = MemoryDataset(samples)

    # Create task with model-graded scoring
    return Task(
        dataset=inspect_dataset,
        solver=generate(),
        scorer=model_graded_fact(model=judge_model),
    )


def _run_inspect_eval(
    dataset: EvalDataset,
    subject_llm: str,
    judge_llm: Dict[str, str],
    output_dir: Path,
) -> bool:
    """Run a fresh Inspect eval for one capability/LLM pair."""
    # Format model names for Inspect (provider/model)
    judge_model = f"{judge_llm['provider']}/{judge_llm['name']}"

    try:
        # Create Inspect task
        task = _create_inspect_task(dataset, judge_model)

        # Run evaluation
        # Inspect saves logs to the specified directory
        output_dir.mkdir(parents=True, exist_ok=True)

        inspect_eval(
            task,
            model=subject_llm,
            log_dir=str(output_dir),
            log_format="json",
        )

        return True

    except Exception as e:
        logger.error(
            "Inspect evaluation failed for %s/%s with %s: %s",
            dataset.area_id,
            dataset.capability_id,
            subject_llm,
            e,
        )
        return False


def _run_inspect_retry(
    retry_log_path: Path,
    output_dir: Path,
) -> bool:
    """Run Inspect eval_retry from a prior failed log."""
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        inspect_eval_retry(
            str(retry_log_path),
            log_dir=str(output_dir),
            log_format="json",
        )
        return True
    except Exception as e:
        logger.error("Inspect eval_retry failed for %s: %s", retry_log_path, e)
        return False


def run_eval_stage1(
    cfg: DictConfig,
    validation_tag: str,
    eval_tag: Optional[str] = None,
) -> str:
    """Run Stage 1 evals and return the eval tag."""
    # Derive paths from config
    exp_id = cfg.exp_cfg.exp_id
    output_base_dir = Path(cfg.global_cfg.output_dir)
    experiment_dir = output_base_dir / exp_id

    # Load eval_config from Stage 0
    datasets_dir = experiment_dir / "eval" / "datasets" / validation_tag
    eval_config_path = datasets_dir / "eval_config.json"
    if not eval_config_path.exists():
        raise ValueError(
            f"eval_config.json not found at {eval_config_path}. Run Stage 0 first."
        )
    eval_config, _ = load_eval_config(eval_config_path)

    # Create eval_tag for this run (or reuse existing one for resume)
    is_resume = eval_tag is not None
    if eval_tag is None:
        eval_tag = timestamp_tag()

    logger.info(
        "Eval Stage 1: Running evaluations (eval_tag=%s, resume=%s)",
        eval_tag,
        is_resume,
    )

    # Find datasets (saved under validation_tag from Stage 0)
    dataset_paths = _find_datasets(datasets_dir)
    logger.info("Found %d datasets", len(dataset_paths))

    if not dataset_paths:
        raise ValueError(f"No datasets found in {datasets_dir}. Run Stage 0 first.")

    # Load datasets
    datasets = [load_eval_dataset(p) for p in dataset_paths]

    # Setup results directory under eval_tag
    eval_dir = experiment_dir / "eval" / "results" / eval_tag
    results_dir = eval_dir

    # Update eval_config with the tag and save it to results dir
    eval_config.eval_tag = eval_tag
    metadata = PipelineMetadata(
        experiment_id=exp_id,
        output_base_dir=str(output_base_dir),
        timestamp=iso_timestamp(),
        input_stage_tag=validation_tag,
        output_stage_tag=eval_tag,
        resume=is_resume,
    )
    results_config_path = eval_dir / "eval_config.json"
    save_eval_config(eval_config, metadata, results_config_path)
    logger.info("Saved eval_config.json to %s", results_config_path)

    # Run evaluations
    subject_llms = eval_config.subject_llms
    judge_llm = eval_config.judge_llm

    num_completed_this_run = 0
    num_skipped_completed = 0
    num_failed = 0
    num_incomplete = 0
    num_resumed = 0
    total_combinations = len(datasets) * len(subject_llms)

    for dataset in datasets:
        expected_task_ids = {str(task["id"]) for task in dataset.tasks}
        for llm_config in subject_llms:
            llm_name = llm_config["name"]
            # Construct full model string: provider/model_name
            subject_model = f"{llm_config['provider']}/{llm_name}"

            # Check if already completed (resume)
            if _check_eval_completed(
                results_dir,
                llm_name,
                dataset.area_id,
                dataset.capability_id,
                expected_task_ids,
            ):
                logger.info(
                    "  Skipping %s/%s with %s (already completed)",
                    dataset.area_id,
                    dataset.capability_id,
                    llm_name,
                )
                num_skipped_completed += 1
                continue

            # Run evaluation
            output_dir = (
                results_dir / llm_name / dataset.area_id / dataset.capability_id
            )

            retry_log = (
                _find_retry_log(output_dir, expected_task_ids) if is_resume else None
            )
            if retry_log is not None:
                logger.info(
                    "  Resuming %s/%s with %s from %s",
                    dataset.area_id,
                    dataset.capability_id,
                    subject_model,
                    retry_log.name,
                )
                success = _run_inspect_retry(
                    retry_log_path=retry_log,
                    output_dir=output_dir,
                )
                num_resumed += 1
            else:
                logger.info(
                    "  Evaluating %s/%s with %s",
                    dataset.area_id,
                    dataset.capability_id,
                    subject_model,
                )

                success = _run_inspect_eval(
                    dataset=dataset,
                    subject_llm=subject_model,
                    judge_llm=judge_llm,
                    output_dir=output_dir,
                )

            if success:
                if _check_eval_completed(
                    results_dir,
                    llm_name,
                    dataset.area_id,
                    dataset.capability_id,
                    expected_task_ids,
                ):
                    num_completed_this_run += 1
                else:
                    logger.warning(
                        "  Incomplete evaluation output for %s/%s with %s "
                        "(task IDs mismatch: missing or extra scored tasks)",
                        dataset.area_id,
                        dataset.capability_id,
                        llm_name,
                    )
                    num_incomplete += 1
            else:
                num_failed += 1

    logger.info(
        "Eval Stage 1 summary: completed_this_run=%d skipped_completed=%d "
        "resumed=%d failed=%d incomplete=%d total=%d",
        num_completed_this_run,
        num_skipped_completed,
        num_resumed,
        num_failed,
        num_incomplete,
        total_combinations,
    )

    return eval_tag
