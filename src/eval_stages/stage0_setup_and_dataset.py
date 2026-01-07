"""Eval Stage 0: Setup and Dataset Preparation.

This stage:
1. Validates that required generation outputs exist
2. Converts validated tasks to Inspect-compatible format

No LLM calls, deterministic transformation. Datasets are saved under
eval/datasets/<validation_tag>/ since they are tied to the validation source.
"""

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from omegaconf import DictConfig

from src.eval_stages.prompts import DEFAULT_EVAL_PROMPT_TEMPLATE
from src.schemas.eval_io_utils import save_eval_dataset
from src.schemas.eval_schemas import EvalConfig, EvalDataset
from src.schemas.validation_schemas import ValidationResult


logger = logging.getLogger(__name__)


class EvalSetupError(Exception):
    """Error during evaluation setup."""

    pass


def _validate_inputs(
    experiment_dir: Path,
    validation_tag: str,
    eval_cfg: dict,
) -> None:
    """Validate all required inputs exist.

    Args:
        experiment_dir: Path to experiment directory
        validation_tag: Tag from generation Stage 5
        eval_cfg: Evaluation config section

    Raises
    ------
        EvalSetupError: If validation fails
    """
    # Check experiment.json exists
    experiment_json = experiment_dir / "experiment.json"
    if not experiment_json.exists():
        raise EvalSetupError(f"Experiment file not found: {experiment_json}")

    # Check validation directory exists
    validation_dir = experiment_dir / "validation" / validation_tag
    if not validation_dir.exists():
        raise EvalSetupError(f"Validation directory not found: {validation_dir}")

    # Check validation files exist
    validation_files = list(validation_dir.rglob("*.json"))
    if not validation_files:
        raise EvalSetupError(f"No validation files found in: {validation_dir}")

    # Check subject_llms configured
    if not eval_cfg.get("subject_llms"):
        raise EvalSetupError("subject_llms must be specified in eval_cfg")

    # Check judge_llm configured
    if not eval_cfg.get("judge_llm"):
        raise EvalSetupError("judge_llm must be specified in eval_cfg")


def _find_validated_tasks(
    experiment_dir: Path, validation_tag: str
) -> List[Tuple[Path, ValidationResult]]:
    """Find all validated tasks (verification=true) for a given tag.

    Args:
        experiment_dir: Path to experiment directory
        validation_tag: Tag from generation Stage 5

    Returns
    -------
        List of (file_path, ValidationResult) tuples for verified tasks
    """
    validation_dir = experiment_dir / "validation" / validation_tag

    validated_tasks = []
    for vf in validation_dir.rglob("*.json"):
        with open(vf, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Skip metadata-only files
        if "verification" not in data:
            continue

        # Only include verified tasks
        if data.get("verification", False):
            validation_result = ValidationResult.from_dict(data)
            validated_tasks.append((vf, validation_result))

    return validated_tasks


def _group_by_capability(
    validated_tasks: List[Tuple[Path, ValidationResult]],
) -> Dict[Tuple[str, str], List[ValidationResult]]:
    """Group validated tasks by capability.

    Args:
        validated_tasks: List of (file_path, ValidationResult) tuples

    Returns
    -------
        Dict mapping (area_id, capability_id) to list of ValidationResults
    """
    grouped = defaultdict(list)
    for _, validation in validated_tasks:
        task_solution = validation.task_solution
        area_id = task_solution.task_obj.capability.area.area_id
        cap_id = task_solution.task_obj.capability.capability_id
        grouped[(area_id, cap_id)].append(validation)
    return grouped


def _create_eval_dataset(
    area_id: str,
    capability_id: str,
    validations: List[ValidationResult],
    prompt_template: str = DEFAULT_EVAL_PROMPT_TEMPLATE,
) -> EvalDataset:
    """Create EvalDataset from validated tasks.

    Args:
        area_id: Area identifier
        capability_id: Capability identifier
        validations: List of ValidationResults for this capability
        prompt_template: Template for formatting task prompts

    Returns
    -------
        EvalDataset dataclass
    """
    # Get capability info from first validation
    first = validations[0]
    capability = first.task_solution.task_obj.capability

    # Build tasks list
    tasks = []
    for v in validations:
        ts = v.task_solution
        tasks.append(
            {
                "id": ts.task_id,
                "input": ts.task,
                "target": ts.solution,
            }
        )

    return EvalDataset(
        area_id=area_id,
        capability_id=capability_id,
        capability_name=capability.name,
        domain=capability.area.domain.name,
        tasks=tasks,
        num_tasks=len(tasks),
        prompt_template=prompt_template,
    )


def run_eval_stage0(
    cfg: DictConfig,
    validation_tag: str,
) -> EvalConfig:
    """Eval Stage 0: Setup and Dataset Preparation.

    Validates inputs and creates datasets for evaluation.

    Args:
        cfg: Configuration object
        validation_tag: Tag from generation Stage 5 (required)

    Returns
    -------
        EvalConfig object for use in subsequent stages

    Raises
    ------
        EvalSetupError: If validation fails
    """
    # Get experiment info from config
    exp_id = cfg.exp_cfg.exp_id
    output_base_dir = Path(cfg.global_cfg.output_dir)
    experiment_dir = output_base_dir / exp_id
    eval_cfg = cfg.get("eval_cfg", {})

    logger.info(
        "Eval Stage 0: exp_id=%s | validation_tag=%s",
        exp_id,
        validation_tag,
    )

    # Validate all inputs
    _validate_inputs(experiment_dir, validation_tag, eval_cfg)
    logger.info("Validation checks passed")

    # Create EvalConfig (no tag yet - that's created in Stage 1)
    eval_config = EvalConfig(
        experiment_id=exp_id,
        eval_tag="",  # Will be set in Stage 1
        subject_llms=eval_cfg.get("subject_llms"),
        judge_llm=eval_cfg.get("judge_llm"),
        validation_tag=validation_tag,
    )

    # Find all validated tasks
    validated_tasks = _find_validated_tasks(experiment_dir, validation_tag)
    logger.info("Found %d validated tasks", len(validated_tasks))

    if not validated_tasks:
        raise EvalSetupError(
            f"No validated tasks (verification=true) found in: {validation_tag}"
        )

    # Group by capability
    grouped = _group_by_capability(validated_tasks)
    logger.info("Found %d capabilities with validated tasks", len(grouped))

    # Create and save datasets (tied to validation_tag, not eval_tag)
    datasets_dir = experiment_dir / "eval" / "datasets" / validation_tag
    num_created = 0

    for (area_id, cap_id), validations in grouped.items():
        # Check if dataset already exists (idempotent)
        dataset_path = datasets_dir / area_id / cap_id / "dataset.json"
        if dataset_path.exists():
            logger.info("  Skipping %s/%s (already exists)", area_id, cap_id)
            continue

        # Create dataset
        dataset = _create_eval_dataset(area_id, cap_id, validations)

        # Save dataset
        save_eval_dataset(dataset, dataset_path)
        logger.info(
            "  Created dataset for %s/%s (%d tasks)",
            area_id,
            cap_id,
            dataset.num_tasks,
        )
        num_created += 1

    logger.info(
        "Eval Stage 0: Created %d datasets in %s",
        num_created,
        datasets_dir,
    )

    return eval_config
