"""Eval Stage 1: Evaluation Execution.

This stage runs Inspect AI evaluation for each capability with each subject LLM.
Creates eval_tag for this evaluation run since this is where LLM calls happen.

See: https://inspect.aisi.org.uk/
"""

import logging
from pathlib import Path
from typing import List

from inspect_ai import Task
from inspect_ai import eval as inspect_eval
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import model_graded_fact
from inspect_ai.solver import generate
from omegaconf import DictConfig

from src.schemas.eval_io_utils import load_eval_dataset, save_eval_config
from src.schemas.eval_schemas import EvalConfig, EvalDataset
from src.schemas.metadata_schemas import PipelineMetadata
from src.utils.timestamp_utils import iso_timestamp, timestamp_tag


logger = logging.getLogger(__name__)


def _find_datasets(datasets_dir: Path) -> List[Path]:
    """Find all dataset files.

    Args:
        datasets_dir: Path to datasets directory

    Returns
    -------
        List of paths to dataset.json files
    """
    if not datasets_dir.exists():
        return []
    return list(datasets_dir.rglob("dataset.json"))


def _check_eval_completed(
    results_dir: Path, subject_llm: str, area_id: str, capability_id: str
) -> bool:
    """Check if evaluation was already completed for this combination.

    Args:
        results_dir: Path to results directory
        subject_llm: Subject LLM name
        area_id: Area identifier
        capability_id: Capability identifier

    Returns
    -------
        True if evaluation results exist
    """
    result_dir = results_dir / subject_llm / area_id / capability_id
    # Check if directory exists and has any log files
    if result_dir.exists():
        log_files = list(result_dir.glob("*.json"))
        return len(log_files) > 0
    return False


def _create_inspect_task(
    dataset: EvalDataset,
    judge_model: str,
) -> "Task":
    """Create an Inspect Task from EvalDataset.

    Args:
        dataset: EvalDataset with tasks
        judge_model: Model to use for grading (e.g., "openai/gpt-4o-mini")

    Returns
    -------
        Inspect Task object
    """
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
    judge_llm: dict,
    output_dir: Path,
) -> bool:
    """Run Inspect evaluation for a single capability/LLM combination.

    Args:
        dataset: EvalDataset to evaluate
        subject_llm: Subject LLM (e.g., "openai/gpt-4o")
        judge_llm: Judge LLM config dict
        output_dir: Directory to save Inspect logs

    Returns
    -------
        True if evaluation succeeded
    """
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


def run_eval_stage1(
    cfg: DictConfig,
    eval_config: EvalConfig,
) -> str:
    """Eval Stage 1: Evaluation Execution.

    Runs Inspect evaluation for each capability with each subject LLM.
    Creates eval_tag since this is where LLM calls happen.

    Args:
        cfg: Configuration object
        eval_config: EvalConfig from Stage 0

    Returns
    -------
        The eval_tag for this evaluation run
    """
    # Derive paths from config
    exp_id = cfg.exp_cfg.exp_id
    output_base_dir = Path(cfg.global_cfg.output_dir)
    experiment_dir = output_base_dir / exp_id
    validation_tag = eval_config.validation_tag

    # Create eval_tag for this run
    eval_tag = timestamp_tag()

    logger.info(
        "Eval Stage 1: Running evaluations (eval_tag=%s)",
        eval_tag,
    )

    # Find datasets (saved under validation_tag from Stage 0)
    datasets_dir = experiment_dir / "eval" / "datasets" / validation_tag
    dataset_paths = _find_datasets(datasets_dir)
    logger.info("Found %d datasets", len(dataset_paths))

    if not dataset_paths:
        raise ValueError(f"No datasets found in {datasets_dir}. Run Stage 0 first.")

    # Load datasets
    datasets = [load_eval_dataset(p) for p in dataset_paths]

    # Setup results directory under eval_tag
    eval_dir = experiment_dir / "eval" / "results" / eval_tag
    results_dir = eval_dir

    # Update eval_config with the tag and save it
    eval_config.eval_tag = eval_tag
    metadata = PipelineMetadata(
        experiment_id=exp_id,
        output_base_dir=str(output_base_dir),
        timestamp=iso_timestamp(),
        input_stage_tag=validation_tag,
        output_stage_tag=eval_tag,
        resume=False,
    )
    eval_config_path = eval_dir / "eval_config.json"
    save_eval_config(eval_config, metadata, eval_config_path)
    logger.info("Saved eval_config.json to %s", eval_config_path)

    # Run evaluations
    subject_llms = eval_config.subject_llms
    judge_llm = eval_config.judge_llm

    num_evals = 0
    total_combinations = len(datasets) * len(subject_llms)

    for dataset in datasets:
        for llm_config in subject_llms:
            llm_name = llm_config["name"]
            # Construct full model string: provider/model_name
            subject_model = f"{llm_config['provider']}/{llm_name}"

            # Check if already completed (resume)
            if _check_eval_completed(
                results_dir, llm_name, dataset.area_id, dataset.capability_id
            ):
                logger.info(
                    "  Skipping %s/%s with %s (already completed)",
                    dataset.area_id,
                    dataset.capability_id,
                    llm_name,
                )
                continue

            # Run evaluation
            output_dir = (
                results_dir / llm_name / dataset.area_id / dataset.capability_id
            )

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
                num_evals += 1

    logger.info(
        "Eval Stage 1: Completed %d/%d evaluations",
        num_evals,
        total_combinations,
    )

    return eval_tag
