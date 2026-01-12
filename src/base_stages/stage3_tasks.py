"""Stage 3: Task generation.

This stage generates tasks (questions with options) for each capability.
The correct answer is NOT determined here — that happens in Stage 4.
"""

import logging
from pathlib import Path

from omegaconf import DictConfig

from src.base_stages.generate_diverse_tasks_pipeline import (
    generate_diverse_tasks_for_capability,
)
from src.schemas.io_utils import load_capabilities, save_tasks
from src.schemas.metadata_schemas import PipelineMetadata
from src.utils import constants
from src.utils.model_client_utils import get_standard_model_client
from src.utils.timestamp_utils import iso_timestamp, timestamp_tag


logger = logging.getLogger(__name__)


def run_stage3(
    cfg: DictConfig,
    capabilities_tag: str,
    tasks_tag: str = None,
) -> str:
    """Stage 3: Generate tasks for each capability.

    This stage generates Task objects (questions with options) using a
    two-step process:
    1. Generate the question text
    2. Generate 4 options for the question

    The correct answer is NOT determined here — that happens in Stage 4
    (Solution Generation) where an LLM solves each task.

    Args:
        cfg: Configuration object
        capabilities_tag: Tag from Stage 2 to load capabilities from
        tasks_tag: Optional resume tag

    Returns
    -------
        The tasks_tag for this generation
    """
    experiment_id = cfg.exp_cfg.exp_id
    output_base_dir = Path(cfg.global_cfg.output_dir)

    # Determine tasks tag (resume or new)
    is_resume = tasks_tag is not None
    if is_resume:
        logger.info(f"Resuming Stage 3 with tasks_tag: {tasks_tag}")
    else:
        tasks_tag = timestamp_tag()
        logger.info(f"Starting new Stage 3 with tasks_tag: {tasks_tag}")

    # Initialize scientist LLM client using task_generation config
    scientist_llm_gen_cfg = dict(cfg.scientist_llm.generation_cfg.task_generation)
    scientist_llm_client = get_standard_model_client(
        cfg.scientist_llm.name,
        seed=scientist_llm_gen_cfg.get("seed", cfg.exp_cfg.seed),
        temperature=scientist_llm_gen_cfg.get(
            "temperature", constants.DEFAULT_TEMPERATURE
        ),
        max_tokens=scientist_llm_gen_cfg.get(
            "max_tokens", constants.DEFAULT_MAX_TOKENS
        ),
    )

    # Get task generation parameters from config
    tasks_per_blueprint = cfg.task_generation_cfg.get("tasks_per_blueprint", 3)
    min_subtopics = cfg.task_generation_cfg.get("min_subtopics", 3)
    max_subtopics = cfg.task_generation_cfg.get("max_subtopics", 8)

    # Find all area directories under capabilities/<capabilities_tag>/
    capabilities_base_dir = (
        output_base_dir / experiment_id / "capabilities" / capabilities_tag
    )
    area_dirs = [d for d in capabilities_base_dir.iterdir() if d.is_dir()]

    logger.info(f"Found {len(area_dirs)} area directories")

    # Process each area
    for area_dir in area_dirs:
        area_id = area_dir.name
        logger.info(f"Processing area: {area_id}")

        # Load capabilities for this area
        capabilities_path = area_dir / "capabilities.json"
        capabilities, _ = load_capabilities(capabilities_path)
        logger.info(f"Loaded {len(capabilities)} capabilities from {area_id}")

        # Process each capability
        for capability in capabilities:
            capability_id = capability.capability_id

            # Check if tasks already exist for this capability (resume logic)
            tasks_path = (
                output_base_dir
                / experiment_id
                / "tasks"
                / tasks_tag
                / area_id
                / capability_id
                / "tasks.json"
            )

            if is_resume and tasks_path.exists():
                logger.info(
                    f"Skipping {area_id}/{capability_id} - "
                    f"tasks already exist at {tasks_path}"
                )
                continue

            logger.info(
                f"Generating tasks for capability: {capability.capability_name} "
                f"({area_id}/{capability_id})"
            )

            try:
                # Generate diverse tasks
                tasks = generate_diverse_tasks_for_capability(
                    capability=capability,
                    tasks_per_blueprint=tasks_per_blueprint,
                    client=scientist_llm_client,
                    min_subtopics=min_subtopics,
                    max_subtopics=max_subtopics,
                )

                logger.info(
                    f"Generated {len(tasks)} tasks for {capability.capability_name}"
                )

                # Save tasks
                metadata = PipelineMetadata(
                    experiment_id=experiment_id,
                    output_base_dir=str(output_base_dir),
                    timestamp=iso_timestamp(),
                    input_stage_tag=capabilities_tag,
                    output_stage_tag=tasks_tag,
                    resume=is_resume,
                )

                save_tasks(tasks, metadata, tasks_path)

                logger.info(
                    f"Stage 3: saved {len(tasks)} tasks to "
                    f"tasks/{tasks_tag}/{area_id}/{capability_id}/tasks.json"
                )

            except Exception as e:
                logger.error(
                    f"Error generating tasks for {area_id}/{capability_id}: {e}",
                    exc_info=True,
                )
                # Continue with next capability instead of failing completely
                continue

    return tasks_tag
