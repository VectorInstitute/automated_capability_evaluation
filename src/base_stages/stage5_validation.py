"""Stage 5: Task validation.

This stage validates generated task solutions to ensure they are correct
and align with the task requirements.
"""

import logging
from pathlib import Path
from typing import Optional

from omegaconf import DictConfig

from src.base_stages.validate_tasks import validate_tasks
from src.schemas.io_utils import load_solution, save_validation
from src.schemas.metadata_schemas import PipelineMetadata
from src.utils import constants
from src.utils.model_client_utils import get_standard_model_client
from src.utils.timestamp_utils import iso_timestamp, timestamp_tag


logger = logging.getLogger(__name__)


def run_stage5(
    cfg: DictConfig,
    solution_tag: str,
    validation_tag: Optional[str] = None,
) -> str:
    """Stage 5: Validate generated task solutions.

    Args:
        cfg: Configuration object
        solution_tag: Tag from Stage 4 to load solutions from
        validation_tag: Optional resume tag

    Returns
    -------
        The validation_tag for this validation
    """
    experiment_id = cfg.exp_cfg.exp_id
    output_base_dir = Path(cfg.global_cfg.output_dir)

    # Determine validation tag (resume or new)
    is_resume = validation_tag is not None
    if is_resume:
        logger.info(f"Resuming Stage 5 with validation_tag: {validation_tag}")
    else:
        validation_tag = timestamp_tag()
        logger.info(f"Starting new Stage 5 with validation_tag: {validation_tag}")

    # Initialize validator LLM client using task_verify config
    validator_llm_gen_cfg = dict(cfg.scientist_llm.generation_cfg.task_verify)
    validator_llm_client = get_standard_model_client(
        cfg.scientist_llm.name,
        seed=validator_llm_gen_cfg.get("seed", cfg.exp_cfg.seed),
        temperature=validator_llm_gen_cfg.get(
            "temperature", constants.DEFAULT_TEMPERATURE
        ),
        max_tokens=validator_llm_gen_cfg.get(
            "max_tokens", constants.DEFAULT_MAX_TOKENS
        ),
    )

    # Find all solutions directories
    solutions_base_dir = output_base_dir / experiment_id / "solutions" / solution_tag

    if not solutions_base_dir.exists():
        logger.error(f"Solutions directory not found: {solutions_base_dir}")
        assert validation_tag is not None
        return validation_tag

    # Find all area directories
    area_dirs = [d for d in solutions_base_dir.iterdir() if d.is_dir()]
    logger.info(f"Found {len(area_dirs)} area directories")

    # Process each area
    for area_dir in area_dirs:
        area_id = area_dir.name
        logger.info(f"Processing area: {area_id}")

        # Find all capability directories
        capability_dirs = [d for d in area_dir.iterdir() if d.is_dir()]

        for capability_dir in capability_dirs:
            capability_id = capability_dir.name

            # Check if validation already exists for this capability (resume logic)
            validation_cap_dir = (
                output_base_dir
                / experiment_id
                / "validation"
                / validation_tag
                / area_id
                / capability_id
            )

            if is_resume and validation_cap_dir.exists():
                existing_validations = list(
                    validation_cap_dir.glob("*/validation.json")
                )
                if existing_validations:
                    logger.info(
                        f"Skipping {area_id}/{capability_id} - "
                        f"{len(existing_validations)} validations already exist"
                    )
                    continue

            # Find all task solution directories (each task has its own directory)
            task_dirs = [d for d in capability_dir.iterdir() if d.is_dir()]

            if not task_dirs:
                logger.warning(f"No solutions found in {area_id}/{capability_id}")
                continue

            logger.info(
                f"Validating {len(task_dirs)} solutions for {area_id}/{capability_id}"
            )

            # Load all task solutions for this capability
            task_solutions = []

            for task_dir in task_dirs:
                solution_file = task_dir / "solution.json"
                if solution_file.exists():
                    task_solution, _ = load_solution(solution_file)
                    task_solutions.append(task_solution)

            if not task_solutions:
                logger.warning(
                    f"No valid solutions loaded for {area_id}/{capability_id}"
                )
                continue

            # Validate all tasks for this capability
            try:
                validation_results = validate_tasks(
                    task_solutions=task_solutions,
                    client=validator_llm_client,
                )

                # Save individual validation results
                for validation_result in validation_results:
                    task_id = validation_result.task_id
                    validation_path = validation_cap_dir / task_id / "validation.json"

                    metadata = PipelineMetadata(
                        experiment_id=experiment_id,
                        output_base_dir=str(output_base_dir),
                        timestamp=iso_timestamp(),
                        input_stage_tag=solution_tag,
                        output_stage_tag=validation_tag,
                        resume=is_resume,
                    )

                    save_validation(validation_result, metadata, validation_path)

                logger.info(
                    f"Validated {area_id}/{capability_id}: "
                    f"{len(validation_results)} task(s) validated"
                )

            except Exception as e:
                logger.error(
                    f"Error validating tasks for {area_id}/{capability_id}: {e}",
                    exc_info=True,
                )
                continue

    logger.info(f"Stage 5 completed. Validation tag: {validation_tag}")
    assert validation_tag is not None
    return validation_tag
