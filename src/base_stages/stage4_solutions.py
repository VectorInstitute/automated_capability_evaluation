"""Stage 4: Solution generation.

This stage solves tasks by having an LLM determine the correct answer
for each multiple-choice question.
"""

import logging
from pathlib import Path

from omegaconf import DictConfig

from src.base_stages.solve_tasks import solve_tasks
from src.schemas.io_utils import load_tasks, save_solution
from src.schemas.metadata_schemas import PipelineMetadata
from src.utils import constants
from src.utils.model_client_utils import get_standard_model_client
from src.utils.timestamp_utils import iso_timestamp, timestamp_tag


logger = logging.getLogger(__name__)


def run_stage4(
    cfg: DictConfig,
    tasks_tag: str,
    solution_tag: str = None,
) -> str:
    """Stage 4: Generate solutions for tasks.

    This stage takes Task objects and determines the correct answer by
    having an LLM solve each task.

    Args:
        cfg: Configuration object
        tasks_tag: Tag from Stage 3 to load tasks from
        solution_tag: Optional resume tag

    Returns
    -------
        The solution_tag for this generation
    """
    experiment_id = cfg.exp_cfg.exp_id
    output_base_dir = Path(cfg.global_cfg.output_dir)

    # Determine solution tag (resume or new)
    is_resume = solution_tag is not None
    if is_resume:
        logger.info(f"Resuming Stage 4 with solution_tag: {solution_tag}")
    else:
        solution_tag = timestamp_tag()
        logger.info(f"Starting new Stage 4 with solution_tag: {solution_tag}")

    # Initialize solver LLM client using task_solve config
    solver_llm_gen_cfg = dict(cfg.scientist_llm.generation_cfg.task_solve)
    solver_llm_client = get_standard_model_client(
        cfg.scientist_llm.name,
        seed=solver_llm_gen_cfg.get("seed", cfg.exp_cfg.seed),
        temperature=solver_llm_gen_cfg.get(
            "temperature", constants.DEFAULT_TEMPERATURE
        ),
        max_tokens=solver_llm_gen_cfg.get("max_tokens", constants.DEFAULT_MAX_TOKENS),
    )

    # Find all task directories under tasks/<tasks_tag>/
    tasks_base_dir = output_base_dir / experiment_id / "tasks" / tasks_tag

    if not tasks_base_dir.exists():
        logger.error(f"Tasks directory not found: {tasks_base_dir}")
        return solution_tag

    area_dirs = [d for d in tasks_base_dir.iterdir() if d.is_dir()]
    logger.info(f"Found {len(area_dirs)} area directories")

    # Process each area
    for area_dir in area_dirs:
        area_id = area_dir.name
        logger.info(f"Processing area: {area_id}")

        # Find all capability directories
        capability_dirs = [d for d in area_dir.iterdir() if d.is_dir()]

        for capability_dir in capability_dirs:
            capability_id = capability_dir.name

            # Check if solutions already exist for this capability (resume logic)
            solutions_dir = (
                output_base_dir
                / experiment_id
                / "solutions"
                / solution_tag
                / area_id
                / capability_id
            )

            if (
                is_resume
                and solutions_dir.exists()
                and any(solutions_dir.glob("*/solution.json"))
            ):
                logger.info(
                    f"Skipping {area_id}/{capability_id} - "
                    f"solutions already exist at {solutions_dir}"
                )
                continue

            # Load tasks for this capability
            tasks_path = capability_dir / "tasks.json"
            if not tasks_path.exists():
                logger.warning(f"No tasks found at {tasks_path}")
                continue

            tasks, _ = load_tasks(tasks_path)
            logger.info(f"Loaded {len(tasks)} tasks for {area_id}/{capability_id}")

            try:
                # Solve tasks
                task_solutions = solve_tasks(tasks=tasks, client=solver_llm_client)

                logger.info(
                    f"Generated {len(task_solutions)} solutions for "
                    f"{area_id}/{capability_id}"
                )

                # Save each solution
                metadata = PipelineMetadata(
                    experiment_id=experiment_id,
                    output_base_dir=str(output_base_dir),
                    timestamp=iso_timestamp(),
                    input_stage_tag=tasks_tag,
                    output_stage_tag=solution_tag,
                    resume=is_resume,
                )

                for task_solution in task_solutions:
                    solution_path = (
                        output_base_dir
                        / experiment_id
                        / "solutions"
                        / solution_tag
                        / area_id
                        / capability_id
                        / task_solution.task_id
                        / "solution.json"
                    )
                    save_solution(task_solution, metadata, solution_path)

                logger.info(
                    f"Stage 4: saved {len(task_solutions)} solutions to "
                    f"solutions/{solution_tag}/{area_id}/{capability_id}/"
                )

            except Exception as e:
                logger.error(
                    f"Error solving tasks for {area_id}/{capability_id}: {e}",
                    exc_info=True,
                )
                continue

    return solution_tag
