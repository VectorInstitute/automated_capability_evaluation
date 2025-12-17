"""Base pipeline for capability and task generation.

This module orchestrates the complete base (non-agentic) generation pipeline:
- Stage 0: Experiment and domain setup
- Stage 1: Area generation
- Stage 2: Capability generation and filtering
- Stage 3: Task generation
- Stage 4: Solution generation
- Stage 5: Task validation

Usage:
    # Run all stages
    python -m src.run_base_pipeline stage=all

    # Run specific stage
    python -m src.run_base_pipeline stage=0
    python -m src.run_base_pipeline stage=1
    python -m src.run_base_pipeline stage=2 areas_tag=_YYYYMMDD_HHMMSS
    python -m src.run_base_pipeline stage=3 capabilities_tag=_YYYYMMDD_HHMMSS
    python -m src.run_base_pipeline stage=4 tasks_tag=_YYYYMMDD_HHMMSS
    python -m src.run_base_pipeline stage=5 solution_tag=_YYYYMMDD_HHMMSS
"""

import logging

import hydra
from omegaconf import DictConfig

from src.base_stages import (
    run_stage0,
    run_stage1,
    run_stage2,
    run_stage3,
    run_stage4,
    run_stage5,
)


logger = logging.getLogger(__name__)


def _validate_stage_inputs(
    stage: int | str,
    areas_tag: str | None,
    capabilities_tag: str | None,
    tasks_tag: str | None,
    solution_tag: str | None,
) -> bool:
    """Validate required inputs for standalone stage execution.

    Returns True if validation passes, False otherwise.
    """
    if stage == 2 and not areas_tag:
        logger.error("areas_tag is required when running stage 2 standalone")
        logger.error(
            "Usage: python -m src.run_base_pipeline stage=2 areas_tag=_YYYYMMDD_HHMMSS"
        )
        logger.error(
            "Optional: capabilities_tag=_YYYYMMDD_HHMMSS to resume from existing run"
        )
        return False

    if stage == 3 and not capabilities_tag:
        logger.error("capabilities_tag is required when running stage 3 standalone")
        logger.error(
            "Usage: python -m src.run_base_pipeline stage=3 "
            "capabilities_tag=_YYYYMMDD_HHMMSS"
        )
        logger.error("Optional: tasks_tag=_YYYYMMDD_HHMMSS to resume from existing run")
        return False

    if stage == 4 and not tasks_tag:
        logger.error("tasks_tag is required when running stage 4 standalone")
        logger.error(
            "Usage: python -m src.run_base_pipeline stage=4 tasks_tag=_YYYYMMDD_HHMMSS"
        )
        logger.error(
            "Optional: solution_tag=_YYYYMMDD_HHMMSS to resume from existing run"
        )
        return False

    if stage == 5 and not solution_tag:
        logger.error("solution_tag is required when running stage 5 standalone")
        logger.error(
            "Usage: python -m src.run_base_pipeline stage=5 "
            "solution_tag=_YYYYMMDD_HHMMSS"
        )
        logger.error(
            "Optional: validation_tag=_YYYYMMDD_HHMMSS to resume from existing run"
        )
        return False

    return True


@hydra.main(version_base=None, config_path="cfg", config_name="run_cfg")
def main(cfg: DictConfig) -> None:
    """Run specific pipeline stages based on configuration.

    Stage 0: Experiment and domain setup
    Stage 1: Area generation
    Stage 2: Capability generation and filtering
    Stage 3: Task generation
    Stage 4: Solution generation
    Stage 5: Task validation
    "all": Run all stages sequentially
    """
    # Suppress httpx and autogen_core INFO logs
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("autogen_core.events").setLevel(logging.WARNING)

    # Get stage from config (can be overridden via command line)
    stage = cfg.get("stage", "all")

    # Convert string to int if numeric
    if isinstance(stage, str) and stage.isdigit():
        stage = int(stage)

    logger.info(f"Running stage: {stage}")

    # Track tags across stages
    areas_tag = cfg.get("areas_tag", None)
    capabilities_tag = cfg.get("capabilities_tag", None)
    tasks_tag = cfg.get("tasks_tag", None)
    solution_tag = cfg.get("solution_tag", None)

    # Validate required inputs for standalone stages
    if not _validate_stage_inputs(
        stage, areas_tag, capabilities_tag, tasks_tag, solution_tag
    ):
        return

    # Stage 0: Experiment and Domain Setup
    if stage in {0, "all"}:
        logger.info("=" * 60)
        logger.info("STAGE 0: Experiment and Domain Setup")
        logger.info("=" * 60)
        run_stage0(cfg)
        if stage == 0:
            return

    # Stage 1: Area Generation
    if stage in {1, "all"}:
        logger.info("=" * 60)
        logger.info("STAGE 1: Area Generation")
        logger.info("=" * 60)
        areas_tag = run_stage1(cfg)
        logger.info("Stage 1 areas tag: %s", areas_tag)
        if stage == 1:
            return

    # Stage 2: Capability Generation and Filtering
    if stage in {2, "all"}:
        logger.info("=" * 60)
        logger.info("STAGE 2: Capability Generation and Filtering")
        logger.info("=" * 60)

        # Check if resuming
        resume_capabilities_tag = (
            cfg.get("capabilities_tag", None) if stage == 2 else None
        )
        if resume_capabilities_tag:
            logger.info(
                f"Resume mode: Will skip areas that already have capabilities "
                f"in tag {resume_capabilities_tag}"
            )

        capabilities_tag = run_stage2(
            cfg=cfg,
            areas_tag=areas_tag,
            capabilities_tag=resume_capabilities_tag,
        )
        logger.info("Stage 2 capabilities tag: %s", capabilities_tag)
        if stage == 2:
            return

    # Stage 3: Task Generation
    if stage in {3, "all"}:
        logger.info("=" * 60)
        logger.info("STAGE 3: Task Generation")
        logger.info("=" * 60)

        # Check if resuming
        resume_tasks_tag = cfg.get("tasks_tag", None) if stage == 3 else None
        if resume_tasks_tag:
            logger.info(
                f"Resume mode: Will skip capabilities that already have tasks "
                f"in tag {resume_tasks_tag}"
            )

        tasks_tag = run_stage3(
            cfg=cfg,
            capabilities_tag=capabilities_tag,
            tasks_tag=resume_tasks_tag,
        )
        logger.info("Stage 3 tasks tag: %s", tasks_tag)
        if stage == 3:
            return

    # Stage 4: Solution Generation
    if stage in {4, "all"}:
        logger.info("=" * 60)
        logger.info("STAGE 4: Solution Generation")
        logger.info("=" * 60)

        # Check if resuming
        resume_solution_tag = cfg.get("solution_tag", None) if stage == 4 else None
        if resume_solution_tag:
            logger.info(
                f"Resume mode: Will skip tasks that already have solutions "
                f"in tag {resume_solution_tag}"
            )

        solution_tag = run_stage4(
            cfg=cfg,
            tasks_tag=tasks_tag,
            solution_tag=resume_solution_tag,
        )
        logger.info("Stage 4 solution tag: %s", solution_tag)
        if stage == 4:
            return

    # Stage 5: Task Validation
    if stage in {5, "all"}:
        logger.info("=" * 60)
        logger.info("STAGE 5: Task Validation")
        logger.info("=" * 60)

        # Check if resuming
        resume_validation_tag = cfg.get("validation_tag", None) if stage == 5 else None
        if resume_validation_tag:
            logger.info(
                f"Resume mode: Will skip tasks that already have validations "
                f"in tag {resume_validation_tag}"
            )

        validation_tag = run_stage5(
            cfg=cfg,
            solution_tag=solution_tag,
            validation_tag=resume_validation_tag,
        )
        logger.info("Stage 5 validation tag: %s", validation_tag)


if __name__ == "__main__":
    main()
