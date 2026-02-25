"""Evaluation pipeline for running LLM evaluations on generated tasks.

This module orchestrates the evaluation pipeline:
- Stage 0: Setup and Dataset Preparation
- Stage 1: Evaluation Execution (runs subject LLMs, creates eval_tag)
- Stage 2: Score Aggregation

Usage:
    # Run all stages
    python -m src.run_eval_pipeline validation_tag=_YYYYMMDD_HHMMSS

    # Run specific stage
    python -m src.run_eval_pipeline stage=0 validation_tag=_YYYYMMDD_HHMMSS
    python -m src.run_eval_pipeline stage=1 validation_tag=_YYYYMMDD_HHMMSS
    python -m src.run_eval_pipeline stage=1 validation_tag=_YYYYMMDD_HHMMSS \
        eval_tag=_YYYYMMDD_HHMMSS
    python -m src.run_eval_pipeline stage=2 eval_tag=_YYYYMMDD_HHMMSS
"""

import logging

import hydra
from omegaconf import DictConfig

from src.eval_stages import (
    EvalSetupError,
    run_eval_stage0,
    run_eval_stage1,
    run_eval_stage2,
)


logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="cfg", config_name="run_cfg")
def main(cfg: DictConfig) -> None:
    """Run the evaluation pipeline."""
    # Get stage to run (default: "all")
    stage = cfg.get("stage", "all")
    if isinstance(stage, str) and stage.isdigit():
        stage = int(stage)

    # Get tags from config
    validation_tag = cfg.get("validation_tag")
    eval_tag = cfg.get("eval_tag")

    logger.info("=" * 60)
    logger.info("EVALUATION PIPELINE")
    logger.info("=" * 60)
    logger.info("Stage: %s", stage)
    logger.info("Experiment ID: %s", cfg.exp_cfg.exp_id)
    logger.info("validation_tag: %s", validation_tag)
    logger.info("eval_tag: %s", eval_tag)
    logger.info("=" * 60)

    # Run all stages sequentially
    if stage == "all":
        if not validation_tag:
            logger.error("validation_tag is required")
            logger.error(
                "Usage: python -m src.run_eval_pipeline validation_tag=_YYYYMMDD_HHMMSS"
            )
            return

        try:
            # Stage 0: Setup and Dataset Preparation
            logger.info("Running Eval Stage 0: Setup and Dataset Preparation")
            run_eval_stage0(cfg, validation_tag)
            logger.info("Eval Stage 0 complete.")

            # Stage 1: Evaluation Execution
            logger.info("Running Eval Stage 1: Evaluation Execution")
            eval_tag = run_eval_stage1(cfg, validation_tag, eval_tag)
            logger.info("Eval Stage 1 complete. eval_tag=%s", eval_tag)

            # Stage 2: Score Aggregation
            logger.info("Running Eval Stage 2: Score Aggregation")
            run_eval_stage2(cfg, eval_tag)
            logger.info("Eval Stage 2 complete.")

        except EvalSetupError as e:
            logger.error("Evaluation setup failed: %s", e)
            return
        except ValueError as e:
            logger.error("Evaluation failed: %s", e)
            return

    # Run specific stage
    elif stage == 0:
        if not validation_tag:
            logger.error("validation_tag is required for stage 0")
            logger.error(
                "Usage: python -m src.run_eval_pipeline stage=0 "
                "validation_tag=_YYYYMMDD_HHMMSS"
            )
            return

        try:
            run_eval_stage0(cfg, validation_tag)
            logger.info("Eval Stage 0 complete. Datasets created.")
        except EvalSetupError as e:
            logger.error("Evaluation setup failed: %s", e)

    elif stage == 1:
        if not validation_tag:
            logger.error("validation_tag is required for stage 1")
            logger.error(
                "Usage: python -m src.run_eval_pipeline stage=1 "
                "validation_tag=_YYYYMMDD_HHMMSS"
            )
            return

        try:
            # Stage 1 reads eval_config from Stage 0's output
            eval_tag = run_eval_stage1(cfg, validation_tag, eval_tag)
            logger.info("Eval Stage 1 complete. eval_tag=%s", eval_tag)
        except ValueError as e:
            logger.error("Stage 1 failed: %s", e)

    elif stage == 2:
        if not eval_tag:
            logger.error("eval_tag is required for stage 2")
            logger.error(
                "Usage: python -m src.run_eval_pipeline stage=2 "
                "eval_tag=_YYYYMMDD_HHMMSS"
            )
            return

        try:
            run_eval_stage2(cfg, eval_tag)
            logger.info("Eval Stage 2 complete.")
        except ValueError as e:
            logger.error("Stage 2 failed: %s", e)

    else:
        logger.error("Invalid stage: %s. Use 'all', 0, 1, or 2", stage)


if __name__ == "__main__":
    main()
