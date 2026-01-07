"""Evaluation pipeline for running LLM evaluations on generated tasks.

This module orchestrates the evaluation pipeline:
- Stage 0: Setup and Dataset Preparation (no LLM calls, no tag)
- Stage 1: Evaluation Execution (runs subject LLMs, creates eval_tag)
- Stage 2: Score Aggregation (no LLM calls)

Usage:
    # Run all stages
    python -m src.run_eval_pipeline validation_tag=_YYYYMMDD_HHMMSS

    # Run specific stage
    python -m src.run_eval_pipeline stage=0 validation_tag=_YYYYMMDD_HHMMSS
    python -m src.run_eval_pipeline stage=1 validation_tag=_YYYYMMDD_HHMMSS
    python -m src.run_eval_pipeline stage=2 eval_tag=_YYYYMMDD_HHMMSS
"""

import logging
from pathlib import Path

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
            eval_config = run_eval_stage0(cfg, validation_tag)
            logger.info("Eval Stage 0 complete.")

            # Stage 1: Evaluation Execution
            logger.info("Running Eval Stage 1: Evaluation Execution")
            eval_tag = run_eval_stage1(cfg, eval_config)
            logger.info("Eval Stage 1 complete. eval_tag=%s", eval_tag)

            # Stage 2: Score Aggregation
            logger.info("Running Eval Stage 2: Score Aggregation")
            run_eval_stage2(cfg, eval_tag)
            logger.info("Eval Stage 2 complete.")

            # Get results dir for final message
            exp_id = cfg.exp_cfg.exp_id
            output_base_dir = Path(cfg.global_cfg.output_dir)
            scores_dir = output_base_dir / exp_id / "eval" / "scores" / eval_tag

            logger.info("=" * 60)
            logger.info("EVALUATION PIPELINE COMPLETE")
            logger.info("Scores in: %s", scores_dir)
            logger.info("=" * 60)

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
            eval_config = run_eval_stage0(cfg, validation_tag)
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
            # Run Stage 0 first to get eval_config
            eval_config = run_eval_stage0(cfg, validation_tag)
            eval_tag = run_eval_stage1(cfg, eval_config)
            logger.info("Eval Stage 1 complete. eval_tag=%s", eval_tag)
        except (EvalSetupError, ValueError) as e:
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
