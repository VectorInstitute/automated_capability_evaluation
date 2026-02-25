"""Evaluation pipeline stages.

Stage 0: Setup and Dataset Preparation (no LLM calls)
Stage 1: Evaluation Execution (runs subject LLMs, creates eval_tag)
Stage 2: Score Aggregation (no LLM calls)
"""

from src.eval_stages.stage0_setup_and_dataset import EvalSetupError, run_eval_stage0
from src.eval_stages.stage1_eval_execution import run_eval_stage1
from src.eval_stages.stage2_score_aggregation import run_eval_stage2


__all__ = [
    "run_eval_stage0",
    "run_eval_stage1",
    "run_eval_stage2",
    "EvalSetupError",
]
