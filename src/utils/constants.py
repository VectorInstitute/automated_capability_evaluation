"""Module containing constants used throughout the codebase."""

import os
from enum import Enum


BASE_ARTIFACTS_DIR = "/fs01/projects/aieng/public/ace/artifacts"
GCP_BASE_ARTIFACTS_DIR = "gs://ace-artifacts"
BASE_INSPECT_EVALS_DIR = "/fs01/projects/aieng/public/ace/inspect_evals/src/ace_evals"

SEED_CAPABILITIES_SCORE_DIR = os.path.join(
    GCP_BASE_ARTIFACTS_DIR, "seed_capabilities_results"
)
NON_SEED_CAPABILITIES_SCORE_DIR = os.path.join(
    GCP_BASE_ARTIFACTS_DIR, "capabilities_results"
)


TAB_W_SPACES = "    "


# Score functions for various seed capability datasets
# Score function is based on https://github.com/UKGovernmentBEIS/inspect_evals/blob/main/src/inspect_evals/mathematics/utils.py#L57
MATHEMATICS_SCORE_FUNC = f"""def score(t: dict, submission: str) -> float | None:\n{TAB_W_SPACES}{TAB_W_SPACES}from .utils import parse_submission, evaluate_with_llm_judge\n{TAB_W_SPACES}{TAB_W_SPACES}answer = parse_submission(submission)\n{TAB_W_SPACES}{TAB_W_SPACES}correct = evaluate_with_llm_judge(answer, t["answer"])\n{TAB_W_SPACES}{TAB_W_SPACES}return 1.0 if correct else 0.0"""
# Score function is based on https://github.com/UKGovernmentBEIS/inspect_evals/blob/main/src/inspect_evals/mathematics/utils.py#L57
GSM8K_SCORE_FUNC = f"""def score(t: dict, submission: str) -> float | None:\n{TAB_W_SPACES}{TAB_W_SPACES}return 1.0 if submission==t["answer"] else 0.0"""

DATASET_NAME_MAP = {
    "mathematics": "competition_math",
    "gsm8k": "word_problems",
}


NO_ANSWER_STR = "NO_ANSWER"


C_STATE_INITIALIZED_STR = "initialized"
C_STATE_FILTERED_OUT_STR = "filtered_out"
C_STATE_TASK_GENERATION_FAILED_STR = "task_generation_failed"
C_STATE_TASK_GENERATION_PARTIALLY_COMPLETED_STR = "task_generation_partially_completed"
C_STATE_TASK_GENERATION_COMPLETED_STR = "task_generation_completed"


class VecInfStatus(Enum):
    """Enum class for vector inference status."""

    PENDING = "PENDING"
    LAUNCHING = "LAUNCHING"
    READY = "READY"
    FAILED = "FAILED"
    SHUTDOWN = "SHUTDOWN"
    UNAVAILABLE = "UNAVAILABLE"


DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"

DEFAULT_CAPABILITY_GENERATION_RETRY_ATTEMPTS = 3
DEFAULT_TASK_GENERATION_RETRY_ATTEMPTS = 3

DEFAULT_RANDOM_SEED = 42

DEFAULT_CONCURRENCY = 5
DEFAULT_TASK_SOLVER_CONCURRENCY = 5
DEFAULT_TASK_VERIFIER_CONCURRENCY = 5
DEFAULT_TASK_EVAL_CONCURRENCY = 5
