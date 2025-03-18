"""Module containing constants used throughout the codebase."""

import os


BASE_ARTIFACTS_DIR = "/fs01/projects/aieng/public/ace_artifacts"

SEED_CAPABILITIES_SCORE_DIR = os.path.join(
    BASE_ARTIFACTS_DIR, "seed_capabilities_results"
)
NON_SEED_CAPABILITIES_SCORE_DIR = os.path.join(
    BASE_ARTIFACTS_DIR, "capabilities_results"
)


TAB_W_SPACES = "    "


# Score functions for various seed capability datasets
# Score function is based on https://github.com/UKGovernmentBEIS/inspect_evals/blob/main/src/inspect_evals/mathematics/utils.py#L57
MATHEMATICS_SCORE_FUNC = f"""def score(t: dict, submission: str) -> float | None:\n{TAB_W_SPACES}{TAB_W_SPACES}ans_pattern_line = r"(?i)ANSWER\\s*:\\s*([^\\n]+)"\n{TAB_W_SPACES}{TAB_W_SPACES}match = re.search(ans_pattern_line, submission)\n{TAB_W_SPACES}{TAB_W_SPACES}if match:\n{TAB_W_SPACES}{TAB_W_SPACES}{TAB_W_SPACES}answer = match.group(1)\n{TAB_W_SPACES}{TAB_W_SPACES}{TAB_W_SPACES}correct = is_equiv(answer, t["answer"])\n{TAB_W_SPACES}{TAB_W_SPACES}else:\n{TAB_W_SPACES}{TAB_W_SPACES}{TAB_W_SPACES}correct = False\n{TAB_W_SPACES}{TAB_W_SPACES}return 1.0 if correct else 0.0"""
# Score function is based on https://github.com/UKGovernmentBEIS/inspect_evals/blob/main/src/inspect_evals/mathematics/utils.py#L57
GSM8K_SCORE_FUNC = f"""def score(t: dict, submission: str) -> float | None:\n{TAB_W_SPACES}{TAB_W_SPACES}return 1.0 if submission==t["answer"] else 0.0"""
