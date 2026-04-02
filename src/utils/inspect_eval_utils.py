"""
The inspect_eval_utils module for the automatic_benchmark_generation project.

It contains utility functions for inspect evals.
"""

import os
import re
from typing import Any, Dict

from inspect_ai.model import GenerateConfig as InspectGenerateConfig
from inspect_ai.model import Model as InspectModel
from inspect_ai.model import get_model


# Define this constant and prompt directly in file since
# it will be ported to the inspect folder
INSPECT_JUDGE_LLM = "openai/gpt-4o-mini"
INSPECT_JUDGE_LLM_GEN_CONFIG: Dict[str, Any] = {}

LLM_JUDGE_PROMPT = """
Look at the submission and the target string for a given task and judge whether they are equivalent or not. If they are semantically equivalent, respond with "Yes". Respond with only "Yes" or "No" (without quotes). Do not include a rationale.

Submission: {submission}
Target: {target}
"""


# Borrowed from:
# https://github.com/UKGovernmentBEIS/inspect_ai/blob/main/src/inspect_ai/_util/pattern.py#L3
def parse_submission(submission: str) -> str:
    """
    Parse the submission string to extract the final answer.

    This function is used in the capability class score method.

    Args
    ----
        submission (str): The submission string to parse.

    Returns
    -------
        str: The extracted answer from the submission, or an empty string
            if no match is found.
    """
    if not submission:
        return ""

    def _maybe_extract_option_letter(text: str) -> str | None:
        """
        Normalize common answer formats (e.g., "ANSWER: B", "Option B", "\\boxed{ANSWER: B}")
        into a single letter like "B".
        """
        if not text:
            return None
        s = text.strip()
        # Normalize common LaTeX wrappers like "\text{ANSWER: B}" -> "ANSWER: B".
        s = re.sub(r"\\text\s*\{([^}]*)\}", r"\1", s, flags=re.IGNORECASE)
        s = s.replace("{", "").replace("}", "")
        # Handle cases like "ANSWER: B" (possibly with trailing punctuation).
        m = re.search(
            r"(?im)(?:final\s+answer|correct\s+option|answer)\s*[:\-]?\s*([A-Z])\b",
            s,
        )
        if m:
            return m.group(1).upper()
        m = re.fullmatch(
            r"(?im)\s*(?:(?:final\s+answer|correct\s+option|answer))\s*[:\-]?\s*([A-Z])\s*[\s.]*",
            s,
        )
        if m:
            return m.group(1).upper()
        m = re.fullmatch(
            r"(?im)\s*(?:(?:option|choice))\s*[:\-]?\s*([A-Z])\s*[\s.)]*",
            s,
        )
        if m:
            return m.group(1).upper()
        # Sometimes the boxed content is already just "B".
        m = re.fullmatch(r"(?im)\s*([A-Z])\s*", s)
        if m:
            return m.group(1).upper()
        return None

    patterns = [
        r"(?im)^\s*(?:final\s+answer|answer)\s*:\s*(.+?)\s*$",
        r"(?im)^\s*(?:the\s+correct\s+option|correct\s+option)\s+is\s*[:\-]?\s*(.+?)\s*$",
        r"(?im)^\s*(?:option|choice)\s*[:\-]?\s*([A-Z])\s*$",
        r"\\boxed\{([^}]+)\}",
    ]

    for pattern in patterns:
        matches = re.findall(pattern, submission)
        if matches:
            extracted = matches[-1].strip()
            if extracted:
                normalized_letter = _maybe_extract_option_letter(extracted)
                if normalized_letter is not None:
                    return normalized_letter
                return extracted

    lines = [line.strip() for line in submission.splitlines() if line.strip()]
    if not lines:
        return ""

    for line in reversed(lines[-5:]):
        option_match = re.fullmatch(
            r"(?:option|choice)?\s*[:\-]?\s*([A-Z])(?:[.)])?",
            line,
            flags=re.IGNORECASE,
        )
        if option_match:
            return option_match.group(1).upper()

    return ""


async def evaluate_with_llm_judge(
    submission: str,
    target: str,
    llm_model: str | InspectModel = INSPECT_JUDGE_LLM,
    gen_cfg: Dict[str, Any] = INSPECT_JUDGE_LLM_GEN_CONFIG,
) -> bool:
    """
    Evaluate the submission using an LLM judge.

    This function uses the LLM judge to determine if
    the submission aligns with the target.

    Args
    ----
        submission (str): The submission string to evaluate.
        target (str): The target answer string.
        llm_model (str | InspectModel): The LLM model to use for evaluation.
        gen_cfg (Dict[str, Any]): The generation configuration for the LLM.

    Returns
    -------
        bool: True if the submission is correct, False otherwise.
    """
    prompt = LLM_JUDGE_PROMPT.format(
        submission=submission,
        target=target,
    )
    result = await get_model(
        model=llm_model,
        base_url=os.getenv("INSPECT_JUDGE_LLM_BASE_URL", None),
        api_key=os.getenv("INSPECT_JUDGE_LLM_API_KEY", None),
    ).generate(
        input=prompt,
        config=InspectGenerateConfig(
            temperature=gen_cfg.get("temperature"),
            max_tokens=gen_cfg.get("max_tokens"),
        ),
    )
    return bool(result.completion.lower() == "yes")
