"""
The capability_utils module for the automatic_benchmark_generation project.

It contains utility functions for capabilities.
"""

import json
import re
from typing import Any, Dict

from inspect_ai import eval as inspect_eval
from inspect_ai.model import GenerateConfig as InspectGenerateConfig
from inspect_ai.model import Model as InspectModel
from inspect_ai.model import get_model
from langsmith import traceable

from src.model import Model
from src.utils.constants import DEFAULT_INSPECT_GRADER_MODEL
from src.utils.data_utils import read_json_file
from src.utils.prompts import LLM_JUDGE_PROMPT


CAPABILITY_SCORER_MAP = {
    "math": "expression_equivalence",
    "gsm8k": "match",
}


def read_score_inspect_json(json_file: str) -> float:
    """
    Read a JSON file containing scores.

    Args:
        json_file (str): The path to the JSON file.

    Returns
    -------
        float: The score value.
    """
    scores = read_json_file(json_file)

    def clean_name(x: str) -> str:
        return x.split("/")[-1]

    capability_name = (
        clean_name(scores["eval"]["master_task"])
        if "master_task" in scores["eval"]
        else clean_name(scores["eval"]["task"])
    )
    scorer_name = CAPABILITY_SCORER_MAP.get(capability_name, "match")
    scores = [elm for elm in scores["results"]["scores"] if elm["name"] == scorer_name][
        0
    ]
    return float(scores["metrics"]["accuracy"]["value"])


def parse_python_class_str(class_str: str) -> str:
    """
    Parse the python class string and return the formatted class string.

    Args
    ----
        class_str (str): The class string to parse with python tag.

    Returns
    -------
        str: The formatted class string.
    """
    return class_str.split("```python\n")[1].split("\n```")[0].strip()


def extract_and_parse_response(response: str) -> Dict[str, Any]:
    """
    Extract the thought string and response JSON data from the response string.

    Args
    ----
        response (str): The response string containing the thought and JSON data.

    Returns
    -------
        Dict[str, Any]: A dictionary with two keys:
            - "thought" (str): The extracted thought string.
            - "parsed_response" (List[Any]): A list of parsed JSON objects.

    Raises
    ------
        ValueError: If there is an error parsing the thought or JSON data.
    """
    try:
        thought_str = (
            response.split("THOUGHT:")[1].split("RESPONSE JSON")[0].strip().strip("\n")
        )
    except (IndexError, json.JSONDecodeError) as e:
        print(f"Error parsing thought string: {e}")
        raise

    try:
        response_str = response.split("RESPONSE JSON:\n")[1].strip().strip("\n")
        response_json = json.loads(response_str)
        parsed_response = []
        for _, v in response_json.items():
            parsed_response.append(v)
    except (IndexError, json.JSONDecodeError) as e:
        print(f"Error parsing capabilities json: {e}")
        raise

    return {"thought": thought_str, "parsed_response": parsed_response}


def run_inspect_evals(path: str, model: Model, log_dir: str, **kwargs: Any) -> None:
    """
    Run the inspect evals command for a given capability and model.

    Args
    ----
    path : str
        The path to the evaluation file for the capability.
    model : Model
        The model object to evaluate.
    log_dir : str
        The directory to store evaluation logs.
    kwargs : Any
        Additional arguments for the command.

    Returns
    -------
    None
    """
    model_name = model.get_model_name(with_provider=True)

    # Create langsmith metadata
    ls_metadata = {
        "ls_provider": model.model_provider,
        "ls_model_name": model.get_model_name(with_provider=False),
        "ls_model_type": "chat",
    }
    ls_metadata.update({f"ls_{k}": v for k, v in kwargs.items()})

    @traceable(
        run_type="llm",
        metadata=ls_metadata,
    )
    def _run_inspect_evals() -> Dict[str, Any]:
        """
        Run the inspect evals command for a given capability and model.

        Local function to enable tracing using langsmith.
        """
        print(f"Running inspect evals for {path} capability using {model_name}")
        eval_log = inspect_eval(
            tasks=path,
            model=model_name,
            log_dir=log_dir,
            log_format="json",
            **kwargs,
        )[0]
        # Return usage stats
        eval_model_usage = eval_log.stats.model_usage[model_name]
        usage_metadata = {
            "input_tokens": eval_model_usage.input_tokens,
            "output_tokens": eval_model_usage.output_tokens,
            "total_tokens": eval_model_usage.total_tokens,
            "reasoning_tokens": eval_model_usage.reasoning_tokens,
        }
        return {
            "inspect_eval_log": eval_log,
            "usage_metadata": usage_metadata,
        }

    output = _run_inspect_evals()

    eval_log = output["inspect_eval_log"]
    if eval_log.status == "error":
        # TODO: Add option to retry with limit
        raise ValueError(
            f"Error running inspect evals for {path} capability using {model_name}: {eval_log.error}"
        )


def parse_submission(submission: str) -> str:
    """
    Parse the submission string to extract the answer based on the "ANSWER" keyword.

    This function is used in the capability class score method.

    Args
    ----
        submission (str): The submission string to parse.

    Returns
    -------
        str: The extracted answer from the submission, or an empty string
            if no match is found.
    """
    answer_pattern = r"(?i)ANSWER\s*:\s*([^\n]+)"
    match = re.search(answer_pattern, submission)
    return match.group(1) if match else ""


async def evaluate_with_llm_judge(
    submission: str,
    target: str,
    llm_model: str | InspectModel = DEFAULT_INSPECT_GRADER_MODEL,
    **kwargs: Any,
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
        **kwargs: Additional arguments for the LLM model.

    Returns
    -------
        bool: True if the submission is correct, False otherwise.
    """
    prompt = LLM_JUDGE_PROMPT.format(
        submission=submission,
        target=target,
    )
    result = await get_model(llm_model).generate(
        input=prompt,
        config=InspectGenerateConfig(
            temperature=kwargs.get("temperature"),
            max_tokens=kwargs.get("max_tokens"),
        ),
    )
    return bool(result.completion.lower() == "yes")
