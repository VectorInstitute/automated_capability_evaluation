"""
The capability_utils module for the automated_capability_evaluation project.

It contains utility functions for capabilities.
"""

import json
import logging
import os
import random
from typing import Any, Dict, List, Tuple

import numpy as np
from inspect_ai import eval as inspect_eval
from inspect_ai.scorer import CORRECT
from langsmith import traceable, tracing_context

from src.model import Model
from src.utils import constants
from src.utils.data_utils import read_json_file
from src.utils.inspect_eval_utils import INSPECT_JUDGE_LLM


CAPABILITY_SCORER_MAP = {
    "math": "expression_equivalence",
    "gsm8k": "match",
}


logger = logging.getLogger(__name__)


def read_score_inspect_json(
    json_file: str, num_tasks: int = -1, seed: int = constants.DEFAULT_RANDOM_SEED
) -> Dict[str, float]:
    """
    Read a JSON file containing scores.

    Args:
        json_file (str): The path to the JSON file.

    Returns
    -------
        float: The score value.
    """
    random.seed(seed)

    scores = read_json_file(json_file)
    all_tasks = scores["samples"]

    def clean_name(x: str) -> str:
        return x.split("/")[-1]

    capability_name = (
        clean_name(scores["eval"]["master_task"])
        if "master_task" in scores["eval"]
        else clean_name(scores["eval"]["task"])
    )
    scorer_name = CAPABILITY_SCORER_MAP.get(
        capability_name, constants.DEFAULT_INSPECT_SCORER_NAME
    )

    if num_tasks == -1:
        num_tasks = len(all_tasks)
    elif num_tasks == 0:
        logger.warning(
            f"[{capability_name}] Number of tasks for scoring is zero. Using all tasks for scoring."
        )
        num_tasks = len(all_tasks)
    elif num_tasks > len(all_tasks):
        logger.warning(
            f"[{capability_name}] Number of tasks ({num_tasks}) for scoring is greater than or equal to total existing tasks ({len(all_tasks)}). Using all tasks for scoring."
        )
        num_tasks = len(all_tasks)

    # Select num_tasks at random for scoring
    selected_tasks = random.sample(all_tasks, num_tasks)
    # Get mean and std error for selected tasks
    mean, std_err = get_inspect_score(selected_tasks, scorer_name)

    return {"mean": float(mean), "std_err": float(std_err)}


def get_inspect_score(
    tasks: List[Dict[str, Any]], scorer_name: str
) -> Tuple[float, float]:
    """
    Get the inspect score for a list of tasks.

    Args
    ----
        tasks (List[Dict[str, Any]]): The list of tasks to score.
        score_func (str): The scoring function to use.

    Returns
    -------
        float: The inspect score.
    """
    if not tasks:
        return (0.0, 0.0)

    n = len(tasks)
    scores = []
    for task_id, task in enumerate(tasks):
        try:
            scores.append(int(task["scores"][scorer_name]["value"] == CORRECT))
        except Exception as e:
            logger.warning(f"Error obtaining score for task {task_id}: {repr(e)}")
            scores.append(0)

    # Calculate the mean score
    score_mean = np.mean(scores)

    # Calculate the standard error
    # Borrowed from: https://github.com/UKGovernmentBEIS/inspect_ai/blob/76266abac6b84fca380226691d60eccf0fd6e0ca/src/inspect_ai/scorer/_metrics/std.py#L114C5-L129C43
    if (n - 1) < 1:
        score_std_err = 0.0
    score_std = np.std(scores, ddof=1)
    score_std_err = score_std / np.sqrt(n)

    return (float(score_mean), float(score_std_err))


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


def extract_and_parse_response(
    response: str,
    has_thought: bool = True,
    parse_kw: str = "RESPONSE JSON",
    response_type: str = "json",
) -> Dict[str, Any]:
    """
    Extract the thought string and response JSON data from the response string.

    Args
    ----
        response (str): The response string containing the thought and JSON data.
        has_thought (bool): Whether the response contains a thought string.
        parse_kw (str): The keyword to split the response string for JSON parsing.
        response_type (str): The type of response to parse (default is "json").

    Returns
    -------
        Dict[str, Any]: A dictionary with two keys:
            - "thought" (str, optional): The extracted thought string if present.
            - "parsed_response" (Union[List[Any], str]): The parsed response data.

    Raises
    ------
        ValueError: If there is an error parsing the thought or JSON data.
    """
    if has_thought:
        try:
            thought_str = (
                response.split("THOUGHT:")[1].split(parse_kw)[0].strip().strip("\n")
            )
        except Exception as e:
            logger.error(f"Error parsing thought string: {repr(e)}")
            logger.error(f"Response: {response}")
            raise

    try:
        try:
            response_str = response.split(f"{parse_kw}:")[1].strip().strip("\n")
        except IndexError as e:
            if "list index out of range" in str(e):
                # Handle case where parse_kw is not found
                logger.warning(
                    f"Parse keyword '{parse_kw}' not found in response. "
                    "Assuming the entire response is the JSON data starting with '```json' and ending in '```'."
                )
                response_str = (
                    "{"
                    + response.split("```json\n{")[1].split("}\n```")[0].strip()
                    + "}"
                )
            else:
                logger.error(
                    f"Parse keyword '{parse_kw}' not found in response: {repr(e)}"
                )
                raise
        if response_type == "json":
            response_json = json.loads(response_str)
            parsed_response_list = []
            for _, v in response_json.items():
                parsed_response_list.append(v)
        elif response_type == "str_yes_no":
            parsed_response_str = response_str.lower()
            assert parsed_response_str in ["yes", "no"], (
                f"Invalid response: {parsed_response_str}. Expected 'yes' or 'no'."
            )
        else:
            raise ValueError(f"Unsupported response type: {response_type}")
    except Exception as e:
        logger.error(f"Error parsing response json: {repr(e)}")
        logger.error(f"Response: {response}")
        raise

    output: Dict[str, Any] = {}
    output["thought"] = thought_str if has_thought else None
    if response_type == "json":
        output.update({"parsed_response": parsed_response_list})
    elif response_type == "str_yes_no":
        output.update({"parsed_response": parsed_response_str})

    return output


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
    # Create langsmith metadata
    model_name = model.get_model_name(with_provider=True)
    ls_metadata: Dict[str, Any] = {
        "ls_provider": model.model_provider,
        "ls_model_name": model.get_model_name(with_provider=False),
        "ls_model_type": "chat",
        "exp_id": kwargs.pop("run_id", None),
        "capability_name": path,
        "log_dir": log_dir,
    }
    ls_metadata.update({f"ls_{k}": v for k, v in kwargs.items()})

    @traceable(
        run_type="llm",
    )
    def _run_inspect_evals() -> Dict[str, Any]:
        """
        Run the inspect evals command for a given capability and model.

        Local function to enable tracing using langsmith.
        """
        logger.info(f"Running inspect evals for {path} capability using {model_name}")
        eval_log = inspect_eval(
            tasks=path,
            model=inspect_model_name,
            log_dir=log_dir,
            log_format="json",
            **kwargs,
        )[0]
        # Return usage stats
        if inspect_model_name in eval_log.stats.model_usage:
            eval_model_usage = eval_log.stats.model_usage[inspect_model_name]
            # [IMP] TODO: How to track usage for judge llm?
            usage_metadata = {
                "input_tokens": eval_model_usage.input_tokens,
                "output_tokens": eval_model_usage.output_tokens,
                "total_tokens": eval_model_usage.total_tokens,
                "reasoning_tokens": eval_model_usage.reasoning_tokens,
                "judge_llm_usage": eval_log.stats.model_usage.get(judge_llm_name, None),
            }
        else:
            usage_metadata = {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "reasoning_tokens": 0,
                "judge_llm_usage": None,
            }
        return {
            "inspect_eval_log": eval_log,
            "usage_metadata": usage_metadata,
        }

    judge_llm_name = kwargs.pop("judge_llm_name", INSPECT_JUDGE_LLM)
    if model.model_provider == "local":
        # Set OPENAI_BASE_URL to local model URL and replace "local" with "openai"
        # See: https://inspect.aisi.org.uk/providers.html#vllm-server
        # TODO: How to ensure this doesn't affect other processes
        # if running in parallel?
        os.environ["OPENAI_BASE_URL"] = model.model_url
        inspect_model_name = model_name.replace("local", "openai")
    else:
        inspect_model_name = model_name

    with tracing_context(
        enabled=True,
        tags=["run_inspect_evals"],
        metadata=ls_metadata,
    ):
        output = _run_inspect_evals()

    if model.model_provider == "local":
        # Reset OPENAI_BASE_URL to actual openai URL
        os.environ["OPENAI_BASE_URL"] = os.getenv(
            "ORIGINAL_OPENAI_BASE_URL", constants.DEFAULT_OPENAI_BASE_URL
        )

    eval_log = output["inspect_eval_log"]
    if eval_log.status == "error":
        # TODO: Add option to retry with limit
        raise ValueError(
            f"Error running inspect evals for {path} capability using {model_name}: {eval_log.error}"
        )

    # Analyze tokens metadata for evaluation
    usage_metadata = output["usage_metadata"]
    tokens_summary = {
        "total_input_tokens": usage_metadata["input_tokens"],
        "total_output_tokens": usage_metadata["output_tokens"],
        "total_tokens": usage_metadata["total_tokens"],
        "input_tokens_per_task": usage_metadata["input_tokens"] / len(eval_log.samples),
        "output_tokens_per_task": usage_metadata["output_tokens"]
        / len(eval_log.samples),
        "total_tokens_per_task": usage_metadata["total_tokens"] / len(eval_log.samples),
    }
    logger.info(
        f"[{path}] Task evaluation tokens summary:\n{json.dumps(tokens_summary, indent=4)}"
    )
