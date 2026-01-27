"""
The capability_utils module for the automatic_benchmark_generation project.

It contains utility functions for capabilities.
"""

import json
import logging
import random
from typing import Any, Dict, List, Tuple

import numpy as np
from inspect_ai.scorer import CORRECT

from src.utils import constants
from src.utils.data_utils import read_json_file


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

    Args
    ----
        json_file (str): The path to the JSON file.
        num_tasks (int): The number of tasks to score. If -1, all tasks are used.
        seed (int): The random seed for selecting tasks.

    Returns
    -------
        Dict[str, float]: A dictionary containing the mean and standard error
            of the scores.
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
        scorer_name (str): The name of the scorer to use.

    Returns
    -------
        Tuple[float, float]: A tuple containing the mean score and
            the standard error of the scores.
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
