"""
The capability_utils module for the automatic_benchmark_generation project.

It contains utility functions for capabilities.
"""

import json
import os
from typing import Any, Dict

from inspect_ai import eval as inspect_eval
from langsmith import traceable

from src.model import Model
from src.utils.constants import DEFAULT_OPENAI_BASE_URL
from src.utils.data_utils import read_json_file
from src.utils.inspect_eval_utils import INSPECT_JUDGE_LLM


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

    judge_llm_name = kwargs.pop("judge_llm_name", INSPECT_JUDGE_LLM)

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
        if model.model_provider == "local":
            # Set OPENAI_BASE_URL to local model URL and replace "local" with "openai"
            # See: https://inspect.aisi.org.uk/providers.html#vllm-server
            # TODO: How to ensure this doesn't affect other processes
            # if running in parallel?
            os.environ["OPENAI_BASE_URL"] = model.model_url
            inspect_model_name = model_name.replace("local", "openai")
        else:
            inspect_model_name = model_name
        eval_log = inspect_eval(
            tasks=path,
            model=inspect_model_name,
            log_dir=log_dir,
            log_format="json",
            **kwargs,
        )[0]
        # Return usage stats
        eval_model_usage = eval_log.stats.model_usage[inspect_model_name]
        # [IMP] TODO: How to track usage for judge llm?
        usage_metadata = {
            "input_tokens": eval_model_usage.input_tokens,
            "output_tokens": eval_model_usage.output_tokens,
            "total_tokens": eval_model_usage.total_tokens,
            "reasoning_tokens": eval_model_usage.reasoning_tokens,
            "judge_llm_usage": eval_log.stats.model_usage[judge_llm_name],
        }
        if model.model_provider == "local":
            # Reset OPENAI_BASE_URL to actual openai URL
            os.environ["OPENAI_BASE_URL"] = os.getenv(
                "ORIGINAL_OPENAI_BASE_URL", DEFAULT_OPENAI_BASE_URL
            )
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
