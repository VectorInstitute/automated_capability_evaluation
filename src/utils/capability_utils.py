"""
The capability_utils module for the automatic_benchmark_generation project.

It contains utility functions for capabilities.
"""

import json
from typing import Any, Dict


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
    with open(json_file, "r") as f:
        scores = json.load(f)

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
