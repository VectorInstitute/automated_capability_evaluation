"""
The capability_utils module for the automatic_benchmark_generation project.

It contains utility functions for capabilities.
"""

import json


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
