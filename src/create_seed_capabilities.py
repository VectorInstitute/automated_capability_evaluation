"""Create seed capabilities for the mathematics and GSM8K datasets."""

import json
import logging
import os
import random
import shutil
from collections import defaultdict
from typing import Any, Dict, List

import hydra
from omegaconf import DictConfig

from capability import CapabilitySeedDataset
from utils import constants, templates


logger = logging.getLogger(__name__)


def populate_seed_capability_dir(
    base_dir: str,
    capability_name: str,
    capability_description: str,
    capability_domain: str,
    capability_data: List[Dict[str, str]],
    capability_repr_tasks: List[Dict[str, str]],
    capability_instructions: str,
    capability_score_func: str,
    source_dataset: str,
    capability_subject: str | None = None,
) -> None:
    """
    Populate a directory with seed capability files.

    Create a JSON configuration and a Python script.

    Args
    ----
        base_dir (str): The base directory where the capability directory
            will be created.
        capability_name (str): The name of the capability.
        capability_description (str): A description of the capability.
        capability_domain (str): The domain to which the capability belongs.
        capability_data (List[Dict]): A list of dictionaries containing capability data.
        capability_repr_tasks (List[Dict]): A list of dictionaries
        containing representative tasks for the capability.
        capability_instructions (str): Instructions for the capability.
        capability_score_func (str): The scoring function for the capability.
        source_dataset (str): The name of the source dataset.
        capability_subject (str | None): The subject of the capability.
    """
    # Create capability dir
    capability_dir = os.path.join(base_dir, capability_name)
    os.makedirs(capability_dir, exist_ok=True)

    # Create capability json
    capability_json = {
        "capability_name": capability_name,
        "capability_description": capability_description,
        "source_dataset": source_dataset,
        "capability_domain": capability_domain,
        "capability_instructions": capability_instructions,
        "capability_data": capability_data,
    }
    if capability_subject:
        capability_json.update({"capability_subject": capability_subject})
    with open(os.path.join(capability_dir, "capability.json"), "w") as f:
        json.dump(capability_json, f, indent=4)

    # Create capability utils file, which includes utils for evaluation
    # Copy the contents of `utils/inspect_eval_utils.py` here
    shutil.copyfile(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "utils/inspect_eval_utils.py",
        ),
        os.path.join(capability_dir, "utils.py"),
    )

    # Create capability python file
    capability_tasks_dict = {
        f"{idx + 1}": task for idx, task in enumerate(capability_repr_tasks)
    }
    capability_class_str = templates.CAPABILITY_CLASS_TEMPLATE.format(
        capability_tasks_dict=json.dumps(capability_tasks_dict, indent=4),
        capability_instructions=capability_instructions,
        capability_score_func=capability_score_func,
    ).lstrip("\n")
    with open(os.path.join(capability_dir, "capability.py"), "w") as f:
        f.write(capability_class_str)

    # Create _state.json file with task generation completed state
    with open(
        os.path.join(capability_dir, "_state.json"),
        "w",
    ) as f:
        json.dump(
            {
                "state": constants.C_STATE_TASK_GENERATION_COMPLETED_STR,
            },
            f,
            indent=4,
        )


# Next 2 functions borrowed from: https://github.com/UKGovernmentBEIS/inspect_evals/blob/main/src/inspect_evals/mathematics/utils.py#L398C1-L441C18
def remove_boxed(s: str) -> str:
    r"""
    Remove the LaTeX boxed notation from the given string.

    This function handles two cases:
    1. If the string starts with "\\boxed ", it removes this prefix.
    2. If the string starts with "\\boxed{" and ends with "}", it removes these
       enclosing characters.

    Args
    ----
        s (str): The input string containing the LaTeX boxed notation.

    Returns
    -------
        str: The string with the LaTeX boxed notation removed.

    Raises
    ------
        AssertionError: If the input string does not start with the expected
                        LaTeX boxed notation.
    """
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]

    left = "\\boxed{"

    assert s[: len(left)] == left
    assert s[-1] == "}"

    return s[len(left) : -1]


def last_boxed_only_string(string: str) -> str | None:
    r"""
    Extract the last boxed substring from a given string.

    This function searches for the last occurrence of
    either "\\boxed" or "\\fbox" in the input string.
    If found, it returns the substring enclosed within
    the last occurrence of these box commands.
    If no such boxed substring is found, it returns None.

    Args
    ----
        string (str): The input string to search for boxed substrings.

    Returns
    -------
        str | None: The last boxed substring if found, otherwise None.
    """
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    return None if right_brace_idx is None else string[idx : right_brace_idx + 1]


@hydra.main(version_base=None, config_path="cfg", config_name="run_cfg")
def main(cfg: DictConfig) -> None:
    """
    Create seed capabilities based on the provided configuration.

    Args
    ----
        cfg (DictConfig): Configuration object containing capability settings.

    The function processes capabilities from the configuration and
    creates seed capabilities for different datasets.
    It supports the following datasets:
    - "mathematics": Processes mathematics dataset, organizes them by subject,
      and populates the seed capability directory with descriptions, instructions,
      and representative tasks.
    - "gsm8k": Processes GSM8K dataset, extracts solutions and answers, and populates
      the seed capability directory with descriptions, instructions, and representative
      tasks.

    The function uses a fixed random seed for reproducibility and prints the number
    of tasks created for each capability.
    """
    random.seed(42)

    domain = cfg.capabilities_cfg.domain
    seed_capability_dir = os.path.join(
        cfg.capabilities_cfg.capabilities_dir, "seed_capabilities", domain
    )

    for dataset_cfg in cfg.capabilities.capability_cfgs.values():
        dataset = CapabilitySeedDataset(dataset_cfg)
        if dataset.name == "mathematics":
            capabilities: Dict[str, Dict[str, Any]] = defaultdict()
            for task in dataset._data:
                subject = task["type"].lower()
                capability_name = f"{'_'.join(subject.split(' '))}".replace("&", "and")
                if capability_name not in capabilities:
                    capabilities[capability_name] = defaultdict()
                    capabilities[capability_name]["type"] = subject
                    capabilities[capability_name]["tasks"] = []
                task["answer"] = remove_boxed(
                    last_boxed_only_string(task["solution"]) or task["solution"]
                )
                capabilities[capability_name]["tasks"].append(task)

            for capability_name, math_tasks in capabilities.items():
                subject = math_tasks["type"]
                capability_desc = dataset.description.format(
                    name=capability_name,
                    size=len(math_tasks["tasks"]),
                    subject=subject,
                )
                capability_instructions = dataset.instructions.format(
                    subject=subject, problem='{t["problem"]}'
                )

                # Create task IDs
                tasks = []
                for task_id, task in enumerate(math_tasks["tasks"]):
                    task["id"] = str(task_id + 1)
                    tasks.append(task)

                # Create representative tasks by selecting the tasks in order
                # Only keep problem and answer
                capability_repr_tasks = [
                    {"problem": s["problem"], "answer": s["answer"]}
                    for s in tasks[: dataset._cfg["data_args"]["num_repr_tasks"]]
                ]

                populate_seed_capability_dir(
                    base_dir=seed_capability_dir,
                    capability_name=capability_name,
                    capability_description=capability_desc,
                    capability_domain=dataset.domain,
                    capability_subject=subject,
                    capability_data=tasks,
                    capability_repr_tasks=capability_repr_tasks,
                    capability_instructions=capability_instructions,
                    capability_score_func=constants.MATHEMATICS_SCORE_FUNC.strip("\n"),
                    source_dataset=dataset.name,
                )
                logger.info(
                    f"Created capability {capability_name} with {len(math_tasks['tasks'])} tasks."
                )
        elif dataset.name == "gsm8k":
            capability_name = f"{constants.DATASET_NAME_MAP[dataset.name]}"
            # Create task IDs, answers and replace "question" with "problem"
            gsm_tasks = []
            for task_id, task in enumerate(dataset._data):
                task["id"] = str(task_id + 1)
                task["problem"] = task.pop("question")
                task["solution"] = task["answer"]
                task["answer"] = task["answer"].split("####").pop().strip()
                gsm_tasks.append(task)

            capability_instructions = dataset.instructions.format(
                problem='{t["problem"]}'
            )

            # Create representative tasks by selecting the tasks in order
            # Only keep problem and answer
            capability_repr_tasks = [
                {"problem": s["problem"], "answer": s["answer"]}
                for s in tasks[: dataset._cfg["data_args"]["num_repr_tasks"]]
            ]

            populate_seed_capability_dir(
                base_dir=seed_capability_dir,
                capability_name=capability_name,
                capability_description=dataset.description,
                capability_domain=dataset.domain,
                capability_data=gsm_tasks,
                capability_repr_tasks=capability_repr_tasks,
                capability_instructions=capability_instructions,
                capability_score_func=constants.GSM8K_SCORE_FUNC.strip("\n"),
                source_dataset=dataset.name,
            )
            logger.info(
                f"Created capability {capability_name} with {len(gsm_tasks)} tasks."
            )


if __name__ == "__main__":
    main()
