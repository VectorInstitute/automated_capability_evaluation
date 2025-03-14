import json  # noqa: D100
import os
import random
from collections import defaultdict
from typing import Any, Dict, List

import hydra  # noqa: D100
from omegaconf import DictConfig

from task import TaskSeedDataset
from utils.templates import TASK_CLASS_TEMPLATE


def populate_seed_task_dir(
    base_dir: str,
    task_name: str,
    task_description: str,
    task_domain: str,
    task_family: str,
    task_data: List[Dict[str, str]],
    task_repr_samples: List[Dict[str, str]],
    task_instructions: str,
    task_score_func: str,
    source_dataset: str,
    task_subject: str | None = None,
) -> None:
    """
    Populate a directory with seed task files.

    Create a JSON configuration and a Python script.

    Args:
        base_dir (str): The base directory where the task directory will be created.
        task_name (str): The name of the task.
        task_description (str): A description of the task.
        task_domain (str): The domain to which the task belongs.
        task_family (str): The family to which the task belongs.
        task_data (List[Dict]): A list of dictionaries containing task data.
        task_repr_samples (List[Dict]): A list of dictionaries
        containing representative samples for the task.
        task_instructions (str): Instructions for the task.
        task_score_func (str): The scoring function for the task.
        source_dataset (str): The name of the source dataset.

    Returns
    -------
        None
    """
    # Create task dir
    task_dir = os.path.join(base_dir, task_name)
    os.makedirs(task_dir, exist_ok=True)

    # Create task json
    task_json = {
        "task_name": task_name,
        "task_description": task_description,
        "source_dataset": source_dataset,
        "task_domain": task_domain,
        "task_family": task_family,
        "task_instructions": task_instructions,
        "task_data": task_data,
    }
    if task_subject:
        task_json.update({"task_subject": task_subject})
    with open(os.path.join(task_dir, "task.json"), "w") as f:
        json.dump(task_json, f, indent=4)

    # Create task python file
    task_data_samples_dict = {
        f"{idx + 1}": sample for idx, sample in enumerate(task_repr_samples)
    }
    task_class_str = TASK_CLASS_TEMPLATE.format(
        task_data_samples_dict=json.dumps(task_data_samples_dict, indent=4),
        task_instructions=task_instructions,
        task_score_func=task_score_func,
    ).lstrip("\n")
    with open(os.path.join(task_dir, "task.py"), "w") as f:
        f.write(task_class_str)


# Next 2 functions borrowed from: https://github.com/UKGovernmentBEIS/inspect_evals/blob/main/src/inspect_evals/mathematics/utils.py#L398C1-L441C18
def remove_boxed(s: str) -> str:
    r"""
    Remove the LaTeX boxed notation from the given string.

    This function handles two cases:
    1. If the string starts with "\\boxed ", it removes this prefix.
    2. If the string starts with "\\boxed{" and ends with "}", it removes these
       enclosing characters.

    Args:
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

    Args:
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


tab_w_spaces = "    "
# Score function is based on https://github.com/UKGovernmentBEIS/inspect_evals/blob/main/src/inspect_evals/mathematics/utils.py#L57
mathematics_score_func = f"""def score(t: dict, submission: str) -> float | None:\n{tab_w_spaces}{tab_w_spaces}ans_pattern_line = r"(?i)ANSWER\\s*:\\s*([^\\n]+)"\n{tab_w_spaces}{tab_w_spaces}match = re.search(ans_pattern_line, submission)\n{tab_w_spaces}{tab_w_spaces}if match:\n{tab_w_spaces}{tab_w_spaces}{tab_w_spaces}answer = match.group(1)\n{tab_w_spaces}{tab_w_spaces}{tab_w_spaces}correct = is_equiv(answer, t["answer"])\n{tab_w_spaces}{tab_w_spaces}else:\n{tab_w_spaces}{tab_w_spaces}{tab_w_spaces}correct = False\n{tab_w_spaces}{tab_w_spaces}return 1.0 if correct else 0.0"""
# Score function is based on https://github.com/UKGovernmentBEIS/inspect_evals/blob/main/src/inspect_evals/mathematics/utils.py#L57
gsm8k_score_func = f"""def score(t: dict, submission: str) -> float | None:\n{tab_w_spaces}{tab_w_spaces}return 1.0 if submission==t["answer"] else 0.0"""


@hydra.main(version_base=None, config_path="cfg", config_name="run_cfg")
def main(cfg: DictConfig) -> None:
    """
    Create seed tasks based on the provided configuration.

    Args:
        cfg (DictConfig): Configuration object containing task settings.

    The function processes tasks from the configuration and creates seed tasks
    for different datasets. It supports the following datasets:
    - "mathematics": Processes mathematical tasks, organizes them by subject,
      and populates the seed task directory with task descriptions, instructions,
      and representative samples.
    - "gsm8k": Processes GSM8K tasks, extracts solutions and answers, and populates
      the seed task directory with task descriptions, instructions, and representative
      samples.

    The function uses a fixed random seed for reproducibility and prints the number
    of samples created for each task.
    """
    random.seed(42)

    domain = cfg.tasks_cfg.domain
    seed_task_dir = os.path.join(cfg.tasks_cfg.tasks_dir, "seed_tasks", domain)

    for dataset_cfg in cfg.tasks.task_cfgs.values():
        dataset = TaskSeedDataset(dataset_cfg)
        if dataset.name == "mathematics":
            tasks: Dict[str, Dict[str, Any]] = defaultdict()
            for sample in dataset._data:
                subject = sample["type"].lower()
                task_name = (
                    f"{dataset.domain}_{dataset.family}_{'_'.join(subject.split(' '))}"
                )
                if task_name not in tasks:
                    tasks[task_name] = defaultdict()
                    tasks[task_name]["type"] = subject
                    tasks[task_name]["samples"] = []
                sample["answer"] = remove_boxed(
                    last_boxed_only_string(sample["solution"]) or sample["solution"]
                )
                tasks[task_name]["samples"].append(sample)

            for task_name, math_samples in tasks.items():
                subject = math_samples["type"]
                task_desc = dataset.description.format(
                    name=task_name,
                    size=len(math_samples["samples"]),
                    subject=subject,
                )
                task_instructions = dataset.instructions.format(
                    subject=subject, problem='{t["problem"]}'
                )

                task_repr_samples = random.sample(
                    math_samples["samples"],
                    dataset._cfg["data_args"]["num_repr_samples"],
                )
                # Only keep problem and answer
                task_repr_samples = [
                    {"problem": s["problem"], "answer": s["answer"]}
                    for s in task_repr_samples
                ]

                populate_seed_task_dir(
                    base_dir=seed_task_dir,
                    task_name=task_name,
                    task_description=task_desc,
                    task_domain=dataset.domain,
                    task_family=dataset.family,
                    task_subject=subject,
                    task_data=math_samples["samples"],
                    task_repr_samples=task_repr_samples,
                    task_instructions=task_instructions,
                    task_score_func=gsm8k_score_func.strip(
                        "\n"
                    ),  # TODO: Change this to mathematics_score_func after figuring out how to implement complex score functions
                    source_dataset=dataset.name,
                )
                print(
                    f"Created task {task_name} with {len(math_samples['samples'])} samples."
                )
        elif dataset.name == "gsm8k":
            task_name = f"{dataset.domain}_{dataset.family}"
            gsm_samples = []
            for sample in dataset._data:
                sample["solution"] = sample["answer"]
                sample["answer"] = sample["answer"].split("####").pop().strip()
                gsm_samples.append(sample)

            task_instructions = dataset.instructions.format(problem='{t["question"]}')

            task_repr_samples = random.sample(
                gsm_samples, dataset._cfg["data_args"]["num_repr_samples"]
            )
            # Only keep question and answer
            task_repr_samples = [
                {"question": s["question"], "answer": s["answer"]}
                for s in task_repr_samples
            ]

            populate_seed_task_dir(
                base_dir=seed_task_dir,
                task_name=task_name,
                task_description=dataset.description,
                task_domain=dataset.domain,
                task_family=dataset.family,
                task_data=gsm_samples,
                task_repr_samples=task_repr_samples,
                task_instructions=task_instructions,
                task_score_func=gsm8k_score_func.strip("\n"),
                source_dataset=dataset.name,
            )
            print(f"Created task {task_name} with {len(gsm_samples)} samples.")


if __name__ == "__main__":
    main()
