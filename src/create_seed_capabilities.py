import json  # noqa: D100
import os
import random
from collections import defaultdict
from typing import Any, Dict, List

import hydra  # noqa: D100
from omegaconf import DictConfig

from capability import CapabilitySeedDataset
from utils.templates import CAPABILITY_CLASS_TEMPLATE


def populate_seed_capability_dir(
    base_dir: str,
    capability_name: str,
    capability_description: str,
    capability_domain: str,
    capability_family: str,
    capability_data: List[Dict[str, str]],
    capability_repr_samples: List[Dict[str, str]],
    capability_instructions: str,
    capability_score_func: str,
    source_dataset: str,
    capability_subject: str | None = None,
) -> None:
    """
    Populate a directory with seed capability files.

    Create a JSON configuration and a Python script.

    Args:
        base_dir (str): The base directory where the capability directory
            will be created.
        capability_name (str): The name of the capability.
        capability_description (str): A description of the capability.
        capability_domain (str): The domain to which the capability belongs.
        capability_family (str): The family to which the capability belongs.
        capability_data (List[Dict]): A list of dictionaries containing capability data.
        capability_repr_samples (List[Dict]): A list of dictionaries
        containing representative samples for the capability.
        capability_instructions (str): Instructions for the capability.
        capability_score_func (str): The scoring function for the capability.
        source_dataset (str): The name of the source dataset.

    Returns
    -------
        None
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
        "capability_family": capability_family,
        "capability_instructions": capability_instructions,
        "capability_data": capability_data,
    }
    if capability_subject:
        capability_json.update({"capability_subject": capability_subject})
    with open(os.path.join(capability_dir, "capability.json"), "w") as f:
        json.dump(capability_json, f, indent=4)

    # Create capability python file
    capability_data_samples_dict = {
        f"{idx + 1}": sample for idx, sample in enumerate(capability_repr_samples)
    }
    capability_class_str = CAPABILITY_CLASS_TEMPLATE.format(
        capability_data_samples_dict=json.dumps(capability_data_samples_dict, indent=4),
        capability_instructions=capability_instructions,
        capability_score_func=capability_score_func,
    ).lstrip("\n")
    with open(os.path.join(capability_dir, "capability.py"), "w") as f:
        f.write(capability_class_str)


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
    Create seed capabilities based on the provided configuration.

    Args:
        cfg (DictConfig): Configuration object containing capability settings.

    The function processes capabilities from the configuration and
    creates seed capabilities for different datasets.
    It supports the following datasets:
    - "mathematics": Processes mathematics dataset, organizes them by subject,
      and populates the seed capability directory with descriptions, instructions,
      and representative samples.
    - "gsm8k": Processes GSM8K dataset, extracts solutions and answers, and populates
      the seed capability directory with descriptions, instructions, and representative
      samples.

    The function uses a fixed random seed for reproducibility and prints the number
    of samples created for each capability.
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
            for sample in dataset._data:
                subject = sample["type"].lower()
                capability_name = (
                    f"{dataset.domain}_{dataset.family}_{'_'.join(subject.split(' '))}"
                )
                if capability_name not in capabilities:
                    capabilities[capability_name] = defaultdict()
                    capabilities[capability_name]["type"] = subject
                    capabilities[capability_name]["samples"] = []
                sample["answer"] = remove_boxed(
                    last_boxed_only_string(sample["solution"]) or sample["solution"]
                )
                capabilities[capability_name]["samples"].append(sample)

            for capability_name, math_samples in capabilities.items():
                subject = math_samples["type"]
                capability_desc = dataset.description.format(
                    name=capability_name,
                    size=len(math_samples["samples"]),
                    subject=subject,
                )
                capability_instructions = dataset.instructions.format(
                    subject=subject, problem='{t["problem"]}'
                )

                capability_repr_samples = random.sample(
                    math_samples["samples"],
                    dataset._cfg["data_args"]["num_repr_samples"],
                )
                # Only keep problem and answer
                capability_repr_samples = [
                    {"problem": s["problem"], "answer": s["answer"]}
                    for s in capability_repr_samples
                ]

                populate_seed_capability_dir(
                    base_dir=seed_capability_dir,
                    capability_name=capability_name,
                    capability_description=capability_desc,
                    capability_domain=dataset.domain,
                    capability_family=dataset.family,
                    capability_subject=subject,
                    capability_data=math_samples["samples"],
                    capability_repr_samples=capability_repr_samples,
                    capability_instructions=capability_instructions,
                    capability_score_func=gsm8k_score_func.strip(
                        "\n"
                    ),  # TODO: Change this to mathematics_score_func after figuring out how to implement complex score functions
                    source_dataset=dataset.name,
                )
                print(
                    f"Created capability {capability_name} with {len(math_samples['samples'])} samples."
                )
        elif dataset.name == "gsm8k":
            capability_name = f"{dataset.domain}_{dataset.family}"
            gsm_samples = []
            for sample in dataset._data:
                sample["solution"] = sample["answer"]
                sample["answer"] = sample["answer"].split("####").pop().strip()
                gsm_samples.append(sample)

            capability_instructions = dataset.instructions.format(
                problem='{t["question"]}'
            )

            capability_repr_samples = random.sample(
                gsm_samples, dataset._cfg["data_args"]["num_repr_samples"]
            )
            # Only keep question and answer
            capability_repr_samples = [
                {"question": s["question"], "answer": s["answer"]}
                for s in capability_repr_samples
            ]

            populate_seed_capability_dir(
                base_dir=seed_capability_dir,
                capability_name=capability_name,
                capability_description=dataset.description,
                capability_domain=dataset.domain,
                capability_family=dataset.family,
                capability_data=gsm_samples,
                capability_repr_samples=capability_repr_samples,
                capability_instructions=capability_instructions,
                capability_score_func=gsm8k_score_func.strip("\n"),
                source_dataset=dataset.name,
            )
            print(
                f"Created capability {capability_name} with {len(gsm_samples)} samples."
            )


if __name__ == "__main__":
    main()
