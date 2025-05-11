"""Compare the scores of mock seed capabilities with actual seed capabilities."""

import json
import logging
import os
from typing import Dict, List

import hydra
import numpy as np
from omegaconf import DictConfig

from src.generate_capabilities import (
    get_previous_capabilities,
)
from src.utils import constants
from src.utils.capability_utils import (
    read_score_inspect_json,
)


def generate_latex_table(
    output_dict: Dict[str, Dict[str, Dict[str, Dict[str, float]]]],
    subject_llms: List[str],
    output_dir: str,
    output_file_name: str,
) -> None:
    """
    Generate a LaTeX table comparing mock and actual scores for capabilities.

    Args:
        output_dict (Dict[str, Dict[str, Dict[str, Dict[str, float]]]]):
            Dictionary containing scores for each capability.
        subject_llms (List[str]): List of subject LLMs.
        output_dir (str): Directory for saving the LaTeX table.
        output_file_name (str): Name of the output LaTeX file.
    """
    # Start LaTeX table with multi-level columns
    latex_table = (
        "\\begin{table}[h!]\n\\centering\n\\begin{tabular}{|l|"
        + "c|c|" * len(subject_llms)
        + "}\n\\hline\n"
    )
    latex_table += "Capability"
    for subject_llm in subject_llms:
        latex_table += f" & \\multicolumn{{2}}{{c|}}{{{subject_llm}}}"
    latex_table += " \\\\\n\\hline\n"

    # Add sub-columns for Mock and Actual
    latex_table += (
        " & " + " & ".join(["Mock & Actual"] * len(subject_llms)) + " \\\\\n\\hline\n"
    )

    # Add rows for each capability
    for capability_name, llm_scores in output_dict.items():
        if capability_name == "all_capabilities":
            continue
        row = [capability_name.replace("_", " ").replace("and", "\\&")]
        for subject_llm in subject_llms:
            mock_mean = llm_scores[subject_llm]["mock_scores"]["mean"]
            actual_mean = llm_scores[subject_llm]["actual_scores"]["mean"]
            row.append(f"{mock_mean:.2f} & {actual_mean:.2f}")
        latex_table += " & ".join(row) + " \\\\\n"

    # Add summary row for all capabilities
    latex_table += "\\hline\nAll Capabilities"
    for subject_llm in subject_llms:
        mock_mean = output_dict["all_capabilities"][subject_llm]["mock_scores"]["mean"]
        actual_mean = output_dict["all_capabilities"][subject_llm]["actual_scores"][
            "mean"
        ]
        latex_table += f" & {mock_mean:.2f} & {actual_mean:.2f}"
    latex_table += " \\\\\n\\hline\n"

    # End LaTeX table
    latex_table += "\\end{tabular}\n\\caption{Comparison of Mock and Actual Scores for MATH Capabilities}\n\\label{tab:capability_scores}\n\\end{table}"

    # Save LaTeX table to a file
    output_file = os.path.join(output_dir, output_file_name)
    with open(output_file, "w") as f:
        f.write(latex_table)

    logger.info(f"LaTeX table saved to {output_file}")


@hydra.main(
    version_base=None,
    config_path="example_cfg",
    config_name="compare_seed_capability_results_cfg",
)
def main(cfg: DictConfig) -> None:
    """
    Compare the scores of mock seed capabilities with actual seed capabilities.

    Args:
        cfg (DictConfig): Configuration for the model.
    """
    run_id = cfg.exp_id
    num_tasks = cfg.num_tasks
    subject_llms = list(cfg.subject_llms)

    # Load output_dict from JSON file if it exists
    output_json_path = os.path.join(
        constants.BASE_ARTIFACTS_DIR,
        "paper_artifacts",
        f"{run_id}.json",
    )
    if os.path.exists(output_json_path):
        with open(output_json_path, "r") as f:
            output_dict = json.load(f)
        logger.info(f"Loaded output_dict from {output_json_path}")
    else:
        logger.info(
            f"No existing output_dict found at {output_json_path}, generating a new one."
        )

        # Set the base capability directory
        base_capability_dir = os.path.join(
            constants.BASE_ARTIFACTS_DIR,
            f"capabilities_{run_id}",
            cfg.domain,
        )
        # Read the capabilities from the base directory
        capabilities = get_previous_capabilities(
            capability_dir=base_capability_dir,
            score_dir_suffix=run_id,
        )
        capabilities = sorted(capabilities, key=lambda x: x.name)
        logger.info(f"Capability names:\n{capabilities}")

        output_dict = {}
        for capability in capabilities:
            capability_name = capability.name

            if num_tasks == -1:
                num_tasks = len(capability.get_tasks())
            elif num_tasks > len(capability.get_tasks()):
                logger.warning(
                    f"[{capability.name}] Requested number of tasks ({num_tasks}) is greater than the available tasks ({len(capability.get_tasks())}). Setting num_tasks to the available tasks."
                )
                num_tasks = len(capability.get_tasks())

            output_dict[capability_name] = {}
            for subject_llm in subject_llms:
                logger.info(
                    f"Comparing scores for capability: {capability_name} with subject LLM: {subject_llm}"
                )
                # Get scores for generated mock seed capability tasks
                capability.load_scores(
                    subject_llm_name=subject_llm, num_tasks=num_tasks, seed=cfg.seed
                )
                mock_scores = capability.scores[subject_llm]

                # Get scores for num_tasks tasks from the
                # original seed capability dataset
                # Set seed capability directory
                seed_capability_dir = os.path.join(
                    constants.SEED_CAPABILITIES_SCORE_DIR,
                    subject_llm,
                    cfg.domain,
                )
                actual_scores = read_score_inspect_json(
                    os.path.join(
                        seed_capability_dir, capability_name, f"{capability_name}.json"
                    ),
                    num_tasks=num_tasks,
                    seed=cfg.seed,
                )

                output_dict[capability_name][subject_llm] = {
                    "mock_scores": mock_scores,
                    "actual_scores": actual_scores,
                    "num_tasks": num_tasks,
                }

        # Calculate the mean across all capabilities for each subject LLM
        all_capabilities = {}
        for subject_llm in subject_llms:
            mock_means = []
            actual_means = []
            for _, capability_scores in output_dict.items():
                mock_means.append(capability_scores[subject_llm]["mock_scores"]["mean"])
                actual_means.append(
                    capability_scores[subject_llm]["actual_scores"]["mean"]
                )

            all_capabilities[subject_llm] = {
                "mock_scores": {
                    "mean": np.mean(mock_means),
                },
                "actual_scores": {
                    "mean": np.mean(actual_means),
                },
            }
        output_dict.update({"all_capabilities": all_capabilities})

        # Save the output_dict to a JSON file
        with open(
            os.path.join(
                constants.BASE_ARTIFACTS_DIR,
                "paper_artifacts",
                f"{run_id}.json",
            ),
            "w",
        ) as f:
            json.dump(output_dict, f, indent=4)

    logger.info(f"Score comparison:\n{json.dumps(output_dict, indent=4)}")

    generate_latex_table(
        output_dict=output_dict,
        subject_llms=subject_llms,
        output_dir=os.path.join(constants.BASE_ARTIFACTS_DIR, "paper_artifacts"),
        output_file_name=f"{run_id}.tex",
    )


if __name__ == "__main__":
    logger = logging.getLogger(__name__)

    main()
