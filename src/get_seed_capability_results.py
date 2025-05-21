"""Seed capability results extraction script."""

import json
import logging
import os

import hydra
import numpy as np
from omegaconf import DictConfig

from utils.data_utils import copy_file, list_dir, read_json_file, write_json_file


logger = logging.getLogger(__name__)


def extract_math_capability_logs(
    log_file: str, capability_name: str, subject: str, out_dir: str
) -> None:
    """
    Extract and processes math capability logs from a given log file.

    Filter the logs based on the specified subject,
    update the log structure, and
    write the processed logs to an output directory.

    Args
    ----
        log_file (str): Path to the input log file
        containing the capability logs in JSON format.
        capability_name (str): Name of the capability to be used
            in the output file and log updates.
        subject (str): Subject to filter the log tasks by.
        out_dir (str): Directory where the processed log file will be saved.
    """
    logs = read_json_file(log_file)

    # Note: Inspect logs refer to capabilities as tasks, hence
    # keeping dict keys consistent in logs
    master_capability = logs["eval"]["task"]
    logs["eval"].update({"task": f"inspect_evals/{capability_name}"})
    logs["eval"].update({"master_task": master_capability})

    logs_dataset = logs["eval"].pop("dataset")
    logs_results = logs.pop("results")
    logs_tasks = logs.pop("samples")
    logs_reductions = logs.pop("reductions")
    _ = logs.pop("stats")

    # Filter for subject
    logs_tasks = [task for task in logs_tasks if task["metadata"]["subject"] == subject]
    task_ids = {task["id"] for task in logs_tasks}

    # Update logs
    for scorer in logs_reductions:
        scorer.update(
            {
                "samples": [
                    elm for elm in scorer["samples"] if elm["sample_id"] in task_ids
                ]
            }
        )
    logs_dataset.update(
        {
            "samples": len(logs_tasks),
            "sample_ids": list(task_ids),
        }
    )
    logs_results.update(
        {
            "total_samples": len(logs_tasks),
            "completed_samples": len(logs_tasks),
        }
    )
    for scorer in logs_results["scores"]:
        reduction_tasks = [
            elm for elm in logs_reductions if elm["scorer"] == scorer["name"]
        ][0]["samples"]
        scorer["metrics"]["accuracy"]["value"] = np.mean(
            [elm["value"] for elm in reduction_tasks]
        )
        _ = scorer["metrics"].pop("stderr")

    # Add all log elements back
    logs["eval"].update({"dataset": logs_dataset})
    logs.update({"results": logs_results})
    logs.update({"samples": logs_tasks})
    logs.update({"reductions": logs_reductions})

    # Write to output file
    out_file_name = os.path.join(out_dir, capability_name, f"{capability_name}.json")
    write_json_file(file_path=out_file_name, data=logs)


@hydra.main(version_base=None, config_path="cfg", config_name="run_cfg")
def main(cfg: DictConfig) -> None:
    """
    Obtain seed capability results.

    This function performs the following steps:
    1. Defines directories for seed capabilities, seed capability results,
        and seed datasets logs.
    2. Iterates over each capability directory in the seed capabilities directory.
    3. Reads the capability configuration from the "capability.json" file.
    4. Determines the dataset name and capability details
    from the capability configuration.
    5. Iterates over results for all subject models
    in the seed datasets log directory.
    6. For each log file that matches the dataset name,
    processes the log file based on the dataset type:
        - For "math" dataset, extracts math capability logs.
        - For "gsm8k" dataset, copies the log file to the output directory.

    Args
    ----
        cfg (DictConfig): Configuration object containing paths and settings.
    """
    domain = cfg.capabilities_cfg.domain
    seed_capability_dir = os.path.join(
        cfg.capabilities_cfg.capabilities_dir, "seed_capabilities", domain
    )
    seed_capability_result_dir = os.path.join(
        cfg.capabilities_cfg.results_dir, "seed_capabilities_results"
    )
    seed_datasets_log_dir = os.path.join(
        cfg.capabilities_cfg.results_dir, "seed_datasets_inspect_logs"
    )

    for capability_dir in os.listdir(seed_capability_dir):
        with open(
            os.path.join(seed_capability_dir, capability_dir, "capability.json"), "r"
        ) as f:
            capability_json = json.load(f)
        dataset_name = (
            "math"
            if capability_json["source_dataset"] == "mathematics"
            else capability_json["source_dataset"]
        )
        capability_name = capability_json["capability_name"]
        if dataset_name == "math":
            subject = capability_json["capability_subject"]

        # Iterate over results for all subject models
        for subject_model_dir in list_dir(seed_datasets_log_dir):
            subject_model_log_path = os.path.join(
                seed_datasets_log_dir, subject_model_dir
            )
            for log_file in list_dir(subject_model_log_path):
                if dataset_name not in log_file:
                    continue

                out_dir = os.path.join(
                    seed_capability_result_dir, subject_model_dir, domain
                )

                # For math dataset, extract math capability logs
                if "math" in log_file:
                    extract_math_capability_logs(
                        log_file=os.path.join(subject_model_log_path, log_file),
                        capability_name=capability_name,
                        subject=subject,
                        out_dir=out_dir,
                    )

                # For gsm8k dataset, copy log file to output directory
                elif "gsm8k" in log_file:
                    # No changes to log file, just copy it to output directory
                    copy_file(
                        src=os.path.join(subject_model_log_path, log_file),
                        dest=os.path.join(out_dir, f"{capability_name}.json"),
                    )

                logger.info(
                    f"Extracted {subject_model_dir} result for {capability_name} capability."
                )


if __name__ == "__main__":
    main()
