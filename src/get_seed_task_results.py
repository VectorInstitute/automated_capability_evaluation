import json  # noqa: D100
import os
import shutil

import numpy as np


def extract_math_task_logs(
    log_file: str, task_name: str, subject: str, out_dir: str
) -> None:
    """
    Extract and processes math task logs from a given log file.

    Filter the logs based on the specified subject,
    update the log structure, and
    write the processed logs to an output directory.

    Args:
        log_file (str): Path to the input log file
        containing the task logs in JSON format.
        task_name (str): Name of the task to be used in the output file and log updates.
        subject (str): Subject to filter the log samples by.
        out_dir (str): Directory where the processed log file will be saved.

    Returns
    -------
        None
    """
    out_file_name = f"{out_dir}/{task_name}.json"
    if os.path.exists(out_file_name):
        return

    with open(log_file, "r") as f:
        logs = json.load(f)

    master_task = logs["eval"]["task"]
    logs["eval"].update({"task": f"inspect_evals/{task_name}"})
    logs["eval"].update({"master_task": master_task})

    logs_dataset = logs["eval"].pop("dataset")
    logs_results = logs.pop("results")
    logs_samples = logs.pop("samples")
    logs_reductions = logs.pop("reductions")
    _ = logs.pop("stats")

    # Filter for subject
    logs_samples = [
        sample for sample in logs_samples if sample["metadata"]["subject"] == subject
    ]
    sample_ids = {sample["id"] for sample in logs_samples}

    # Update logs
    for scorer in logs_reductions:
        scorer.update(
            {
                "samples": [
                    elm for elm in scorer["samples"] if elm["sample_id"] in sample_ids
                ]
            }
        )
    logs_dataset.update(
        {
            "samples": len(logs_samples),
            "sample_ids": list(sample_ids),
        }
    )
    logs_results.update(
        {
            "total_samples": len(logs_samples),
            "completed_samples": len(logs_samples),
        }
    )
    for scorer in logs_results["scores"]:
        reduction_samples = [
            elm for elm in logs_reductions if elm["scorer"] == scorer["name"]
        ][0]["samples"]
        scorer["metrics"]["accuracy"]["value"] = np.mean(
            [elm["value"] for elm in reduction_samples]
        )
        _ = scorer["metrics"].pop("stderr")

    # Add all log elements back
    logs["eval"].update({"dataset": logs_dataset})
    logs.update({"results": logs_results})
    logs.update({"samples": logs_samples})
    logs.update({"reductions": logs_reductions})

    # Write to output file
    with open(out_file_name, "w") as f:
        json.dump(logs, f, indent=4)


def main() -> None:
    """
    Obtain seed task results.

    This function performs the following steps:
    1. Defines directories for seed tasks, seed task results, and seed datasets logs.
    2. Iterates over each task directory in the seed tasks directory.
    3. Reads the task configuration from the "task.json" file.
    4. Determines the dataset name and task details
    from the task configuration.
    5. Iterates over results for all candidate models
    in the seed datasets log directory.
    6. For each log file that matches the dataset name,
    processes the log file based on the dataset type:
        - For "math" dataset, extracts math task logs.
        - For "gsm8k" dataset, copies the log file to the output directory.
    """
    domain = "math"
    seed_task_dir = f"./seed_tasks/{domain}"
    seed_task_result_dir = "./seed_tasks_results"
    seed_datasets_log_dir = "./seed_datasets_inspect_logs"

    for task_dir in os.listdir(seed_task_dir):
        with open(os.path.join(seed_task_dir, task_dir, "task.json"), "r") as f:
            task_json = json.load(f)
        dataset_name = (
            "math"
            if task_json["source_dataset"] == "mathematics"
            else task_json["source_dataset"]
        )
        task_name = task_json["task_name"]
        if dataset_name == "math":
            subject = task_json["task_subject"]

        # Iterate over results for all candidate models
        for candidate_model_dir in os.listdir(seed_datasets_log_dir):
            candidate_model_log_path = os.path.join(
                seed_datasets_log_dir, candidate_model_dir
            )
            for log_file in os.listdir(candidate_model_log_path):
                if dataset_name not in log_file:
                    continue

                out_dir = os.path.join(seed_task_result_dir, candidate_model_dir)
                out_dir = os.path.join(out_dir, domain)
                os.makedirs(out_dir, exist_ok=True)

                # For math dataset, extract math task logs
                if "math" in log_file:
                    extract_math_task_logs(
                        log_file=os.path.join(candidate_model_log_path, log_file),
                        task_name=task_name,
                        subject=subject,
                        out_dir=out_dir,
                    )

                # For gsm8k dataset, copy log file to output directory
                elif "gsm8k" in log_file:
                    # No changes to log file, just copy it to output directory
                    shutil.copyfile(
                        src=os.path.join(candidate_model_log_path, log_file),
                        dst=os.path.join(
                            out_dir,
                            f"{task_name}.json",
                        ),
                    )

                print(f"Extracted {candidate_model_dir} result for {task_name} task.")


if __name__ == "__main__":
    main()
