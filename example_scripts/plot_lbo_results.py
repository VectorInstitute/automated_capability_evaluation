"""Plot LBO results for the paper artifacts."""

import json
import logging
import os

import hydra
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig

from src.utils import constants


logger = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="example_cfg",
    config_name="plot_lbo_results_cfg",
)
def main(cfg: DictConfig) -> None:
    """
    Plot LBO results for the paper artifacts.

    Args:
        cfg (DictConfig): Configuration for the model.
    """
    lbo_results_dir = os.path.join(
        constants.BASE_ARTIFACTS_DIR,
        "lbo_results",
    )
    output_dir = os.path.join(
        constants.BASE_ARTIFACTS_DIR,
        "paper_artifacts",
    )

    # Load the LBO results from the JSON file
    all_results = {}  # type: ignore
    for subject_llm_name in cfg.subject_llms:
        all_results[subject_llm_name] = {}
        for acquisition_function in cfg.lbo_cfg.acquisition_functions:
            results_file_name = f"lbo_results_{cfg.exp_cfg.exp_id}_{subject_llm_name}_{cfg.lbo_cfg.pipeline_id}_F{cfg.lbo_cfg.train_frac}_I{cfg.lbo_cfg.num_initial_train}_LR{cfg.lbo_cfg.num_lbo_runs}_AF{acquisition_function}_K{cfg.lbo_cfg.select_k}.json"
            with open(os.path.join(lbo_results_dir, results_file_name), "r") as f:
                results_dict = json.load(f)
            all_results[subject_llm_name][results_dict["acquisition_function_tag"]] = {}
            for metric in cfg.lbo_cfg.metrics:
                all_results[subject_llm_name][results_dict["acquisition_function_tag"]][
                    metric
                ] = results_dict["lbo_error_dict"][metric]
            num_entries = len(results_dict["new_capabilities"])
            if num_entries < cfg.lbo_cfg.num_lbo_runs:
                logger.warning(
                    f"[{results_file_name}] Number of iterations in the results file ({num_entries}) is less than the number of LBO runs ({cfg.lbo_cfg.num_lbo_runs})."
                )

    methods = ["ALC"]

    for method in methods:
        for metric in cfg.lbo_cfg.metrics:
            plt.figure(figsize=(6, 4))
            x_ub = cfg.lbo_cfg.num_lbo_runs + 1
            x_vals = np.arange(0, x_ub)

            for subject_llm_name in cfg.subject_llms:
                mean_values = np.array(all_results[subject_llm_name][method][metric])[
                    :x_ub
                ]
                if mean_values.shape[0] < x_vals.shape[0]:
                    plt.plot(
                        x_vals[: mean_values.shape[0]],
                        mean_values,
                        label=subject_llm_name,
                    )
                else:
                    plt.plot(x_vals, mean_values, label=subject_llm_name)

            y_label = "RMSE" if metric == "rmse" else "Average Standard Deviation"
            plt_file_name = f"lbo_plot_{cfg.exp_cfg.exp_id}_{cfg.lbo_cfg.pipeline_id}_F{cfg.lbo_cfg.train_frac}_I{cfg.lbo_cfg.num_initial_train}_LR{cfg.lbo_cfg.num_lbo_runs}_{method}_{metric}.pdf"

            plt.xlabel("Active Learning Iteration", fontsize=14)
            plt.ylabel(y_label, fontsize=14)
            x_ticks = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
            plt.xticks(x_ticks, fontsize=12)
            plt.yticks(fontsize=12)
            plt.legend(
                fontsize=12, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2
            )
            plt.grid(True)
            plt.savefig(
                os.path.join(output_dir, plt_file_name),
                format="pdf",
                bbox_inches="tight",
            )


if __name__ == "__main__":
    main()
