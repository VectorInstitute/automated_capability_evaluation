"""Plot LBO results for the paper artifacts."""

import json
import logging
import os
import re
from collections import defaultdict

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
        constants.BASE_ARTIFACTS_DIR, "lbo_results", "per_area"
    )
    output_dir = os.path.join(
        constants.BASE_ARTIFACTS_DIR,
        "farnaz_plots",
    )
    os.makedirs(output_dir, exist_ok=True)

    all_results: dict[str, dict[str, dict[str, dict[str, list[float]]]]]
    all_results = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    def _area_slug(name: str) -> str:
        return re.sub(r"[^0-9A-Za-z_-]", "_", name.lower())

    def _discover_areas(subject_name: str, acquisition_fn: str) -> list[str]:
        if cfg.lbo_cfg.pipeline_id != "no_discovery":
            return ["all"]
        areas_found = set()
        prefix = f"lbo_results_{cfg.exp_cfg.exp_id}_{subject_name}_{cfg.lbo_cfg.pipeline_id}_"
        for file in os.listdir(lbo_results_dir):
            if not file.startswith(prefix):
                continue
            if f"AF{acquisition_fn}" not in file:
                continue
            with open(os.path.join(lbo_results_dir, file), "r") as f:
                results_dict = json.load(f)
            area_value = results_dict.get("area")
            if area_value:
                areas_found.add(area_value)
        return sorted(areas_found)

    for subject_llm_name in cfg.subject_llms:
        for acquisition_function in cfg.lbo_cfg.acquisition_functions:
            for area_name in _discover_areas(subject_llm_name, acquisition_function):
                print(f"area_name: {area_name}")
                suffix_parts = [
                    f"lbo_results_{cfg.exp_cfg.exp_id}_{subject_llm_name}_{cfg.lbo_cfg.pipeline_id}"
                ]
                if cfg.lbo_cfg.pipeline_id == "no_discovery":
                    suffix_parts.append(_area_slug(area_name))

                # suffix_parts.append(f"F{cfg.lbo_cfg.train_frac}")
                # suffix_parts.append(f"I{cfg.lbo_cfg.num_initial_train}")
                # suffix_parts.append(f"LR{cfg.lbo_cfg.num_lbo_runs}")
                # suffix_parts.append(f"AF{acquisition_function}")
                file_glob = "_".join(suffix_parts)
                print(f"file_glob: {lbo_results_dir}/{file_glob}")
                candidates = [
                    file
                    for file in os.listdir(lbo_results_dir)
                    if file.startswith(file_glob)
                ]
                if not candidates:
                    logger.warning(
                        "No results file found for subject=%s, area=%s, acquisition=%s",
                        subject_llm_name,
                        area_name,
                        acquisition_function,
                    )
                    continue

                results_file_name = sorted(candidates)[-1]
                with open(os.path.join(lbo_results_dir, results_file_name), "r") as f:
                    results_dict = json.load(f)

                acquisition_tag = results_dict.get(
                    "acquisition_function_tag", acquisition_function.upper()
                )
                area_key = area_name
                for metric in cfg.lbo_cfg.metrics:
                    metric_values = results_dict["lbo_error_dict"].get(metric, [])
                    all_results[area_key][acquisition_tag][subject_llm_name][metric] = (
                        metric_values
                    )

    for area_name, acquisition_dict in all_results.items():
        for acquisition_tag, subject_dict in acquisition_dict.items():
            for metric in cfg.lbo_cfg.metrics:
                plt.figure(figsize=(6, 4))
                for subject_llm_name, metric_dict in subject_dict.items():
                    values = metric_dict.get(metric, [])
                    if not values:
                        logger.warning(
                            "No %s values for subject=%s, area=%s, acquisition=%s",
                            metric,
                            subject_llm_name,
                            area_name,
                            acquisition_tag,
                        )
                        continue
                    x_vals = np.arange(len(values))
                    plt.plot(x_vals, values, label=subject_llm_name)

                y_label = "RMSE" if metric == "rmse" else "Average Standard Deviation"
                plt.xlabel("Active Learning Iteration", fontsize=14)
                plt.ylabel(y_label, fontsize=14)
                plt.xticks(fontsize=12)
                plt.yticks(fontsize=12)
                plt.legend(
                    fontsize=12, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2
                )
                plt.grid(True)
                area_slug = _area_slug(area_name)
                plt_file_name = f"lbo_plot_{cfg.exp_cfg.exp_id}_{cfg.lbo_cfg.pipeline_id}_{area_slug}_{acquisition_tag}_{metric}.pdf"
                plt.savefig(
                    os.path.join(output_dir, plt_file_name),
                    format="pdf",
                    bbox_inches="tight",
                )
                plt.savefig(
                    os.path.join(output_dir, plt_file_name.replace(".pdf", ".png")),
                    format="png",
                    bbox_inches="tight",
                )
                print(f"Saved plot to {os.path.join(output_dir, plt_file_name)}")


if __name__ == "__main__":
    main()
