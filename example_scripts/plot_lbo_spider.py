"""Generate per-area LBO spider charts across training fractions."""

import json
import logging
import math
import os
from collections import defaultdict
from typing import DefaultDict

import hydra
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from omegaconf import DictConfig

from src.utils import constants


logger = logging.getLogger(__name__)


def _slugify(value: str) -> str:
    """Convert an arbitrary string to a filesystem-friendly slug."""
    return "".join(
        ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value.lower()
    )


@hydra.main(
    version_base=None,
    config_path="example_cfg",
    config_name="plot_lbo_spider_cfg",
)
def main(cfg: DictConfig) -> None:
    """Generate spider charts summarising LBO performance per area."""
    lbo_results_dir = os.path.join(
        constants.BASE_ARTIFACTS_DIR,
        "lbo_results",
        "per_area",
    )
    if not os.path.isdir(lbo_results_dir):
        logger.error("LBO results directory does not exist: %s", lbo_results_dir)
        return

    output_dir = os.path.join(constants.BASE_ARTIFACTS_DIR, "farnaz_plots")
    os.makedirs(output_dir, exist_ok=True)

    target_train_fracs = [float(frac) for frac in cfg.lbo_cfg.train_fracs]
    metric = cfg.plot_cfg.metric

    # subject -> acquisition_fn -> train_frac -> area -> list[metric]
    data: DefaultDict[
        str, DefaultDict[str, DefaultDict[float, DefaultDict[str, list[float]]]]
    ] = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))

    for file_name in os.listdir(lbo_results_dir):
        if not file_name.endswith(".json"):
            continue
        file_path = os.path.join(lbo_results_dir, file_name)
        with open(file_path, "r") as f:
            try:
                results = json.load(f)
            except json.JSONDecodeError as e:
                logger.warning("Skipping invalid JSON %s: %s", file_path, e)
                continue

        if cfg.exp_cfg.get("exp_id") and results.get("run_id") != cfg.exp_cfg.exp_id:
            continue

        pipeline_filter = cfg.lbo_cfg.get("pipeline_id")
        if (
            pipeline_filter
            and results.get("pipeline_id", pipeline_filter) != pipeline_filter
        ):
            continue

        subject_name = results.get("subject_llm_name")
        subject_filter = set(cfg.get("subject_llms", []))
        if subject_filter and subject_name not in subject_filter:
            continue

        acquisition_fn = results.get("acquisition_function")
        acq_filter = set(cfg.lbo_cfg.get("acquisition_functions", []))
        if acq_filter and acquisition_fn not in acq_filter:
            continue

        area_name = results.get("area")
        if not area_name:
            continue
        metric_series = results.get("lbo_error_dict", {}).get(metric, [])
        if not metric_series:
            continue

        initial_train_size = len(results.get("initial_train_capabilities", []))
        candidate_size = len(results.get("candidate_capabilities", []))
        additional_available = max(0, len(metric_series) - 1)
        max_iteratable = initial_train_size + additional_available
        total_capacity = max(initial_train_size + candidate_size, max_iteratable)

        for target_frac in target_train_fracs:
            clamped_frac = max(0.0, min(1.0, target_frac))
            target_total = max(1, math.ceil(total_capacity * clamped_frac))
            target_total = min(target_total, max_iteratable)
            additional_needed = max(0, target_total - initial_train_size)
            index = min(additional_needed, additional_available)
            if index >= len(metric_series):
                continue
            value = float(metric_series[index])
            if math.isnan(value):
                continue
            data[subject_name][acquisition_fn][target_frac][area_name].append(value)

    if not data:
        logger.error("No matching LBO results found under %s", lbo_results_dir)
        return

    for subject_name, acquisition_dict in data.items():
        for acquisition_fn, frac_dict in acquisition_dict.items():
            # Collect sorted list of areas present across all fractions for this subject/acquisition
            areas = sorted(
                {area for frac_data in frac_dict.values() for area in frac_data}
            )
            if not areas:
                logger.warning(
                    "No areas found for subject=%s acquisition=%s",
                    subject_name,
                    acquisition_fn,
                )
                continue

            angles = np.linspace(0, 2 * np.pi, len(areas), endpoint=False)
            angles = np.concatenate([angles, [angles[0]]])

            fig, ax = plt.subplots(subplot_kw={"polar": True}, figsize=(7, 7))

            for train_frac in target_train_fracs:
                frac_data = frac_dict.get(train_frac)
                if not frac_data:
                    logger.warning(
                        "Missing data for subject=%s acquisition=%s train_frac=%.3f",
                        subject_name,
                        acquisition_fn,
                        train_frac,
                    )
                    continue

                values = []
                for area in areas:
                    area_values = frac_data.get(area)
                    if not area_values:
                        values.append(np.nan)
                    else:
                        values.append(float(np.mean(area_values)))

                if all(np.isnan(values)):
                    continue

                values = np.array(values)
                values = np.where(np.isnan(values), np.nanmean(values), values)
                values = np.concatenate([values, [values[0]]])

                ax.plot(
                    angles,
                    values,
                    label=f"{int(train_frac * 100)}% train",
                )
                ax.fill(angles, values, alpha=0.1)

            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)
            ax.set_thetagrids(np.degrees(angles[:-1]), labels=areas)
            ax.set_title(
                f"{subject_name} – {acquisition_fn.upper()} ({metric.upper()})",
                fontsize=14,
            )
            ax.yaxis.set_major_locator(MaxNLocator(integer=False, nbins=5))
            ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))

            output_file = (
                f"lbo_spider_{_slugify(cfg.exp_cfg.exp_id)}_"
                f"{_slugify(subject_name)}_{_slugify(acquisition_fn)}_{metric}.pdf"
            )
            fig.savefig(
                os.path.join(output_dir, output_file),
                format="pdf",
                bbox_inches="tight",
            )
            # Save as PNG for easier viewing in the browser
            fig.savefig(
                os.path.join(output_dir, output_file.replace(".pdf", ".png")),
                format="png",
                bbox_inches="tight",
            )
            plt.close(fig)

            logger.info(
                "Saved spider chart for subject=%s acquisition=%s to %s",
                subject_name,
                acquisition_fn,
                output_file,
            )


if __name__ == "__main__":
    main()
