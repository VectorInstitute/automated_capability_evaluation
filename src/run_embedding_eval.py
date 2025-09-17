"""Script to discover new capabilities using LBO."""

import logging
import os
from collections import defaultdict

import hydra
import numpy as np
from omegaconf import DictConfig
from scipy import stats
from tqdm import tqdm

from src.lbo import calculate_lbo_error, fit_lbo
from src.utils import constants
from src.utils.capability_discovery_utils import (
    select_complete_capabilities,
)
from src.utils.capability_management_utils import (
    get_previous_capabilities,
)
from src.utils.data_utils import check_cfg, get_run_id
from src.utils.embedding_utils import (
    apply_dimensionality_reduction,
    generate_and_set_capabilities_embeddings,
)
from src.utils.lbo_utils import get_lbo_train_set


@hydra.main(version_base=None, config_path="cfg", config_name="run_cfg")
def main(cfg: DictConfig) -> None:
    """
    Discover new capabilities using LBO.

    Args:
        cfg (DictConfig): Configuration for the model.
    """
    check_cfg(cfg, logger)
    run_id = get_run_id(cfg)

    # Set the base capability directory
    base_capability_dir = os.path.join(
        constants.BASE_ARTIFACTS_DIR,
        f"capabilities_{run_id}",
        cfg.capabilities_cfg.domain,
    )
    logger.info(f"base_capability_dir: {base_capability_dir}")
    # Read the capabilities from the base directory
    capabilities = get_previous_capabilities(
        capability_dir=base_capability_dir,
        score_dir_suffix=run_id,
    )
    capabilities = sorted(capabilities, key=lambda x: x.name)
    logger.info(f"All capability names:\n{capabilities}")

    # Select complete capabilities (same set of capabilities were evaluated)
    capabilities = select_complete_capabilities(
        capabilities=capabilities,
        strict=False,
        num_tasks_lower_bound=int(
            cfg.capabilities_cfg.num_gen_tasks_per_capability
            * (1 - cfg.capabilities_cfg.num_gen_tasks_buffer)
        ),
    )
    capabilities = sorted(capabilities, key=lambda x: x.name)
    logger.info(f"Selected capability names:\n{capabilities}")

    # Pre-load capability scores
    for capability in tqdm(capabilities, desc="Loading capability scores"):
        capability.load_scores(
            subject_llm_name=cfg.subject_llm.name,
        )
        logger.info(f"Capability: {capability.name}, scores: {capability.scores}")

    # Number of runs to calculate confidence intervals.
    num_runs = 20

    rmse_dict = defaultdict(list)
    avg_std_dict = defaultdict(list)
    for dim_reduction_method in ["t-sne", "pca"]:
        for rep_string_order in ["n", "nd", "and"]:
            # Embed capabilities using openai embedding model
            generate_and_set_capabilities_embeddings(
                capabilities=capabilities,
                embedding_model_name=cfg.embedding_cfg.embedding_model,
                embed_dimensions=cfg.embedding_cfg.embedding_size,
                rep_string_order=rep_string_order,
            )
            # Fit the dimensionality reduction model and transform all capabilities
            _ = apply_dimensionality_reduction(
                capabilities=capabilities,
                dim_reduction_method_name=dim_reduction_method,
                output_dimension_size=cfg.dimensionality_reduction_cfg.no_discovery_reduced_dimensionality_size,
                embedding_model_name=cfg.embedding_cfg.embedding_model,
                random_seed=cfg.exp_cfg.seed,
            )
            for i in range(num_runs):
                # Train-test split.
                train_capabilities, test_capabilities = get_lbo_train_set(
                    input_data=capabilities,
                    train_frac=0.8,
                    input_categories=[capability.area for capability in capabilities],
                    seed=i,
                )
                # Fit model to training capabilities
                # acquisition_function is unused.
                lbo = fit_lbo(
                    capabilities=train_capabilities,
                    embedding_name=dim_reduction_method,
                    subject_llm_name=cfg.subject_llm.name,
                    acquisition_function="expected_variance_reduction",
                )
                # Measure test set error.
                rmse, avg_std = calculate_lbo_error(
                    lbo_model=lbo,
                    capabilities=test_capabilities,
                    embedding_name=dim_reduction_method,
                    subject_llm_name=cfg.subject_llm.name,
                )
                str_key = dim_reduction_method + "," + rep_string_order
                rmse_dict[str_key].append(rmse)
                avg_std_dict[str_key].append(avg_std)

    # Compute mean and 95% confidence interval
    mean_values = {}
    for key, values in rmse_dict.items():
        arr = np.array(values)
        mean = arr.mean()
        sem = stats.sem(arr)
        ci = stats.t.interval(0.95, len(arr) - 1, loc=mean, scale=sem)
        mean_values[key] = {"mean": mean, "95p_ci": ci}
    for key, stats_dict in mean_values.items():
        ci_low, ci_high = stats_dict["95p_ci"]
        logger.info(
            f"{key}: RMSE={stats_dict['mean']:.3f}, 95% Confidence Interval=({ci_low:.3f}, {ci_high:.3f})"
        )

    std_values = {}
    for key, values in avg_std_dict.items():
        arr = np.array(values)
        mean = arr.mean()
        sem = stats.sem(arr)
        ci = stats.t.interval(0.95, len(arr) - 1, loc=mean, scale=sem)
        std_values[key] = {"mean": mean, "95p_ci": ci}
    for key, stats_dict in std_values.items():
        ci_low, ci_high = stats_dict["95p_ci"]
        logger.info(
            f"{key}: Standard Deviation (Uncertainty)={stats_dict['mean']:.3f}, 95% Confidence Interval=({ci_low:.3f}, {ci_high:.3f})"
        )


if __name__ == "__main__":
    logger = logging.getLogger(__name__)

    main()
