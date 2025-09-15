"""Script to discover new capabilities using LBO."""

import json
import logging
import os
import shutil

import hydra
from omegaconf import DictConfig
from tqdm import tqdm

from src.generate_tasks import (
    generate_tasks_using_llm,
)
from src.lbo import calculate_lbo_error, fit_lbo, select_capabilities_using_lbo
from src.model import Model
from src.utils import constants, prompts
from src.utils.capability_discovery_utils import (
    capability_satisfies_criterion,
    knn_based_capability_discovery,
    score_based_capability_discovery,
    select_complete_capabilities,
)
from src.utils.capability_management_utils import (
    get_previous_capabilities,
)
from src.utils.data_utils import check_cfg, get_run_id
from src.utils.embedding_utils import (
    apply_dimensionality_reduction,
    apply_dimensionality_reduction_to_test_capabilities,
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

    rmse_dict: Dict[str, float] = {}
    avg_std_dict: Dict[str, float] = {}
    for dim_reduction_method in ["t-sne", "pca"]:
        for rep_string_order in ["n", "nd", "and"]:
            # Embed capabilities using openai embedding model
            generate_and_set_capabilities_embeddings(
                capabilities=capabilities,
                embedding_model_name=cfg.embedding_cfg.embedding_model,
                embed_dimensions=cfg.embedding_cfg.embedding_size,
                rep_string_order=rep_string_order,
            )
            # Train-test split.
            train_capabilities, test_capabilities = get_lbo_train_set(
                input_data=capabilities,
                train_frac=0.8,
                input_categories=[capability.area for capability in capabilities],
                seed=cfg.exp_cfg.seed,
            )
            logger.info(f"num train_capabilities: {len(train_capabilities)}")
            logger.info(f"num test_capabilities: {len(test_capabilities)}")
            # Fit the dimensionality reduction model and transform all capabilities
            _ = apply_dimensionality_reduction(
                capabilities=capabilities,
                dim_reduction_method_name=dim_reduction_method,
                output_dimension_size=cfg.dimensionality_reduction_cfg.no_discovery_reduced_dimensionality_size,
                embedding_model_name=cfg.embedding_cfg.embedding_model,
                random_seed=cfg.exp_cfg.seed,
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
            logger.info(f"RMSE: {rmse}, avg_std: {avg_std}")
            key = dim_reduction_method + "," + rep_string_order
            rmse_dict[key] = rmse
            avg_std_dict[key] = avg_std

    logger.info(f"rmse_dict: {rmse_dict}")
    logger.info(f"avg_std_dict: {avg_std_dict}")


if __name__ == "__main__":
    logger = logging.getLogger(__name__)

    main()
