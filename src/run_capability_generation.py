"""Script to generate capabilities and tasks using the scientist LLM."""

import logging
import os

import hydra
from omegaconf import DictConfig

from generate_capabilities import (
    filter_capabilities,
    generate_and_set_capabilities_embeddings,
    generate_capabilities,
    get_previous_capabilities,
)
from generate_tasks import (
    generate_tasks_using_llm,
)
from model import Model
from utils import constants
from utils.data_utils import check_cfg, get_run_id


@hydra.main(version_base=None, config_path="cfg", config_name="run_cfg")
def main(cfg: DictConfig) -> None:
    """
    Run capability generation with the specified configuration.

    This includes generating capabilities and generating, solving and verifying tasks.

    Args:
        cfg (DictConfig): Configuration for the model.
    """
    check_cfg(cfg, logger)
    run_id = get_run_id(cfg)
    logger.info(f"Run ID: {run_id}")

    # Initialize the scientist LLM model
    scientist_llm = Model(
        model_name=cfg.scientist_llm.name,
        model_provider=cfg.scientist_llm.provider,
    )
    scientist_llm_gen_cfg = cfg.scientist_llm.generation_cfg

    # Generate initial capabilities
    # Set the base capability directory
    base_capability_dir = os.path.join(
        constants.BASE_ARTIFACTS_DIR,
        f"capabilities_{run_id}",
        cfg.capabilities_cfg.domain,
    )
    target_num_capabilities = cfg.capabilities_cfg.num_gen_capabilities
    if os.path.exists(base_capability_dir):
        # Fetch previously generated capabilities
        logger.info(
            f"Base capability directory already exists: {base_capability_dir}. "
            "Fetching previously generated capabilities."
        )
        capabilities = get_previous_capabilities(capability_dir=base_capability_dir)
    else:
        os.makedirs(base_capability_dir, exist_ok=False)
        logger.info("Starting capability generation ...")
        num_capabilities = int(
            target_num_capabilities
            * (1 + cfg.capabilities_cfg.num_gen_capabilities_buffer)
        )
        capabilities = generate_capabilities(
            domain=cfg.capabilities_cfg.domain,
            num_capabilities=num_capabilities,
            num_capabilities_per_run=cfg.capabilities_cfg.num_gen_capabilities_per_run,
            base_capability_dir=base_capability_dir,
            scientist_llm=scientist_llm,
            num_seed_capabilities=cfg.capabilities_cfg.num_seed_capabilities,
            scientist_llm_gen_cfg=dict(scientist_llm_gen_cfg.capability_generation),
            method=cfg.capabilities_cfg.method,
            num_capability_areas=cfg.capabilities_cfg.num_capability_areas,
            exclude_seed_capability_names=["word_problems"],
            run_id=run_id,
            trial_run=cfg.exp_cfg.trial_run,
            seed=cfg.exp_cfg.seed,
            retry_attempts=cfg.capabilities_cfg.capabilities_gen_retry_attempts,
        )
    capabilities = sorted(capabilities, key=lambda x: x.name)
    logger.info(f"Capability names ({len(capabilities)}):\n{capabilities}")
    if len(capabilities) < target_num_capabilities:
        logger.warning(
            f"Only {len(capabilities)} capabilities were created. "
            f"Target number of capabilities not reached: {target_num_capabilities}. "
            "It is recommended to increase the buffer."
        )

    # Embed capabilities using openai embedding model
    generate_and_set_capabilities_embeddings(
        capabilities=capabilities,
        embedding_model_name=cfg.embedding_cfg.embedding_model,
        embed_dimensions=cfg.embedding_cfg.embedding_size,
    )
    # Filter capabilities based on their embeddings
    filtered_capabilities = filter_capabilities(
        capabilities,
        embedding_model_name=cfg.embedding_cfg.embedding_model,
        similarity_threshold=cfg.embedding_cfg.filtering_similarity_threshold,
    )
    logger.info(
        f"Capabilities retained after filtering ({len(filtered_capabilities)}/{len(capabilities)}): {filtered_capabilities}"
    )

    # TODO: Run this asynchronosly
    for capability in filtered_capabilities:
        # Generate tasks for each capability
        generate_tasks_using_llm(
            capability=capability,
            scientist_llm=scientist_llm,
            num_tasks=cfg.capabilities_cfg.num_gen_tasks_per_capability,
            num_tasks_buffer=cfg.capabilities_cfg.num_gen_tasks_buffer,
            scientist_llm_gen_cfg_task_gen=dict(scientist_llm_gen_cfg.task_generation),
            scientist_llm_gen_cfg_task_solve=dict(scientist_llm_gen_cfg.task_solve),
            scientist_llm_gen_cfg_task_verify=dict(scientist_llm_gen_cfg.task_verify),
            solve_sample_tasks=True,  # TODO: Update this based on checkpointing
            few_shot=cfg.capabilities_cfg.task_gen_few_shot,
            run_id=run_id,
            tasks_gen_retry_attempts=cfg.capabilities_cfg.tasks_gen_retry_attempts,
            concurrency_task_solver=cfg.capabilities_cfg.concurrency_task_solver,
            concurrency_task_verifier=cfg.capabilities_cfg.concurrency_task_verifier,
            seed=cfg.exp_cfg.seed,
        )


if __name__ == "__main__":
    logger = logging.getLogger(__name__)

    main()
