"""Script to discover new capabilities using LBO."""

import logging
import os

import hydra
from omegaconf import DictConfig

from generate_capabilities import (
    apply_dimensionality_reduction,
    apply_dimensionality_reduction_to_test_capabilities,
    capability_satisfies_criterion,
    generate_and_set_capabilities_embeddings,
    get_previous_capabilities,
    knn_based_capability_discovery,
    score_based_capability_discovery,
    select_complete_capabilities,
)
from generate_tasks import (
    generate_tasks_using_llm,
)
from lbo import fit_lbo, select_capabilities_using_lbo, select_k_capabilities
from model import Model
from utils import constants, prompts
from utils.data_utils import check_cfg, get_run_id
from utils.lbo_utils import get_lbo_train_set


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
    # Read the capabilities from the base directory
    capabilities = get_previous_capabilities(
        capability_dir=base_capability_dir,
        score_dir_suffix=run_id,
    )
    capabilities = sorted(capabilities, key=lambda x: x.name)
    logger.info(f"ALl capability names:\n{capabilities}")
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

    # Embed capabilities using openai embedding model
    generate_and_set_capabilities_embeddings(
        capabilities=capabilities,
        embedding_model_name=cfg.embedding_cfg.embedding_model,
        embed_dimensions=cfg.embedding_cfg.embedding_size,
    )

    num_lbo_runs = cfg.lbo_cfg.num_lbo_runs
    if cfg.lbo_cfg.pipeline_id == "no_discovery":
        # Reduce the dimensionality of all capability embeddings
        dim_reduction_method_name = (
            cfg.dimensionality_reduction_cfg.no_discovery_reduced_dimensionality_method
        )
        _ = apply_dimensionality_reduction(
            capabilities=capabilities,
            dim_reduction_method_name=dim_reduction_method_name,
            output_dimension_size=cfg.dimensionality_reduction_cfg.no_discovery_reduced_dimensionality_size,
            embedding_model_name=cfg.embedding_cfg.embedding_model,
            random_seed=cfg.exp_cfg.seed,
        )
        # For pipeline_id=="no_discovery", the set of
        # generated capabilities are split into two sets
        train_capabilities, candidate_capabilities = get_lbo_train_set(
            input_data=capabilities,
            train_frac=cfg.lbo_cfg.train_frac,
            min_train_size=cfg.lbo_cfg.min_train_size,
            stratified=cfg.capabilities_cfg.method == "hierarchical",
            seed=cfg.exp_cfg.seed,
        )
        logger.info(
            f"Train capabilities ({len(train_capabilities)}):\n{train_capabilities}"
        )
        if num_lbo_runs > len(candidate_capabilities):
            logger.warning(
                f"Number of LBO runs ({num_lbo_runs}) exceeds the number of "
                + f"candidate capabilities ({len(candidate_capabilities)}). "
                + f"Setting the number of LBO runs to {len(candidate_capabilities)}."
            )
            num_lbo_runs = len(candidate_capabilities)
        # Run LBO to select capabilities
        new_capabilities = select_capabilities_using_lbo(
            capabilities=train_capabilities,
            embedding_name=dim_reduction_method_name,  # Since this embedding is reduced
            capabilities_pool=candidate_capabilities,
            subject_llm_name=cfg.subject_llm.name,
            num_lbo_iterations=num_lbo_runs,
        )

    elif "discover_new" in cfg.lbo_cfg.pipeline_id:
        # Reduce the dimensionality of all capability embeddings
        dim_reduction_method_name = (
            cfg.dimensionality_reduction_cfg.discover_new_reduced_dimensionality_method
        )
        dim_reduction_model = apply_dimensionality_reduction(
            capabilities=capabilities,
            dim_reduction_method_name=dim_reduction_method_name,
            output_dimension_size=cfg.dimensionality_reduction_cfg.discover_new_reduced_dimensionality_size,
            embedding_model_name=cfg.embedding_cfg.embedding_model,
            random_seed=cfg.exp_cfg.seed,
        )

        # Initialize the scientist LLM model for new capabilities'
        # task generation, solving, verification and evaluation (as judge)
        scientist_llm = Model(
            model_name=cfg.scientist_llm.name,
            model_provider=cfg.scientist_llm.provider,
        )
        scientist_llm_gen_cfg = cfg.scientist_llm.generation_cfg
        scientist_llm_gen_cfg_cap_gen = dict(
            scientist_llm_gen_cfg.capability_generation
        )
        # Initialize the subject LLM model for task evaluation
        subject_llm = Model(
            model_name=cfg.subject_llm.name,
            model_provider=cfg.subject_llm.provider,
            **dict(cfg.subject_llm.local_launch_cfg),
        )
        subject_llm_gen_cfg = dict(cfg.subject_llm.generation_cfg)
        subject_llm_gen_cfg.update(
            {
                "limit": cfg.capabilities_cfg.num_eval_tasks_per_capability,
            }
        )

        extended_run_id = f"{run_id}_{cfg.lbo_cfg.pipeline_id}"
        base_new_capability_dir = base_capability_dir.replace(
            f"capabilities_{run_id}",
            f"capabilities_{extended_run_id}",
        )

        if cfg.lbo_cfg.pipeline_id == "discover_new_lbo_knn":
            # Create LBO model by fitting on all capabilities
            lbo_model = fit_lbo(
                capabilities=capabilities,
                embedding_name=dim_reduction_method_name,
                subject_llm_name=subject_llm.get_model_name(),
            )

        random_seed = cfg.exp_cfg.seed
        new_capabilities = []
        for lbo_run_id in range(num_lbo_runs):
            num_retries = 0
            while num_retries < cfg.lbo_cfg.discover_new_retry_attempts:
                if cfg.lbo_cfg.pipeline_id == "discover_new_llm":
                    # Generate a new capability directly using the scientist LLM
                    # based on the previous capabilities and their scores
                    response = score_based_capability_discovery(
                        prev_capabilities=capabilities,
                        domain=cfg.capabilities_cfg.domain,
                        base_capability_dir=base_new_capability_dir,
                        user_prompt=prompts.SCORE_BASED_NEW_CAPABILITY_DISCOVERY_USER_PROMPT,
                        scientist_llm=scientist_llm,
                        scientist_llm_gen_cfg=scientist_llm_gen_cfg_cap_gen,
                        subject_llm_name=subject_llm.get_model_name(),
                        run_id=extended_run_id,
                        seed=random_seed,
                        retry_attempts=cfg.lbo_cfg.discover_new_llm_retry_attempts,
                    )
                elif cfg.lbo_cfg.pipeline_id == "discover_new_lbo_knn":
                    # Select K capabilities using LBO
                    knn_capabilities = select_k_capabilities(
                        lbo_model=lbo_model,
                        capabilities=capabilities,
                        select_k=cfg.lbo_cfg.select_k,
                        embedding_name=dim_reduction_method_name,
                    )
                    response = knn_based_capability_discovery(
                        knn_capabilities=knn_capabilities,
                        prev_capabilities=capabilities,
                        domain=cfg.capabilities_cfg.domain,
                        base_capability_dir=base_new_capability_dir,
                        user_prompt=prompts.KNN_BASED_NEW_CAPABILITY_DISCOVERY_USER_PROMPT,
                        scientist_llm=scientist_llm,
                        scientist_llm_gen_cfg=scientist_llm_gen_cfg_cap_gen,
                        run_id=extended_run_id,
                        seed=random_seed,
                        retry_attempts=cfg.lbo_cfg.discover_new_lbo_knn_retry_attempts,
                    )

                new_capability = response["capability"]

                # Generate tasks for the new capability
                generate_tasks_using_llm(
                    capability=new_capability,
                    scientist_llm=scientist_llm,
                    num_tasks=cfg.capabilities_cfg.num_gen_tasks_per_capability,
                    num_tasks_buffer=cfg.capabilities_cfg.num_gen_tasks_buffer,
                    scientist_llm_gen_cfg_task_gen=dict(
                        scientist_llm_gen_cfg.task_generation
                    ),
                    scientist_llm_gen_cfg_task_solve=dict(
                        scientist_llm_gen_cfg.task_solve
                    ),
                    scientist_llm_gen_cfg_task_verify=dict(
                        scientist_llm_gen_cfg.task_verify
                    ),
                    solve_sample_tasks=True,
                    few_shot=cfg.capabilities_cfg.task_gen_few_shot,
                    run_id=extended_run_id,
                    tasks_gen_retry_attempts=cfg.capabilities_cfg.tasks_gen_retry_attempts,
                    concurrency_task_solver=cfg.capabilities_cfg.concurrency_task_solver,
                    concurrency_task_verifier=cfg.capabilities_cfg.concurrency_task_verifier,
                    seed=random_seed,
                )

                # Verify if the new capability is complete
                if capability_satisfies_criterion(
                    capability=new_capability,
                    strict=False,
                    num_tasks_lower_bound=int(
                        cfg.capabilities_cfg.num_gen_tasks_per_capability
                        * (1 - cfg.capabilities_cfg.num_gen_tasks_buffer)
                    ),
                ):
                    break

                # Update the seed to generate a different capability since
                # the new capability is not complete
                logger.warning(
                    f"[Attempt #{num_retries + 1}] New capability {new_capability.name} is not complete. "
                    + "Updating seed to generate a different capability."
                )
                random_seed += 1
                scientist_llm_gen_cfg_cap_gen["seed"] += 1
                num_retries += 1

            # TODO: Raise error if the new capability is not complete even after
            # cfg.lbo_cfg.discover_new_retry_attempts retries? OR
            # use the last generated capability?
            # Not raising an error for now
            logger.info(
                f"Iteration {lbo_run_id + 1}/{num_lbo_runs}: "
                f"Generated capability {new_capability.name}"
            )

            # Evaluate the new capability using the subject LLM
            new_capability.evaluate(
                subject_llms=[subject_llm],
                gen_args=[subject_llm_gen_cfg],
                judge_llm=scientist_llm,  # Use scientist LLM as judge
                judge_llm_gen_args=dict(scientist_llm_gen_cfg.judge_llm),
                run_id=extended_run_id,
                concurrency_task_eval=cfg.capabilities_cfg.concurrency_task_eval,
            )

            # Add the new capability to the list
            new_capabilities.append(new_capability)
            capabilities.append(new_capability)

            if cfg.lbo_cfg.pipeline_id == "discover_new_lbo_knn":
                # Prepare the new capability for LBO
                # Embed the new capability using the same embedding model
                generate_and_set_capabilities_embeddings(
                    capabilities=[new_capability],
                    embedding_model_name=cfg.embedding_cfg.embedding_model,
                    embed_dimensions=cfg.embedding_cfg.embedding_size,
                )
                # Apply dimensionality reduction
                apply_dimensionality_reduction_to_test_capabilities(
                    capabilities=[new_capability],
                    dim_reduction_method=dim_reduction_model,
                    embedding_model_name=cfg.embedding_cfg.embedding_model,
                )
                # Get subject LLM score
                new_capability_score = new_capability.load_scores(
                    subject_llm_name=subject_llm.get_name(),
                )[subject_llm.get_name()]["mean"]
                # Update the LBO model with the new capability
                lbo_model.update(
                    new_capability.get_embedding(dim_reduction_model.method_name),
                    new_capability_score,
                )

        logger.info(f"New capabilities: {new_capabilities}")


if __name__ == "__main__":
    logger = logging.getLogger(__name__)

    main()
