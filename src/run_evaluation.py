"""Script to run capability evaluation on subject LLMs."""

import logging
import os

import hydra
from omegaconf import DictConfig

from generate_capabilities import (
    get_previous_capabilities,
    select_complete_capabilities,
)
from model import Model
from utils import constants
from utils.data_utils import check_cfg, get_run_id


@hydra.main(version_base=None, config_path="cfg", config_name="run_cfg")
def main(cfg: DictConfig) -> None:
    """
    Run capability evaluation on subject LLMs using the specified configuration.

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
    # Select the capabilities to evaluate
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

    # Initialize the scientist LLM model to be used as a judge
    scientist_llm = Model(
        model_name=cfg.scientist_llm.name,
        model_provider=cfg.scientist_llm.provider,
    )
    scientist_llm_gen_cfg = cfg.scientist_llm.generation_cfg
    # Initialize the subject LLM model
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

    for capability in capabilities:
        # Evaluate subject LLM on each capability
        capability.evaluate(
            subject_llms=[subject_llm],
            gen_args=[subject_llm_gen_cfg],
            judge_llm=scientist_llm,  # Use scientist LLM as judge
            judge_llm_gen_args=dict(scientist_llm_gen_cfg.judge_llm),
            run_id=run_id,
            concurrency_task_eval=cfg.capabilities_cfg.concurrency_task_eval,
        )
        if cfg.exp_cfg.trial_run:
            logger.info("Trial run completed, exiting after one capability.")
            break


if __name__ == "__main__":
    logger = logging.getLogger(__name__)

    main()
