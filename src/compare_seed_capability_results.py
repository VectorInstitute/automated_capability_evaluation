"""Compare the scores of mock seed capabilities with actual seed capabilities."""

import json
import logging
import os

import hydra
from omegaconf import DictConfig

from src.generate_capabilities import (
    get_previous_capabilities,
)
from src.utils import constants
from src.utils.capability_utils import (
    read_score_inspect_json,
)


@hydra.main(
    version_base=None,
    config_path="cfg",
    config_name="compare_seed_capability_results_cfg",
)
def main(cfg: DictConfig) -> None:
    """
    Compare the scores of mock seed capabilities with actual seed capabilities.

    Args:
        cfg (DictConfig): Configuration for the model.
    """
    run_id = cfg.exp_id
    subject_llm_name = cfg.subject_llm.name
    num_tasks = cfg.num_tasks

    # Set the base capability directory
    base_capability_dir = os.path.join(
        constants.BASE_ARTIFACTS_DIR,
        f"capabilities_{run_id}",
        cfg.domain,
    )
    # Read the capabilities from the base directory
    capabilities = get_previous_capabilities(
        capability_dir=base_capability_dir,
        score_dir_suffix=run_id,
    )
    capabilities = sorted(capabilities, key=lambda x: x.name)
    logger.info(f"Capability names:\n{capabilities}")

    # Set seed capability directory
    seed_capability_dir = os.path.join(
        constants.SEED_CAPABILITIES_SCORE_DIR,
        subject_llm_name,
        cfg.domain,
    )

    output_dict = {}
    for capability in capabilities:
        if num_tasks == -1:
            num_tasks = len(capability.get_tasks())
        elif num_tasks > len(capability.get_tasks()):
            logger.warning(
                f"[{capability.name}] Requested number of tasks ({num_tasks}) is greater than the available tasks ({len(capability.get_tasks())}). Setting num_tasks to the available tasks."
            )
            num_tasks = len(capability.get_tasks())

        # Get scores for generated mock seed capability tasks
        mock_scores = capability.load_scores(num_tasks=num_tasks, seed=cfg.seed)[
            subject_llm_name
        ]

        # Get scores for num_tasks tasks from the original seed capability dataset
        capability_name = capability.name
        actual_scores = read_score_inspect_json(
            os.path.join(
                seed_capability_dir, capability_name, f"{capability_name}.json"
            ),
            num_tasks=num_tasks,
            seed=cfg.seed,
        )

        output_dict[capability_name] = {
            "mock_scores": mock_scores,
            "actual_scores": actual_scores,
            "num_tasks": num_tasks,
        }

    logger.info(f"Score comparison:\n{json.dumps(output_dict, indent=4)}")


if __name__ == "__main__":
    logger = logging.getLogger(__name__)

    main()
