import json  # noqa: D100
import logging  # noqa: D100
import os  # noqa: D100

import hydra
from omegaconf import DictConfig

from src.generate_capabilities import (
    get_previous_capabilities,
)


logger = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="example_cfg",
    config_name="capability_score_visualization",
)
def main(cfg: DictConfig) -> None:
    """Plot capability scores across areas for each subject LLM."""
    # Set the base capability directory
    capability_dir = os.path.join(
        cfg.capabilities_cfg.saved_capabilities_dir,
        cfg.capabilities_cfg.domain,
    )

    # Fetch previously generated capabilities
    capabilities = get_previous_capabilities(capability_dir=capability_dir)
    logger.info(f"Loaded {len(capabilities)} capabilities from {capability_dir}")
    # Assert that the capabilities list is not empty
    assert capabilities, "No capabilities found in the specified directory."

    # Load capability embeddings and scores
    all_llms_scores = {}
    for subject_llm in cfg.score_cfg.subject_llm_names:
        llm_capability_scores = {}
        # Set the base capability score directory
        capability_score_file = os.path.join(
            cfg.score_cfg.read_score_dir,
            f"{subject_llm}_capability_scores.json",
        )

        logger.info(f"Loading scores for {subject_llm}")
        # Load capability scores from the specified file
        with open(capability_score_file, "r") as f:
            capability_scores = json.load(f)

        for cap_name, value in capability_scores.items():
            llm_capability_scores[cap_name] = value["score"]

        all_llms_scores[subject_llm] = llm_capability_scores
        print(
            f"Loaded {len(llm_capability_scores)} capability scores for {subject_llm}"
        )

        # extract the area
        capability_dir = os.path.join(
            cfg.capabilities_cfg.saved_capabilities_dir,
            cfg.capabilities_cfg.domain,
        )

    # Fetch previously generated capabilities
    capabilities = get_previous_capabilities(capability_dir=capability_dir)
    logger.info(f"Loaded {len(capabilities)} capabilities from {capability_dir}")


if __name__ == "__main__":
    main()
