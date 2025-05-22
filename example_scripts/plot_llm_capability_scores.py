import logging  # noqa: D100
import os  # noqa: D100

import hydra
from omegaconf import DictConfig
from tqdm import tqdm

from src.generate_capabilities import (
    get_previous_capabilities,
    plot_capability_scores_spider_and_bar_chart,
    select_complete_capabilities,
)
from src.utils.data_utils import get_run_id


logger = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="example_cfg",
    config_name="capability_score_visualization",
)
def main(cfg: DictConfig) -> None:
    """Plot capability scores across areas for each subject LLM."""
    run_id = get_run_id(cfg)
    # Set the base capability directory
    capability_dir = os.path.join(
        cfg.capabilities_cfg.saved_capabilities_dir,
        cfg.capabilities_cfg.domain,
    )

    # Fetch previously generated capabilities
    # Read the capabilities from the base directory
    capabilities = get_previous_capabilities(
        capability_dir=capability_dir,
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
    # Sort capabilities by name
    capabilities = sorted(capabilities, key=lambda x: x.name)
    # Pre-load capability scores
    for subject_llm_name in cfg.score_cfg.subject_llm_names:
        for capability in tqdm(capabilities, desc="Loading capability scores"):
            capability.load_scores(
                subject_llm_name=subject_llm_name,
            )

    # Plot capability scores based on area --> spider and bar charts.
    plot_capability_scores_spider_and_bar_chart(
        capabilities,
        cfg.score_cfg.subject_llm_names,
        cfg.score_cfg.plot_capabilities_score_dir,
        plot_name="llm_scores",
        plot_spider_chart=True,
        plot_grouped_bars=True,
    )


if __name__ == "__main__":
    main()
