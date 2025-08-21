"""Multi-agent task generation system for generating tasks for each capability."""

import asyncio
import logging
import traceback

import hydra
from omegaconf import DictConfig, OmegaConf

from .task_generation import generate_tasks


log = logging.getLogger("agentic_task_gen")


@hydra.main(version_base=None, config_path="cfg", config_name="agentic_config")
def main(cfg: DictConfig) -> None:
    """Run the multi-agent task generation system."""
    log.info("Starting multi-agent task generation")
    log.info("Configuration:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    # Check for capabilities_tag parameter
    capabilities_tag = cfg.pipeline_tags.capabilities_tag
    if capabilities_tag:
        log.info(f"Using capabilities from tag: {capabilities_tag}")
    else:
        log.warning(
            "No capabilities_tag provided. Please provide --pipeline_tags.capabilities_tag=<tag> to specify which capabilities to use."
        )
        return

    try:
        asyncio.run(generate_tasks(cfg, capabilities_tag))
    except Exception as e:
        log.error(f"Task generation failed: {e}")
        log.error(f"Full traceback: {traceback.format_exc()}")
        raise


if __name__ == "__main__":
    main()
