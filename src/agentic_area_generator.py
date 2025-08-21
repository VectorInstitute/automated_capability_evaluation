"""Multi-agent debate system for generating capability areas."""

import asyncio
import logging
import traceback

import hydra
from omegaconf import DictConfig, OmegaConf

from .area_generation import generate_areas


log = logging.getLogger("agentic_area_gen")


@hydra.main(version_base=None, config_path="cfg", config_name="agentic_config")
def main(cfg: DictConfig) -> None:
    """Run the multi-agent debate-based area generation system."""
    log.info("Starting multi-agent debate-based area generation")
    log.info("Configuration:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    try:
        asyncio.run(generate_areas(cfg))
    except Exception as e:
        log.error(f"Area generation failed: {e}")
        log.error(f"Full traceback: {traceback.format_exc()}")
        raise


if __name__ == "__main__":
    main()
