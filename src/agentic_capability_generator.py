"""Multi-agent debate system for generating capabilities for each area."""

import asyncio
import logging
import traceback

import hydra
from omegaconf import DictConfig, OmegaConf

from .capability_generation import generate_capabilities

log = logging.getLogger("agentic_cap_gen")


@hydra.main(version_base=None, config_path="cfg", config_name="agentic_config")
def main(cfg: DictConfig) -> None:
    """Run the multi-agent debate-based capability generation system."""
    log.info("Starting multi-agent debate-based capability generation")
    log.info("Configuration:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    try:
        asyncio.run(generate_capabilities(cfg))
    except Exception as e:
        log.error(f"Capability generation failed: {e}")
        log.error(f"Full traceback: {traceback.format_exc()}")
        raise


if __name__ == "__main__":
    main()
