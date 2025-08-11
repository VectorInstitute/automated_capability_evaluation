"""Main area generation orchestration function."""

import logging
from pathlib import Path

from autogen_core import DefaultTopicId, SingleThreadedAgentRuntime
from autogen_ext.models.openai import OpenAIChatCompletionClient
from omegaconf import DictConfig

from .messages import Domain
from .moderator import AreaModerator
from .scientist import AreaScientist

log = logging.getLogger("agentic_area_gen.generator")

DEFAULT_NUM_SCIENTISTS = 2


async def generate_areas(cfg: DictConfig) -> None:
    """Generate areas using multi-agent debate system."""
    try:
        log.info("Starting area generation process")

        max_round = cfg.debate_cfg.max_round
        runtime = SingleThreadedAgentRuntime()

        output_dir = (
            Path.home()
            / cfg.debate_cfg.output_dir
            / cfg.capabilities_cfg.domain
            / cfg.exp_cfg.exp_id
            / "areas"
        )
        log.info(f"Output directory: {output_dir}")

        await AreaScientist.register(
            runtime,
            "AreaScientistA",
            lambda: AreaScientist(
                model_client=OpenAIChatCompletionClient(
                    model=cfg.agents.scientist_a.name,
                    seed=cfg.agents.scientist_a.seed,
                ),
                scientist_id="A",
                max_round=max_round,
            ),
        )

        await AreaScientist.register(
            runtime,
            "AreaScientistB",
            lambda: AreaScientist(
                model_client=OpenAIChatCompletionClient(
                    model=cfg.agents.scientist_b.name,
                    seed=cfg.agents.scientist_b.seed,
                ),
                scientist_id="B",
                max_round=max_round,
            ),
        )

        await AreaModerator.register(
            runtime,
            "AreaModerator",
            lambda: AreaModerator(
                model_client=OpenAIChatCompletionClient(
                    model=cfg.agents.moderator.name,
                    seed=cfg.agents.moderator.seed,
                ),
                num_scientists=DEFAULT_NUM_SCIENTISTS,
                num_final_areas=cfg.capabilities_cfg.num_capability_areas,
                max_round=max_round,
                output_dir=output_dir,
            ),
        )

        # Use domain from config
        domain = Domain(name=cfg.capabilities_cfg.domain)
        runtime.start()
        await runtime.publish_message(domain, DefaultTopicId())
        log.info(f"Domain message published: {domain.name}")

        # Wait for the runtime to stop when idle.
        try:
            await runtime.stop_when_idle()
            log.info("Runtime stopped when idle")
        except Exception as e:
            log.error(f"Error while waiting for runtime to stop: {e}")
            raise

    except Exception as e:
        log.error(f"Error in generate_areas: {e}")
        raise 