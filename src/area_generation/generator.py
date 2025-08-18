"""Main area generation orchestration function."""

import logging
from pathlib import Path

import openlit
from autogen_core import DefaultTopicId, SingleThreadedAgentRuntime
from autogen_ext.models.openai import OpenAIChatCompletionClient
from langfuse import Langfuse
from omegaconf import DictConfig

from ..langfuse_client import LangfuseChatCompletionClient, start_root_span
from .messages import Domain
from .moderator import AreaModerator
from .scientist import AreaScientist


log = logging.getLogger("agentic_area_gen.generator")

DEFAULT_NUM_SCIENTISTS = 2


async def generate_areas(cfg: DictConfig) -> None:
    """Generate areas using multi-agent debate system."""
    try:
        log.info("Starting area generation process")

        langfuse = Langfuse(
            blocked_instrumentation_scopes=["autogen SingleThreadedAgentRuntime"]
        )
        if langfuse.auth_check():
            log.info("Langfuse client is authenticated and ready!")
        else:
            log.error("Authentication failed. Please check your credentials and host.")

        openlit.init(tracer=langfuse._otel_tracer, disable_batch=True)

        max_round = cfg.debate_cfg.max_round
        runtime = SingleThreadedAgentRuntime()

        output_dir = (
            Path.home()
            / cfg.debate_cfg.output_dir
            / cfg.capabilities_cfg.domain.replace(" ", "_")
            / cfg.exp_cfg.exp_id
            / "areas"
        )
        log.info(f"Output directory: {output_dir}")

        with start_root_span(
            name=f"ace_area_generation:{cfg.capabilities_cfg.domain}",
            input={"exp_id": cfg.exp_cfg.exp_id, "domain": cfg.capabilities_cfg.domain},
        ):
            await AreaScientist.register(
                runtime,
                "AreaScientistA",
                lambda: AreaScientist(
                    model_client=LangfuseChatCompletionClient(
                        OpenAIChatCompletionClient(
                            model=cfg.agents.scientist_a.name,
                            seed=cfg.agents.scientist_a.seed,
                        ),
                        agent_role="ScientistA",
                        default_model=cfg.agents.scientist_a.name,  # optional but nice to log
                    ),
                    scientist_id="A",
                    max_round=max_round,
                ),
            )

            await AreaScientist.register(
                runtime,
                "AreaScientistB",
                lambda: AreaScientist(
                    model_client=LangfuseChatCompletionClient(
                        OpenAIChatCompletionClient(
                            model=cfg.agents.scientist_b.name,
                            seed=cfg.agents.scientist_b.seed,
                        ),
                        agent_role="ScientistB",
                        default_model=cfg.agents.scientist_b.name,
                    ),
                    scientist_id="B",
                    max_round=max_round,
                ),
            )

            await AreaModerator.register(
                runtime,
                "AreaModerator",
                lambda: AreaModerator(
                    model_client=LangfuseChatCompletionClient(
                        OpenAIChatCompletionClient(
                            model=cfg.agents.moderator.name,
                            seed=cfg.agents.moderator.seed,
                        ),
                        agent_role="Moderator",
                        default_model=cfg.agents.moderator.name,
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
