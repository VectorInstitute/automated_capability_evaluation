"""Main area generation orchestration function."""

import logging
from datetime import datetime
from pathlib import Path

import openlit
from autogen_core import DefaultTopicId, SingleThreadedAgentRuntime
from autogen_ext.models.openai import OpenAIChatCompletionClient
from langfuse import Langfuse
from omegaconf import DictConfig

from .messages import Domain
from .moderator import AreaModerator
from .scientist import AreaScientist


log = logging.getLogger("agentic_area_gen.generator")

langfuse = Langfuse(
    blocked_instrumentation_scopes=["autogen SingleThreadedAgentRuntime"]
)
openlit.init(tracer=langfuse._otel_tracer, disable_batch=True)


DEFAULT_NUM_SCIENTISTS = 2


async def generate_areas(cfg: DictConfig) -> None:
    """Generate areas using multi-agent debate system."""
    domain_name = cfg.capabilities_cfg.domain
    exp_id = cfg.exp_cfg.exp_id
    areas_tag = f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log.info(f"Areas will be saved with tag: {areas_tag}")

    with langfuse.start_as_current_span(
        name=f"ace_area_generation:{domain_name}:{exp_id}:{areas_tag}"
    ) as span:
        try:
            log.info("Starting area generation process")

            max_round = cfg.debate_cfg.max_round
            runtime = SingleThreadedAgentRuntime()

            output_dir = (
                Path.home()
                / cfg.debate_cfg.output_dir
                / domain_name.replace(" ", "_")
                / exp_id
                / "areas"
                / f"{areas_tag}"
            )
            log.info(f"Output directory: {output_dir}")

            span.update_trace(
                metadata={
                    "domain": domain_name,
                    "exp_id": exp_id,
                    "max_round": max_round,
                    "num_capability_areas": cfg.capabilities_cfg.num_capability_areas,
                    "output_dir": str(output_dir),
                    "areas_tag": areas_tag,
                },
                tags=["area_generation", exp_id],
            )

            # Register agents with the runtime
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
            domain = Domain(name=domain_name)
            runtime.start()
            await runtime.publish_message(domain, DefaultTopicId())
            log.info(f"Domain message published: {domain.name}")

            # Wait for the runtime to stop when idle.
            try:
                await runtime.stop_when_idle()
                log.info("Runtime stopped when idle")
            except Exception as e:
                log.error(f"Error while waiting for runtime to stop: {e}")
                span.update(level="ERROR", status_message=str(e))
                raise

            print(f"Areas generated with tag: {areas_tag}")

        except Exception as e:
            log.error(f"Error in generate_areas: {e}")
            span.update(level="ERROR", status_message=str(e))
            langfuse.flush()
            span.end()
            raise
