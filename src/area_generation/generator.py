"""Main area generation orchestration function."""

import logging
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

from autogen_core import (
    EVENT_LOGGER_NAME,
    ROOT_LOGGER_NAME,
    TRACE_LOGGER_NAME,
    DefaultTopicId,
    SingleThreadedAgentRuntime,
)
from autogen_ext.models.openai import OpenAIChatCompletionClient
from langfuse import Langfuse
from omegaconf import DictConfig

from .messages import Domain
from .moderator import AreaModerator
from .scientist import AreaScientist


log = logging.getLogger("agentic_area_gen.generator")
logging.getLogger(ROOT_LOGGER_NAME).setLevel(logging.WARNING)
logging.getLogger(TRACE_LOGGER_NAME).setLevel(logging.WARNING)
logging.getLogger(EVENT_LOGGER_NAME).setLevel(logging.WARNING)


async def generate_areas(cfg: DictConfig, langfuse_client: Langfuse = None) -> None:
    """Generate areas using multi-agent debate system."""
    domain_name = cfg.capabilities_cfg.domain
    exp_id = cfg.exp_cfg.exp_id
    max_round = cfg.debate_cfg.max_round
    areas_tag = f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    lf = langfuse_client
    if lf is None:
        lf = Langfuse(
            blocked_instrumentation_scopes=["autogen SingleThreadedAgentRuntime"]
        )

    with (
        lf.start_as_current_span(
            name=f"ace_area_generation:{domain_name}:{exp_id}:{areas_tag}"
        )
        if lf
        else nullcontext() as span
    ):
        try:
            msg = f"Areas will be saved with tag: {areas_tag}"
            log.info(msg)
            if span:
                span.update(
                    metadata={
                        "generation_started": msg,
                        "areas_tag": areas_tag,
                        "domain": domain_name,
                        "exp_id": exp_id,
                    }
                )

            output_dir = (
                Path.home()
                / cfg.debate_cfg.output_dir
                / domain_name.replace(" ", "_")
                / exp_id
                / "areas"
                / f"{areas_tag}"
            )

            msg = f"Output directory: {output_dir}"
            log.info(msg)
            if span:
                span.update(
                    metadata={
                        "output_directory_configured": msg,
                        "output_dir": str(output_dir),
                    }
                )

            if span:
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

            runtime = SingleThreadedAgentRuntime()

            await AreaScientist.register(
                runtime,
                "AreaScientistA",
                lambda: AreaScientist(
                    model_client=OpenAIChatCompletionClient(
                        model=cfg.agents.scientist_a.model_name,
                        seed=cfg.agents.scientist_a.seed,
                    ),
                    scientist_id="A",
                    langfuse_client=lf,
                ),
            )

            await AreaScientist.register(
                runtime,
                "AreaScientistB",
                lambda: AreaScientist(
                    model_client=OpenAIChatCompletionClient(
                        model=cfg.agents.scientist_b.model_name,
                        seed=cfg.agents.scientist_b.seed,
                    ),
                    scientist_id="B",
                    langfuse_client=lf,
                ),
            )

            await AreaModerator.register(
                runtime,
                "AreaModerator",
                lambda: AreaModerator(
                    model_client=OpenAIChatCompletionClient(
                        model=cfg.agents.moderator.model_name,
                        seed=cfg.agents.moderator.seed,
                    ),
                    num_scientists=2,
                    num_final_areas=cfg.capabilities_cfg.num_capability_areas,
                    max_round=max_round,
                    output_dir=output_dir,
                    langfuse_client=lf,
                ),
            )

            if span:
                span.update(
                    metadata={
                        "agents_registered": "All agents registered successfully",
                        "scientists": ["A", "B"],
                        "moderator": True,
                        "max_rounds": max_round,
                    }
                )

            domain = Domain(name=domain_name)
            runtime.start()
            await runtime.publish_message(domain, DefaultTopicId())

            msg = f"Domain message published: {domain.name}"
            log.info(msg)
            if span:
                span.update(
                    metadata={"domain_published": msg, "domain_name": domain.name}
                )

            try:
                await runtime.stop_when_idle()

                msg = "Runtime stopped when idle"
                log.info(msg)
                if span:
                    span.update(metadata={"runtime_completed": msg})
            except Exception as e:
                msg = f"Error while waiting for runtime to stop: {e}"
                log.error(msg)
                if span:
                    span.update(
                        level="ERROR",
                        status_message=str(e),
                        metadata={"runtime_error": msg, "error": str(e)},
                    )
                raise

            msg = f"Areas generated with tag: {areas_tag}"
            print(msg)
            if span:
                span.update(
                    metadata={"generation_completed": msg, "areas_tag": areas_tag}
                )

        except Exception as e:
            msg = f"Error in generate_areas: {e}"
            log.error(msg)
            if span:
                span.update(
                    level="ERROR",
                    status_message=str(e),
                    metadata={"generation_error": msg, "error": str(e)},
                )
            if langfuse_client is None:
                lf.flush()
            raise
