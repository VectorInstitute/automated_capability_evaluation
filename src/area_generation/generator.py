"""Main area generation orchestration function."""

import logging
import traceback
from datetime import datetime
from pathlib import Path

from autogen_core import (
    EVENT_LOGGER_NAME,
    ROOT_LOGGER_NAME,
    TRACE_LOGGER_NAME,
    DefaultTopicId,
    SingleThreadedAgentRuntime,
)
from langfuse import Langfuse
from omegaconf import DictConfig

from src.area_generation.messages import Domain
from src.area_generation.moderator import AreaModerator
from src.area_generation.scientist import AreaScientist
from src.utils.model_client_utils import get_standard_model_client


log = logging.getLogger("agentic_area_gen.generator")
logging.getLogger(ROOT_LOGGER_NAME).setLevel(logging.WARNING)
logging.getLogger(TRACE_LOGGER_NAME).setLevel(logging.WARNING)
logging.getLogger(EVENT_LOGGER_NAME).setLevel(logging.WARNING)


async def generate_areas(cfg: DictConfig, langfuse_client: Langfuse) -> None:
    """Generate areas using multi-agent debate system."""
    domain_name = cfg.global_cfg.domain
    exp_id = cfg.exp_cfg.exp_id
    max_round = cfg.debate_cfg.max_round
    num_areas = cfg.area_generation.num_areas
    areas_tag = f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    with langfuse_client.start_as_current_span(
        name=f"ace_area_generation:{domain_name}:{exp_id}:{areas_tag}"
    ) as span:
        try:
            msg = f"Areas will be saved with tag: {areas_tag}"
            log.info(msg)
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
                / cfg.global_cfg.output_dir
                / domain_name.replace(" ", "_")
                / exp_id
                / "areas"
                / f"{areas_tag}"
            )
            output_dir.mkdir(parents=True, exist_ok=True)

            msg = f"Output directory: {output_dir}"
            log.info(msg)
            span.update(
                metadata={
                    "output_directory_configured": msg,
                    "output_dir": str(output_dir),
                }
            )

            span.update_trace(
                metadata={
                    "domain": domain_name,
                    "exp_id": exp_id,
                    "max_round": max_round,
                    "num_areas": num_areas,
                    "areas_tag": areas_tag,
                },
                tags=["area_generation_process", exp_id],
            )

            runtime = SingleThreadedAgentRuntime()

            await AreaScientist.register(
                runtime,
                "AreaScientistA",
                lambda: AreaScientist(
                    model_client=get_standard_model_client(
                        model_name=cfg.agents.scientist_a.model_name,
                        seed=cfg.agents.scientist_a.seed,
                    ),
                    scientist_id="A",
                    langfuse_client=langfuse_client,
                ),
            )

            await AreaScientist.register(
                runtime,
                "AreaScientistB",
                lambda: AreaScientist(
                    model_client=get_standard_model_client(
                        model_name=cfg.agents.scientist_b.model_name,
                        seed=cfg.agents.scientist_b.seed,
                    ),
                    scientist_id="B",
                    langfuse_client=langfuse_client,
                ),
            )

            await AreaModerator.register(
                runtime,
                "AreaModerator",
                lambda: AreaModerator(
                    model_client=get_standard_model_client(
                        model_name=cfg.agents.moderator.model_name,
                        seed=cfg.agents.moderator.seed,
                    ),
                    num_scientists=2,
                    num_final_areas=num_areas,
                    max_round=max_round,
                    output_dir=output_dir,
                    langfuse_client=langfuse_client,
                ),
            )

            msg = "All area agents registered successfully"
            log.info(msg)
            span.update(
                metadata={
                    "agents_registered": msg,
                    "scientists": ["A", "B"],
                    "moderator": True,
                    "max_rounds": max_round,
                    "expected_areas": num_areas,
                }
            )

            runtime.start()

            domain_message = Domain(name=domain_name)
            await runtime.publish_message(domain_message, DefaultTopicId())

            msg = f"Domain message published: {domain_name}"
            log.info(msg)
            span.update(
                metadata={
                    "domain_published": msg,
                    "domain_name": domain_name,
                }
            )

            try:
                await runtime.stop_when_idle()

                msg = "Runtime stopped - area generation completed"
                log.info(msg)
                span.update(metadata={"runtime_completed": msg})

                print(f"Areas generated with tag: {areas_tag}")
            except Exception as e:
                msg = f"Error while waiting for runtime to stop: {e}"
                log.error(msg)
                span.update(
                    level="ERROR",
                    status_message=str(e),
                    metadata={
                        "runtime_error": msg,
                        "error": str(e),
                    },
                )
                raise

        except Exception as e:
            error_msg = f"Error in generate_areas: {e}"
            traceback_msg = f"Traceback: {traceback.format_exc()}"

            log.error(error_msg)
            log.error(traceback_msg)

            span.update(
                level="ERROR",
                status_message=str(e),
                metadata={
                    "generation_error": error_msg,
                    "error": str(e),
                    "traceback": traceback_msg,
                },
            )

            if langfuse_client is None:
                langfuse_client.flush()
            raise
