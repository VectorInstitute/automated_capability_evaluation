"""Main capability generation orchestration functions."""

import asyncio
import json
import logging
import traceback
from datetime import datetime
from pathlib import Path

import openlit
from autogen_core import DefaultTopicId, SingleThreadedAgentRuntime
from autogen_ext.models.openai import OpenAIChatCompletionClient
from langfuse import Langfuse
from omegaconf import DictConfig

from .messages import Area
from .moderator import CapabilityModerator
from .scientist import CapabilityScientist


log = logging.getLogger("agentic_cap_gen.generator")
langfuse = Langfuse(
    blocked_instrumentation_scopes=["autogen SingleThreadedAgentRuntime"]
)
openlit.init(tracer=langfuse._otel_tracer, disable_batch=True)


async def generate_capabilities_for_area(
    cfg: DictConfig, area: Area, output_dir: Path
) -> None:
    """Generate capabilities for a single area."""
    try:
        log.info(f"Generating capabilities for area: {area.name}")

        domain_name = cfg.capabilities_cfg.domain

        runtime = SingleThreadedAgentRuntime()

        await CapabilityScientist.register(
            runtime,
            "CapabilityScientistA",
            lambda: CapabilityScientist(
                model_client=OpenAIChatCompletionClient(
                    model=cfg.agents.scientist_a.name,
                    seed=cfg.agents.scientist_a.seed,
                ),
                scientist_id="A",
                max_round=cfg.debate_cfg.max_round,
                expected_capabilities=cfg.capabilities_cfg.num_capabilities_per_area,
                domain=domain_name,
            ),
        )

        await CapabilityScientist.register(
            runtime,
            "CapabilityScientistB",
            lambda: CapabilityScientist(
                model_client=OpenAIChatCompletionClient(
                    model=cfg.agents.scientist_b.name,
                    seed=cfg.agents.scientist_b.seed,
                ),
                scientist_id="B",
                max_round=cfg.debate_cfg.max_round,
                expected_capabilities=cfg.capabilities_cfg.num_capabilities_per_area,
                domain=domain_name,
            ),
        )

        await CapabilityModerator.register(
            runtime,
            "CapabilityModerator",
            lambda: CapabilityModerator(
                model_client=OpenAIChatCompletionClient(
                    model=cfg.agents.moderator.name,
                    seed=cfg.agents.moderator.seed,
                ),
                num_scientists=2,
                num_capabilities=cfg.capabilities_cfg.num_capabilities_per_area,
                max_round=cfg.debate_cfg.max_round,
                output_dir=output_dir,
                domain=domain_name,
            ),
        )

        # Start runtime and process the area
        runtime.start()
        await runtime.publish_message(area, DefaultTopicId())
        log.info(f"Area message published: {area.name}")

        # Wait for the runtime to stop when idle
        try:
            await runtime.stop_when_idle()
            log.info(f"Completed generating capabilities for area: {area.name}")
        except Exception as e:
            log.error(f"Error while generating capabilities for area {area.name}: {e}")
            raise

    except Exception as e:
        log.error(f"Error in generating capabilities for {area.name}: {e}")
        log.error(f"Traceback: {traceback.format_exc()}")
        raise


async def generate_capabilities(cfg: DictConfig, areas_tag: str) -> None:
    """Generate capabilities using multi-agent debate system for each area."""
    domain_name = cfg.capabilities_cfg.domain
    exp_id = cfg.exp_cfg.exp_id
    capabilities_tag = f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log.info(f"Capabilities will be saved with tag: {capabilities_tag}")

    with langfuse.start_as_current_span(
        name=f"ace_capability_generation:{domain_name}:{exp_id}:{capabilities_tag}"
    ) as span:
        try:
            log.info("Starting capability generation process")
            max_round = cfg.debate_cfg.max_round

            span.update_trace(
                metadata={
                    "domain": domain_name,
                    "exp_id": exp_id,
                    "max_round": max_round,
                    "num_capabilities_per_area": cfg.capabilities_cfg.num_capabilities_per_area,
                    "areas_tag": areas_tag,
                    "capabilities_tag": capabilities_tag,
                },
                tags=["capability_generation_process", exp_id],
            )

            areas_file = (
                Path.home()
                / cfg.debate_cfg.output_dir
                / domain_name.replace(" ", "_")
                / exp_id
                / "areas"
                / areas_tag
                / "areas.json"
            )

            if not areas_file.exists():
                raise FileNotFoundError(f"Areas file not found: {areas_file}")

            with open(areas_file, "r", encoding="utf-8") as f:
                areas_data = json.load(f)

            # Parse areas from the JSON data
            areas = []
            if isinstance(areas_data, dict) and "areas" in areas_data:
                for area_dict in areas_data["areas"]:
                    if (
                        isinstance(area_dict, dict)
                        and "name" in area_dict
                        and "description" in area_dict
                    ):
                        areas.append(
                            Area(
                                name=area_dict["name"],
                                description=area_dict["description"],
                            )
                        )

            if not areas:
                raise ValueError(f"No valid areas found in {areas_file}")

            log.info(
                f"Found {len(areas)} areas to process: {[area.name for area in areas]}"
            )

            output_dir = (
                Path.home()
                / cfg.debate_cfg.output_dir
                / domain_name.replace(" ", "_")
                / exp_id
                / "capabilities"
                / capabilities_tag
            )
            log.info(f"Output directory: {output_dir}")
            span.update(metadata={"output_dir": str(output_dir)})

            # Process each area individually with fresh agents
            for i, area in enumerate(areas):
                log.info(f"Processing area {i + 1}/{len(areas)}: {area.name}")

                await generate_capabilities_for_area(cfg, area, output_dir)

                log.info(f"Completed area {i + 1}/{len(areas)}: {area.name}")

                await asyncio.sleep(1)

            print(f"Capabilities generated with tag: {capabilities_tag}")

        except Exception as e:
            log.error(f"Error in generate_capabilities: {e}")
            log.error(f"Traceback: {traceback.format_exc()}")
            span.update(level="ERROR", status_message=str(e))
            langfuse.flush()
            span.end()
            raise
