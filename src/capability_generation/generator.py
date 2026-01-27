"""Main capability generation orchestration functions."""

import asyncio
import json
import logging
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

from autogen_core import (
    EVENT_LOGGER_NAME,
    ROOT_LOGGER_NAME,
    TRACE_LOGGER_NAME,
    DefaultTopicId,
    SingleThreadedAgentRuntime,
)
from langfuse import Langfuse
from omegaconf import DictConfig

from src.capability_generation.messages import Area
from src.capability_generation.moderator import CapabilityModerator
from src.capability_generation.scientist import CapabilityScientist
from src.utils.model_client_utils import get_standard_model_client


log = logging.getLogger("agentic_cap_gen.generator")
logging.getLogger(ROOT_LOGGER_NAME).setLevel(logging.WARNING)
logging.getLogger(TRACE_LOGGER_NAME).setLevel(logging.WARNING)
logging.getLogger(EVENT_LOGGER_NAME).setLevel(logging.WARNING)


async def generate_capabilities_for_area(
    cfg: DictConfig, area: Area, output_dir: Path, langfuse_client: Langfuse
) -> None:
    """Generate capabilities for a single area."""
    with langfuse_client.start_as_current_span(
        name=f"capability_generation_for_area:{area.name}"
    ) as span:
        try:
            msg = f"Generating capabilities for area: {area.name}"
            log.info(msg)
            span.update(
                metadata={
                    "area_generation_started": msg,
                    "area_name": area.name,
                    "area_description": area.description,
                }
            )

            domain_name = cfg.global_cfg.domain
            max_round = cfg.debate_cfg.max_round
            expected_capabilities = cfg.capability_generation.num_capabilities_per_area

            runtime = SingleThreadedAgentRuntime()

            await CapabilityScientist.register(
                runtime,
                "CapabilityScientistA",
                lambda: CapabilityScientist(
                    model_client=get_standard_model_client(
                        model_name=cfg.agents.scientist_a.model_name,
                        seed=cfg.agents.scientist_a.seed,
                    ),
                    scientist_id="A",
                    langfuse_client=langfuse_client,
                ),
            )

            await CapabilityScientist.register(
                runtime,
                "CapabilityScientistB",
                lambda: CapabilityScientist(
                    model_client=get_standard_model_client(
                        model_name=cfg.agents.scientist_b.model_name,
                        seed=cfg.agents.scientist_b.seed,
                    ),
                    scientist_id="B",
                    langfuse_client=langfuse_client,
                ),
            )

            await CapabilityModerator.register(
                runtime,
                "CapabilityModerator",
                lambda: CapabilityModerator(
                    model_client=get_standard_model_client(
                        model_name=cfg.agents.moderator.model_name,
                        seed=cfg.agents.moderator.seed,
                    ),
                    num_scientists=2,
                    num_capabilities=expected_capabilities,
                    max_round=max_round,
                    output_dir=output_dir,
                    domain=domain_name,
                    langfuse_client=langfuse_client,
                ),
            )

            span.update(
                metadata={
                    "agents_registered": "All capability agents registered successfully",
                    "scientists": ["A", "B"],
                    "moderator": True,
                    "max_rounds": max_round,
                    "expected_capabilities": expected_capabilities,
                }
            )

            runtime.start()
            await runtime.publish_message(area, DefaultTopicId())

            msg = f"Area message published: {area.name}"
            log.info(msg)
            span.update(metadata={"area_published": msg, "area_name": area.name})

            try:
                await runtime.stop_when_idle()

                msg = f"Runtime stopped for area: {area.name}"
                log.info(msg)
                span.update(metadata={"runtime_completed": msg})
            except Exception as e:
                msg = (
                    f"Error while waiting for runtime to stop for area {area.name}: {e}"
                )
                log.error(msg)
                span.update(
                    level="ERROR",
                    status_message=str(e),
                    metadata={
                        "runtime_error": msg,
                        "error": str(e),
                        "area_name": area.name,
                    },
                )
                raise

        except Exception as e:
            error_msg = f"Error generating capabilities for area {area.name}: {e}"
            log.error(error_msg)
            span.update(
                level="ERROR",
                status_message=str(e),
                metadata={
                    "area_generation_error": error_msg,
                    "error": str(e),
                    "area_name": area.name,
                },
            )
            raise


async def generate_capabilities(
    cfg: DictConfig,
    areas_tag: str,
    langfuse_client: Langfuse,
    resume_tag: Optional[str] = None,
) -> None:
    """Generate capabilities using multi-agent debate system for each area."""
    domain_name = cfg.global_cfg.domain
    exp_id = cfg.exp_cfg.exp_id
    max_round = cfg.debate_cfg.max_round

    # Use resume_tag if provided, otherwise create new tag
    if resume_tag:
        capabilities_tag = resume_tag
        log.info(
            f"Resuming capability generation with existing tag: {capabilities_tag}"
        )
    else:
        capabilities_tag = f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    lf = langfuse_client
    if lf is None:
        lf = Langfuse()

    with langfuse_client.start_as_current_span(
        name=f"ace_capability_generation:{domain_name}:{exp_id}:{capabilities_tag}"
    ) as span:
        try:
            msg = f"Capabilities will be saved with tag: {capabilities_tag}"
            log.info(msg)
            span.update(
                metadata={
                    "generation_started": msg,
                    "capabilities_tag": capabilities_tag,
                    "domain": domain_name,
                    "exp_id": exp_id,
                }
            )

            msg = "Starting capability generation process"
            log.info(msg)
            span.update(metadata={"process_started": msg})

            span.update_trace(
                metadata={
                    "domain": domain_name,
                    "exp_id": exp_id,
                    "max_round": max_round,
                    "num_capabilities_per_area": cfg.capability_generation.num_capabilities_per_area,
                    "areas_tag": areas_tag,
                    "capabilities_tag": capabilities_tag,
                },
                tags=["capability_generation_process", exp_id],
            )

            areas_file = (
                Path.home()
                / cfg.global_cfg.output_dir
                / domain_name.replace(" ", "_")
                / exp_id
                / "areas"
                / areas_tag
                / "areas.json"
            )

            if not areas_file.exists():
                error_msg = f"Areas file not found: {areas_file}"
                log.error(error_msg)
                span.update(
                    level="ERROR",
                    status_message="Areas file not found",
                    metadata={
                        "file_not_found_error": error_msg,
                        "areas_file": str(areas_file),
                    },
                )
                raise FileNotFoundError(error_msg)

            with open(areas_file, "r", encoding="utf-8") as f:
                areas_data = json.load(f)

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
                error_msg = f"No valid areas found in {areas_file}"
                span.update(
                    level="ERROR",
                    status_message="No valid areas found",
                    metadata={
                        "no_areas_error": error_msg,
                        "areas_file": str(areas_file),
                    },
                )
                raise ValueError(error_msg)

            msg = (
                f"Found {len(areas)} areas to process: {[area.name for area in areas]}"
            )
            log.info(msg)
            span.update(
                metadata={
                    "areas_loaded": msg,
                    "num_areas": len(areas),
                    "area_names": [area.name for area in areas],
                }
            )

            output_dir = (
                Path.home()
                / cfg.global_cfg.output_dir
                / domain_name.replace(" ", "_")
                / exp_id
                / "capabilities"
                / capabilities_tag
            )

            msg = f"Output directory: {output_dir}"
            log.info(msg)
            span.update(
                metadata={
                    "output_directory_configured": msg,
                    "output_dir": str(output_dir),
                }
            )

            # Check for existing capabilities if resuming
            existing_capabilities = set()
            if resume_tag and output_dir.exists():
                for area_dir in output_dir.iterdir():
                    if area_dir.is_dir() and (area_dir / "capabilities.json").exists():
                        existing_capabilities.add(area_dir.name)

                if existing_capabilities:
                    msg = f"Found {len(existing_capabilities)} existing capabilities: {list(existing_capabilities)}"
                    log.info(msg)
                    span.update(metadata={"existing_capabilities": msg})
                else:
                    log.info("No existing capabilities found, will generate all areas")

            processed_areas = 0
            skipped_areas = 0

            for i, area in enumerate(areas):
                area_dir_name = area.name.replace(" ", "_")

                # Skip if capabilities already exist for this area
                if resume_tag and area_dir_name in existing_capabilities:
                    msg = f"Skipping area {i + 1}/{len(areas)}: {area.name} (already exists)"
                    log.info(msg)
                    span.update(
                        metadata={
                            f"area_{i + 1}_skipped": msg,
                            "skipped_area": area.name,
                            "progress": f"{i + 1}/{len(areas)}",
                        }
                    )
                    skipped_areas += 1
                    continue

                msg = f"Processing area {i + 1}/{len(areas)}: {area.name}"
                log.info(msg)
                span.update(
                    metadata={
                        f"area_{i + 1}_started": msg,
                        "current_area": area.name,
                        "progress": f"{i + 1}/{len(areas)}",
                    }
                )

                await generate_capabilities_for_area(cfg, area, output_dir, lf)

                msg = f"Completed area {i + 1}/{len(areas)}: {area.name}"
                log.info(msg)
                span.update(
                    metadata={
                        f"area_{i + 1}_completed": msg,
                        "completed_area": area.name,
                    }
                )

                processed_areas += 1
                await asyncio.sleep(1)

            if resume_tag:
                msg = f"Capability generation resumed with tag: {capabilities_tag} - Processed: {processed_areas}, Skipped: {skipped_areas}, Total: {len(areas)}"
            else:
                msg = f"Capabilities generated with tag: {capabilities_tag} - Processed: {processed_areas} areas"

            print(msg)
            span.update(
                metadata={
                    "generation_completed": msg,
                    "capabilities_tag": capabilities_tag,
                    "total_areas": len(areas),
                    "processed_areas": processed_areas,
                    "skipped_areas": skipped_areas,
                    "resumed": bool(resume_tag),
                }
            )

        except Exception as e:
            error_msg = f"Error in generate_capabilities: {e}"
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
                lf.flush()
            raise
