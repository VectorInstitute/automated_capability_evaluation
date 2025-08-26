"""Main task generation orchestration functions."""

import asyncio
import json
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
from autogen_ext.models.openai import OpenAIChatCompletionClient
from langfuse import Langfuse
from omegaconf import DictConfig

from src.task_generation.messages import Capability
from src.task_generation.moderator import TaskModerator
from src.task_generation.scientist import TaskScientist


log = logging.getLogger("agentic_task_gen.generator")
logging.getLogger(ROOT_LOGGER_NAME).setLevel(logging.WARNING)
logging.getLogger(TRACE_LOGGER_NAME).setLevel(logging.WARNING)
logging.getLogger(EVENT_LOGGER_NAME).setLevel(logging.WARNING)


async def generate_tasks_for_capability(
    cfg: DictConfig, capability: Capability, output_dir: Path, langfuse_client: Langfuse
) -> None:
    """Generate tasks for a single capability."""
    with langfuse_client.start_as_current_span(
        name=f"task_generation_for_capability:{capability.name}"
    ) as span:
        try:
            msg = f"Generating tasks for capability: {capability.name}"
            log.info(msg)
            span.update(
                metadata={
                    "capability_generation_started": msg,
                    "capability_name": capability.name,
                    "capability_description": capability.description,
                }
            )

            domain_name = cfg.global_cfg.domain

            runtime = SingleThreadedAgentRuntime()

            # Register scientists
            await TaskScientist.register(
                runtime,
                "TaskScientistA",
                lambda: TaskScientist(
                    model_client=OpenAIChatCompletionClient(
                        model=cfg.agents.scientist_a.model_name,
                        seed=cfg.agents.scientist_a.seed,
                    ),
                    scientist_id="A",
                    domain=domain_name,
                    langfuse_client=langfuse_client,
                ),
            )

            await TaskScientist.register(
                runtime,
                "TaskScientistB",
                lambda: TaskScientist(
                    model_client=OpenAIChatCompletionClient(
                        model=cfg.agents.scientist_b.model_name,
                        seed=cfg.agents.scientist_b.seed,
                    ),
                    scientist_id="B",
                    domain=domain_name,
                    langfuse_client=langfuse_client,
                ),
            )

            # Register moderator
            await TaskModerator.register(
                runtime,
                "TaskModerator",
                lambda: TaskModerator(
                    model_client=OpenAIChatCompletionClient(
                        model=cfg.agents.moderator.model_name,
                        seed=cfg.agents.moderator.seed,
                    ),
                    num_scientists=2,
                    num_final_problems=cfg.task_generation.num_final_problems_per_capability,
                    buffer_param=cfg.task_generation.buffer_param,
                    agreement_threshold=cfg.task_generation.agreement_threshold,
                    output_dir=output_dir,
                    domain=domain_name,
                    langfuse_client=langfuse_client,
                ),
            )

            span.update(
                metadata={
                    "agents_registered": "All task agents registered successfully",
                    "scientists": ["A", "B"],
                    "moderator": True,
                }
            )

            # Start runtime and process the capability
            runtime.start()
            await runtime.publish_message(capability, DefaultTopicId())

            msg = f"Capability message published: {capability.name}"
            log.info(msg)
            span.update(
                metadata={
                    "capability_published": msg,
                    "capability_name": capability.name,
                }
            )

            # Wait for the runtime to stop when idle
            try:
                await runtime.stop_when_idle()

                msg = f"Completed generating tasks for capability: {capability.name}"
                log.info(msg)
                span.update(metadata={"runtime_completed": msg})
            except Exception as e:
                msg = f"Error while generating tasks for capability {capability.name}: {e}"
                log.error(msg)
                span.update(
                    level="ERROR",
                    status_message=str(e),
                    metadata={
                        "runtime_error": msg,
                        "error": str(e),
                        "capability_name": capability.name,
                    },
                )
                raise

        except Exception as e:
            error_msg = f"Error in generating tasks for {capability.name}: {e}"
            traceback_msg = f"Traceback: {traceback.format_exc()}"

            log.error(error_msg)
            log.error(traceback_msg)

            span.update(
                level="ERROR",
                status_message=str(e),
                metadata={
                    "capability_generation_error": error_msg,
                    "error": str(e),
                    "traceback": traceback_msg,
                },
            )
            raise


async def generate_tasks(
    cfg: DictConfig, capabilities_tag: str, langfuse_client: Langfuse
) -> None:
    """Generate tasks for all capabilities."""
    domain_name = cfg.global_cfg.domain
    exp_id = cfg.exp_cfg.exp_id
    tasks_tag = f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    with langfuse_client.start_as_current_span(
        name=f"ace_task_generation:{domain_name}:{exp_id}:{tasks_tag}"
    ) as span:
        try:
            msg = f"Tasks will be saved with tag: {tasks_tag}"
            log.info(msg)
            span.update(
                metadata={
                    "generation_started": msg,
                    "tasks_tag": tasks_tag,
                    "domain": domain_name,
                    "exp_id": exp_id,
                }
            )

            msg = "Starting task generation process"
            log.info(msg)
            span.update(metadata={"process_started": msg})

            span.update_trace(
                metadata={
                    "domain": domain_name,
                    "exp_id": exp_id,
                    "tasks_tag": tasks_tag,
                    "capabilities_tag": capabilities_tag,
                    "num_problems_per_capability": cfg.task_generation.num_final_problems_per_capability,
                },
                tags=["task_generation_process", exp_id],
            )

            # Read capabilities from the timestamped capabilities directory
            capabilities_dir = (
                Path.home()
                / cfg.global_cfg.output_dir
                / domain_name.replace(" ", "_")
                / exp_id
                / "capabilities"
                / capabilities_tag
            )

            if not capabilities_dir.exists():
                error_msg = f"Capabilities directory not found: {capabilities_dir}"
                log.error(error_msg)
                span.update(
                    level="ERROR",
                    status_message="Capabilities directory not found",
                    metadata={
                        "directory_not_found_error": error_msg,
                        "capabilities_dir": str(capabilities_dir),
                    },
                )
                raise FileNotFoundError(error_msg)

            capabilities = []

            # Iterate through area directories
            for area_dir in capabilities_dir.iterdir():
                if area_dir.is_dir():
                    capabilities_file = area_dir / "capabilities.json"
                    if capabilities_file.exists():
                        with open(capabilities_file, "r", encoding="utf-8") as f:
                            capabilities_data = json.load(f)

                        if (
                            isinstance(capabilities_data, dict)
                            and "capabilities" in capabilities_data
                        ):
                            for cap_dict in capabilities_data["capabilities"]:
                                if (
                                    isinstance(cap_dict, dict)
                                    and "name" in cap_dict
                                    and "description" in cap_dict
                                ):
                                    capabilities.append(
                                        Capability(
                                            name=cap_dict["name"],
                                            description=cap_dict["description"],
                                            domain=cap_dict.get("domain", domain_name),
                                            area=cap_dict.get("area", area_dir.name),
                                        )
                                    )

            if not capabilities:
                error_msg = f"No valid capabilities found in {capabilities_dir}"
                span.update(
                    level="ERROR",
                    status_message="No valid capabilities found",
                    metadata={
                        "no_capabilities_error": error_msg,
                        "capabilities_dir": str(capabilities_dir),
                    },
                )
                raise ValueError(error_msg)

            msg = f"Found {len(capabilities)} capabilities to process"
            log.info(msg)
            span.update(
                metadata={
                    "capabilities_loaded": msg,
                    "num_capabilities": len(capabilities),
                    "capability_names": [cap.name for cap in capabilities],
                }
            )

            # Create timestamped output directory for tasks
            output_dir = (
                Path.home()
                / cfg.global_cfg.output_dir
                / domain_name.replace(" ", "_")
                / exp_id
                / "tasks"
                / tasks_tag
            )

            msg = f"Output directory: {output_dir}"
            log.info(msg)
            span.update(
                metadata={
                    "output_directory_configured": msg,
                    "output_dir": str(output_dir),
                }
            )

            # Print the timestamp for future reference
            print(f"Tasks generated with tag: {tasks_tag}")

            # Process each capability individually
            for i, capability in enumerate(capabilities):
                msg = f"Processing capability {i + 1}/{len(capabilities)}: {capability.name}"
                log.info(msg)
                span.update(
                    metadata={
                        f"capability_{i + 1}_started": msg,
                        "current_capability": capability.name,
                        "progress": f"{i + 1}/{len(capabilities)}",
                    }
                )

                await generate_tasks_for_capability(
                    cfg, capability, output_dir, langfuse_client
                )

                msg = f"Completed capability {i + 1}/{len(capabilities)}: {capability.name}"
                log.info(msg)
                span.update(
                    metadata={
                        f"capability_{i + 1}_completed": msg,
                        "completed_capability": capability.name,
                    }
                )

                await asyncio.sleep(1)

        except Exception as e:
            error_msg = f"Error in generate_tasks: {e}"
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

            raise
