"""Multi-agent task generation system for generating tasks for each capability."""

import asyncio
import logging
import os
import traceback

import hydra
import openlit
from langfuse import Langfuse
from omegaconf import DictConfig, OmegaConf

from src.task_generation import generate_tasks


# Suppress OpenTelemetry console output
os.environ["OTEL_LOG_LEVEL"] = "ERROR"
os.environ["OTEL_METRICS_EXPORTER"] = "none"
os.environ["OTEL_PYTHON_LOG_CORRELATION"] = "false"
os.environ["OTEL_PYTHON_LOG_LEVEL"] = "ERROR"

log = logging.getLogger("agentic_task_gen")

lf = Langfuse()
openlit.init(
    tracer=lf._otel_tracer, disable_batch=True, disable_metrics=True
)

@hydra.main(version_base=None, config_path="cfg", config_name="agentic_config")
def main(cfg: DictConfig) -> None:
    """Run the multi-agent task generation system."""
    capabilities_tag = cfg.pipeline_tags.capabilities_tag
    resume_tag = getattr(cfg.pipeline_tags, "resume_tasks_tag", None)
    domain_name = cfg.global_cfg.domain
    exp_id = cfg.exp_cfg.exp_id

    with lf.start_as_current_span(
        name=f"ace_agentic_task_generation:{domain_name}:{exp_id}"
    ) as span:
        try:
            msg = "Starting multi-agent task generation"
            log.info(msg)
            span.update(metadata={"system_started": msg})

            config_yaml = OmegaConf.to_yaml(cfg, resolve=True)
            msg = "Configuration loaded"
            log.info("Configuration:\n%s", config_yaml)
            span.update(
                metadata={
                    "configuration_loaded": msg,
                    "config": config_yaml,
                    "domain": domain_name,
                    "exp_id": exp_id,
                }
            )

            if capabilities_tag:
                msg = f"Using capabilities from tag: {capabilities_tag}"
                log.info(msg)
                span.update(
                    metadata={
                        "capabilities_tag_found": msg,
                        "capabilities_tag": capabilities_tag,
                    }
                )
            else:
                error_msg = "No capabilities_tag provided. Please provide pipeline_tags.capabilities_tag=<tag> to specify which capabilities to use."
                log.warning(error_msg)
                span.update(
                    level="ERROR",
                    status_message="Missing capabilities_tag",
                    metadata={"capabilities_tag_missing": error_msg},
                )
                return
                
            if resume_tag:
                msg = f"Resuming task generation from tag: {resume_tag}"
                log.info(msg)
                span.update(metadata={"resume_tag_found": msg, "resume_tag": resume_tag})
                

            span.update_trace(
                metadata={
                    "domain": domain_name,
                    "exp_id": exp_id,
                    "capabilities_tag": capabilities_tag,
                    "resume_tag": resume_tag,
                    "config": config_yaml,
                },
                tags=["agentic_task_generation", exp_id],
            )

            asyncio.run(generate_tasks(cfg, capabilities_tag, lf, resume_tag))

            msg = "Multi-agent task generation completed successfully"
            log.info(msg)
            span.update(metadata={"system_completed": msg})

        except Exception as e:
            error_msg = f"Task generation failed: {e}"
            traceback_msg = f"Full traceback: {traceback.format_exc()}"

            log.error(error_msg)
            log.error(traceback_msg)

            span.update(
                level="ERROR",
                status_message=str(e),
                metadata={
                    "system_error": error_msg,
                    "error": str(e),
                    "traceback": traceback_msg,
                },
            )

            raise


if __name__ == "__main__":
    main()
