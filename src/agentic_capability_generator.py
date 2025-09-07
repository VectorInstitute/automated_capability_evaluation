"""Multi-agent debate system for generating capabilities for each area."""

import asyncio
import logging
import os
import traceback
from typing import Optional

import hydra
import openlit
from langfuse import Langfuse
from omegaconf import DictConfig, OmegaConf

from src.capability_generation.generator import generate_capabilities


# Suppress OpenTelemetry console output
os.environ["OTEL_LOG_LEVEL"] = "ERROR"
os.environ["OTEL_METRICS_EXPORTER"] = "none"
os.environ["OTEL_PYTHON_LOG_CORRELATION"] = "false"
os.environ["OTEL_PYTHON_LOG_LEVEL"] = "ERROR"

log = logging.getLogger("agentic_cap_gen")

lf = Langfuse()
openlit.init(tracer=lf._otel_tracer, disable_batch=True, disable_metrics=True)


@hydra.main(version_base=None, config_path="cfg", config_name="agentic_config")
def main(cfg: DictConfig) -> None:
    """Run the multi-agent debate-based capability generation system."""
    areas_tag = cfg.pipeline_tags.areas_tag
    resume_tag: Optional[str] = getattr(
        cfg.pipeline_tags, "resume_capabilities_tag", None
    )
    domain_name = cfg.global_cfg.domain
    exp_id = cfg.exp_cfg.exp_id
    num_capabilities_per_area = cfg.capability_generation.num_capabilities_per_area

    with lf.start_as_current_span(
        name=f"ace_agentic_capability_generation:{domain_name}:{exp_id}"
    ) as span:
        try:
            msg = "Starting multi-agent debate-based capability generation"
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
                    "num_capabilities_per_area": num_capabilities_per_area,
                }
            )

            if areas_tag:
                msg = f"Using areas from tag: {areas_tag}"
                log.info(msg)
                span.update(metadata={"areas_tag_found": msg, "areas_tag": areas_tag})
            else:
                error_msg = "No areas_tag provided. Please provide pipeline_tags.areas_tag=<tag> to specify which areas to use."
                log.warning(error_msg)
                span.update(
                    level="ERROR",
                    status_message="Missing areas_tag",
                    metadata={"areas_tag_missing": error_msg},
                )
                return

            if resume_tag:
                msg = f"Resuming capability generation from tag: {resume_tag}"
                log.info(msg)
                span.update(
                    metadata={"resume_tag_found": msg, "resume_tag": resume_tag}
                )

            span.update_trace(
                metadata={
                    "domain": domain_name,
                    "exp_id": exp_id,
                    "areas_tag": areas_tag,
                    "resume_tag": resume_tag,
                    "num_capabilities_per_area": num_capabilities_per_area,
                    "config": config_yaml,
                },
                tags=["agentic_capability_generation", exp_id],
            )

            asyncio.run(generate_capabilities(cfg, areas_tag, lf, resume_tag))

            msg = (
                "Multi-agent debate-based capability generation completed successfully"
            )
            log.info(msg)
            span.update(metadata={"system_completed": msg})

        except Exception as e:
            error_msg = f"Capability generation failed: {e}"
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

            lf.flush()
            raise


if __name__ == "__main__":
    main()
