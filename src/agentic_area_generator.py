"""Multi-agent debate system for generating capability areas."""

import asyncio
import logging
import os
import traceback

import hydra
from langfuse import Langfuse
from omegaconf import DictConfig, OmegaConf

from .area_generation import generate_areas


# Suppress OpenTelemetry console output
os.environ["OTEL_LOG_LEVEL"] = "ERROR"
os.environ["OTEL_METRICS_EXPORTER"] = "none"
os.environ["OTEL_PYTHON_LOG_CORRELATION"] = "false"
os.environ["OTEL_PYTHON_LOG_LEVEL"] = "ERROR"

log = logging.getLogger("agentic_area_gen")

lf = Langfuse()


@hydra.main(version_base=None, config_path="cfg", config_name="agentic_config")
def main(cfg: DictConfig) -> None:
    """Run the multi-agent debate-based area generation system."""
    domain_name = cfg.capabilities_cfg.domain
    exp_id = cfg.exp_cfg.exp_id

    with lf.start_as_current_span(
        name=f"ace_agentic_area_generation:{domain_name}:{exp_id}"
    ) as span:
        try:
            msg = "Starting multi-agent debate-based area generation"
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

            span.update_trace(
                metadata={
                    "domain": domain_name,
                    "exp_id": exp_id,
                    "config": config_yaml,
                },
                tags=["agentic_area_generation", exp_id],
            )

            asyncio.run(generate_areas(cfg, lf))

            msg = "Multi-agent debate-based area generation completed successfully"
            log.info(msg)
            span.update(metadata={"system_completed": msg})

        except Exception as e:
            error_msg = f"Area generation failed: {e}"
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
