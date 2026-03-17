"""Multi-agent debate system for solving generated tasks."""

import asyncio
import logging
import os
import traceback

import hydra
import openlit
from langfuse import Langfuse
from omegaconf import DictConfig, OmegaConf

from src.task_solver import solve_tasks


# Suppress OpenTelemetry console output
os.environ["OTEL_LOG_LEVEL"] = "ERROR"
os.environ["OTEL_METRICS_EXPORTER"] = "none"
os.environ["OTEL_PYTHON_LOG_CORRELATION"] = "false"
os.environ["OTEL_PYTHON_LOG_LEVEL"] = "ERROR"

log = logging.getLogger("agentic_task_solver")

langfuse_client = Langfuse()
openlit.init(
    tracer=langfuse_client._otel_tracer, disable_batch=True, disable_metrics=True
)


@hydra.main(version_base=None, config_path="cfg", config_name="agentic_config")
def main(cfg: DictConfig) -> None:
    """Run the multi-agent debate-based task solving system."""
    tasks_tag = cfg.pipeline_tags.get("tasks_tag")
    resume_tag = getattr(cfg.pipeline_tags, "resume_solutions_tag", None)
    domain_name = cfg.global_cfg.domain
    exp_id = cfg.exp_cfg.exp_id

    with langfuse_client.start_as_current_span(
        name=f"ace_agentic_task_solver:{domain_name}:{exp_id}"
    ) as span:
        try:
            msg = "Starting multi-agent debate-based task solver"
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

            if tasks_tag:
                msg = f"Using tasks from tag: {tasks_tag}"
                log.info(msg)
                span.update(
                    metadata={
                        "tasks_tag_found": msg,
                        "tasks_tag": tasks_tag,
                    }
                )
            else:
                error_msg = "No tasks_tag provided. Please provide pipeline_tags.tasks_tag=<tag> to specify which tasks to solve."
                log.warning(error_msg)
                span.update(
                    level="ERROR",
                    status_message="Missing tasks_tag",
                    metadata={"tasks_tag_missing": error_msg},
                )
                return

            if resume_tag:
                msg = f"Resuming task solving from tag: {resume_tag}"
                log.info(msg)
                span.update(
                    metadata={"resume_tag_found": msg, "resume_tag": resume_tag}
                )

            span.update_trace(
                metadata={
                    "domain": domain_name,
                    "exp_id": exp_id,
                    "tasks_tag": tasks_tag,
                    "resume_tag": resume_tag,
                    "config": config_yaml,
                },
                tags=["agentic_task_solver", exp_id],
            )

            asyncio.run(solve_tasks(cfg, tasks_tag, langfuse_client, resume_tag))

            msg = "Multi-agent debate-based task solving completed successfully"
            log.info(msg)
            span.update(metadata={"system_completed": msg})

        except Exception as e:
            error_msg = f"Task solving failed: {e}"
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

        finally:
            langfuse_client.flush()


if __name__ == "__main__":
    main()
