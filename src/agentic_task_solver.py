"""Multi-agent debate system for solving generated tasks."""

import asyncio
import logging
import os
import traceback
from pathlib import Path

import hydra
import openlit
from langfuse import Langfuse
from omegaconf import DictConfig, OmegaConf

from src.task_solving.generator import solve_tasks_with_debate, load_tasks_from_file


# Suppress OpenTelemetry console output
os.environ["OTEL_LOG_LEVEL"] = "ERROR"
os.environ["OTEL_METRICS_EXPORTER"] = "none"
os.environ["OTEL_PYTHON_LOG_CORRELATION"] = "false"
os.environ["OTEL_PYTHON_LOG_LEVEL"] = "ERROR"

log = logging.getLogger("agentic_task_solving")

lf = Langfuse()
openlit.init(tracer=lf._otel_tracer, disable_batch=True, disable_metrics=True)


@hydra.main(version_base=None, config_path="cfg", config_name="agentic_config")
def main(cfg: DictConfig) -> None:
    """Run the multi-agent debate-based task solving system."""
    domain_name = cfg.global_cfg.domain
    exp_id = cfg.exp_cfg.exp_id
    output_dir = cfg.global_cfg.output_dir
    max_tasks = cfg.task_solving.get("max_tasks", 0)

    with lf.start_as_current_span(
        name=f"ace_agentic_task_solving:{domain_name}:{exp_id}"
    ) as span:
        try:
            msg = "Starting multi-agent debate-based task solving"
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

            # Load tasks from the specified file or use pipeline tags to find them
            tasks_file = None
            if cfg.pipeline_tags.get("tasks_tag"):
                # Look for tasks file using the tag
                tasks_dir = Path(output_dir) / domain_name / "tasks"
                tasks_file = tasks_dir / f"tasks_{cfg.pipeline_tags.tasks_tag}.json"
            elif cfg.task_solving.get("input_file"):
                tasks_file = Path(cfg.task_solving.input_file)
            else:
                raise ValueError("Either pipeline_tags.tasks_tag or task_solving.input_file must be specified")

        if not tasks_file.exists():
            raise FileNotFoundError(f"Tasks file not found: {tasks_file}")

        log.info(f"Loading tasks from: {tasks_file}")
        tasks = load_tasks_from_file(tasks_file)
        log.info(f"Loaded {len(tasks)} tasks")

        # Limit number of tasks if specified
            if max_tasks > 0:
                tasks = tasks[:max_tasks]
            log.info(f"Limited to {len(tasks)} tasks")

        # Run task solving
            msg = f"Running task solving for {len(tasks)} tasks"
            log.info(msg)
            span.update(metadata={"task_solving_started": msg})

            results = asyncio.run(solve_tasks_with_debate(
            cfg=cfg,
            tasks=tasks,
                langfuse_client=lf,
            ))

        # Print summary
        consensus_count = sum(1 for result in results.values() if result.get("consensus_reached", False))
        no_consensus_count = len(results) - consensus_count

            msg = f"Task solving completed. Consensus: {consensus_count}, No consensus: {no_consensus_count}"
            log.info(msg)
            span.update(
                metadata={
                    "task_solving_completed": msg,
                    "total_tasks": len(results),
                    "consensus_reached": consensus_count,
                    "no_consensus": no_consensus_count,
                }
            )

        # Print detailed results if requested
            if cfg.task_solving.get("print_results", False):
            for task_id, result in results.items():
                log.info(f"\nTask {task_id}:")
                log.info(f"  Solution: {result['solution'][:100]}...")
                log.info(f"  Consensus: {result['consensus_reached']}")
                log.info(f"  Rounds: {result['total_rounds']}")

    except Exception as e:
            error_msg = f"Error in agentic task solving: {str(e)}"
            log.error(error_msg)
            log.error(traceback.format_exc())
            span.update(metadata={"error": error_msg})
        raise
    finally:
            lf.flush()


if __name__ == "__main__":
    main()
