"""Main task solver orchestration function."""

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

from src.task_solver.messages import Task
from src.task_solver.moderator import TaskSolverModerator
from src.task_solver.scientist import TaskSolverScientist
from src.utils.model_client_utils import get_model_client


log = logging.getLogger("task_solver.generator")
logging.getLogger(ROOT_LOGGER_NAME).setLevel(logging.WARNING)
logging.getLogger(TRACE_LOGGER_NAME).setLevel(logging.WARNING)
logging.getLogger(EVENT_LOGGER_NAME).setLevel(logging.WARNING)


async def solve_task(
    cfg: DictConfig, task: Task, output_dir: Path, langfuse_client: Langfuse
) -> None:
    """Solve a task using multi-agent debate system."""
    max_rounds = cfg.task_solver.max_rounds
    task_id = task.task_id
    capability_name = task.capability_name
    area_name = task.area_name

    with langfuse_client.start_as_current_span(
        name=f"task_solver_for_task:{task_id}, capability:{capability_name}, area: {area_name}"
    ) as span:
        try:
            msg = f"Generating solutions for task: {task_id}, capability: {capability_name}, area: {area_name}"
            log.info(msg)
            span.update(
                metadata={
                    "single_task_solver_started": msg,
                    "task_id": task_id,
                    "problem": task.problem,
                    "capability_name": capability_name,
                    "area_name": area_name,
                }
            )

            runtime = SingleThreadedAgentRuntime()

            # Register moderator
            await TaskSolverModerator.register(
                runtime,
                "TaskSolverModerator",
                lambda: TaskSolverModerator(
                    model_client=get_model_client(
                        model_name=cfg.agents.moderator.model_name,
                        seed=cfg.agents.moderator.get("seed"),
                    ),
                    num_solvers=2,
                    max_rounds=max_rounds,
                    output_dir=output_dir,
                    langfuse_client=langfuse_client,
                ),
            )

            # Register scientist agents
            await TaskSolverScientist.register(
                runtime,
                "TaskSolverScientistA",
                lambda: TaskSolverScientist(
                    model_client=get_model_client(
                        model_name=cfg.agents.scientist_a.model_name,
                        seed=cfg.agents.scientist_a.get("seed"),
                    ),
                    scientist_id="A",
                    langfuse_client=langfuse_client,
                ),
            )

            await TaskSolverScientist.register(
                runtime,
                "TaskSolverScientistB",
                lambda: TaskSolverScientist(
                    model_client=get_model_client(
                        model_name=cfg.agents.scientist_b.model_name,
                        seed=cfg.agents.scientist_b.get("seed"),
                    ),
                    scientist_id="B",
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

            # Start runtime
            runtime.start()

            await runtime.publish_message(task, DefaultTopicId())

            msg = f"Task message published: {task_id}, capability: {capability_name}, area: {area_name}"
            log.info(msg)
            span.update(
                metadata={
                    "task_published": msg,
                    "task_id": task_id,
                    "capability_name": capability_name,
                    "area_name": area_name,
                }
            )

            try:
                await runtime.stop_when_idle()
                msg = f"Completed solving task: {task_id}, capability: {capability_name}, area: {area_name}"
                log.info(msg)
                span.update(metadata={"runtime_completed": msg})
            except Exception as e:
                msg = f"Error while solving task {task_id}, capability: {capability_name}, area: {area_name}: {e}"
                log.error(msg)
                span.update(
                    level="ERROR",
                    status_message=str(e),
                    metadata={
                        "runtime_error": msg,
                        "error": str(e),
                        "task_id": task_id,
                        "capability_name": capability_name,
                        "area_name": {area_name},
                    },
                )
                raise
        except Exception as e:
            error_msg = f"Error in task solver: {str(e)}"
            log.error(error_msg)
            log.error(traceback.format_exc())
            span.update(metadata={"error": error_msg})
            raise


async def solve_tasks(
    cfg: DictConfig,
    tasks_tag: str,
    langfuse_client: Langfuse,
    resume_tag: Optional[str] = None,
) -> None:
    """Solve tasks using multi-agent debate system."""
    domain_name = cfg.global_cfg.domain
    exp_id = cfg.exp_cfg.exp_id

    if resume_tag:
        solutions_tag = resume_tag
        log.info(f"Resuming task solver with existing tag: {solutions_tag}")
    else:
        solutions_tag = f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    output_dir = (
        Path.home()
        / cfg.global_cfg.output_dir
        / domain_name.replace(" ", "_")
        / exp_id
        / "task_solutions"
        / solutions_tag
    )

    with langfuse_client.start_as_current_span(
        name=f"ace_task_solver:{domain_name}:{exp_id}:{solutions_tag}"
    ) as span:
        try:
            msg = f"Solutions will be saved with tag: {solutions_tag}"
            print(msg)
            log.info(msg)
            span.update(
                metadata={
                    "solver_started": msg,
                    "solutions_tag": solutions_tag,
                    "resume_tag": resume_tag,
                    "output_dir": output_dir,
                    "tasks_tag": tasks_tag,
                    "domain": domain_name,
                    "exp_id": exp_id,
                },
                tags=["task_solver_process", exp_id],
            )

            tasks_dir = (
                Path.home()
                / cfg.global_cfg.output_dir
                / domain_name.replace(" ", "_")
                / exp_id
                / "tasks"
                / tasks_tag
            )

            if not tasks_dir.exists():
                error_msg = f"Tasks directory not found: {tasks_dir}"
                log.error(error_msg)
                span.update(
                    level="ERROR",
                    metadata={
                        "directory_not_found_error": error_msg,
                        "tasks_dir": str(tasks_dir),
                    },
                )
                raise FileNotFoundError(error_msg)

            for per_area_capability_dir in tasks_dir.iterdir():
                tasks_file = per_area_capability_dir / "tasks.json"

                if not tasks_file.exists():
                    msg = f"Tasks file not found: {tasks_file}"
                    log.error(msg)
                    span.update(metadata={"warning": msg})
                    continue

                with open(tasks_file, "r", encoding="utf-8") as f:
                    tasks = json.load(f)["tasks"]
                    output_solver_dir = Path(output_dir) / per_area_capability_dir.name

                    for task_id, task_data in tasks.items():
                        if (
                            output_solver_dir.exists()
                            and f"{task_id}_solution.json"
                            in list(output_solver_dir.iterdir())
                        ):
                            msg = f"Task {task_id} already solved"
                            log.info(msg)
                            span.update(metadata={"task_solver_skipped": msg})
                            continue

                        task = Task(
                            task_id=task_id,
                            problem=task_data["task"],
                            capability_name=task_data["capability_id"],
                            area_name=task_data["area_id"],
                        )
                        await solve_task(cfg, task, output_solver_dir, langfuse_client)

        except Exception as e:
            error_msg = f"Error in task solver: {str(e)}"
            log.error(error_msg)
            log.error(f"Traceback: {traceback.format_exc()}")
            span.update(metadata={"error": error_msg})
            raise
