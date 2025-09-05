"""Main task solving orchestration function."""

import json
import logging
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from autogen_core import (
    EVENT_LOGGER_NAME,
    ROOT_LOGGER_NAME,
    TRACE_LOGGER_NAME,
    DefaultTopicId,
    SingleThreadedAgentRuntime,
)
from langfuse import Langfuse
from omegaconf import DictConfig

from src.task_solving.messages import Task
from src.task_solving.moderator import TaskSolvingModerator
from src.task_solving.scientist import TaskSolvingScientist
from src.utils.model_client_utils import get_model_client


log = logging.getLogger("task_solving.generator")
logging.getLogger(ROOT_LOGGER_NAME).setLevel(logging.WARNING)
logging.getLogger(TRACE_LOGGER_NAME).setLevel(logging.WARNING)
logging.getLogger(EVENT_LOGGER_NAME).setLevel(logging.WARNING)


async def solve_tasks_with_debate(
    cfg: DictConfig, 
    tasks: List[Dict], 
    langfuse_client: Langfuse = None
) -> Dict[str, Dict]:
    """
    Solve tasks using multi-agent debate system.
    
    Args:
        cfg: Configuration containing debate and model settings
        tasks: List of tasks to solve, each containing task_id, task content, and capability_id
        langfuse_client: Langfuse client for tracing
        
    Returns:
        Dictionary mapping task_id to final solution data
    """
    domain_name = cfg.global_cfg.domain
    exp_id = cfg.exp_cfg.exp_id
    max_rounds = cfg.debate_cfg.max_round
    num_solvers = 2  # scientist_a and scientist_b
    solutions_tag = f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    with langfuse_client.start_as_current_span(
        name=f"ace_task_solving:{domain_name}:{exp_id}:{solutions_tag}"
    ) as span:
        try:
            msg = f"Solutions will be saved with tag: {solutions_tag}"
            log.info(msg)
            span.update(
                metadata={
                    "solving_started": msg,
                    "solutions_tag": solutions_tag,
                    "domain": domain_name,
                    "exp_id": exp_id,
                    "num_tasks": len(tasks),
                    "num_solvers": num_solvers,
                    "max_rounds": max_rounds,
                }
            )

            # Create output directory
            output_dir = Path(cfg.global_cfg.output_dir) / "task_solutions" / f"{domain_name}_{exp_id}{solutions_tag}"
            output_dir.mkdir(parents=True, exist_ok=True)

            # Set up runtime
            runtime = SingleThreadedAgentRuntime()

            # Create model clients for each agent
            scientist_a_client = get_model_client(
                cfg.agents.scientist_a.model_name, 
                seed=cfg.agents.scientist_a.get("seed")
            )
            scientist_b_client = get_model_client(
                cfg.agents.scientist_b.model_name, 
                seed=cfg.agents.scientist_b.get("seed")
            )
            moderator_client = get_model_client(
                cfg.agents.moderator.model_name, 
                seed=cfg.agents.moderator.get("seed")
            )

            # Register moderator
            moderator_agent_type = await TaskSolvingModerator.register(
                runtime,
                "task_solving_moderator",
                lambda: TaskSolvingModerator(
                    model_client=moderator_client,
                    num_solvers=num_solvers,
                    max_rounds=max_rounds,
                    output_dir=output_dir,
                    langfuse_client=langfuse_client,
                ),
            )

            # Register scientist agents
            scientist_a_type = await TaskSolvingScientist.register(
                runtime,
                "task_scientist_a",
                lambda: TaskSolvingScientist(
                    model_client=scientist_a_client,
                    scientist_id="scientist_a",
                    langfuse_client=langfuse_client,
                ),
            )
            
            scientist_b_type = await TaskSolvingScientist.register(
                runtime,
                "task_scientist_b", 
                lambda: TaskSolvingScientist(
                    model_client=scientist_b_client,
                    scientist_id="scientist_b",
                    langfuse_client=langfuse_client,
                ),
            )

            # Start runtime
            runtime.start()

            log.info(f"Starting task solving for {len(tasks)} tasks with {num_solvers} scientists")

            # Process each task
            for i, (task_id, task_data) in enumerate(tasks.items()):
                # Handle both old and new task formats
                if isinstance(task_data, dict) and "task" in task_data:
                    # New format: {"task": "problem text", "capability_id": "cap_name"}
                    capability_id = task_data.get("capability_id", "unknown")
                    task_content = task_data
                else:
                    # Old format or other formats
                    capability_id = task_data.get("capability_id", "unknown") if isinstance(task_data, dict) else "unknown"
                    task_content = {"task": str(task_data)} if not isinstance(task_data, dict) else task_data
                
                # Create task message
                task = Task(
                    task_id=task_id,
                    task_content=task_content,
                    capability_id=capability_id,
                )

                # Send task to moderator
                await runtime.publish_message(
                    task, 
                    topic_id=DefaultTopicId()
                )

                log.info(f"Submitted task {task_id} for solving")

            # Wait for all tasks to complete
            # Note: In a real implementation, you might want to add a timeout
            # and check for completion status
            await runtime.stop_when_idle()

            # Collect results
            results = {}
            for solution_file in output_dir.glob("task_*_solution.json"):
                try:
                    with open(solution_file, "r") as f:
                        solution_data = json.load(f)
                        results[solution_data["task_id"]] = solution_data
                except Exception as e:
                    log.error(f"Error loading solution from {solution_file}: {e}")

            log.info(f"Task solving completed. Processed {len(results)} tasks.")
            
            span.update(
                metadata={
                    "solving_completed": f"Processed {len(results)} tasks",
                    "output_dir": str(output_dir),
                    "results_count": len(results),
                }
            )

            return results

        except Exception as e:
            error_msg = f"Error in task solving: {str(e)}"
            log.error(error_msg)
            log.error(traceback.format_exc())
            span.update(metadata={"error": error_msg})
            raise


def load_tasks_from_file(tasks_file: Path) -> List[Dict]:
    """
    Load tasks from a JSON file.
    
    Args:
        tasks_file: Path to the tasks file
        
    Returns:
        List of task dictionaries
    """
    try:
        with open(tasks_file, "r") as f:
            tasks_data = json.load(f)
        
        # Handle different task file formats
        if isinstance(tasks_data, list):
            # Old format: list of tasks
            return {f"task_{i+1}": task for i, task in enumerate(tasks_data)}
        elif isinstance(tasks_data, dict):
            # If it's a dict, try to extract tasks
            if "tasks" in tasks_data:
                # New format: {"tasks": {"task_1": {...}, "task_2": {...}}}
                return tasks_data["tasks"]
            else:
                # Convert dict to single task
                return {"task_1": tasks_data}
        else:
            raise ValueError(f"Unexpected task file format: {type(tasks_data)}")
            
    except Exception as e:
        log.error(f"Error loading tasks from {tasks_file}: {e}")
        raise 