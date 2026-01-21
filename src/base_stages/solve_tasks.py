"""Solve tasks and generate TaskSolution objects (Stage 4).

This module takes Task objects (questions with options) and determines
the correct answer by having an LLM solve each task, creating TaskSolution
objects with the solution and reasoning.
"""

import asyncio
import logging
from typing import List, Optional

from autogen_core.models import ChatCompletionClient

from src.schemas.solution_schemas import TaskSolution
from src.schemas.task_schemas import Task
from src.utils.base_generation_prompts import format_solution_prompt
from src.utils.model_client_utils import ModelCallMode, async_call_model


logger = logging.getLogger(__name__)


def solve_tasks(
    tasks: List[Task],
    client: ChatCompletionClient,
) -> List[TaskSolution]:
    """Solve tasks and generate TaskSolution objects.

    For each task, this function has an LLM solve the multiple-choice
    question and creates a TaskSolution with the solution and reasoning.

    The solver determines the correct answer by actually solving the problem,
    not by looking up any pre-stored answer.

    Args:
        tasks: List of Task objects to solve
        client: ChatCompletionClient for API calls

    Returns
    -------
        List of TaskSolution objects
    """
    logger.info(f"Solving {len(tasks)} tasks...")

    task_solutions = []

    for i, task in enumerate(tasks):
        logger.info(f"Solving task {i + 1}/{len(tasks)}: {task.task_id}")

        try:
            capability = task.capability
            task_gen_metadata = task.generation_metadata or {}

            system_prompt, user_prompt = format_solution_prompt(
                capability_domain=capability.area.domain.domain_name,
                capability_area=capability.area.area_name,
                capability_name=capability.capability_name,
                capability_description=capability.capability_description,
                task_text=task.task,
            )

            response = asyncio.run(
                async_call_model(
                    client,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    mode=ModelCallMode.JSON_PARSE,
                )
            )

            solution = response.get("answer", "")
            reasoning = response.get("reasoning", "")

            generation_metadata = {
                "method": "solve_tasks",
                # Include original task generation metadata
                "task_generation_metadata": task_gen_metadata,
            }

            task_solution = TaskSolution(
                task=task,
                solution=solution,
                reasoning=reasoning,
                generation_metadata=generation_metadata,
            )
            task_solutions.append(task_solution)

            logger.info(f"  Solved: answer={solution}")

        except Exception as e:
            logger.error(f"  Failed to solve {task.task_id}: {e}")
            continue

    logger.info(f"Solved {len(task_solutions)}/{len(tasks)} tasks")

    return task_solutions


def solve_single_task(
    task: Task,
    client: ChatCompletionClient,
) -> Optional[TaskSolution]:
    """Solve a single task and return a TaskSolution.

    Args:
        task: Task object to solve
        client: ChatCompletionClient for API calls

    Returns
    -------
        TaskSolution object if successful, None otherwise
    """
    result = solve_tasks(tasks=[task], client=client)
    return result[0] if result else None
