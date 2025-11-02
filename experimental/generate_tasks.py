"""Generate multiple-choice questions for each blueprint."""

import json
import logging
from typing import Callable

from diverse_task_dataclasses import Blueprint, Capability, Task
from diverse_task_prompts import format_task_prompt


logger = logging.getLogger(__name__)


def generate_tasks(
    capability: Capability,
    blueprints: list[Blueprint],
    call_llm: Callable,
    tasks_per_blueprint: int = 3,
) -> list[Task]:
    """Generate multiple-choice questions for each blueprint."""
    logger.info("Generating tasks from blueprints...")

    all_tasks = []

    for blueprint in blueprints:
        logger.info(
            f"Generating {tasks_per_blueprint} tasks for blueprint "
            f"{blueprint.combination_id}: {blueprint.subtopic} | "
            f"{blueprint.difficulty} | {blueprint.reasoning}"
        )

        # Generate multiple tasks for this blueprint
        for j in range(tasks_per_blueprint):
            system_prompt, user_prompt = format_task_prompt(
                capability_name=capability.name,
                capability_description=capability.description,
                capability_domain=capability.domain,
                capability_area=capability.area,
                blueprint_description=blueprint.blueprint,
            )

            response = call_llm(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_format={"type": "json_object"},
            )

            task_data = json.loads(response)

            # Create Task object
            task_id = f"task_{blueprint.combination_id}_{j}"
            task = Task(
                task_id=task_id,
                blueprint_id=blueprint.combination_id,
                subtopic=blueprint.subtopic,
                difficulty=blueprint.difficulty,
                reasoning=blueprint.reasoning,
                question=task_data["question"],
                choices=task_data["options"],
                correct_answer=task_data["correct_answer"],
            )
            all_tasks.append(task)

        logger.info(f"  Generated {tasks_per_blueprint} tasks")

    logger.info(f"Generated {len(all_tasks)} total tasks")

    return all_tasks
