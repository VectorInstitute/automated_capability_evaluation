"""Generate multiple-choice questions for each blueprint."""

import asyncio
import logging
from typing import List

from autogen_core.models import ChatCompletionClient

from src.base_stages.task_dataclasses import Blueprint
from src.schemas.task_schemas import Task
from src.utils.base_generation_prompts import (
    format_options_prompt,
    format_question_prompt,
)
from src.utils.model_client_utils import ModelCallMode, async_call_model


logger = logging.getLogger(__name__)


def generate_tasks_from_blueprints(
    capability,
    blueprints: list[Blueprint],
    client: ChatCompletionClient,
    tasks_per_blueprint: int = 3,
) -> List[Task]:
    """Generate multiple-choice questions for each blueprint.

    This function generates Task objects using a two-step process:
    1. Generate the question text
    2. Generate 4 options for the question

    The correct answer is NOT determined here — that happens in Stage 4
    (Solution Generation) where an LLM solves each task.

    Args:
        capability: Capability object
        blueprints: List of Blueprint objects
        client: ChatCompletionClient for API calls
        tasks_per_blueprint: Number of tasks to generate per blueprint

    Returns
    -------
        List of Task objects (questions + options, no answers)
    """
    logger.info("Generating tasks from blueprints...")

    all_tasks = []

    for blueprint in blueprints:
        logger.info(
            f"Generating {tasks_per_blueprint} tasks for blueprint "
            f"{blueprint.combination_id}: {blueprint.subtopic} | "
            f"{blueprint.difficulty} | {blueprint.reasoning}"
        )

        for _j in range(tasks_per_blueprint):
            task_id = f"task_{len(all_tasks):03d}"

            try:
                # Step 1: Generate the question
                logger.debug(f"  {task_id}: Generating question...")
                question_system, question_user = format_question_prompt(
                    capability_name=capability.name,
                    capability_description=capability.description,
                    capability_domain=capability.area.domain.name,
                    capability_area=capability.area.name,
                    blueprint_description=blueprint.blueprint,
                )

                question_response = asyncio.run(
                    async_call_model(
                        client,
                        system_prompt=question_system,
                        user_prompt=question_user,
                        mode=ModelCallMode.JSON_PARSE,
                    )
                )

                question_text = question_response["question"]
                logger.debug(f"  {task_id}: Question generated")

                # Step 2: Generate the options
                logger.debug(f"  {task_id}: Generating options...")
                options_system, options_user = format_options_prompt(
                    capability_name=capability.name,
                    capability_description=capability.description,
                    capability_domain=capability.area.domain.name,
                    capability_area=capability.area.name,
                    question=question_text,
                )

                options_response = asyncio.run(
                    async_call_model(
                        client,
                        system_prompt=options_system,
                        user_prompt=options_user,
                        mode=ModelCallMode.JSON_PARSE,
                    )
                )

                options = options_response["options"]
                logger.debug(f"  {task_id}: Options generated")

                # Combine question and options into task text
                task_text = f"{question_text}\n\n"
                for choice_key, choice_text in options.items():
                    task_text += f"{choice_key}. {choice_text}\n"

                choices_structured = [
                    {"label": label, "solution": text}
                    for label, text in options.items()
                ]

                task = Task(
                    task_id=task_id,
                    task=task_text,
                    task_type="multiple_choice",
                    solution_type="multiple_choice",
                    difficulty=blueprint.difficulty,
                    bloom_level=blueprint.reasoning,
                    choices=choices_structured,
                    capability=capability,
                    generation_metadata={
                        "method": "diverse_task_generation",
                        "blueprint_id": blueprint.combination_id,
                        "blueprint": blueprint.blueprint,
                        "subtopic": blueprint.subtopic,
                    },
                )
                all_tasks.append(task)

            except Exception as e:
                logger.error(f"  Failed to generate {task_id}: {e}")
                continue

        tasks_for_blueprint = [
            t
            for t in all_tasks
            if t.generation_metadata.get("blueprint_id") == blueprint.combination_id
        ]
        logger.info(f"  Generated {len(tasks_for_blueprint)} tasks for this blueprint")

    logger.info(f"Generated {len(all_tasks)} total tasks")

    return all_tasks
