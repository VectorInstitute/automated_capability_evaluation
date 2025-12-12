"""Generate multiple-choice questions for each blueprint."""

import asyncio
import logging

from autogen_core.models import ChatCompletionClient

from src.base_task_generation.diverse_task_dataclasses import Blueprint
from src.base_task_generation.diverse_task_prompts import format_task_prompt


logger = logging.getLogger(__name__)


def generate_tasks(
    capability,
    blueprints: list[Blueprint],
    client: ChatCompletionClient,
    tasks_per_blueprint: int = 3,
):
    """Generate multiple-choice questions for each blueprint.

    Args:
        capability: Schema Capability object with area.domain.name, area.name, etc.
        blueprints: List of Blueprint objects
        client: ChatCompletionClient for API calls
        tasks_per_blueprint: Number of tasks to generate per blueprint

    Returns
    -------
        List of schema TaskSolution objects
    """
    logger.info("Generating tasks from blueprints...")

    # Import here to avoid circular dependency
    from src.schemas.solution_schemas import TaskSolution
    from src.schemas.task_schemas import Task
    from src.utils.model_client_utils import ModelCallMode, async_call_model

    all_task_solutions = []

    for blueprint in blueprints:
        logger.info(
            f"Generating {tasks_per_blueprint} tasks for blueprint "
            f"{blueprint.combination_id}: {blueprint.subtopic} | "
            f"{blueprint.difficulty} | {blueprint.reasoning}"
        )

        # Generate multiple tasks for this blueprint
        for _j in range(tasks_per_blueprint):
            task_id = f"task_{len(all_task_solutions):03d}"

            try:
                system_prompt, user_prompt = format_task_prompt(
                    capability_name=capability.name,
                    capability_description=capability.description,
                    capability_domain=capability.area.domain.name,
                    capability_area=capability.area.name,
                    blueprint_description=blueprint.blueprint,
                )

                response = asyncio.run(
                    async_call_model(
                        client,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        mode=ModelCallMode.JSON_PARSE,
                    )
                )

                # Format the task question with multiple choice options
                task_text = f"{response['question']}\n\n"
                for choice_key, choice_text in response["options"].items():
                    task_text += f"{choice_key}. {choice_text}\n"

                # Create metadata dict
                generation_metadata = {
                    "method": "diverse_task_generation",
                    "blueprint_id": blueprint.combination_id,
                    "subtopic": blueprint.subtopic,
                    "difficulty": blueprint.difficulty,
                    "reasoning": blueprint.reasoning,
                    "correct_answer": response["correct_answer"],
                    "explanation": response.get("explanation", ""),
                    "alignment_notes": response.get("alignment_notes", ""),
                }

                # Create schema Task object (without generation_metadata)
                task = Task(
                    task_id=task_id,
                    task=task_text,
                    capability=capability,
                )

                # Create TaskSolution object with the task and metadata
                task_solution = TaskSolution(
                    task_id=task_id,
                    task=task_text,
                    solution=response["correct_answer"],
                    reasoning=response.get("explanation", ""),
                    task_obj=task,
                    generation_metadata=generation_metadata,
                )
                all_task_solutions.append(task_solution)

            except Exception as e:
                logger.error(f"  Failed to generate {task_id}: {e}")
                # Skip failed tasks and continue
                continue

        logger.info(
            f"  Generated {len([ts for ts in all_task_solutions if ts.generation_metadata.get('blueprint_id') == blueprint.combination_id])} task solutions for this blueprint"
        )

    logger.info(f"Generated {len(all_task_solutions)} total task solutions")

    return all_task_solutions
