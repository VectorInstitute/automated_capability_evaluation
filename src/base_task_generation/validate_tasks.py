"""Validate that generated tasks align with intended dimensions."""

import asyncio
import logging

from autogen_core.models import ChatCompletionClient

from src.base_task_generation.diverse_task_prompts import format_verification_prompt
from src.schemas.solution_schemas import TaskSolution
from src.schemas.task_schemas import Task
from src.schemas.validation_schemas import ValidationResult
from src.utils.model_client_utils import ModelCallMode, async_call_model


logger = logging.getLogger(__name__)


def validate_tasks(
    task_solutions: list[TaskSolution],
    client: ChatCompletionClient,
) -> list[ValidationResult]:
    """Validate that generated tasks align with intended dimensions.

    Args:
        task_solutions: List of schema TaskSolution objects (non-empty)
        client: ChatCompletionClient for API calls

    Returns
    -------
        List of ValidationResult objects for validated tasks
    """
    logger.info("Validateing task alignment...")

    validation_results = []

    for i, task_solution in enumerate(task_solutions):
        logger.info(
            f"Validateing task {i + 1}/{len(task_solutions)}: {task_solution.task_id}"
        )
        capability = task_solution.task_obj.capability

        try:
            # Get blueprint info from generation_metadata
            blueprint_info = task_solution.generation_metadata or {}
            blueprint_text = blueprint_info.get("blueprint", "N/A")

            # Parse the task to extract question and choices
            task_lines = task_solution.task.strip().split("\n")
            question = task_lines[0] if task_lines else ""

            # Extract choices (A, B, C, D)
            choices = {}
            for task_line in task_lines[1:]:
                line = task_line.strip()
                if line and len(line) > 2 and line[1] == ".":
                    choice_letter = line[0]
                    choice_text = line[3:].strip()
                    choices[choice_letter] = choice_text

            system_prompt, user_prompt = format_verification_prompt(
                capability_domain=capability.area.domain.name,
                capability_area=capability.area.name,
                capability_name=capability.name,
                capability_description=capability.description,
                task_blueprint=blueprint_text,
                question=question,
                option_a=choices.get("A", ""),
                option_b=choices.get("B", ""),
                option_c=choices.get("C", ""),
                option_d=choices.get("D", ""),
                correct_answer=task_solution.solution,
            )

            response = asyncio.run(
                async_call_model(
                    client,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    mode=ModelCallMode.JSON_PARSE,
                )
            )

            # Map verification response to result format
            overall_aligned = response.get("overall_verdict", "Fail") == "Pass"

            # Create Task object
            task = Task(
                task_id=task_solution.task_id,
                task=task_solution.task,
                capability=capability,
            )

            # Create ValidationResult schema object
            validation_result = ValidationResult(
                task_id=task_solution.task_id,
                task=task_solution.task,
                verification=overall_aligned,
                feedback=response.get("explanation", ""),
                task_obj=task,
                generation_metadata={
                    "method": "validate_tasks",
                    "subtopic_aligned": response.get("blueprint_alignment", "No")
                    == "Yes",
                    "difficulty_aligned": response.get(
                        "difficulty_reasoning_match", "No"
                    )
                    == "Yes",
                    "reasoning_aligned": response.get("capability_alignment", "No")
                    == "Yes",
                    "choices_appropriate": response.get("single_correct_answer", "No")
                    == "Yes",
                    "suggested_improvements": response.get(
                        "suggested_improvements", ""
                    ),
                    **task_solution.generation_metadata,
                },
            )
            validation_results.append(validation_result)

            status = "✓ PASS" if overall_aligned else "✗ FAIL"
            logger.info(f"  {status}")

        except Exception as e:
            logger.error(f"  Failed to validate {task_solution.task_id}: {e}")
            logger.info("  ✗ ERROR - Skipping this task")
            # Skip tasks that fail verification
            continue

    return validation_results
