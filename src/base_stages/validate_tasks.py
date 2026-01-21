"""Validate that generated tasks align with intended dimensions."""

import asyncio
import logging

from autogen_core.models import ChatCompletionClient

from src.schemas.solution_schemas import TaskSolution
from src.schemas.validation_schemas import ValidationResult
from src.utils.base_generation_prompts import format_verification_prompt
from src.utils.model_client_utils import ModelCallMode, async_call_model


logger = logging.getLogger(__name__)


def validate_tasks(
    task_solutions: list[TaskSolution],
    client: ChatCompletionClient,
) -> list[ValidationResult]:
    """Validate that generated tasks align with intended dimensions.

    Args:
        task_solutions: List of TaskSolution objects
        client: ChatCompletionClient for API calls

    Returns
    -------
        List of ValidationResult objects
    """
    logger.info("Validating task alignment...")

    validation_results = []

    for i, task_solution in enumerate(task_solutions):
        logger.info(
            f"Validating task {i + 1}/{len(task_solutions)}: {task_solution.task_id}"
        )
        capability = task_solution.task.capability

        try:
            task_obj = task_solution.task

            # Use structured fields from Task if available, fallback to metadata/parsing
            blueprint_text = "N/A"
            if task_obj.difficulty and task_obj.bloom_level:
                blueprint_text = (
                    f"Difficulty: {task_obj.difficulty}, "
                    f"Bloom's Level: {task_obj.bloom_level}"
                )
            elif task_solution.generation_metadata:
                blueprint_text = task_solution.generation_metadata.get(
                    "blueprint", "N/A"
                )

            # Extract question (first part before choices)
            task_lines = task_solution.task_text.strip().split("\n")
            question = task_lines[0] if task_lines else task_solution.task_text

            # Use structured choices if available, otherwise parse from text
            choices = {}
            if task_obj.choices:
                for choice in task_obj.choices:
                    choices[choice["label"]] = choice["solution"]
            else:
                # Fallback: parse from task text
                for task_line in task_lines[1:]:
                    line = task_line.strip()
                    if line and len(line) > 2 and line[1] == ".":
                        choice_letter = line[0]
                        choice_text = line[3:].strip()
                        choices[choice_letter] = choice_text

            system_prompt, user_prompt = format_verification_prompt(
                capability_domain=capability.area.domain.domain_name,
                capability_area=capability.area.area_name,
                capability_name=capability.capability_name,
                capability_description=capability.capability_description,
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

            overall_aligned = response.get("overall_verdict", "Fail") == "Pass"

            validation_result = ValidationResult(
                task_solution=task_solution,
                verification=overall_aligned,
                feedback=response.get("explanation", ""),
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

            status = "PASS" if overall_aligned else "FAIL"
            logger.info(f"  {status}")

        except Exception as e:
            logger.error(f"Error validating {task_solution.task_id}: {e}")
            logger.info("ERROR - Skipping this task")
            continue

    return validation_results
