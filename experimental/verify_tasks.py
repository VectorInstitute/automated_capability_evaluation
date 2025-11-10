"""Verify that generated tasks align with intended dimensions."""

import json
import logging
from typing import Callable

from diverse_task_dataclasses import Blueprint, Capability, Task, VerificationResult
from diverse_task_prompts import format_verification_prompt


logger = logging.getLogger(__name__)


def verify_tasks(
    capability: Capability,
    tasks: list[Task],
    blueprints: list[Blueprint],
    call_llm: Callable,
) -> VerificationResult:
    """Verify that generated tasks align with intended dimensions."""
    logger.info("Verifying task alignment...")

    # Create blueprint lookup
    blueprint_dict = {bp.combination_id: bp for bp in blueprints}

    verification_results = []

    for i, task in enumerate(tasks):
        logger.info(f"Verifying task {i + 1}/{len(tasks)}: {task.task_id}")

        try:
            # Skip verification for tasks that failed generation
            if task.question.startswith("ERROR:"):
                logger.warning("  Skipping verification (task generation failed)")
                verification = VerificationResult(
                    task_id=task.task_id,
                    subtopic_aligned=False,
                    difficulty_aligned=False,
                    reasoning_aligned=False,
                    choices_appropriate=False,
                    overall_aligned=False,
                    feedback="Task generation failed - verification skipped",
                )
                verification_results.append(verification)
                logger.info("  ✗ SKIPPED")
                continue

            # Get blueprint for this task
            blueprint = blueprint_dict.get(task.blueprint_id)
            blueprint_text = blueprint.blueprint if blueprint else "N/A"

            system_prompt, user_prompt = format_verification_prompt(
                capability_domain=capability.domain,
                capability_area=capability.area,
                capability_name=capability.name,
                capability_description=capability.description,
                task_blueprint=blueprint_text,
                question=task.question,
                option_a=task.choices.get("A", ""),
                option_b=task.choices.get("B", ""),
                option_c=task.choices.get("C", ""),
                option_d=task.choices.get("D", ""),
                correct_answer=task.correct_answer,
            )

            response = call_llm(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_format={"type": "json_object"},
            )

            verification_data = json.loads(response)

            # Map new verification format to old format
            overall_aligned = verification_data.get("overall_verdict", "Fail") == "Pass"

            verification = VerificationResult(
                task_id=task.task_id,
                subtopic_aligned=verification_data.get("blueprint_alignment", "No")
                == "Yes",
                difficulty_aligned=verification_data.get(
                    "difficulty_reasoning_match", "No"
                )
                == "Yes",
                reasoning_aligned=verification_data.get("capability_alignment", "No")
                == "Yes",
                choices_appropriate=verification_data.get("single_correct_answer", "No")
                == "Yes",
                overall_aligned=overall_aligned,
                feedback=verification_data.get("explanation", ""),
            )
            verification_results.append(verification)

            status = "✓ PASS" if verification.overall_aligned else "✗ FAIL"
            logger.info(f"  {status}")

        except Exception as e:
            logger.error(f"  Failed to verify {task.task_id}: {e}")
            # Create a verification result with error information
            verification = VerificationResult(
                task_id=task.task_id,
                subtopic_aligned=False,
                difficulty_aligned=False,
                reasoning_aligned=False,
                choices_appropriate=False,
                overall_aligned=False,
                feedback=f"Verification failed: {str(e)}",
            )
            verification_results.append(verification)
            logger.info("  ✗ ERROR")

    # Calculate statistics
    total = len(verification_results)
    passed = sum(1 for v in verification_results if v.overall_aligned)
    failed = total - passed

    # Convert to dict for JSON serialization
    verification_details_dict = [
        {
            "task_id": v.task_id,
            "subtopic_aligned": v.subtopic_aligned,
            "difficulty_aligned": v.difficulty_aligned,
            "reasoning_aligned": v.reasoning_aligned,
            "choices_appropriate": v.choices_appropriate,
            "overall_aligned": v.overall_aligned,
            "feedback": v.feedback,
            "suggested_improvements": v.suggested_improvements,
        }
        for v in verification_results
    ]

    summary = {
        "total_tasks": total,
        "passed": passed,
        "failed": failed,
        "pass_rate": passed / total if total > 0 else 0,
        "verification_details": verification_details_dict,
    }

    logger.info("\nVerification Summary:")
    logger.info(f"  Total tasks: {total}")
    logger.info(f"  Passed: {passed}")
    logger.info(f"  Failed: {failed}")
    logger.info(f"  Pass rate: {summary['pass_rate']:.1%}")

    return summary
