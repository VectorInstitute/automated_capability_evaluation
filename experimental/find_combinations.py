"""Find valid combinations of (Content, Difficulty, Reasoning)."""

import json
import logging

from diverse_task_constants import BLOOMS_TAXONOMY, DIFFICULTY_LEVELS
from diverse_task_dataclasses import Capability, Combination, SubTopic
from diverse_task_prompts import format_combination_prompt


logger = logging.getLogger(__name__)


def find_valid_combinations(
    capability: Capability, subtopics: list[SubTopic], call_llm, config: dict
) -> list[Combination]:
    """Find valid combinations of Content, Difficulty, and Reasoning."""
    logger.info("Finding valid combinations...")

    # Get difficulty levels and reasoning types from constants
    difficulty_levels = list(DIFFICULTY_LEVELS.keys())
    reasoning_types = list(BLOOMS_TAXONOMY.keys())

    # Generate all possible combinations
    all_combinations = []
    for subtopic in subtopics:
        for difficulty in difficulty_levels:
            for reasoning in reasoning_types:
                all_combinations.append(
                    {
                        "content": subtopic.name,
                        "difficulty": difficulty,
                        "reasoning": reasoning,
                    }
                )

    logger.info(f"Generated {len(all_combinations)} total combinations to validate")

    # Format combinations as a numbered list for the LLM
    content_list = "\n".join(
        [
            f"{i + 1}. Content: {c['content']}, Difficulty: {c['difficulty']}, Reasoning: {c['reasoning']}"
            for i, c in enumerate(all_combinations)
        ]
    )

    system_prompt, user_prompt = format_combination_prompt(
        capability_name=capability.name,
        capability_description=capability.description,
        capability_domain=capability.domain,
        capability_area=capability.area,
        content_list=content_list,
    )

    response = call_llm(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        response_format={"type": "json_object"},
    )

    result = json.loads(response)
    combinations_data = result.get("valid_combinations", [])

    # Create Combination objects
    combinations = [
        Combination(
            content=combo["content"],
            difficulty=combo["difficulty"],
            reasoning=combo["reasoning"],
        )
        for combo in combinations_data
    ]

    logger.info(
        f"Found {len(combinations)} valid combinations out of {len(all_combinations)} total:"
    )
    for i, combo in enumerate(combinations[:5]):  # Show first 5
        logger.info(
            f"  {i + 1}. {combo.content} | {combo.difficulty} | {combo.reasoning}"
        )
    if len(combinations) > 5:
        logger.info(f"  ... and {len(combinations) - 5} more")

    return combinations
