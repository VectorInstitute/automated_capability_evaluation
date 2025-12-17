"""Find valid (Content, Difficulty, Reasoning) combinations."""

import asyncio
import logging

from autogen_core.models import ChatCompletionClient

from src.base_stages.prompts import format_combination_prompt
from src.base_stages.task_constants import (
    BLOOMS_TAXONOMY,
    DIFFICULTY_LEVELS,
)
from src.base_stages.task_dataclasses import Combination, SubTopic
from src.utils.model_client_utils import ModelCallMode, async_call_model


logger = logging.getLogger(__name__)


def find_valid_combinations(
    capability, subtopics: list[SubTopic], client: ChatCompletionClient
) -> list[Combination]:
    """Find valid combinations of Content, Difficulty, and Reasoning.

    Args:
        capability: Capability object
        subtopics: List of SubTopic objects
        client: ChatCompletionClient for API calls

    Returns
    -------
        List of Combination objects
    """
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

    content_list = "\n".join(
        [
            f"{i + 1}. Content: {c['content']}, Difficulty: {c['difficulty']}, Reasoning: {c['reasoning']}"
            for i, c in enumerate(all_combinations)
        ]
    )

    system_prompt, user_prompt = format_combination_prompt(
        capability_name=capability.name,
        capability_description=capability.description,
        capability_domain=capability.area.domain.name,
        capability_area=capability.area.name,
        content_list=content_list,
    )

    response = asyncio.run(
        async_call_model(
            client,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            mode=ModelCallMode.JSON_PARSE,
        )
    )

    combinations_data = response.get("valid_combinations", [])

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
    for i, combo in enumerate(combinations[:5]):
        logger.info(
            f"  {i + 1}. {combo.content} | {combo.difficulty} | {combo.reasoning}"
        )
    if len(combinations) > 5:
        logger.info(f"  ... and {len(combinations) - 5} more")

    return combinations
