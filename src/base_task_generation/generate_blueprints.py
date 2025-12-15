"""Generate task blueprints for each combination."""

import asyncio
import logging

from autogen_core.models import ChatCompletionClient

from src.base_task_generation.diverse_task_constants import (
    BLOOMS_TAXONOMY,
    DIFFICULTY_LEVELS,
)
from src.base_task_generation.diverse_task_dataclasses import Blueprint, Combination
from src.base_task_generation.diverse_task_prompts import format_blueprint_prompt
from src.utils.model_client_utils import ModelCallMode, async_call_model


logger = logging.getLogger(__name__)


def generate_blueprints(
    capability,
    combinations: list[Combination],
    client: ChatCompletionClient,
) -> list[Blueprint]:
    """Generate task blueprints for each valid combination.

    Args:
        capability: Capability object
        combinations: List of Combination objects
        client: ChatCompletionClient for API calls

    Returns
    -------
        List of Blueprint objects
    """
    logger.info("Generating task blueprints...")

    blueprints = []

    for i, combo in enumerate(combinations):
        logger.info(
            f"Generating blueprint {i + 1}/{len(combinations)}: "
            f"{combo.content} | {combo.difficulty} | {combo.reasoning}"
        )

        system_prompt, user_prompt = format_blueprint_prompt(
            capability_name=capability.name,
            capability_description=capability.description,
            capability_domain=capability.area.domain.name,
            capability_area=capability.area.name,
            subtopic=combo.content,
            difficulty=combo.difficulty,
            difficulty_description=DIFFICULTY_LEVELS[combo.difficulty.lower()][
                "description"
            ],
            reasoning=combo.reasoning,
            reasoning_description=BLOOMS_TAXONOMY[combo.reasoning]["description"],
        )

        response = asyncio.run(
            async_call_model(
                client,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                mode=ModelCallMode.JSON_PARSE,
            )
        )

        if "blueprint" not in response:
            logger.error(
                f"Response missing 'blueprint' key. Response keys: {response.keys()}"
            )
            logger.error(f"Full response: {response}")
            raise ValueError(
                "Invalid blueprint response format: missing 'blueprint' key"
            )

        blueprint = Blueprint(
            combination_id=i,
            subtopic=combo.content,
            difficulty=combo.difficulty,
            reasoning=combo.reasoning,
            blueprint=response["blueprint"],
            rationale=combo.rationale,
        )
        blueprints.append(blueprint)

    logger.info(f"Generated {len(blueprints)} blueprints")

    return blueprints
