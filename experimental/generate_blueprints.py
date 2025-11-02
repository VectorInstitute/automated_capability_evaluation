"""Generate task blueprints for each valid combination."""

import json
import logging
from typing import Callable

from diverse_task_dataclasses import Blueprint, Capability, Combination
from diverse_task_prompts import format_blueprint_prompt


logger = logging.getLogger(__name__)


def generate_blueprints(
    capability: Capability,
    combinations: list[Combination],
    call_llm: Callable,
    config: dict,
) -> list[Blueprint]:
    """Generate task blueprints for each valid combination."""
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
            capability_domain=capability.domain,
            capability_area=capability.area,
            subtopic=combo.content,
            difficulty=combo.difficulty,
            difficulty_description=config["difficulty_levels"][
                combo.difficulty.lower()
            ]["description"],
            reasoning=combo.reasoning,
            reasoning_description=config["blooms_taxonomy"][combo.reasoning][
                "description"
            ],
        )

        response = call_llm(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_format={"type": "json_object"},
        )

        blueprint_data = json.loads(response)
        blueprint = Blueprint(
            combination_id=i,
            subtopic=combo.content,
            difficulty=combo.difficulty,
            reasoning=combo.reasoning,
            blueprint=blueprint_data["blueprint"],
            rationale=combo.rationale,
        )
        blueprints.append(blueprint)

    logger.info(f"Generated {len(blueprints)} blueprints")

    return blueprints
