"""Find valid combinations of (Content, Difficulty, Reasoning)."""

import json
import logging

from diverse_task_dataclasses import Capability, Combination, SubTopic
from diverse_task_prompts import format_combination_prompt


logger = logging.getLogger(__name__)


def find_valid_combinations(
    capability: Capability, subtopics: list[SubTopic], call_llm
) -> list[Combination]:
    """Find valid combinations of for the capability."""
    logger.info("Finding valid combinations...")

    # Prepare subtopics description
    subtopics_desc = "\n".join([f"- {st.name}" for st in subtopics])

    system_prompt, user_prompt = format_combination_prompt(
        capability_name=capability.name,
        capability_description=capability.description,
        capability_domain=capability.domain,
        capability_area=capability.area,
        subtopics_desc=subtopics_desc,
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

    logger.info(f"Found {len(combinations)} valid combinations:")
    for i, combo in enumerate(combinations[:5]):  # Show first 5
        logger.info(
            f"  {i + 1}. {combo.content} | {combo.difficulty} | {combo.reasoning}"
        )
    if len(combinations) > 5:
        logger.info(f"  ... and {len(combinations) - 5} more")

    return combinations
