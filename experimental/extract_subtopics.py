"""Extract sub-topics for a capability."""

import json
import logging
from typing import Callable

from diverse_task_dataclasses import Capability, SubTopic
from diverse_task_prompts import format_subtopic_prompt


logger = logging.getLogger(__name__)


def extract_subtopics(capability: Capability, call_llm: Callable) -> list[SubTopic]:
    """Extract sub-topics for the given capability."""
    logger.info("Extracting sub-topics...")

    system_prompt, user_prompt = format_subtopic_prompt(
        capability_name=capability.name,
        capability_description=capability.description,
        capability_domain=capability.domain,
        capability_area=capability.area,
    )

    response = call_llm(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        response_format={"type": "json_object"},
    )

    result = json.loads(response)
    subtopic_names = result.get("sub_topics", [])

    # Create SubTopic objects
    subtopics = [SubTopic(name=name) for name in subtopic_names]

    logger.info(f"Extracted {len(subtopics)} sub-topics:")
    for st in subtopics:
        logger.info(f"  - {st.name}")

    return subtopics
