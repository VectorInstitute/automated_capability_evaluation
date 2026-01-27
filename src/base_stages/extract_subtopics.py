"""Extract sub-topics for a capability."""

import asyncio
import logging

from autogen_core.models import ChatCompletionClient

from src.base_stages.task_dataclasses import SubTopic
from src.schemas.capability_schemas import Capability
from src.utils.base_generation_prompts import format_subtopic_prompt
from src.utils.model_client_utils import ModelCallMode, async_call_model


logger = logging.getLogger(__name__)


def extract_subtopics(
    capability: Capability,
    client: ChatCompletionClient,
    min_subtopics: int = 3,
    max_subtopics: int = 8,
) -> list[SubTopic]:
    """Extract sub-topics for the given capability.

    Args:
        capability: Capability object
        client: ChatCompletionClient for API calls
        min_subtopics: Minimum number of subtopics to generate
        max_subtopics: Maximum number of subtopics to generate

    Returns
    -------
        List of SubTopic objects
    """
    logger.info(f"Extracting sub-topics (range: {min_subtopics}-{max_subtopics}) ...")

    system_prompt, user_prompt = format_subtopic_prompt(
        capability_name=capability.capability_name,
        capability_description=capability.capability_description,
        capability_domain=capability.area.domain.domain_name,
        capability_area=capability.area.area_name,
        min_subtopics=min_subtopics,
        max_subtopics=max_subtopics,
    )

    response = asyncio.run(
        async_call_model(
            client,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            mode=ModelCallMode.JSON_PARSE,
        )
    )

    subtopic_names = response.get("sub_topics", [])

    subtopics = [SubTopic(name=name) for name in subtopic_names]

    logger.info(f"Extracted {len(subtopics)} sub-topics:")
    for st in subtopics:
        logger.info(f"  - {st.name}")

    return subtopics
