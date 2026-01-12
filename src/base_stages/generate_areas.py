"""Generate areas using the scientist LLM."""

import asyncio
import logging
from typing import List

from autogen_core.models import ChatCompletionClient

from src.schemas.area_schemas import Area
from src.schemas.domain_schemas import Domain
from src.utils.base_generation_prompts import (
    AREAS_GENERATION_RESPONSE_JSON_FORMAT,
    AREAS_GENERATION_USER_PROMPT,
)
from src.utils.model_client_utils import ModelCallMode, async_call_model


logger = logging.getLogger(__name__)


def generate_areas(
    domain: Domain,
    num_areas: int,
    num_capabilities_per_area: int,
    client: ChatCompletionClient,
) -> List[Area]:
    """Generate areas for the specified domain.

    Args:
        domain: Domain object
        num_areas: Number of areas to generate
        num_capabilities_per_area: Number of capabilities per area
        client: ChatCompletionClient for API calls

    Returns
    -------
        List of generated Area objects
    """
    logger.info(f"Generating {num_areas} areas ...")
    user_prompt = AREAS_GENERATION_USER_PROMPT.format(
        num_areas=num_areas,
        num_capabilities_per_area=num_capabilities_per_area,
        domain=domain.domain_name,
        response_json_format=AREAS_GENERATION_RESPONSE_JSON_FORMAT,
    )

    response = asyncio.run(
        async_call_model(
            client,
            system_prompt="",
            user_prompt=user_prompt,
            mode=ModelCallMode.JSON_PARSE,
        )
    )

    areas = []
    for idx, area_name in enumerate(response.get("areas", [])):
        area = Area(
            area_name=area_name,
            area_id=f"area_{idx:03d}",
            domain=domain,
            area_description="",
        )
        areas.append(area)

    logger.info(f"Generated {len(areas)} areas")

    return areas
