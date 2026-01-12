"""Generate capabilities using the scientist LLM."""

import asyncio
import logging
from typing import List

import numpy as np
from autogen_core.models import ChatCompletionClient

from src.schemas.area_schemas import Area
from src.schemas.capability_schemas import Capability
from src.utils.base_generation_prompts import (
    CAPABILITY_GENERATION_SYSTEM_PROMPT,
    CAPABILITY_GENERATION_USER_PROMPT,
)
from src.utils.model_client_utils import ModelCallMode, async_call_model


logger = logging.getLogger(__name__)


def generate_capabilities(
    area: Area,
    num_capabilities: int,
    num_capabilities_per_run: int,
    client: ChatCompletionClient,
) -> List[Capability]:
    """Generate capabilities for a given area.

    Args:
        area: Area object
        num_capabilities: Total number of capabilities to generate
        num_capabilities_per_run: Number of capabilities per LLM call
        client: ChatCompletionClient for API calls

    Returns
    -------
        List of generated Capability objects
    """
    capabilities = []

    # Calculate number of runs needed
    num_runs = int(np.ceil(num_capabilities / num_capabilities_per_run))

    # Generate capabilities in batches
    num_capabilities_left = num_capabilities
    for run in range(num_runs):
        logger.info(f"Capability generation for area: {area.name} at run {run}")

        run_capabilities = generate_capabilities_using_llm(
            area=area,
            num_capabilities=min(num_capabilities_per_run, num_capabilities_left),
            client=client,
            prev_capabilities=capabilities,
            id_offset=len(capabilities),  # Pass offset for unique IDs
        )
        capabilities.extend(run_capabilities)
        num_capabilities_left -= len(run_capabilities)

    return capabilities


def generate_capabilities_using_llm(
    area: Area,
    num_capabilities: int,
    client: ChatCompletionClient,
    prev_capabilities: List[Capability],
    id_offset: int = 0,
) -> List[Capability]:
    """Generate capabilities using LLM.

    Args:
        area: Area object
        num_capabilities: Number of capabilities to generate
        client: ChatCompletionClient for API calls
        prev_capabilities: Previously generated capabilities
        id_offset: Offset for capability IDs to ensure uniqueness across batches

    Returns
    -------
        List of generated Capability objects
    """
    sys_prompt = CAPABILITY_GENERATION_SYSTEM_PROMPT
    user_prompt = CAPABILITY_GENERATION_USER_PROMPT.format(
        area=area.name,
        domain=area.domain.name,
        num_capabilities=num_capabilities,
        prev_capabilities="\n".join([elm.name for elm in prev_capabilities]),
    )

    response = asyncio.run(
        async_call_model(
            client,
            system_prompt=sys_prompt,
            user_prompt=user_prompt,
            mode=ModelCallMode.JSON_PARSE,
        )
    )

    gen_capabilities_dict = response.get("capabilities", [])
    capabilities = []

    for idx, capability_dict in enumerate(gen_capabilities_dict):
        try:
            capability_id = f"cap_{(idx + id_offset):03d}"
            capability = Capability(
                name=capability_dict["name"],
                capability_id=capability_id,
                area=area,
                description=capability_dict["description"],
            )
        except Exception as e:
            logger.warning(
                f"Error creating capability {capability_dict['name']}, skipping: {e}"
            )
            continue
        else:
            capabilities.append(capability)

    if len(capabilities) != len(gen_capabilities_dict):
        logger.warning(
            f"Only {len(capabilities)} capabilities were created out of "
            f"{len(gen_capabilities_dict)} generated capabilities."
        )

    # Truncate to requested number if LLM returned more
    if len(capabilities) > num_capabilities:
        logger.info(
            f"LLM returned {len(capabilities)} capabilities, "
            f"truncating to requested {num_capabilities}"
        )
        capabilities = capabilities[:num_capabilities]

    logger.info(f"Generated {len(capabilities)} capabilities.")

    return capabilities
