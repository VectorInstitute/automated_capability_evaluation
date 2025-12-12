"""Generate capabilities using the scientist LLM."""

import asyncio
import logging
from typing import Any, Dict, List

import numpy as np
from autogen_core.models import ChatCompletionClient

from src.schemas.area_schemas import Area
from src.schemas.capability_schemas import Capability
from src.schemas.domain_schemas import Domain
from src.utils import prompts
from src.utils.model_client_utils import ModelCallMode, async_call_model


logger = logging.getLogger(__name__)


def generate_areas(
    domain: Domain,
    num_areas: int,
    num_capabilities_per_area: int,
    scientist_llm_client: ChatCompletionClient,
) -> List[Area]:
    """
    Generate areas for the specified domain.

    Args
    ----
        domain (str): The domain name.
        num_areas (int): The number of areas to generate.
        num_capabilities_per_area (int): The number of capabilities per area.
        scientist_llm (ChatCompletionClient): The scientist LLM client.

    Returns
    -------
        Dict[str, Any]: A dictionary containing the generated areas
        and metadata about the generation process.
    """
    logger.info(f"Generating {num_areas} areas ...")
    user_prompt = prompts.AREAS_GENERATION_USER_PROMPT.format(
        num_areas=num_areas,
        num_capabilities_per_area=num_capabilities_per_area,
        domain=domain.name,
        response_json_format=prompts.AREAS_GENERATION_RESPONSE_JSON_FORMAT,
    )

    # Use async_call_model with asyncio.run() for sync execution
    response = asyncio.run(
        async_call_model(
            scientist_llm_client,
            system_prompt="",
            user_prompt=user_prompt,
            mode=ModelCallMode.JSON_PARSE,
        )
    )

    areas = []
    for idx, area_name in enumerate(response.get("areas", [])):
        area = Area(
            name=area_name,
            area_id=f"area_{idx:03d}",
            domain=domain,
            description="",
        )
        areas.append(area)

    logger.info(f"Generated {len(areas)} areas")

    return areas


def generate_capabilities(
    area: Area,
    num_capabilities: int,
    num_capabilities_per_run: int,
    scientist_llm_client: ChatCompletionClient,
) -> List[Capability]:
    """
    Generate capabilities for a given area.

    Args
    ----
        area (Area): The area object containing domain information.
        num_capabilities (int): The number of capabilities to generate.
        num_capabilities_per_run (int): The number of capabilities to generate per run.
        scientist_llm (ChatCompletionClient): The scientist LLM client.
        scientist_llm_gen_cfg (Dict[str, Any]): The generation configuration
            for the scientist LLM.
        **kwargs (Any): Additional keyword arguments.

    Returns
    -------
        List[Capability]: The generated capabilities.
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
            scientist_llm_client=scientist_llm_client,
            prev_capabilities=capabilities,
        )
        capabilities.extend(run_capabilities)
        num_capabilities_left -= len(run_capabilities)

    return capabilities


def generate_capabilities_using_llm(
    area: Area,
    num_capabilities: int,
    scientist_llm_client: ChatCompletionClient,
    prev_capabilities: List[Capability],
) -> Dict[str, Any]:
    """
    Generate capabilities using the scientist LLM.

    Prompt the scientist LLM with instructions to generate initial capabilities.

    Args
    ----
        domain_name (str): The domain name.
        area_name (str): The area name.
        area (Area): The area object.
        num_capabilities (int): The number of capabilities to generate.
        scientist_llm (ChatCompletionClient): The scientist LLM client.
        scientist_llm_gen_cfg (Dict[str, Any]): The generation configuration
            for the scientist LLM.
        sys_prompt (str): The system prompt.
        user_prompt (str): The user prompt.
        prev_capabilities (List[Capability]): The list of previously
            generated capabilities.
        **kwargs (Any): Additional keyword arguments.

    Returns
    -------
        Dict[str, Any]: A dictionary containing the generated capabilities
        and metadata about the generation process.
    """
    sys_prompt = prompts.CAPABILITY_GENERATION_SYSTEM_PROMPT
    user_prompt = prompts.HIERARCHICAL_CAPABILITY_GENERATION_USER_PROMPT.format(
        area=area.name,
        domain=area.domain.name,
        num_capabilities=num_capabilities,
        prev_capabilities="\n".join([elm.name for elm in prev_capabilities]),
    )

    # Use async_call_model with asyncio.run() for sync execution
    # Retry logic is handled inside async_call_model
    response = asyncio.run(
        async_call_model(
            scientist_llm_client,
            system_prompt=sys_prompt,
            user_prompt=user_prompt,
            mode=ModelCallMode.JSON_PARSE,
        )
    )

    # Response is already a parsed dict from JSON_PARSE mode
    gen_capabilities_dict = response.get("capabilities", [])
    capabilities = []

    for idx, capability_dict in enumerate(gen_capabilities_dict):
        try:
            # Create capability object without saving to disk
            capability_id = f"cap_{idx:03d}"
            capability = Capability(
                name=capability_dict["name"],
                capability_id=capability_id,
                area=area,
                description=capability_dict["description"],
            )
        except Exception as e:
            logger.warning(
                f"Error creating capability object {capability_dict['name']}, hence skipping it: {e}"
            )
            # Skip this capability
            continue
        else:
            capabilities.append(capability)

    if len(capabilities) != len(gen_capabilities_dict):
        logger.warning(
            f"Only {len(capabilities)} capabilities were created out of {len(gen_capabilities_dict)} generated capabilities."
        )

    logger.info(f"Generated {len(capabilities)} capabilities.")

    return capabilities
