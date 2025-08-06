"""Generate capabilities using the scientist LLM."""

import json
import logging
import os
import shutil
from typing import Any, Dict, List, Optional, Union

import numpy as np
from langsmith import tracing_context
from tenacity import Retrying, stop_after_attempt

from src.capability import Capability
from src.model import Model
from src.utils import constants, prompts
from src.utils.capability_management_utils import (
    _sample_seed_capabilities,
    get_previous_capabilities,
)
from src.utils.capability_utils import extract_and_parse_response


logger = logging.getLogger(__name__)


def generate_capability_areas(
    domain: str,
    num_areas: int,
    num_capabilities_per_area: int,
    scientist_llm: Model,
    user_prompt: str,
    scientist_llm_gen_cfg: Dict[str, Any],
    sys_prompt: Union[str, None] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Generate capability areas for the specified domain.

    Args
    ----
        domain (str): The domain name.
        num_areas (int): The number of capability areas to generate.
        num_capabilities_per_area (int): The number of capabilities per area.
        scientist_llm (Model): The scientist LLM model.
        user_prompt (str): The user prompt for generating capability areas.
        scientist_llm_gen_cfg (Dict[str, Any]): The generation configuration
            for the scientist LLM.
        sys_prompt (str | None): The system prompt for the scientist LLM.
        **kwargs (Any): Additional keyword arguments.

    Returns
    -------
        Dict[str, Any]: A dictionary containing the generated capability areas
        and metadata about the generation process.
    """
    logger.info(f"Generating {num_areas} capability areas ...")
    # Generate output using the model with specified generation arguments
    user_prompt = user_prompt.format(
        num_areas=num_areas,
        num_capabilities_per_area=num_capabilities_per_area,
        domain=domain,
        response_json_format=prompts.CAPABILITY_AREAS_GENERATION_RESPONSE_JSON_FORMAT,
    )
    with tracing_context(
        enabled=True,
        tags=["generate_capability_areas"],
        metadata={
            "ls_provider": scientist_llm.model_provider,
            "ls_model_name": scientist_llm.get_model_name(with_provider=False),
            "ls_model_type": "chat",
            "exp_id": kwargs.get("run_id"),
            "domain": domain,
            "num_areas": num_areas,
            **{f"ls_{k}": v for k, v in scientist_llm_gen_cfg.items()},
        },
    ):
        response, metadata = scientist_llm.generate(
            sys_prompt=sys_prompt if sys_prompt else "",
            user_prompt=user_prompt,
            generation_config=scientist_llm_gen_cfg,
        )

    parsed_response = extract_and_parse_response(response, has_thought=False)
    capability_areas = parsed_response["parsed_response"]

    logger.info(
        f"Capability areas generation tokens summary:\n{json.dumps(metadata, indent=4)}"
    )

    if len(capability_areas) > num_areas:
        logger.warning(
            f"Generated {len(capability_areas)} capability areas, but only {num_areas} are needed. "
            + f"Keeping the first {num_areas} areas."
        )
        capability_areas = capability_areas[:num_areas]

    logger.info(f"Generated capability areas:\n{capability_areas}")

    return {
        "capability_areas": capability_areas,
        "metadata": {
            "model": scientist_llm.get_model_name(),
            "api_metadata": metadata,
        },
    }


def generate_capabilities(
    domain: str,
    num_capabilities: int,
    num_capabilities_per_run: int,
    base_capability_dir: str,
    scientist_llm: Model,
    num_seed_capabilities: int,
    scientist_llm_gen_cfg: Dict[str, Any],
    method: str = "flat",
    include_seed_capability_names: Optional[List[str]] = None,
    exclude_seed_capability_names: Optional[List[str]] = None,
    **kwargs: Any,
) -> List[Capability]:
    """
    Generate initial capabilities for the specified domain.

    Args
    ----
        domain (str): The domain name.
        num_capabilities (int): The number of capabilities to generate.
        num_capabilities_per_run (int): The number of capabilities to generate per run.
        base_capability_dir (str): The base directory to store
            the generated capabilities for the specified domain.
        scientist_llm (Model): The scientist LLM model.
        num_seed_capabilities (int): The number of seed capabilities to use.
        scientist_llm_gen_cfg (Dict[str, Any]): The generation configuration
            for the scientist LLM.
        method (str): The method to use for generating capabilities.
            Choose from "flat" or "hierarchical".
        include_seed_capability_names (List[str] | None): A list of seed capability
            names to include in the generation process.
        exclude_seed_capability_names (List[str] | None): A list of seed capability
            names to exclude from the generation process.
        **kwargs (Any): Additional keyword arguments.

    Returns
    -------
        List[Capability]: The generated capabilities.
    """
    gen_capabilities = []
    run_metadata = []

    if method == "hierarchical":
        assert "num_capability_areas" in kwargs, (
            "`num_capability_areas` should be specified for hierarchical generation."
        )
        num_capability_areas = kwargs["num_capability_areas"]
        assert num_capabilities >= num_capability_areas, (
            "Number of capabilities should be greater than or equal to the number of capability areas, "
            + "so that each area can have at least one capability."
        )
        # Uniformly distribute num_capabilities across num_capability_areas
        num_capabilities_per_area = [
            num_capabilities // num_capability_areas
        ] * num_capability_areas
        for i in range(num_capabilities % num_capability_areas):
            num_capabilities_per_area[i] += 1
        num_runs = [
            int(np.ceil(num / num_capabilities_per_run))
            for num in num_capabilities_per_area
        ]

        # Generate capability areas for the specified domain
        response = generate_capability_areas(
            domain=domain,
            num_areas=kwargs["num_capability_areas"],
            num_capabilities_per_area=num_capabilities_per_area[0],
            scientist_llm=scientist_llm,
            user_prompt=prompts.HIERARCHICAL_CAPABILITY_AREAS_GENERATION_USER_PROMPT,
            scientist_llm_gen_cfg=scientist_llm_gen_cfg,
            **kwargs,
        )
        capability_areas = response["capability_areas"]
        # Select only the specified number of capability areas
        # even if more are generated
        capability_areas = capability_areas[:num_capability_areas]
    else:
        num_capabilities_per_area = [num_capabilities]
        num_runs = [int(np.ceil(num_capabilities / num_capabilities_per_run))]
        # No capability areas for flat generation, use the domain as the area
        capability_areas = [domain]

    for idx, capability_area in enumerate(capability_areas):
        if method == "hierarchical":
            logger.info(f"Generating capabilities for area: {capability_area}")
            # Fetch previously generated capabilities, if any
            prev_capabilities = get_previous_capabilities(
                capability_dir=base_capability_dir, capability_area=capability_area
            )
            user_prompt = prompts.HIERARCHICAL_CAPABILITY_GENERATION_USER_PROMPT.format(
                capability_area=capability_area,
            )
        else:
            prev_capabilities = get_previous_capabilities(
                capability_dir=base_capability_dir
            )
            user_prompt = prompts.CAPABILITY_GENERATION_USER_PROMPT

        # Add all seed capabilities to the list of prev_capabilities
        seed_capability_dir = os.path.join(
            constants.BASE_ARTIFACTS_DIR, "seed_capabilities", domain
        )
        prev_capabilities.extend(
            _sample_seed_capabilities(
                seed_capability_dir=seed_capability_dir,
                num_seed_capabilities=-1,
                random_seed=int(kwargs.get("seed", constants.DEFAULT_RANDOM_SEED)),
            )
        )

        num_capabilities_left = num_capabilities_per_area[idx]
        for run_id in range(num_runs[idx]):
            logger.info(f"Run ID: {run_id}")
            # Generate capabilities using the scientist LLM

            response = generate_capabilities_using_llm(
                domain=domain,
                num_capabilities=min(
                    num_capabilities_per_run,
                    num_capabilities_left,
                ),
                scientist_llm=scientist_llm,
                sys_prompt=prompts.CAPABILITY_GENERATION_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                num_seed_capabilities=num_seed_capabilities,
                seed_capability_dir=seed_capability_dir,
                prev_capabilities=prev_capabilities,
                scientist_llm_gen_cfg=scientist_llm_gen_cfg,
                base_capability_dir=base_capability_dir,
                include_seed_capability_names=include_seed_capability_names,
                exclude_seed_capability_names=exclude_seed_capability_names,
                capability_area=capability_area if method == "hierarchical" else None,
                local_run_id=run_id,
                **kwargs,
            )
            gen_capabilities.extend(response["capabilities"])
            num_capabilities_left -= len(response["capabilities"])
            run_metadata.append(response["metadata"])

            # Update the list of previously generated capabilities
            prev_capabilities.extend(response["capabilities"])

    # Analyze tokens metadata for capability generation
    total_input_tokens = sum([m["api_metadata"]["input_tokens"] for m in run_metadata])
    total_output_tokens = sum(
        [m["api_metadata"]["output_tokens"] for m in run_metadata]
    )
    tokens_summary = {
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_input_tokens + total_output_tokens,
        "input_tokens_per_run": int(total_input_tokens / sum(num_runs)),
        "output_tokens_per_run": int(total_output_tokens / sum(num_runs)),
        "total_tokens_per_run": int(
            (total_input_tokens + total_output_tokens) / sum(num_runs)
        ),
        "input_tokens_per_capability": int(total_input_tokens / len(gen_capabilities)),
        "output_tokens_per_capability": int(
            total_output_tokens / len(gen_capabilities)
        ),
        "total_tokens_per_capability": int(
            (total_input_tokens + total_output_tokens) / len(gen_capabilities)
        ),
    }
    logger.info(
        f"Capability generation tokens summary:\n{json.dumps(tokens_summary, indent=4)}"
    )

    return gen_capabilities


def generate_capabilities_using_llm(
    domain: str,
    num_capabilities: int,
    scientist_llm: Model,
    sys_prompt: str,
    user_prompt: str,
    num_seed_capabilities: int,
    seed_capability_dir: str,
    prev_capabilities: List[Capability],
    scientist_llm_gen_cfg: Dict[str, Any],
    base_capability_dir: str,
    include_seed_capability_names: Optional[List[str]] = None,
    exclude_seed_capability_names: Optional[List[str]] = None,
    capability_area: Union[str, None] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Generate capabilities using the scientist LLM.

    Prompt the scientist LLM with instructions and
    seed capabilities for the specified domain
    to generate initial capabilities.

    Args
    ----
        domain (str): The domain name.
        num_capabilities (int): The number of capabilities to generate.
        scientist_llm (Model): The scientist LLM model name.
        sys_prompt (str): The system prompt.
        user_prompt (str): The user prompt.
        num_seed_capabilities (int): The number of seed capabilities to use.
        seed_capability_dir (str): The directory containing the seed capabilities.
        prev_capabilities (List[Capability]): The list of previously
            generated capabilities.
        scientist_llm_gen_cfg (Dict[str, Any]): The generation configuration
            for the scientist LLM.
        base_capability_dir (str): The base directory to store
            the generated capabilities for the specified domain.
        include_seed_capability_names (List[str] | None): A list of seed capability
            names to include in the generation process.
        exclude_seed_capability_names (List[str] | None): A list of seed capability
            names to exclude from the generation process.
        capability_area (str | None): The capability area for the generation
        **kwargs (Any): Additional keyword arguments.

    Returns
    -------
        Dict[str, Any]: A dictionary containing the generated capabilities
        and metadata about the generation process.
    """
    # Sample seed capabilities for the generation process
    seed_capabilities = _sample_seed_capabilities(
        seed_capability_dir=seed_capability_dir,
        num_seed_capabilities=num_seed_capabilities,
        include_capability_names=include_seed_capability_names,
        exclude_capability_names=exclude_seed_capability_names,
        random_seed=int(kwargs.get("seed", constants.DEFAULT_RANDOM_SEED)),
    )
    # Get capability JSON strings (without scores)
    seed_capabilities_repr = [
        capability.to_json_str() for capability in seed_capabilities
    ]

    # LLM input
    user_prompt = user_prompt.format(
        sample_capability_json="\n".join(seed_capabilities_repr),
        prev_capabilities="\n".join([elm.name for elm in prev_capabilities]),
        domain=domain,
        num_gen_capabilities=num_capabilities,
    )

    # Generate output using the model with specified generation arguments
    num_attempts = kwargs.get(
        "retry_attempts", constants.DEFAULT_CAPABILITY_GENERATION_RETRY_ATTEMPTS
    )
    try:
        # Retry the generation process if an error occurs
        # Common errors:
        # - [ill-formatted python class]
        #   - SyntaxError: unterminated triple-quoted string literal
        for attempt in Retrying(
            stop=stop_after_attempt(num_attempts),
            reraise=True,
        ):
            with attempt:
                # Update the seed for each attempt
                scientist_llm_gen_cfg["seed"] += 1
                with tracing_context(
                    enabled=True,
                    tags=["generate_capabilities_using_llm"],
                    metadata={
                        "ls_provider": scientist_llm.model_provider,
                        "ls_model_name": scientist_llm.get_model_name(
                            with_provider=False
                        ),
                        "ls_model_type": "chat",
                        "exp_id": kwargs.get("run_id"),
                        "run_id": kwargs.get("local_run_id"),
                        "domain": domain,
                        "capability_area": capability_area,
                        "num_capabilities": num_capabilities,
                        "seed_capabilities": [elm.name for elm in seed_capabilities],
                        "prev_capabilities": [elm.name for elm in prev_capabilities],
                        **{f"ls_{k}": v for k, v in scientist_llm_gen_cfg.items()},
                    },
                ):
                    response, metadata = scientist_llm.generate(
                        sys_prompt=sys_prompt,
                        user_prompt=user_prompt,
                        generation_config=scientist_llm_gen_cfg,
                    )

                parsed_response = extract_and_parse_response(response)
                gen_capabilities = parsed_response["parsed_response"]
                # Convert JSON string to dict if needed
                gen_capabilities_dict = []
                for capability in gen_capabilities:
                    if isinstance(capability, dict):
                        capability_dict = capability
                    elif isinstance(capability, str):
                        try:
                            capability_dict = json.loads(capability)
                        except json.JSONDecodeError as e:
                            logger.warning(
                                f"Error decoding JSON string: {capability}: {repr(e)}"
                            )
                            continue
                    else:
                        logger.warning(
                            f"Invalid capability format: {capability}. Expected str or dict."
                        )
                        continue
                    gen_capabilities_dict.append(capability_dict)
                gen_capabilities_clean = []
                for capability in gen_capabilities_dict:
                    try:
                        if capability_area is not None:
                            # Add the capability area to the generated capabilities
                            capability["area"] = capability_area
                        capability_obj = Capability.from_dict(
                            capability_dict=capability,
                            base_dir=base_capability_dir,
                            score_dir_suffix=(kwargs.get("run_id")),
                        )
                    except FileExistsError:
                        # 1. Same name as existing capability
                        # Do not delete the capability directory if it already exists
                        logger.warning(
                            f"Capability {capability['name']} already exists. Skipping it."
                        )
                        # Skip this capability
                        continue
                    except Exception as e:
                        # 2. "problem" replaced with "riddle" or some other keyword
                        #   leads to KeyError
                        # 3. Ill-formatted `capability.py` file due to missing quotes
                        logger.warning(
                            f"Error creating capability object {capability['name']}, hence skipping it: {e}"
                        )
                        # Delete the capability directory if it exists
                        capability_dir = os.path.join(
                            base_capability_dir, capability["name"]
                        )
                        if os.path.exists(capability_dir):
                            shutil.rmtree(capability_dir)
                        # Skip this capability
                        continue
                    else:
                        gen_capabilities_clean.append(capability_obj)
                if len(gen_capabilities_clean) != len(gen_capabilities):
                    logger.warning(
                        f"Only {len(gen_capabilities_clean)} capabilities were created out of {len(gen_capabilities)} generated capabilities."
                    )
    except Exception as e:
        logger.error(f"Error generating capabilities: {e}")
        logger.error(f"Response:\n{response}")
        raise e

    logger.info(
        f"Generated {len(gen_capabilities_clean)} capabilities:\n{gen_capabilities_clean}"
    )

    return {
        "capabilities": gen_capabilities_clean,
        "metadata": {
            "model": scientist_llm.get_model_name(),
            "thought": parsed_response["thought"],
            "api_metadata": metadata,
        },
    }
