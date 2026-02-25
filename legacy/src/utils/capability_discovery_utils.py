"""Utility functions for capability discovery and selection."""

import json
import logging
import os
import random
import shutil
from collections import defaultdict
from typing import Any, Dict, List

from langsmith import tracing_context
from tenacity import Retrying, stop_after_attempt

from src.capability import Capability
from src.model import Model
from src.utils import constants, prompts
from src.utils.capability_utils import extract_and_parse_response


logger = logging.getLogger(__name__)


def score_based_capability_discovery(
    prev_capabilities: List[Capability],
    domain: str,
    base_capability_dir: str,
    user_prompt: str,
    scientist_llm: Model,
    scientist_llm_gen_cfg: Dict[str, Any],
    subject_llm_name: str,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Generate new capabilities based on existing ones using the scientist LLM.

    This function leverages the scores of previously generated capabilities
    to guide the generation of new capabilities. It uses the scientist LLM
    to generate a new capability based on existing capability and their
    associated scores.

    Args
    ----
        prev_capabilities (List[Capability]): The list of previously
            generated capabilities.
        domain (str): The domain name.
        base_capability_dir (str): The base directory to store the
            generated capabilities.
        user_prompt (str): The user prompt for generating new capabilities.
        scientist_llm (Model): The scientist LLM model.
        scientist_llm_gen_cfg (Dict[str, Any]): The generation configuration
            for the scientist LLM.
        subject_llm_name (str): The name of the subject LLM used for scoring.
        **kwargs (Any): Additional keyword arguments.

    Returns
    -------
        Dict[str, Any]: A dictionary containing the newly discovered capability
        and metadata about the generation process.
    """
    random.seed(int(kwargs.get("seed", constants.DEFAULT_RANDOM_SEED)))

    # Get capability names with scores
    capability_score_dict = {}
    for capability in prev_capabilities:
        capability_score_dict[capability.name] = capability.scores[subject_llm_name][
            "mean"
        ]

    # Randomly sample a capability from the existing capabilities
    sample_capability = random.choice(prev_capabilities)

    # Build the user prompt
    user_prompt = user_prompt.format(
        sample_capability_json=sample_capability.to_json_str(),
        prev_capabilities_and_scores=json.dumps(capability_score_dict, indent=4),
        domain=domain,
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
                    tags=["score_based_capability_discovery"],
                    metadata={
                        "ls_provider": scientist_llm.model_provider,
                        "ls_model_name": scientist_llm.get_model_name(
                            with_provider=False
                        ),
                        "ls_model_type": "chat",
                        "exp_id": kwargs.get("run_id"),
                        "domain": domain,
                        "subject_llm_name": subject_llm_name,
                        "prev_capabilities": [elm.name for elm in prev_capabilities],
                        **{f"ls_{k}": v for k, v in scientist_llm_gen_cfg.items()},
                    },
                ):
                    response, metadata = scientist_llm.generate(
                        sys_prompt=prompts.CAPABILITY_GENERATION_SYSTEM_PROMPT,
                        user_prompt=user_prompt,
                        generation_config=scientist_llm_gen_cfg,
                    )

                if response is None:
                    raise ValueError("Response is None")
                parsed_response = extract_and_parse_response(response)
                # Fetch the first capability from the response if multiple are generated
                gen_capability = parsed_response["parsed_response"][0]
                # Convert JSON string to dict if needed
                if isinstance(gen_capability, dict):
                    gen_capability_dict = gen_capability
                elif isinstance(gen_capability, str):
                    gen_capability_dict = json.loads(gen_capability)
                else:
                    raise ValueError(
                        f"Invalid capability format: {gen_capability}. Expected str or dict."
                    )
                # Load the capability object
                try:
                    gen_capability_obj = Capability.from_dict(
                        capability_dict=gen_capability_dict,
                        base_dir=base_capability_dir,
                        score_dir_suffix=(kwargs.get("run_id")),
                    )
                except FileExistsError as e:
                    # Do not delete the capability directory if it already exists
                    logger.error(
                        f"Capability {gen_capability_dict['name']} already exists. Updating seed to generate a new capability."
                    )
                    raise e
                except Exception as e:
                    logger.error(
                        f"Error creating capability object {gen_capability_dict['name']}: {repr(e)}"
                    )
                    # Delete the capability directory if it exists
                    capability_dir = os.path.join(
                        base_capability_dir, gen_capability_dict["name"]
                    )
                    if os.path.exists(capability_dir):
                        shutil.rmtree(capability_dir)
                    raise e
    except Exception as e:
        logger.error(f"Error generating capability: {e}")
        logger.error(f"Response:\n{response}")
        raise e

    logger.info(f"Generated capability: {gen_capability_obj.name}")
    logger.info(f"Capability generation tokens summary\n{metadata}")

    return {
        "capability": gen_capability_obj,
        "metadata": {
            "model": scientist_llm.get_model_name(),
            "thought": parsed_response["thought"],
            "api_metadata": metadata,
        },
    }


def knn_based_capability_discovery(
    knn_capabilities: List[Capability],
    prev_capabilities: List[Capability],
    domain: str,
    base_capability_dir: str,
    user_prompt: str,
    scientist_llm: Model,
    scientist_llm_gen_cfg: Dict[str, Any],
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Utilize a LBO KNN approach to guide the generation of new capabilities.

    This function leverages the scores and structure of previously generated
    capabilities to create new ones. The scientist LLM is used to generate
    these capabilities based on a user-defined prompt and configuration.

    Args
    ----
        knn_capabilities (List[Capability]): A list of capabilities identified
            as nearest neighbors to guide the generation process.
        prev_capabilities (List[Capability]): The list of previously generated
            capabilities used for sampling and context.
        domain (str): The domain name for which the capabilities are being
            generated.
        base_capability_dir (str): The base directory to store the generated
            capabilities.
        user_prompt (str): The user-defined prompt for generating new capabilities.
        scientist_llm (Model): The scientist LLM model used for capability generation.
        scientist_llm_gen_cfg (Dict[str, Any]): The generation configuration for the
            scientist LLM, including parameters like seed and temperature.
        **kwargs (Any): Additional keyword arguments, including:
            - seed (int): Random seed for reproducibility.
            - retry_attempts (int): Number of retry attempts for generation.
            - run_id (str): Experiment or run identifier for tracking.

    Returns
    -------
        Dict[str, Any]: A dictionary containing:
            - "capability" (Capability): The newly discovered capability object.
            - "metadata" (Dict[str, Any]): Metadata about the generation process,
              including model details, thought process, and API usage.

    Raises
    ------
        ValueError: If the generated capability format is invalid.
        Exception: If an error occurs during capability generation or object creation.
    """
    random.seed(int(kwargs.get("seed", constants.DEFAULT_RANDOM_SEED)))

    # Randomly sample a capability from the existing capabilities
    sample_capability = random.choice(prev_capabilities)

    # Get JSON string for KNN capabilities
    knn_capabilities_json_str = [
        capability.to_json_str() for capability in knn_capabilities
    ]

    # Build the user prompt
    user_prompt = user_prompt.format(
        num_input_capabilities=len(knn_capabilities),
        prev_capabilities=json.dumps(knn_capabilities_json_str, indent=4),
        sample_capability_json=sample_capability.to_json_str(),
        domain=domain,
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
                    tags=["knn_based_capability_discovery"],
                    metadata={
                        "ls_provider": scientist_llm.model_provider,
                        "ls_model_name": scientist_llm.get_model_name(
                            with_provider=False
                        ),
                        "ls_model_type": "chat",
                        "exp_id": kwargs.get("run_id"),
                        "domain": domain,
                        "knn_capabilities": [elm.name for elm in knn_capabilities],
                        **{f"ls_{k}": v for k, v in scientist_llm_gen_cfg.items()},
                    },
                ):
                    response, metadata = scientist_llm.generate(
                        sys_prompt=prompts.CAPABILITY_GENERATION_SYSTEM_PROMPT,
                        user_prompt=user_prompt,
                        generation_config=scientist_llm_gen_cfg,
                    )

                if response is None:
                    raise ValueError("Response is None")
                parsed_response = extract_and_parse_response(response)
                # Fetch the first capability from the response if multiple are generated
                gen_capability = parsed_response["parsed_response"][0]
                # Convert JSON string to dict if needed
                if isinstance(gen_capability, dict):
                    gen_capability_dict = gen_capability
                elif isinstance(gen_capability, str):
                    gen_capability_dict = json.loads(gen_capability)
                else:
                    raise ValueError(
                        f"Invalid capability format: {gen_capability}. Expected str or dict."
                    )
                # Load the capability object
                try:
                    gen_capability_obj = Capability.from_dict(
                        capability_dict=gen_capability_dict,
                        base_dir=base_capability_dir,
                        score_dir_suffix=(kwargs.get("run_id")),
                    )
                except FileExistsError as e:
                    # Do not delete the capability directory if it already exists
                    logger.error(
                        f"Capability {gen_capability_dict['name']} already exists. Updating seed to generate a new capability."
                    )
                    raise e
                except Exception as e:
                    logger.error(
                        f"Error creating capability object {gen_capability_dict['name']}: {repr(e)}"
                    )
                    # Delete the capability directory if it exists
                    capability_dir = os.path.join(
                        base_capability_dir, gen_capability_dict["name"]
                    )
                    if os.path.exists(capability_dir):
                        shutil.rmtree(capability_dir)
                    raise e
    except Exception as e:
        logger.error(f"Error generating capability: {e}")
        logger.error(f"Response:\n{response}")
        raise e

    logger.info(f"Generated capability: {gen_capability_obj.name}")
    logger.info(f"Capability generation tokens summary\n{metadata}")

    return {
        "capability": gen_capability_obj,
        "metadata": {
            "model": scientist_llm.get_model_name(),
            "thought": parsed_response["thought"],
            "api_metadata": metadata,
        },
    }


def select_complete_capabilities(
    capabilities: List[Capability],
    strict: bool = True,
    num_tasks_lower_bound: int = 0,
) -> List[Capability]:
    """
    Select and summarize the generated capabilities with a specific state.

    This function filters the capabilities to include only those with the
    state `TASK_GENERATION_COMPLETED` and provides a summary of the states
    of all capabilities.

    Args
    ----
        capabilities (List[Capability]): The list of generated capabilities.
        strict (bool): If True, only capabilities with the state
            `TASK_GENERATION_COMPLETED` are selected. If False, capabilities
            with at least `num_tasks_lower_bound` tasks are also selected.
        num_tasks_lower_bound (int): The minimum number of tasks required

    Returns
    -------
        List[Capability]: A list of capabilities with the state
        `TASK_GENERATION_COMPLETED`.
    """
    keep_capabilities = []
    cap_state_count: Dict[str, int] = defaultdict(int)

    for capability in capabilities:
        # Get the state of the capability
        cap_state_count[capability.get_state().value] += 1

        if capability_satisfies_criterion(
            capability=capability,
            strict=strict,
            num_tasks_lower_bound=num_tasks_lower_bound,
        ):
            # If the capability satisfies the criterion, keep it
            keep_capabilities.append(capability)

    logger.info(
        f"Capability generation summary:\n{json.dumps(cap_state_count, indent=4)}"
    )
    logger.info(
        f"Selected {len(keep_capabilities)} capabilities with state {constants.C_STATE_TASK_GENERATION_COMPLETED_STR}"
        + (f" or with at least {num_tasks_lower_bound} tasks" if not strict else "")
    )

    return keep_capabilities


def capability_satisfies_criterion(
    capability: Capability, strict: bool = True, num_tasks_lower_bound: int = 0
) -> bool:
    """
    Determine whether a given capability satisfies the specified criteria.

    This function evaluates if a capability meets the conditions for being
    considered complete. The criteria can be adjusted based on the `strict`
    parameter and the minimum number of tasks required.

    Args
    ----
        capability (Capability): The capability object to evaluate.
        strict (bool, optional): If True, only capabilities with the
            TASK_GENERATION_COMPLETED state are considered valid. If False,
            capabilities with at least `num_tasks_lower_bound` tasks are also
            considered valid. Defaults to True.
        num_tasks_lower_bound (int, optional): The minimum number of tasks
            required for a capability to be considered valid when `strict` is
            False. Defaults to 0.

    Returns
    -------
        bool: True if the capability satisfies the criteria, False otherwise.
    """
    return (
        # Keep only capabilities with TASK_GENERATION_COMPLETED state
        capability.get_state().value == constants.C_STATE_TASK_GENERATION_COMPLETED_STR
    ) or (
        # If strict is False, keep capabilities with at least
        # num_tasks_lower_bound tasks
        not strict and len(capability.get_tasks()) >= num_tasks_lower_bound
    )
