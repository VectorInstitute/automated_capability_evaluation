import json  # noqa: D100
import os
import random
from typing import Any, Dict, List, Optional

from capability import Capability
from model import Model
from utils.capability_utils import extract_and_parse_response
from utils.constants import BASE_ARTIFACTS_DIR
from utils.prompts import (
    CAPABILITY_GENERATION_SYSTEM_PROMPT,
    CAPABILITY_GENERATION_USER_PROMPT,
)


def select_seed_capabilities(
    seed_capability_dir: str,
    num_seed_capabilities: int = -1,
    include_capabilities: List[str] | None = None,
    random_seed: int = 42,
) -> List[Capability]:
    """
    Select `num_seed_capabilities` seed capabilities from the specified directory.

    Args
    ----
        seed_capability_dir (str): The directory containing the seed capabilities.
        num_seed_capabilities (int): The number of seed capabilities to select.
        include_capabilities (List[str] | None): A list of capability names to include.
        random_seed (int): The seed for the random number generator.

    Returns
    -------
        List[Capability]: A list of capability objects.
    """
    random.seed(random_seed)

    selected_seed_capabilities = []
    all_seed_capability_paths = os.listdir(seed_capability_dir)

    # Select all capabilities if num_seed_capabilities is -1
    if num_seed_capabilities == -1:
        num_seed_capabilities = len(all_seed_capability_paths)
        include_capabilities = None

    # Force include some capabilities
    if include_capabilities is not None:
        assert num_seed_capabilities >= len(include_capabilities), (
            "Number of seed capabilities is less than the number of capabilities to include."
        )
        for capability_name in include_capabilities:
            capability = Capability(os.path.join(seed_capability_dir, capability_name))
            selected_seed_capabilities.append(capability)
            all_seed_capability_paths.remove(capability_name)
        num_seed_capabilities -= len(include_capabilities)

    # TODO: Enhance the selection criterion
    for capability_path in random.sample(
        all_seed_capability_paths, num_seed_capabilities
    ):
        capability = Capability(os.path.join(seed_capability_dir, capability_path))
        selected_seed_capabilities.append(capability)

    return selected_seed_capabilities


def get_previous_capabilities(
    capability_dir: str,
) -> List[Capability]:
    """
    Get the previously generated capabilities for the specified domain.

    Args
    ----
        capability_dir (str): The directory containing the generated capabilities.

    Returns
    -------
        List[Capability]: A list of capabilities.
    """
    prev_capabilities = []
    for capability_path in os.listdir(capability_dir):
        try:
            capability = Capability(os.path.join(capability_dir, capability_path))
        except Exception as e:
            print(f"{capability_path} could not be loaded: {e}")
            continue
        prev_capabilities.append(capability)
    return prev_capabilities


def get_capability_repr_with_score(capability: Capability, model_name: str) -> str:
    """
    Get the capability representation with score for the specified model.

    Args
    ----
        capability (Capability): The capability to get the representation for.
        model_name (str): The name of the model to use for scoring the capability.

    Returns
    -------
        str: A JSON string containing the capability representation and score.
    """
    model_score = capability.load_scores()[model_name]
    capability_dict = capability._to_dict()
    capability_dict["score"] = model_score
    return json.dumps(capability_dict, indent=4)


def generate_capabilities_using_llm(
    domain: str,
    num_capabilities: int,
    scientist_llm: str,
    sys_prompt: str,
    user_prompt: str,
    num_seed_capabilities: int,
    scientist_llm_gen_cfg: Dict[str, Any],
    include_seed_capabilities: Optional[List[str]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Generate capabilities using the scientist LLM.

    Prompt the scientist LLM with instructions and
    seed capability representations for the specified domain
    to generate initial capabilities.

    Args
    ----
        domain (str): The domain name.
        num_capabilities (int): The number of capabilities to generate.
        scientist_llm (str): The scientist LLM model name.
        sys_prompt (str): The system prompt.
        user_prompt (str): The user prompt.
        num_seed_capabilities (int): The number of seed capabilities to use.
        scientist_llm_gen_cfg (Dict[str, Any]): The generation configuration
            for the scientist LLM.
        include_seed_capabilities (List[str] | None): A list of seed capability
            names to include in the generation process.

    Returns
    -------
        List[str]: The generated capability names.
    """
    if "trial_run" in kwargs:
        base_capability_dir = os.path.join(
            BASE_ARTIFACTS_DIR, f"capabilities_{kwargs['run_id']}"
        )
        os.makedirs(base_capability_dir, exist_ok=True)
    else:
        base_capability_dir = os.path.join(BASE_ARTIFACTS_DIR, "capabilities")

    # Select seed capabilities
    seed_capability_dir = os.path.join(BASE_ARTIFACTS_DIR, "seed_capabilities", domain)
    seed_capabilities = select_seed_capabilities(
        seed_capability_dir=seed_capability_dir,
        num_seed_capabilities=num_seed_capabilities,
        include_capabilities=include_seed_capabilities,
    )
    # Get previous capability names
    capability_dir = os.path.join(base_capability_dir, domain)
    os.makedirs(capability_dir, exist_ok=True)
    prev_capabilities = [
        elm.name for elm in get_previous_capabilities(capability_dir=capability_dir)
    ]

    # Create an instance of the Model class with the specified model name
    model = Model(
        model_name=scientist_llm,
        sys_msg=sys_prompt,
    )

    # Get capability representations (without scores)
    seed_capabilities_repr = [
        capability.to_json_str() for capability in seed_capabilities
    ]
    # LLM input
    sample_input = user_prompt.format(
        seed_capabilities="\n".join(seed_capabilities_repr),
        prev_capabilities="\n".join(prev_capabilities),
        domain=domain,
        num_gen_capabilities=num_capabilities,
    )

    # Generate output using the model with specified generation arguments
    response, metadata = model.generate(
        prompt=sample_input,
        generation_config=scientist_llm_gen_cfg,
    )

    # Print the output
    print(f"Model: {model.get_model_name()}")
    print(f"Output:\n\n{response}\n\n")
    print(f"Metadata: {metadata}")

    parsed_response = extract_and_parse_response(response)
    gen_capabilities = parsed_response["capabilities"]
    gen_capabilities = [
        Capability.from_dict(capability_dict=capability, base_dir=capability_dir)
        for capability in gen_capabilities
    ]
    gen_capabilities_names = [capability.name for capability in gen_capabilities]

    return {
        "capabilities": gen_capabilities_names,
        "metadata": {
            "model": model.get_model_name(),
            "thought": parsed_response["thought"],
            "api_metadata": metadata,
        },
    }


def filter_capabilities(
    capabilities: List[str],
) -> List[str]:
    """
    Filter capabilities based on multiple criterion.

    Remove repeated, irrelevant, and ill-formed capabilities.

    Args
    ----
        capabilities (List[str]): The list of capabilities.

    Returns
    -------
        List[str]: The filtered capability names.
    """
    # TODO: Implement capability filtering
    return capabilities


def generate_capabilities(
    domain: str,
    num_capabilities: int,
    scientist_llm: str,
    num_seed_capabilities: int,
    scientist_llm_gen_cfg: Dict[str, Any],
    include_seed_capabilities: Optional[List[str]] = None,
    **kwargs: Any,
) -> List[str]:
    """
    Generate initial capabilities for the specified domain.

    Args
    ----
        domain (str): The domain name.
        num_capabilities (int): The number of capabilities to generate.
        scientist_llm (str): The scientist LLM model name.
        num_seed_capabilities (int): The number of seed capabilities to use.
        scientist_llm_gen_cfg (Dict[str, Any]): The generation configuration
            for the scientist LLM.
        include_seed_capabilities (List[str] | None): A list of seed capability
            names to include in the generation process.

    Returns
    -------
        List[str]: The generated capability names.
    """
    # Generate capabilities using the scientist LLM
    response = generate_capabilities_using_llm(
        domain=domain,
        num_capabilities=num_capabilities,
        scientist_llm=scientist_llm,
        sys_prompt=CAPABILITY_GENERATION_SYSTEM_PROMPT,
        user_prompt=CAPABILITY_GENERATION_USER_PROMPT,
        num_seed_capabilities=num_seed_capabilities,
        scientist_llm_gen_cfg=scientist_llm_gen_cfg,
        include_seed_capabilities=include_seed_capabilities,
        **kwargs,
    )
    print(response)
    gen_capabilities = response["capabilities"]

    return filter_capabilities(gen_capabilities)
