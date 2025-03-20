from typing import List  # noqa: D100

from utils.prompts import (
    CAPABILITY_GENERATION_SYSTEM_PROMPT,
    CAPABILITY_GENERATION_USER_PROMPT,
)


def generate_capabilities_using_llm(
    domain: str,
    num_capabilities: int,
    scientist_llm: str,
    sys_prompt: str,
    user_prompt: str,
) -> List[str]:
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

    Returns
    -------
        List[str]: The generated capability names.
    """
    raise NotImplementedError


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
    raise NotImplementedError


def generate_capabilities(
    domain: str,
    num_capabilities: int,
    scientist_llm: str,
) -> List[str]:
    """
    Generate initial capabilities for the specified domain.

    Args
    ----
        domain (str): The domain name.
        num_capabilities (int): The number of capabilities to generate.
        scientist_llm (str): The scientist LLM model name.

    Returns
    -------
        List[str]: The generated capability names.
    """
    # Generate capabilities using the scientist LLM
    capabilities = generate_capabilities_using_llm(
        domain=domain,
        num_capabilities=num_capabilities,
        scientist_llm=scientist_llm,
        sys_prompt=CAPABILITY_GENERATION_SYSTEM_PROMPT,
        user_prompt=CAPABILITY_GENERATION_USER_PROMPT,
    )

    return filter_capabilities(capabilities)
