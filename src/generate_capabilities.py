import json  # noqa: D100
import logging
import os
import random
from typing import Any, Dict, List, Optional

import numpy as np
from langsmith import tracing_context
from tenacity import Retrying, retry_if_exception_type, stop_after_attempt

from src.capability import Capability
from src.generate_embeddings import (
    DimensionalityReductionTechnique,
    EmbeddingGenerator,
    EmbeddingModelName,
    filter_embeddings,
    reduce_embeddings_dimensions,
)
from src.model import Model
from src.utils import constants, prompts
from src.utils.capability_utils import extract_and_parse_response


logger = logging.getLogger(__name__)


def _sample_seed_capabilities(
    seed_capability_dir: str,
    num_seed_capabilities: int = -1,
    include_capability_names: List[str] | None = None,
    exclude_capability_names: List[str] | None = None,
    random_seed: int = 42,
) -> List[Capability]:
    """
    Sample `num_seed_capabilities` seed capabilities from the specified directory.

    These sampled seed capabilities are used in the input prompt
    to generate new capabilities.

    Args
    ----
        seed_capability_dir (str): The directory containing the seed capabilities.
        num_seed_capabilities (int): The number of seed capabilities to sample.
        include_capability_names (List[str] | None): A list of
            capability names to include.
        exclude_capability_names (List[str] | None): A list of
            capability names to exclude.
        random_seed (int): The seed for the random number generator.

    Returns
    -------
        List[Capability]: A list of capability objects.
    """
    random.seed(random_seed)

    sampled_seed_capabilities = []
    all_seed_capability_paths = os.listdir(seed_capability_dir)

    if exclude_capability_names is not None:
        assert num_seed_capabilities != -1, (
            "Number of seed capabilities should be specified when excluding capabilities."
        )
        assert len(exclude_capability_names) < len(all_seed_capability_paths), (
            "Number of excluded capabilities should be less than the total number of seed capabilities."
        )
        assert (
            len(all_seed_capability_paths) - len(exclude_capability_names)
        ) >= num_seed_capabilities, (
            "Number of remaining seed capabilities should be greater than or equal to the number of seed capabilities to sample."
        )
        # Remove the excluded capabilities from the list
        all_seed_capability_paths = [
            path
            for path in all_seed_capability_paths
            if path not in exclude_capability_names
        ]

    # Select all capabilities if num_seed_capabilities is -1
    if num_seed_capabilities == -1:
        num_seed_capabilities = len(all_seed_capability_paths)
        include_capability_names = None

    # Force include some capabilities
    if include_capability_names is not None:
        assert num_seed_capabilities >= len(include_capability_names), (
            "Number of seed capabilities is less than the number of capabilities to include."
        )
        for capability_name in include_capability_names:
            assert os.path.exists(os.path.join(seed_capability_dir, capability_name)), (
                f"{capability_name} does not exist in {seed_capability_dir}."
            )
            capability = Capability(os.path.join(seed_capability_dir, capability_name))
            sampled_seed_capabilities.append(capability)
            all_seed_capability_paths.remove(capability_name)
        num_seed_capabilities -= len(include_capability_names)

    # TODO: Enhance the selection criterion
    for capability_path in random.sample(
        all_seed_capability_paths, num_seed_capabilities
    ):
        capability = Capability(os.path.join(seed_capability_dir, capability_path))
        sampled_seed_capabilities.append(capability)

    return sampled_seed_capabilities


def _get_previous_capabilities(
    capability_dir: str,
    capability_area: str | None = None,
) -> List[Capability]:
    """
    Get the previously generated capabilities for the specified domain.

    These are included in the input prompt to generate new capabilities.

    Args
    ----
        capability_dir (str): The directory containing the generated capabilities.

    Returns
    -------
        List[Capability]: A list of capabilities.
    """
    prev_capabilities = []
    for capability_path in os.listdir(capability_dir):
        capability = Capability(os.path.join(capability_dir, capability_path))
        if capability_area is not None and capability.area != capability_area:
            continue
        prev_capabilities.append(capability)
    return prev_capabilities


def get_capability_repr_with_score(capability: Capability, model_name: str) -> str:
    """
    Get the capability JSON string with score for the specified model.

    Args
    ----
        capability (Capability): The capability to get the JSON string for.
        model_name (str): The name of the model to use for scoring the capability.

    Returns
    -------
        str: A JSON string containing the capability JSON string and score.
    """
    model_score = capability.load_scores()[model_name]
    capability_dict = capability._to_dict()
    capability_dict["score"] = model_score
    return json.dumps(capability_dict, indent=4)


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
    capability_area: str | None = None,
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
        # Retry the generation process if a SyntaxError occurs
        # Common errors:
        # - [ill-formatted python class]
        #   - SyntaxError: unterminated triple-quoted string literal
        # - [same capability name]
        #   - FileExistsError: Capability directory already exists
        for attempt in Retrying(
            stop=stop_after_attempt(num_attempts),
            retry=retry_if_exception_type((SyntaxError, FileExistsError)),
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
                if capability_area is not None:
                    # Add the capability area to the generated capabilities
                    for capability in gen_capabilities:
                        capability["area"] = capability_area
                gen_capabilities = [
                    Capability.from_dict(
                        capability_dict=capability,
                        base_dir=base_capability_dir,
                        score_dir_suffix=(
                            kwargs.get("run_id") if kwargs.get("trial_run") else None
                        ),
                    )
                    for capability in gen_capabilities
                ]
    except Exception as e:
        logger.error(f"Error generating capabilities: {e}")
        logger.error(f"Response:\n{response}")
        raise e

    return {
        "capabilities": gen_capabilities,
        "metadata": {
            "model": scientist_llm.get_model_name(),
            "thought": parsed_response["thought"],
            "api_metadata": metadata,
        },
    }


def apply_dimensionality_reduction(
    capabilities: List[Capability],
    dim_reduction_method: str,
    output_dimension_size: int,
    embedding_model_name: str,
    seed: int = 42,
) -> None:  # noqa: D205
    """Apply dimensionality reduction to the capabilities.

    This function applies dimensionality reduction on a list of Capabilities.
    The reduced embedding is stored in the `embedding_dict` of
    each capability object with embedding_name corresponding to the dimensionality
    reductio algorithm name.

    The T-SNE algorithm operates by reducing the dimensionality of the embeddings
    generated by the OpenAI embedding model to the specified `output_dimension_size`.
    Since T-SNE is a non-parametric algorithm, all capabilities must be provided
    at once, as it cannot process new points independently later.

    Args
    ----
        capabilities (List[Capability]): A list of capabilities with
            valid embeddings.
        dim_reduction_method (str): The dimensionality reduction method to use.
        output_dimension_size (int): The number of dimensions to reduce to.
        embedding_model_name (str): The name of the OpenAI embedding model used for
            generating the embeddings.
        seed (int): The random seed for reproducibility.

    Returns
    -------
        None
    """
    # First, generate embeddings using the specified embedding model,
    # then apply the dimensionality reduction technique (e.g., T-SNE).
    # The T-SNE module should be fitted on the entire set of capabilities.
    embeddings = []
    for capability in capabilities:
        embedding = capability.get_embedding(embedding_model_name)
        assert embedding is not None, (
            f"Capability {capability} does not have a valid embedding."
        )
        embeddings.append(embedding)

    # fit_transform() the t-sne module to the entire set of capabilities.
    # This is a non parametric module, so it only works with all the points
    # available at this time.
    reduced_embeddings = reduce_embeddings_dimensions(
        embeddings,
        output_dimensions=output_dimension_size,
        dim_reduction_technique=DimensionalityReductionTechnique(dim_reduction_method),
        seed=seed,
    )
    # Set the reduced embeddings for each capability.
    for capability, reduced_embedding in zip(capabilities, reduced_embeddings):
        capability.set_embedding(
            embedding_name=dim_reduction_method, embedding_tensor=reduced_embedding
        )


def generate_and_set_capabilities_embeddings(
    capabilities: List[Capability],
    embedding_model_name: str,
    embed_dimensions: int,
) -> None:
    """Generate the capabilities embeddings using the OpenAI embedding model.

    The embedding of each capability is set in the capability object.

    Args
    ----
        capabilities (List[Capability]): The list of capabilities.
        embedding_model_name (str): The name of the embedding model to use.
        embed_dimensions (int): The number of dimensions for the embeddings.

    """
    # Convert the embedding model name to `EmbeddingModelName` to ensure
    # that the provided model name is valid and supported.
    embedding_generator = EmbeddingGenerator(
        model_name=EmbeddingModelName(
            embedding_model_name
        ),  # Conversion of model name makes sure embedding_model_name is supported.
        embed_dimensions=embed_dimensions,
    )
    # Generate embeddings for the capabilities, all at the same time.
    embeddings = embedding_generator.generate_embeddings(
        texts=[
            capability.to_json_str(attribute_names=["name", "description", "domain"])
            for capability in capabilities
        ]
    )
    # Set embeddings for capabilities.
    for i, capability in enumerate(capabilities):
        capability.set_embedding(
            embedding_name=embedding_model_name, embedding_tensor=embeddings[i]
        )


def filter_capabilities(
    capabilities: List[Capability],
    embedding_model_name: str,
    similarity_threshold: float,
) -> List[Capability]:
    """Filter capabilities based embedding similarity.

    Calls filter_embeddings that eliminates all closely similar
    capability embeddings (neighbors) while minimizing the number of
    removed capabilities.

    Args
    ----
        capabilities (List[Capability]): The list of capabilities.
        embedding_model_name (str): The name of the OpenAI embedding model used for
            generating the embeddings.
        similarity_threshold (float): The threshold for cosine similarity
                        above which capabilities are considered duplicates.

    Returns
    -------
        List[Capability]: The list of remaining capabilities.
    """
    embeddings = [
        capability.get_embedding(embedding_model_name) for capability in capabilities
    ]
    remaining_indices = filter_embeddings(embeddings, similarity_threshold)
    return [capabilities[i] for i in remaining_indices]


def generate_capability_areas(
    domain: str,
    num_areas: int,
    num_capabilities_per_area: int,
    scientist_llm: Model,
    user_prompt: str,
    scientist_llm_gen_cfg: Dict[str, Any],
    sys_prompt: str | None = None,
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

    # Set the base capability directory
    if "trial_run" in kwargs:
        base_capability_dir = os.path.join(
            constants.BASE_ARTIFACTS_DIR, f"capabilities_{kwargs['run_id']}", domain
        )
        os.makedirs(base_capability_dir, exist_ok=True)
    else:
        base_capability_dir = os.path.join(
            constants.BASE_ARTIFACTS_DIR, "capabilities", domain
        )

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
            prev_capabilities = _get_previous_capabilities(
                capability_dir=base_capability_dir, capability_area=capability_area
            )
            user_prompt = prompts.HIERARCHICAL_CAPABILITY_GENERATION_USER_PROMPT.format(
                capability_area=capability_area,
            )
        else:
            prev_capabilities = _get_previous_capabilities(
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
