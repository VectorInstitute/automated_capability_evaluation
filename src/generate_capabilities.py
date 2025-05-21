"""Generate capabilities using the scientist LLM."""

import json
import logging
import os
import random
import shutil
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from langsmith import tracing_context
from tenacity import Retrying, stop_after_attempt

from src.capability import Capability
from src.dimensionality_reduction import DimensionalityReductionMethod
from src.generate_embeddings import (
    EmbeddingGenerator,
    EmbeddingModelName,
    filter_embeddings,
    hierarchical_2d_visualization,
    save_embedding_heatmap,
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

    for capability_path in random.sample(
        all_seed_capability_paths, num_seed_capabilities
    ):
        capability = Capability(os.path.join(seed_capability_dir, capability_path))
        sampled_seed_capabilities.append(capability)

    return sampled_seed_capabilities


def get_previous_capabilities(
    capability_dir: str,
    capability_area: str | None = None,
    **kwargs: Any,
) -> List[Capability]:
    """
    Get the previously generated capabilities for the specified domain.

    These are included in the input prompt to generate new capabilities.

    Args
    ----
        capability_dir (str): The directory containing the generated capabilities.
        capability_area (str | None): The capability area to filter by.
        **kwargs (Any): Additional keyword arguments.

    Returns
    -------
        List[Capability]: A list of capabilities.
    """
    prev_capabilities = []
    for capability_path in os.listdir(capability_dir):
        capability = Capability(
            capability_dir=os.path.join(capability_dir, capability_path),
            score_dir_suffix=kwargs.get("score_dir_suffix"),
        )
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
    if not hasattr(capability, "scores") or model_name not in capability.scores:
        capability.load_scores(subject_llm_name=model_name)
    model_score = capability.scores[model_name]
    capability_dict = capability.to_dict()
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
                        # 2. “problem” replaced with “riddle” or some other keyword
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


def plot_hierarchical_capability_2d_embeddings(
    capabilities: List[Capability],
    dim_reduction_method: str,
    plot_name: str,
    save_dir: str,
    show_point_ids: bool,
) -> None:
    """Visualize the hierarchical capability embeddings.

    Embeddings are retrieved based on the defined dim_reduction_method,
    and they should be 2D.

    Args
    ----
        capabilities (List[Capability]): The list of capabilities.
        dim_reduction_method (str): The dimensionality reduction method to use.
        plot_name (str): The name of the plot to save.
        save_dir (str): The directory to save the plot.
        show_point_ids (bool): Whether to show point IDs in the plot. Set this to
            False for large datasets to avoid cluttering the plot.
    """
    # Get the reduced embeddings.
    reduced_embeddings = [
        capability.get_embedding(dim_reduction_method) for capability in capabilities
    ]
    area_names = [capability.get_attribute("area") for capability in capabilities]

    # Populate embeddings_by_area, and points_area_name_ids
    embeddings_by_area: dict[str, List[torch.Tensor]] = {}
    points_area_name_ids: dict[str, dict[str, int]] = {}
    for idx in range(len(reduced_embeddings)):
        area_name = area_names[idx]
        if area_name not in embeddings_by_area:
            embeddings_by_area[area_name] = []
            points_area_name_ids[area_name] = {}
        embeddings_by_area[area_name].append(reduced_embeddings[idx])
        points_area_name_ids[area_name][capabilities[idx].name] = idx

    hierarchical_2d_visualization(
        embeddings_by_area=embeddings_by_area,
        save_dir=save_dir,
        plot_name=plot_name,
        points_area_name_ids=points_area_name_ids if show_point_ids else None,
    )


def generate_capability_heatmap(
    capabilities: List[Capability],
    embedding_model_name: str,
    plot_name: str,
    save_dir: str,
    add_squares: bool,
) -> None:
    """
    Generate and save a heatmap of the capabilities based on their embeddings.

    Args:
        capabilities (List[Capability]): the list of capabilities.
        embedding_model_name (str): name of the embedding model used
            to generate the embeddings.
        plot_name (str): name of the plot file to save.
        save_dir (str): directory to save the plot.
        add_squares (bool): whether to add squares to the heatmap.
    """
    # Get the embeddings based on the specified embedding model name.
    embeddings = [
        capability.get_embedding(embedding_model_name) for capability in capabilities
    ]
    # Process capabilities to populate embeddings_by_area and
    # capability_names_by_area.
    area_names = [capability.area for capability in capabilities]
    embeddings_by_area: dict[str, List[torch.Tensor]] = {}
    capability_names_by_area: dict[str, List[str]] = {}
    for idx in range(len(capabilities)):
        embedding_group = area_names[idx]
        if embedding_group not in embeddings_by_area:
            embeddings_by_area[embedding_group] = []
            capability_names_by_area[embedding_group] = []
        embeddings_by_area[embedding_group].append(embeddings[idx])
        capability_names_by_area[embedding_group].append(capabilities[idx].name)

    save_embedding_heatmap(
        embeddings_by_area=embeddings_by_area,
        capability_names_by_area=capability_names_by_area,
        save_dir=save_dir,
        plot_name=plot_name,
        add_squares=add_squares,
    )


def apply_dimensionality_reduction(
    capabilities: List[Capability],
    dim_reduction_method_name: str,
    output_dimension_size: int,
    embedding_model_name: str,
    tsne_perplexity: int | None = None,
    random_seed: int = constants.DEFAULT_RANDOM_SEED,
    normalize_output: bool = True,
) -> DimensionalityReductionMethod:  # noqa: D205
    """Apply dimensionality reduction to the capabilities.

    This function applies dimensionality reduction on a list of Capabilities.
    The reduced embedding is stored in the `embedding_dict` of
    each capability object with embedding_name corresponding to the dimensionality
    reduction algorithm name.

    Args
    ----
        capabilities (List[Capability]): A list of capabilities with
            valid embeddings.
        dim_reduction_method_name (str): The dimensionality reduction method to use.
        output_dimension_size (int): The number of dimensions to reduce to.
        embedding_model_name (str): The name of the OpenAI embedding model used for
            generating the embeddings.
        tsne_perplexity (int | None): The perplexity parameter for T-SNE.
        random_seed (int): The seed for the random number generator.
        normalize_output (bool): Whether to normalize the output embeddings.

    Returns
    -------
        dim_reduction (DimensionalityReductionMethod):
            The dimensionality reduction object. This object
            can be used to transform new embeddings.
    """
    # First, generate embeddings using the specified embedding model,
    # then apply the dimensionality reduction technique (e.g., T-SNE).
    embeddings = []
    for capability in capabilities:
        embedding = capability.get_embedding(embedding_model_name)
        assert embedding is not None, (
            f"Capability {capability} does not have a valid embedding."
        )
        embeddings.append(embedding)

    dim_reduction = DimensionalityReductionMethod.from_name(
        dim_reduction_method_name,
        output_dimension_size,
        random_seed=random_seed,
        normalize_output=normalize_output,
        tsne_perplexity=tsne_perplexity,
    )
    # fit_transform() the dimensionality reduction module on the embeddings.
    reduced_embeddings = dim_reduction.fit_transform(embeddings)

    # Set the reduced embeddings for each capability.
    for capability, reduced_embedding in zip(capabilities, reduced_embeddings):
        capability.set_embedding(
            embedding_name=dim_reduction_method_name, embedding_tensor=reduced_embedding
        )
    return dim_reduction


def apply_dimensionality_reduction_to_test_capabilities(
    capabilities: List[Capability],
    dim_reduction_method: DimensionalityReductionMethod,
    embedding_model_name: str,
) -> None:
    """Apply dimensionality reduction to the test capabilities.

    This function applies dimensionality reduction on a list of Capabilities.
    The reduced embedding is stored in the `embedding_dict` of
    each capability object with embedding_name corresponding to the dimensionality
    reduction algorithm name.

    Args
    ----
        capabilities (List[Capability]): A list of capabilities with
            valid embeddings.
        dim_reduction_method (DimensionalityReductionMethod): The dimensionality
            reduction method to use.
        embedding_model_name (str): The name of the embedding model used for
            generating the embeddings.
    """
    # Apply the dimensionality reduction technique on test capabilities.
    reduced_embeddings = dim_reduction_method.transform_new_points(
        [capability.get_embedding(embedding_model_name) for capability in capabilities]
    )

    # Set the reduced embeddings for each capability.
    for capability, reduced_embedding in zip(capabilities, reduced_embeddings):
        capability.set_embedding(
            embedding_name=dim_reduction_method.method_name,
            embedding_tensor=reduced_embedding,
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
    # Embeddings are generated based on the capability name and description.
    texts = []
    for capability in capabilities:
        capability_dict = capability.to_dict(attribute_names=["name", "description"])
        texts.append(f"{capability_dict['name']}: {capability_dict['description']}")
    embeddings = embedding_generator.generate_embeddings(texts)
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
    # Update state of filtered capabilities
    filtered_out_capabilities = []
    for capability in (
        cap for i, cap in enumerate(capabilities) if i not in remaining_indices
    ):
        capability.set_state(
            constants.C_STATE_FILTERED_OUT_STR,
        )
        filtered_out_capabilities.append(capability)
    logger.info(
        f"Filtered out {len(filtered_out_capabilities)} capabilities:\n{filtered_out_capabilities}"
    )
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
