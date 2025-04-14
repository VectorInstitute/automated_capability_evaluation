import json  # noqa: D100
import os
import random
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.capability import Capability
from src.generate_embeddings import (
    DimensionalityReductionTechnique,
    EmbeddingGenerator,
    EmbeddingModelName,
    reduce_embeddings_dimensions,
)
from src.model import Model
from src.utils.capability_utils import extract_and_parse_response
from src.utils.constants import BASE_ARTIFACTS_DIR
from src.utils.prompts import (
    CAPABILITY_GENERATION_SYSTEM_PROMPT,
    CAPABILITY_GENERATION_USER_PROMPT,
)


def _sample_seed_capabilities(
    seed_capability_dir: str,
    num_seed_capabilities: int = -1,
    include_capability_names: List[str] | None = None,
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
        random_seed (int): The seed for the random number generator.

    Returns
    -------
        List[Capability]: A list of capability objects.
    """
    random.seed(random_seed)

    sampled_seed_capabilities = []
    all_seed_capability_paths = os.listdir(seed_capability_dir)

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
    prev_capabilities: List[Capability],
    scientist_llm_gen_cfg: Dict[str, Any],
    base_capability_dir: str,
    include_seed_capability_names: Optional[List[str]] = None,
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
        prev_capabilities (List[Capability]): The list of previously
            generated capabilities.
        scientist_llm_gen_cfg (Dict[str, Any]): The generation configuration
            for the scientist LLM.
        base_capability_dir (str): The base directory to store
            the generated capabilities for the specified domain.
        include_seed_capability_names (List[str] | None): A list of seed capability
            names to include in the generation process.
        **kwargs (Any): Additional keyword arguments.

    Returns
    -------
        Dict[str, Any]: A dictionary containing the generated capabilities
        and metadata about the generation process.
    """
    # Select seed capabilities
    seed_capability_dir = os.path.join(BASE_ARTIFACTS_DIR, "seed_capabilities", domain)
    seed_capabilities = _sample_seed_capabilities(
        seed_capability_dir=seed_capability_dir,
        num_seed_capabilities=num_seed_capabilities,
        include_capability_names=include_seed_capability_names,
    )
    # Get capability JSON strings (without scores)
    seed_capabilities_repr = [
        capability.to_json_str() for capability in seed_capabilities
    ]

    # LLM input
    user_prompt = user_prompt.format(
        seed_capabilities="\n".join(seed_capabilities_repr),
        prev_capabilities="\n".join([elm.name for elm in prev_capabilities]),
        domain=domain,
        num_gen_capabilities=num_capabilities,
    )

    # Generate output using the model with specified generation arguments
    response, metadata = scientist_llm.generate(
        sys_prompt=sys_prompt,
        user_prompt=user_prompt,
        generation_config=scientist_llm_gen_cfg,
    )

    # Print the output
    print(f"Model: {scientist_llm.get_model_name()}")
    print(f"Output:\n\n{response}\n\n")
    print(f"Metadata: {metadata}")

    parsed_response = extract_and_parse_response(response)
    gen_capabilities = parsed_response["parsed_response"]
    gen_capabilities = [
        Capability.from_dict(capability_dict=capability, base_dir=base_capability_dir)
        for capability in gen_capabilities
    ]

    return {
        "capabilities": gen_capabilities,
        "metadata": {
            "model": scientist_llm.get_model_name(),
            "thought": parsed_response["thought"],
            "api_metadata": metadata,
        },
    }


def apply_dimensionality_reduction(
    filtered_capabilities: List[Capability],
    dim_reduction_method: str,
    output_dimensions: int,
    embedding_model_name: str,
) -> None:  # noqa: D205
    """Apply dimensionality reduction to the filtered capabilities.

    This function encodes the provided capabilities by applying the dimensionality
    reduction. The encoded output is stored in the `encoder_output_dict` of
    each capability object under the key corresponding to the encoder model name.
    Currently, multiple dimensionality reduction techniques are supported.

    The T-SNE algorithm operates by reducing the dimensionality of the embeddings
    generated by the OpenAI embedding model to the specified `output_dimensions`.
    Since T-SNE is a non-parametric algorithm, all capabilities must be provided
    at once, as it cannot process new points independently later.

    Args
    ----
        filtered_capabilities (List[Capability]): A list of capabilities with
            valid embeddings.
        dim_reduction_method (str): The dimensionality reduction method to use.
        output_dimensions (int): The number of dimensions to reduce to.
        embedding_model_name (str): The name of the OpenAI embedding model used for
            generating the embeddings.

    Returns
    -------
        None
    """
    # First, generate embeddings using the specified embedding model,
    # then apply the dimensionality reduction technique (e.g., T-SNE).
    # The T-SNE module should be fitted on the entire set of capabilities.
    embeddings = []
    for capability in filtered_capabilities:
        embedding = capability.encode(embedding_model_name)
        assert embedding is not None, (
            f"Capability {capability} does not have a valid embedding."
        )
        embeddings.append(embedding)

    # fit_transform() the t-sne module to the entire set of capabilities.
    # This is a non parametric module, so it only works with all the points
    # available at this time.
    reduced_embeddings = reduce_embeddings_dimensions(
        embeddings,
        output_dimensions=output_dimensions,
        dim_reduction_technique=DimensionalityReductionTechnique(dim_reduction_method),
    )
    # Set the reduced embeddings for each capability.
    for capability, reduced_embedding in zip(filtered_capabilities, reduced_embeddings):
        capability.set_encoder_output(
            encoder_name=dim_reduction_method, encoder_output=reduced_embedding
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
        capability.set_encoder_output(
            encoder_name=embedding_model_name, encoder_output=embeddings[i]
        )


def filter_capabilities(
    capabilities: List[Capability],
    embedding_model_name: str,
    similarity_threshold: float,
) -> List[Capability]:
    """Filter capabilities based embedding similarity.

    This filtering eliminates all closely similar capabilities (neighbors)
    while minimizing the number of removed capabilities.

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
        capability.encode(embedding_model_name) for capability in capabilities
    ]
    # Remove capabilities with close embeddings.
    similarity_matrix = cosine_similarity(embeddings)
    binary_matrix = (similarity_matrix > similarity_threshold).astype(int)
    # Getting the neighbor pairs, and ignoring the diagonal (self neighbors)
    close_pairs = np.argwhere(
        (binary_matrix == 1) & ~np.eye(binary_matrix.shape[0], dtype=bool)
    )
    # Iterate through the similarity matrix
    num_neighbors = {}
    for row_inx in range(len(similarity_matrix)):
        # Count the number of neighbors for each row and
        # subtract 1 to ignore the diagonal (self connection).
        num_neighbors[row_inx] = sum(binary_matrix[row_inx]) - 1
    # Sort the keys in the dictionary by their values in descending order
    sorted_indices = sorted(num_neighbors, key=lambda x: num_neighbors[x], reverse=True)

    # Eliminate all closely similar neighbors while minimizing the number of
    # removed points.
    idx = -1
    remove_indices = set()
    while close_pairs.size > 0:
        idx += 1
        # While there are close embeddings (connections),
        # remove the first index from sorted_indices
        current_connected_index = sorted_indices[idx]
        # Remove any trace of current_connected_index from
        # the close_pairs list because this point is removed.
        pair_idx = 0
        while pair_idx < len(close_pairs):
            if current_connected_index in close_pairs[pair_idx]:
                # Remove the pair_idx from close_pairs np array
                close_pairs = np.delete(close_pairs, pair_idx, axis=0)
                remove_indices.add(current_connected_index)
            else:
                pair_idx += 1
    # Remaining points that are left in sorted_indices are filtered capability indices.
    remaining_indices = set(sorted_indices) - remove_indices
    return [capabilities[i] for i in remaining_indices]


def generate_capabilities(
    domain: str,
    num_capabilities: int,
    num_capabilities_per_run: int,
    scientist_llm: Model,
    num_seed_capabilities: int,
    scientist_llm_gen_cfg: Dict[str, Any],
    include_seed_capability_names: Optional[List[str]] = None,
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
        include_seed_capability_names (List[str] | None): A list of seed capability
            names to include in the generation process.

    Returns
    -------
        List[Capability]: The generated capabilities.
    """
    num_runs = int(np.ceil(num_capabilities / num_capabilities_per_run))
    gen_capabilities = []
    run_metadata = []

    # Set the base capability directory
    if "trial_run" in kwargs:
        base_capability_dir = os.path.join(
            BASE_ARTIFACTS_DIR, f"capabilities_{kwargs['run_id']}", domain
        )
        os.makedirs(base_capability_dir, exist_ok=True)
    else:
        base_capability_dir = os.path.join(BASE_ARTIFACTS_DIR, "capabilities", domain)

    # Fetch previously generated capabilities, if any
    prev_capabilities = _get_previous_capabilities(capability_dir=base_capability_dir)

    for run_id in range(num_runs):
        print("Run ID:", run_id)
        # Generate capabilities using the scientist LLM
        response = generate_capabilities_using_llm(
            domain=domain,
            num_capabilities=num_capabilities_per_run,
            scientist_llm=scientist_llm,
            sys_prompt=CAPABILITY_GENERATION_SYSTEM_PROMPT,
            user_prompt=CAPABILITY_GENERATION_USER_PROMPT,
            num_seed_capabilities=num_seed_capabilities,
            prev_capabilities=prev_capabilities,
            scientist_llm_gen_cfg=scientist_llm_gen_cfg,
            base_capability_dir=base_capability_dir,
            include_seed_capability_names=include_seed_capability_names,
            **kwargs,
        )
        gen_capabilities.extend(response["capabilities"])
        run_metadata.append(response["metadata"])

        # Update the list of previously generated capabilities
        prev_capabilities.extend(response["capabilities"])

    return gen_capabilities
