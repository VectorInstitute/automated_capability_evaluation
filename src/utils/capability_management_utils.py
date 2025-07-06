"""Utility functions for capability management, loading, filtering, and scoring."""

import json
import logging
import os
import random
from typing import Any, List, Union

from src.capability import Capability
from src.generate_embeddings import filter_embeddings
from src.utils import constants


logger = logging.getLogger(__name__)


def _sample_seed_capabilities(
    seed_capability_dir: str,
    num_seed_capabilities: int = -1,
    include_capability_names: Union[List[str], None] = None,
    exclude_capability_names: Union[List[str], None] = None,
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
    capability_area: Union[str, None] = None,
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
