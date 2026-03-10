"""Utility functions for capability management, loading, filtering, and scoring."""

import logging
import os
from typing import Any, List, Tuple

import torch

from src.capability import Capability
from src.generate_embeddings import filter_embeddings
from src.utils import constants


logger = logging.getLogger(__name__)


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


def filter_schema_capabilities_by_embeddings(
    capabilities: List[Any],  # List of schema Capability objects
    embeddings: List[torch.Tensor],
    similarity_threshold: float,
) -> Tuple[List[Any], List[int]]:
    """Filter schema capabilities based on embedding similarity.

    This function filters capabilities without mutating them, returning both
    the filtered list and the indices of retained capabilities.

    Args
    ----
        capabilities (List[Any]): The list of schema Capability objects.
        embeddings (List[torch.Tensor]): The embeddings corresponding to capabilities.
        similarity_threshold (float): The threshold for cosine similarity
            above which capabilities are considered duplicates.

    Returns
    -------
        Tuple[List[Any], List[int]]:
            - List of filtered capabilities
            - List of indices of retained capabilities
    """
    if len(capabilities) != len(embeddings):
        raise ValueError(
            f"Number of capabilities ({len(capabilities)}) must match "
            f"number of embeddings ({len(embeddings)})"
        )

    remaining_indices = filter_embeddings(embeddings, similarity_threshold)

    logger.info(
        f"Filtered out {len(capabilities) - len(remaining_indices)} "
        f"capabilities out of {len(capabilities)}"
    )

    filtered_capabilities = [capabilities[i] for i in remaining_indices]
    return filtered_capabilities, list(remaining_indices)
