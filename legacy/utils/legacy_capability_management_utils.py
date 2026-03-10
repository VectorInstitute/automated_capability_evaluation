"""Legacy capability management helpers."""

import logging
import os
from typing import Any, List

from legacy.src.capability import Capability
from legacy.utils import legacy_constants as constants
from src.utils.embedding_utils import filter_embeddings


logger = logging.getLogger(__name__)


def get_previous_capabilities(
    capability_dir: str,
    capability_area: str | None = None,
    **kwargs: Any,
) -> List[Capability]:
    """Load legacy `Capability` objects from a directory."""
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
    """Filter legacy capabilities by embedding similarity."""
    embeddings = [
        capability.get_embedding(embedding_model_name) for capability in capabilities
    ]
    remaining_indices = filter_embeddings(embeddings, similarity_threshold)

    filtered_out_capabilities = []
    for capability in (
        cap for i, cap in enumerate(capabilities) if i not in remaining_indices
    ):
        capability.set_state(
            constants.C_STATE_FILTERED_OUT_STR,
        )
        filtered_out_capabilities.append(capability)
    logger.info(
        "Filtered out %d capabilities:\n%s",
        len(filtered_out_capabilities),
        filtered_out_capabilities,
    )
    return [capabilities[i] for i in remaining_indices]
