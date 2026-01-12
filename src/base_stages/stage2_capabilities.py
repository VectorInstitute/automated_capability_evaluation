"""Stage 2: Capability generation and filtering.

This stage generates capabilities for each area, embeds them, and filters
by similarity to remove duplicates.
"""

import logging
import math
from pathlib import Path

from omegaconf import DictConfig

from src.base_stages.generate_capabilities import generate_capabilities
from src.schemas.io_utils import load_areas, save_capabilities
from src.schemas.metadata_schemas import PipelineMetadata
from src.utils import constants
from src.utils.capability_management_utils import (
    filter_schema_capabilities_by_embeddings,
)
from src.utils.embedding_utils import generate_capability_embeddings
from src.utils.model_client_utils import get_standard_model_client
from src.utils.timestamp_utils import iso_timestamp, timestamp_tag


logger = logging.getLogger(__name__)


def run_stage2(
    cfg: DictConfig,
    areas_tag: str,
    capabilities_tag: str = None,
) -> str:
    """Stage 2: Generate capabilities, embed, and filter.

    Args:
        cfg: Configuration object
        areas_tag: Tag from Stage 1 to load areas from
        capabilities_tag: Optional resume tag

    Returns
    -------
        The capabilities_tag for this generation
    """
    experiment_id = cfg.exp_cfg.exp_id
    output_base_dir = Path(cfg.global_cfg.output_dir)

    # Load areas from Stage 1 output
    areas_path = output_base_dir / experiment_id / "areas" / areas_tag / "areas.json"
    areas, _ = load_areas(areas_path)
    logger.info(f"Loaded {len(areas)} area(s) from Stage 1")

    # Initialize scientist LLM client directly with generation parameters
    scientist_llm_gen_cfg = dict(cfg.scientist_llm.generation_cfg.capability_generation)
    scientist_llm_client = get_standard_model_client(
        cfg.scientist_llm.name,
        seed=scientist_llm_gen_cfg.get("seed", cfg.exp_cfg.seed),
        temperature=scientist_llm_gen_cfg.get(
            "temperature", constants.DEFAULT_TEMPERATURE
        ),
        max_tokens=scientist_llm_gen_cfg.get(
            "max_tokens", constants.DEFAULT_MAX_TOKENS
        ),
    )

    # Determine capabilities tag (resume or new)
    is_resume = capabilities_tag is not None
    if is_resume:
        logger.info(f"Resuming Stage 2 with capabilities_tag: {capabilities_tag}")
    else:
        capabilities_tag = timestamp_tag()
        logger.info(f"Starting new Stage 2 with capabilities_tag: {capabilities_tag}")

    # Calculate target capabilities per area
    target_num_capabilities_per_area = math.ceil(
        cfg.capabilities_cfg.num_capabilities / len(areas)
    )
    num_capabilities_per_area = int(
        target_num_capabilities_per_area
        * (1 + cfg.capabilities_cfg.num_capabilities_buffer)
    )

    # Process each area
    for area in areas:
        # Check if capabilities already exist for this area (resume logic)
        capabilities_path = (
            output_base_dir
            / experiment_id
            / "capabilities"
            / capabilities_tag
            / area.area_id
            / "capabilities.json"
        )

        if is_resume and capabilities_path.exists():
            logger.info(
                f"Skipping area {area.name} ({area.area_id}) - "
                f"capabilities already exist at {capabilities_path}"
            )
            continue

        logger.info(
            f"Generating {num_capabilities_per_area} capabilities for area: "
            f"{area.name} ({area.area_id}) [target: {target_num_capabilities_per_area}]"
        )

        # Generate capabilities
        capabilities = generate_capabilities(
            area=area,
            num_capabilities=num_capabilities_per_area,
            num_capabilities_per_run=cfg.capabilities_cfg.num_gen_capabilities_per_run,
            client=scientist_llm_client,
        )

        # Sort capabilities
        capabilities = sorted(capabilities, key=lambda x: x.name)
        if len(capabilities) < target_num_capabilities_per_area:
            logger.warning(
                f"Only {len(capabilities)} capabilities were created. "
                f"Target number not reached: {target_num_capabilities_per_area}. "
                "It is recommended to increase the buffer."
            )

        # Skip embedding/filtering if no capabilities were generated
        if not capabilities:
            logger.warning(f"No capabilities generated for area {area.name}. Skipping.")
            continue

        # Generate embeddings for capabilities
        embeddings = generate_capability_embeddings(
            capabilities=capabilities,
            embedding_model_name=cfg.embedding_cfg.embedding_model,
            embed_dimensions=cfg.embedding_cfg.embedding_size,
        )

        # Filter capabilities based on embedding similarity
        filtered_capabilities, retained_indices = (
            filter_schema_capabilities_by_embeddings(
                capabilities=capabilities,
                embeddings=embeddings,
                similarity_threshold=cfg.embedding_cfg.filtering_similarity_threshold,
            )
        )

        logger.info(
            f"Capabilities retained after filtering: "
            f"{len(filtered_capabilities)}/{len(capabilities)}"
        )

        for idx, cap in enumerate(filtered_capabilities):
            cap.generation_metadata = {
                "embedding_model": cfg.embedding_cfg.embedding_model,
                "similarity_threshold": cfg.embedding_cfg.filtering_similarity_threshold,
                "original_index": idx,
            }

        # Save capabilities for this area
        metadata = PipelineMetadata(
            experiment_id=experiment_id,
            output_base_dir=str(output_base_dir),
            timestamp=iso_timestamp(),
            input_stage_tag=areas_tag,
            output_stage_tag=capabilities_tag,
            resume=is_resume,
        )

        save_capabilities(filtered_capabilities, metadata, capabilities_path)
        logger.info(
            f"Stage 2: saved {len(filtered_capabilities)} capabilities to "
            f"{capabilities_path}"
        )

    return capabilities_tag
