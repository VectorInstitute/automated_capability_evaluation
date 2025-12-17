"""Stage 1: Area generation.

This stage generates capability areas for the domain.
"""

import logging
from pathlib import Path

from omegaconf import DictConfig

from src.base_stages.generate_areas import generate_areas
from src.schemas.io_utils import load_domain, save_areas
from src.schemas.metadata_schemas import PipelineMetadata
from src.utils import constants
from src.utils.model_client_utils import get_standard_model_client
from src.utils.timestamp_utils import iso_timestamp, timestamp_tag


logger = logging.getLogger(__name__)


def run_stage1(cfg: DictConfig) -> str:
    """Stage 1: Generate capability areas.

    Args:
        cfg: Configuration object

    Returns
    -------
        The areas_tag for this generation
    """
    experiment_id = cfg.exp_cfg.exp_id
    output_base_dir = Path(cfg.global_cfg.output_dir)

    # Load domain from Stage 0 output
    domain_path = output_base_dir / experiment_id / "domain" / "domain.json"
    domain, _ = load_domain(domain_path)

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

    num_areas = cfg.areas_cfg.num_areas
    logger.info(f"Generating {num_areas} capability areas for domain: {domain.name}")

    areas = generate_areas(
        domain=domain,
        num_areas=num_areas,
        num_capabilities_per_area=cfg.capabilities_cfg.num_capabilities // num_areas,
        client=scientist_llm_client,
    )

    # Convert area names to Area objects
    if len(areas) > num_areas:
        logger.warning(
            f"Generated {len(areas)} areas, but only {num_areas} are needed. "
            f"Keeping the first {num_areas} areas."
        )
        areas = areas[:num_areas]

    # Save areas
    areas_tag = timestamp_tag()
    areas_path = output_base_dir / experiment_id / "areas" / areas_tag / "areas.json"
    metadata = PipelineMetadata(
        experiment_id=experiment_id,
        output_base_dir=str(output_base_dir),
        timestamp=iso_timestamp(),
        input_stage_tag=None,
        output_stage_tag=areas_tag,
        resume=False,
    )
    save_areas(areas, metadata, areas_path)

    logger.info(f"Stage 1: saved {len(areas)} areas to {areas_path}")
    return areas_tag
