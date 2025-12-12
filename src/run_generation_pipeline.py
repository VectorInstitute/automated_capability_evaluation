"""Script to generate capabilities (Stage 2) and tasks (Stage 3).

This module keeps the existing behavior but makes the flow explicit:
- Stage 0: setup (config validation, run id, model init)
- Stage 1: create a minimal single area artifact (schema alignment)
- Stage 2: generate capabilities, embed + filter
- Stage 3: generate tasks for retained capabilities

Usage:
    # Run specific stage using Hydra override syntax
    python -m src.run_capability_generation stage=0
    python -m src.run_capability_generation stage=1
    python -m src.run_capability_generation stage=2 areas_tag=_20251211_214002
    python -m src.run_capability_generation stage=3 capabilities_tag=_20251211_220000

    # Run all stages
    python -m src.run_capability_generation stage=all
    python -m src.run_capability_generation  # defaults to "all"
"""

import json
import logging
import math
from datetime import datetime
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from src.generate_capabilities import generate_areas, generate_capabilities
from src.generate_diverse_tasks import generate_diverse_tasks_for_capability
from src.schemas.domain_schemas import Domain
from src.schemas.experiment_schemas import Experiment
from src.schemas.io_utils import (
    load_areas,
    load_capabilities,
    load_domain,
    save_areas,
    save_capabilities,
    save_domain,
    save_experiment,
    save_solution,
    save_validation,
)
from src.schemas.metadata_schemas import PipelineMetadata
from src.schemas.validation_schemas import ValidationResult
from src.utils import constants
from src.utils.capability_management_utils import (
    filter_schema_capabilities_by_embeddings,
)
from src.utils.data_utils import check_cfg
from src.utils.embedding_utils import (
    generate_schema_capabilities_embeddings,
)
from src.utils.model_client_utils import get_standard_model_client


logger = logging.getLogger(__name__)


def stage0_setup(
    cfg: DictConfig,
) -> None:
    """Stage 0: basic setup (config check, run id, base dir, schema files)."""
    check_cfg(cfg, logger)
    exp_id = cfg.exp_cfg.exp_id
    output_base_dir = Path(cfg.global_cfg.output_dir)
    domain_name = cfg.global_cfg.domain
    pipeline_type = cfg.global_cfg.pipeline_type
    logger.info(
        "Stage 0: exp_id=%s | domain=%s | output_base_dir=%s | pipeline_type=%s",
        exp_id,
        domain_name,
        output_base_dir,
        pipeline_type,
    )

    domain_id = "domain_000"
    domain_obj = Domain(
        name=domain_name,
        domain_id=domain_id,
        description=None,
    )

    # Convert entire config to dictionary for experiment configuration
    config_dict = OmegaConf.to_container(cfg, resolve=True)

    experiment_obj = Experiment(
        experiment_id=exp_id,
        domain=domain_name,
        domain_id=domain_id,
        pipeline_type=pipeline_type,
        configuration=config_dict,
    )

    metadata = PipelineMetadata(
        experiment_id=exp_id,
        output_base_dir=str(output_base_dir),
        timestamp=_iso_timestamp(),
        input_stage_tag=None,
        output_stage_tag=None,
        resume=False,
    )
    save_experiment(
        experiment=experiment_obj,
        metadata=metadata,
        output_path=output_base_dir / exp_id / "experiment.json",
    )
    save_domain(
        domain=domain_obj,
        metadata=metadata,
        output_path=output_base_dir / exp_id / "domain" / "domain.json",
    )
    logger.info(
        "Stage 0: saved experiment and domain artifacts under %s", output_base_dir
    )


def _timestamp_tag() -> str:
    """Return a timestamp tag in `_YYYYMMDD_HHMMSS` format."""
    return f"_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"


def _iso_timestamp() -> str:
    """Return an ISO 8601 formatted timestamp with UTC timezone."""
    return datetime.utcnow().isoformat() + "Z"


def stage1_generate_areas(cfg: DictConfig) -> str:
    """
    Stage 1: Generate capability areas using hierarchical method.

    Uses LLM to generate multiple areas within the domain.

    Args:
        cfg: The configuration object

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

    num_areas = cfg.capabilities_cfg.num_areas
    logger.info(f"Generating {num_areas} capability areas for domain: {domain.name}")

    areas = generate_areas(
        domain=domain,
        num_areas=num_areas,
        num_capabilities_per_area=cfg.capabilities_cfg.num_capabilities // num_areas,
        scientist_llm_client=scientist_llm_client,
    )

    # Convert area names to Area objects
    if len(areas) > num_areas:
        logger.warning(
            f"Generated {len(areas)} areas, but only {num_areas} are needed."
            + f"Keeping the first {num_areas} areas."
        )
        areas = areas[:num_areas]

    # Save areas
    areas_tag = _timestamp_tag()
    areas_path = output_base_dir / experiment_id / "areas" / areas_tag / "areas.json"
    metadata = PipelineMetadata(
        experiment_id=experiment_id,
        output_base_dir=str(output_base_dir),
        timestamp=_iso_timestamp(),
        input_stage_tag=None,
        output_stage_tag=areas_tag,
        resume=False,
    )
    save_areas(areas, metadata, areas_path)
    logger.info(f"Stage 1: saved {len(areas)} areas to {areas_path}")
    return areas_tag


def stage2_generate_and_filter_capabilities(
    cfg: DictConfig,
    areas_tag: str,
    capabilities_tag: str = None,
) -> str:
    """Stage 2: generate capabilities, embed, and filter per schema intent.

    Args:
        cfg: The configuration object
        areas_tag: The tag from Stage 1 to load areas from
        capabilities_tag: Optional resume tag. If provided, resumes from existing tag.

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
        capabilities_tag = _timestamp_tag()
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
                f"Skipping area {area.name} ({area.area_id}) - capabilities already exist at {capabilities_path}"
            )
            continue

        logger.info(f"Generating capabilities for area: {area.name} ({area.area_id})")

        # Generate capabilities using existing function
        capabilities = generate_capabilities(
            area=area,
            num_capabilities=num_capabilities_per_area,
            num_capabilities_per_run=cfg.capabilities_cfg.num_gen_capabilities_per_run,
            scientist_llm_client=scientist_llm_client,
        )

        # Sort capabilities
        capabilities = sorted(capabilities, key=lambda x: x.name)
        if len(capabilities) < target_num_capabilities_per_area:
            logger.warning(
                f"Only {len(capabilities)} capabilities were created. "
                f"Target number not reached: {target_num_capabilities_per_area}. "
                "It is recommended to increase the buffer."
            )

        # Generate embeddings for schema capabilities
        embeddings = generate_schema_capabilities_embeddings(
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
            f"Capabilities retained after filtering: {len(filtered_capabilities)}/{len(capabilities)}"
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
            timestamp=_iso_timestamp(),
            input_stage_tag=areas_tag,
            output_stage_tag=capabilities_tag,
            resume=is_resume,
        )

        save_capabilities(filtered_capabilities, metadata, capabilities_path)
        logger.info(
            f"Stage 2: saved {len(filtered_capabilities)} capabilities to {capabilities_path}"
        )

    return capabilities_tag


def stage3_generate_tasks(
    cfg: DictConfig,
    capabilities_tag: str,
    tasks_tag: str = None,
) -> str:
    """Stage 3: Generate diverse tasks with solutions for each capability.

    Generates tasks using the diverse task generation method and creates
    TaskSolution objects with the correct answer and explanation.

    Args:
        cfg: The configuration object
        capabilities_tag: The tag from Stage 2 to load capabilities from
        tasks_tag: Optional resume tag. If provided, resumes from existing tag.

    Returns
    -------
        The tasks_tag for this generation
    """
    experiment_id = cfg.exp_cfg.exp_id
    output_base_dir = Path(cfg.global_cfg.output_dir)

    # Determine tasks tag (resume or new)
    is_resume = tasks_tag is not None
    if is_resume:
        logger.info(f"Resuming Stage 3 with tasks_tag: {tasks_tag}")
    else:
        tasks_tag = _timestamp_tag()
        logger.info(f"Starting new Stage 3 with tasks_tag: {tasks_tag}")

    # Initialize scientist LLM client using task_generation config
    scientist_llm_gen_cfg = dict(cfg.scientist_llm.generation_cfg.task_generation)
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

    # Get task generation parameters from config
    tasks_per_blueprint = cfg.task_generation_cfg.get("tasks_per_blueprint", 3)
    min_subtopics = cfg.task_generation_cfg.get("min_subtopics", 3)
    max_subtopics = cfg.task_generation_cfg.get("max_subtopics", 8)

    # Find all area directories under capabilities/<capabilities_tag>/
    capabilities_base_dir = (
        output_base_dir / experiment_id / "capabilities" / capabilities_tag
    )
    area_dirs = [d for d in capabilities_base_dir.iterdir() if d.is_dir()]

    logger.info(f"Found {len(area_dirs)} area directories")

    # Process each area
    for area_dir in area_dirs:
        area_id = area_dir.name
        logger.info(f"Processing area: {area_id}")

        # Load capabilities for this area
        capabilities_path = area_dir / "capabilities.json"
        capabilities, _ = load_capabilities(capabilities_path)
        logger.info(f"Loaded {len(capabilities)} capabilities from {area_id}")

        # Process each capability
        for capability in capabilities:
            capability_id = capability.capability_id

            # Check if task solutions already exist for this capability (resume logic)
            solutions_dir = (
                output_base_dir
                / experiment_id
                / "task_solutions"
                / tasks_tag
                / area_id
                / capability_id
            )

            if (
                is_resume
                and solutions_dir.exists()
                and any(solutions_dir.glob("*.json"))
            ):
                logger.info(
                    f"Skipping {area_id}/{capability_id} - task solutions already exist at {solutions_dir}"
                )
                continue

            logger.info(
                f"Generating tasks for capability: {capability.name} ({area_id}/{capability_id})"
            )

            try:
                # Generate diverse tasks with solutions
                task_solutions = generate_diverse_tasks_for_capability(
                    capability=capability,
                    tasks_per_blueprint=tasks_per_blueprint,
                    client=scientist_llm_client,
                    min_subtopics=min_subtopics,
                    max_subtopics=max_subtopics,
                )

                logger.info(
                    f"Generated {len(task_solutions)} task solutions for {capability.name}"
                )

                # Save each task solution
                metadata = PipelineMetadata(
                    experiment_id=experiment_id,
                    output_base_dir=str(output_base_dir),
                    timestamp=_iso_timestamp(),
                    input_stage_tag=capabilities_tag,
                    output_stage_tag=tasks_tag,
                    resume=is_resume,
                )

                # Save task solutions in task_solutions directory
                for task_solution in task_solutions:
                    solution_path = (
                        output_base_dir
                        / experiment_id
                        / "task_solutions"
                        / tasks_tag
                        / area_id
                        / capability_id
                        / f"{task_solution.task_id}.json"
                    )
                    save_solution(task_solution, metadata, solution_path)

                logger.info(
                    f"Stage 3: saved {len(task_solutions)} task solutions to "
                    f"task_solutions/{tasks_tag}/{area_id}/{capability_id}/"
                )

            except Exception as e:
                logger.error(
                    f"Error generating tasks for {area_id}/{capability_id}: {e}",
                    exc_info=True,
                )
                # Continue with next capability instead of failing completely
                continue

    return tasks_tag


def stage5_validate_tasks(
    cfg: DictConfig,
    solution_tag: str,
    validation_tag: str = None,
) -> str:
    """Stage 5: Validate generated task solutions.

    Args:
        cfg: The configuration object
        solution_tag: The tag from Stage 3 to load task solutions from
        validation_tag: Optional resume tag. If provided, resumes from existing tag.

    Returns
    -------
        The validation_tag for this validation
    """
    experiment_id = cfg.exp_cfg.exp_id
    output_base_dir = Path(cfg.global_cfg.output_dir)

    # Determine validation tag (resume or new)
    is_resume = validation_tag is not None
    if is_resume:
        logger.info(f"Resuming Stage 5 with validation_tag: {validation_tag}")
    else:
        validation_tag = _timestamp_tag()
        logger.info(f"Starting new Stage 5 with validation_tag: {validation_tag}")

    # Initialize validator LLM client
    validator_llm_gen_cfg = dict(
        cfg.get("validator_llm", {})
        .get("generation_cfg", {})
        .get("task_validation", {})
    )
    validator_llm_name = cfg.get("validator_llm", {}).get(
        "name", cfg.scientist_llm.name
    )
    validator_llm_client = get_standard_model_client(
        validator_llm_name,
        seed=validator_llm_gen_cfg.get("seed", cfg.exp_cfg.seed),
        temperature=validator_llm_gen_cfg.get(
            "temperature", constants.DEFAULT_TEMPERATURE
        ),
        max_tokens=validator_llm_gen_cfg.get(
            "max_tokens", constants.DEFAULT_MAX_TOKENS
        ),
    )

    # Get validation parameters from config
    pass_threshold = cfg.get("task_verification_cfg", {}).get("pass_threshold", 0.8)
    strict_mode = cfg.get("task_verification_cfg", {}).get("strict_mode", False)

    # Find all task_solutions directories
    task_solutions_base_dir = (
        output_base_dir / experiment_id / "task_solutions" / solution_tag
    )

    if not task_solutions_base_dir.exists():
        logger.error(f"Task solutions directory not found: {task_solutions_base_dir}")
        return validation_tag

    # Find all area directories
    area_dirs = [d for d in task_solutions_base_dir.iterdir() if d.is_dir()]
    logger.info(f"Found {len(area_dirs)} area directories")

    # Process each area
    for area_dir in area_dirs:
        area_id = area_dir.name
        logger.info(f"Processing area: {area_id}")

        # Find all capability directories
        capability_dirs = [d for d in area_dir.iterdir() if d.is_dir()]

        for capability_dir in capability_dirs:
            capability_id = capability_dir.name

            # Find all task solution files
            task_solution_files = list(capability_dir.glob("*.json"))

            if not task_solution_files:
                logger.warning(f"No task solutions found in {area_id}/{capability_id}")
                continue

            logger.info(
                f"Validating {len(task_solution_files)} task solutions for {area_id}/{capability_id}"
            )

            for task_solution_file in task_solution_files:
                task_id = task_solution_file.stem

                # Check if validation already exists (resume logic)
                validation_path = (
                    output_base_dir
                    / experiment_id
                    / "validations"
                    / validation_tag
                    / area_id
                    / capability_id
                    / f"{task_id}.json"
                )

                if is_resume and validation_path.exists():
                    logger.info(
                        f"Skipping {area_id}/{capability_id}/{task_id} - validation already exists"
                    )
                    continue

                try:
                    # Load task solution
                    with open(task_solution_file, "r") as f:
                        task_solution_data = json.load(f)

                    # Extract necessary information
                    task_text = task_solution_data.get("task", "")
                    solution = task_solution_data.get("solution", "")
                    reasoning = task_solution_data.get("reasoning", "")
                    generation_metadata = task_solution_data.get(
                        "generation_metadata", {}
                    )

                    # For validation, we need to check if the task is well-formed
                    # Simple validation: check if task has content and solution exists
                    verification = (
                        len(task_text.strip()) > 0 and len(solution.strip()) > 0
                    )
                    feedback = (
                        "Task validation passed"
                        if verification
                        else "Task validation failed: missing content"
                    )

                    # Create Task object for ValidationResult
                    from src.schemas.area_schemas import Area
                    from src.schemas.capability_schemas import Capability
                    from src.schemas.domain_schemas import Domain
                    from src.schemas.task_schemas import Task

                    domain = Domain(
                        name=task_solution_data.get("domain", ""),
                        domain_id=task_solution_data.get("domain_id", ""),
                        description="",
                    )
                    area = Area(
                        name=task_solution_data.get("area", ""),
                        area_id=task_solution_data.get("area_id", ""),
                        domain=domain,
                        description=task_solution_data.get("area_description", ""),
                    )
                    capability = Capability(
                        name=task_solution_data.get("capability", ""),
                        capability_id=task_solution_data.get("capability_id", ""),
                        area=area,
                        description=task_solution_data.get(
                            "capability_description", ""
                        ),
                    )
                    task = Task(
                        task_id=task_id,
                        task=task_text,
                        capability=capability,
                    )

                    # Create ValidationResult
                    validation_result = ValidationResult(
                        task_id=task_id,
                        task=task_text,
                        verification=verification,
                        feedback=feedback,
                        task_obj=task,
                        generation_metadata={
                            "method": "simple_validation",
                            "pass_threshold": pass_threshold,
                            "strict_mode": strict_mode,
                            **generation_metadata,
                        },
                    )

                    # Save validation
                    metadata = PipelineMetadata(
                        experiment_id=experiment_id,
                        output_base_dir=str(output_base_dir),
                        timestamp=_iso_timestamp(),
                        input_stage_tag=solution_tag,
                        output_stage_tag=validation_tag,
                        resume=is_resume,
                    )

                    save_validation(validation_result, metadata, validation_path)
                    logger.info(
                        f"Validated {task_id}: {'✓ PASS' if verification else '✗ FAIL'}"
                    )

                except Exception as e:
                    logger.error(
                        f"Error validating {area_id}/{capability_id}/{task_id}: {e}",
                        exc_info=True,
                    )
                    continue

    logger.info(f"Stage 5 completed. Validation tag: {validation_tag}")
    return validation_tag


@hydra.main(version_base=None, config_path="cfg", config_name="run_cfg")
def main(cfg: DictConfig) -> None:
    """Run specific pipeline stages based on configuration.

    Stage 0: Experiment and domain setup
    Stage 1: Area generation
    Stage 2: Capability generation and filtering
    Stage 3: Task generation with solutions
    Stage 5: Task validation
    "all": Run all stages sequentially
    """
    # Suppress httpx and autogen_core INFO logs
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("autogen_core.events").setLevel(logging.WARNING)

    # Get stage from config (can be overridden via command line)
    stage = cfg.get("stage", "all")

    # Convert string to int if numeric
    if isinstance(stage, str) and stage.isdigit():
        stage = int(stage)

    logger.info(f"Running stage: {stage}")

    # Track tags across stages
    areas_tag = cfg.get("areas_tag", None)
    capabilities_tag = cfg.get("capabilities_tag", None)
    solution_tag = cfg.get("solution_tag", None)
    validation_tag = None

    if stage == 0 or stage == "all":
        logger.info("=" * 60)
        logger.info("STAGE 0: Experiment and Domain Setup")
        logger.info("=" * 60)
        stage0_setup(cfg)
        if stage == 0:
            return

    if stage == 1 or stage == "all":
        logger.info("=" * 60)
        logger.info("STAGE 1: Area Generation (Hierarchical)")
        logger.info("=" * 60)
        areas_tag = stage1_generate_areas(cfg)
        logger.info("Stage 1 areas tag: %s", areas_tag)
        if stage == 1:
            return

    if stage == 2 or stage == "all":
        logger.info("=" * 60)
        logger.info("STAGE 2: Capability Generation and Filtering")
        logger.info("=" * 60)

        # When running stage 2 standalone, areas_tag must be provided
        if stage == 2 and not areas_tag:
            logger.error("areas_tag is required when running stage 2 standalone")
            logger.error(
                "Usage: python -m src.run_capability_generation stage=2 areas_tag=_YYYYMMDD_HHMMSS"
            )
            logger.error(
                "Optional: capabilities_tag=_YYYYMMDD_HHMMSS to resume from existing run"
            )
            return

        # Check if resuming
        resume_capabilities_tag = (
            cfg.get("capabilities_tag", None) if stage == 2 else None
        )
        if resume_capabilities_tag:
            logger.info(
                f"Resume mode: Will skip areas that already have capabilities in tag {resume_capabilities_tag}"
            )

        capabilities_tag = stage2_generate_and_filter_capabilities(
            cfg=cfg,
            areas_tag=areas_tag,
            capabilities_tag=resume_capabilities_tag,
        )
        logger.info("Stage 2 capabilities tag: %s", capabilities_tag)
        if stage == 2:
            return

    if stage == 3 or stage == "all":
        logger.info("=" * 60)
        logger.info("STAGE 3: Diverse Task Generation")
        logger.info("=" * 60)

        # When running stage 3 standalone, capabilities_tag must be provided
        if stage == 3 and not capabilities_tag:
            logger.error("capabilities_tag is required when running stage 3 standalone")
            logger.error(
                "Usage: python -m src.run_capability_generation stage=3 capabilities_tag=_YYYYMMDD_HHMMSS"
            )
            logger.error(
                "Optional: tasks_tag=_YYYYMMDD_HHMMSS to resume from existing run"
            )
            return

        # Check if resuming
        resume_tasks_tag = cfg.get("tasks_tag", None) if stage == 3 else None
        if resume_tasks_tag:
            logger.info(
                f"Resume mode: Will skip capabilities that already have tasks in tag {resume_tasks_tag}"
            )

        solution_tag = stage3_generate_tasks(
            cfg=cfg,
            capabilities_tag=capabilities_tag,
            tasks_tag=resume_tasks_tag,
        )
        logger.info("Stage 3 solution tag: %s", solution_tag)
        if stage == 3:
            return

    if stage == 5 or stage == "all":
        logger.info("=" * 60)
        logger.info("STAGE 5: Task Validation")
        logger.info("=" * 60)

        # When running stage 5 standalone, solution_tag must be provided
        if stage == 5 and not solution_tag:
            logger.error("solution_tag is required when running stage 5 standalone")
            logger.error(
                "Usage: python -m src.run_capability_generation stage=5 solution_tag=_YYYYMMDD_HHMMSS"
            )
            logger.error(
                "Optional: validation_tag=_YYYYMMDD_HHMMSS to resume from existing run"
            )
            return

        # Check if resuming
        resume_validation_tag = cfg.get("validation_tag", None) if stage == 5 else None
        if resume_validation_tag:
            logger.info(
                f"Resume mode: Will skip tasks that already have validations in tag {resume_validation_tag}"
            )

        validation_tag = stage5_validate_tasks(
            cfg=cfg,
            solution_tag=solution_tag,
            validation_tag=resume_validation_tag,
        )
        logger.info("Stage 5 validation tag: %s", validation_tag)
        if stage == 5:
            return


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    main()
