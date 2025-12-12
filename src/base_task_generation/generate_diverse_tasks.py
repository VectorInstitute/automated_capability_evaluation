"""Generate diverse tasks using base task generation multi-dimensional approach.

This module adapts the base task generator for use in Stage 3
of the standardized pipeline.
"""

import logging
from typing import List

from autogen_core.models import ChatCompletionClient

# Import base task generation components
from src.base_task_generation.extract_subtopics import extract_subtopics
from src.base_task_generation.find_combinations import find_valid_combinations
from src.base_task_generation.generate_blueprints import generate_blueprints
from src.base_task_generation.generate_tasks import generate_tasks

# Import schema objects
from src.schemas.capability_schemas import Capability
from src.schemas.solution_schemas import TaskSolution


logger = logging.getLogger(__name__)


def generate_diverse_tasks_for_capability(
    capability: Capability,
    tasks_per_blueprint: int,
    client: ChatCompletionClient,
    min_subtopics: int = 3,
    max_subtopics: int = 8,
) -> List[TaskSolution]:
    """Generate diverse tasks with solutions for a single capability.

    Uses multi-dimensional task generation approach.

    Args:
        capability: Schema Capability object
        tasks_per_blueprint: Number of tasks to generate per blueprint
        client: ChatCompletionClient from get_standard_model_client
        min_subtopics: Minimum number of subtopics to generate
        max_subtopics: Maximum number of subtopics to generate

    Returns
    -------
        List of schema TaskSolution objects
    """
    logger.info(f"Generating diverse tasks for capability: {capability.name}")

    # Step 1: Extract sub-topics
    logger.info("Step 1: Extracting sub-topics")
    subtopics = extract_subtopics(capability, client, min_subtopics, max_subtopics)
    logger.info(f"Extracted {len(subtopics)} sub-topics")

    # Step 2: Find valid combinations
    logger.info("Step 2: Finding valid combinations")
    combinations = find_valid_combinations(capability, subtopics, client)
    logger.info(f"Found {len(combinations)} valid combinations")

    # Step 3: Generate blueprints
    logger.info("Step 3: Generating blueprints")
    blueprints = generate_blueprints(capability, combinations, client)
    logger.info(f"Generated {len(blueprints)} blueprints")

    # Step 4: Generate tasks (returns schema TaskSolution objects directly)
    logger.info("Step 4: Generating tasks with solutions")
    task_solutions = generate_tasks(capability, blueprints, client, tasks_per_blueprint)
    logger.info(f"Generated {len(task_solutions)} task solutions")

    return task_solutions
