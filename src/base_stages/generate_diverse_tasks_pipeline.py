"""Generate diverse tasks using multi-dimensional approach."""

import logging
from typing import List

from autogen_core.models import ChatCompletionClient

from src.base_stages.extract_subtopics import extract_subtopics
from src.base_stages.find_combinations import find_valid_combinations
from src.base_stages.generate_blueprints import generate_blueprints
from src.base_stages.generate_tasks_from_blueprints import (
    generate_tasks_from_blueprints,
)
from src.schemas.capability_schemas import Capability
from src.schemas.task_schemas import Task


logger = logging.getLogger(__name__)


def generate_diverse_tasks_for_capability(
    capability: Capability,
    tasks_per_blueprint: int,
    client: ChatCompletionClient,
    min_subtopics: int = 3,
    max_subtopics: int = 8,
) -> List[Task]:
    """Generate diverse tasks for a single capability.

    This function generates Task objects (questions with 4 options). The
    correct answer is NOT determined here — that happens in Stage 4
    (Solution Generation) where an LLM solves each task.

    Args:
        capability: Capability object
        tasks_per_blueprint: Number of tasks to generate per blueprint
        client: ChatCompletionClient for API calls
        min_subtopics: Minimum number of subtopics to generate
        max_subtopics: Maximum number of subtopics to generate

    Returns
    -------
        List of Task objects (questions + options, no answers)
    """
    logger.info(f"Generating diverse tasks for capability: {capability.name}")

    logger.info("Step 1: Extracting sub-topics")
    subtopics = extract_subtopics(capability, client, min_subtopics, max_subtopics)
    logger.info(f"Extracted {len(subtopics)} sub-topics")

    logger.info("Step 2: Finding valid combinations")
    combinations = find_valid_combinations(capability, subtopics, client)
    logger.info(f"Found {len(combinations)} valid combinations")

    logger.info("Step 3: Generating blueprints")
    blueprints = generate_blueprints(capability, combinations, client)
    logger.info(f"Generated {len(blueprints)} blueprints")

    logger.info("Step 4: Generating tasks")
    tasks = generate_tasks_from_blueprints(
        capability, blueprints, client, tasks_per_blueprint
    )
    logger.info(f"Generated {len(tasks)} tasks")

    return tasks
