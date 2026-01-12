"""Base (non-agentic) pipeline stages and utilities.

This module contains all base pipeline stages and the utilities they use:

Stages:
- stage0_setup: Experiment and domain setup
- stage1_areas: Area generation
- stage2_capabilities: Capability generation and filtering
- stage3_tasks: Task generation
- stage4_solutions: Solution generation
- stage5_validation: Task validation

Utilities:
- generate_areas: Area generation using LLM
- generate_capabilities: Capability generation using LLM
- generate_diverse_tasks_pipeline: Orchestrates subtopic→combination→blueprint→task
  pipeline
- generate_tasks_from_blueprints: Task (question + options) generation from blueprints
- solve_tasks: Task solving to determine correct answers
- validate_tasks: Task validation

Supporting modules:
- task_constants: Bloom's taxonomy, difficulty levels
- task_dataclasses: SubTopic, Combination, Blueprint, etc.
- task_prompts: All LLM prompts for task generation pipeline
- extract_subtopics: Sub-topic extraction
- find_combinations: Valid combination finding
- generate_blueprints: Blueprint generation
"""

from src.base_stages.stage0_setup import run_stage0
from src.base_stages.stage1_areas import run_stage1
from src.base_stages.stage2_capabilities import run_stage2
from src.base_stages.stage3_tasks import run_stage3
from src.base_stages.stage4_solutions import run_stage4
from src.base_stages.stage5_validation import run_stage5


__all__ = [
    "run_stage0",
    "run_stage1",
    "run_stage2",
    "run_stage3",
    "run_stage4",
    "run_stage5",
]
