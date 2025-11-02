"""Standalone script for generating diverse tasks for a single capability."""

import argparse
import json
import logging
import os
from dataclasses import asdict
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any

import yaml
from diverse_task_dataclasses import (
    Blueprint,
    Capability,
    Combination,
    SubTopic,
    Task,
    VerificationResult,
)
from extract_subtopics import extract_subtopics
from find_combinations import find_valid_combinations
from generate_blueprints import generate_blueprints
from model_utils import call_model
from openai import OpenAI
from verify_tasks import verify_tasks

from generate_tasks import generate_tasks


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class DiverseTaskGenerator:
    """Generate diverse tasks for a capability using multi-dimensional approach."""

    def __init__(
        self,
        capability_dict: dict,
        config: dict,
    ) -> None:
        """Initialize the diverse task generator."""
        # Extract example tasks from capability_data if present
        example_tasks = (
            capability_dict.get("capability_data", [])[:3]
            if "capability_data" in capability_dict
            else []
        )

        self.capability = Capability(
            name=capability_dict["capability_name"],
            description=capability_dict["capability_description"],
            domain=capability_dict["capability_domain"],
            area=capability_dict.get("capability_area"),
            example_tasks=example_tasks,
        )

        # Store configuration
        self.config = config

        # Use config values
        self.model_name = self.config["model"]["name"]
        self.temperature = self.config["model"]["temperature"]
        self.max_tokens = self.config["model"]["max_tokens"]
        self.output_dir = Path(self.config["output"]["base_dir"])

        # Initialize OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        self.client = OpenAI(api_key=api_key)

        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_output_dir = self.output_dir / f"{self.capability.name}_{timestamp}"
        self.run_output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("=" * 80)
        logger.info(f"Initialized DiverseTaskGenerator for: {self.capability.name}")
        logger.info(f"Model: {self.model_name}")
        logger.info(f"Temperature: {self.temperature}")
        logger.info(f"Max tokens: {self.max_tokens}")
        logger.info(f"Output directory: {self.run_output_dir}")
        logger.info("=" * 80)

        # Create API caller with pre-configured parameters
        self._call_api = partial(
            call_model,
            self.client,
            model_name=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

    def _save_json(self, filename: str, data_key: str, data: Any) -> Path:
        """Save data to JSON file."""
        output_file = self.run_output_dir / filename
        # Convert dataclass objects to dicts if needed
        if data and hasattr(
            data[0] if isinstance(data, list) else data, "__dataclass_fields__"
        ):
            data = (
                [asdict(item) for item in data]
                if isinstance(data, list)
                else asdict(data)
            )

        with open(output_file, "w") as f:
            json.dump({data_key: data} if data_key else data, f, indent=2)
        logger.info(f"Saved to: {output_file}")
        return output_file

    def extract_and_save_subtopics(self) -> list[SubTopic]:
        """Extract sub-topics and save results."""
        subtopics = extract_subtopics(self.capability, self._call_api)
        self._save_json("subtopics.json", "sub_topics", subtopics)
        return subtopics

    def find_and_save_combinations(
        self, subtopics: list[SubTopic]
    ) -> list[Combination]:
        """Find valid combinations and save results."""
        combinations = find_valid_combinations(
            self.capability, subtopics, self._call_api
        )
        self._save_json("combinations.json", "valid_combinations", combinations)
        return combinations

    def generate_and_save_blueprints(
        self, combinations: list[Combination]
    ) -> list[Blueprint]:
        """Generate blueprints and save results."""
        blueprints = generate_blueprints(
            self.capability, combinations, self._call_api, self.config
        )
        self._save_json("blueprints.json", "blueprints", blueprints)
        return blueprints

    def generate_and_save_tasks(self, blueprints: list[Blueprint]) -> list[Task]:
        """Generate tasks and save results."""
        tasks_per_blueprint = self.config["generation"]["tasks_per_blueprint"]
        tasks = generate_tasks(
            self.capability, blueprints, self._call_api, tasks_per_blueprint
        )
        self._save_json("tasks.json", "tasks", tasks)
        return tasks

    def verify_and_save_tasks(
        self, tasks: list[Task], blueprints: list[Blueprint]
    ) -> VerificationResult:
        """Verify tasks and save results."""
        verification = verify_tasks(self.capability, tasks, blueprints, self._call_api)
        self._save_json("verification.json", None, verification)
        return verification

    def run_full_pipeline(self) -> dict:
        """Run the complete diverse task generation pipeline."""
        logger.info("=" * 80)
        logger.info("Starting Diverse Task Generation Pipeline")
        logger.info(f"Capability: {self.capability.name}")
        logger.info(f"Model: {self.model_name}")
        logger.info("=" * 80)

        # Extract sub-topics
        subtopics = self.extract_and_save_subtopics()

        # Find valid combinations
        combinations = self.find_and_save_combinations(subtopics)

        # Generate blueprints
        blueprints = self.generate_and_save_blueprints(combinations)

        # Generate tasks
        tasks = self.generate_and_save_tasks(blueprints)

        # Verify tasks
        verification = self.verify_and_save_tasks(tasks, blueprints)

        # Compile final results
        results = {
            "capability_name": self.capability.name,
            "capability_description": self.capability.description,
            "capability_domain": self.capability.domain,
            "model_name": self.model_name,
            "timestamp": datetime.now().isoformat(),
            "subtopics": [asdict(st) for st in subtopics],
            "combinations": [asdict(c) for c in combinations],
            "blueprints": [asdict(bp) for bp in blueprints],
            "tasks": [asdict(t) for t in tasks],
            "verification": verification,
        }

        # Save final results
        self._save_json("final_results.json", None, results)

        logger.info("=" * 80)
        logger.info("Pipeline Complete!")
        logger.info(f"All results saved to: {self.run_output_dir}")
        logger.info("=" * 80)

        return results


def load_capability_from_json(capability_json_path: str) -> dict:
    """Load capability information from a JSON file."""
    with open(capability_json_path, "r") as f:
        return json.load(f)


def main() -> None:
    """Generate diverse tasks for a single capability."""
    parser = argparse.ArgumentParser(
        description="Generate diverse tasks for a capability from JSON file"
    )
    parser.add_argument(
        "--capability-json-path",
        type=str,
        help="Path to capability JSON file (default: from config file)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        help="OpenAI model name (default: from config file)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory (default: from config file)",
    )

    args = parser.parse_args()

    # Load config
    config_file = Path(__file__).parent / "diverse_task_config.yaml"
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    # Override config with command-line arguments
    if args.model_name:
        config["model"]["name"] = args.model_name
    if args.output_dir:
        config["output"]["base_dir"] = args.output_dir
    if args.capability_json_path:
        config["input"]["capability_json_path"] = args.capability_json_path

    logger.info(f"Loading capability from: {config['input']['capability_json_path']}")
    capability_dict = load_capability_from_json(config["input"]["capability_json_path"])

    # Initialize and run generator
    generator = DiverseTaskGenerator(
        capability_dict=capability_dict,
        config=config,
    )
    generator.run_full_pipeline()

    logger.info("Done!")


if __name__ == "__main__":
    main()
