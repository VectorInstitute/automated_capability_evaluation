"""Area categorization system for static datasets using LLM with resume capability."""

import glob
import json
import logging
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import hydra
from omegaconf import DictConfig

from src.model import Model
from wikipedia.prompts import (
    SYSTEM_PROMPT_MATH_TAXONOMIST,
    get_area_categorization_prompt,
)


logger = logging.getLogger(__name__)


@dataclass
class CapabilityInfo:
    """Data class to hold capability information."""

    name: str
    description: str
    area: str
    domain: str = "math"


@dataclass
class AreaInfo:
    """Data class to hold area information."""

    name: str
    capabilities: List[CapabilityInfo]


class DatasetQuestionCategorizer:
    """Class to categorize questions from selected dataset using two-step LLM approach."""

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.areas: List[AreaInfo] = []
        self.capabilities_by_area: Dict[str, List[CapabilityInfo]] = {}
        self.llm_model: Optional[Model] = None

        # Initialize LLM model
        try:
            self.llm_model = Model(
                model_name=cfg.llm_cfg.model_name,
                model_provider=cfg.llm_cfg.model_provider,
            )
            logger.info(f"Initialized LLM model: {cfg.llm_cfg.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize LLM model: {e}")
            raise e

    def extract_areas_and_capabilities_from_generated(
        self, generated_dir: str
    ) -> Tuple[List[AreaInfo], Dict[str, List[CapabilityInfo]]]:
        """Extract all areas and capabilities from the generated capabilities directory."""
        logger.info("Extracting areas and capabilities from generated capabilities...")

        areas: List[AreaInfo] = []
        capabilities_by_area: Dict[str, List[CapabilityInfo]] = {}

        # Get all capability directories (handle nested structure like math/<capability_name>/)
        capability_dirs = glob.glob(os.path.join(generated_dir, "*/"))
        print(f"Found {len(capability_dirs)} capability directories")

        for cap_dir in capability_dirs:
            capability_json_path = os.path.join(cap_dir, "capability.json")
            if os.path.exists(capability_json_path):
                try:
                    with open(capability_json_path, "r") as f:
                        cap_data = json.load(f)

                    print(
                        f"Loaded capability data: {cap_data.get('capability_name', ''), cap_data.get('capability_description', ''), cap_data.get('capability_area', 'Unknown')}"
                    )
                    capability = CapabilityInfo(
                        name=cap_data.get("capability_name", ""),
                        description=cap_data.get("capability_description", ""),
                        area=cap_data.get("capability_area", "Unknown"),
                    )

                    # Group capabilities by area
                    if capability.area not in capabilities_by_area:
                        capabilities_by_area[capability.area] = []
                    capabilities_by_area[capability.area].append(capability)

                except Exception as e:
                    logger.warning(
                        f"Error loading capability from {capability_json_path}: {e}"
                    )

        # Create area objects
        for area_name, capabilities in capabilities_by_area.items():
            area = AreaInfo(name=area_name, capabilities=capabilities)
            areas.append(area)

        total_caps = sum(len(caps) for caps in capabilities_by_area.values())
        logger.info(
            f"Extracted {len(areas)} areas with {total_caps} total capabilities"
        )
        return areas, capabilities_by_area

    def extract_areas_and_capabilities_from_wikipedia(
        self, wikipedia_dir: str
    ) -> Tuple[List[AreaInfo], Dict[str, List[CapabilityInfo]]]:
        """Extract all areas and capabilities from the Wikipedia pages directory containing individual JSON files."""
        logger.info(
            f"Extracting areas and capabilities from Wikipedia pages directory: {wikipedia_dir}"
        )

        areas: List[AreaInfo] = []
        capabilities_by_area: Dict[str, List[CapabilityInfo]] = {}

        try:
            # Get all JSON files in the Wikipedia pages directory
            json_files = glob.glob(os.path.join(wikipedia_dir, "*.json"))
            logger.info(f"Found {len(json_files)} Wikipedia capability files")

            for json_file in json_files:
                try:
                    with open(json_file, "r", encoding="utf-8") as f:
                        wikipedia_data = json.load(f)

                    # Extract capability information from the individual JSON file
                    capability_name = wikipedia_data.get("capability_name", "")
                    description = wikipedia_data.get("description", "")
                    area = wikipedia_data.get("area", "Unknown")

                    if capability_name and description:
                        capability = CapabilityInfo(
                            name=capability_name,
                            description=description,
                            area=area,
                            domain="math",
                        )

                        # Group capabilities by area
                        if area not in capabilities_by_area:
                            capabilities_by_area[area] = []
                        capabilities_by_area[area].append(capability)

                except Exception as e:
                    logger.warning(
                        f"Error loading Wikipedia capability from {json_file}: {e}"
                    )
                    continue

            # Create area objects
            for area_name, capabilities in capabilities_by_area.items():
                if capabilities:  # Only create areas that have capabilities
                    area = AreaInfo(name=area_name, capabilities=capabilities)
                    areas.append(area)
                    logger.info(
                        f"Loaded area '{area_name}' with {len(capabilities)} capabilities"
                    )

            total_caps = sum(len(caps) for caps in capabilities_by_area.values())
            logger.info(
                f"Extracted {len(areas)} areas with {total_caps} total capabilities from Wikipedia pages"
            )

        except Exception as e:
            logger.error(
                f"Error loading Wikipedia capabilities from {wikipedia_dir}: {e}"
            )
            raise e

        return areas, capabilities_by_area

    def extract_areas_and_capabilities(
        self, generated_dir: Optional[str] = None, wikipedia_dir: Optional[str] = None
    ) -> Tuple[List[AreaInfo], Dict[str, List[CapabilityInfo]]]:
        """Extract areas and capabilities using the configured method."""
        extraction_method = getattr(self.cfg, "categorization_cfg", {}).get(
            "extraction_method", "generated"
        )

        if extraction_method == "wikipedia":
            if not wikipedia_dir:
                wikipedia_dir = self.cfg.data_cfg.wikipedia_dir
            return self.extract_areas_and_capabilities_from_wikipedia(wikipedia_dir)
        if extraction_method == "generated":
            if not generated_dir:
                generated_dir = self.cfg.data_cfg.generated_dir
            return self.extract_areas_and_capabilities_from_generated(generated_dir)
        raise ValueError(
            f"Unknown extraction method: {extraction_method}. Must be 'generated' or 'wikipedia'"
        )

    @staticmethod
    def _normalize_text(text: str) -> str:
        return " ".join(text.strip().lower().split())

    def _find_matching_area_key(self, predicted_area: str) -> str:
        """Find the exact area key that matches the predicted area name."""
        predicted_normalized = self._normalize_text(predicted_area)

        # First try exact match
        if predicted_area in self.capabilities_by_area:
            return predicted_area

        # Try normalized match
        area_key: str
        for area_key in self.capabilities_by_area.keys():
            if self._normalize_text(area_key) == predicted_normalized:
                return area_key

        # Try partial match (contains)
        for area_key in self.capabilities_by_area.keys():
            if (
                predicted_normalized in self._normalize_text(area_key)
                or self._normalize_text(area_key) in predicted_normalized
            ):
                return area_key

        # If no match found, return the original prediction
        logger.warning(
            f"No matching area key found for '{predicted_area}'. Available keys: {list(self.capabilities_by_area.keys())}"
        )
        return predicted_area

    @classmethod
    def _select_best_match(cls, response_text: str, allowed_names: List[str]) -> str:
        """Select the best matching allowed name from a free-form response.

        Uses a hierarchical matching strategy:
        1. Exact normalized match
        2. Substring match (contains or starts with)
        3. Return "Unknown" if no match found
        """
        if not response_text or not allowed_names:
            return "Unknown"

        # Normalize response once
        normalized_response = cls._normalize_text(response_text)

        # Pre-compute normalized names for efficiency
        normalized_names = {cls._normalize_text(name): name for name in allowed_names}

        # 1. Exact match (highest priority)
        if normalized_response in normalized_names:
            return normalized_names[normalized_response]

        # 2. Substring match (contains or starts with)
        for norm_name, original_name in normalized_names.items():
            if (
                normalized_response.startswith(norm_name)
                or norm_name in normalized_response
            ):
                return original_name

        # 3. No match found
        return "Unknown"

    def load_gsm8k_questions(self, jsonl_path: str) -> List[Dict[str, Any]]:
        """Load GSM8K questions from JSONL file."""
        logger.info(f"Loading GSM8K questions from {jsonl_path}...")

        questions = []
        with open(jsonl_path, "r") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    question_data = json.loads(line.strip())
                    question_data["line_number"] = line_num
                    questions.append(question_data)
                except json.JSONDecodeError as e:
                    logger.warning(f"Error parsing line {line_num}: {e}")

        logger.info(f"Loaded {len(questions)} questions")
        return questions

    def load_math_questions(self, math_data_dir: str) -> List[Dict[str, Any]]:
        """Load MATH dataset questions from a directory containing JSON files (recursive)."""
        logger.info(f"Loading MATH questions from {math_data_dir}...")
        questions: List[Dict[str, Any]] = []

        json_files = glob.glob(os.path.join(math_data_dir, "**/*.json"), recursive=True)
        logger.info(f"Found {len(json_files)} JSON files in MATH dataset")

        for json_file in json_files:
            try:
                with open(json_file, "r") as f:
                    problem_data = json.load(f)
                question_data = {
                    "question": problem_data.get("problem", ""),
                    "answer": problem_data.get("solution", ""),
                    "level": problem_data.get("level", ""),
                    "type": problem_data.get("type", ""),
                    "source_file": json_file,
                }
                questions.append(question_data)
            except Exception as e:
                logger.warning(f"Error loading problem from {json_file}: {e}")

        logger.info(f"Loaded {len(questions)} MATH problems")
        return questions

    def load_questions_by_dataset(
        self, dataset_name: str, dataset_path: str
    ) -> List[Dict[str, Any]]:
        """Select and load questions for the specified dataset.

        Currently supported datasets:
        - "gsm8k": expects a JSONL file path (same as existing behavior)
        - "math": expects a directory path containing JSON files (recursive)

        Args:
            dataset_name: The logical name of the dataset (e.g., "gsm8k").
            dataset_path: The path to the dataset file/directory as required by the dataset loader.
        """
        if not dataset_name:
            raise ValueError("dataset_name must be provided")
        name = dataset_name.strip().lower()

        if name == "gsm8k":
            return self.load_gsm8k_questions(dataset_path)
        if name == "math":
            return self.load_math_questions(dataset_path)

        raise ValueError(
            f"Unsupported dataset '{dataset_name}'. Supported: ['gsm8k', 'math']"
        )

    def load_checkpoint(self, checkpoint_path: str) -> Tuple[List[Dict[str, Any]], int]:
        """Load existing checkpoint and return processed questions and last processed index."""
        if not os.path.exists(checkpoint_path):
            logger.info(
                f"No checkpoint found at {checkpoint_path}, starting from beginning"
            )
            return [], 0

        logger.info(f"Loading checkpoint from {checkpoint_path}")
        try:
            with open(checkpoint_path, "r") as f:
                checkpoint_data = json.load(f)

            if isinstance(checkpoint_data, list):
                # Direct list of categorized questions
                processed_questions = checkpoint_data
                last_index = len(processed_questions)
            else:
                # Structured checkpoint with metadata
                processed_questions = checkpoint_data.get("categorized_questions", [])
                last_index = checkpoint_data.get(
                    "last_processed_index", len(processed_questions)
                )

            logger.info(
                f"Loaded checkpoint with {len(processed_questions)} processed questions, resuming from index {last_index}"
            )
            return processed_questions, last_index

        except Exception as e:
            logger.warning(f"Error loading checkpoint: {e}, starting from beginning")
            return [], 0

    def save_checkpoint(
        self,
        categorized_questions: List[Dict[str, Any]],
        checkpoint_path: str,
        last_index: int,
    ) -> None:
        """Save checkpoint with processed questions and metadata."""
        logger.info(
            f"Saving checkpoint with {len(categorized_questions)} questions to {checkpoint_path}"
        )

        checkpoint_data = {
            "categorized_questions": categorized_questions,
            "last_processed_index": last_index,
            "total_processed": len(categorized_questions),
        }

        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint_data, f, indent=2)

        logger.info("Checkpoint saved successfully")

    async def categorize_question_by_area(
        self, question: str, areas: List[AreaInfo], **kwargs: Any
    ) -> str:
        """Categorize a question into one of the available areas (returns exact area name)."""
        area_names = [area.name for area in areas]
        area_bullets = "\n".join([f"- {area.name}" for area in areas])

        sys_prompt = SYSTEM_PROMPT_MATH_TAXONOMIST
        user_prompt = get_area_categorization_prompt(area_bullets, question)

        generation_config = {
            "temperature": 0.1,
            "max_tokens": 128,
            "seed": 42,
        }

        if self.llm_model is None:
            logger.warning("LLM model not initialized")
            return "Unknown"

        try:
            response, metadata = await self.llm_model.async_generate(
                sys_prompt=sys_prompt,
                user_prompt=user_prompt,
                generation_config=generation_config,
            )
            raw = (response or "").strip()
            # Print what model returned
            print(f"[AREA CATEGORIZED] {self._select_best_match(raw, area_names)}")
            # Select best match from allowed list
            return self._select_best_match(raw, area_names)

        except Exception as e:
            logger.warning(f"Error in area categorization: {e}")
            return "Unknown"

    async def categorize_questions(
        self,
        questions: List[Dict[str, Any]],
        checkpoint_path: str,
        save_every_n: int | None = None,
        output_path: str | None = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Perform two-step categorization of all questions with resume capability."""
        logger.info(
            f"Starting two-step categorization of {len(questions)} questions with resume capability..."
        )

        # Load existing checkpoint
        categorized_questions, start_index = self.load_checkpoint(checkpoint_path)

        if start_index >= len(questions):
            logger.info("All questions have already been processed")
            return categorized_questions

        logger.info(f"Resuming from question {start_index + 1}/{len(questions)}")

        for i in range(start_index, len(questions)):
            question_data = questions[i]

            if i % 10 == 0:
                logger.info(f"Processing question {i + 1}/{len(questions)}")

            question_text = question_data.get("question", "")

            # Step 1: Categorize by area
            predicted_area = await self.categorize_question_by_area(
                question_text, self.areas, **kwargs
            )

            # Step 2: Find the exact area key and get capabilities within that area
            area_key = self._find_matching_area_key(predicted_area)

            # Create categorized question data
            categorized_question = {
                **question_data,
                "categorized_area": area_key,
                "processing_order": i + 1,
            }

            categorized_questions.append(categorized_question)

            # Periodic checkpoint saving
            if save_every_n and (i + 1) % save_every_n == 0:
                try:
                    logger.info(
                        f"Checkpoint: saving results at {i + 1} questions to {checkpoint_path}"
                    )
                    self.save_checkpoint(categorized_questions, checkpoint_path, i + 1)
                except Exception as e:
                    logger.warning(f"Failed to write checkpoint at {i + 1}: {e}")

        # Final checkpoint save
        if save_every_n:
            try:
                logger.info(
                    f"Final checkpoint: saving all {len(categorized_questions)} questions to {checkpoint_path}"
                )
                self.save_checkpoint(
                    categorized_questions, checkpoint_path, len(questions)
                )
            except Exception as e:
                logger.warning(f"Failed to write final checkpoint: {e}")

        logger.info("Completed two-step categorization with resume capability")
        return categorized_questions

    def save_categorized_questions(
        self, categorized_questions: List[Dict[str, Any]], output_path: str
    ) -> None:
        """Save categorized questions to JSON file."""
        logger.info(
            f"Saving {len(categorized_questions)} categorized questions to {output_path}"
        )

        with open(output_path, "w") as f:
            json.dump(categorized_questions, f, indent=2)

        logger.info("Categorized questions saved successfully")

    def print_categorization_summary(
        self, categorized_questions: List[Dict[str, Any]]
    ) -> None:
        """Print summary of categorization results."""
        print("\n" + "=" * 80)
        dataset_name = getattr(self.cfg.data_cfg, "dataset_name", "dataset")
        print(f"{str(dataset_name).upper()} QUESTION CATEGORIZATION SUMMARY")
        print("=" * 80)

        # Count by area
        area_counts: Dict[str, int] = defaultdict(int)

        for q in categorized_questions:
            area = q.get("categorized_area", "Unknown")
            area_counts[area] += 1

        print(f"\nTotal questions categorized: {len(categorized_questions)}")

        print("\nQuestions by Area:")
        for area, count in sorted(
            area_counts.items(), key=lambda x: x[1], reverse=True
        ):
            print(f"  {area}: {count}")

    def run_categorization(self) -> None:
        """Run the complete categorization process."""
        logger.info("Starting question categorization...")

        # Extract areas and capabilities using the configured method
        self.areas, self.capabilities_by_area = self.extract_areas_and_capabilities()

        # Determine dataset and load questions
        dataset_name = getattr(self.cfg.data_cfg, "dataset_name", "gsm8k")
        dataset_path = getattr(self.cfg.data_cfg, "dataset_path", None)

        if not dataset_path or not dataset_name:
            raise ValueError(
                "dataset_path and dataset_name must be provided in the config"
            )
        questions = self.load_questions_by_dataset(dataset_name, dataset_path)

        # Prepare checkpoint and output paths
        checkpoint_path = os.path.join(
            self.cfg.output_cfg.results_dir,
            f"{dataset_name}_categorization_checkpoint_{len(questions)}.json",
        )
        output_path = os.path.join(
            self.cfg.output_cfg.results_dir, self.cfg.output_cfg.output_filename
        )
        os.makedirs(self.cfg.output_cfg.results_dir, exist_ok=True)
        save_every_n = getattr(
            getattr(self.cfg, "processing_cfg", {}), "save_every_n", 100
        )

        # Perform categorization with resume
        import asyncio

        categorized_questions = asyncio.run(
            self.categorize_questions(
                questions,
                checkpoint_path=checkpoint_path,
                save_every_n=save_every_n,
                output_path=output_path,
            )
        )

        # Save results
        self.save_categorized_questions(categorized_questions, output_path)

        # Print summary
        self.print_categorization_summary(categorized_questions)

        print(f"\nResults saved to: {output_path}")
        print(f"Checkpoint saved to: {checkpoint_path}")


@hydra.main(
    version_base=None,
    config_path="cfg",
    config_name="static_vs_generated",
)
def main(cfg: DictConfig) -> None:
    """Main function to run question categorization."""
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Create categorizer
    categorizer = DatasetQuestionCategorizer(cfg)

    # Run categorization
    categorizer.run_categorization()


if __name__ == "__main__":
    main()
