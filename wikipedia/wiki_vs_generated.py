"""Match Wikipedia capabilities with generated capabilities.

Uses pre-categorized Wikipedia data to load and match capabilities.
"""

import glob
import json
import logging
import os
from typing import Dict, List

import hydra
from omegaconf import DictConfig

from src.model import Model
from wikipedia.prompts import (
    SYSTEM_PROMPT_MATH_CAPABILITIES,
    get_generated_to_wikipedia_prompt,
    get_wikipedia_to_generated_prompt,
)


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WikipediaCapability:
    """Represents a Wikipedia capability."""

    def __init__(self, name: str, description: str, area: str) -> None:
        self.name = name
        self.description = description
        self.area = area


class GeneratedCapability:
    """Represents a generated capability."""

    def __init__(self, name: str, description: str, area: str) -> None:
        self.name = name
        self.description = description
        self.area = area


class GeneratedVsWikipedia:
    """Match Wikipedia capabilities with generated capabilities using batching."""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.model = Model(
            model_name=cfg.llm_cfg.model_name, model_provider=cfg.llm_cfg.model_provider
        )
        self.results: Dict[str, str] = {}
        self.match_direction: str = getattr(
            getattr(cfg, "processing_cfg", {}),
            "match_direction",
            "generated_to_wikipedia",
        )

    def load_wikipedia_capabilities(self) -> Dict[str, List[WikipediaCapability]]:
        """Load Wikipedia capabilities from JSON format in wikipedia/pages/.

        Returns
        -------
            Dictionary mapping area names to lists of WikipediaCapability objects
        """
        capabilities_by_area: Dict[str, List[WikipediaCapability]] = {}

        # Path to the Wikipedia pages directory (from config)
        wikipedia_pages_dir = self.cfg.data_cfg.wikipedia_pages_dir

        if not os.path.exists(wikipedia_pages_dir):
            logger.error(f"Wikipedia pages directory not found: {wikipedia_pages_dir}")
            return capabilities_by_area

        try:
            # Load all JSON files from the Wikipedia pages directory
            json_files = glob.glob(os.path.join(wikipedia_pages_dir, "*.json"))

            for json_file in json_files:
                try:
                    with open(json_file, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    # Extract capability information from the new format
                    cap_name = data.get("capability_name", "")
                    description = data.get("description", "")
                    area = data.get("area", "Unknown")

                    if cap_name and description:
                        # Create capability object
                        capability = WikipediaCapability(
                            name=cap_name, description=description, area=area
                        )

                        # Group by area
                        if area not in capabilities_by_area:
                            capabilities_by_area[area] = []
                        capabilities_by_area[area].append(capability)

                except Exception as e:
                    logger.warning(f"Error loading {json_file}: {e}")
                    continue

            total_caps = sum(len(caps) for caps in capabilities_by_area.values())
            logger.info(f"Loaded {total_caps} Wikipedia capabilities")
            logger.info(f"Areas: {list(capabilities_by_area.keys())}")

        except Exception as e:
            logger.error(f"Error loading Wikipedia capabilities: {e}")

        return capabilities_by_area

    def load_generated_capabilities(
        self, generated_dir: str
    ) -> List[GeneratedCapability]:
        """
        Load capabilities from generated directory structure.

        Args:
            generated_dir: Directory containing generated capabilities

        Returns
        -------
            List of GeneratedCapability objects
        """
        capabilities = []

        # Look for capability.json files in the directory structure
        capability_files = glob.glob(
            os.path.join(generated_dir, "**/capability.json"), recursive=True
        )

        logger.info(f"Found {len(capability_files)} generated capability files")

        for file_path in capability_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    cap_data = json.load(f)

                capability = GeneratedCapability(
                    name=cap_data.get("capability_name", ""),
                    description=cap_data.get("capability_description", ""),
                    area=cap_data.get("capability_area", "mathematics"),
                )

                capabilities.append(capability)

            except Exception as e:
                logger.warning(f"Error loading {file_path}: {e}")
                continue

        logger.info(f"Successfully loaded {len(capabilities)} generated capabilities")
        return capabilities

    def match_wikipedia_to_generated_capabilities(
        self,
        wikipedia_cap: WikipediaCapability,
        generated_caps: List[GeneratedCapability],
    ) -> str:
        """
        Match Wikipedia capability to generated capabilities using batching.

        Args:
            wikipedia_cap: Wikipedia capability
            generated_caps: List of generated capabilities in the matched area

        Returns
        -------
            Name of the matched capability or "none" if no match
        """
        if not generated_caps:
            return "none"

        # Batch size to avoid context length issues
        batch_size = 20

        logger.info(
            f"  Processing {len(generated_caps)} capabilities in batches of {batch_size}"
        )

        # Process capabilities in batches
        for i in range(0, len(generated_caps), batch_size):
            batch = generated_caps[i : i + batch_size]
            logger.info(
                f"  Processing batch {i // batch_size + 1}/{(len(generated_caps) + batch_size - 1) // batch_size}"
            )

            result = self.match_wikipedia_to_generated_batch(wikipedia_cap, batch)

            # If we found a match in this batch, return it
            if result != "none":
                logger.info(f"  Found match in batch: {result}")
                return result

        # No match found in any batch
        logger.info("  No match found in any batch")
        return "none"

    def match_wikipedia_to_generated_batch(
        self,
        wikipedia_cap: WikipediaCapability,
        generated_caps_batch: List[GeneratedCapability],
    ) -> str:
        """Match a Wikipedia capability against a batch of generated capabilities.

        Returns generated capability name if matched, otherwise "none".
        """
        if not generated_caps_batch:
            return "none"

        capabilities_list = "\n".join(
            [f"- {cap.name}: {cap.description}" for cap in generated_caps_batch]
        )

        user_prompt = get_wikipedia_to_generated_prompt(
            wikipedia_cap.name, wikipedia_cap.description, capabilities_list
        )

        try:
            response, metadata = self.model.generate(
                sys_prompt=SYSTEM_PROMPT_MATH_CAPABILITIES,
                user_prompt=user_prompt,
                generation_config={"temperature": 0.0, "max_tokens": 100},
            )

            response_str: str = str(response).strip()
            capability_names = [cap.name for cap in generated_caps_batch]
            if response_str in capability_names:
                return response_str
            return "none"
        except Exception as e:
            logger.error(f"Error matching Wikipedia capability to generated batch: {e}")
            return "none"

    def match_generated_to_wikipedia_capabilities(
        self,
        generated_cap: GeneratedCapability,
        wikipedia_caps: List[WikipediaCapability],
    ) -> str:
        """
        Match generated capability to Wikipedia capabilities using batching.

        Args:
            generated_cap: Generated capability
            wikipedia_caps: List of Wikipedia capabilities in the matched area

        Returns
        -------
            Name of the matched Wikipedia capability or "none" if no match
        """
        if not wikipedia_caps:
            return "none"

        # Batch size to avoid context length issues
        batch_size = 40

        logger.info(
            f"  Processing {len(wikipedia_caps)} Wikipedia capabilities in batches of {batch_size}"
        )

        # Process capabilities in batches
        for i in range(0, len(wikipedia_caps), batch_size):
            batch = wikipedia_caps[i : i + batch_size]
            logger.info(
                f"  Processing batch {i // batch_size + 1}/{(len(wikipedia_caps) + batch_size - 1) // batch_size}"
            )

            result = self.match_generated_to_wikipedia_batch(generated_cap, batch)

            # If we found a match in this batch, return it
            if result != "none":
                logger.info(f"  Found match in batch: {result}")
                return result

        # No match found in any batch
        logger.info("  No match found in any batch")
        return "none"

    def match_generated_to_wikipedia_batch(
        self,
        generated_cap: GeneratedCapability,
        wikipedia_caps_batch: List[WikipediaCapability],
    ) -> str:
        """
        Match generated capability to a batch of Wikipedia capabilities.

        Args:
            generated_cap: Generated capability to match
            wikipedia_caps_batch: Batch of Wikipedia capabilities to match against

        Returns
        -------
            Name of the matched Wikipedia capability or "none" if no match
        """
        if not wikipedia_caps_batch:
            return "none"

        capabilities_list = "\n".join(
            [f"- {cap.name}: {cap.description}" for cap in wikipedia_caps_batch]
        )

        user_prompt = get_generated_to_wikipedia_prompt(
            generated_cap.name, generated_cap.description, capabilities_list
        )

        try:
            response, metadata = self.model.generate(
                sys_prompt=SYSTEM_PROMPT_MATH_CAPABILITIES,
                user_prompt=user_prompt,
                generation_config={"temperature": 0.0, "max_tokens": 100},
            )

            # Clean the response
            response_str: str = str(response).strip()

            # Check if the response matches one of the available capabilities
            capability_names = [cap.name for cap in wikipedia_caps_batch]
            if response_str in capability_names:
                return response_str
            return "none"

        except Exception as e:
            logger.error(f"Error matching artifact to Wikipedia batch: {e}")
            return "none"

    def match_capabilities(
        self,
        generated_caps: List[GeneratedCapability],
        categorized_wikipedia_caps: Dict[str, List[WikipediaCapability]],
    ) -> Dict[str, str]:
        """Match capabilities based on configured direction.

        Returns a mapping:
        - generated_to_wikipedia: {generated_name -> wikipedia_name}
        - wikipedia_to_generated: {wikipedia_name -> generated_name}
        """
        results: Dict[str, str] = {}

        # Flatten lists and build area groupings
        all_wikipedia_caps: List[WikipediaCapability] = []
        for area_caps in categorized_wikipedia_caps.values():
            all_wikipedia_caps.extend(area_caps)

        generated_caps_by_area: Dict[str, List[GeneratedCapability]] = {}
        for cap in generated_caps:
            generated_caps_by_area.setdefault(cap.area, []).append(cap)

        if self.match_direction == "generated_to_wikipedia":
            logger.info("Starting two-step matching process (GENERATED -> WIKIPEDIA):")
            logger.info(f"  - {len(generated_caps)} generated capabilities")
            logger.info(f"  - {len(all_wikipedia_caps)} Wikipedia capabilities")
            logger.info(
                f"  - {len(generated_caps_by_area)} generated areas: {list(generated_caps_by_area.keys())}"
            )

            for i, generated_cap in enumerate(generated_caps):
                logger.info(
                    f"\nProcessing generated capability {i + 1}/{len(generated_caps)}: {generated_cap.name}"
                )
                logger.info(f"  Generated area: {generated_cap.area}")

                wikipedia_area = generated_cap.area
                area_wikipedia_caps = categorized_wikipedia_caps.get(wikipedia_area, [])
                if not area_wikipedia_caps:
                    logger.info(
                        f"  - NO WIKIPEDIA CAPABILITIES in area '{wikipedia_area}'"
                    )
                    results[generated_cap.name] = "none"
                    continue

                logger.info(
                    f"  + Found {len(area_wikipedia_caps)} Wikipedia capabilities in area '{wikipedia_area}'"
                )
                matched = self.match_generated_to_wikipedia_capabilities(
                    generated_cap, area_wikipedia_caps
                )
                if matched == "none":
                    logger.info(
                        f"  - NO WIKIPEDIA MATCH: {generated_cap.name} in area '{wikipedia_area}'"
                    )
                else:
                    logger.info(
                        f"  + WIKIPEDIA MATCH: {generated_cap.name} -> {matched} (in area '{wikipedia_area}')"
                    )
                results[generated_cap.name] = matched

        elif self.match_direction == "wikipedia_to_generated":
            logger.info("Starting two-step matching process (WIKIPEDIA -> GENERATED):")
            logger.info(f"  - {len(all_wikipedia_caps)} Wikipedia capabilities")
            logger.info(f"  - {len(generated_caps)} generated capabilities")
            logger.info(
                f"  - {len(generated_caps_by_area)} generated areas: {list(generated_caps_by_area.keys())}"
            )

            for i, wikipedia_cap in enumerate(all_wikipedia_caps):
                logger.info(
                    f"\nProcessing Wikipedia capability {i + 1}/{len(all_wikipedia_caps)}: {wikipedia_cap.name}"
                )
                logger.info(f"  Wikipedia area: {wikipedia_cap.area}")

                generated_area = wikipedia_cap.area
                area_generated_caps = generated_caps_by_area.get(generated_area, [])
                if not area_generated_caps:
                    logger.info(
                        f"  - NO GENERATED CAPABILITIES in area '{generated_area}'"
                    )
                    results[wikipedia_cap.name] = "none"
                    continue

                logger.info(
                    f"  + Found {len(area_generated_caps)} generated capabilities in area '{generated_area}'"
                )
                matched = self.match_wikipedia_to_generated_capabilities(
                    wikipedia_cap, area_generated_caps
                )
                if matched == "none":
                    logger.info(
                        f"  - NO GENERATED MATCH: {wikipedia_cap.name} in area '{generated_area}'"
                    )
                else:
                    logger.info(
                        f"  + GENERATED MATCH: {wikipedia_cap.name} -> {matched} (in area '{generated_area}')"
                    )
                results[wikipedia_cap.name] = matched

        else:
            raise ValueError(
                "processing_cfg.match_direction must be 'generated_to_wikipedia' or 'wikipedia_to_generated'"
            )

        return results

    def save_results(
        self,
        results: Dict[str, str],
        output_path: str,
        generated_caps: List[GeneratedCapability],
        categorized_wikipedia_caps: Dict[str, List[WikipediaCapability]],
    ) -> None:
        """
        Save results to JSON file with detailed information.

        Args:
            results: Dictionary of matching results
            output_path: Path to save the results
            generated_caps: List of generated capabilities
            categorized_wikipedia_caps: Dictionary of categorized Wikipedia capabilities
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Flatten Wikipedia capabilities for output
        all_wikipedia_caps = []
        for _area, caps in categorized_wikipedia_caps.items():
            all_wikipedia_caps.extend(caps)

        # Create detailed results with metadata
        detailed_results = {
            "metadata": {
                "total_generated_capabilities": len(generated_caps),
                "total_wikipedia_capabilities": len(all_wikipedia_caps),
                "categorized_wikipedia_areas": len(categorized_wikipedia_caps),
                "matched_capabilities": sum(1 for v in results.values() if v != "none"),
                "unmatched_capabilities": sum(
                    1 for v in results.values() if v == "none"
                ),
                "match_rate": sum(1 for v in results.values() if v != "none")
                / len(results)
                if results
                else 0,
                "matching_direction": self.match_direction,
            },
            "matching_results": results,
            "generated_capabilities": [
                {"name": cap.name, "description": cap.description, "area": cap.area}
                for cap in generated_caps
            ],
            "wikipedia_capabilities": [
                {"name": cap.name, "description": cap.description, "area": cap.area}
                for cap in all_wikipedia_caps
            ],
            "wikipedia_capabilities_by_area": {
                area: [
                    {"name": cap.name, "description": cap.description, "area": cap.area}
                    for cap in caps
                ]
                for area, caps in categorized_wikipedia_caps.items()
            },
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(detailed_results, f, indent=2, ensure_ascii=False)

        logger.info(f"Detailed results saved to: {output_path}")

    def print_results(self, results: Dict[str, str]) -> None:
        """
        Print results in a formatted way.

        Args:
            results: Dictionary of matching results
        """
        print("\n" + "=" * 80)
        if self.match_direction == "generated_to_wikipedia":
            print("GENERATED → WIKIPEDIA MATCHING RESULTS")
        else:
            print("WIKIPEDIA → GENERATED MATCHING RESULTS")
        print("=" * 80)

        if self.match_direction == "generated_to_wikipedia":
            for generated_name, wikipedia_name in results.items():
                if wikipedia_name == "none":
                    print(f"[NO MATCH] {generated_name} -> NO MATCH")
                else:
                    print(f"[MATCH] {generated_name} -> {wikipedia_name}")
        else:
            for wikipedia_name, generated_name in results.items():
                if generated_name == "none":
                    print(f"[NO MATCH] {wikipedia_name} -> NO MATCH")
                else:
                    print(f"[MATCH] {wikipedia_name} -> {generated_name}")

        print("=" * 80)
        if self.match_direction == "generated_to_wikipedia":
            print(f"Total generated capabilities: {len(results)}")
        else:
            print(f"Total Wikipedia capabilities: {len(results)}")
        matched_count = sum(1 for v in results.values() if v != "none")
        print(f"Matched capabilities: {matched_count}")
        print(f"Unmatched capabilities: {len(results) - matched_count}")
        print("=" * 80)


@hydra.main(version_base=None, config_path="cfg", config_name="wiki_vs_generated")
def main(cfg: DictConfig) -> None:
    """Run generated-Wikipedia capability matching.

    Args:
        cfg: Configuration for the matching process
    """
    logger.info(
        "Starting Generated-Wikipedia Matcher V2 Fixed (Generated -> Wikipedia Version)"
    )

    # Initialize matcher
    matcher = GeneratedVsWikipedia(cfg)

    # Load capabilities
    logger.info("Loading generated capabilities...")
    generated_caps = matcher.load_generated_capabilities(cfg.data_cfg.generated_dir)

    logger.info("Loading Wikipedia capabilities...")
    categorized_wikipedia_caps = matcher.load_wikipedia_capabilities()

    if not generated_caps:
        logger.error("No generated capabilities found!")
        return

    if not categorized_wikipedia_caps:
        logger.error("No categorized Wikipedia capabilities found!")
        return

    # Match capabilities (generated -> Wikipedia direction)
    logger.info("Starting capability matching (generated -> Wikipedia)...")
    results = matcher.match_capabilities(generated_caps, categorized_wikipedia_caps)

    # Print results
    matcher.print_results(results)

    # Save results (append _generated_to_wikipedia to filename)
    orig_filename = cfg.output_cfg.output_filename
    name, ext = os.path.splitext(orig_filename)
    filename_with_suffix = f"{name}_generated_to_wikipedia{ext or '.json'}"
    output_path = os.path.join(cfg.output_cfg.results_dir, filename_with_suffix)
    matcher.save_results(
        results, output_path, generated_caps, categorized_wikipedia_caps
    )
    print(f"Output saved to: {output_path}")

    logger.info("Generated-Wikipedia matching completed!")


if __name__ == "__main__":
    main()
