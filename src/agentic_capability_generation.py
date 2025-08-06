"""Generate hierarchical capabilities using a multi-agent debate system."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import hydra
from omegaconf import DictConfig, OmegaConf

from src.capability import Capability
from src.utils.agentic_helpers import (
    _debate_once,
    _make_moderator,
    _make_scientist,
    _stub_class,
    _to_autogen_cfg,
)
from src.utils.agentic_prompts import SCIENTIST_AREA_PROMPT, SCIENTIST_CAP_PROMPT


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
log = logging.getLogger("agentic_cap_gen")


def _persist_capabilities(
    all_caps: Dict[str, Dict[str, Any]],
    domain: str,
    base_dir: Path,
) -> List[Capability]:
    """Create METR Capability directories."""
    base_dir.mkdir(parents=True, exist_ok=True)
    created: List[Capability] = []

    for area_id, bundle in all_caps.items():
        for cap in bundle["capabilities"]:
            meta = {
                "name": cap["name"],
                "description": cap.get("description", ""),
                "domain": domain,
                "area": int(area_id),
                "class": _stub_class(cap["name"], cap.get("description", "")),
            }

            try:
                created.append(Capability.from_dict(meta, base_dir=str(base_dir)))
                log.info(f"Created capability: {cap['name']}")
            except FileExistsError:
                log.warning("Capability %s already exists – skipping", cap["name"])
            except Exception as e:
                log.error(f"Failed to create capability {cap['name']}: {e}")

    return created


def generate_capabilities(cfg: DictConfig) -> List[Capability]:
    """
    Run the complete multi-agent debate pipeline for capability generation.

    This implements the enhanced ACE framework with:
    1. Multi-agent capability hierarchy design
    2. Robust error handling and validation
    """
    try:
        # Extract configuration
        domain = cfg.capabilities_cfg.domain
        n_areas = cfg.capabilities_cfg.num_capability_areas
        total_caps = cfg.capabilities_cfg.num_gen_capabilities
        caps_per_area = max(1, total_caps // max(1, n_areas))

        # LLM configuration with proper error handling
        g = cfg.scientist_llm.generation_cfg.capability_generation
        base_llm = {
            "name": cfg.scientist_llm.name,
            "temperature": g.get("temperature", 0.7),
            "max_tokens": g.get("max_tokens", 4096),
            "seed": g.get("seed", 42),
        }

        log.info(f"Generating capabilities for domain: {domain}")
        log.info(f"Target: {n_areas} areas, {total_caps} capabilities")

    except AttributeError as e:
        log.error("Missing required configuration: %s", e)
        raise ValueError(f"Configuration error: {e}") from e

    # Create agent instances with enhanced configurations
    scientist_a = _make_scientist("Scientist_A", _to_autogen_cfg(base_llm), domain)
    scientist_b = _make_scientist(
        "Scientist_B",
        _to_autogen_cfg(base_llm, {"temperature": base_llm["temperature"] + 0.3}),
        domain,
    )
    moderator = _make_moderator(_to_autogen_cfg(base_llm, {"temperature": 0.2}))

    # Phase 1: Collaborative area design
    log.info("Phase 1: Designing capability areas through multi-agent debate")
    areas_bundle = _debate_once(
        scientist_a,
        scientist_b,
        moderator,
        SCIENTIST_AREA_PROMPT.format(role="A/B", n=n_areas, domain=domain),
    )

    areas = areas_bundle.get("areas", [])
    log.info(f"Generated {len(areas)} areas: {[area['name'] for area in areas]}")

    # Phase 2: Collaborative capability design per area
    log.info("Phase 2: Designing capabilities for each area")
    all_caps: Dict[str, Dict[str, Any]] = {}

    for area in areas:
        log.info(f"Generating capabilities for area: {area['name']}")
        task = SCIENTIST_CAP_PROMPT.format(
            role="A/B",
            domain=domain,
            area_name=area["name"],
            area_id=area["id"],
            n=caps_per_area,
        )

        cap_bundle = _debate_once(
            scientist_a,
            scientist_b,
            moderator,
            task,
        )

        all_caps[str(area["id"])] = cap_bundle
        caps = cap_bundle.get("capabilities", [])
        log.info(f"Generated {len(caps)} capabilities for area {area['name']}")

    # Phase 3: Persistence and artifact creation
    log.info("Phase 3: Persisting capabilities and creating artifacts")
    run_id = (
        getattr(cfg.exp_cfg, "exp_id", "debate_run")
        if "exp_cfg" in cfg
        else "debate_run"
    )
    base_dir = (
        Path(cfg.capabilities_cfg.capabilities_dir) / f"capabilities_{run_id}" / domain
    )

    capability_objs = _persist_capabilities(all_caps, domain, base_dir)

    # Save comprehensive audit trail
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_dir = base_dir / "_json"
    json_dir.mkdir(parents=True, exist_ok=True)

    # Save all intermediate results for transparency
    (json_dir / f"areas_{domain}_{timestamp}.json").write_text(
        json.dumps(areas_bundle, indent=2, ensure_ascii=False)
    )
    (json_dir / f"capabilities_{domain}_{timestamp}.json").write_text(
        json.dumps(all_caps, indent=2, ensure_ascii=False)
    )

    # Summary statistics
    total_capabilities = sum(
        len(bundle.get("capabilities", [])) for bundle in all_caps.values()
    )
    log.info("Multi-agent debate complete")
    log.info(f"Generated: {len(areas)} areas, {total_capabilities} capabilities")
    log.info(f"Artifacts saved to: {base_dir}")

    return capability_objs


@hydra.main(version_base=None, config_path="cfg", config_name="agentic_config")
def main(cfg: DictConfig) -> None:
    """Run the multi-agent capability generation system."""
    log.info("Starting Multi-Agent Debate-Based Capability Generation")
    log.info("Configuration:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    try:
        caps = generate_capabilities(cfg)
        print(
            f"Successfully generated {len(caps)} capabilities using multi-agent debate"
        )
        print("METR-compatible artifacts are ready for evaluation")
    except Exception as e:
        log.error(f"Generation failed: {e}")
        raise


if __name__ == "__main__":
    main()
