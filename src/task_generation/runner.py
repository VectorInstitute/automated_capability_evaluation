"""Script to run the task generation pipeline over a corpus of book chapters."""
import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv

from src.schemas.area_schemas import Area
from src.schemas.capability_schemas import Capability
from src.schemas.domain_schemas import Domain
from src.schemas.io_utils import save_tasks
from src.schemas.metadata_schemas import PipelineMetadata
from src.schemas.task_schemas import Task
from src.task_generation.agentic_pipeline import run_task_generation_loop
from src.task_generation.dedup_utils import (
    assign_chapter_level_task_ids,
    deduplicate_tasks_for_chapter,
    mark_discarded_metadata,
)
from src.task_generation.designer_agent import DesignerAgent
from src.task_generation.verifier_agent import VerifierAgent


load_dotenv()

ROOT_DIR = Path(__file__).resolve().parents[2]
LOG_DIR = ROOT_DIR / "src" / "task_generation" / "logs"
logger = logging.getLogger(__name__)


def configure_logging() -> None:
    """Configure logging for CLI runs without overriding existing app logging."""
    root_logger = logging.getLogger()
    if root_logger.handlers:
        return

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"task_gen_{run_id}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def load_yaml_config(path: Path) -> Dict[str, Any]:
    """
    Load YAML config from file, expanding environment variables.

    Args:
        path: Path to the YAML config file.

    Returns
    -------
        Parsed YAML content as a dictionary.
    """
    with open(path, "r", encoding="utf-8") as f:
        content = os.path.expandvars(f.read())

    obj: Any = yaml.safe_load(content)
    if not isinstance(obj, dict):
        raise ValueError(f"YAML root must be a mapping, got {type(obj)} in {path}")
    return obj


def create_tag(dt: datetime) -> str:
    """
    Make a tag string from datetime in format: _YYYYMMDD_HHMMSS.

    Args:
        dt: The datetime object.

    Returns
    -------
        The formatted tag string.
    """
    return "_" + dt.strftime("%Y%m%d_%H%M%S")


def check_tag(s: str) -> bool:
    """
    Check if string looks like a tag in format: _YYYYMMDD_HHMMSS.

    Args:
        s: The string to check.

    Returns
    -------
        True if it looks like a tag, False otherwise.
    """
    if not isinstance(s, str) or len(s) != 16 or not s.startswith("_"):
        return False
    return s[1:9].isdigit() and s[10:16].isdigit() and s[9] == "_"


def create_diff_blueprint_combo(difficulty: str, blooms: str) -> str:
    """
    Create a slug from difficulty and blooms level.

    Args:
        difficulty: Difficulty string.
        blooms: Bloom's level string.

    Returns
    -------
        Slugified string.
    """

    def _clean(x: str) -> str:
        return "".join(ch if ch.isalnum() else "_" for ch in x.strip()).strip("_")

    return f"{_clean(difficulty)}_{_clean(blooms)}"


def get_book_name_from_relpath(chapter_relpath: str) -> str:
    """
    Assumes chapter_relpath looks like: <book_name>/<chapter_file>.txt.

    Args:
        chapter_relpath: The chapter relative path.

    Returns
    -------
        The book name extracted from the relative path.
    """
    parts = chapter_relpath.split("/")
    return parts[0] if parts else "unknown_book"


def build_placeholder_capability(
    *, chapter_id: str, chapter_index: int, domain_name: str
) -> Capability:
    """Schema-valid placeholder capability until Stage-2 capabilities are wired."""
    domain = Domain(
        domain_name=domain_name or "unknown_domain",
        domain_id="domain_000",
        domain_description="Placeholder domain for task-generation runner",
    )
    area = Area(
        area_name=f"placeholder_area_{chapter_id}",
        area_id=f"area_{chapter_index:03d}",
        domain=domain,
        area_description="Placeholder area for task-generation runner",
    )
    return Capability(
        capability_name=f"placeholder_capability_{chapter_id}",
        capability_id="cap_000",
        area=area,
        capability_description="Placeholder capability for task-generation runner",
    )


def load_runner_configs(config_path: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Load runner agent and pipeline configs."""
    agent_cfg = load_yaml_config(config_path / "agent_config.yaml")
    pipeline_cfg = load_yaml_config(config_path / "pipeline_config.yaml")
    return agent_cfg, pipeline_cfg


def load_blueprints(blueprints_path: Path) -> Tuple[List[Dict[str, Any]], str]:
    """Load blueprint combinations and domain from blueprint file."""
    if not blueprints_path.exists():
        raise FileNotFoundError(f"Blueprints JSON not found: {blueprints_path}")

    blueprints_obj: Dict[str, Any] = json.loads(
        blueprints_path.read_text(encoding="utf-8")
    )
    combinations = blueprints_obj.get("combinations", [])
    if not isinstance(combinations, list) or not combinations:
        raise ValueError(
            f"Blueprints JSON must contain a non-empty 'combinations' list: {blueprints_path}"
        )
    blueprint_domain = str(blueprints_obj.get("domain"))
    return combinations, blueprint_domain


def init_model_clients(
    agent_cfg: Dict[str, Any],
) -> Tuple[OpenAIChatCompletionClient, OpenAIChatCompletionClient]:
    """Initialize designer/verifier model clients."""
    designer_model_cfg = agent_cfg["agents"]["designer"]["model_config"]["config_list"][
        0
    ]
    verifier_model_cfg = agent_cfg["agents"]["verifier"]["model_config"]["config_list"][
        0
    ]

    designer_client = OpenAIChatCompletionClient(
        model=designer_model_cfg["model"],
        api_key=designer_model_cfg["api_key"],
        model_info={
            "family": "openai",
            "vision": False,
            "function_calling": False,
            "json_output": False,
            "structured_output": False,
        },
    )

    verifier_client = OpenAIChatCompletionClient(
        model=verifier_model_cfg["model"],
        api_key=verifier_model_cfg["api_key"],
        model_info={
            "family": "openai",
            "vision": False,
            "function_calling": False,
            "json_output": False,
            "structured_output": False,
        },
    )

    logger.info(f"Designer Agent loaded with config: {designer_model_cfg}")
    logger.info(f"Verifier Agent loaded with config: {verifier_model_cfg}")
    return designer_client, verifier_client


async def run_pipeline(
    *,
    experiment_id_override: Optional[str] = None,
    output_base_dir_override: Optional[Path] = None,
    tasks_tag_override: Optional[str] = None,
    capabilities_tag_override: Optional[str] = None,
) -> str:
    """Run task generation pipeline and return the Stage-3 tasks tag."""
    configure_logging()
    config_path = ROOT_DIR / "src" / "cfg" / "task_generation"
    agent_cfg, pipeline_cfg = load_runner_configs(config_path)

    # ---- required pipeline config ----
    experiment_id = experiment_id_override or pipeline_cfg["pipeline"]["experiment_id"]
    output_base_dir = output_base_dir_override or Path(
        pipeline_cfg["pipeline"]["output_base_dir"]
    )

    # ---- chapter corpus root ----
    book_chapter_dir = pipeline_cfg["pipeline"].get(
        "book_chapter_dir", "book_chapter_text_files"
    )
    chapter_root_dir = (
        ROOT_DIR / "src" / "task_generation" / book_chapter_dir
    ).resolve()
    if not chapter_root_dir.exists():
        raise FileNotFoundError(f"book_chapter_dir not found: {chapter_root_dir}")

    # ---- blueprints ----
    blueprints_file = pipeline_cfg["pipeline"].get(
        "blueprints_file", "finance_blueprints.json"
    )
    blueprints_path = (
        ROOT_DIR / "src" / "task_generation" / "blueprints" / blueprints_file
    ).resolve()
    combinations, blueprint_domain = load_blueprints(blueprints_path)
    logger.info(f"Using blueprints file: {blueprints_path}")

    # ---- runtime params ----
    max_retries = int(pipeline_cfg["pipeline"].get("max_retries", 3))
    default_num_tasks_per_combo = int(
        pipeline_cfg["pipeline"].get("num_tasks_per_combo", 5)
    )
    configured_num_tasks = pipeline_cfg["pipeline"].get("num_tasks")
    num_tasks = int(
        configured_num_tasks
        if configured_num_tasks is not None
        else combinations[0].get("num_tasks", default_num_tasks_per_combo)
    )
    capability_source_mode = str(
        pipeline_cfg["pipeline"].get("capability_source_mode", "placeholder")
    ).strip().lower()
    if capability_source_mode not in {"placeholder", "from_stage2"}:
        raise ValueError(
            "capability_source_mode must be one of: placeholder, from_stage2. "
            f"Got: {capability_source_mode}"
        )
    if capability_source_mode == "from_stage2":
        raise NotImplementedError(
            "capability_source_mode=from_stage2 is not wired yet. "
            "Use capability_source_mode=placeholder for now."
        )
    capabilities_input_tag = str(
        capabilities_tag_override
        or pipeline_cfg["pipeline"].get("capabilities_tag")
        or "placeholder_capabilities_tag"
    )

    resume_tag = pipeline_cfg["pipeline"].get("resume_tag")  # optional
    if resume_tag and not check_tag(resume_tag):
        raise ValueError(f"resume_tag must match _YYYYMMDD_HHMMSS, got: {resume_tag}")

    out_tag = tasks_tag_override or resume_tag or create_tag(datetime.now(timezone.utc))
    designer_client, verifier_client = init_model_clients(agent_cfg)

    try:
        # ---- discover chapters on disk ----
        chapter_files = sorted(chapter_root_dir.rglob("*.txt"))
        if not chapter_files:
            logger.error(f"No chapter .txt files found under: {chapter_root_dir}")
            return out_tag

        logger.info(f"Found {len(chapter_files)} chapter files under {chapter_root_dir}")
        logger.info(f"Found {len(combinations)} blueprint combinations")
        logger.info(f"Capability source mode: {capability_source_mode}")
        logger.info(f"num_tasks per capability/chapter: {num_tasks}")
        if blueprint_domain:
            logger.info(f"Blueprint domain (informational): {blueprint_domain}")

        def make_designer_agent() -> DesignerAgent:
            return DesignerAgent(name="Designer", model_client=designer_client)

        def make_verifier_agent() -> VerifierAgent:
            return VerifierAgent(name="Verifier", model_client=verifier_client)

        for chapter_idx, chapter_path in enumerate(chapter_files):
            chapter_relpath = chapter_path.relative_to(chapter_root_dir).as_posix()
            chapter_id = chapter_path.stem
            book_name = get_book_name_from_relpath(chapter_relpath)
            placeholder_capability = build_placeholder_capability(
                chapter_id=chapter_id,
                chapter_index=chapter_idx,
                domain_name=blueprint_domain,
            )

            context_text = chapter_path.read_text(encoding="utf-8")

            # Stage-3-compatible path: tasks/<tag>/<area_id>/<capability_id>/tasks.json
            chapter_out_path = (
                output_base_dir
                / experiment_id
                / "tasks"
                / out_tag
                / placeholder_capability.area.area_id
                / placeholder_capability.capability_id
                / "tasks.json"
            )
            chapter_out_path.parent.mkdir(parents=True, exist_ok=True)

            checkpoint_dir = chapter_out_path.parent / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            # Make it deterministic so resume works across reruns
            checkpoint_path = checkpoint_dir / "passed_tasks_checkpoint.json"

            now_ckpt = datetime.now(timezone.utc)
            checkpoint_metadata = PipelineMetadata(
                experiment_id=experiment_id,
                output_base_dir=str(output_base_dir),
                timestamp=now_ckpt.isoformat().replace("+00:00", "Z"),
                input_stage_tag=capabilities_input_tag,
                output_stage_tag=out_tag,
                resume=bool(resume_tag),
            )

            if resume_tag and chapter_out_path.exists():
                logger.info(
                    f"Skipping (resume) {placeholder_capability.area.area_id}/"
                    f"{placeholder_capability.capability_id} ({book_name}/{chapter_id}) "
                    "because tasks.json exists."
                )
                continue

            # ---- chapter knowledge extraction (ONCE per chapter) ----
            designer = make_designer_agent()
            chapter_knowledge_obj, _chapter_knowledge_prompt = await designer.summarize_chapter_knowledge(
                chapter_excerpts=context_text
            )
            if not isinstance(chapter_knowledge_obj, dict):
                logger.warning("Chapter knowledge summary not dict; using raw text block.")
            chapter_knowledge_text = (
                json.dumps(chapter_knowledge_obj, indent=2, ensure_ascii=False)
                if isinstance(chapter_knowledge_obj, dict)
                else str(chapter_knowledge_obj)
            )

            logger.info(
                f"Chapter {chapter_idx + 1}/{len(chapter_files)}: {chapter_relpath}"
            )

            all_tasks: List[Task] = []
            chapter_q_counter = 0
            chapter_verification_logs: List[Dict[str, Any]] = []
            chapter_prev_questions: List[str] = []

            # Blueprints are currently placeholders; use first combination for now.
            difficulty = (
                str(combinations[0].get("difficulty", "")).strip().split("-")[0].strip()
            )
            blooms_level = (
                str(combinations[0].get("blooms_level", "")).strip().split("-")[0].strip()
            )
            blueprint = combinations[0].get("blueprint", "")

            diff_bpt_combo = create_diff_blueprint_combo(difficulty, blooms_level)

            logger.info(
                f"Generating tasks for {chapter_id} with tasks (n={num_tasks})"
            )

            tasks: Optional[List[Task]] = await run_task_generation_loop(
                designer_factory=make_designer_agent,
                verifier_factory=make_verifier_agent,
                capability=placeholder_capability,
                capability_source_mode=capability_source_mode,
                domain=blueprint_domain,
                context_text=context_text,
                chapter_knowledge_text=chapter_knowledge_text,
                max_retries=max_retries,
                difficulty=difficulty,
                blooms_level=blooms_level,
                blueprint=blueprint,
                num_tasks=num_tasks,
                chapter_id=chapter_id,
                chapter_relpath=chapter_relpath,
                blueprint_key=diff_bpt_combo,
                chapter_q_start=chapter_q_counter,
                verification_log=chapter_verification_logs,
                previous_questions=chapter_prev_questions,
                checkpoint_path=checkpoint_path,
                checkpoint_every=2,
                checkpoint_metadata=checkpoint_metadata,
                resume_from_checkpoint=True,
            )

            if not tasks:
                logger.error(
                    f"Failed to generate tasks for {chapter_id} | {difficulty} - {blooms_level}"
                )
                continue

            chapter_q_counter += len(tasks)
            all_tasks.extend(tasks)
            logger.info(
                f"Accumulated {len(all_tasks)} total tasks so far for {chapter_id}"
            )

            if not all_tasks:
                logger.error(
                    f"No tasks generated for chapter {book_name}/{chapter_id}; nothing to save."
                )
                continue

            # ---- deduplication ----
            dedup_cfg = agent_cfg.get("dedup", {}) or {}
            if bool(dedup_cfg.get("enabled", False)):
                threshold = float(dedup_cfg.get("threshold", 0.90))
                embedding_model = str(
                    dedup_cfg.get("embedding_model", "text-embedding-3-small")
                )
                keep_policy = str(dedup_cfg.get("keep_policy", "first"))
                cache_embeddings = bool(dedup_cfg.get("cache_embeddings", True))
                save_discarded = bool(dedup_cfg.get("save_discarded", True))

                cache_path = None
                if cache_embeddings:
                    cache_path = chapter_out_path.parent / "embedding_cache.json"

                kept, discarded, report = deduplicate_tasks_for_chapter(
                    tasks=all_tasks,
                    chapter_id=chapter_id,
                    embedding_model=embedding_model,
                    threshold=threshold,
                    keep_policy=keep_policy,
                    cache_path=cache_path,
                )

                kept = assign_chapter_level_task_ids(
                    kept_tasks=kept,
                    chapter_id=chapter_id,
                )

                report_path = chapter_out_path.parent / "dedup_report.json"
                report_path.write_text(
                    json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
                )
                logger.info(
                    f"Dedup: kept {len(kept)}/{len(all_tasks)} tasks. Report → {report_path}"
                )

                if save_discarded and discarded:
                    discarded = mark_discarded_metadata(
                        discarded_tasks=discarded,
                        chapter_id=chapter_id,
                        dedup_report=report,
                    )
                    discarded_path = chapter_out_path.parent / "discarded_tasks.json"
                    discarded_tasks_to_save = discarded
                else:
                    discarded_tasks_to_save = None

                all_tasks = kept
            else:
                discarded_tasks_to_save = None

            now = datetime.now(timezone.utc)
            metadata = PipelineMetadata(
                experiment_id=experiment_id,
                output_base_dir=str(output_base_dir),
                timestamp=now.isoformat().replace("+00:00", "Z"),
                input_stage_tag=capabilities_input_tag,
                output_stage_tag=out_tag,
                resume=bool(resume_tag),
            )

            stats_path = chapter_out_path.parent / "verification_stats.json"
            stats_payload = {
                "chapter_id": chapter_id,
                "chapter_relpath": chapter_relpath,
                "book_name": book_name,
                "num_verifier_calls": len(chapter_verification_logs),
                "verification_logs": chapter_verification_logs,
            }
            stats_path.write_text(
                json.dumps(stats_payload, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
            logger.info(f"Saved verification stats → {stats_path}")

            save_tasks(all_tasks, metadata, chapter_out_path)
            logger.info(f"Saved {len(all_tasks)} tasks → {chapter_out_path}")

            if discarded_tasks_to_save:
                discarded_path = chapter_out_path.parent / "discarded_tasks.json"
                save_tasks(discarded_tasks_to_save, metadata, discarded_path)
                logger.info(
                    f"Saved {len(discarded_tasks_to_save)} discarded tasks → {discarded_path}"
                )
    finally:
        await designer_client.close()
        await verifier_client.close()
    logger.info("Done.")
    return out_tag


def run_from_stage3(
    *,
    experiment_id: str,
    output_base_dir: Path,
    capabilities_tag: str,
    tasks_tag: Optional[str] = None,
) -> str:
    """Run agentic task generation from Stage 3 and return tasks_tag."""
    return asyncio.run(
        run_pipeline(
            experiment_id_override=experiment_id,
            output_base_dir_override=output_base_dir,
            tasks_tag_override=tasks_tag,
            capabilities_tag_override=capabilities_tag,
        )
    )


async def main() -> None:
    """CLI entrypoint."""
    await run_pipeline()


if __name__ == "__main__":
    asyncio.run(main())
