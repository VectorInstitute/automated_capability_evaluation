"""Source file for the agentic pipeline to generate and verify tasks."""

import json
import logging
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from src.schemas.capability_schemas import Capability
from src.schemas.metadata_schemas import PipelineMetadata
from src.schemas.task_schemas import Task
from src.task_generation.designer_agent import DesignerAgent
from src.task_generation.json_response_utils import parse_json_like
from src.task_generation.verifier_agent import VerifierAgent


logger = logging.getLogger(__name__)


@dataclass
class CandidateState:
    """In-flight candidate state carried through repair and verification steps."""

    qcore: Union[Dict[str, Any], List[Any], str]
    trace_part: Dict[str, Any]
    solution_part: Dict[str, Any]
    candidate_label: str


def _qa_pair_text(t: Task) -> str:
    """Create QA pair for anti-duplication checks."""
    q = (t.task_statement or "").strip()
    meta = t.generation_metadata or {}
    ca = str(meta.get("correct_answer") or "").strip()

    ans_text = ""
    for ch in t.choices or []:
        if str(ch.get("label", "")).strip() == ca:
            ans_text = str(ch.get("solution", "")).strip()
            break

    a = ans_text or ca
    return f"Question: {q}; Answer: {a}" if q else ""


def _prompt_memory_seed_index(task: Task) -> Optional[int]:
    """Extract the seed-generation index used for prompt-memory grouping."""
    meta = task.generation_metadata or {}
    raw_seed_index = meta.get("seed_generation_index")
    if raw_seed_index is None:
        return None
    try:
        return int(raw_seed_index)
    except (TypeError, ValueError):
        return None


def _rebuild_prompt_previous_questions(tasks: List[Task]) -> List[str]:
    """Rebuild prompt memory with one representative QA pair per seed generation."""
    prompt_previous_questions: List[str] = []
    seen_seed_indexes: set[int] = set()

    for task in tasks:
        qa = _qa_pair_text(task)
        if not qa:
            continue

        seed_index = _prompt_memory_seed_index(task)
        if seed_index is None:
            # Fall back to keeping the entry when legacy checkpoints lack seed metadata.
            prompt_previous_questions.append(qa)
            continue

        if seed_index in seen_seed_indexes:
            continue

        seen_seed_indexes.add(seed_index)
        prompt_previous_questions.append(qa)

    return prompt_previous_questions


def _is_passing(report: Dict[str, Any]) -> bool:
    """Determine if the verification report indicates a passing result."""
    overall = report.get("overall_verdict")
    if isinstance(overall, str):
        return overall.strip().lower() == "pass"
    return False


def _norm_yes_no(value: Any) -> str:
    """Normalize verifier status values to a stable yes/no vocabulary."""
    s = str(value or "").strip().lower()
    if s in {"yes", "pass", "true"}:
        return "yes"
    if s in {"no", "fail", "false"}:
        return "no"
    return s


def _split_parts(
    one_obj: Union[Dict[str, Any], str],
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Split the question dict into three parts.

    1. The main question content (with trace/solution fields removed)
    2. The trace part (fields related to solution graph or steps)
    3. The solution part (fields related to complete solution or explanation)

    Args:
        one_obj: The original question dict.

    Returns
    -------
        A tuple of (main_content, trace_part, solution_part) where:
    """
    if isinstance(one_obj, str):
        q_obj: Dict[str, Any] = {"question": one_obj}
        return q_obj, {}, {}

    q_obj = dict(one_obj)  # shallow copy

    trace_part: Dict[str, Any] = {}
    solution_part: Dict[str, Any] = {}

    # ---- extract trace ----
    sg = (
        q_obj.pop("solution_graph", None)
        or q_obj.pop("reasoning_graph", None)
        or q_obj.pop("graph", None)
    )
    if sg is not None:
        trace_part["solution_graph"] = sg

    ss = q_obj.pop("solution_steps", None) or q_obj.pop("steps", None)
    if ss is not None:
        trace_part["solution_steps"] = ss

    # ---- extract full solution ----
    cs = (
        q_obj.pop("complete_solution", None)
        or q_obj.pop("solution", None)
        or q_obj.pop("explanation", None)
    )
    if cs is not None:
        solution_part["complete_solution"] = cs

    return q_obj, trace_part, solution_part


def is_qcore_dict(x: Any) -> bool:
    """Check if the given object conforms to the expected qcore dict structure."""
    return (
        isinstance(x, dict)
        and isinstance(x.get("question"), str)
        and x["question"].strip()
        and isinstance(x.get("options"), dict)
        and isinstance(x.get("correct_answer"), str)
    )


def _looks_like_verification_report(x: Any) -> bool:
    """Detect verifier/fallback report-shaped payloads."""
    if not isinstance(x, dict):
        return False
    report_keys = {
        "overall_verdict",
        "json_format_valid",
        "mcq_integrity",
        "blooms_alignment",
        "constraint_compliance",
        "question_evaluation",
    }
    return len(report_keys.intersection(x.keys())) >= 2


def _wrap_qcore(obj: Union[Dict[str, Any], str]) -> Dict[str, Any]:
    """
    Normalize any single-question representation into a qcore dict.

    Args:
        obj: The original question content, which can be a dict or a raw string.

    Returns
    -------
        A dict with at least a "question" field, and optionally "options" and "
    """
    if isinstance(obj, dict):
        # Already a dict; assume it's the qcore
        return obj

    # If it's a raw string, treat it as the question stem
    s = str(obj).strip()
    return {"question": s, "options": {}, "correct_answer": ""}


def _ensure_json_string(content: Union[Dict[str, Any], List[Any], str]) -> str:
    """Ensure the content is a JSON-formatted string."""
    if isinstance(content, (dict, list)):
        return json.dumps(content, indent=2, ensure_ascii=False)

    s = str(content)
    parsed = parse_json_like(s)
    if parsed is not None:
        return json.dumps(parsed, indent=2, ensure_ascii=False)
    return s


def _load_checkpoint(
    path: Path,
) -> Tuple[List[Task], List[Dict[str, Any]], List[Dict[str, Any]], int]:
    """
    Load tasks and verifier logs from a checkpoint JSON.

    Args:
        path: The path to the checkpoint file.

    Returns
    -------
        Tuple[List[Task], List[Dict[str, Any]], List[Dict[str, Any]], int]:
        Loaded tasks, verifier logs, token usage logs, and processed seed
        generation attempts.
    """
    if not path.exists():
        return [], [], [], 0

    data = json.loads(path.read_text(encoding="utf-8"))
    raw_tasks = data.get("tasks", [])
    raw_verification_logs = data.get("verification_logs", [])
    raw_token_usage_logs = data.get("token_usage_logs", [])
    raw_generation_state = data.get("generation_state", {})

    tasks = [Task.from_dict(td) for td in raw_tasks]
    verification_logs = (
        [log for log in raw_verification_logs if isinstance(log, dict)]
        if isinstance(raw_verification_logs, list)
        else []
    )
    token_usage_logs = (
        [log for log in raw_token_usage_logs if isinstance(log, dict)]
        if isinstance(raw_token_usage_logs, list)
        else []
    )
    generation_attempts = 0
    if isinstance(raw_generation_state, dict):
        generation_attempts = int(raw_generation_state.get("generation_attempts", 0))
    return tasks, verification_logs, token_usage_logs, max(generation_attempts, 0)


def _save_checkpoint_snapshot(
    passed_tasks: List[Task],
    verification_log: List[Dict[str, Any]],
    token_usage_log: Optional[List[Dict[str, Any]]],
    checkpoint_path: Optional[Path],
    checkpoint_metadata: Optional[PipelineMetadata],
    generation_attempts: int,
) -> None:
    """Save a snapshot of the current passed tasks and verifier logs."""
    if checkpoint_path is None or checkpoint_metadata is None:
        return
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": checkpoint_metadata.to_dict(),
        "tasks": [task.to_dict() for task in passed_tasks],
        "verification_logs": verification_log,
        "token_usage_logs": token_usage_log or [],
        "generation_state": {
            "generation_attempts": generation_attempts,
        },
    }
    checkpoint_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def _pack_to_schema(
    content: Union[Dict[str, Any], List[Any], str],
    solution_trace: Optional[Dict[str, Any]],
    solution_full: Optional[Dict[str, Any]],
    *,
    capability: Capability,
    capability_source_mode: str,
    num_tasks: int,
    chapter_id: Optional[str],
    chapter_relpath: Optional[str],
    difficulty: str,
    blooms_level: str,
    blueprint_key: str,
    chapter_q_start: int = 0,
    task_id_start: int = 0,
) -> List[Task]:
    """
    Convert model output into List[Task].

    Args:
        content: The main question content (dict, list, or raw string).
        solution_trace: Optional dict containing solution graph or steps.
        solution_full: Optional dict containing complete solution or explanation.
        num_tasks: Number of tasks to produce (if content contains multiple).
        chapter_id: Optional chapter identifier for metadata.
        chapter_relpath: Optional chapter relative path for metadata.
        difficulty: Difficulty level for metadata.
        blooms_level: Bloom's taxonomy level for metadata.
        blueprint_key: Key representing the blueprint for metadata.
        chapter_q_start: Starting index for chapter question numbering (for metadata).
        task_id_start: Starting index for task IDs (for uniqueness within batch).

    Returns
    -------
        List[Task]: A list of Task objects conforming to the schema.
    """
    # ---- Parse content into Python object ----
    obj: Any = content
    if isinstance(content, str):
        try:
            obj = json.loads(content)
        except json.JSONDecodeError:
            obj = content

    # ---- Pull items list ----
    items: List[Any] = []
    if isinstance(obj, dict):
        if isinstance(obj.get("questions"), list):
            items = obj["questions"]
        elif isinstance(obj.get("tasks"), list):
            items = obj["tasks"]
        else:
            items = [obj]
    elif isinstance(obj, list):
        items = obj
    else:
        items = [{"question": str(obj), "options": {}, "correct_answer": None}]

    if isinstance(num_tasks, int) and num_tasks > 0:
        items = items[:num_tasks]

    # ---- Normalize trace/solution payloads into flat canonical fields ----
    trace_meta: Dict[str, Any] = {}
    if solution_trace is not None:
        # Expected shape: {"solution_graph": ..., "solution_steps": ...}
        if isinstance(solution_trace, dict):
            if (
                "solution_graph" in solution_trace
                and solution_trace["solution_graph"] is not None
            ):
                trace_meta["solution_graph"] = solution_trace["solution_graph"]
            # accept both names, just in case
            if (
                "solution_steps" in solution_trace
                and solution_trace["solution_steps"] is not None
            ):
                trace_meta["solution_steps"] = solution_trace["solution_steps"]
            elif "steps" in solution_trace and solution_trace["steps"] is not None:
                trace_meta["solution_steps"] = solution_trace["steps"]
        else:
            # If someone passed the raw graph object directly
            trace_meta["solution_graph"] = solution_trace

    solution_meta: Dict[str, Any] = {}
    if solution_full is not None:
        # Expected shape: {"complete_solution": ...}
        if isinstance(solution_full, dict):
            if (
                "complete_solution" in solution_full
                and solution_full["complete_solution"] is not None
            ):
                solution_meta["complete_solution"] = solution_full["complete_solution"]
            elif "solution" in solution_full and solution_full["solution"] is not None:
                solution_meta["complete_solution"] = solution_full["solution"]
            elif (
                "explanation" in solution_full
                and solution_full["explanation"] is not None
            ):
                solution_meta["complete_solution"] = solution_full["explanation"]
        else:
            # If someone passed a raw solution string/object directly
            solution_meta["complete_solution"] = solution_full

    # ---- Build Task objects ----
    tasks: List[Task] = []

    for idx, item in enumerate(items):
        k = task_id_start + idx
        task_id = f"task_{k:03d}"

        task_statement = ""
        correct_answer = None
        choices: Optional[List[Dict[str, str]]] = None
        extra_fields: Dict[str, Any] = {}

        if isinstance(item, str):
            task_statement = item

        elif isinstance(item, dict):
            task_statement = (item.get("question") or item.get("task") or "").strip()
            correct_answer = item.get("correct_answer")
            options = item.get("options")

            if isinstance(options, dict):
                order = ["A", "B", "C", "D", "E"]
                labels = [k for k in order if k in options] + [
                    k
                    for k in options.keys()  # noqa: SIM118
                    if k not in order  # noqa: SIM118
                ]
                choices = [
                    {"label": str(k), "solution": str(options[k])} for k in labels
                ]
            elif isinstance(options, list):
                if (
                    options
                    and isinstance(options[0], dict)
                    and "label" in options[0]
                    and "solution" in options[0]
                ):
                    choices = [
                        {"label": str(o["label"]), "solution": str(o["solution"])}
                        for o in options
                    ]
                else:
                    labels = ["A", "B", "C", "D", "E"]
                    choices = [
                        {"label": labels[i] if i < 5 else str(i), "solution": str(o)}
                        for i, o in enumerate(options)
                    ]
            else:
                choices = None

            extra_fields = {
                k: v
                for k, v in item.items()
                if k not in {"question", "task", "options", "correct_answer"}
            }

            if not task_statement:
                task_statement = json.dumps(item, ensure_ascii=False)

        else:
            task_statement = str(item)

        validated_mcq = _normalize_and_validate_mcq_fields(
            task_id=task_id,
            task_statement=task_statement,
            choices=choices,
            correct_answer=correct_answer,
        )
        if validated_mcq is None:
            continue
        task_statement, choices, correct_answer = validated_mcq

        generation_metadata: Dict[str, Any] = {
            "chapter_id": chapter_id,
            "chapter_relpath": chapter_relpath,
            "capability_source_mode": capability_source_mode,
            "blueprint_key": blueprint_key,
            "correct_answer": correct_answer,
            "chapter_question_id": f"{chapter_id}_q_{(chapter_q_start + idx):03d}",
            **trace_meta,
            **solution_meta,
            **extra_fields,
        }

        tasks.append(
            Task(
                task_id,
                task_statement,
                capability,
                task_type="multiple_choice",
                solution_type="multiple_choice",
                difficulty=difficulty,
                bloom_level=blooms_level,
                choices=choices,
                generation_metadata=generation_metadata,
            )
        )

    return tasks


def _append_token_usage_record(
    token_usage_log: Optional[List[Dict[str, Any]]],
    *,
    usage: Optional[Dict[str, Any]],
    stage: str,
    model_role: str,
    task_batch_id: str,
    chapter_id: Optional[str],
    chapter_relpath: Optional[str],
    blueprint_key: Optional[str],
    difficulty: str,
    blooms_level: str,
    seed_generation_index: Optional[int] = None,
    candidate_label: Optional[str] = None,
    candidate_index_within_seed: Optional[int] = None,
    attempt_index: Optional[int] = None,
    hardening_round_index: Optional[int] = None,
) -> None:
    """Append a normalized token-usage record for one model call."""
    if token_usage_log is None:
        return
    usage = usage or {}
    token_usage_log.append(
        {
            "task_batch_id": task_batch_id,
            "stage": stage,
            "model_role": model_role,
            "chapter_id": chapter_id,
            "chapter_relpath": chapter_relpath,
            "blueprint_key": blueprint_key,
            "difficulty": difficulty,
            "blooms_level": blooms_level,
            "seed_generation_index": seed_generation_index,
            "candidate_label": candidate_label,
            "candidate_index_within_seed": candidate_index_within_seed,
            "attempt_index": attempt_index,
            "hardening_round_index": hardening_round_index,
            "input_tokens": usage.get("input_tokens"),
            "output_tokens": usage.get("output_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "usage_available": bool(usage.get("usage_available", False)),
        }
    )


def _append_choices_to_task_statement(
    task_statement: str,
    choices: Optional[List[Dict[str, str]]],
) -> str:
    """Append a options block to task_statement for backward compatibility."""
    if not task_statement or not choices:
        return task_statement

    normalized_statement = task_statement.rstrip()
    if "\nOptions:\n" in normalized_statement:
        return normalized_statement

    option_lines: List[str] = []
    for choice in choices:
        label = str(choice.get("label", "")).strip()
        solution = str(choice.get("solution", "")).strip()
        if label and solution:
            option_lines.append(f"{label}. {solution}")

    if not option_lines:
        return normalized_statement

    return normalized_statement + "\n\nOptions:\n" + "\n".join(option_lines)


def _normalize_and_validate_mcq_fields(  # noqa: PLR0911
    *,
    task_id: str,
    task_statement: str,
    choices: Optional[List[Dict[str, str]]],
    correct_answer: Any,
) -> Optional[Tuple[str, List[Dict[str, str]], str]]:
    """Normalize MCQ fields and reject inconsistent answer-key/choice combinations."""
    if not task_statement.strip():
        logger.warning("Skipping %s because task_statement is empty.", task_id)
        return None
    if not choices:
        logger.warning("Skipping %s because choices are missing.", task_id)
        return None

    normalized_choices: List[Dict[str, str]] = []
    seen_labels: set[str] = set()
    for choice in choices:
        label = str(choice.get("label", "")).strip().upper()
        solution = str(choice.get("solution", "")).strip()
        if not label or not solution:
            logger.warning(
                "Skipping %s because a choice has missing label or solution.", task_id
            )
            return None
        if label in seen_labels:
            logger.warning(
                "Skipping %s because choice label %s is duplicated.", task_id, label
            )
            return None
        seen_labels.add(label)
        normalized_choices.append({"label": label, "solution": solution})

    normalized_answer = str(correct_answer or "").strip().upper()
    if not normalized_answer:
        logger.warning("Skipping %s because correct_answer is missing.", task_id)
        return None

    answer_choice = next(
        (
            choice
            for choice in normalized_choices
            if choice["label"] == normalized_answer
        ),
        None,
    )
    if answer_choice is None:
        logger.warning(
            "Skipping %s because correct_answer=%s is not present in choices.",
            task_id,
            normalized_answer,
        )
        return None

    normalized_statement = _append_choices_to_task_statement(
        task_statement,
        normalized_choices,
    )
    if f"{normalized_answer}. {answer_choice['solution']}" not in normalized_statement:
        logger.warning(
            "Skipping %s because task_statement options block is inconsistent with choices.",
            task_id,
        )
        return None

    return normalized_statement, normalized_choices, normalized_answer


def _format_feedback(report: Dict[str, Any]) -> str:
    """Format the verifier feedback from the report."""
    top_keys = [
        "json_format_valid",
        "mcq_integrity",
        "blooms_alignment",
        "constraint_compliance",
    ]
    verdict = report.get("overall_verdict") or "Unknown"
    explanation = report.get("explanation") or "No explanation provided."

    checks = [
        f"{k}={report[k].strip()}"
        for k in top_keys
        if isinstance(report.get(k), str) and report[k].strip()
    ]
    checks_line = " | ".join(checks)

    q_lines: List[str] = []
    evals = report.get("question_evaluation", {})
    if isinstance(evals, dict):
        evals = [dict(evals, question_index=1)]
    if isinstance(evals, list):
        for it in evals:
            if not isinstance(it, dict):
                continue
            q = it.get("question_index", "?")
            distractors_plausible = str(it.get("distractors_plausible", "")).strip()
            issues = it.get("main_issues", [])
            fix = it.get("fix", "")
            issue_list = (
                [str(x).strip() for x in issues]
                if isinstance(issues, list)
                else ([str(issues).strip()] if issues else [])
            )
            issue_list = [x for x in issue_list if x]
            fix = fix.strip() if isinstance(fix, str) else ""
            if issue_list or fix:
                parts = []
                if distractors_plausible:
                    parts.append(f"distractors_plausible={distractors_plausible}")
                if issue_list:
                    parts.append("Issues: " + "; ".join(issue_list))
                if fix:
                    parts.append("Fix: " + fix)
                q_lines.append(f"Q{q}: " + " | ".join(parts))

    out = [f"Verdict: {verdict}", f"Overall: {explanation}"]
    if checks_line:
        out.append(f"Checks: {checks_line}")
    if q_lines:
        out.append("Per-question:\n" + "\n".join(q_lines))
    return "\n".join(out)


# Task Generation Loop (one-by-one generation with verification-driven repair)
async def run_task_generation_loop(
    designer_factory: Callable[[], DesignerAgent],
    verifier_factory: Callable[[], VerifierAgent],
    capability: Capability,
    domain: str,  # TODO: remove domain from args if not needed in prompts
    context_text: str,
    chapter_knowledge_text: str,
    previous_questions: List[str],
    capability_source_mode: str = "placeholder",
    max_retries: int = 5,
    difficulty: str = "Easy",  # TODO: remove structured difficulty schema or enum
    blooms_level: str = "Remember",  # TODO: remove structured blooms_level schema or enum
    num_tasks: int = 100,
    hardening_rounds: int = 5,
    seed_generation_target: Optional[int] = None,
    chapter_id: Optional[str] = None,
    chapter_relpath: Optional[str] = None,
    blueprint_key: Optional[str] = None,
    chapter_q_start: int = 0,
    verification_log: Optional[List[Dict[str, Any]]] = None,
    token_usage_log: Optional[List[Dict[str, Any]]] = None,
    checkpoint_path: Optional[Path] = None,
    checkpoint_every: int = 10,
    checkpoint_metadata: Optional[PipelineMetadata] = None,
    resume_from_checkpoint: bool = False,
) -> Optional[List[Task]]:
    """
    Run the agentic pipeline for task generation.

    - Generates ONE problem at a time (instead of generating a batch of problems).
    - For each generated problem, run Steps 2–8.
    - Repeats over a fixed number of seed generations.

    Args:
        designer_factory: Factory function to create a DesignerAgent.
        verifier_factory: Factory function to create a VerifierAgent.
        capability: Capability metadata to attach to each generated Task.
        capability_source_mode: Source mode for capability metadata (traceability).
        domain: The domain or subject area for the tasks.
        context_text: Relevant excerpts from the chapter (source grounding).
        chapter_knowledge_text: Chapter-specific knowledge text.
        blueprint: The blueprint or template for the tasks.
        previous_questions: Previously accepted Q/A pairs from this chapter (anti-dup).
        max_retries: Maximum number of retries for verification failures (per-question).
        difficulty: The difficulty level for the tasks.
        blooms_level: The Bloom's taxonomy level for the tasks.
        num_tasks: Maximum number of passing tasks to keep.
        hardening_rounds: Maximum number of hardening rounds per seed task.
        seed_generation_target: Number of unique seed generations to run.
        chapter_id: Optional chapter identifier.
        chapter_relpath: Optional chapter relative path.
        blueprint_key: Optional key representing the blueprint.
        chapter_q_start: Starting index for chapter question numbering.
        verification_log: Optional list to log verification reports.
        token_usage_log: Optional list to log token usage across model calls.

    Returns
    -------
        List[Task] on success, None if none pass.
    """
    task_batch_id = f"batch_{uuid.uuid4().hex[:6]}"
    logger.info(
        f"[{task_batch_id}] Starting generation for chapter={chapter_id} combo={difficulty}/{blooms_level}"
    )

    logger.info(f"[{task_batch_id}] Step 1: Generating one question at a time...")

    passed_tasks: List[Task] = []
    prompt_previous_questions = previous_questions
    prompt_previous_questions.clear()
    prompt_memory_seed_indexes: set[int] = set()
    task_seq = chapter_q_start
    if seed_generation_target is None:
        seed_generation_target = num_tasks

    generation_attempts = 0
    max_generation_attempts = max(seed_generation_target, 0)
    logger.info(
        f"[{task_batch_id}] Will run {max_generation_attempts} seed generation(s) to collect up to {num_tasks} passing hardened tasks."
    )

    # ---- Resume from checkpoint if enabled ----
    if resume_from_checkpoint and checkpoint_path and checkpoint_path.exists():
        (
            loaded_tasks,
            loaded_verification_logs,
            loaded_token_usage_logs,
            loaded_generation_attempts,
        ) = _load_checkpoint(checkpoint_path)
        if verification_log is not None:
            verification_log.clear()
            verification_log.extend(loaded_verification_logs)
        if token_usage_log is not None:
            current_token_usage_logs = list(token_usage_log)
            token_usage_log.clear()
            token_usage_log.extend(loaded_token_usage_logs)
            token_usage_log.extend(current_token_usage_logs)
        if loaded_tasks:
            passed_tasks = loaded_tasks

            # Rebuild prompt memory with one representative per seed generation.
            prompt_previous_questions.extend(
                _rebuild_prompt_previous_questions(passed_tasks)
            )
            for task in passed_tasks:
                seed_index = _prompt_memory_seed_index(task)
                if seed_index is not None:
                    prompt_memory_seed_indexes.add(seed_index)

            # Advance task_seq so new tasks get new IDs
            task_seq = chapter_q_start + len(passed_tasks)
        generation_attempts = loaded_generation_attempts

        if loaded_tasks or loaded_generation_attempts > 0:
            logger.info(
                f"[{task_batch_id}] Resumed from checkpoint: {len(passed_tasks)} passed tasks loaded "
                f"from {checkpoint_path}; processed_seed_generations={generation_attempts}/{max_generation_attempts}"
            )

    while generation_attempts < max_generation_attempts:
        i = generation_attempts
        generation_attempts += 1

        logger.info(
            f"[{task_batch_id}] Seed generation {i + 1}/{max_generation_attempts}: current_passed={len(passed_tasks)}/{num_tasks}"
        )

        # --- Step 1: INITIAL GENERATION ---
        designer = designer_factory()
        one_content, one_prompt, one_usage = await designer.generate_draft(
            chapter_excerpts=context_text,
            chapter_knowledge_text=chapter_knowledge_text,
            difficulty=difficulty,
            blooms_level=blooms_level,
            previous_questions=prompt_previous_questions,
        )
        _append_token_usage_record(
            token_usage_log,
            usage=one_usage,
            stage="generate_draft",
            model_role="designer",
            task_batch_id=task_batch_id,
            chapter_id=chapter_id,
            chapter_relpath=chapter_relpath,
            blueprint_key=blueprint_key,
            difficulty=difficulty,
            blooms_level=blooms_level,
            seed_generation_index=i + 1,
        )
        # Normalize generation output into dict
        one_obj: Any = one_content
        if isinstance(one_obj, str):
            preview = (one_obj[:200] + "…") if len(one_obj) > 200 else one_obj
            logger.warning(
                f"[{task_batch_id}] Q{i + 1} generator returned non-JSON; retrying Step 1 once. Preview={preview!r}"
            )

            # Try one more time with schema reminder prompt if response is non-dict.
            designer = designer_factory()
            (
                one_content_retry,
                one_prompt_retry,
                one_usage_retry,
            ) = await designer.generate_draft(
                chapter_excerpts=context_text,
                chapter_knowledge_text=chapter_knowledge_text,
                difficulty=difficulty,
                blooms_level=blooms_level,
                previous_questions=prompt_previous_questions,
            )
            _append_token_usage_record(
                token_usage_log,
                usage=one_usage_retry,
                stage="generate_draft_retry",
                model_role="designer",
                task_batch_id=task_batch_id,
                chapter_id=chapter_id,
                chapter_relpath=chapter_relpath,
                blueprint_key=blueprint_key,
                difficulty=difficulty,
                blooms_level=blooms_level,
                seed_generation_index=i + 1,
            )
            one_obj = one_content_retry
            if isinstance(one_obj, str):
                preview = (one_obj[:200] + "…") if len(one_obj) > 200 else one_obj
                logger.warning(
                    f"[{task_batch_id}] Q{i + 1} generator retry still non-JSON; skipping. Preview={preview!r}"
                )
                if (
                    checkpoint_every > 0
                    and checkpoint_path
                    and checkpoint_metadata
                    and verification_log is not None
                ):
                    _save_checkpoint_snapshot(
                        passed_tasks,
                        verification_log,
                        token_usage_log,
                        checkpoint_path,
                        checkpoint_metadata,
                        generation_attempts,
                    )
                continue

        if not str(one_obj.get("question") or "").strip():
            logger.warning(
                f"[{task_batch_id}] Q{i + 1} missing 'question' after Step 1; skipping."
            )
            if (
                checkpoint_every > 0
                and checkpoint_path
                and checkpoint_metadata
                and verification_log is not None
            ):
                _save_checkpoint_snapshot(
                    passed_tasks,
                    verification_log,
                    token_usage_log,
                    checkpoint_path,
                    checkpoint_metadata,
                    generation_attempts,
                )
            continue

        # --- Step 1.5: ITERATIVE TASK HARDENING (carry every round forward) ---
        logger.info(
            f"[{task_batch_id}] Q{i + 1} Step 1.5: Iterative hardening (max rounds={hardening_rounds})..."
        )
        iter_obj: Dict[str, Any] = dict(one_obj)
        hidden_answer = str(iter_obj.get("correct_answer") or "").strip().upper()
        hardened_round_candidates: List[Dict[str, Any]] = []

        for h_idx in range(hardening_rounds):
            round_num = h_idx + 1
            logger.info(
                f"[{task_batch_id}] Q{i + 1} Step 1.5 Round {round_num}/{hardening_rounds}: harden candidate..."
            )

            candidate_for_hardening_obj = {
                "question": str(iter_obj.get("question") or "").strip(),
                "options": iter_obj.get("options", {}),
                "solution_graph": iter_obj.get("solution_graph"),
                "complete_solution": iter_obj.get("complete_solution"),
            }
            candidate_for_hardening = _ensure_json_string(candidate_for_hardening_obj)

            designer = designer_factory()
            (
                hardened_content,
                hardening_prompt,
                harden_usage,
            ) = await designer.harden_task(
                chapter_excerpts=context_text,
                chapter_knowledge_summary=chapter_knowledge_text,
                difficulty=difficulty,
                blooms_level=blooms_level,
                candidate_question_and_solution_graph=candidate_for_hardening,
            )
            _append_token_usage_record(
                token_usage_log,
                usage=harden_usage,
                stage="harden_task",
                model_role="designer",
                task_batch_id=task_batch_id,
                chapter_id=chapter_id,
                chapter_relpath=chapter_relpath,
                blueprint_key=blueprint_key,
                difficulty=difficulty,
                blooms_level=blooms_level,
                seed_generation_index=i + 1,
                hardening_round_index=round_num,
            )
            if (
                isinstance(hardened_content, dict)
                and str(hardened_content.get("question") or "").strip()
            ):
                # Preserve hidden answer if the hardener omits it.
                if not str(hardened_content.get("correct_answer") or "").strip():
                    hardened_content["correct_answer"] = hidden_answer
                iter_obj = hardened_content
                hidden_answer = (
                    str(iter_obj.get("correct_answer") or "").strip().upper()
                )
                hardened_round_candidates.append(dict(iter_obj))
            else:
                logger.warning(
                    f"[{task_batch_id}] Q{i + 1} Step 1.5 Round {round_num}: invalid hardened payload; skipping this round output."
                )

            logger.debug(f"[{task_batch_id}] Hardening content: {hardened_content}")
            logger.debug(f"[{task_batch_id}] Hardening prompt: {hardening_prompt}")

        if not hardened_round_candidates:
            logger.warning(
                f"[{task_batch_id}] Q{i + 1} Step 1.5: no valid hardened outputs; proceeding with the original seed candidate only."
            )

        candidate_records: List[Dict[str, Any]] = [
            {
                "candidate_obj": dict(one_obj),
                "candidate_origin": "seed",
                "is_seed_task": True,
                "hardening_round_index": 0,
                "candidate_label": f"SeedGeneration {i + 1}/{max_generation_attempts} SeedCandidate",
            }
        ]
        for h_idx, hardened_obj in enumerate(hardened_round_candidates, start=1):
            candidate_records.append(
                {
                    "candidate_obj": hardened_obj,
                    "candidate_origin": "hardening",
                    "is_seed_task": False,
                    "hardening_round_index": h_idx,
                    "candidate_label": (
                        f"SeedGeneration {i + 1}/{max_generation_attempts} HardeningRound "
                        f"{h_idx}/{len(hardened_round_candidates)}"
                    ),
                }
            )

        logger.info(
            f"[{task_batch_id}] Q{i + 1} Step 1.5: forwarding {len(candidate_records)} candidate(s) into the downstream pipeline "
            f"(seed=1, hardened={len(hardened_round_candidates)})."
        )

        for candidate_idx, candidate_record in enumerate(candidate_records, start=1):
            if len(passed_tasks) >= num_tasks:
                break

            candidate_obj = candidate_record["candidate_obj"]
            q_obj, trace_part, solution_part = _split_parts(candidate_obj)
            candidate_state = CandidateState(
                qcore=_wrap_qcore(q_obj),
                trace_part=trace_part,
                solution_part=solution_part,
                candidate_label=str(candidate_record["candidate_label"]),
            )

            logger.info(
                f"[{task_batch_id}] {candidate_state.candidate_label}: starting per-question pipeline"
            )

            # Per-question retries (verification-driven repair loop)
            for attempt in range(max_retries + 1):
                logger.info(
                    f"[{task_batch_id}] {candidate_state.candidate_label} Attempt {attempt + 1}/{max_retries + 1}"
                )

                # --- Step 2: INCLUDE NOTATION DEFINITIONS / CLARIFICATIONS ---
                logger.info(
                    f"[{task_batch_id}] {candidate_state.candidate_label} Step 2: Including clarification info..."
                )

                designer = designer_factory()
                current_qcore_as_str = _ensure_json_string(candidate_state.qcore)
                (
                    clarified_qcore,
                    clarification_prompt,
                    clarification_usage,
                ) = await designer.include_clarification_info(
                    candidate_question=current_qcore_as_str,
                )
                _append_token_usage_record(
                    token_usage_log,
                    usage=clarification_usage,
                    stage="include_clarification_info",
                    model_role="designer",
                    task_batch_id=task_batch_id,
                    chapter_id=chapter_id,
                    chapter_relpath=chapter_relpath,
                    blueprint_key=blueprint_key,
                    difficulty=difficulty,
                    blooms_level=blooms_level,
                    seed_generation_index=i + 1,
                    candidate_label=candidate_state.candidate_label,
                    candidate_index_within_seed=candidate_idx,
                    attempt_index=attempt,
                )

                if not is_qcore_dict(clarified_qcore):
                    retry_prompt = (
                        current_qcore_as_str + "\n\n[SCHEMA REMINDER]\n"
                        "Return ONLY a single JSON object.\n"
                        'Keys: "question" (string), "options" (object A-E strings), "correct_answer" (A-E).\n'
                        "Do not drop or rename keys.\n\n"
                        "Output format example (do not add any text outside JSON):\n"
                        "{\n"
                        '  "question": "<self-contained MCQ stem>",\n'
                        '  "options": { "A": "...", "B": "...", "C": "...", "D": "...", "E": "None of the above" },\n'
                        '  "correct_answer": "<one of: A|B|C|D|E>"\n'
                        "}\n"
                    )
                    (
                        clarified_qcore,
                        _,
                        clarification_retry_usage,
                    ) = await designer.include_clarification_info(
                        candidate_question=retry_prompt,
                    )
                    _append_token_usage_record(
                        token_usage_log,
                        usage=clarification_retry_usage,
                        stage="include_clarification_info_retry",
                        model_role="designer",
                        task_batch_id=task_batch_id,
                        chapter_id=chapter_id,
                        chapter_relpath=chapter_relpath,
                        blueprint_key=blueprint_key,
                        difficulty=difficulty,
                        blooms_level=blooms_level,
                        seed_generation_index=i + 1,
                        candidate_label=candidate_state.candidate_label,
                        candidate_index_within_seed=candidate_idx,
                        attempt_index=attempt,
                    )

                if is_qcore_dict(clarified_qcore):
                    candidate_state.qcore = clarified_qcore
                else:
                    logger.warning(
                        f"[{task_batch_id}] {candidate_state.candidate_label} Step 2 failed twice; skipping this candidate."
                    )
                    break

                logger.debug(
                    f"[{task_batch_id}] Clarification content: {clarified_qcore}"
                )
                logger.debug(
                    f"[{task_batch_id}] Clarification prompt: {clarification_prompt}"
                )
                # --- Step 3: VERIFY CORRECTNESS / MCQ INTEGRITY ---
                logger.info(
                    f"[{task_batch_id}] {candidate_state.candidate_label} Step 3: Verifying MCQ integrity..."
                )
                verifier = verifier_factory()
                integrity_input_str = _ensure_json_string(candidate_state.qcore)
                qcore_before_integrity = candidate_state.qcore

                (
                    mcq_fixed_full,
                    mcq_fixed_full_prompt,
                    mcq_integrity_usage,
                ) = await verifier.check_and_revise_mcq_option(
                    candidate_question=integrity_input_str,
                    chapter_excerpts=context_text,
                    chapter_knowledge_text=chapter_knowledge_text,
                    blooms_level=blooms_level,
                    solution_trace=candidate_state.trace_part,
                    solution_full=candidate_state.solution_part,
                )
                _append_token_usage_record(
                    token_usage_log,
                    usage=mcq_integrity_usage,
                    stage="check_and_revise_mcq_option",
                    model_role="verifier",
                    task_batch_id=task_batch_id,
                    chapter_id=chapter_id,
                    chapter_relpath=chapter_relpath,
                    blueprint_key=blueprint_key,
                    difficulty=difficulty,
                    blooms_level=blooms_level,
                    seed_generation_index=i + 1,
                    candidate_label=candidate_state.candidate_label,
                    candidate_index_within_seed=candidate_idx,
                    attempt_index=attempt,
                )

                mcq_fixed_full_str = _ensure_json_string(mcq_fixed_full)

                try:
                    mcq_fixed_full_obj = json.loads(mcq_fixed_full_str)
                except json.JSONDecodeError:
                    mcq_fixed_full_obj = None

                if isinstance(mcq_fixed_full_obj, dict):
                    q_obj_step3, trace_part_step3, solution_part_step3 = _split_parts(
                        mcq_fixed_full_obj
                    )
                    candidate_qcore = _wrap_qcore(q_obj_step3)
                    if is_qcore_dict(candidate_qcore):
                        candidate_state.qcore = candidate_qcore
                        candidate_state.trace_part = (
                            trace_part_step3 or candidate_state.trace_part
                        )
                        candidate_state.solution_part = (
                            solution_part_step3 or candidate_state.solution_part
                        )
                    else:
                        logger.warning(
                            f"[{task_batch_id}] {candidate_state.candidate_label} Step 3 produced non-MCQ payload; keeping prior candidate."
                        )
                        candidate_state.qcore = qcore_before_integrity
                else:
                    logger.warning(
                        f"[{task_batch_id}] {candidate_state.candidate_label} Step 3 produced non-MCQ payload; keeping prior candidate."
                    )
                    candidate_state.qcore = qcore_before_integrity

                logger.debug(
                    f"[{task_batch_id}] MCQ-integrity content: {mcq_fixed_full}"
                )
                logger.debug(
                    f"[{task_batch_id}] MCQ-integrity prompt: {mcq_fixed_full_prompt}"
                )

                # --- Step 4: REMOVE REDUNDANT INFO ---
                logger.info(
                    f"[{task_batch_id}] {candidate_state.candidate_label} Step 4: Removing redundant info..."
                )

                designer = designer_factory()
                mcq_integrity_as_str = _ensure_json_string(candidate_state.qcore)
                (
                    no_redundant_content,
                    no_redundant_prompt,
                    no_redundant_usage,
                ) = await designer.remove_redundant_info(
                    candidate_question=mcq_integrity_as_str,
                )
                _append_token_usage_record(
                    token_usage_log,
                    usage=no_redundant_usage,
                    stage="remove_redundant_info",
                    model_role="designer",
                    task_batch_id=task_batch_id,
                    chapter_id=chapter_id,
                    chapter_relpath=chapter_relpath,
                    blueprint_key=blueprint_key,
                    difficulty=difficulty,
                    blooms_level=blooms_level,
                    seed_generation_index=i + 1,
                    candidate_label=candidate_state.candidate_label,
                    candidate_index_within_seed=candidate_idx,
                    attempt_index=attempt,
                )
                if is_qcore_dict(
                    no_redundant_content
                ) and not _looks_like_verification_report(no_redundant_content):
                    candidate_state.qcore = no_redundant_content
                else:
                    logger.warning(
                        f"[{task_batch_id}] {candidate_state.candidate_label} Step 4 produced invalid payload; keeping prior candidate."
                    )

                logger.debug(
                    f"[{task_batch_id}] No-redundant content: {no_redundant_content}"
                )
                logger.debug(
                    f"[{task_batch_id}] No-redundant prompt: {no_redundant_prompt}"
                )
                # --- Step 5: REMOVE SOURCE REFERENCES ---
                logger.info(
                    f"[{task_batch_id}] {candidate_state.candidate_label} Step 5: Removing source references..."
                )

                designer = designer_factory()
                no_redundant_content_as_str = _ensure_json_string(candidate_state.qcore)
                (
                    no_source_content,
                    no_source_prompt,
                    no_source_usage,
                ) = await designer.remove_references(
                    candidate_question=no_redundant_content_as_str,
                )
                _append_token_usage_record(
                    token_usage_log,
                    usage=no_source_usage,
                    stage="remove_references",
                    model_role="designer",
                    task_batch_id=task_batch_id,
                    chapter_id=chapter_id,
                    chapter_relpath=chapter_relpath,
                    blueprint_key=blueprint_key,
                    difficulty=difficulty,
                    blooms_level=blooms_level,
                    seed_generation_index=i + 1,
                    candidate_label=candidate_state.candidate_label,
                    candidate_index_within_seed=candidate_idx,
                    attempt_index=attempt,
                )
                if is_qcore_dict(
                    no_source_content
                ) and not _looks_like_verification_report(no_source_content):
                    candidate_state.qcore = no_source_content
                else:
                    logger.warning(
                        f"[{task_batch_id}] {candidate_state.candidate_label} Step 5 produced invalid payload; keeping prior candidate."
                    )

                logger.debug(
                    f"[{task_batch_id}] No-source content: {no_source_content}"
                )
                logger.debug(f"[{task_batch_id}] No-source prompt: {no_source_prompt}")
                # --- Step 6: Check Soundness ---
                logger.info(
                    f"[{task_batch_id}] {candidate_state.candidate_label} Step 6: Checking soundness..."
                )

                designer = designer_factory()
                no_source_content_as_str = _ensure_json_string(candidate_state.qcore)
                (
                    clean_content,
                    soundness_prompt,
                    soundness_usage,
                ) = await designer.check_soundness(
                    candidate_question=no_source_content_as_str,
                )
                _append_token_usage_record(
                    token_usage_log,
                    usage=soundness_usage,
                    stage="check_soundness",
                    model_role="designer",
                    task_batch_id=task_batch_id,
                    chapter_id=chapter_id,
                    chapter_relpath=chapter_relpath,
                    blueprint_key=blueprint_key,
                    difficulty=difficulty,
                    blooms_level=blooms_level,
                    seed_generation_index=i + 1,
                    candidate_label=candidate_state.candidate_label,
                    candidate_index_within_seed=candidate_idx,
                    attempt_index=attempt,
                )
                if is_qcore_dict(clean_content) and not _looks_like_verification_report(
                    clean_content
                ):
                    candidate_state.qcore = clean_content
                else:
                    logger.warning(
                        f"[{task_batch_id}] {candidate_state.candidate_label} Step 6 produced invalid payload; keeping prior candidate."
                    )
                    clean_content = candidate_state.qcore

                logger.debug(f"[{task_batch_id}] Soundness content: {clean_content}")
                logger.debug(f"[{task_batch_id}] Soundness prompt: {soundness_prompt}")
                # --- Step 7: FINAL VERIFICATION (MCQ INTEGRITY, JSON FORMAT CHECK) ---
                logger.info(
                    f"[{task_batch_id}] {candidate_state.candidate_label} Step 7: Verifying..."
                )

                verifier = verifier_factory()
                verification_report, verification_usage = await verifier.verify_task(
                    candidate_output=clean_content,
                    blooms_level=blooms_level,
                )
                _append_token_usage_record(
                    token_usage_log,
                    usage=verification_usage,
                    stage="verify_task",
                    model_role="verifier",
                    task_batch_id=task_batch_id,
                    chapter_id=chapter_id,
                    chapter_relpath=chapter_relpath,
                    blueprint_key=blueprint_key,
                    difficulty=difficulty,
                    blooms_level=blooms_level,
                    seed_generation_index=i + 1,
                    candidate_label=candidate_state.candidate_label,
                    candidate_index_within_seed=candidate_idx,
                    attempt_index=attempt,
                )

                if is_qcore_dict(clean_content) and not _looks_like_verification_report(
                    clean_content
                ):
                    candidate_state.qcore = clean_content
                if verification_log is not None:
                    verification_log.append(
                        {
                            "task_batch_id": task_batch_id,
                            "attempt_index": attempt,
                            "attempt_human": f"{attempt + 1}/{max_retries + 1}",
                            "chapter_id": chapter_id,
                            "chapter_relpath": chapter_relpath,
                            "blueprint_key": blueprint_key,
                            "difficulty": difficulty,
                            "blooms_level": blooms_level,
                            "question_index_in_batch": task_seq,
                            "seed_generation_index": i + 1,
                            "seed_generation_target": max_generation_attempts,
                            "candidate_origin": candidate_record["candidate_origin"],
                            "is_seed_task": candidate_record["is_seed_task"],
                            "hardening_round_index": candidate_record[
                                "hardening_round_index"
                            ],
                            "candidate_index_within_seed": candidate_idx,
                            "candidate_total_within_seed": len(candidate_records),
                            "hardening_round_candidate_index": candidate_idx,
                            "hardening_round_candidate_total": len(candidate_records),
                            "summary": {
                                "overall_verdict": verification_report.get(
                                    "overall_verdict"
                                ),
                                "json_format_valid": verification_report.get(
                                    "json_format_valid"
                                ),
                                "mcq_integrity": verification_report.get(
                                    "mcq_integrity"
                                ),
                                "blooms_alignment": verification_report.get(
                                    "blooms_alignment"
                                ),
                                "constraint_compliance": verification_report.get(
                                    "constraint_compliance"
                                ),
                            },
                        }
                    )

                logger.info(
                    f"[{task_batch_id}] {candidate_state.candidate_label} Verification report: {verification_report}"
                )
                logger.debug(
                    f"[{task_batch_id}] Clean content for verification: {clean_content}"
                )
                if _is_passing(verification_report):
                    packed_clean_content: Dict[str, Any]
                    if isinstance(clean_content, dict):
                        packed_clean_content = dict(clean_content)
                    else:
                        packed_clean_content = {
                            "question": str(clean_content),
                            "options": {},
                            "correct_answer": "",
                        }
                    packed_clean_content["hardening_round_candidate_index"] = (
                        candidate_idx
                    )
                    packed_clean_content["hardening_round_candidate_total"] = len(
                        candidate_records
                    )
                    packed_clean_content["seed_generation_index"] = i + 1
                    packed_clean_content["seed_generation_target"] = (
                        max_generation_attempts
                    )
                    packed_clean_content["candidate_origin"] = candidate_record[
                        "candidate_origin"
                    ]
                    packed_clean_content["is_seed_task"] = candidate_record[
                        "is_seed_task"
                    ]
                    packed_clean_content["hardening_round_index"] = candidate_record[
                        "hardening_round_index"
                    ]
                    packed_clean_content["candidate_index_within_seed"] = candidate_idx
                    packed_clean_content["candidate_total_within_seed"] = len(
                        candidate_records
                    )

                    one = _pack_to_schema(
                        packed_clean_content,
                        solution_trace=candidate_state.trace_part,
                        solution_full=candidate_state.solution_part,
                        capability=capability,
                        capability_source_mode=capability_source_mode,
                        num_tasks=1,
                        chapter_id=chapter_id,
                        chapter_relpath=chapter_relpath,
                        difficulty=difficulty,
                        blooms_level=blooms_level,
                        blueprint_key=blueprint_key
                        or f"{difficulty.split('-')[0].strip()}_{blooms_level.split('-')[0].strip()}",  # noqa: PLC0207
                        chapter_q_start=task_seq,
                        task_id_start=task_seq,
                    )
                    passed_tasks.extend(one)

                    if (
                        checkpoint_every > 0
                        and checkpoint_path
                        and checkpoint_metadata
                        and verification_log is not None
                        and len(passed_tasks) % checkpoint_every == 0
                    ):
                        _save_checkpoint_snapshot(
                            passed_tasks,
                            verification_log,
                            token_usage_log,
                            checkpoint_path,
                            checkpoint_metadata,
                            generation_attempts,
                        )
                        logger.info(
                            f"[{task_batch_id}] Checkpoint saved -> {checkpoint_path} (passed={len(passed_tasks)})"
                        )

                    qa_pair = _qa_pair_text(one[0]) if one else ""
                    current_seed_index = i + 1
                    if qa_pair and current_seed_index not in prompt_memory_seed_indexes:
                        prompt_previous_questions.append(qa_pair)
                        prompt_memory_seed_indexes.add(current_seed_index)

                    task_seq += 1
                    logger.info(
                        f"[{task_batch_id}] {candidate_state.candidate_label} PASSED (prompt_prev_questions={len(prompt_previous_questions)})"
                    )
                    break

                if attempt < max_retries:
                    logger.info(
                        f"[{task_batch_id}] {candidate_state.candidate_label} Step 8: fix_bug (attempt {attempt + 1})"
                    )
                    feedback_str = _format_feedback(verification_report)
                    json_bad = (
                        _norm_yes_no(verification_report.get("json_format_valid"))
                        == "no"
                    )
                    mcq_ok = (
                        _norm_yes_no(verification_report.get("mcq_integrity")) == "yes"
                    )
                    constraint_ok = (
                        _norm_yes_no(verification_report.get("constraint_compliance"))
                        == "yes"
                    )

                    json_only_case = json_bad and mcq_ok and constraint_ok

                    designer = designer_factory()

                    if json_only_case:
                        (
                            revised_content,
                            json_fix_usage,
                        ) = await designer.fix_json_format_only(
                            previous_candidate_output=_ensure_json_string(
                                candidate_state.qcore
                            ),
                            verifier_feedback=feedback_str,
                        )
                        _append_token_usage_record(
                            token_usage_log,
                            usage=json_fix_usage,
                            stage="fix_json_format_only",
                            model_role="designer",
                            task_batch_id=task_batch_id,
                            chapter_id=chapter_id,
                            chapter_relpath=chapter_relpath,
                            blueprint_key=blueprint_key,
                            difficulty=difficulty,
                            blooms_level=blooms_level,
                            seed_generation_index=i + 1,
                            candidate_label=candidate_state.candidate_label,
                            candidate_index_within_seed=candidate_idx,
                            attempt_index=attempt,
                        )
                    else:
                        (
                            revised_content,
                            repair_usage,
                        ) = await designer.fix_mcq_with_trace(
                            previous_candidate_output=_ensure_json_string(
                                candidate_state.qcore
                            ),
                            verifier_feedback=feedback_str,
                            chapter_material=f"{context_text}\n\n[CHAPTER_KNOWLEDGE_SUMMARY]\n{chapter_knowledge_text}",
                            chapter_knowledge_text=chapter_knowledge_text,
                            difficulty=difficulty,
                            blooms_level=blooms_level,
                            solution_trace=_ensure_json_string(
                                candidate_state.trace_part
                            ),
                            previous_questions=prompt_previous_questions,
                        )
                        _append_token_usage_record(
                            token_usage_log,
                            usage=repair_usage,
                            stage="fix_mcq_with_trace",
                            model_role="designer",
                            task_batch_id=task_batch_id,
                            chapter_id=chapter_id,
                            chapter_relpath=chapter_relpath,
                            blueprint_key=blueprint_key,
                            difficulty=difficulty,
                            blooms_level=blooms_level,
                            seed_generation_index=i + 1,
                            candidate_label=candidate_state.candidate_label,
                            candidate_index_within_seed=candidate_idx,
                            attempt_index=attempt,
                        )

                    if is_qcore_dict(
                        revised_content
                    ) and not _looks_like_verification_report(revised_content):
                        candidate_state.qcore = revised_content
                    else:
                        logger.warning(
                            f"[{task_batch_id}] {candidate_state.candidate_label} Step 8 produced invalid payload; keeping prior candidate."
                        )

            else:
                logger.warning(
                    f"[{task_batch_id}] {candidate_state.candidate_label} FAILED after {max_retries + 1} attempts; skipping."
                )

        if (
            checkpoint_every > 0
            and checkpoint_path
            and checkpoint_metadata
            and verification_log is not None
        ):
            _save_checkpoint_snapshot(
                passed_tasks,
                verification_log,
                token_usage_log,
                checkpoint_path,
                checkpoint_metadata,
                generation_attempts,
            )

    if len(passed_tasks) < num_tasks:
        logger.warning(
            f"[{task_batch_id}] Only generated {len(passed_tasks)}/{num_tasks} passing tasks "
            f"after {generation_attempts} seed generation attempts."
        )

    return passed_tasks or None
