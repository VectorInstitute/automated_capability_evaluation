"""Source file for the agentic pipeline to generate and verify tasks."""
import json
import logging
import re
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from src.schemas.capability_schemas import Capability
from src.schemas.io_utils import save_tasks
from src.schemas.metadata_schemas import PipelineMetadata
from src.schemas.task_schemas import Task
from src.task_generation.designer_agent import DesignerAgent
from src.task_generation.verifier_agent import VerifierAgent


logger = logging.getLogger(__name__)

def _qa_pair_text(t: Task) -> str:
    """Create QA pair for anti-duplication checks."""
    q = (t.task_statement or "").strip()
    meta = t.generation_metadata or {}
    ca = str(meta.get("correct_answer") or "").strip()

    ans_text = ""
    for ch in (t.choices or []):
        if str(ch.get("label", "")).strip() == ca:
            ans_text = str(ch.get("solution", "")).strip()
            break

    a = ans_text or ca
    return f"Question: {q}; Answer: {a}" if q else ""


def _is_passing(report: Dict[str, Any]) -> bool:
    """Determine if the verification report indicates a passing result."""
    overall = report.get("overall_verdict")
    if isinstance(overall, str):
        return overall.strip().lower() == "pass"
    return False


def _split_parts(one_obj: Union[Dict[str, Any], str]) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
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
    sg = q_obj.pop("solution_graph", None) or q_obj.pop("reasoning_graph", None) or q_obj.pop("graph", None)
    if sg is not None:
        trace_part["solution_graph"] = sg

    ss = q_obj.pop("solution_steps", None) or q_obj.pop("steps", None)
    if ss is not None:
        trace_part["solution_steps"] = ss

    # ---- extract full solution ----
    cs = q_obj.pop("complete_solution", None) or q_obj.pop("solution", None) or q_obj.pop("explanation", None)
    if cs is not None:
        solution_part["complete_solution"] = cs

    return q_obj, trace_part, solution_part


def is_qcore_dict(x: Any) -> bool:
    """Check if the given object conforms to the expected qcore dict structure."""
    return (
        isinstance(x, dict)
        and isinstance(x.get("question"), str) and x["question"].strip()
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
        "clarity_well_posed",
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
    # 1) Try direct parse first.
    try:
        parsed = json.loads(s)
        return json.dumps(parsed, indent=2, ensure_ascii=False)
    except json.JSONDecodeError:
        pass

    # 2) Parse from fenced ```json ... ``` blocks.
    blocks = re.findall(r"```json\s*(.*?)\s*```", s, flags=re.DOTALL | re.IGNORECASE)
    for b in blocks:
        candidate = b.strip()
        try:
            parsed = json.loads(candidate)
            return json.dumps(parsed, indent=2, ensure_ascii=False)
        except json.JSONDecodeError:
            continue

    # 3) Best-effort parse from outermost JSON object/array slice.
    obj_start, obj_end = s.find("{"), s.rfind("}")
    if obj_start != -1 and obj_end != -1 and obj_end > obj_start:
        candidate = s[obj_start : obj_end + 1]
        try:
            parsed = json.loads(candidate)
            return json.dumps(parsed, indent=2, ensure_ascii=False)
        except json.JSONDecodeError:
            pass

    return s


def _load_tasks_from_checkpoint(path: Path) -> List[Task]:
    """
    Load tasks from a checkpoint JSON written by save_tasks().

    Args:
        path: The path to the checkpoint file.

    Returns
    -------
        List[Task]: A list of Task objects loaded from the checkpoint, or an empty list.
    """
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    raw_tasks = data.get("tasks", [])
    return [Task.from_dict(td) for td in raw_tasks]


def _save_checkpoint_snapshot(
    passed_tasks: List[Task],
    checkpoint_path: Optional[Path],
    checkpoint_metadata: Optional[PipelineMetadata],
) -> None:
    """Save a snapshot of the current passed tasks to a checkpoint file."""
    if checkpoint_path is None or checkpoint_metadata is None:
        return
    save_tasks(passed_tasks, checkpoint_metadata, checkpoint_path)

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
            if "solution_graph" in solution_trace and solution_trace["solution_graph"] is not None:
                trace_meta["solution_graph"] = solution_trace["solution_graph"]
            # accept both names, just in case
            if "solution_steps" in solution_trace and solution_trace["solution_steps"] is not None:
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
            if "complete_solution" in solution_full and solution_full["complete_solution"] is not None:
                solution_meta["complete_solution"] = solution_full["complete_solution"]
            elif "solution" in solution_full and solution_full["solution"] is not None:
                solution_meta["complete_solution"] = solution_full["solution"]
            elif "explanation" in solution_full and solution_full["explanation"] is not None:
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
                labels = [k for k in order if k in options] + [k for k in options.keys() if k not in order]
                choices = [{"label": str(k), "solution": str(options[k])} for k in labels]
            elif isinstance(options, list):
                if options and isinstance(options[0], dict) and "label" in options[0] and "solution" in options[0]:
                    choices = [{"label": str(o["label"]), "solution": str(o["solution"])} for o in options]
                else:
                    labels = ["A", "B", "C", "D", "E"]
                    choices = [{"label": labels[i] if i < 5 else str(i), "solution": str(o)} for i, o in enumerate(options)]
            else:
                choices = None

            extra_fields = {k: v for k, v in item.items() if k not in {"question", "task", "options", "correct_answer"}}

            if not task_statement:
                task_statement = json.dumps(item, ensure_ascii=False)

        else:
            task_statement = str(item)

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
                task_id=task_id,
                task_statement=task_statement,
                capability=capability,
                task_type="multiple_choice",
                solution_type="multiple_choice",
                difficulty=difficulty,
                bloom_level=blooms_level,
                choices=choices,
                generation_metadata=generation_metadata,
            )
        )

    return tasks


def _format_feedback(report: Dict[str, Any]) -> str:
    """Format the verifier feedback from the report."""
    top_keys = [
        "json_format_valid",
        "chapter_scope_verifiable",
        "blueprint_alignment",
        "domain_consistency",
        "difficulty_bloom_match",
        "mcq_integrity",
        "clarity_well_posed",
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
    domain: str, #TODO: remove domain from args if not needed in prompts
    context_text: str,
    chapter_knowledge_text: str,
    blueprint: str, # TODO: remove blueprint from args if not needed in prompts (or replace with more specific fields)
    previous_questions: List[str],
    capability_source_mode: str = "placeholder",
    max_retries: int = 5,
    difficulty: str = "Easy", #TODO: remove structured difficulty schema or enum
    blooms_level: str = "Remember", #TODO: remove structured blooms_level schema or enum
    num_tasks: int = 100,
    chapter_id: Optional[str] = None,
    chapter_relpath: Optional[str] = None,
    blueprint_key: Optional[str] = None,
    chapter_q_start: int = 0,
    verification_log: Optional[List[Dict[str, Any]]] = None,

    checkpoint_path: Optional[Path] = None,
    checkpoint_every: int = 10,
    checkpoint_metadata: Optional[PipelineMetadata] = None,
    resume_from_checkpoint: bool = False,
) -> Optional[List[Task]]:
    """
    Run the agentic pipeline for task generation.

    - Generates ONE problem at a time (instead of generating a batch of problems).
    - For each generated problem, run Steps 2–8.
    - Repeats until `num_tasks` passing tasks are collected.

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
        num_tasks: Number of tasks to produce (passing).
        chapter_id: Optional chapter identifier.
        chapter_relpath: Optional chapter relative path.
        blueprint_key: Optional key representing the blueprint.
        chapter_q_start: Starting index for chapter question numbering.
        verification_log: Optional list to log verification reports.

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
    task_seq = chapter_q_start

    generation_attempts = 0
    max_generation_attempts = max(10, num_tasks * 3)
    logger.info(f"[{task_batch_id}] Will attempt up to {max_generation_attempts} generations to get {num_tasks} passing tasks.")

    # ---- Resume from checkpoint if enabled ----
    if resume_from_checkpoint and checkpoint_path and checkpoint_path.exists():
        loaded_tasks = _load_tasks_from_checkpoint(checkpoint_path)
        if loaded_tasks:
            passed_tasks = loaded_tasks

            # Rebuild previous_questions for anti-dup to remain consistent
            previous_questions.clear()
            for t in passed_tasks:
                qa = _qa_pair_text(t)
                if qa:
                    previous_questions.append(qa)

            # Advance task_seq so new tasks get new IDs
            task_seq = chapter_q_start + len(passed_tasks)

            logger.info(
                f"[{task_batch_id}] Resumed from checkpoint: {len(passed_tasks)} passed tasks loaded "
                f"from {checkpoint_path}"
            )

    while len(passed_tasks) < num_tasks and generation_attempts < max_generation_attempts:
        i = len(passed_tasks)
        generation_attempts += 1

        logger.info(
            f"[{task_batch_id}] Q{i+1}/{num_tasks}: generation attempt {generation_attempts}/{max_generation_attempts}"
        )

        # --- Step 1: INITIAL GENERATION ---
        designer = designer_factory()
        one_content, one_prompt = await designer.generate_draft(
            chapter_excerpts=context_text,
            chapter_knowledge_text=chapter_knowledge_text,
            previous_questions=previous_questions,
        )

        # Normalize generation output into dict
        one_obj: Any = one_content
        if isinstance(one_obj, str):
            preview = (one_obj[:200] + "…") if len(one_obj) > 200 else one_obj
            logger.warning(
                f"[{task_batch_id}] Q{i+1} generator returned non-JSON; retrying Step 1 once. Preview={preview!r}"
            )

            # Try one more time with schema reminder prompt if response is non-dict.
            designer = designer_factory()
            one_content_retry, one_prompt_retry = await designer.generate_draft(
                chapter_excerpts=context_text,
                chapter_knowledge_text=chapter_knowledge_text,
                previous_questions=previous_questions,
            )
            one_obj = one_content_retry
            if isinstance(one_obj, str):
                preview = (one_obj[:200] + "…") if len(one_obj) > 200 else one_obj
                logger.warning(
                    f"[{task_batch_id}] Q{i+1} generator retry still non-JSON; skipping. Preview={preview!r}"
                )
                continue
            one_prompt = one_prompt_retry

        if not str(one_obj.get("question") or "").strip():
            logger.warning(f"[{task_batch_id}] Q{i+1} missing 'question' after Step 1; skipping.")
            continue

        # one_obj is the single-question dict from generation
        q_obj, trace_part, solution_part = _split_parts(one_obj)

        # now only pass question core to cleaning steps
        current_qcore: Union[Dict[str, Any], List[Any], str] = _wrap_qcore(q_obj)
        last_prompt_text_i = one_prompt

        logger.info(f"[{task_batch_id}] Q{i+1}/{num_tasks}: starting per-question pipeline")

        # Per-question retries (verification-driven repair loop)
        for attempt in range(max_retries + 1):
            logger.info(f"[{task_batch_id}] Q{i+1}/{num_tasks} Attempt {attempt+1}/{max_retries+1}")


            # --- Step 2: INCLUDE NOTATION DEFINITIONS / CLARIFICATIONS ---
            logger.info(f"[{task_batch_id}] Q{i+1} Step 2: Including clarification info...")

            designer = designer_factory()
            current_qcore_as_str = _ensure_json_string(current_qcore)
            clarified_qcore, clarification_prompt = await designer.include_clarification_info(candidate_question=current_qcore_as_str)

            if not is_qcore_dict(clarified_qcore):
                # Retry once with explicit schema reminder + error
                retry_prompt = (
                    current_qcore_as_str
                    + "\n\n[SCHEMA REMINDER]\n"
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
                clarified_qcore, _ = await designer.include_clarification_info(candidate_question=retry_prompt)

            if is_qcore_dict(clarified_qcore):
                current_qcore = clarified_qcore
            else:
                # If still bad, skip this attempt (don’t poison state)
                logger.warning(f"[{task_batch_id}] clarification failed twice; skipping attempt.")
                break

            logger.debug(f"[{task_batch_id}] Clarification content: {clarified_qcore}")
            logger.debug(f"[{task_batch_id}] Clarification prompt: {clarification_prompt}")


            # --- Step 3: VERIFY CORRECTNESS / MCQ INTEGRITY ---
            logger.info(f"[{task_batch_id}] Q{i+1} Step 3: Verifying MCQ integrity...")

            verifier = verifier_factory()
            integrity_input_str = _ensure_json_string(current_qcore)
            qcore_before_integrity = current_qcore

            mcq_fixed_full, mcq_fixed_full_prompt = await verifier.check_and_revise_mcq_option(
                candidate_question=integrity_input_str,
                chapter_excerpts=context_text,
                chapter_knowledge_text=chapter_knowledge_text,
                solution_trace=trace_part,
                solution_full=solution_part,
            )

            mcq_fixed_full_str = _ensure_json_string(mcq_fixed_full)

            try:
                mcq_fixed_full_obj = json.loads(mcq_fixed_full_str)
            except json.JSONDecodeError:
                # not valid JSON text
                mcq_fixed_full_obj = None

            if isinstance(mcq_fixed_full_obj, dict):
                q_obj_step3, trace_part_step3, solution_part_step3 = _split_parts(mcq_fixed_full_obj)
                candidate_qcore = _wrap_qcore(q_obj_step3)
                if is_qcore_dict(candidate_qcore):
                    current_qcore = candidate_qcore
                    trace_part = trace_part_step3 or trace_part
                    solution_part = solution_part_step3 or solution_part
                else:
                    logger.warning(
                        f"[{task_batch_id}] Q{i+1} Step 3 produced non-MCQ payload; keeping prior candidate."
                    )
                    current_qcore = qcore_before_integrity
            else:
                logger.warning(
                    f"[{task_batch_id}] Q{i+1} Step 3 produced non-MCQ payload; keeping prior candidate."
                )
                current_qcore = qcore_before_integrity

            logger.debug(f"[{task_batch_id}] MCQ-integrity content: {mcq_fixed_full}")
            logger.debug(f"[{task_batch_id}] MCQ-integrity prompt: {mcq_fixed_full_prompt}")


            # --- Step 4: REMOVE REDUNDANT INFO ---
            logger.info(f"[{task_batch_id}] Q{i+1} Step 4: Removing redundant info...")

            designer = designer_factory()
            mcq_integrity_as_str = _ensure_json_string(current_qcore)
            no_redundant_content, no_redundant_prompt = await designer.remove_redundant_info(
                candidate_question=mcq_integrity_as_str,
            )
            if is_qcore_dict(no_redundant_content) and not _looks_like_verification_report(no_redundant_content):
                current_qcore = no_redundant_content
            else:
                logger.warning(
                    f"[{task_batch_id}] Q{i+1} Step 4 produced invalid payload; keeping prior candidate."
                )

            logger.debug(f"[{task_batch_id}] No-redundant content: {no_redundant_content}")
            logger.debug(f"[{task_batch_id}] No-redundant prompt: {no_redundant_prompt}")


            # --- Step 5: REMOVE SOURCE REFERENCES ---
            logger.info(f"[{task_batch_id}] Q{i+1} Step 5: Removing source references...")

            designer = designer_factory()
            no_redundant_content_as_str = _ensure_json_string(current_qcore)
            no_source_content, no_source_prompt = await designer.remove_references(
                candidate_question=no_redundant_content_as_str,
            )
            if is_qcore_dict(no_source_content) and not _looks_like_verification_report(no_source_content):
                current_qcore = no_source_content
            else:
                logger.warning(
                    f"[{task_batch_id}] Q{i+1} Step 5 produced invalid payload; keeping prior candidate."
                )

            logger.debug(f"[{task_batch_id}] No-source content: {no_source_content}")
            logger.debug(f"[{task_batch_id}] No-source prompt: {no_source_prompt}")

            # --- Step 6: Check Soundness ---
            logger.info(f"[{task_batch_id}] Q{i+1} Step 6: Checking soundness...")

            designer = designer_factory()
            no_source_content_as_str = _ensure_json_string(current_qcore)
            clean_content, soundness_prompt = await designer.check_soundness(
                candidate_question=no_source_content_as_str,
            )
            if is_qcore_dict(clean_content) and not _looks_like_verification_report(clean_content):
                current_qcore = clean_content
            else:
                logger.warning(
                    f"[{task_batch_id}] Q{i+1} Step 6 produced invalid payload; keeping prior candidate."
                )
                clean_content = current_qcore

            logger.debug(f"[{task_batch_id}] Soundness content: {clean_content}")
            logger.debug(f"[{task_batch_id}] Soundness prompt: {soundness_prompt}")

            # --- Step 7: FINAL VERIFICATION (MCQ INTEGRITY, JSON FORMAT CHECK) ---
            logger.info(f"[{task_batch_id}] Q{i+1} Step 7: Verifying...")

            verifier = verifier_factory()
            verification_report = await verifier.verify_task(
                candidate_output=clean_content,
            )

            if is_qcore_dict(clean_content) and not _looks_like_verification_report(clean_content):
                current_qcore = clean_content
            # Log verification summary
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
                        "summary": {
                            "overall_verdict": verification_report.get("overall_verdict"),
                            "json_format_valid": verification_report.get("json_format_valid"),
                            "mcq_integrity": verification_report.get("mcq_integrity"),
                            "clarity_well_posed": verification_report.get("clarity_well_posed"),
                            "constraint_compliance": verification_report.get("constraint_compliance"),
                        },
                    }
                )

            logger.info(f"[{task_batch_id}] Q{i+1} Verification report: {verification_report}")
            logger.debug(f"[{task_batch_id}] Clean content for verification: {clean_content}")

            # Save if passed, else loop to Step 8 for fixes and retries
            if _is_passing(verification_report):
                one = _pack_to_schema(
                    clean_content,
                    solution_trace=trace_part,
                    solution_full=solution_part,
                    capability=capability,
                    capability_source_mode=capability_source_mode,
                    num_tasks=1,
                    chapter_id=chapter_id,
                    chapter_relpath=chapter_relpath,
                    difficulty=difficulty,
                    blooms_level=blooms_level,
                    blueprint_key=blueprint_key
                    or f"{difficulty.split('-')[0].strip()}_{blooms_level.split('-')[0].strip()}",
                    chapter_q_start=task_seq,
                    task_id_start=task_seq,
                )
                passed_tasks.extend(one)

                # ---- checkpoint snapshot (save_tasks schema) ----
                if checkpoint_every > 0 and checkpoint_path and checkpoint_metadata and len(passed_tasks) % checkpoint_every == 0:
                    _save_checkpoint_snapshot(passed_tasks, checkpoint_path, checkpoint_metadata)
                    logger.info(
                        f"[{task_batch_id}] Checkpoint saved → {checkpoint_path} (passed={len(passed_tasks)})"
                    )

                # Update previous_questions with the newly accepted Q/A pair for dedup
                qa_pair = _qa_pair_text(one[0]) if one else ""
                if qa_pair:
                    previous_questions.append(qa_pair)

                task_seq += 1
                logger.info(
                    f"[{task_batch_id}] Q{i+1} PASSED (prev_questions={len(previous_questions)})"
                )
                break

            # --- Step 8: FIX ISSUES IF NOT PASSED (upto max retries) ---
            if attempt < max_retries:
                logger.info(f"[{task_batch_id}] Q{i+1} Step 8: fix_bug (attempt {attempt+1})")
                feedback_str = _format_feedback(verification_report)
                designer = designer_factory()
                json_bad = str(verification_report.get("json_format_valid", "")).strip().lower() == "no"
                mcq_ok = str(verification_report.get("mcq_integrity", "")).strip().lower() == "yes"
                clarity_ok = str(verification_report.get("clarity_well_posed", "")).strip().lower() == "yes"
                constraint_ok = str(verification_report.get("constraint_compliance", "")).strip().lower() == "yes"

                json_only_case = json_bad and mcq_ok and clarity_ok and constraint_ok

                designer = designer_factory()

                if json_only_case:
                    # B) Fix JSON format only (preserve content)
                    revised_content = await designer.fix_json_format_only(
                        previous_candidate_output=_ensure_json_string(current_qcore),
                        verifier_feedback=feedback_str,
                    )
                else:
                    # A) MCQ fix grounded in trace + chapter + knowledge
                    revised_content = await designer.fix_mcq_with_trace(
                        previous_candidate_output=_ensure_json_string(current_qcore),
                        verifier_feedback=feedback_str,
                        chapter_material=f"{context_text}\n\n[CHAPTER_KNOWLEDGE_SUMMARY]\n{chapter_knowledge_text}",
                        chapter_knowledge_text=chapter_knowledge_text,
                        solution_trace=_ensure_json_string(trace_part),
                        previous_questions=previous_questions,
                    )

                if is_qcore_dict(revised_content) and not _looks_like_verification_report(revised_content):
                    current_qcore = revised_content
                else:
                    logger.warning(
                        f"[{task_batch_id}] Q{i+1} Step 8 produced invalid payload; keeping prior candidate."
                    )
                last_prompt_text_i = f"{last_prompt_text_i}\n\n[REVISION FEEDBACK]\n{feedback_str}"

        else:
            logger.warning(
                f"[{task_batch_id}] Q{i+1} FAILED after {max_retries+1} attempts; skipping."
            )
            continue

    if len(passed_tasks) < num_tasks:
        logger.warning(
            f"[{task_batch_id}] Only generated {len(passed_tasks)}/{num_tasks} passing tasks "
            f"after {generation_attempts} generation attempts."
        )

    return passed_tasks or None
