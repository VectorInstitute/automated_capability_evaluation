"""Source file for the agentic pipeline to generate and verify tasks."""

import json
import logging
import re
import uuid
from typing import Any, Callable, Dict, List, Optional, Union

from src.schemas.capability_schemas import Capability
from src.schemas.task_schemas import Task
from src.task_generation.designer_agent import DesignerAgent
from src.task_generation.verifier_agent import VerifierAgent


logger = logging.getLogger(__name__)


def _qa_pair_text(t: Task) -> str:
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


def _make_default_capability() -> Capability:
    """Create a Capability class placeholder."""
    default_capability_dict = {
        "capability_name": "__placeholder__",
        "capability_id": "capability_placeholder",
        "capability_description": "__placeholder__",
        "domain_name": "__placeholder__",
        "domain_id": "domain_placeholder",
        "area_name": "__placeholder__",
        "area_id": "area_placeholder",
        "area_description": "__placeholder__",
        "generation_metadata": {"is_placeholder": True},
    }

    return Capability.from_dict(default_capability_dict)


async def run_task_generation_loop(
    designer_factory: Callable[[], DesignerAgent],
    verifier_factory: Callable[[], VerifierAgent],
    domain: str,
    context_text: str,
    blueprint: str,
    previous_questions: List[str],
    max_retries: int = 5,
    difficulty: str = "Easy",
    blooms_level: str = "Remember",
    num_tasks: int = 3,
    chapter_id: Optional[str] = None,
    chapter_relpath: Optional[str] = None,
    blueprint_key: str = "",
    chapter_q_start: int = 0,
    verification_log: Optional[List[Dict[str, Any]]] = None,
) -> Optional[List[Task]]:
    """
    Run the agentic pipeline for task generation.

    Args:
        designer_factory: Factory function to create a DesignerAgent.
        verifier_factory: Factory function to create a VerifierAgent.
        domain: The domain or subject area for the tasks.
        context_text: Relevant excerpts from the chapter.
        blueprint: The blueprint or template for the tasks.
        max_retries: Maximum number of retries for verification failures.
        difficulty: The difficulty level for the tasks.
        blooms_level: The Bloom's taxonomy level for the tasks.
        num_tasks: Number of tasks to generate.
        chapter_id: Optional chapter identifier.
        chapter_relpath: Optional chapter relative path.
        blueprint_key: Optional key representing the blueprint.
        chapter_q_start: Starting index for chapter question numbering.
        verification_log: Optional list to log verification reports.
        previous_questions: Optional list of previously generated questions.

    Returns
    -------
        List[Task] on success, None on failure.
    """
    task_batch_id = f"batch_{uuid.uuid4().hex[:6]}"
    logger.info(
        f"[{task_batch_id}] Starting generation for chapter={chapter_id} combo={difficulty}/{blooms_level}"
    )

    # --- Step 1: Design (Drafting) with JSON-parse retries ---
    logger.info(f"[{task_batch_id}] Step 1: Generating draft...")

    draft_max_retries = 3
    draft_obj: Any = None
    draft_content: Any = None
    draft_prompt: str = ""

    for draft_attempt in range(1, draft_max_retries + 1):
        logger.info(
            f"[{task_batch_id}] Draft attempt {draft_attempt}/{draft_max_retries}"
        )
        designer = designer_factory()

        draft_content, draft_prompt = await designer.generate_draft(
            domain=domain,
            chapter_excerpts=context_text,
            difficulty=difficulty,
            blooms_level=blooms_level,
            task_blueprint=blueprint,
            chapter_id=chapter_id,
            chapter_relpath=chapter_relpath,
            previous_questions=previous_questions,
        )

        logger.debug(f"[{task_batch_id}] Initial draft content: {draft_content}")
        logger.debug(
            f"[{task_batch_id}] Initial draft prompt chars={len(draft_prompt)}"
        )

        draft_obj = draft_content
        if isinstance(draft_obj, str):
            s = draft_obj.strip()
            m = re.search(
                r"```(?:json)?\s*(\{.*\}|\[.*\])\s*```",
                s,
                flags=re.DOTALL | re.IGNORECASE,
            )
            if m:
                s = m.group(1).strip()

            try:
                draft_obj = json.loads(s)
            except json.JSONDecodeError as e:
                logger.warning(f"[{task_batch_id}] Draft JSON parse failed: {e}")
                draft_obj = None

        # validate shape early (so we retry if wrong shape too)
        questions: List[Dict[str, Any]] = []
        if isinstance(draft_obj, dict) and isinstance(draft_obj.get("questions"), list):
            questions = draft_obj["questions"]
        elif isinstance(draft_obj, list):
            questions = [q for q in draft_obj if isinstance(q, dict)]

        if len(questions) >= num_tasks:
            questions = questions[:num_tasks]
            break

        logger.warning(
            f"[{task_batch_id}] Draft invalid shape/length. Got {len(questions)} questions, need {num_tasks}."
        )
        draft_obj = None

    if draft_obj is None:
        logger.error(
            f"[{task_batch_id}] Draft failed after {draft_max_retries} attempts."
        )
        return None

    # keep these as before
    current_content = draft_content
    last_prompt_text = draft_prompt
    logger.info(
        f"[{task_batch_id}] Draft generated. Extracted {len(questions)} questions (need {num_tasks})."
    )

    passed_tasks: List[Task] = []

    for i, q in enumerate(questions):
        logger.info(
            f"[{task_batch_id}] Q{i + 1}/{num_tasks}: starting per-question pipeline"
        )

        current_content = {"questions": [q]}
        last_prompt_text_i = last_prompt_text

        for attempt in range(max_retries + 1):
            logger.info(
                f"[{task_batch_id}] Q{i + 1}/{num_tasks} Attempt {attempt + 1}/{max_retries + 1}"
            )

            # --- Step 2: INCLUDE Notation Definitions / Clarifications ---
            logger.info(
                f"[{task_batch_id}] Q{i + 1} Step 2: Including clarification info..."
            )

            designer = designer_factory()
            current_content_as_str = _ensure_json_string(current_content)
            (
                clarification_content,
                clarification_prompt,
            ) = await designer.include_clarification_info(
                candidate_question=current_content_as_str,
            )

            logger.debug(f"[{task_batch_id}] Clean content: {clarification_content}")
            logger.debug(f"[{task_batch_id}] Last prompt text: {clarification_prompt}")

            # --- Step 3: Remove Redundant References ---
            logger.info(
                f"[{task_batch_id}] Q{i + 1} Step 3: Removing redundant info..."
            )
            designer = designer_factory()
            clarification_content_as_str = _ensure_json_string(clarification_content)
            (
                no_redundant_content,
                no_redundant_prompt,
            ) = await designer.remove_redundant_info(
                candidate_question=clarification_content_as_str,
            )

            logger.debug(f"[{task_batch_id}] Clean content: {no_redundant_content}")
            logger.debug(f"[{task_batch_id}] Last prompt text: {no_redundant_prompt}")

            # --- Step 4: Remove Source References ---
            logger.info(
                f"[{task_batch_id}] Q{i + 1} Step 4: Removing source references..."
            )

            designer = designer_factory()
            no_redundant_content_as_str = _ensure_json_string(no_redundant_content)
            no_source_content, no_source_prompt = await designer.remove_references(
                candidate_question=no_redundant_content_as_str,
            )

            logger.debug(f"[{task_batch_id}] Clean content: {no_source_content}")
            logger.debug(f"[{task_batch_id}] Last prompt text: {no_source_prompt}")

            # --- Step 5: Check Soundness ---
            logger.info(f"[{task_batch_id}] Q{i + 1} Step 5: Checking soundness...")

            designer = designer_factory()
            no_source_content_as_str = _ensure_json_string(no_source_content)
            soundness_content, soundness_prompt = await designer.check_soundness(
                candidate_question=no_source_content_as_str,
            )

            logger.debug(f"[{task_batch_id}] Soundness content: {soundness_content}")
            logger.debug(f"[{task_batch_id}] Last prompt text: {soundness_prompt}")

            # --- Step 6: Verify correct option ---
            logger.info(
                f"[{task_batch_id}] Q{i + 1} Step 6: Verifying correct option..."
            )

            verifier = verifier_factory()
            soundness_content_as_str = _ensure_json_string(soundness_content)
            clean_content, clean_prompt = await verifier.check_and_revise_mcq_option(
                candidate_question=soundness_content_as_str,
            )

            logger.debug(f"[{task_batch_id}] Clean content: {clean_content}")
            logger.debug(f"[{task_batch_id}] Last prompt text: {clean_prompt}")
            # --- Step 7: Verify Correctness ---
            logger.info(f"[{task_batch_id}] Q{i + 1} Step 7: Verifying...")

            verifier = verifier_factory()
            verification_report = await verifier.verify_task(
                candidate_output=clean_content,
            )

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
                        "question_index_in_batch": i,
                        "summary": {
                            "overall_verdict": verification_report.get(
                                "overall_verdict"
                            ),
                            "json_format_valid": verification_report.get(
                                "json_format_valid"
                            ),
                            "mcq_integrity": verification_report.get("mcq_integrity"),
                            "clarity_well_posed": verification_report.get(
                                "clarity_well_posed"
                            ),
                            "constraint_compliance": verification_report.get(
                                "constraint_compliance"
                            ),
                            "single_correct_answer_verified": verification_report.get(
                                "single_correct_answer_verified"
                            ),
                        },
                    }
                )

            logger.debug(
                f"[{task_batch_id}] Q{i + 1} Verification report: {verification_report}"
            )
            logger.debug(
                f"[{task_batch_id}] Clean content for verification: {clean_content}"
            )

            cap = _make_default_capability()
            if _is_passing(verification_report):
                one = _pack_to_schema(
                    clean_content,
                    num_tasks=1,
                    chapter_id=chapter_id,
                    chapter_relpath=chapter_relpath,
                    difficulty=difficulty,
                    blooms_level=blooms_level,
                    blueprint_key=blueprint_key,
                    task_id_prefix=blueprint_key,
                    chapter_q_start=chapter_q_start + i,
                    task_id_start=i,
                    capability=cap,
                )
                passed_tasks.extend(one)

                qa_pair = _qa_pair_text(one[0]) if one else ""
                if qa_pair:
                    previous_questions.append(qa_pair)

                logger.info(
                    f"[{task_batch_id}] Q{i + 1} PASSED (prev_questions={len(previous_questions)})"
                )
                break

            # --- Step 8 fix bugs if not passed ---
            if attempt < max_retries:
                logger.info(
                    f"[{task_batch_id}] Q{i + 1} Step 8: fix_bug (attempt {attempt + 1})"
                )
                feedback_str = _format_feedback(verification_report)
                designer = designer_factory()
                revised_content = await designer.fix_bug(
                    previous_candidate_output=_ensure_json_string(clean_content),
                    verifier_feedback=feedback_str,
                    chapter_material=context_text,
                    previous_questions=previous_questions,
                )
                current_content = revised_content
                last_prompt_text_i = (
                    f"{last_prompt_text_i}\n\n[REVISION FEEDBACK]\n{feedback_str}"
                )

        else:
            logger.warning(
                f"[{task_batch_id}] Q{i + 1} FAILED after {max_retries + 1} attempts; skipping."
            )
            continue

    return passed_tasks or None


def _is_passing(report: Dict[str, Any]) -> bool:
    """
    Determine if the verification report indicates a passing result.

    Args:
        report: The verification report dictionary.

    Returns
    -------
        True if passing, False otherwise.
    """
    overall = report.get("overall_verdict")
    if isinstance(overall, str):
        return overall.strip().lower() == "pass"

    return False


def _ensure_json_string(content: Union[Dict[str, Any], List[Any], str]) -> str:
    """
    Ensure the content is a JSON-formatted string.

    Args:
        content: The content to ensure as a JSON string.

    Returns
    -------
        JSON-formatted string.
    """
    if isinstance(content, (dict, list)):
        return json.dumps(content, indent=2, ensure_ascii=False)

    s = str(content)
    try:
        parsed = json.loads(s)
        return json.dumps(parsed, indent=2, ensure_ascii=False)
    except json.JSONDecodeError:
        return s


def _pack_to_schema(
    content: Union[Dict[str, Any], List[Any], str],
    *,
    num_tasks: int,
    chapter_id: Optional[str],
    chapter_relpath: Optional[str],
    difficulty: str,
    blooms_level: str,
    blueprint_key: str,
    capability: Capability,
    task_id_prefix: Optional[str] = None,
    chapter_q_start: int = 0,
    task_id_start: int = 0,
) -> List[Task]:
    """
    Convert model output into List[Task].

    Args:
        content: The raw content from the model output.
        num_tasks: Number of tasks to include.
        chapter_id: Optional chapter identifier.
        chapter_relpath: Optional chapter relative path.
        difficulty: The difficulty level for the tasks.
        blooms_level: The Bloom's taxonomy level for the tasks.
        blueprint_key: Key representing the blueprint.
        task_id_prefix: Optional prefix for task IDs.
        chapter_q_start: Starting index for chapter question numbering.
        task_id_start: Starting index for task numbering.
        capability: Capability dict.

    Returns
    -------
        List[Task]
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

    # ---- Build Task objects ----
    if not task_id_prefix:
        task_id_prefix = blueprint_key or f"{difficulty}_{blooms_level}"

    tasks: List[Task] = []

    for idx, item in enumerate(items):
        k = task_id_start + idx
        base_task_id = f"task_{k:03d}"
        task_id = f"{task_id_prefix}__{base_task_id}"

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
                order = ["A", "B", "C", "D"]
                labels = [k for k in order if k in options] + [
                    k for k, v in options.items() if k not in order
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
                    labels = ["A", "B", "C", "D"]
                    choices = [
                        {"label": labels[i] if i < 4 else str(i), "solution": str(o)}
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

        generation_metadata: Dict[str, Any] = {
            "chapter_id": chapter_id,
            "chapter_relpath": chapter_relpath,
            "blueprint_key": blueprint_key,
            "correct_answer": correct_answer,
            "chapter_question_id": f"{chapter_id}_q_{(chapter_q_start + idx):03d}",
            **extra_fields,
        }

        tasks.append(
            Task(
                task_id=task_id,
                task_statement=task_statement,
                task_type="multiple_choice",
                solution_type="multiple_choice",
                difficulty=difficulty,
                bloom_level=blooms_level,
                choices=choices,
                generation_metadata=generation_metadata,
                capability=capability,
            )
        )

    return tasks


def _format_feedback(report: Dict[str, Any]) -> str:
    """
    Format the verifier feedback from the report.

    Args:
        report: The verification report dictionary.

    Returns
    -------
        Formatted feedback string.
    """
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
    evals = report.get("per_question_evaluation", [])
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
