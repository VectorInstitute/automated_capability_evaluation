"""Source file for the verifier agent used in task verification."""

import logging
from typing import Any, Dict, List, Tuple, Union

from src.task_generation.json_response_utils import parse_json_like, stringify_payload
from src.task_generation.prompts import (
    MCQ_INTEGRITY_OUTPUT_FORMAT,
    SYSTEM_MCQ_INTEGRITY_CHECK_AND_REVISE_PROMPT,
    SYSTEM_TASK_DIFFICULTY_ASSESSMENT_PROMPT,
    SYSTEM_TASK_VERIFICATION_PROMPT,
    USER_MCQ_INTEGRITY_CHECK_AND_REVISE_PROMPT,
    USER_TASK_DIFFICULTY_ASSESSMENT_PROMPT,
    USER_TASK_VERIFICATION_PROMPT,
)
from src.utils.model_client_utils import ModelCallMode, async_call_model


logger = logging.getLogger(__name__)


class VerifierAgent:
    """Verifier agent for task verification."""

    def __init__(self, name: str, model_client: Any, **kwargs: Any) -> None:
        self.name = name
        self.model_client = model_client

    async def verify_task(
        self,
        candidate_output: Union[Dict[str, Any], str],
    ) -> Dict[str, Any]:
        """
        Verify the candidate task output against the provided blueprint and context.

        Args:
            candidate_output: The candidate task output to be verified.

        Returns
        -------
            A dictionary containing the verification results.
        """
        candidate_str = stringify_payload(candidate_output)

        user_prompt = USER_TASK_VERIFICATION_PROMPT.format(
            candidate_output=candidate_str,
        )
        task = SYSTEM_TASK_VERIFICATION_PROMPT + "\n\n" + user_prompt
        text = await self._call_text_prompt(task)
        return self._extract_verification_report(text)

    async def check_and_revise_mcq_option(
        self,
        candidate_question: str,
        chapter_excerpts: str,
        chapter_knowledge_text: str,
        solution_trace: Dict[str, Any],
        solution_full: Dict[str, Any],
    ) -> Tuple[Union[Dict[str, Any], str], str]:
        """
        Check the correctness of a candidate question.

        Args:
            candidate_question: The candidate problem to process.
            chapter_excerpts: Relevant excerpts from the chapter.
            chapter_knowledge_text: Structured knowledge about the chapter.
            solution_trace: The step-by-step solution trace for the candidate question.
            solution_full: The full solution for the candidate question.

        Returns
        -------
            A tuple containing the processed problems and the full prompt used.
        """
        task = USER_MCQ_INTEGRITY_CHECK_AND_REVISE_PROMPT.format(
            candidate_question=candidate_question,
            chapter_excerpts=chapter_excerpts,
            chapter_knowledge_text=chapter_knowledge_text,
            solution_trace=stringify_payload(solution_trace),
            solution_full=stringify_payload(solution_full),
        )
        task = (
            SYSTEM_MCQ_INTEGRITY_CHECK_AND_REVISE_PROMPT
            + "\n\n"
            + task
            + "\n\n"
            + MCQ_INTEGRITY_OUTPUT_FORMAT
        )
        text = await self._call_text_prompt(task)
        return self._extract_mcq_payload(text), task

    async def assess_task_difficulty(
        self,
        candidate_question: Union[Dict[str, Any], str],
    ) -> Tuple[Dict[str, Any], str]:
        """
        Solve and assess candidate question difficulty with structured feedback.

        Args:
            candidate_question: Candidate question payload.

        Returns
        -------
            A tuple of (difficulty_feedback_json, full_prompt).
        """
        candidate_str = stringify_payload(candidate_question)
        user_prompt = USER_TASK_DIFFICULTY_ASSESSMENT_PROMPT.format(
            candidate_question=candidate_str,
        )
        task = SYSTEM_TASK_DIFFICULTY_ASSESSMENT_PROMPT + "\n\n" + user_prompt
        text = await self._call_text_prompt(task)
        return self._extract_difficulty_feedback(text), task

    async def _call_text_prompt(self, task: str) -> str:
        """Call the model in plain-text mode for a fully assembled prompt."""
        return await async_call_model(
            self.model_client,
            user_prompt=task,
            mode=ModelCallMode.TEXT,
        )

    def _extract_mcq_payload(self, content: str) -> Union[Dict[str, Any], str]:
        """
        Extract payload for Step-3 (MCQ repair path).

        No verifier-report fallback here; return {} on parse failure.
        """
        obj = parse_json_like(content)
        if isinstance(obj, dict):
            return obj
        if obj is None:
            logger.warning("Verifier MCQ payload parse failed; returning empty dict.")
            return {}
        return {"result": obj}

    def _extract_verification_report(self, content: str) -> Dict[str, Any]:
        """
        Extract payload for Step-7 (verification report path).

        Uses fail-report fallback to keep verification loop deterministic.
        """
        obj = parse_json_like(content)
        if isinstance(obj, dict):
            return obj
        return self._fallback_report("json parse failed")

    def _extract_difficulty_feedback(self, content: str) -> Dict[str, Any]:
        """
        Extract structured difficulty feedback with strict schema normalization.

        Guarantees a dict that always includes all expected keys.
        """
        obj = parse_json_like(content)
        if not isinstance(obj, dict):
            return self._fallback_difficulty_feedback("json parse failed")

        difficulty_raw = str(obj.get("difficulty", "")).strip().lower()
        if difficulty_raw not in {"easy", "medium", "hard"}:
            difficulty_raw = "medium"

        confidence_raw = obj.get("confidence", 0.0)
        try:
            confidence = float(confidence_raw)
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))

        def _to_list_of_str(x: Any) -> List[str]:
            if isinstance(x, list):
                return [str(v).strip() for v in x if str(v).strip()]
            if x is None:
                return []
            s = str(x).strip()
            return [s] if s else []

        return {
            "difficulty": difficulty_raw,
            "solver_answer": str(obj.get("solver_answer", "")).strip(),
            "solver_rationale": str(obj.get("solver_rationale", "")).strip(),
            "difficulty_rationale": str(obj.get("difficulty_rationale", "")).strip(),
            "identified_issues": _to_list_of_str(obj.get("identified_issues")),
            "confidence": confidence,
        }

    def _fallback_report(self, msg: str) -> Dict[str, Any]:
        """Structured fail report fallback for verification stage only."""
        return {
            "overall_verdict": "Fail",
            "json_format_valid": "No",
            "mcq_integrity": "No",
            "constraint_compliance": "No",
            "explanation": f"System Error: {msg}",
            "question_evaluation": {
                "distractors_plausible": "No",
                "main_issues": [f"parse_error: {msg}"],
                "fix": "Return a single valid JSON object matching the verification schema.",
            },
        }

    def _fallback_difficulty_feedback(self, msg: str) -> Dict[str, Any]:
        """Structured fallback for difficulty assessment path."""
        return {
            "difficulty": "medium",
            "solver_answer": "",
            "solver_rationale": "",
            "difficulty_rationale": f"System Error: {msg}",
            "identified_issues": [f"parse_error: {msg}"],
            "confidence": 0.0,
        }
