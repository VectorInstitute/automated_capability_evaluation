"""Source file for the verifier agent used in task verification."""
import json
import logging
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage

from src.task_generation.prompts import (
    SYSTEM_MCQ_INTEGRITY_CHECK_AND_REVISE_PROMPT,
    SYSTEM_TASK_VERIFICATION_PROMPT,
    USER_MCQ_INTEGRITY_CHECK_AND_REVISE_PROMPT,
    USER_TASK_VERIFICATION_PROMPT,
)


logger = logging.getLogger(__name__)


def _strip_agent_terminator(text: str) -> str:
    """Remove trailing agent terminator tokens (e.g., TERMINATE) from model output."""
    if not text:
        return ""
    return re.sub(r"\s*TERMINATE\s*$", "", text, flags=re.IGNORECASE).strip()


def _escape_invalid_json_backslashes(s: str) -> str:
    """
    Escape backslashes that are invalid in JSON escape sequences.

    Example repaired patterns:
    - "\\dot" -> "\\\\dot"
    - "\\frac" -> "\\\\frac"

    Valid JSON escapes (\\", \\\\, \\/, \\b, \\f, \\n, \\r, \\t, \\uXXXX) are preserved.
    For \\u, preservation only applies when followed by exactly 4 hex digits.

    Args:
        s: The input string to process.

    Returns
    -------
        A new string with invalid JSON backslashes escaped.
    """  # noqa: D301
    out: List[str] = []
    i = 0
    n = len(s)
    while i < n:
        ch = s[i]
        if ch != "\\":
            out.append(ch)
            i += 1
            continue

        if i + 1 >= n:
            out.append("\\\\")
            i += 1
            continue

        nxt = s[i + 1]
        if nxt in ['"', "\\", "/", "b", "f", "n", "r", "t"]:
            out.append("\\")
            out.append(nxt)
            i += 2
            continue

        if nxt == "u":
            if i + 5 < n and re.fullmatch(r"[0-9a-fA-F]{4}", s[i + 2 : i + 6]):
                out.append("\\u")
                out.append(s[i + 2 : i + 6])
                i += 6
                continue
            out.append("\\\\u")
            i += 2
            continue

        out.append("\\\\")
        out.append(nxt)
        i += 2

    return "".join(out)


class VerifierAgent(AssistantAgent):
    """Verifier agent for task verification."""

    def __init__(self, name: str, model_client: Any, **kwargs: Any) -> None:
        super().__init__(
            name=name,
            model_client=model_client,
            **kwargs,
        )

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
        candidate_str = (
            json.dumps(candidate_output, indent=2, ensure_ascii=False)
            if isinstance(candidate_output, dict)
            else str(candidate_output)
        )

        user_prompt = USER_TASK_VERIFICATION_PROMPT.format(
            candidate_output=candidate_str,
        )
        task = SYSTEM_TASK_VERIFICATION_PROMPT + "\n\n" + user_prompt
        result = await self.run(task=task)
        text = self._extract_message(result.messages)
        return self._extract_verification_report(text)


    async def check_and_revise_mcq_option(
        self,
        candidate_question: str,
        chapter_excerpts: str,
        chapter_knowledge_text: str,
        solution_trace: Dict[str, Any],
        solution_full: Dict[str, Any]
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
            solution_trace=json.dumps(solution_trace, indent=2, ensure_ascii=False),
            solution_full=json.dumps(solution_full, indent=2, ensure_ascii=False),
        )
        task = SYSTEM_MCQ_INTEGRITY_CHECK_AND_REVISE_PROMPT + "\n\n" + task
        result = await self.run(task=task)
        text = self._extract_message(result.messages)
        return self._extract_mcq_payload(text), task


    def _extract_message(self, messages: Sequence[Any]) -> str:
        """
        Extract text content from messages.

        Args:
            messages: List of message objects.

        Returns
        -------
            Extracted text content.
        """
        for m in reversed(messages):
            if isinstance(m, TextMessage):
                return m.content
            if hasattr(m, "content") and isinstance(m.content, str):
                return m.content
        return ""

    def _parse_json_like(self, content: str) -> Optional[Any]:  # noqa: PLR0911
        """Parse JSON from raw/fenced/braced text with invalid-escape repair."""
        content = _strip_agent_terminator((content or "").strip())
        if not content:
            return None

        def _loads_with_repair(candidate: str) -> Optional[Any]:
            try:
                return json.loads(candidate)
            except json.JSONDecodeError as e:
                repaired = _escape_invalid_json_backslashes(candidate)
                if repaired != candidate:
                    try:
                        return json.loads(repaired)
                    except json.JSONDecodeError:
                        pass
                return None

        # If output is fenced markdown, try fenced parsing first.
        blocks = re.findall(
            r"```json\s*(.*?)\s*```", content, flags=re.DOTALL | re.IGNORECASE
        )
        if blocks:
            for b in blocks:
                obj = _loads_with_repair(_strip_agent_terminator(b.strip()))
                if obj is not None:
                    return obj
        else:
            obj = _loads_with_repair(content)
            if obj is not None:
                return obj

        start, end = content.find("{"), content.rfind("}")
        if start != -1 and end != -1 and end > start:
            obj = _loads_with_repair(
                _strip_agent_terminator(content[start : end + 1])
            )
            if obj is not None:
                return obj

        return None

    def _extract_mcq_payload(self, content: str) -> Union[Dict[str, Any], str]:
        """
        Extract payload for Step-3 (MCQ repair path).

        No verifier-report fallback here; return {} on parse failure.
        """
        obj = self._parse_json_like(content)
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
        obj = self._parse_json_like(content)
        if isinstance(obj, dict):
            return obj
        return self._fallback_report("json parse failed")

    def _fallback_report(self, msg: str) -> Dict[str, Any]:
        """Structured fail report fallback for verification stage only."""
        return {
            "overall_verdict": "fail",
            "json_format_valid": "fail",
            "mcq_integrity": "fail",
            "clarity_well_posed": "fail",
            "constraint_compliance": "fail",
            "explanation": f"System Error: {msg}",
            "question_evaluation": {},
        }
