"""Source file for the designer agent used in task generation and revision."""
import json
import logging
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage

from src.task_generation.prompts import (
    INCLUDE_CLARIFICATION_PROMPT,
    REMOVE_REDUNDANT_INFO_PROMPT,
    REMOVE_SOURCE_INFO_PROMPT,
    SOUNDNESS_CHECK_PROMPT,
    SYSTEM_CHAPTER_KNOWLEDGE_SUMMARY_PROMPT,
    SYSTEM_GRAPH_TASK_GENERATION_PROMPT,
    SYSTEM_GRAPH_TASK_GENERATION_PROMPT_UNIQUE,
    SYSTEM_TASK_REVISION_PROMPT_JSON_ONLY,
    SYSTEM_TASK_REVISION_PROMPT_MCQ_FIX,
    USER_CHAPTER_KNOWLEDGE_SUMMARY_PROMPT,
    USER_GRAPH_TASK_GENERATION_PROMPT,
    USER_GRAPH_TASK_GENERATION_PROMPT_UNIQUE,
    USER_TASK_REVISION_PROMPT_JSON_ONLY,
    USER_TASK_REVISION_PROMPT_MCQ_FIX,
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

        # Trailing backslash: escape it.
        if i + 1 >= n:
            out.append("\\\\")
            i += 1
            continue

        nxt = s[i + 1]

        # Always-valid one-char JSON escapes.
        if nxt in ['"', "\\", "/", "b", "f", "n", "r", "t"]:
            out.append("\\")
            out.append(nxt)
            i += 2
            continue

        # \u must be followed by exactly 4 hex digits to remain valid.
        if nxt == "u":
            if i + 5 < n and re.fullmatch(r"[0-9a-fA-F]{4}", s[i + 2 : i + 6]):
                out.append("\\u")
                out.append(s[i + 2 : i + 6])
                i += 6
                continue
            out.append("\\\\u")
            i += 2
            continue

        # Any other escape starter is invalid in JSON; escape the backslash itself.
        out.append("\\\\")
        out.append(nxt)
        i += 2

    return "".join(out)


class DesignerAgent(AssistantAgent):
    """Designer agent for task generation and revision."""

    def __init__(
        self,
        name: str,
        model_client: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name=name,
            model_client=model_client,
            **kwargs,
        )

    async def generate_draft(
        self,
        chapter_excerpts: str,
        chapter_knowledge_text: str,
        previous_questions: Optional[List[str]] = None,
    ) -> Tuple[Union[Dict[str, Any], str], str]:
        """
        Generate draft tasks based on the provided blueprint and context.

        Now includes `chapter_knowledge_text` (structured summary) in the prompt.

        Args:
            chapter_excerpts: Relevant excerpts from the chapter.
            chapter_knowledge_text: Structured knowledge about the chapter.
            previous_questions: List of previously generated questions for the chapter.

        Returns
        -------
            A tuple containing the generated tasks and the full prompt used
        """
        # Normalize (avoid None / huge whitespace)
        chapter_knowledge_text = (chapter_knowledge_text or "").strip()
        chapter_excerpts = (chapter_excerpts or "").strip()

        if previous_questions:
            previous_questions_str = "\n".join(f"- {q}" for q in previous_questions)
            user_prompt = USER_GRAPH_TASK_GENERATION_PROMPT_UNIQUE.format(
                chapter_excerpts=chapter_excerpts,
                chapter_knowledge_text=chapter_knowledge_text,
                previous_questions=previous_questions_str,
            )
            task = SYSTEM_GRAPH_TASK_GENERATION_PROMPT_UNIQUE + "\n\n" + user_prompt
        else:
            user_prompt = USER_GRAPH_TASK_GENERATION_PROMPT.format(
                chapter_excerpts=chapter_excerpts,
                chapter_knowledge_text=chapter_knowledge_text,
            )
            task = SYSTEM_GRAPH_TASK_GENERATION_PROMPT + "\n\n" + user_prompt

        result = await self.run(task=task)
        text = self._extract_message(result.messages)
        return self._extract_message_content(text), task


    async def summarize_chapter_knowledge(
        self,
        chapter_excerpts: str,
    ) -> Tuple[Union[Dict[str, Any], str], str]:
        """
        Summarize chapter knowledge from provided excerpts.

        Args:
            chapter_excerpts: Relevant excerpts from the chapter.

        Returns
        -------
            A tuple containing the summarized chapter knowledge and the prompt used.
        """
        task = SYSTEM_CHAPTER_KNOWLEDGE_SUMMARY_PROMPT + "\n\n" + USER_CHAPTER_KNOWLEDGE_SUMMARY_PROMPT.format(chapter_excerpts=chapter_excerpts)
        result = await self.run(task=task)
        text = self._extract_message(result.messages)
        return self._extract_message_content(text), task


    async def include_clarification_info(
        self, candidate_question: str
    ) -> Tuple[Union[Dict[str, Any], str], str]:
        """
        Include clarification information in candidate problem.

        Args:
            candidate_question: The candidate problem to process.

        Returns
        -------
            A tuple containing the processed problems and the full prompt used.
        """
        task = INCLUDE_CLARIFICATION_PROMPT.format(
            candidate_question=candidate_question,
        )
        result = await self.run(task=task)
        text = self._extract_message(result.messages)
        return self._extract_message_content(text), task

    async def remove_redundant_info(
        self, candidate_question: str
    ) -> Tuple[Union[Dict[str, Any], str], str]:
        """
        Remove redundant information from candidate problem.

        Args:
            candidate_question: The candidate problem to process.

        Returns
        -------
            A tuple containing the processed problems and the full prompt used.
        """
        task = REMOVE_REDUNDANT_INFO_PROMPT.format(
            candidate_question=candidate_question,
        )
        result = await self.run(task=task)
        text = self._extract_message(result.messages)
        return self._extract_message_content(text), task

    async def remove_references(
        self, candidate_question: str
    ) -> Tuple[Union[Dict[str, Any], str], str]:
        """
        Remove reference to chapter excerpts from candidate problem.

        Args:
            candidate_question: The candidate problem to process.

        Returns
        -------
            A tuple containing the processed problems and the full prompt used.
        """
        task = REMOVE_SOURCE_INFO_PROMPT.format(
            candidate_question=candidate_question,
        )
        result = await self.run(task=task)
        text = self._extract_message(result.messages)
        return self._extract_message_content(text), task

    async def check_soundness(
        self, candidate_question: str
    ) -> Tuple[Union[Dict[str, Any], str], str]:
        """
        Check the soundness of a candidate question.

        Args:
            candidate_question: The candidate problem to process.

        Returns
        -------
            A tuple containing the processed problems and the full prompt used.
        """
        task = SOUNDNESS_CHECK_PROMPT.format(
            candidate_question=candidate_question,
        )
        result = await self.run(task=task)
        text = self._extract_message(result.messages)
        return self._extract_message_content(text), task


    async def fix_mcq_with_trace(
        self,
        previous_candidate_output: str,
        verifier_feedback: str,
        chapter_material: str,
        chapter_knowledge_text: str,
        solution_trace: str,
        previous_questions: List[str],
    ) -> Union[Dict[str, Any], str]:
        """Fix the MCQ question based on the verifier feedback and solution trace.

        Args:
            previous_candidate_output: The previous candidate question output.
            verifier_feedback: The feedback from the verifier agent.
            chapter_material: The relevant chapter material.
            chapter_knowledge_text: The relevant chapter knowledge.
            solution_trace: The solution trace.
            previous_questions: A list of previously generated questions.

        Returns
        -------
            A dictionary containing the revised question or an error message.
        """
        previous_questions_str = "\n".join(f"- {q}" for q in previous_questions)
        user_prompt = USER_TASK_REVISION_PROMPT_MCQ_FIX.format(
            previous_candidate_output=previous_candidate_output,
            verifier_llm_feedback=verifier_feedback,
            chapter_material=chapter_material,
            chapter_knowledge_text=chapter_knowledge_text,
            solution_trace=solution_trace,
            previous_questions=previous_questions_str,
        )
        task = SYSTEM_TASK_REVISION_PROMPT_MCQ_FIX + "\n\n" + user_prompt
        result = await self.run(task=task)
        text = self._extract_message(result.messages)
        return self._extract_message_content(text)


    async def fix_json_format_only(
        self,
        previous_candidate_output: str,
        verifier_feedback: str,
    ) -> Union[Dict[str, Any], str]:
        """Fix the candidate output's JSON format based on the verifier feedback."""
        user_prompt = USER_TASK_REVISION_PROMPT_JSON_ONLY.format(
            previous_candidate_output=previous_candidate_output,
            verifier_llm_feedback=verifier_feedback,
        )
        task = SYSTEM_TASK_REVISION_PROMPT_JSON_ONLY + "\n\n" + user_prompt
        result = await self.run(task=task)
        text = self._extract_message(result.messages)
        return self._extract_message_content(text)

    def _extract_message(self, messages: Sequence[Any]) -> str:
        """Extract the last assistant text from an AgentChat result.messages list."""
        if not messages:
            return ""
        for m in reversed(messages):
            # Most common case
            if isinstance(m, TextMessage):
                return (m.content or "").strip()

            # Fallback for other message types
            content = getattr(m, "content", None)
            if isinstance(content, str):
                return content.strip()

            # Some message types may store text differently
            text = getattr(m, "text", None)
            if isinstance(text, str):
                return text.strip()

        return ""

    def _extract_message_content(  # noqa: PLR0911
        self, reply: Union[str, Dict[str, Any], List[Any], None]
    ) -> Union[Dict[str, Any], str]:
        """
        Extract content and attempt to parse JSON safely.

        Handles common AutoGen return shapes:
          - str
          - dict with "content"
          - list of messages (dicts), take the last
          - nested dict-like shapes (best-effort)

        Args:
            reply: The raw reply from the agent.

        Returns
        -------
            Parsed JSON object if successful, else raw string.
        """
        content = self._normalize_reply_to_text(reply)
        content = _strip_agent_terminator(content)
        if not content:
            return ""

        # A) If output is fenced markdown, parse fenced blocks first.
        blocks = re.findall(
            r"```json\s*(.*?)\s*```", content, flags=re.DOTALL | re.IGNORECASE
        )
        if blocks:
            for b in blocks:
                candidate = _strip_agent_terminator(b.strip())
                try:
                    parsed_candidate = json.loads(candidate)
                    if isinstance(parsed_candidate, dict):
                        return parsed_candidate
                except json.JSONDecodeError as e:
                    repaired = _escape_invalid_json_backslashes(candidate)
                    if repaired != candidate:
                        try:
                            parsed_candidate = json.loads(repaired)
                            if isinstance(parsed_candidate, dict):
                                logger.warning(
                                    "Recovered Designer JSON after escape repair (fenced block). "
                                    "Original error at line %s col %s: %s",
                                    e.lineno,
                                    e.colno,
                                    e.msg,
                                )
                                return parsed_candidate
                        except json.JSONDecodeError:
                            pass
        else:
            # B) Try parsing raw string directly
            try:
                parsed_content = json.loads(content)
                if isinstance(parsed_content, dict):
                    return parsed_content
            except json.JSONDecodeError as e:
                # Retry once after escaping invalid backslashes (common with LaTeX in JSON strings).
                repaired = _escape_invalid_json_backslashes(content)
                if repaired != content:
                    try:
                        parsed_content = json.loads(repaired)
                        if isinstance(parsed_content, dict):
                            logger.warning(
                                "Recovered Designer JSON after escape repair (raw content). "
                                "Original error at line %s col %s: %s",
                                e.lineno,
                                e.colno,
                                e.msg,
                            )
                            return parsed_content
                    except json.JSONDecodeError:
                        pass

        # C) Heuristic: find outermost braces and attempt to parse
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1 and end > start:
            json_candidate = _strip_agent_terminator(content[start : end + 1])
            try:
                parsed_json_candidate = json.loads(json_candidate)
                if isinstance(parsed_json_candidate, dict):
                    return parsed_json_candidate
            except json.JSONDecodeError as e:
                repaired = _escape_invalid_json_backslashes(json_candidate)
                if repaired != json_candidate:
                    try:
                        parsed_json_candidate = json.loads(repaired)
                        if isinstance(parsed_json_candidate, dict):
                            logger.warning(
                                "Recovered Designer JSON after escape repair (brace slice). "
                                "Original error at line %s col %s: %s",
                                e.lineno,
                                e.colno,
                                e.msg,
                            )
                            return parsed_json_candidate
                    except json.JSONDecodeError:
                        pass

        logger.warning(
            "Failed to parse JSON from Designer output. Returning raw string. Preview=%r",
            (content[:200] + "…") if len(content) > 200 else content,
        )
        return content

    def _normalize_reply_to_text(  # noqa: PLR0911
        self, reply: Union[str, Dict[str, Any], List[Any], None]
    ) -> str:
        """
        Best-effort normalization of AutoGen replies into a plain text string.

        Args:
            reply: The raw reply from the agent.

        Returns
        -------
            Normalized text string.
        """
        if reply is None:
            return ""

        if isinstance(reply, str):
            return reply.strip()

        if isinstance(reply, list) and reply:
            return self._normalize_reply_to_text(reply[-1])

        if isinstance(reply, dict):
            if "content" in reply:
                return str(reply["content"]).strip()

            if "message" in reply:
                return self._normalize_reply_to_text(reply["message"])

            if (
                "choices" in reply
                and isinstance(reply["choices"], list)
                and reply["choices"]
            ):
                return self._normalize_reply_to_text(reply["choices"][-1])

        try:
            return str(reply).strip()
        except Exception:
            return ""
