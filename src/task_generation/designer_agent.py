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
    SYSTEM_TASK_GENERATION_PROMPT,
    SYSTEM_TASK_GENERATION_PROMPT_UNIQUE,
    SYSTEM_TASK_REVISION_PROMPT,
    USER_TASK_GENERATION_PROMPT,
    USER_TASK_GENERATION_PROMPT_UNIQUE,
    USER_TASK_REVISION_PROMPT,
)


logger = logging.getLogger(__name__)


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
        domain: str,
        chapter_excerpts: str,
        difficulty: str,
        blooms_level: str,
        task_blueprint: str,
        chapter_id: Optional[str] = None,
        chapter_relpath: Optional[str] = None,
        previous_questions: Optional[List[str]] = None,
    ) -> Tuple[Union[Dict[str, Any], str], str]:
        """
        Generate draft tasks based on the provided blueprint and context.

        Args:
            domain: The domain or subject area for the tasks.
            chapter_excerpts: Relevant excerpts from the chapter.
            difficulty: The difficulty level for the tasks.
            blooms_level: The Bloom's taxonomy level for the tasks.
            task_blueprint: The blueprint or template for the tasks.
            chapter_id: Optional chapter identifier.
            chapter_relpath: Optional chapter relative path.
            previous_questions: Optional list of previously generated questions.

        Returns
        -------
            A tuple containing the generated tasks and the full prompt used.
        """
        if previous_questions:
            previous_questions_str = "\n".join(f"- {q}" for q in previous_questions)
            user_prompt = USER_TASK_GENERATION_PROMPT_UNIQUE.format(
                domain=domain,
                difficulty=difficulty,
                blooms_level=blooms_level,
                task_blueprint=task_blueprint,
                chapter_excerpts=chapter_excerpts,
                chapter_id=chapter_id or "",
                chapter_relpath=chapter_relpath or "",
                previous_questions=previous_questions_str,
            )

            task = SYSTEM_TASK_GENERATION_PROMPT_UNIQUE + "\n\n" + user_prompt
        else:
            user_prompt = USER_TASK_GENERATION_PROMPT.format(
                domain=domain,
                difficulty=difficulty,
                blooms_level=blooms_level,
                task_blueprint=task_blueprint,
                chapter_excerpts=chapter_excerpts,
                chapter_id=chapter_id or "",
                chapter_relpath=chapter_relpath or "",
            )
            task = SYSTEM_TASK_GENERATION_PROMPT + "\n\n" + user_prompt
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

    async def fix_bug(
        self,
        previous_candidate_output: str,
        verifier_feedback: str,
        chapter_material: str,
        previous_questions: List[str],
    ) -> Union[Dict[str, Any], str]:
        """
        Fix bugs in previously generated tasks based on verifier feedback.

        Args:
            previous_candidate_output: The previously generated candidate output.
            verifier_feedback: Feedback from the verifier indicating issues.
            chapter_material: Relevant chapter material for context.

        Returns
        -------
            The revised tasks (as a dict or str).
        """
        previous_questions_str = "\n".join(f"- {q}" for q in previous_questions)
        user_prompt = USER_TASK_REVISION_PROMPT.format(
            previous_candidate_output=previous_candidate_output,
            verifier_llm_feedback=verifier_feedback,
            chapter_material=chapter_material,
            previous_questions=previous_questions_str,
        )
        task = SYSTEM_TASK_REVISION_PROMPT + "\n\n" + user_prompt
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

    def _extract_message_content(
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
        if not content:
            return ""

        # A) Try parsing the raw string directly
        try:
            parsed_content = json.loads(content)
            if isinstance(parsed_content, dict):
                return parsed_content
        except json.JSONDecodeError:
            pass

        # B) Extract JSON from ```json ... ``` code blocks (try all blocks)
        blocks = re.findall(
            r"```json\s*(.*?)\s*```", content, flags=re.DOTALL | re.IGNORECASE
        )
        for b in blocks:
            candidate = b.strip()
            try:
                parsed_candidate = json.loads(candidate)
                if isinstance(parsed_candidate, dict):
                    return parsed_candidate
            except json.JSONDecodeError:
                continue

        # C) Heuristic: find outermost braces and attempt to parse
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1 and end > start:
            json_candidate = content[start : end + 1]
            try:
                parsed_json_candidate = json.loads(json_candidate)
                if isinstance(parsed_json_candidate, dict):
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
