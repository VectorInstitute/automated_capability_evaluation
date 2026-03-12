"""Source file for the designer agent used in task generation and revision."""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

from src.schemas.task_gen_io_utils import strip_agent_terminator
from src.task_generation.json_response_utils import (
    normalize_reply_to_text,
    parse_json_like,
)
from src.task_generation.prompts import (
    INCLUDE_CLARIFICATION_PROMPT,
    REMOVE_REDUNDANT_INFO_PROMPT,
    REMOVE_SOURCE_INFO_PROMPT,
    SOUNDNESS_CHECK_PROMPT,
    SYSTEM_CHAPTER_KNOWLEDGE_SUMMARY_PROMPT,
    SYSTEM_GRAPH_TASK_GENERATION_PROMPT,
    SYSTEM_GRAPH_TASK_GENERATION_PROMPT_UNIQUE,
    SYSTEM_TASK_HARDENING_PROMPT,
    SYSTEM_TASK_REVISION_PROMPT_JSON_ONLY,
    SYSTEM_TASK_REVISION_PROMPT_MCQ_FIX,
    USER_CHAPTER_KNOWLEDGE_SUMMARY_PROMPT,
    USER_GRAPH_TASK_GENERATION_PROMPT,
    USER_GRAPH_TASK_GENERATION_PROMPT_UNIQUE,
    USER_TASK_HARDENING_PROMPT,
    USER_TASK_REVISION_PROMPT_JSON_ONLY,
    USER_TASK_REVISION_PROMPT_MCQ_FIX,
)
from src.utils.model_client_utils import ModelCallMode, async_call_model


logger = logging.getLogger(__name__)


class DesignerAgent:
    """Designer agent for task generation and revision."""

    def __init__(
        self,
        name: str,
        model_client: Any,
        **kwargs: Any,
    ) -> None:
        self.name = name
        self.model_client = model_client

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

        text = await self._call_text_prompt(task)
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
        task = (
            SYSTEM_CHAPTER_KNOWLEDGE_SUMMARY_PROMPT
            + "\n\n"
            + USER_CHAPTER_KNOWLEDGE_SUMMARY_PROMPT.format(
                chapter_excerpts=chapter_excerpts
            )
        )
        text = await self._call_text_prompt(task)
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
        text = await self._call_text_prompt(task)
        return self._extract_message_content(text), task

    async def harden_task(
        self,
        chapter_excerpts: str,
        chapter_knowledge_summary: str,
        candidate_question_and_solution_graph: str,
    ) -> Tuple[Union[Dict[str, Any], str], str]:
        """
        Harden a generated candidate question by increasing reasoning complexity.

        Args:
            chapter_excerpts: Relevant excerpts from the chapter.
            chapter_knowledge_summary: Structured chapter knowledge summary.
            candidate_question_and_solution_graph: Candidate question JSON as string.

        Returns
        -------
            A tuple containing the hardened candidate and the full prompt used.
        """
        user_prompt = USER_TASK_HARDENING_PROMPT.format(
            chapter_excerpts=(chapter_excerpts or "").strip(),
            chapter_knowledge_summary=(chapter_knowledge_summary or "").strip(),
            candidate_question_and_solution_graph=candidate_question_and_solution_graph,
        )
        task = SYSTEM_TASK_HARDENING_PROMPT + "\n\n" + user_prompt
        text = await self._call_text_prompt(task)
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
        text = await self._call_text_prompt(task)
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
        text = await self._call_text_prompt(task)
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
        text = await self._call_text_prompt(task)
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
        text = await self._call_text_prompt(task)
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
        text = await self._call_text_prompt(task)
        return self._extract_message_content(text)

    async def _call_text_prompt(self, task: str) -> str:
        """Call the model in plain-text mode for a fully assembled prompt."""
        return await async_call_model(
            self.model_client,
            user_prompt=task,
            mode=ModelCallMode.TEXT,
        )

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
        content = normalize_reply_to_text(reply)
        content = strip_agent_terminator(content)
        if not content:
            return ""
        parsed = parse_json_like(
            content,
            on_repair=lambda msg: logger.warning("Designer %s", msg),
        )
        if isinstance(parsed, dict):
            return parsed

        logger.warning(
            "Failed to parse JSON from Designer output. Returning raw string. Preview=%r",
            (content[:200] + "…") if len(content) > 200 else content,
        )
        return content
