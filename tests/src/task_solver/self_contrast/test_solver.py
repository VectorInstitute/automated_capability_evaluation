"""Regression tests for the Self-Contrast solver."""

from __future__ import annotations

import threading

import pytest

from src.task_solver.self_contrast.perspectives import (
    BASE_PERSPECTIVES,
    TOOL_PERSPECTIVE,
    PerspectiveResponse,
)
from src.task_solver.self_contrast.prompts import (
    CONTRAST_SYSTEM_PROMPT,
    PERSPECTIVE_CODE_SYSTEM_PROMPT,
    PERSPECTIVE_SYSTEM_PROMPT,
)
from src.task_solver.self_contrast.solver import SelfContrastSolver


class NoopClient:
    """Client double that should never be called."""

    def call(self, *_args, **_kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("Unexpected model call in test.")


class ConsensusClient:
    """Client double returning the same JSON answer for each perspective."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self._lock = threading.Lock()

    def call(
        self,
        system_prompt: str,
        _user_prompt: str,
        *,
        force_json: bool = False,
        temperature: float | None = None,
    ) -> str:
        del force_json, temperature
        with self._lock:
            self.calls.append(system_prompt)

        if system_prompt == PERSPECTIVE_SYSTEM_PROMPT:
            return '{ "answer": "1.0", "rationale": "The statement is true." }'
        if system_prompt == CONTRAST_SYSTEM_PROMPT:
            raise AssertionError("Contrast should have been skipped by local consensus.")
        raise AssertionError(f"Unexpected system prompt: {system_prompt}")


class MissingCodeThenCodeClient:
    """Client double that omits code once, then returns valid Python."""

    def __init__(self) -> None:
        self.code_calls = 0
        self._lock = threading.Lock()

    def call(
        self,
        system_prompt: str,
        _user_prompt: str,
        *,
        force_json: bool = False,
        temperature: float | None = None,
    ) -> str:
        del force_json, temperature
        if system_prompt == PERSPECTIVE_CODE_SYSTEM_PROMPT:
            with self._lock:
                self.code_calls += 1
                if self.code_calls == 1:
                    return "The answer is 4."
            return "```python\nprint(4)\n```"

        if system_prompt == PERSPECTIVE_SYSTEM_PROMPT:
            return '{ "answer": "4", "rationale": "Used the Python result." }'

        raise AssertionError(f"Unexpected system prompt: {system_prompt}")


@pytest.mark.asyncio
async def test_solver_skips_contrast_when_perspectives_already_agree():
    """Local consensus should bypass the expensive contrast call."""

    solver = SelfContrastSolver(
        ConsensusClient(),
        list(BASE_PERSPECTIVES),
        tool_mode="none",
    )

    result = await solver.solve_problem(
        {
            "id": "bool_1",
            "task": "bool",
            "question": "Is 2 greater than 1?",
            "ground_truth": "1.0",
        }
    )

    assert result["prediction"] == "1.0"
    assert result["contrast_details"]["contrast"]["skipped"] is True
    assert result["contrast_details"]["final"]["decision_source"] == "local_consensus"


@pytest.mark.asyncio
async def test_solver_retries_when_tool_response_lacks_code_block():
    """Tool-enabled solving should retry when the model forgets the code fence."""

    client = MissingCodeThenCodeClient()
    solver = SelfContrastSolver(
        client,
        [TOOL_PERSPECTIVE],
        tool_mode="tool_perspective_only",
    )

    response = await solver._solve_with_perspective(
        "Question: What is 2 + 2?",
        "calcu",
        TOOL_PERSPECTIVE,
        tool_context={"needs_tools": True, "documentation": "", "selected_modules": []},
    )

    assert client.code_calls == 2
    assert response.python_code == "print(4)"
    assert response.python_output == "4\n"
    assert response.answer == "4"


def test_solver_merges_decimal_and_percentage_versions_of_same_answer():
    """A decimal rate and percentage form should vote together, not conflict."""

    solver = SelfContrastSolver(NoopClient(), list(BASE_PERSPECTIVES), tool_mode="none")
    responses = [
        PerspectiveResponse(
            perspective_id="top_down",
            label="Top-Down",
            answer="10.4375",
            rationale="",
            raw_response="",
        ),
        PerspectiveResponse(
            perspective_id="bottom_up",
            label="Bottom-Up",
            answer="10.437494",
            rationale="",
            raw_response="",
        ),
        PerspectiveResponse(
            perspective_id="tool_assisted",
            label="Tool-Assisted",
            answer="0.104374942",
            rationale="",
            raw_response="",
            python_code="print(0.104374942)",
            python_output="0.104374942\n",
        ),
    ]

    majority = solver._select_majority_answer(responses, "calcu")

    assert majority is not None
    assert float(majority["answer"]) == pytest.approx(10.437497, rel=1e-6)
    assert solver._has_local_consensus(responses, "calcu") is True
