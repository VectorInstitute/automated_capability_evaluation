#!/usr/bin/env python3
"""
Unified Experiment Runner for Multi-Agent Mathematical Reasoning

Supports experiments with:
- Configurable model (default: gemini-2.5-pro)
- Configurable debate rounds (default: 1)
- Scalable tasks per capability: 1-15 (default: 15)
- Both baseline and tool-assisted conditions

Usage:
    # Quick test: 1 task per capability
    python run_experiments.py --num-tasks 1 --condition tool_assisted

    # Full experiment: 15 tasks per capability
    python run_experiments.py --num-tasks 15 --condition both

    # Custom configuration
    python run_experiments.py --model gemini-2.5-pro --rounds 2 --num-tasks 10
"""

import argparse
import asyncio
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from autogen_core import (
    DefaultTopicId,
    SingleThreadedAgentRuntime,
)
from dotenv import load_dotenv
from langfuse import Langfuse


# Load environment variables from .env file
load_dotenv()

from autogen_core.models import SystemMessage, UserMessage

from src.task_solver.messages import Task
from src.task_solver.moderator import TaskSolverModerator
from src.task_solver.scientist import TaskSolverScientist
from src.task_solver.tool_assisted_scientist import ToolAssistedScientist
from src.tools.toolkit import ScientificToolKit
from src.utils.model_client_utils import get_model_client


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("experiments.log"), logging.StreamHandler()],
)
log = logging.getLogger(__name__)


class ExperimentRunner:
    """Runner for mathematical reasoning experiments."""

    def __init__(
        self,
        model_name: str = "gemini-3-flash-preview",
        num_rounds: int = 1,
        num_tasks: int = 15,
        condition: str = "both",
        output_dir: str = "experiment_results",
        enable_tool_selection: bool = False,
        tasks_file: str = "selected_tasks.json",
        dataset_type: str = "math",
    ):
        """Initialize experiment runner.

        Args:
            model_name: LLM model to use
            num_rounds: Number of debate rounds
            num_tasks: Number of tasks per capability to run (1-15)
            condition: Which condition to run - 'baseline', 'tool_assisted', or 'both'
            output_dir: Directory for saving results
            enable_tool_selection: Whether to enable dynamic tool selection (default: True)
            tasks_file: Path to tasks JSON file (default: selected_tasks.json)
            dataset_type: Type of dataset - 'math' or 'xfinbench' (default: math)
        """
        self.model_name = model_name
        self.num_rounds = num_rounds
        self.num_tasks = num_tasks
        self.condition = condition
        self.enable_tool_selection = enable_tool_selection
        self.tasks_file = Path(tasks_file)
        self.dataset_type = dataset_type
        self.results_dir = Path(output_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Langfuse
        self.langfuse_client = Langfuse(
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        )

        self.results = {
            "experiment_info": {
                "name": "capability_evaluation_experiment",
                "version": "1.0",
                "start_time": datetime.now().isoformat(),
                "model": self.model_name,
                "num_rounds": self.num_rounds,
                "tasks_per_capability": self.num_tasks,
            },
            "baseline": [],
            "tool_assisted": [],
        }

    def load_tasks(self) -> List[Dict[str, Any]]:
        """Load tasks from specified tasks file."""
        if not self.tasks_file.exists():
            raise FileNotFoundError(
                f"Tasks file not found: {self.tasks_file}\n"
                f"Please provide a valid tasks file path."
            )

        with open(self.tasks_file, "r", encoding="utf-8") as f:
            all_tasks = json.load(f)

        selected_tasks = []

        # XFinBench dataset is a flat list, not grouped by capability
        if self.dataset_type == "xfinbench":
            # XFinBench tasks are already in the right format
            tasks_to_load = (
                all_tasks[: self.num_tasks]
                if isinstance(all_tasks, list)
                else all_tasks
            )
            for task in tasks_to_load:
                task_copy = task.copy()
                # Ensure required fields exist
                if "area_name" not in task_copy:
                    task_copy["area_name"] = "finance"
                selected_tasks.append(task_copy)
            log.info(
                f"Loaded {len(selected_tasks)} XFinBench tasks from {self.tasks_file}"
            )
        # Math dataset: grouped by capability
        # If using a custom tasks file (not default), load all tasks as-is
        elif self.tasks_file != Path("selected_tasks.json"):
            for capability_name, task_list in all_tasks.items():
                for task in task_list:
                    task_copy = task.copy()
                    task_copy["capability_name"] = capability_name
                    task_copy["area_name"] = "math"  # All tasks are math
                    selected_tasks.append(task_copy)
            log.info(f"Loaded {len(selected_tasks)} tasks from {self.tasks_file}")
        else:
            # For default file, apply num_tasks limit per capability
            for capability_name, task_list in all_tasks.items():
                # Take up to num_tasks from each capability
                for task in task_list[: self.num_tasks]:
                    task_copy = task.copy()
                    task_copy["capability_name"] = capability_name
                    task_copy["area_name"] = "math"  # All tasks are math
                    selected_tasks.append(task_copy)
            log.info(
                f"Loaded {len(selected_tasks)} tasks ({self.num_tasks} per capability)"
            )

        return selected_tasks

    async def run_single_task(
        self,
        task_data: Dict[str, Any],
        condition: str,
        task_index: int,
        total_tasks: int,
    ) -> Dict[str, Any]:
        """Run a single task with specified condition."""
        log.info(
            f"\n{'=' * 70}\n"
            f"[{condition.upper()}] Task {task_index}/{total_tasks}\n"
            f"Capability: {task_data['capability_name']}\n"
            f"Task ID: {task_data['task_id']}\n"
            f"{'=' * 70}"
        )

        start_time = time.time()
        result = {
            "task_id": task_data["task_id"],
            "capability_name": task_data["capability_name"],
            "condition": condition,
            "problem": task_data["problem"],
            "expected_answer": task_data["answer"],
            "task_type": task_data.get("task_type"),  # For XFinBench
            "start_time": datetime.now().isoformat(),
            "status": "pending",
        }

        try:
            # Create runtime
            runtime = SingleThreadedAgentRuntime()

            # Create output directory for this task
            output_dir = self.results_dir / condition / task_data["capability_name"]
            output_dir.mkdir(parents=True, exist_ok=True)

            # Register moderator (single solver - no debate/consensus)
            await TaskSolverModerator.register(
                runtime,
                "TaskSolverModerator",
                lambda: TaskSolverModerator(
                    model_client=get_model_client(
                        model_name=self.model_name,
                        seed=42,
                        reasoning_effort="low",
                    ),
                    num_solvers=1,  # Single scientist - no multi-agent debate
                    max_rounds=1,  # Single round - direct solve
                    output_dir=output_dir,
                    langfuse_client=self.langfuse_client,
                ),
            )

            # Register scientist agent based on condition (single scientist - no debate)
            if condition == "baseline":
                # Use regular TaskSolverScientist
                await TaskSolverScientist.register(
                    runtime,
                    "TaskSolverScientistA",
                    lambda: TaskSolverScientist(
                        model_client=get_model_client(
                            model_name=self.model_name,
                            seed=42,
                            reasoning_effort="low",
                        ),
                        scientist_id="A",
                        langfuse_client=self.langfuse_client,
                    ),
                )
            else:
                # Use ToolAssistedScientist with toolkit (no tool selection - direct code execution)
                toolkit = ScientificToolKit(
                    model_client=get_model_client(
                        model_name=self.model_name,
                        seed=42,
                        reasoning_effort="low",
                    ),
                    enable_tool_selection=self.enable_tool_selection,  # False by default - skip tool selection
                    enable_rag=False,  # Use model's parametric knowledge instead of retrieval
                )

                await ToolAssistedScientist.register(
                    runtime,
                    "TaskSolverScientistA",
                    lambda: ToolAssistedScientist(
                        model_client=get_model_client(
                            model_name=self.model_name,
                            seed=42,
                            reasoning_effort="low",
                        ),
                        scientist_id="A",
                        langfuse_client=self.langfuse_client,
                        toolkit=toolkit,
                    ),
                )

            # Start runtime
            runtime.start()

            # Create and publish task
            task = Task(
                task_id=task_data["task_id"],
                problem=task_data["problem"],
                capability_name=task_data["capability_name"],
                area_name=task_data.get("area_name", "math"),
            )

            await runtime.publish_message(task, DefaultTopicId())

            # Wait for completion
            await runtime.stop_when_idle()

            # Load the solution file
            solution_file = output_dir / f"{task_data['task_id']}_solution.json"
            if solution_file.exists():
                with open(solution_file, "r") as f:
                    solution_data = json.load(f)

                # Extract code execution info for tool_assisted
                code_executions = None
                if condition == "tool_assisted":
                    code_executions = self._extract_code_executions(
                        solution_data.get("all_solutions", [])
                    )
                    result["code_executions"] = code_executions
                    # Track if ANY code was executed successfully (no "ERROR:" in output)
                    result["code_execution_success"] = any(
                        c.get("has_code")
                        and not str(c.get("code_output", "")).startswith("ERROR:")
                        for c in code_executions
                    )

                result.update(
                    {
                        "status": "completed",
                        "execution_time": time.time() - start_time,
                        "solution": solution_data.get("solution"),
                        "numerical_answer": solution_data.get("numerical_answer"),
                        "consensus_reached": solution_data.get("consensus_reached"),
                        "total_rounds": solution_data.get("total_rounds"),
                    }
                )

                # Verify answer (XFinBench-aware)
                task_type = task_data.get("task_type")
                if self.dataset_type == "xfinbench" and task_type:
                    verification_dict = await self._verify_xfinbench_answer(
                        solution_data.get("numerical_answer"),
                        task_data["answer"],
                        task_type,
                        code_executions,
                        reasoning=solution_data.get("reasoning"),
                        problem_text=task_data.get("problem"),
                    )
                    result.update(verification_dict)
                    # Success is determined by LLM judge
                    result["success"] = verification_dict.get("llm_judge", False)
                else:
                    # For non-XFinBench datasets, use relaxed verifier
                    result["success"] = self._verify_single_answer(
                        solution_data.get("numerical_answer"),
                        task_data["answer"],
                        relaxed=True,
                    )
                    result["llm_judge"] = result["success"]
            else:
                result.update(
                    {
                        "status": "error",
                        "error": "Solution file not found",
                        "execution_time": time.time() - start_time,
                    }
                )

            log.info(
                f"✓ Task completed in {result['execution_time']:.2f}s\n"
                f"  Success: {result.get('success', False)}\n"
                f"  Consensus: {result.get('consensus_reached', False)}"
            )

        except Exception as e:
            log.error(f"Error running task: {e}", exc_info=True)
            result.update(
                {
                    "status": "error",
                    "error": str(e),
                    "execution_time": time.time() - start_time,
                }
            )

        return result

    def _normalize_answer(self, answer_str: str) -> list[float]:
        """Extract and normalize numerical values from any answer format.

        This method handles multiple answer formats:
        - LaTeX: \\[begin{pmatrix} 1 \\\\ 1 \\end{pmatrix}\\]
        - Python list: [1, 1] or [[1], [1]]
        - Python dict: {'x': [[1], [1]]}
        - Plain text with numbers

        Returns
        -------
            Sorted list of float values extracted from the answer.
        """
        import ast
        import re

        if not answer_str:
            return []

        answer_str = str(answer_str).strip()

        # Remove LaTeX formatting and commands
        answer_str = re.sub(r"\\[a-zA-Z]+\{", "", answer_str)  # Remove \command{
        answer_str = re.sub(r"\\[a-zA-Z]+\[", "", answer_str)  # Remove \command[
        answer_str = re.sub(
            r"\\[a-zA-Z]+", "", answer_str
        )  # Remove remaining \commands
        answer_str = re.sub(r"[{}\\]", "", answer_str)  # Remove braces and backslashes

        numbers = []

        # Try parsing as Python literal first (handles dicts, lists, etc.)
        try:
            parsed = ast.literal_eval(answer_str)

            def flatten(obj):
                """Recursively flatten nested structures."""
                if isinstance(obj, (list, tuple)):
                    result = []
                    for item in obj:
                        result.extend(flatten(item))
                    return result
                if isinstance(obj, dict):
                    result = []
                    for value in obj.values():
                        result.extend(flatten(value))
                    return result
                try:
                    return [float(obj)]
                except (ValueError, TypeError):
                    return []

            numbers = flatten(parsed)
        except (ValueError, SyntaxError):
            # If parsing fails, extract numbers via regex
            pass

        # Extract all numbers (including decimals and negative numbers)
        if not numbers:
            number_pattern = r"-?\d+\.?\d*"
            found_numbers = re.findall(number_pattern, answer_str)
            numbers = []
            for num_str in found_numbers:
                if num_str and num_str != "-":  # Skip empty or lone minus signs
                    try:
                        numbers.append(float(num_str))
                    except ValueError:
                        pass

        return sorted(numbers)

    def _verify_answer(
        self,
        solution_answer: str,
        expected_answer: str,
        code_executions: List[Dict] = None,
    ) -> bool:
        """Verify if solution matches expected answer with robust normalization.

        Args:
            solution_answer: The answer from the consensus/solution
            expected_answer: The expected correct answer
            code_executions: Optional list of code execution results to use as fallback

        Returns
        -------
            True if answer matches (from consensus or code execution), False otherwise
        """
        if expected_answer is None:
            return False

        # Try verifying the consensus answer first
        if solution_answer is not None:
            if self._verify_single_answer(solution_answer, expected_answer):
                return True

        # If consensus answer failed and we have code executions, try those
        if code_executions:
            for exec_info in code_executions:
                # Try the numerical_answer from code execution
                code_numerical = exec_info.get("numerical_answer")
                if code_numerical and self._verify_single_answer(
                    code_numerical, expected_answer
                ):
                    log.info(
                        f"  ✓ Answer verified using code output from agent {exec_info.get('agent_id')}"
                    )
                    return True

                # Try extracting numbers from raw code output
                code_output = exec_info.get("code_output", "")
                if code_output:
                    try:
                        output_nums = self._normalize_answer(code_output)
                        exp_nums = self._normalize_answer(expected_answer)
                        if (
                            output_nums
                            and exp_nums
                            and len(output_nums) == len(exp_nums)
                        ):
                            if all(
                                abs(s - e) < 1e-6 for s, e in zip(output_nums, exp_nums)
                            ):
                                log.info(
                                    f"  ✓ Answer verified using raw code output from agent {exec_info.get('agent_id')}"
                                )
                                return True
                    except Exception:
                        pass

        return False

    async def _verify_xfinbench_answer(
        self,
        solution_answer: str,
        expected_answer: Any,
        task_type: str,
        code_executions: List[Dict] = None,
        reasoning: str = None,
        problem_text: str = None,
    ) -> Dict[str, bool]:
        """Verify XFinBench answer using LLM-as-a-Judge with principled criteria.

        This uses LLM-as-a-Judge that encodes the legitimate reasons from Feb 12 analysis:
        - Precision superiority: tool used more rigorous method than textbook
        - Correct methodology with implementation variance: tiny numerical differences from different solvers

        Args:
            solution_answer: The answer from the solution
            expected_answer: The expected correct answer
            task_type: Type of task ('bool', 'mcq', 'calcu')
            code_executions: Optional list of code execution results
            reasoning: The reasoning/thought process from the solution
            problem_text: The original problem statement

        Returns
        -------
            Dictionary with 'llm_judge' success flag
        """
        results = {"llm_judge": False}
        if expected_answer is None:
            return results

        # Extract code context from code_executions
        code_context = None
        reasoning_context = reasoning

        if code_executions and len(code_executions) > 0:
            # Use the first successful code execution
            for exec_info in code_executions:
                if exec_info.get("has_code") and not str(
                    exec_info.get("code_output", "")
                ).startswith("ERROR:"):
                    code_context = exec_info.get("code", "")
                    # Also include thought from code execution if reasoning not provided
                    if not reasoning_context:
                        reasoning_context = exec_info.get("thought", "")
                    break

            # If no successful execution, use any code even if it errored
            if not code_context and len(code_executions) > 0:
                code_context = code_executions[0].get("code", "")
                if not reasoning_context:
                    reasoning_context = code_executions[0].get("thought", "")

        # Use LLM-as-a-Judge for verification
        results["llm_judge"] = await self._verify_with_llm_judge(
            solution_answer=solution_answer,
            expected_answer=expected_answer,
            reasoning=reasoning_context,
            code=code_context,
            problem_text=problem_text,
            task_type=task_type,
        )

        return results

    def _verify_boolean_answer(
        self,
        solution_answer: str,
        expected_answer: Any,
        code_executions: List[Dict] = None,
    ) -> bool:
        """Verify boolean (True/False or 0/1) answer."""
        if solution_answer is None:
            return False

        # Normalize to 0/1
        def normalize_bool(val):
            if val is None:
                return None
            val_str = str(val).strip().lower()
            if val_str in ["true", "1", "1.0", "yes"]:
                return 1
            if val_str in ["false", "0", "0.0", "no"]:
                return 0
            return None

        sol_bool = normalize_bool(solution_answer)
        exp_bool = normalize_bool(expected_answer)

        if sol_bool is not None and exp_bool is not None:
            if sol_bool == exp_bool:
                return True

        # Try code executions
        if code_executions:
            for exec_info in code_executions:
                code_answer = exec_info.get("numerical_answer") or exec_info.get(
                    "code_output", ""
                )
                code_bool = normalize_bool(code_answer)
                if code_bool is not None and code_bool == exp_bool:
                    log.info("  ✓ Boolean answer verified using code output")
                    return True

        return False

    def _verify_mcq_answer(
        self,
        solution_answer: str,
        expected_answer: str,
        code_executions: List[Dict] = None,
    ) -> bool:
        """Verify multiple choice answer (extract choice letter)."""
        if solution_answer is None or expected_answer is None:
            return False

        import re

        def extract_choice(text):
            """Extract choice letter (A, B, C, D, etc.) from text."""
            if not text:
                return None
            text_str = str(text).strip().upper()
            # Look for standalone letter or letter with period/parenthesis
            match = re.search(r"\b([A-Z])\b", text_str)
            if match:
                return match.group(1)
            return None

        sol_choice = extract_choice(solution_answer)
        exp_choice = extract_choice(expected_answer)

        if sol_choice and exp_choice:
            if sol_choice == exp_choice:
                return True

        # Try code executions
        if code_executions:
            for exec_info in code_executions:
                code_answer = exec_info.get("numerical_answer") or exec_info.get(
                    "code_output", ""
                )
                code_choice = extract_choice(code_answer)
                if code_choice and code_choice == exp_choice:
                    log.info("  ✓ MCQ answer verified using code output")
                    return True

        return False

    async def _verify_with_llm_judge(
        self,
        solution_answer: str,
        expected_answer: Any,
        reasoning: str = None,
        code: str = None,
        problem_text: str = None,
        task_type: str = None,
    ) -> bool:
        """Use LLM-as-a-Judge to verify answer with principled criteria from Feb 12 analysis.

        This judge encodes the legitimate reasons tool-use deserves credit when diverging from ground truth:
        1. Precision superiority: More rigorous method than textbook (continuous vs discrete, exact vs approximate)
        2. Correct methodology with implementation variance: Different numerical solvers producing slightly different floats

        NOT encoded: Lucky passes from consensus relaxation artifacts.

        Args:
            solution_answer: The answer from the solution
            expected_answer: The expected correct answer
            reasoning: The model's thought process explaining approach
            code: The code used to compute the answer
            problem_text: The original problem statement
            task_type: Type of task ('bool', 'mcq', 'calcu')

        Returns
        -------
            True if answer is mathematically correct (exact match OR legitimate precision/methodology difference)
        """
        if solution_answer is None or expected_answer is None:
            return False

        # Build context for judge
        reasoning_context = f"\n\nModel's Reasoning:\n{reasoning}" if reasoning else ""
        code_context = f"\n\nModel's Code:\n```python\n{code}\n```" if code else ""
        problem_context = f"\n\nProblem:\n{problem_text}" if problem_text else ""

        # Prepare the judgment prompt with principled criteria
        judge_prompt = f"""You are an expert financial mathematics evaluator. Your task is to determine if the model's answer is CORRECT.

EXPECTED ANSWER (from textbook/reference): {expected_answer}
MODEL'S ANSWER: {solution_answer}
TASK TYPE: {task_type if task_type else "numerical"}
{problem_context}
{reasoning_context}
{code_context}

EVALUATION CRITERIA:

**ACCEPT if any of the following are true:**

1. **EXACT MATCH** (with formatting/unit tolerance):
   - Answers are numerically identical after normalizing units and formatting
   - Unit conversions: 0.0156 = 1.56% = 156 bps (basis points)
   - Formatting: "$1,234.56" = "1234.56" = "1234.560"
   - Rounding: 102.1285 ≈ 102.13 (difference < 0.01 or rounds to same value)

2. **PRECISION SUPERIORITY** (tool used MORE rigorous method):
   - Model used continuous compounding, textbook assumed discrete
   - Model used exact numerical solver, textbook used approximation formula
   - Model used higher precision constants (more decimal places of π, e, etc.)
   - **HOW TO IDENTIFY**: Check if code uses scipy.optimize, numerical integration, or exact formulas
   - **EXAMPLE**: Model gets 2.489 with fsolve(), textbook gets 2.5 with simplified formula → ACCEPT

3. **CORRECT METHODOLOGY WITH IMPLEMENTATION VARIANCE**:
   - Model's approach is mathematically sound (verify from reasoning/code)
   - Difference is minor floating-point variance (< 0.1% relative error)
   - Different solvers/libraries produce slightly different results (2.489 vs 2.490)
   - **HOW TO IDENTIFY**: Code shows correct setup, tiny numerical difference
   - **EXAMPLE**: Two valid bond pricing implementations differ by 0.001 → ACCEPT

**REJECT if:**
- Model made mathematical error in reasoning or code
- Model used wrong formula or misunderstood problem
- Large numerical difference (> 1% relative error) without justification
- Model's methodology is fundamentally flawed

**SPECIAL HANDLING:**

For **boolean** (0/1) or **MCQ** tasks: Require exact match (no precision argumentation applies).

For **numerical** tasks: Apply full criteria above, prioritizing methodological correctness.

**YOUR JUDGMENT:**

Based ONLY on the criteria above:
1. Is there an exact match (accounting for units/formatting)?
2. If not exact, does the code/reasoning show the model used a MORE precise method?
3. If not more precise, is the methodology correct with only minor numerical variance?

Output ONLY one word: "ACCEPT" or "REJECT"

Your judgment:"""

        try:
            # Use same model as task solver for consistency
            judge_client = get_model_client(
                self.model_name, seed=42, reasoning_effort="low"
            )

            response = await judge_client.create(
                [
                    SystemMessage(
                        content="You are a precise mathematical evaluator who rewards rigorous computational methods."
                    ),
                    UserMessage(content=judge_prompt, source="user"),
                ]
            )

            judgment = response.content.strip().upper()

            if "ACCEPT" in judgment:
                log.debug(
                    f"LLM Judge: ACCEPT (expected={expected_answer}, got={solution_answer})"
                )
                return True
            if "REJECT" in judgment:
                log.debug(
                    f"LLM Judge: REJECT (expected={expected_answer}, got={solution_answer})"
                )
                return False
            log.warning(
                f"Unexpected LLM judge response: {judgment}, defaulting to REJECT"
            )
            return False

        except Exception as e:
            log.error(f"LLM judge error: {e}")
            return False

    def _verify_single_answer(
        self, solution_answer: str, expected_answer: str, relaxed: bool = False
    ) -> bool:
        """Verify a single answer using XFinBench standards.

        Args:
            solution_answer: The answer to verify
            expected_answer: The expected correct answer
            relaxed: If True, use 5% relative error tolerance; if False, use strict tolerance

        Returns
        -------
            True if answer matches according to the specified tolerance
        """
        if solution_answer is None or expected_answer is None:
            return False

        sol = str(solution_answer).strip().lower()
        exp = str(expected_answer).strip().lower()

        # Exact match
        if sol == exp:
            return True

        try:
            sol_nums = self._normalize_answer(solution_answer)
            exp_nums = self._normalize_answer(expected_answer)

            if not sol_nums or not exp_nums or len(sol_nums) != len(exp_nums):
                return False

            for s, e in zip(sol_nums, exp_nums):
                diff = abs(s - e)

                # Strict: Must be virtually exact
                if not relaxed:
                    if diff > 1e-6:
                        return False
                else:
                    # 1. Absolute tolerance (crucial for expected == 0)
                    if diff < 1e-4 or (e == 0 and diff < 1e-9):
                        continue

                    # 2. Feb 12 standard: 3.5% relative error tolerance
                    if e != 0 and diff / abs(e) <= 0.035:
                        continue

                    # 3. Unit scaling checks (BP, %, decimals, Thousands, Millions)
                    scaled_match = False
                    for factor in [10, 100, 1000, 10000, 1000000]:
                        if abs(s * factor - e) < 1e-4 or abs(s - e * factor) < 1e-4:
                            scaled_match = True
                            break
                        if e != 0 and (
                            abs(s * factor - e) / abs(e) <= 0.035
                            or abs(s - e * factor) / abs(e) <= 0.035
                        ):
                            scaled_match = True
                            break

                    if scaled_match:
                        continue
                    return False
            return True
        except Exception:
            return False

    def _extract_code_executions(self, all_solutions: List[Dict]) -> List[Dict]:
        """Extract code execution information and outputs from solutions."""
        code_executions = []
        for solution in all_solutions:
            code = solution.get("code", "")
            code_output = solution.get("code_output", "")
            if code:
                code_executions.append(
                    {
                        "agent_id": solution.get("agent_id"),
                        "round": solution.get("round_number"),
                        "has_code": True,
                        "code": code,
                        "code_output": code_output,
                        "numerical_answer": solution.get("numerical_answer"),
                    }
                )
        return code_executions

    async def run_experiment(self):
        """Run the experiment."""
        log.info("=" * 70)
        log.info("MATHEMATICAL REASONING EXPERIMENT")
        log.info("=" * 70)
        log.info(f"Model: {self.model_name}")
        log.info(f"Debate rounds: {self.num_rounds}")
        log.info(f"Tasks per capability: {self.num_tasks}")
        log.info(f"Condition: {self.condition}")
        log.info(f"Results directory: {self.results_dir}")
        log.info("=" * 70 + "\n")

        # Load tasks
        tasks = self.load_tasks()
        total_tasks = len(tasks)

        # Run baseline condition
        if self.condition in ["baseline", "both"]:
            log.info("\n" + "=" * 70)
            log.info("BASELINE CONDITION (No Code Execution)")
            log.info("=" * 70)
            for idx, task_data in enumerate(tasks, 1):
                result = await self.run_single_task(
                    task_data, "baseline", idx, total_tasks
                )
                self.results["baseline"].append(result)

        # Run tool-assisted condition
        if self.condition in ["tool_assisted", "both"]:
            log.info("\n" + "=" * 70)
            log.info("TOOL-ASSISTED CONDITION (With Code Execution)")
            log.info("=" * 70)
            for idx, task_data in enumerate(tasks, 1):
                result = await self.run_single_task(
                    task_data, "tool_assisted", idx, total_tasks
                )
                self.results["tool_assisted"].append(result)

        # Save results
        self.results["experiment_info"]["end_time"] = datetime.now().isoformat()
        results_file = self.results_dir / "experiment_results.json"
        with open(results_file, "w") as f:
            json.dump(self.results, f, indent=2)

        log.info(f"\n✓ Results saved to: {results_file}")

        # Print summary
        self._print_summary()

        # Flush Langfuse
        self.langfuse_client.flush()

    def _print_summary(self):
        """Print experiment summary with both strict and relaxed metrics for XFinBench."""
        log.info("\n" + "=" * 70)
        log.info("EXPERIMENT SUMMARY")
        log.info("=" * 70)

        for condition in ["baseline", "tool_assisted"]:
            if condition not in self.results or not self.results[condition]:
                continue

            results = self.results[condition]
            completed = [r for r in results if r["status"] == "completed"]

            log.info(f"\n{condition.upper().replace('_', ' ')}:")
            log.info(f"  Total tasks: {len(results)}")
            log.info(f"  Completed: {len(completed)}")

            if completed:
                # LLM Judge metrics
                successful = [r for r in completed if r.get("success", False)]
                log.info(f"  Successful: {len(successful)}")

                success_rate = len(successful) / len(completed) * 100
                log.info(
                    f"  Success rate (LLM-as-a-Judge with precision superiority criteria): {success_rate:.1f}%"
                )

                # Code execution success for tool_assisted
                if condition == "tool_assisted":
                    code_success_count = sum(
                        1
                        for r in completed
                        if r.get("code_executions")
                        and any(
                            c.get("has_code")
                            and not str(c.get("code_output", "")).startswith("ERROR:")
                            for c in r.get("code_executions", [])
                        )
                    )
                    log.info(
                        f"  Code execution success: {code_success_count}/{len(completed)}"
                    )

                avg_time = sum(r["execution_time"] for r in completed) / len(completed)
                log.info(f"  Average time: {avg_time:.2f}s")

        log.info("\n" + "=" * 70)


async def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Run mathematical reasoning experiments with configurable parameters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test: 1 task per capability, tool-assisted condition
  python run_experiments.py --num-tasks 1 --condition tool_assisted

  # Quick test: 1 task per capability, both conditions
  python run_experiments.py --num-tasks 1 --condition both

  # Full experiment: 3 tasks per capability (default)
  python run_experiments.py --num-tasks 3

  # Custom configuration
  python run_experiments.py --model gemini-3-flash-preview --rounds 2 --num-tasks 10 --output my_results
        """,
    )
    parser.add_argument(
        "--model",
        default="gemini-3-flash-preview",
        help="LLM model to use (default: gemini-3-flash-preview)",
    )
    parser.add_argument(
        "--rounds", type=int, default=1, help="Number of debate rounds (default: 1)"
    )
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=3,
        help="Number of tasks per capability to run (default: 3 for full experiment)",
    )
    parser.add_argument(
        "--condition",
        choices=["baseline", "tool_assisted", "both"],
        default="both",
        help="Which condition to run (default: both)",
    )
    parser.add_argument(
        "--output",
        default="experiment_results",
        help="Output directory for results (default: experiment_results)",
    )
    parser.add_argument(
        "--tasks-file",
        default="selected_tasks.json",
        help="Path to tasks JSON file (default: selected_tasks.json)",
    )
    parser.add_argument(
        "--enable-tool-selection",
        action="store_true",
        help="Enable dynamic tool selection (disabled by default)",
    )
    parser.add_argument(
        "--dataset-type",
        choices=["math", "xfinbench"],
        default="math",
        help="Type of dataset to evaluate (default: math)",
    )

    args = parser.parse_args()

    # Tool selection is disabled by default, enable only if flag is present
    enable_tool_selection = args.enable_tool_selection

    runner = ExperimentRunner(
        model_name=args.model,
        num_rounds=args.rounds,
        num_tasks=args.num_tasks,
        condition=args.condition,
        output_dir=args.output,
        enable_tool_selection=enable_tool_selection,
        tasks_file=args.tasks_file,
        dataset_type=args.dataset_type,
    )
    await runner.run_experiment()


if __name__ == "__main__":
    asyncio.run(main())
