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
from typing import Dict, List, Any

from dotenv import load_dotenv
from autogen_core import (
    DefaultTopicId,
    SingleThreadedAgentRuntime,
)
from langfuse import Langfuse

# Load environment variables from .env file
load_dotenv()

from src.task_solver.messages import Task
from src.task_solver.moderator import TaskSolverModerator
from src.task_solver.scientist import TaskSolverScientist
from src.task_solver.tool_assisted_scientist import ToolAssistedScientist
from src.utils.model_client_utils import get_model_client


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('experiments.log'),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)


class ExperimentRunner:
    """Runner for mathematical reasoning experiments."""
    
    def __init__(
        self,
        model_name: str = "gemini-2.5-pro",
        num_rounds: int = 1,
        num_tasks: int = 15,
        condition: str = "both",
        output_dir: str = "experiment_results",
    ):
        """Initialize experiment runner.
        
        Args:
            model_name: LLM model to use
            num_rounds: Number of debate rounds
            num_tasks: Number of tasks per capability to run (1-15)
            condition: Which condition to run - 'baseline', 'tool_assisted', or 'both'
            output_dir: Directory for saving results
        """
        self.model_name = model_name
        self.num_rounds = num_rounds
        self.num_tasks = num_tasks
        self.condition = condition
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
        """Load tasks from selected_tasks.json file."""
        tasks_file = Path("selected_tasks.json")
        
        if not tasks_file.exists():
            raise FileNotFoundError(
                f"Tasks file not found: {tasks_file}\n"
                f"Please run extract_tasks.py first."
            )
        
        with open(tasks_file, "r", encoding="utf-8") as f:
            all_tasks = json.load(f)
        
        # Select specified number of tasks from each capability
        selected_tasks = []
        for capability_name, task_list in all_tasks.items():
            # Take up to num_tasks from each capability
            for task in task_list[:self.num_tasks]:
                task_copy = task.copy()
                task_copy["capability_name"] = capability_name
                task_copy["area_name"] = "math"  # All tasks are math
                selected_tasks.append(task_copy)
        
        log.info(f"Loaded {len(selected_tasks)} tasks ({self.num_tasks} per capability)")
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
            f"\n{'='*70}\n"
            f"[{condition.upper()}] Task {task_index}/{total_tasks}\n"
            f"Capability: {task_data['capability_name']}\n"
            f"Task ID: {task_data['task_id']}\n"
            f"{'='*70}"
        )
        
        start_time = time.time()
        result = {
            "task_id": task_data["task_id"],
            "capability_name": task_data["capability_name"],
            "condition": condition,
            "problem": task_data["problem"],
            "expected_answer": task_data["answer"],
            "start_time": datetime.now().isoformat(),
            "status": "pending",
        }
        
        try:
            # Create runtime
            runtime = SingleThreadedAgentRuntime()
            
            # Create output directory for this task
            output_dir = self.results_dir / condition / task_data["capability_name"]
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Register moderator
            await TaskSolverModerator.register(
                runtime,
                "TaskSolverModerator",
                lambda: TaskSolverModerator(
                    model_client=get_model_client(
                        model_name=self.model_name,
                        seed=42,
                    ),
                    num_solvers=2,
                    max_rounds=self.num_rounds,
                    output_dir=output_dir,
                    langfuse_client=self.langfuse_client,
                ),
            )
            
            # Register scientist agents based on condition
            if condition == "baseline":
                # Use regular TaskSolverScientist
                await TaskSolverScientist.register(
                    runtime,
                    "TaskSolverScientistA",
                    lambda: TaskSolverScientist(
                        model_client=get_model_client(
                            model_name=self.model_name,
                            seed=42,
                        ),
                        scientist_id="A",
                        langfuse_client=self.langfuse_client,
                    ),
                )
                
                await TaskSolverScientist.register(
                    runtime,
                    "TaskSolverScientistB",
                    lambda: TaskSolverScientist(
                        model_client=get_model_client(
                            model_name=self.model_name,
                            seed=42,
                        ),
                        scientist_id="B",
                        langfuse_client=self.langfuse_client,
                    ),
                )
            else:
                # Use ToolAssistedScientist
                await ToolAssistedScientist.register(
                    runtime,
                    "TaskSolverScientistA",
                    lambda: ToolAssistedScientist(
                        model_client=get_model_client(
                            model_name=self.model_name,
                            seed=42,
                        ),
                        scientist_id="A",
                        langfuse_client=self.langfuse_client,
                    ),
                )
                
                await ToolAssistedScientist.register(
                    runtime,
                    "TaskSolverScientistB",
                    lambda: ToolAssistedScientist(
                        model_client=get_model_client(
                            model_name=self.model_name,
                            seed=42,
                        ),
                        scientist_id="B",
                        langfuse_client=self.langfuse_client,
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
                
                result.update({
                    "status": "completed",
                    "execution_time": time.time() - start_time,
                    "solution": solution_data.get("solution"),
                    "numerical_answer": solution_data.get("numerical_answer"),
                    "consensus_reached": solution_data.get("consensus_reached"),
                    "total_rounds": solution_data.get("total_rounds"),
                    "success": self._verify_answer(
                        solution_data.get("numerical_answer"),
                        task_data["answer"],
                        code_executions=code_executions
                    ),
                })
            else:
                result.update({
                    "status": "error",
                    "error": "Solution file not found",
                    "execution_time": time.time() - start_time,
                })
            
            log.info(
                f"✓ Task completed in {result['execution_time']:.2f}s\n"
                f"  Success: {result.get('success', False)}\n"
                f"  Consensus: {result.get('consensus_reached', False)}"
            )
            
        except Exception as e:
            log.error(f"Error running task: {e}", exc_info=True)
            result.update({
                "status": "error",
                "error": str(e),
                "execution_time": time.time() - start_time,
            })
        
        return result
    
    def _normalize_answer(self, answer_str: str) -> list[float]:
        """Extract and normalize numerical values from any answer format.
        
        This method handles multiple answer formats:
        - LaTeX: \\[begin{pmatrix} 1 \\\\ 1 \\end{pmatrix}\\]
        - Python list: [1, 1] or [[1], [1]]
        - Python dict: {'x': [[1], [1]]}
        - Plain text with numbers
        
        Returns:
            Sorted list of float values extracted from the answer.
        """
        import re
        import ast
        
        if not answer_str:
            return []
        
        answer_str = str(answer_str).strip()
        
        # Remove LaTeX formatting and commands
        answer_str = re.sub(r'\\[a-zA-Z]+\{', '', answer_str)  # Remove \command{
        answer_str = re.sub(r'\\[a-zA-Z]+\[', '', answer_str)  # Remove \command[
        answer_str = re.sub(r'\\[a-zA-Z]+', '', answer_str)    # Remove remaining \commands
        answer_str = re.sub(r'[{}\\]', '', answer_str)         # Remove braces and backslashes
        
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
                elif isinstance(obj, dict):
                    result = []
                    for value in obj.values():
                        result.extend(flatten(value))
                    return result
                else:
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
            number_pattern = r'-?\d+\.?\d*'
            found_numbers = re.findall(number_pattern, answer_str)
            numbers = []
            for num_str in found_numbers:
                if num_str and num_str != '-':  # Skip empty or lone minus signs
                    try:
                        numbers.append(float(num_str))
                    except ValueError:
                        pass
        
        return sorted(numbers)
    
    def _verify_answer(self, solution_answer: str, expected_answer: str, code_executions: List[Dict] = None) -> bool:
        """Verify if solution matches expected answer with robust normalization.
        
        Args:
            solution_answer: The answer from the consensus/solution
            expected_answer: The expected correct answer
            code_executions: Optional list of code execution results to use as fallback
        
        Returns:
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
                if code_numerical and self._verify_single_answer(code_numerical, expected_answer):
                    log.info(f"  ✓ Answer verified using code output from agent {exec_info.get('agent_id')}")
                    return True
                
                # Try extracting numbers from raw code output
                code_output = exec_info.get("code_output", "")
                if code_output:
                    try:
                        output_nums = self._normalize_answer(code_output)
                        exp_nums = self._normalize_answer(expected_answer)
                        if output_nums and exp_nums and len(output_nums) == len(exp_nums):
                            if all(abs(s - e) < 1e-6 for s, e in zip(output_nums, exp_nums)):
                                log.info(f"  ✓ Answer verified using raw code output from agent {exec_info.get('agent_id')}")
                                return True
                    except Exception:
                        pass
        
        return False
    
    def _verify_single_answer(self, solution_answer: str, expected_answer: str) -> bool:
        """Verify a single answer against the expected answer."""
        if solution_answer is None or expected_answer is None:
            return False
        
        # First try direct string match (fastest)
        sol = str(solution_answer).strip().lower()
        exp = str(expected_answer).strip().lower()
        
        if sol == exp:
            return True
        
        # Try robust numerical comparison with normalization
        try:
            sol_nums = self._normalize_answer(solution_answer)
            exp_nums = self._normalize_answer(expected_answer)
            
            if not sol_nums or not exp_nums:
                return False
            
            if len(sol_nums) != len(exp_nums):
                return False
            
            # Compare with tolerance
            return all(abs(s - e) < 1e-6 for s, e in zip(sol_nums, exp_nums))
            
        except Exception as e:
            log.debug(f"Error during answer normalization: {e}")
            # Fall back to old method
            try:
                # Handle list/vector comparisons
                if "[" in sol and "[" in exp:
                    sol_list = eval(sol) if isinstance(sol, str) else sol
                    exp_list = eval(exp) if isinstance(exp, str) else exp
                    if isinstance(sol_list, list) and isinstance(exp_list, list):
                        if len(sol_list) != len(exp_list):
                            return False
                        return all(abs(float(s) - float(e)) < 1e-6 for s, e in zip(sol_list, exp_list))
                
                # Simple float comparison
                sol_num = float(sol)
                exp_num = float(exp)
                return abs(sol_num - exp_num) < 1e-6
            except (ValueError, TypeError, SyntaxError):
                pass
        
        return False
    
    def _extract_code_executions(self, all_solutions: List[Dict]) -> List[Dict]:
        """Extract code execution information and outputs from solutions."""
        code_executions = []
        for solution in all_solutions:
            code = solution.get("code", "")
            code_output = solution.get("code_output", "")
            if code:
                code_executions.append({
                    "agent_id": solution.get("agent_id"),
                    "round": solution.get("round_number"),
                    "has_code": True,
                    "code": code,
                    "code_output": code_output,
                    "numerical_answer": solution.get("numerical_answer"),
                })
        return code_executions
    
    async def run_experiment(self):
        """Run the experiment."""
        log.info("="*70)
        log.info(f"MATHEMATICAL REASONING EXPERIMENT")
        log.info("="*70)
        log.info(f"Model: {self.model_name}")
        log.info(f"Debate rounds: {self.num_rounds}")
        log.info(f"Tasks per capability: {self.num_tasks}")
        log.info(f"Condition: {self.condition}")
        log.info(f"Results directory: {self.results_dir}")
        log.info("="*70 + "\n")
        
        # Load tasks
        tasks = self.load_tasks()
        total_tasks = len(tasks)
        
        # Run baseline condition
        if self.condition in ["baseline", "both"]:
            log.info("\n" + "="*70)
            log.info("BASELINE CONDITION (No Code Execution)")
            log.info("="*70)
            for idx, task_data in enumerate(tasks, 1):
                result = await self.run_single_task(
                    task_data, "baseline", idx, total_tasks
                )
                self.results["baseline"].append(result)
        
        # Run tool-assisted condition
        if self.condition in ["tool_assisted", "both"]:
            log.info("\n" + "="*70)
            log.info("TOOL-ASSISTED CONDITION (With Code Execution)")
            log.info("="*70)
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
        """Print experiment summary."""
        log.info("\n" + "="*70)
    def _print_summary(self):
        """Print experiment summary."""
        log.info("\n" + "="*70)
        log.info("EXPERIMENT SUMMARY")
        log.info("="*70)
        
        for condition in ["baseline", "tool_assisted"]:
            if condition not in self.results or not self.results[condition]:
                continue
                
            results = self.results[condition]
            completed = [r for r in results if r["status"] == "completed"]
            successful = [r for r in completed if r.get("success", False)]
            
            log.info(f"\n{condition.upper().replace('_', ' ')}:")
            log.info(f"  Total tasks: {len(results)}")
            log.info(f"  Completed: {len(completed)}")
            log.info(f"  Successful: {len(successful)}")
            
            if completed:
                success_rate = len(successful) / len(completed) * 100
                avg_time = sum(r["execution_time"] for r in completed) / len(completed)
                log.info(f"  Success rate: {success_rate:.1f}%")
                log.info(f"  Average time: {avg_time:.2f}s")
        
        log.info("\n" + "="*70)


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
  python run_experiments.py --model gemini-2.5-pro --rounds 2 --num-tasks 10 --output my_results
        """
    )
    parser.add_argument(
        "--model",
        default="gemini-2.5-pro",
        help="LLM model to use (default: gemini-2.5-pro)"
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=1,
        help="Number of debate rounds (default: 1)"
    )
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=3,
        help="Number of tasks per capability to run (default: 3 for full experiment)"
    )
    parser.add_argument(
        "--condition",
        choices=["baseline", "tool_assisted", "both"],
        default="both",
        help="Which condition to run (default: both)"
    )
    parser.add_argument(
        "--output",
        default="experiment_results",
        help="Output directory for results (default: experiment_results)"
    )
    
    args = parser.parse_args()
    
    runner = ExperimentRunner(
        model_name=args.model,
        num_rounds=args.rounds,
        num_tasks=args.num_tasks,
        condition=args.condition,
        output_dir=args.output,
    )
    await runner.run_experiment()


if __name__ == "__main__":
    asyncio.run(main())
