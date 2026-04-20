#!/usr/bin/env python3
"""V2 -- Single Agent baseline runner.

One LLM call per problem with task-type-specific prompts.  No perspectives,
no contrast.  Included for side-by-side comparison with Self-Contrast
variants.

Example
-------
    python -m src.task_solver.self_contrast.run_single_agent \
        --model gpt-4o --batch-file evaluation_batch.json

    # vec-inf / vLLM endpoint:
    python -m src.task_solver.self_contrast.run_single_agent \
        --model Qwen2.5-72B-Instruct --url http://10.1.1.29:8081/v1
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from src.task_solver.self_contrast._runner_utils import (
    parse_single_agent_response,
    resolve_batch_file,
)
from src.task_solver.self_contrast.evaluator import (
    evaluate_batch,
    evaluate_result,
    print_summary,
    save_results,
)
from src.task_solver.self_contrast.model_client import LLMClient
from src.task_solver.self_contrast.prompts import (
    SINGLE_AGENT_SYSTEM_PROMPT,
    SINGLE_AGENT_USER_PROMPT,
)


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger("self_contrast.v2")

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_DIR = SCRIPT_DIR / "dataset" / "XFinBench"
DEFAULT_RESULTS_DIR = SCRIPT_DIR / "Results"

ANSWER_FORMATS: Dict[str, str] = {
    "mcq": "The option letter (e.g., A, B, C)",
    "bool": "1.0 for True/Yes, 0.0 for False/No",
    "calcu": "The numerical value only (no units)",
}


class SingleAgentSolver:
    """Baseline single-call solver.

    Parameters
    ----------
    client : LLMClient
        Lightweight model client.
    force_json : bool
        Request JSON response format when supported.
    """

    def __init__(self, client: LLMClient, *, force_json: bool = True) -> None:
        self.client = client
        self.force_json = force_json
        self._use_threads = getattr(client, "provider", "") not in (
            "ollama",
            "openai_compatible",
        )

    async def solve_problem(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """Solve a single problem with one LLM call."""
        task_type = problem.get("task") or problem.get("task_type") or ""
        question = problem.get("question", "")
        choices = problem.get("choice", "")

        choices_section = f"Choices:\n{choices}\n" if choices else ""
        answer_format = ANSWER_FORMATS.get(task_type, "Your final answer")

        user_prompt = SINGLE_AGENT_USER_PROMPT.format(
            question=question,
            choices_section=choices_section,
            answer_format=answer_format,
        )

        response = await self._call_llm(
            SINGLE_AGENT_SYSTEM_PROMPT, user_prompt, force_json=self.force_json
        )

        prediction, reasoning = parse_single_agent_response(response, task_type)

        return {
            "id": problem.get("id"),
            "prediction": prediction,
            "reasoning": reasoning,
            "ground_truth": problem.get("ground_truth"),
            "task_type": task_type,
            "raw_response": response,
        }

    async def _call_llm(self, system: str, user: str, *, force_json: bool) -> str:
        if self._use_threads:
            return await asyncio.to_thread(
                self.client.call, system, user, force_json=force_json
            )
        return self.client.call(system, user, force_json=force_json)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for V2 Single Agent."""
    parser = argparse.ArgumentParser(
        description="V2: Single Agent baseline (one LLM call per problem)."
    )
    parser.add_argument("--model", type=str, default="gpt-4o", help="Model name.")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help="Path to dataset directory.",
    )
    parser.add_argument(
        "--batch-file",
        type=str,
        default="evaluation_batch.json",
        help="Batch file name inside dataset-dir.",
    )
    parser.add_argument("--output", type=str, help="Output JSON file name.")
    parser.add_argument("--results-dir", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-problems", type=int)
    parser.add_argument(
        "--url",
        type=str,
        help="Base URL for vec-inf / vLLM / custom endpoint.",
    )
    parser.add_argument("--api-key", type=str)

    json_group = parser.add_mutually_exclusive_group()
    json_group.add_argument("--force-json", dest="force_json", action="store_true")
    json_group.add_argument("--no-force-json", dest="force_json", action="store_false")
    parser.set_defaults(force_json=True)
    return parser


async def run(args: argparse.Namespace) -> None:
    """Execute the V2 Single Agent pipeline."""
    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else DEFAULT_DATASET_DIR
    results_dir = Path(args.results_dir) if args.results_dir else DEFAULT_RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    batch_path = resolve_batch_file(args.batch_file, dataset_dir)
    with open(batch_path, "r", encoding="utf-8") as f:
        problems: List[Dict[str, Any]] = json.load(f)

    if args.max_problems:
        problems = problems[: args.max_problems]

    client = LLMClient(
        args.model,
        base_url=args.url,
        api_key=args.api_key,
        temperature=args.temperature,
    )
    solver = SingleAgentSolver(client, force_json=args.force_json)

    start = time.time()
    results: List[Dict[str, Any]] = []

    for idx, problem in enumerate(problems, start=1):
        pid = problem.get("id", f"problem_{idx}")
        log.info("Problem %s/%s: %s", idx, len(problems), pid)
        try:
            result = await solver.solve_problem(problem)
            result["is_correct"] = evaluate_result(result)
            results.append(result)
            log.info("  -> %s", "CORRECT" if result["is_correct"] else "INCORRECT")
        except Exception as exc:
            log.error("Error on %s: %s", pid, exc)

    elapsed = time.time() - start
    time_fmt = f"{int(elapsed // 60)}m {int(elapsed % 60)}s"

    batch_metrics = evaluate_batch(results)
    extra = {
        "model": args.model,
        "method": "Single Agent",
        "temperature": args.temperature,
        "batch_info": batch_path.stem,
        "execution_time_seconds": round(elapsed, 2),
        "execution_time_formatted": time_fmt,
    }

    safe_model = args.model.replace("/", "-").replace(":", "-")
    version_dir = results_dir / "v2_single_agent" / safe_model
    version_dir.mkdir(parents=True, exist_ok=True)

    if args.output:
        out_path = version_dir / args.output
    else:
        ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_path = version_dir / f"{batch_path.stem}_T{args.temperature}_{ts}.json"

    save_results(batch_metrics, results, out_path, extra_metadata=extra)
    print_summary(
        batch_metrics,
        model=args.model,
        method="Single Agent (V2: Baseline)",
        batch_info=batch_path.stem,
        execution_time=time_fmt,
    )


def main() -> None:
    """CLI entry point for V2 Single Agent."""
    parser = build_parser()
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
