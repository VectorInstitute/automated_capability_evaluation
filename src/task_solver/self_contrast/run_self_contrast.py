#!/usr/bin/env python3
"""V1 -- Base Self-Contrast runner.

Three perspectives (Top-Down, Bottom-Up, Analogical) with contrastive
reconciliation. Uses Python execution only for problems that match the
built-in computation heuristic.

Supports OpenAI, Anthropic, Google Gemini, Ollama, vec-inf, and vLLM
endpoints via ``--url``.

Example
-------
    python -m src.task_solver.self_contrast.run_self_contrast \
        --model gpt-4o --batch-file evaluation_batch.json

    # vec-inf / vLLM endpoint:
    python -m src.task_solver.self_contrast.run_self_contrast \
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

from src.task_solver.self_contrast._runner_utils import resolve_batch_file
from src.task_solver.self_contrast.evaluator import (
    evaluate_batch,
    evaluate_result,
    print_summary,
    save_results,
)
from src.task_solver.self_contrast.model_client import LLMClient
from src.task_solver.self_contrast.perspectives import BASE_PERSPECTIVES
from src.task_solver.self_contrast.solver import SelfContrastSolver


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger("self_contrast.v1")

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_DIR = SCRIPT_DIR / "dataset" / "XFinBench"
DEFAULT_RESULTS_DIR = SCRIPT_DIR / "Results"


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser (shared structure across runners)."""
    parser = argparse.ArgumentParser(
        description="V1: Base Self-Contrast (3 perspectives, contrastive reconciliation)."
    )
    parser.add_argument("--model", type=str, default="gpt-4o", help="Model name.")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help="Path to dataset directory (default: self_contrast/dataset/XFinBench/).",
    )
    parser.add_argument(
        "--batch-file",
        type=str,
        default="evaluation_batch.json",
        help="Batch file name inside dataset-dir.",
    )
    parser.add_argument("--output", type=str, help="Output JSON file name.")
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Custom directory to save results.",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="Sampling temperature."
    )
    parser.add_argument(
        "--max-problems", type=int, help="Maximum number of problems to run."
    )
    parser.add_argument(
        "--url",
        type=str,
        help="Base URL for vec-inf / vLLM / custom OpenAI-compatible endpoint.",
    )
    parser.add_argument("--api-key", type=str, help="API key (or use env vars).")
    parser.add_argument(
        "--prompt-repeat",
        type=int,
        default=1,
        choices=[1, 2, 3],
        help="Repeat perspective prompt N times (arXiv:2512.14982).",
    )

    json_group = parser.add_mutually_exclusive_group()
    json_group.add_argument(
        "--force-json",
        dest="force_json",
        action="store_true",
        help="Force JSON output via response_format (default: on).",
    )
    json_group.add_argument(
        "--no-force-json",
        dest="force_json",
        action="store_false",
        help="Disable forced JSON output.",
    )
    parser.set_defaults(force_json=True)
    return parser


async def run(args: argparse.Namespace) -> None:
    """Execute the V1 Self-Contrast pipeline."""
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

    solver = SelfContrastSolver(
        client,
        list(BASE_PERSPECTIVES),
        tool_mode="none",
        prompt_repeat=args.prompt_repeat,
        force_json=args.force_json,
    )

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
        "method": "Self-Contrast",
        "perspectives": [p["id"] for p in BASE_PERSPECTIVES],
        "prompt_repeat": args.prompt_repeat,
        "temperature": args.temperature,
        "batch_info": batch_path.stem,
        "execution_time_seconds": round(elapsed, 2),
        "execution_time_formatted": time_fmt,
    }

    safe_model = args.model.replace("/", "-").replace(":", "-")
    version_dir = results_dir / "v1_self_contrast" / safe_model
    version_dir.mkdir(parents=True, exist_ok=True)

    if args.output:
        out_path = version_dir / args.output
    else:
        ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_path = version_dir / (
            f"{batch_path.stem}_T{args.temperature}_R{args.prompt_repeat}_{ts}.json"
        )

    save_results(batch_metrics, results, out_path, extra_metadata=extra)
    print_summary(
        batch_metrics,
        model=args.model,
        method="Self-Contrast (V1: Base)",
        batch_info=batch_path.stem,
        execution_time=time_fmt,
    )


def main() -> None:
    """CLI entry point for V1 Self-Contrast."""
    parser = build_parser()
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
