"""
Run ReCAP-style solver on financial problems.

This runner mirrors `run_multi_agent.py` output shape (metrics + results),
but uses a ReCAP controller/worker loop (plan → step → refine).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List

from autogen_core import DefaultTopicId, SingleThreadedAgentRuntime
from autogen_core.models import ModelInfo
from autogen_ext.models.openai import OpenAIChatCompletionClient
from langfuse import Langfuse

from src.task_solve_models.evaluator import evaluate_result
from src.task_solve_models.multi_agent_solver.RECAP.controller import RecapController
from src.task_solve_models.multi_agent_solver.RECAP.messages import RecapTask
from src.task_solve_models.multi_agent_solver.RECAP.worker import RecapWorker
from src.utils.model_client_utils import RetryableModelClient, get_model_client

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).parent
DATASET_DIR = SCRIPT_DIR / "dataset" / "XFinBench"
RESULTS_DIR = SCRIPT_DIR / "Results"


class _NullSpan:
    def update(self, metadata: Dict[str, Any] | None = None) -> None:
        return

    def __enter__(self) -> "_NullSpan":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return


class _NullLangfuse:
    def start_as_current_span(self, name: str, **kwargs: Any) -> _NullSpan:
        return _NullSpan()


def get_client_wrapper(model_name: str, seed: int):
    """Wrapper to get model client, supporting local Ollama models."""
    local_models = ["llama", "mistral", "gemma", "qwen", "phi", "deepseek"]
    if any(k in model_name.lower() for k in local_models):
        log.info("Using local Ollama client for model: %s", model_name)
        client = OpenAIChatCompletionClient(
            model=model_name,
            api_key="EMPTY",
            base_url="http://localhost:11434/v1",
            model_info=ModelInfo(
                vision=False,
                function_calling=False,
                json_output=True,
                structured_output=True,
                family="unknown",
            ),
        )
        return RetryableModelClient(client)
    return get_model_client(model_name, seed=seed)


def _safe_name(s: str) -> str:
    return s.replace("/", "-").replace(":", "-")


async def run_recap_task(
    *,
    problem_data: Dict[str, Any],
    output_dir: Path,
    controller_model: str,
    worker_models: List[str],
    max_steps: int,
    langfuse_client: Any,
) -> Dict[str, Any] | None:
    task_id = problem_data.get("id")
    problem_text = f"Question: {problem_data.get('question')}\n"
    if problem_data.get("choice"):
        problem_text += f"Choices:\n{problem_data.get('choice')}\n"

    task = RecapTask(
        task_id=task_id,
        problem=problem_text,
        capability_name=problem_data.get("fin_capability", "General"),
        area_name="Financial",
        task_type=problem_data.get("task", "unknown"),
    )

    task_output_dir = output_dir / "recap_outputs"
    task_output_dir.mkdir(parents=True, exist_ok=True)

    runtime = SingleThreadedAgentRuntime()

    worker_ids = [chr(ord("A") + i) for i in range(len(worker_models))]

    # Register workers
    for idx, wid in enumerate(worker_ids):
        model_name = worker_models[idx]
        seed = 1000 + idx
        await RecapWorker.register(
            runtime,
            f"RecapWorker{wid}",
            lambda model_name=model_name, wid=wid, seed=seed: RecapWorker(
                model_client=get_client_wrapper(model_name, seed=seed),
                worker_id=wid,
                langfuse_client=langfuse_client,
            ),
        )

    # Register controller
    await RecapController.register(
        runtime,
        "RecapController",
        lambda: RecapController(
            model_client=get_client_wrapper(controller_model, seed=42),
            output_dir=task_output_dir,
            worker_ids=worker_ids,
            langfuse_client=langfuse_client,
            max_steps=max_steps,
        ),
    )

    runtime.start()
    await runtime.publish_message(task, DefaultTopicId())
    await runtime.stop_when_idle()

    solution_file = task_output_dir / f"{task_id}_solution.json"
    if solution_file.exists():
        with open(solution_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run ReCAP-style solver on financial problems.")
    parser.add_argument("--batch-file", type=str, help="Batch file to run (e.g., evaluation_batch.json).")
    parser.add_argument("--save-home", action="store_true", help="Also save results to user's home directory.")

    parser.add_argument("--model", type=str, default="gpt-4o", help="Default model if specific ones not set.")
    parser.add_argument("--model-controller", type=str, help="Model for ReCAP controller.")
    parser.add_argument("--model-worker", type=str, help="Model for ReCAP worker(s) (if specific not set).")
    parser.add_argument("--model-worker-a", type=str, help="Model for worker A.")
    parser.add_argument("--model-worker-b", type=str, help="Model for worker B.")
    parser.add_argument("--model-worker-c", type=str, help="Model for worker C.")
    parser.add_argument("--num-workers", type=int, default=1, help="Number of workers (1-3).")
    parser.add_argument("--max-steps", type=int, default=12, help="Maximum ReCAP steps per task.")

    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        langfuse_client: Any = Langfuse()
    except Exception as e:
        log.warning("Langfuse initialization failed: %s. Tracing disabled.", e)
        langfuse_client = _NullLangfuse()

    if args.batch_file:
        batch_files = [DATASET_DIR / args.batch_file]
        batch_suffix = f"_{Path(args.batch_file).stem}"
    else:
        batch_files = sorted(DATASET_DIR.glob("batch_*.json"))
        batch_suffix = "_all_batches"

    # Resolve models
    controller_model = args.model_controller or args.model
    default_worker = args.model_worker or args.model
    worker_models = [
        args.model_worker_a or default_worker,
        args.model_worker_b or default_worker,
        args.model_worker_c or default_worker,
    ][: max(1, min(3, args.num_workers))]

    models_str = f"ctrl_{controller_model}_wk_" + "_".join(worker_models)
    output_filename = f"recap_results_{_safe_name(models_str)}{batch_suffix}.json"

    total_correct = 0
    total_processed = 0
    all_results: list[dict[str, Any]] = []

    start_time = time.time()

    for batch_file in batch_files:
        if not batch_file.exists():
            log.error("Batch file not found: %s", batch_file)
            continue
        log.info("Processing %s...", batch_file.name)
        try:
            with open(batch_file, "r", encoding="utf-8") as f:
                problems = json.load(f)
        except Exception as e:
            log.error("Error reading batch file %s: %s", batch_file, e)
            continue

        for problem in problems:
            log.info("Solving problem %s...", problem.get("id"))
            try:
                recap_solution = await run_recap_task(
                    problem_data=problem,
                    output_dir=RESULTS_DIR,
                    controller_model=controller_model,
                    worker_models=worker_models,
                    max_steps=args.max_steps,
                    langfuse_client=langfuse_client,
                )

                prediction: Any = None
                reasoning = ""
                if recap_solution:
                    # recap_solution schema is RecapFinalSolution.to_dict()
                    task_type = problem.get("task")
                    if task_type == "calcu":
                        prediction = recap_solution.get("numerical_answer")
                        if prediction in (None, "null"):
                            prediction = recap_solution.get("answer")
                    else:
                        prediction = recap_solution.get("answer")
                    reasoning = recap_solution.get("reasoning", "") or ""

                eval_input = {
                    "id": problem.get("id"),
                    "prediction": prediction,
                    "ground_truth": problem.get("ground_truth"),
                    "task_type": problem.get("task"),
                    "reasoning": reasoning,
                    "recap_info": recap_solution,
                }

                is_correct = evaluate_result(eval_input)
                eval_input["is_correct"] = is_correct
                if is_correct:
                    total_correct += 1
                total_processed += 1
                all_results.append(eval_input)
                log.info("Problem %s: %s", problem.get("id"), "CORRECT" if is_correct else "INCORRECT")
            except Exception as e:
                log.error("Error solving problem %s: %s", problem.get("id"), e)
                log.error(traceback.format_exc())

    end_time = time.time()
    execution_time_seconds = end_time - start_time
    execution_time_formatted = f"{int(execution_time_seconds // 60)}m {int(execution_time_seconds % 60)}s"

    accuracy = total_correct / total_processed if total_processed else 0.0
    metrics = {
        "model_specs": {
            "controller": controller_model,
            "workers": worker_models,
        },
        "batch_info": batch_suffix.strip("_"),
        "total_processed": total_processed,
        "total_correct": total_correct,
        "accuracy": accuracy,
        "mode": "recap",
        "max_steps": args.max_steps,
        "execution_time_seconds": round(execution_time_seconds, 2),
        "execution_time_formatted": execution_time_formatted,
    }

    output_data = {"metrics": metrics, "results": all_results}
    local_output_path = RESULTS_DIR / output_filename
    with open(local_output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    log.info("Results saved to %s", local_output_path)

    if args.save_home:
        home_output_path = Path.home() / output_filename
        with open(home_output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        log.info("Results also saved to %s", home_output_path)


if __name__ == "__main__":
    asyncio.run(main())

