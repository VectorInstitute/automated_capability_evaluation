"""
Run multi-agent debate solver on financial problems.
"""
import json
import os
import logging
import argparse
import asyncio
import traceback
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, List

from autogen_core import (
    SingleThreadedAgentRuntime,
    DefaultTopicId,
)
from langfuse import Langfuse

# Import from the copied multi-agent solver modules
from src.task_solve_models.multi_agent_solver.messages import Task
from src.task_solve_models.multi_agent_solver.moderator import TaskSolverModerator
from src.task_solve_models.multi_agent_solver.scientist import TaskSolverScientist
from src.utils.model_client_utils import get_model_client, RetryableModelClient
from src.task_solve_models.evaluator import evaluate_result

# Import OpenAI client for manual local construction
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import ModelInfo

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# --- Configuration ---
SCRIPT_DIR = Path(__file__).parent
DATASET_DIR = SCRIPT_DIR / "dataset" / "XFinBench"
RESULTS_DIR = SCRIPT_DIR / "Results"

def get_client_wrapper(model_name: str, seed: int):
    """
    Wrapper to get model client, supporting local Ollama models.
    """
    local_models = ["llama", "mistral", "gemma", "qwen", "phi", "deepseek"]
    
    # Check if model name contains any of the known local model keywords
    if any(keyword in model_name.lower() for keyword in local_models):
        log.info(f"Using local Ollama client for model: {model_name}")
        client = OpenAIChatCompletionClient(
            model=model_name,
            api_key="EMPTY",
            base_url="http://localhost:11434/v1",
            model_info=ModelInfo(
                vision=False,
                function_calling=False,
                json_output=True,
                structured_output=True,
                family="unknown"
            )
        )
        return RetryableModelClient(client)
    
    # Fallback to standard utility
    return get_model_client(model_name, seed=seed)

async def run_multi_agent_debate(
    problem_data: Dict[str, Any],
    output_dir: Path,
    model_specs: Dict[str, str],
    max_rounds: int,
    langfuse_client: Langfuse
) -> Dict[str, Any]:
    
    task_id = problem_data.get("id")
    problem_text = f"Question: {problem_data.get('question')}\n"
    if problem_data.get("choice"):
        problem_text += f"Choices:\n{problem_data.get('choice')}\n"
        
    capability = problem_data.get("fin_capability", "General")
    area = "Financial" # Generic area
    
    task = Task(
        task_id=task_id,
        problem=problem_text,
        capability_name=capability,
        area_name=area,
        task_type=problem_data.get("task", "unknown")
    )
    
    # Setup output directory for this specific task solution
    task_output_dir = output_dir / f"{task_id}_solution"
    task_output_dir.mkdir(parents=True, exist_ok=True)
    
    runtime = SingleThreadedAgentRuntime()

    # Register Moderator
    await TaskSolverModerator.register(
        runtime,
        "TaskSolverModerator",
        lambda: TaskSolverModerator(
            model_client=get_client_wrapper(model_specs["moderator"], seed=888),
            num_solvers=2,
            max_rounds=max_rounds,
            output_dir=task_output_dir,
            langfuse_client=langfuse_client,
        ),
    )

    # Register Scientist A
    await TaskSolverScientist.register(
        runtime,
        "TaskSolverScientistA",
        lambda: TaskSolverScientist(
            model_client=get_client_wrapper(model_specs["scientist_a"], seed=8),
            scientist_id="A",
            langfuse_client=langfuse_client,
        ),
    )

    # Register Scientist B
    await TaskSolverScientist.register(
        runtime,
        "TaskSolverScientistB",
        lambda: TaskSolverScientist(
            model_client=get_client_wrapper(model_specs["scientist_b"], seed=88),
            scientist_id="B",
            langfuse_client=langfuse_client,
        ),
    )

    # Start runtime
    runtime.start()
    
    # Publish task
    await runtime.publish_message(task, DefaultTopicId())
    
    # Wait for completion
    await runtime.stop_when_idle()
    
    # Retrieve the result
    solution_file = task_output_dir / f"{task_id}_solution.json"
    if solution_file.exists():
        with open(solution_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

async def main():
    parser = argparse.ArgumentParser(description="Run multi-agent debate solver on financial problems.")
    parser.add_argument("--batch-file", type=str, help="Specific batch file to run (e.g., batch_1.json).")
    parser.add_argument("--save-home", action="store_true", help="Also save results to user's home directory.")
    
    # Model arguments
    parser.add_argument("--model", type=str, default="gpt-4o", help="Default model to use for all agents if specific ones aren't set.")
    parser.add_argument("--model-moderator", type=str, help="Model for Moderator.")
    parser.add_argument("--model-scientist-a", type=str, help="Model for Scientist A.")
    parser.add_argument("--model-scientist-b", type=str, help="Model for Scientist B.")
    
    parser.add_argument("--max-rounds", type=int, default=1, help="Maximum debate rounds.")
    args = parser.parse_args()

    # Construct model specs
    model_specs = {
        "moderator": args.model_moderator or args.model,
        "scientist_a": args.model_scientist_a or args.model,
        "scientist_b": args.model_scientist_b or args.model
    }

    # Ensure results directory exists
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Initialize Langfuse
    try:
        langfuse_client = Langfuse()
    except Exception as e:
        log.warning(f"Langfuse initialization failed: {e}. Tracing disabled.")
        return 

    all_results = []
    
    if args.batch_file:
        batch_files = [DATASET_DIR / args.batch_file]
        batch_suffix = f"_{Path(args.batch_file).stem}"
    else:
        batch_files = sorted(DATASET_DIR.glob("batch_*.json"))
        batch_suffix = "_all_batches"
        
    if not batch_files:
        log.error(f"No batch files found in {DATASET_DIR}")
        return

    # Create a safe name for the output file based on the models
    models_str = f"mod_{model_specs['moderator']}_scA_{model_specs['scientist_a']}_scB_{model_specs['scientist_b']}"
    safe_models_str = models_str.replace("/", "-").replace(":", "-")
    output_filename = f"multi_agent_results_{safe_models_str}{batch_suffix}.json"

    total_correct = 0
    total_processed = 0

    for batch_file in batch_files:
        if not batch_file.exists():
            log.error(f"Batch file not found: {batch_file}")
            continue
            
        log.info(f"Processing {batch_file.name}...")
        try:
            with open(batch_file, 'r', encoding='utf-8') as f:
                problems = json.load(f)
            
            for problem in problems:
                log.info(f"Solving problem {problem.get('id')}...")
                try:
                    # Run debate
                    debate_result = await run_multi_agent_debate(
                        problem, 
                        RESULTS_DIR / "temp_debate_outputs", 
                        model_specs, 
                        args.max_rounds, 
                        langfuse_client
                    )
                    
                    final_prediction = None
                    reasoning = ""
                    
                    if debate_result:
                        # Use the structured 'answer' field if available (New Feature)
                        if debate_result.get("answer") and str(debate_result.get("answer")).lower() != "null":
                             final_prediction = debate_result.get("answer")
                        
                        # Fallback to numerical_answer for calc tasks if answer is missing
                        elif problem.get("task") == "calcu":
                            final_prediction = debate_result.get("numerical_answer")
                            if final_prediction == "null" or final_prediction is None:
                                final_prediction = debate_result.get("solution")
                        else:
                            final_prediction = debate_result.get("solution")
                            
                        reasoning = debate_result.get("reasoning", "")
                        
                    # Construct result object for evaluation
                    eval_input = {
                        "id": problem.get("id"),
                        "prediction": final_prediction,
                        "ground_truth": problem.get("ground_truth"),
                        "task_type": problem.get("task"),
                        "reasoning": reasoning,
                        "debate_info": debate_result
                    }
                    
                    is_correct = evaluate_result(eval_input)
                    eval_input["is_correct"] = is_correct
                    
                    if is_correct:
                        total_correct += 1
                    total_processed += 1
                    
                    all_results.append(eval_input)
                    log.info(f"Problem {problem.get('id')}: {'CORRECT' if is_correct else 'INCORRECT'}")
                    if not is_correct:
                         log.info(f"  Expected: {problem.get('ground_truth')}, Got: {final_prediction}")

                except Exception as e:
                    log.error(f"Error solving problem {problem.get('id')}: {e}")
                    log.error(traceback.format_exc())

        except Exception as e:
            log.error(f"Error reading batch file {batch_file}: {e}")

    # Metrics
    accuracy = total_correct / total_processed if total_processed > 0 else 0
    metrics = {
        "model_specs": model_specs,
        "batch_info": batch_suffix.strip("_"),
        "total_processed": total_processed,
        "total_correct": total_correct,
        "accuracy": accuracy,
        "mode": "multi-agent-debate"
    }
    
    log.info("="*30)
    log.info(f"Multi-Agent Run Results")
    log.info(f"Models: {model_specs}")
    log.info(f"Total Processed: {total_processed}")
    log.info(f"Total Correct: {total_correct}")
    log.info(f"Accuracy: {accuracy:.2%}")
    log.info("="*30)

    # Save
    output_data = {
        "metrics": metrics,
        "results": all_results
    }
    
    local_output_path = RESULTS_DIR / output_filename
    with open(local_output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)
    log.info(f"Results saved to {local_output_path}")
    
    if args.save_home:
        home_output_path = Path.home() / output_filename
        with open(home_output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2)
        log.info(f"Results also saved to {home_output_path}")

if __name__ == "__main__":
    asyncio.run(main())
