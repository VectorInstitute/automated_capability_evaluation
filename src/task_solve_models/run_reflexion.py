"""Script to run the Reflexion agent system."""

import argparse
import asyncio
import json
import logging
from pathlib import Path
import traceback

from src.utils.model_client_utils import get_model_client
from src.task_solve_models.reflexion.core import ReflexionSystem
from src.task_solve_models.reflexion.client import AgentClient
from src.task_solve_models.evaluator import evaluate_result

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("reflexion.run")

async def main(args):
    # 1. Setup Models
    log.info(f"Initializing Solver ({args.model_solver}) and Critic ({args.model_critic})...")
    
    # Use OpenAI adapter for Gemini/Claude if needed, handled by utils
    solver_llm = get_model_client(args.model_solver)
    critic_llm = get_model_client(args.model_critic)
    
    solver_client = AgentClient(solver_llm, "Solver")
    critic_client = AgentClient(critic_llm, "Critic")
    
    system = ReflexionSystem(solver_client, critic_client, max_rounds=args.max_rounds)

    # 2. Load Dataset
    batch_file = Path("src/task_solve_models/dataset/XFinBench") / args.batch_file
    log.info(f"Loading tasks from {batch_file}...")
    with open(batch_file, "r") as f:
        tasks = json.load(f)

    results_dir = Path("src/task_solve_models/Results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    final_results = []
    total_correct = 0

    # 3. Process Tasks
    for i, task in enumerate(tasks):
        task_id = task["id"]
        # Handle different dataset field names if necessary (e.g., 'question' vs 'problem')
        problem = task.get("problem") or task.get("question")
        
        # If choice field exists, append it to the problem text
        if "choice" in task and task["choice"]:
            problem += f"\nChoices:\n{task['choice']}"
            
        ground_truth = task.get("solution") or task.get("ground_truth")
        task_type = task.get("type") or task.get("task") # e.g. "calcu", "bool", "mcq"

        log.info(f"Processing Task {i+1}/{len(tasks)}: {task_id}")
        
        try:
            result = await system.solve(task_id, problem)
            
            # Determine prediction to use for evaluation
            prediction_to_use = result.answer
            if task_type == "calcu" and result.numerical_answer and str(result.numerical_answer).lower() not in ["null", "none"]:
                 prediction_to_use = result.numerical_answer
            
            # 4. Evaluate
            is_correct = evaluate_result({
                "prediction": prediction_to_use,
                "ground_truth": ground_truth,
                "task_type": task_type
            })
            
            if is_correct:
                total_correct += 1
                log.info(f"Task {task_id}: CORRECT")
            else:
                log.info(f"Task {task_id}: INCORRECT (Pred: {result.answer}, GT: {ground_truth})")

            # 5. Save Record
            record = {
                "id": task_id,
                "problem": problem,
                "prediction": result.answer,
                "ground_truth": ground_truth,
                "task_type": task_type,
                "is_correct": is_correct,
                "numerical_answer": result.numerical_answer,
                "reasoning": result.solution,
                "trace": result.reasoning_trace,
                "total_rounds": result.total_rounds
            }
            final_results.append(record)

        except Exception as e:
            log.error(f"Error processing {task_id}: {e}")
            log.error(traceback.format_exc())
            final_results.append({
                "id": task_id,
                "error": str(e),
                "is_correct": False
            })

    # 6. Save Final Results
    output_filename = f"reflexion_results_solver_{args.model_solver}_critic_{args.model_critic}_{args.batch_file}"
    output_path = results_dir / output_filename
    
    output_data = {
        "metrics": {
            "solver_model": args.model_solver,
            "critic_model": args.model_critic,
            "batch": args.batch_file,
            "total": len(tasks),
            "correct": total_correct,
            "accuracy": total_correct / len(tasks) if tasks else 0
        },
        "results": final_results
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    
    log.info(f"Results saved to {output_path}")
    log.info(f"Final Accuracy: {total_correct}/{len(tasks)} ({output_data['metrics']['accuracy']:.2%})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Reflexion Agent System")
    parser.add_argument("--batch-file", type=str, required=True, help="Filename in dataset/XFinBench")
    parser.add_argument("--model-solver", type=str, default="gpt-4o", help="Model for Solver agent")
    parser.add_argument("--model-critic", type=str, default="gpt-4o", help="Model for Critic agent")
    parser.add_argument("--max-rounds", type=int, default=3, help="Max reflection rounds")
    
    args = parser.parse_args()
    asyncio.run(main(args))

