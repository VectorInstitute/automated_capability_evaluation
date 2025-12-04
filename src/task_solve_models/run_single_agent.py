
import json
import os
import logging
import argparse
import asyncio
from pathlib import Path

from src.task_solve_models.solver import FinancialProblemSolver
from src.task_solve_models.evaluator import evaluate_result
from src.utils.model_client_utils import get_model_client

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# --- Configuration ---
SCRIPT_DIR = Path(__file__).parent
DATASET_DIR = SCRIPT_DIR / "dataset" / "XFinBench"
RESULTS_DIR = SCRIPT_DIR / "Results"

async def main():
    parser = argparse.ArgumentParser(description="Run single agent solver on financial problems.")
    parser.add_argument("--batch-file", type=str, help="Specific batch file to run (e.g., batch_1.json). If not provided, runs all.")
    parser.add_argument("--save-home", action="store_true", help="Also save results to user's home directory.")
    parser.add_argument("--model", type=str, default="gpt-4o", help="Model to use (e.g., gpt-4o, claude-3-5-sonnet-20241022, gemini-2.5-pro).")
    args = parser.parse_args()

    # Ensure results directory exists
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        model_client = get_model_client(args.model)
        log.info(f"Initialized client for model: {args.model}")
    except Exception as e:
        log.error(f"Failed to initialize model client: {e}")
        return

    solver = FinancialProblemSolver(model_client)
    
    all_results = []
    
    # Determine which files to process
    if args.batch_file:
        batch_files = [DATASET_DIR / args.batch_file]
        batch_suffix = f"_{Path(args.batch_file).stem}" # e.g. _batch_1
    else:
        batch_files = sorted(DATASET_DIR.glob("batch_*.json"))
        batch_suffix = "_all_batches"
    
    if not batch_files:
        log.error(f"No batch files found in {DATASET_DIR}")
        return

    # Construct dynamic output filename
    # Sanitize model name for filename (replace characters that might be issues)
    safe_model_name = args.model.replace("/", "-").replace(":", "-")
    output_filename = f"results_{safe_model_name}{batch_suffix}.json"

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
                result = await solver.solve_problem(problem)
                is_correct = evaluate_result(result)
                result["is_correct"] = is_correct
                
                if is_correct:
                    total_correct += 1
                total_processed += 1
                
                all_results.append(result)
                log.info(f"Problem {problem.get('id')} ({problem.get('task')}): {'CORRECT' if is_correct else 'INCORRECT'}")
                if not is_correct:
                    log.info(f"  Expected: {result['ground_truth']}, Got: {result['prediction']}")
                
        except Exception as e:
            log.error(f"Error reading/processing batch file {batch_file}: {e}")

    # Calculate metrics
    accuracy = total_correct / total_processed if total_processed > 0 else 0
    metrics = {
        "model": args.model,
        "batch_info": batch_suffix.strip("_"),
        "total_processed": total_processed,
        "total_correct": total_correct,
        "accuracy": accuracy
    }
    
    log.info("="*30)
    log.info(f"Model: {args.model}")
    log.info(f"Batch Info: {batch_suffix.strip('_')}")
    log.info(f"Total Processed: {total_processed}")
    log.info(f"Total Correct: {total_correct}")
    log.info(f"Accuracy: {accuracy:.2%}")
    log.info("="*30)

    # Save results locally in Results folder
    output_data = {
        "metrics": metrics,
        "results": all_results
    }
    
    local_output_path = RESULTS_DIR / output_filename
    try:
        with open(local_output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2)
        log.info(f"Results saved to {local_output_path}")
    except Exception as e:
        log.error(f"Failed to save local results: {e}")

    # Optionally save to home directory
    if args.save_home:
        home_output_path = Path.home() / output_filename
        try:
            with open(home_output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2)
            log.info(f"Results also saved to {home_output_path}")
        except Exception as e:
            log.error(f"Failed to save results to home directory: {e}")

if __name__ == "__main__":
    asyncio.run(main())
