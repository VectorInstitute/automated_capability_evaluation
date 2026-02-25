#!/usr/bin/env python3
"""
XFinBench Task Classifier for Tool-Assisted Evaluation

Classifies XFinBench financial tasks to identify:
1. Tool-solvable vs explanation-focused
2. Quantitative vs qualitative answers
3. Ground truth quality issues
4. Tasks requiring vision (to filter out)
"""

import argparse
import csv
import json
import logging
import os
import random
from pathlib import Path
from typing import Dict, List, Any, Optional

from dotenv import load_dotenv
from google import genai
from google.genai.types import GenerateContentConfig

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)


XFINBENCH_CLASSIFICATION_PROMPT = """You are an expert at analyzing financial problems to determine their suitability for computational tool-assisted solving and evaluation.

Analyze the following financial task and classify it across multiple dimensions:

TASK:
ID: {task_id}
Task Type: {task_type}
Question: {question}
{choices_section}
Expected Answer: {answer}
Financial Capability: {fin_capability}

CLASSIFICATION DIMENSIONS:

1. **Tool Solvability** (tool_solvable: true/false)
   - Can this problem be meaningfully solved or verified using computational tools (Python with NumPy, SciPy, SymPy)?
   - Financial calculations (NPV, IRR, option pricing, ratios) ARE tool-solvable
   - Pure terminology understanding or conceptual explanations are NOT tool-solvable
   - Temporal reasoning requiring quantitative analysis IS tool-solvable
   - Scenario planning with numerical modeling IS tool-solvable

2. **Requires Vision** (requires_vision: true/false)
   - Does the problem reference figures, charts, or visual elements that are essential?
   - Tasks with LaTeX tables/data embedded in text do NOT require vision
   - Tasks explicitly mentioning "see figure" or "shown in chart" DO require vision

3. **Answer Type** (answer_type: "boolean", "mcq", "numerical", "qualitative", "mixed")
   - "boolean": True/False (0/1) answer
   - "mcq": Multiple choice with options
   - "numerical": Specific numerical answer(s)
   - "qualitative": Text explanations, descriptions, conceptual answers
   - "mixed": Both quantitative and qualitative components

4. **Ground Truth Quality** (ground_truth_quality: "good", "problematic", "missing")
   - "good": Clear, well-specified expected answer
   - "problematic": Answer unclear, contradictory, or format issues
   - "missing": No answer provided or placeholder value

5. **Verification Feasibility** (verification_feasibility: "direct", "numeric_check", "hard")
   - "direct": Simple comparison (bool, mcq choice letter)
   - "numeric_check": Requires numerical tolerance comparison
   - "hard": Complex multi-part or subjective evaluation

6. **Problem Difficulty** (problem_difficulty: "easy", "medium", "hard", "very_hard")
   - Based on financial complexity, calculation steps, required knowledge level

7. **Tool Benefit** (tool_benefit: "high", "medium", "low", "none")
   - "high": Complex calculations, numerical modeling, formula verification
   - "medium": Moderate computation, could solve by hand but tools help
   - "low": Simple arithmetic, tools don't add much value
   - "none": Pure conceptual/terminology, no computation involved

8. **Recommendation** (recommend_for_evaluation: true/false)
   - Should this task be included in a tool-assisted evaluation dataset?
   - Consider: tool-solvability, verification feasibility, no vision required
   - CRITICAL: Tasks requiring vision should NOT be recommended
   - Focus on tasks where tools provide measurable benefit

IMPORTANT: Return ONLY a valid JSON object with this exact structure:
{{
    "tool_solvable": true/false,
    "tool_solvability_reasoning": "Brief explanation",
    "requires_vision": true/false,
    "answer_type": "boolean/mcq/numerical/qualitative/mixed",
    "ground_truth_quality": "good/problematic/missing",
    "ground_truth_issues": "Specific issues if any, or null",
    "verification_feasibility": "direct/numeric_check/hard",
    "problem_difficulty": "easy/medium/hard/very_hard",
    "tool_benefit": "high/medium/low/none",
    "recommend_for_evaluation": true/false,
    "recommendation_reasoning": "Why include or exclude this task"
}}

Respond with ONLY the JSON object, no additional text or formatting."""


class XFinBenchClassifier:
    """Classifies XFinBench tasks for quality and tool-solvability."""
    
    def __init__(self, model_name: str = "gemini-3-flash-preview"):
        """Initialize classifier with specified model.
        
        Args:
            model_name: Google AI model to use for classification
        """
        self.model_name = model_name
        self.client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
        
    def classify_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Classify a single task.
        
        Args:
            task_data: Task dictionary from XFinBench CSV
            
        Returns:
            Classification result dictionary
        """
        # Format choices section for MCQ
        choices_section = ""
        if task_data.get("task") == "mcq" and task_data.get("choice"):
            choices_section = f"Choices: {task_data['choice']}"
        
        prompt = XFINBENCH_CLASSIFICATION_PROMPT.format(
            task_id=task_data.get("id", ""),
            task_type=task_data.get("task", ""),
            question=task_data.get("question", ""),
            choices_section=choices_section,
            answer=task_data.get("ground_truth", ""),
            fin_capability=task_data.get("fin_capability", "")
        )
        
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=GenerateContentConfig(
                    temperature=0.1,
                    response_mime_type="application/json"
                )
            )
            
            classification = json.loads(response.text)
            return classification
            
        except Exception as e:
            log.error(f"Error classifying task {task_data.get('id', 'unknown')}: {e}")
            # Return default classification on error
            return {
                "tool_solvable": False,
                "tool_solvability_reasoning": f"Classification error: {e}",
                "requires_vision": False,
                "answer_type": "unknown",
                "ground_truth_quality": "unknown",
                "ground_truth_issues": f"Error during classification: {e}",
                "verification_feasibility": "hard",
                "problem_difficulty": "unknown",
                "tool_benefit": "none",
                "recommend_for_evaluation": False,
                "recommendation_reasoning": "Classification failed"
            }
    
    def filter_recommended_tasks(
        self,
        classified_tasks: List[Dict[str, Any]],
        require_tool_solvable: bool = True,
        exclude_vision: bool = True,
        min_tool_benefit: str = "low"
    ) -> List[Dict[str, Any]]:
        """Filter tasks based on classification criteria.
        
        Args:
            classified_tasks: Tasks with classification metadata
            require_tool_solvable: Only include tool-solvable tasks
            exclude_vision: Exclude tasks requiring vision
            min_tool_benefit: Minimum tool benefit level ("none", "low", "medium", "high")
            
        Returns:
            Filtered task list
        """
        benefit_order = {"none": 0, "low": 1, "medium": 2, "high": 3}
        min_benefit_level = benefit_order.get(min_tool_benefit, 1)
        
        filtered = []
        
        for task in classified_tasks:
            classification = task.get("classification", {})
            
            # Check filters
            if require_tool_solvable and not classification.get("tool_solvable"):
                continue
            if exclude_vision and classification.get("requires_vision"):
                continue
            
            tool_benefit = classification.get("tool_benefit", "none")
            if benefit_order.get(tool_benefit, 0) < min_benefit_level:
                continue
                
            if not classification.get("recommend_for_evaluation"):
                continue
            
            filtered.append(task)
        
        return filtered


def load_xfinbench_csv(csv_path: Path) -> List[Dict[str, Any]]:
    """Load XFinBench validation set from CSV.
    
    Args:
        csv_path: Path to validation_set.csv
        
    Returns:
        List of task dictionaries
    """
    tasks = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Skip rows with missing critical data
            if not row.get('question') or not row.get('id'):
                continue
            
            # Convert ground_truth to appropriate type
            ground_truth = row.get('ground_truth', '')
            if ground_truth and ground_truth != 'nan':
                # Try to convert to float for numerical tasks
                try:
                    ground_truth = float(ground_truth)
                    # Convert 0.0/1.0 to int for bool tasks
                    if row.get('task') == 'bool' and ground_truth in [0.0, 1.0]:
                        ground_truth = int(ground_truth)
                except ValueError:
                    pass  # Keep as string
            else:
                ground_truth = None
            
            tasks.append({
                "id": row.get('id', ''),
                "task": row.get('task', ''),
                "question": row.get('question', ''),
                "choice": row.get('choice', '') if row.get('choice') != 'nan' else None,
                "ground_truth": ground_truth,
                "figure": row.get('figure', '') if row.get('figure') != 'nan' else None,
                "fin_capability": row.get('fin_capability', ''),
                "gold_fin_term_id": row.get('gold_fin_term_id', '')
            })
    
    log.info(f"Loaded {len(tasks)} tasks from {csv_path}")
    return tasks


def stratified_classify_until_target(
    tasks: List[Dict[str, Any]],
    target_count: int,
    classifier: 'XFinBenchClassifier',
    seed: int = 42,
    min_tool_benefit: str = "low"
) -> List[Dict[str, Any]]:
    """Classify tasks iteratively until we have target_count recommended tasks.
    
    Uses stratified sampling across task types and capabilities.
    
    Args:
        tasks: All available tasks
        target_count: Number of high-quality tasks needed
        classifier: XFinBenchClassifier instance
        seed: Random seed for reproducibility
        min_tool_benefit: Minimum tool benefit level
        
    Returns:
        List of classified and recommended tasks
    """
    random.seed(seed)
    
    # Shuffle tasks for random sampling
    shuffled_tasks = tasks.copy()
    random.shuffle(shuffled_tasks)
    
    # Store recommended tasks
    recommended_tasks = []
    total_classified = 0
    
    log.info(f"Starting classification to find {target_count} high-quality tasks")
    log.info(f"Total tasks available: {len(shuffled_tasks)}")
    
    # Task type distribution for reference
    task_types = {}
    for task in tasks:
        task_type = task['task']
        task_types[task_type] = task_types.get(task_type, 0) + 1
    log.info(f"Task type distribution: {task_types}")
    
    # Classify tasks until we have enough recommended ones
    for i, task in enumerate(shuffled_tasks):
        if len(recommended_tasks) >= target_count:
            break
        
        total_classified += 1
        
        log.info(
            f"[{total_classified}] [{len(recommended_tasks)}/{target_count} found] "
            f"Classifying {task['task']} task: {task['id']}"
        )
        
        # Classify the task
        classification = classifier.classify_task(task)
        
        # Check if recommended
        is_recommended = classification.get("recommend_for_evaluation", False)
        is_tool_solvable = classification.get("tool_solvable", False)
        requires_vision = classification.get("requires_vision", False)
        tool_benefit = classification.get("tool_benefit", "none")
        
        log.info(
            f"  Result: Recommend={is_recommended}, Tool-solvable={is_tool_solvable}, "
            f"Vision={requires_vision}, Benefit={tool_benefit}"
        )
        
        # Add classification to task
        task_with_classification = task.copy()
        task_with_classification["classification"] = classification
        
        # If recommended, add to results
        if is_recommended and not requires_vision:
            recommended_tasks.append(task_with_classification)
            log.info(f"  ✓ Added to dataset ({len(recommended_tasks)}/{target_count})")
        else:
            reason = "requires vision" if requires_vision else "not recommended"
            log.info(f"  ✗ Skipped: {reason}")
    
    log.info(f"\n{'='*70}")
    log.info(f"Classification complete!")
    log.info(f"  Tasks classified: {total_classified}")
    log.info(f"  Tasks recommended: {len(recommended_tasks)}")
    if total_classified > 0:
        log.info(f"  Success rate: {len(recommended_tasks)/total_classified*100:.1f}%")
    log.info(f"{'='*70}\n")
    
    return recommended_tasks


def convert_to_experiment_format(
    classified_tasks: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Convert XFinBench tasks to experiment runner format.
    
    Args:
        classified_tasks: Tasks with classification metadata
        
    Returns:
        Tasks in format expected by run_experiments.py
    """
    experiment_tasks = []
    
    for task in classified_tasks:
        # Format for experiment runner
        formatted = {
            "task_id": task["id"],
            "problem": task["question"],
            "answer": task["ground_truth"],
            "capability_name": task["fin_capability"],
            "area_name": "finance",
            # XFinBench-specific metadata
            "task_type": task["task"],
            "choice": task.get("choice"),
            "figure": task.get("figure"),
            "gold_fin_term_id": task.get("gold_fin_term_id"),
            # Classification metadata
            "classification": task.get("classification", {})
        }
        
        # Add task type instructions to clarify expected format
        if task["task"] == "bool":
            formatted["problem"] = (
                f"{task['question']}\n\n"
                f"Answer with 1 for True or 0 for False."
            )
        elif task["task"] == "mcq" and task.get("choice"):
            formatted["problem"] = (
                f"{task['question']}\n\n"
                f"Choices: {task['choice']}\n\n"
                f"Answer with the letter of the correct choice (A, B, C, D, etc.)."
            )
        
        experiment_tasks.append(formatted)
    
    return experiment_tasks


def classify_all_tasks(
    tasks: List[Dict[str, Any]],
    classifier: 'XFinBenchClassifier',
    output_file: Path,
    save_interval: int = 10
) -> List[Dict[str, Any]]:
    """Classify all tasks in the dataset with progressive saving and resume capability.
    
    Args:
        tasks: All available tasks
        classifier: XFinBenchClassifier instance
        output_file: Path to save classified tasks (for progressive saving)
        save_interval: Save after every N classifications (default: 10)
        
    Returns:
        List of all tasks with classification metadata
    """
    # Check if we have existing classifications to resume from
    classified_tasks = []
    classified_ids = set()
    
    if output_file.exists():
        log.info(f"Found existing classification file: {output_file}")
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                classified_tasks = json.load(f)
            classified_ids = {t['id'] for t in classified_tasks}
            log.info(f"Loaded {len(classified_tasks)} already-classified tasks")
            log.info(f"Resuming classification from task {len(classified_tasks) + 1}")
        except Exception as e:
            log.warning(f"Could not load existing classifications: {e}")
            log.info("Starting fresh classification")
            classified_tasks = []
            classified_ids = set()
    
    total = len(tasks)
    remaining = [t for t in tasks if t['id'] not in classified_ids]
    
    log.info(f"Total tasks: {total}")
    log.info(f"Already classified: {len(classified_ids)}")
    log.info(f"Remaining to classify: {len(remaining)}")
    
    if not remaining:
        log.info("All tasks already classified!")
        return classified_tasks
    
    # Classify remaining tasks
    for i, task in enumerate(remaining, 1):
        overall_idx = len(classified_tasks) + 1
        log.info(f"[{overall_idx}/{total}] Classifying {task['task']} task: {task['id']}")
        
        # Classify the task
        classification = classifier.classify_task(task)
        
        # Check if recommended
        is_recommended = classification.get("recommend_for_evaluation", False)
        is_tool_solvable = classification.get("tool_solvable", False)
        requires_vision = classification.get("requires_vision", False)
        tool_benefit = classification.get("tool_benefit", "none")
        
        log.info(
            f"  Recommend={is_recommended}, Tool-solvable={is_tool_solvable}, "
            f"Vision={requires_vision}, Benefit={tool_benefit}"
        )
        
        # Add classification to task
        task_with_classification = task.copy()
        task_with_classification["classification"] = classification
        classified_tasks.append(task_with_classification)
        
        # Progressive save every N tasks
        if i % save_interval == 0 or i == len(remaining):
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(classified_tasks, f, indent=2, ensure_ascii=False)
            log.info(f"  💾 Progress saved: {overall_idx}/{total} tasks")
    
    log.info(f"\n{'='*70}")
    log.info(f"Classification complete!")
    log.info(f"  Total tasks classified: {total}")
    recommended_count = sum(1 for t in classified_tasks 
                           if t['classification'].get('recommend_for_evaluation', False))
    log.info(f"  Tasks recommended: {recommended_count}")
    if total > 0:
        log.info(f"  Success rate: {recommended_count/total*100:.1f}%")
    log.info(f"{'='*70}\n")
    
    return classified_tasks


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Classify XFinBench tasks for tool-assisted evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--csv-file",
        type=Path,
        default=Path("validation_set.csv"),
        help="Path to XFinBench validation CSV file"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("xfinbench_classified.json"),
        help="Output file for all classified tasks"
    )
    parser.add_argument(
        "--model",
        default="gemini-3-flash-preview",
        help="Model to use for classification"
    )
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=100,
        help="Number of filtered tasks to output (default: 100)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42)"
    )
    parser.add_argument(
        "--min-tool-benefit",
        choices=["low", "medium", "high"],
        default="low",
        help="Minimum tool benefit level for filtering (default: low)"
    )
    parser.add_argument(
        "--classify-all",
        action="store_true",
        help="Classify all 1000 tasks (default: True). Use --no-classify-all for iterative mode."
    )
    parser.add_argument(
        "--no-classify-all",
        action="store_true",
        help="Use iterative classification (stop after finding num-tasks)"
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=10,
        help="Save progress after every N classifications (default: 10)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Resume from existing classification file if present (default: True)"
    )
    
    args = parser.parse_args()
    
    # Load XFinBench tasks
    all_tasks = load_xfinbench_csv(args.csv_file)
    
    if not all_tasks:
        log.error("No tasks loaded")
        return
    
    classifier = XFinBenchClassifier(model_name=args.model)
    
    # Classify all tasks or use iterative mode
    if args.no_classify_all:
        # Iterative mode: stop after finding target
        log.info("Using iterative classification mode (stops after finding target)")
        classified_tasks = stratified_classify_until_target(
            tasks=all_tasks,
            target_count=args.num_tasks,
            classifier=classifier,
            seed=args.seed,
            min_tool_benefit=args.min_tool_benefit
        )
        recommended_tasks = classified_tasks
    else:
        # Default: classify everything
        log.info("Using full classification mode (classifies all tasks)")
        classified_tasks = classify_all_tasks(
            tasks=all_tasks,
            classifier=classifier,
            output_file=args.output,
            save_interval=args.save_interval
        )
        
        # Filter to get recommended tasks
        recommended_tasks = [
            t for t in classified_tasks 
            if t['classification'].get('recommend_for_evaluation', False)
            and not t['classification'].get('requires_vision', False)
        ]
        
        log.info(f"Filtered to {len(recommended_tasks)} recommended tasks)")
    
    if not classified_tasks:
        log.error("No tasks classified")
        return
    
    # Save all classified tasks (final save, may already be saved progressively)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(classified_tasks, f, indent=2, ensure_ascii=False)
    log.info(f"✓ Final save: {len(classified_tasks)} classified tasks to {args.output}")
    
    # Convert recommended tasks to experiment format and save
    # Limit to num_tasks if specified
    tasks_to_convert = recommended_tasks[:args.num_tasks] if args.num_tasks else recommended_tasks
    experiment_tasks = convert_to_experiment_format(tasks_to_convert)
    filtered_file = args.output.parent / f"{args.output.stem}_filtered.json"
    with open(filtered_file, 'w', encoding='utf-8') as f:
        json.dump(experiment_tasks, f, indent=2, ensure_ascii=False)
    log.info(f"✓ Saved {len(experiment_tasks)} experiment-ready tasks to {filtered_file}")
    
    # Print summary statistics
    print("\n" + "="*70)
    print("CLASSIFICATION SUMMARY")
    print("="*70)
    
    by_task_type = {}
    by_capability = {}
    by_tool_benefit = {"high": 0, "medium": 0, "low": 0, "none": 0}
    by_difficulty = {"easy": 0, "medium": 0, "hard": 0, "very_hard": 0}
    by_verification = {"direct": 0, "numeric_check": 0, "hard": 0}
    
    for task in recommended_tasks:
        task_type = task["task"]
        by_task_type[task_type] = by_task_type.get(task_type, 0) + 1
        
        capability = task["fin_capability"]
        by_capability[capability] = by_capability.get(capability, 0) + 1
        
        c = task.get("classification", {})
        
        benefit = c.get("tool_benefit", "none")
        by_tool_benefit[benefit] += 1
        
        difficulty = c.get("problem_difficulty", "unknown")
        by_difficulty[difficulty] = by_difficulty.get(difficulty, 0) + 1
        
        verification = c.get("verification_feasibility", "unknown")
        by_verification[verification] = by_verification.get(verification, 0) + 1
    
    total = len(recommended_tasks)
    
    print(f"\nTotal recommended tasks: {total}")
    
    print(f"\nBy task type:")
    for ttype, count in sorted(by_task_type.items(), key=lambda x: x[1], reverse=True):
        print(f"  {ttype}: {count} ({count/total*100:.1f}%)")
    
    print(f"\nBy financial capability:")
    for cap, count in sorted(by_capability.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {cap}: {count} ({count/total*100:.1f}%)")
    
    print(f"\nBy tool benefit:")
    for benefit, count in sorted(by_tool_benefit.items(), key=lambda x: x[1], reverse=True):
        if count > 0:
            print(f"  {benefit}: {count} ({count/total*100:.1f}%)")
    
    print(f"\nBy problem difficulty:")
    for diff, count in sorted(by_difficulty.items()):
        if count > 0:
            print(f"  {diff}: {count} ({count/total*100:.1f}%)")
    
    print(f"\nBy verification feasibility:")
    for verif, count in sorted(by_verification.items()):
        if count > 0:
            print(f"  {verif}: {count} ({count/total*100:.1f}%)")
    
    print("="*70)


if __name__ == "__main__":
    main()
