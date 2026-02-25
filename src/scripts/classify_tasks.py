#!/usr/bin/env python3
"""
Task Classifier for Filtering Tool-Appropriate Math Tasks

Classifies math tasks to identify:
1. Tool-solvable vs explanation-focused
2. Quantitative vs qualitative answers
3. Ground truth quality issues
4. Format specification problems
"""

import argparse
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


CLASSIFICATION_PROMPT = """You are an expert at analyzing mathematical problems to determine their suitability for computational tool-assisted solving and evaluation.

Analyze the following math task and classify it across multiple dimensions:

TASK:
Problem: {problem}
Expected Answer: {answer}
{reasoning_section}
{verification_section}

CLASSIFICATION DIMENSIONS:

1. **Tool Solvability** (tool_solvable: true/false)
   - Can this problem be meaningfully solved or verified using computational tools (Python with NumPy, SciPy, SymPy)?
   - Problems requiring pure conceptual explanations, proofs, or theoretical discussions are NOT tool-solvable
   - Problems with numerical computation, symbolic manipulation, or verification ARE tool-solvable

2. **Answer Type** (answer_type: "quantitative", "qualitative", "mixed", "no_answer")
   - "quantitative": Numerical answer(s), formulas, or specific mathematical objects
   - "qualitative": Explanations, descriptions, procedures, conceptual answers
   - "mixed": Both quantitative and qualitative components
   - "no_answer": NO_ANSWER or similar (task asks for explanation/implementation without specific answer)

3. **Ground Truth Quality** (ground_truth_quality: "good", "problematic", "missing")
   - "good": Clear, well-specified expected answer that matches the problem requirements
   - "problematic": Answer doesn't match problem requirements, contradictory, or under-specified format
   - "missing": NO_ANSWER when a specific answer could be provided, or truly missing
   - IMPORTANT: If verification verdict is "no", this is a strong indicator of problematic quality

4. **Format Specification** (format_clarity: "clear", "ambiguous", "unspecified")
   - "clear": Expected answer format is well-defined and unambiguous
   - "ambiguous": Multiple valid formats possible, unclear what format is expected
   - "unspecified": No clear indication of expected format

5. **Evaluation Difficulty** (evaluation_difficulty: "easy", "medium", "hard")
   - "easy": Simple numerical comparison or exact string match
   - "medium": Requires normalization or format handling
   - "hard": Requires understanding context, multi-part answers, or subjective judgment

6. **Problem Difficulty** (problem_difficulty: "easy", "medium", "hard", "very_hard")
   - Based on mathematical complexity, number of steps, required knowledge level

7. **Recommendation** (recommend_for_evaluation: true/false)
   - Should this task be included in a high-quality tool-assisted evaluation dataset?
   - Consider: tool-solvability, answer quality, format clarity, and evaluation feasibility
   - CRITICAL: Tasks with verification verdict "no" should typically NOT be recommended unless the issue is minor
   - Pay attention to verification reasons - quality issues like unclear problems, wrong answers, or conceptual mismatches should disqualify tasks

IMPORTANT: Return ONLY a valid JSON object with this exact structure:
{{
    "tool_solvable": true/false,
    "tool_solvability_reasoning": "Brief explanation",
    "answer_type": "quantitative/qualitative/mixed/no_answer",
    "ground_truth_quality": "good/problematic/missing",
    "ground_truth_issues": "Specific issues if any, or null",
    "format_clarity": "clear/ambiguous/unspecified",
    "evaluation_difficulty": "easy/medium/hard",
    "problem_difficulty": "easy/medium/hard/very_hard",
    "recommend_for_evaluation": true/false,
    "recommendation_reasoning": "Why include or exclude this task"
}}

Respond with ONLY the JSON object, no additional text or formatting."""


class TaskClassifier:
    """Classifies math tasks for quality and tool-solvability."""
    
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
            task_data: Task dictionary with 'problem', 'answer', optionally 'reasoning' and 'verification'
            
        Returns:
            Classification result dictionary
        """
        # Format reasoning section if available
        reasoning_section = ""
        if task_data.get("reasoning"):
            reasoning_section = f"Reasoning/Solution Provided: {task_data['reasoning']}"
        
        # Format verification section if available
        verification_section = ""
        verification = task_data.get("verification", {})
        if verification:
            verdict = verification.get("verdict", "unknown")
            reason = verification.get("reason", "No reason provided")
            verification_section = f"\nVERIFICATION STATUS:\nVerdict: {verdict}\nReason: {reason}"
        
        prompt = CLASSIFICATION_PROMPT.format(
            problem=task_data.get("problem", ""),
            answer=task_data.get("answer", ""),
            reasoning_section=reasoning_section,
            verification_section=verification_section
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
            log.error(f"Error classifying task {task_data.get('task_id', 'unknown')}: {e}")
            # Return default classification on error
            return {
                "tool_solvable": False,
                "tool_solvability_reasoning": f"Classification error: {e}",
                "answer_type": "unknown",
                "ground_truth_quality": "unknown",
                "ground_truth_issues": f"Error during classification: {e}",
                "format_clarity": "unknown",
                "evaluation_difficulty": "hard",
                "problem_difficulty": "unknown",
                "recommend_for_evaluation": False,
                "recommendation_reasoning": "Classification failed"
            }
    
    def classify_dataset(
        self,
        tasks: Dict[str, List[Dict[str, Any]]],
        output_file: Optional[Path] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Classify all tasks in a dataset.
        
        Args:
            tasks: Dictionary mapping capability names to list of tasks
            output_file: Optional path to save classifications
            
        Returns:
            Tasks with added classification metadata
        """
        classified_tasks = {}
        total_tasks = sum(len(task_list) for task_list in tasks.values())
        processed = 0
        
        log.info(f"Classifying {total_tasks} tasks from {len(tasks)} capabilities...")
        
        for capability_name, task_list in tasks.items():
            log.info(f"Processing capability: {capability_name} ({len(task_list)} tasks)")
            classified_tasks[capability_name] = []
            
            for task in task_list:
                processed += 1
                log.info(f"  [{processed}/{total_tasks}] Classifying task {task.get('task_id', '?')}")
                
                classification = self.classify_task(task)
                
                # Add classification to task
                task_with_classification = task.copy()
                task_with_classification["classification"] = classification
                classified_tasks[capability_name].append(task_with_classification)
                
                # Log key insights
                log.info(
                    f"    Tool-solvable: {classification.get('tool_solvable')}, "
                    f"Answer type: {classification.get('answer_type')}, "
                    f"Recommend: {classification.get('recommend_for_evaluation')}"
                )
        
        # Save if output file specified
        if output_file:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(classified_tasks, f, indent=2, ensure_ascii=False)
            log.info(f"Saved classified tasks to {output_file}")
        
        return classified_tasks
    
    def filter_recommended_tasks(
        self,
        classified_tasks: Dict[str, List[Dict[str, Any]]],
        require_tool_solvable: bool = True,
        exclude_no_answer: bool = True,
        exclude_problematic: bool = True
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Filter tasks based on classification criteria.
        
        Args:
            classified_tasks: Tasks with classification metadata
            require_tool_solvable: Only include tool-solvable tasks
            exclude_no_answer: Exclude tasks with answer_type="no_answer"
            exclude_problematic: Exclude tasks with problematic ground truth
            
        Returns:
            Filtered task dictionary
        """
        filtered = {}
        total_before = 0
        total_after = 0
        
        for capability_name, task_list in classified_tasks.items():
            total_before += len(task_list)
            filtered_list = []
            
            for task in task_list:
                classification = task.get("classification", {})
                
                # Check filters
                if require_tool_solvable and not classification.get("tool_solvable"):
                    continue
                if exclude_no_answer and classification.get("answer_type") == "no_answer":
                    continue
                if exclude_problematic and classification.get("ground_truth_quality") == "problematic":
                    continue
                if not classification.get("recommend_for_evaluation"):
                    continue
                
                filtered_list.append(task)
            
            if filtered_list:
                filtered[capability_name] = filtered_list
                total_after += len(filtered_list)
        
        log.info(f"\nFiltering results:")
        log.info(f"  Before: {total_before} tasks across {len(classified_tasks)} capabilities")
        log.info(f"  After: {total_after} tasks across {len(filtered)} capabilities")
        log.info(f"  Removed: {total_before - total_after} tasks")
        
        return filtered


def load_math_tasks(math_dir: Path = Path("math")) -> Dict[str, List[Dict[str, Any]]]:
    """Load all math tasks from capability directories.
    
    Args:
        math_dir: Base directory containing math capability folders
        
    Returns:
        Dictionary mapping capability names to list of tasks
    """
    all_tasks = {}
    
    if not math_dir.exists():
        log.error(f"Math directory not found: {math_dir}")
        return all_tasks
    
    capability_dirs = sorted([d for d in math_dir.iterdir() if d.is_dir()])
    log.info(f"Found {len(capability_dirs)} capability directories")
    
    for cap_dir in capability_dirs:
        capability_name = cap_dir.name
        cap_file = cap_dir / "capability.json"
        
        if not cap_file.exists():
            continue
        
        try:
            with open(cap_file, 'r', encoding='utf-8') as f:
                cap_data = json.load(f)
            
            # Extract tasks from capability_failed_data (previously failed tasks)
            failed_tasks = cap_data.get("capability_failed_data", [])
            
            if failed_tasks:
                tasks = []
                for task_data in failed_tasks:
                    verification = task_data.get("verification", {})
                    verdict = verification.get("verdict", "")
                    
                    # Include all tasks (not just verdict='no') for comprehensive classification
                    tasks.append({
                        "task_id": task_data.get("id", ""),
                        "problem": task_data.get("problem", ""),
                        "answer": task_data.get("answer", ""),
                        "reasoning": task_data.get("reasoning", ""),
                        "verification": verification,
                        "original_verdict": verdict
                    })
                
                if tasks:
                    all_tasks[capability_name] = tasks
                    log.info(f"Loaded {len(tasks)} tasks from {capability_name}")
        
        except Exception as e:
            log.error(f"Error loading {capability_name}: {e}")
    
    return all_tasks


def stratified_classify_until_target(
    tasks_by_capability: Dict[str, List[Dict[str, Any]]],
    target_count: int,
    classifier: 'TaskClassifier',
    seed: int = 42
) -> Dict[str, List[Dict[str, Any]]]:
    """Classify tasks iteratively until we have target_count recommended tasks.
    
    Uses stratified sampling across capabilities, classifying one task at a time
    from each capability in round-robin fashion until we have enough good tasks.
    
    Args:
        tasks_by_capability: All available tasks grouped by capability
        target_count: Number of high-quality tasks needed
        classifier: TaskClassifier instance
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary of classified tasks that were recommended
    """
    random.seed(seed)
    
    # Shuffle tasks within each capability for random sampling
    shuffled_tasks = {}
    for cap_name, task_list in tasks_by_capability.items():
        shuffled = task_list.copy()
        random.shuffle(shuffled)
        shuffled_tasks[cap_name] = shuffled
    
    # Track which index we're at for each capability
    capability_indices = {cap: 0 for cap in shuffled_tasks.keys()}
    
    # Store recommended tasks
    recommended_tasks = {}
    total_recommended = 0
    total_classified = 0
    
    capabilities = list(shuffled_tasks.keys())
    if not capabilities:
        log.error("No capabilities available")
        return {}
    
    log.info(f"Starting stratified classification to find {target_count} high-quality tasks")
    log.info(f"Available capabilities: {len(capabilities)}")
    log.info(f"Total tasks available: {sum(len(tasks) for tasks in shuffled_tasks.values())}")
    
    # Round-robin through capabilities until we have enough recommended tasks
    capability_idx = 0
    stuck_counter = 0
    max_stuck_iterations = len(capabilities) * 2  # Safety check
    
    while total_recommended < target_count:
        # Check if we've exhausted all capabilities
        if stuck_counter >= max_stuck_iterations:
            log.warning(
                f"Could not find enough tasks. Found {total_recommended}/{target_count} after "
                f"classifying {total_classified} tasks. All capabilities may be exhausted."
            )
            break
        
        # Get current capability
        current_cap = capabilities[capability_idx]
        cap_tasks = shuffled_tasks[current_cap]
        cap_idx = capability_indices[current_cap]
        
        # Check if this capability is exhausted
        if cap_idx >= len(cap_tasks):
            # Move to next capability
            capability_idx = (capability_idx + 1) % len(capabilities)
            stuck_counter += 1
            continue
        
        # Get next task from this capability
        task = cap_tasks[cap_idx]
        capability_indices[current_cap] += 1
        total_classified += 1
        
        log.info(
            f"[{total_classified}] [{total_recommended}/{target_count} found] "
            f"Classifying from {current_cap}: task {task.get('task_id', '?')}"
        )
        
        # Classify the task
        classification = classifier.classify_task(task)
        
        # Check if recommended
        is_recommended = classification.get("recommend_for_evaluation", False)
        is_tool_solvable = classification.get("tool_solvable", False)
        answer_type = classification.get("answer_type", "unknown")
        
        log.info(
            f"  Result: Recommend={is_recommended}, Tool-solvable={is_tool_solvable}, "
            f"Answer type={answer_type}"
        )
        
        # Add classification to task
        task_with_classification = task.copy()
        task_with_classification["classification"] = classification
        task_with_classification["capability_name"] = current_cap
        
        # If recommended, add to results
        if is_recommended:
            if current_cap not in recommended_tasks:
                recommended_tasks[current_cap] = []
            recommended_tasks[current_cap].append(task_with_classification)
            total_recommended += 1
            stuck_counter = 0  # Reset stuck counter on success
            log.info(f"  ✓ Added to dataset ({total_recommended}/{target_count})")
        else:
            log.info(f"  ✗ Not recommended, continuing search")
        
        # Move to next capability for round-robin
        capability_idx = (capability_idx + 1) % len(capabilities)
    
    log.info(f"\n{'='*70}")
    log.info(f"Classification complete!")
    log.info(f"  Tasks classified: {total_classified}")
    log.info(f"  Tasks recommended: {total_recommended}")
    log.info(f"  Success rate: {total_recommended/total_classified*100:.1f}%")
    log.info(f"  Capabilities represented: {len(recommended_tasks)}")
    log.info(f"{'='*70}\n")
    
    return recommended_tasks


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Classify math tasks for tool-assisted evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--math-dir",
        type=Path,
        default=Path("math"),
        help="Directory containing math capability folders"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("classified_tasks.json"),
        help="Output file for classified tasks"
    )
    parser.add_argument(
        "--model",
        default="gemini-3-flash-preview",
        help="Model to use for classification"
    )
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=50,
        help="Number of high-quality tasks to find (default: 50)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42)"
    )
    parser.add_argument(
        "--filter-only",
        action="store_true",
        help="Only filter pre-classified tasks (skip classification step)"
    )
    parser.add_argument(
        "--input-classified",
        type=Path,
        help="Input file with pre-classified tasks (for --filter-only mode)"
    )
    
    args = parser.parse_args()
    
    if args.filter_only:
        if not args.input_classified or not args.input_classified.exists():
            log.error("Must provide --input-classified file in --filter-only mode")
            return
        
        with open(args.input_classified, 'r', encoding='utf-8') as f:
            classified_tasks = json.load(f)
        log.info(f"Loaded pre-classified tasks from {args.input_classified}")
        
        # Filter the pre-classified tasks
        classifier = TaskClassifier(model_name=args.model)
        filtered = classifier.filter_recommended_tasks(classified_tasks)
    else:
        # Load all available tasks
        all_tasks = load_math_tasks(args.math_dir)
        
        if not all_tasks:
            log.error("No tasks loaded")
            return
        
        total_available = sum(len(tasks) for tasks in all_tasks.values())
        log.info(f"Loaded {total_available} tasks from {len(all_tasks)} capabilities")
        
        # Use stratified sampling with iterative classification
        classifier = TaskClassifier(model_name=args.model)
        filtered = stratified_classify_until_target(
            tasks_by_capability=all_tasks,
            target_count=args.num_tasks,
            classifier=classifier,
            seed=args.seed
        )
        
        if not filtered:
            log.error("No recommended tasks found")
            return
    
    # Save filtered/recommended tasks
    filtered_file = args.output.parent / f"{args.output.stem}_filtered.json"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(filtered_file, 'w', encoding='utf-8') as f:
        json.dump(filtered, f, indent=2, ensure_ascii=False)
    log.info(f"Saved recommended tasks to {filtered_file}")
    
    # Print summary statistics
    print("\n" + "="*70)
    print("CLASSIFICATION SUMMARY")
    print("="*70)
    
    tool_solvable = 0
    not_tool_solvable = 0
    by_answer_type = {"quantitative": 0, "qualitative": 0, "mixed": 0, "no_answer": 0}
    by_quality = {"good": 0, "problematic": 0, "missing": 0}
    by_difficulty = {"easy": 0, "medium": 0, "hard": 0, "very_hard": 0}
    by_capability = {}
    recommended = 0
    
    for cap_name, task_list in filtered.items():
        by_capability[cap_name] = len(task_list)
        for task in task_list:
            c = task.get("classification", {})
            if c.get("tool_solvable"):
                tool_solvable += 1
            else:
                not_tool_solvable += 1
            
            answer_type = c.get("answer_type", "unknown")
            by_answer_type[answer_type] = by_answer_type.get(answer_type, 0) + 1
            
            quality = c.get("ground_truth_quality", "unknown")
            by_quality[quality] = by_quality.get(quality, 0) + 1
            
            difficulty = c.get("problem_difficulty", "unknown")
            by_difficulty[difficulty] = by_difficulty.get(difficulty, 0) + 1
            
            if c.get("recommend_for_evaluation"):
                recommended += 1
    
    total = tool_solvable + not_tool_solvable
    if total == 0:
        print("No tasks found")
        return
        
    print(f"\nTotal tasks: {total}")
    print(f"Tool-solvable: {tool_solvable} ({tool_solvable/total*100:.1f}%)")
    print(f"Not tool-solvable: {not_tool_solvable} ({not_tool_solvable/total*100:.1f}%)")
    
    print(f"\nBy capability:")
    for cap, count in sorted(by_capability.items(), key=lambda x: x[1], reverse=True):
        print(f"  {cap}: {count}")
    
    print(f"\nAnswer types:")
    for atype, count in sorted(by_answer_type.items()):
        if count > 0:
            print(f"  {atype}: {count} ({count/total*100:.1f}%)")
    
    print(f"\nGround truth quality:")
    for quality, count in sorted(by_quality.items()):
        if count > 0:
            print(f"  {quality}: {count} ({count/total*100:.1f}%)")
    
    print(f"\nProblem difficulty:")
    for diff, count in sorted(by_difficulty.items()):
        if count > 0:
            print(f"  {diff}: {count} ({count/total*100:.1f}%)")
    
    print(f"\nRecommended for evaluation: {recommended} ({recommended/total*100:.1f}%)")
    print("="*70)


if __name__ == "__main__":
    main()
