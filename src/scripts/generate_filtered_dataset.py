#!/usr/bin/env python3
"""
Generate Filtered Dataset with Metadata

Creates a curated dataset of N tasks randomly sampled from classified math tasks,
with comprehensive metadata for experimental analysis.
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)


def load_classified_tasks(filepath: Path) -> Dict[str, List[Dict[str, Any]]]:
    """Load classified tasks from JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def generate_filtered_dataset(
    classified_tasks: Dict[str, List[Dict[str, Any]]],
    num_tasks: int = 50,
    stratify_by_capability: bool = True,
    seed: int = 42
) -> List[Dict[str, Any]]:
    """Generate filtered dataset with stratified sampling.
    
    Args:
        classified_tasks: Dictionary of classified tasks by capability
        num_tasks: Total number of tasks to sample
        stratify_by_capability: Whether to sample proportionally from each capability
        seed: Random seed for reproducibility
        
    Returns:
        List of sampled tasks with full metadata
    """
    random.seed(seed)
    
    # Flatten tasks with capability metadata
    all_tasks = []
    for capability_name, task_list in classified_tasks.items():
        for task in task_list:
            task_copy = task.copy()
            task_copy["capability_name"] = capability_name
            all_tasks.append(task_copy)
    
    if not all_tasks:
        log.error("No tasks available for sampling")
        return []
    
    log.info(f"Total tasks available: {len(all_tasks)} from {len(classified_tasks)} capabilities")
    
    # Sample tasks
    if stratify_by_capability:
        # Calculate tasks per capability
        capabilities = list(classified_tasks.keys())
        tasks_per_cap = max(1, num_tasks // len(capabilities))
        remainder = num_tasks % len(capabilities)
        
        sampled = []
        capability_counts = defaultdict(int)
        
        for i, capability_name in enumerate(capabilities):
            # Allocate extra tasks to first capabilities to handle remainder
            cap_target = tasks_per_cap + (1 if i < remainder else 0)
            cap_tasks = classified_tasks[capability_name]
            
            if len(cap_tasks) < cap_target:
                log.warning(
                    f"Capability {capability_name} has only {len(cap_tasks)} tasks, "
                    f"requested {cap_target}"
                )
                cap_sample = cap_tasks
            else:
                cap_sample = random.sample(cap_tasks, cap_target)
            
            for task in cap_sample:
                task_copy = task.copy()
                task_copy["capability_name"] = capability_name
                sampled.append(task_copy)
                capability_counts[capability_name] += 1
        
        log.info("Stratified sampling complete:")
        for cap, count in sorted(capability_counts.items()):
            log.info(f"  {cap}: {count} tasks")
    else:
        # Random sampling without stratification
        if len(all_tasks) < num_tasks:
            log.warning(
                f"Requested {num_tasks} tasks but only {len(all_tasks)} available"
            )
            sampled = all_tasks
        else:
            sampled = random.sample(all_tasks, num_tasks)
        log.info(f"Random sampling: selected {len(sampled)} tasks")
    
    return sampled


def enrich_metadata(tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Add derived metadata fields to tasks.
    
    Args:
        tasks: List of tasks with classification
        
    Returns:
        Tasks with enriched metadata
    """
    enriched = []
    
    for i, task in enumerate(tasks):
        enriched_task = task.copy()
        classification = task.get("classification", {})
        
        # Add experimental metadata
        enriched_task["dataset_metadata"] = {
            "dataset_id": i + 1,
            "source_capability": task.get("capability_name", "unknown"),
            "original_task_id": task.get("task_id", "unknown"),
            
            # Classification-derived fields
            "tool_solvable": classification.get("tool_solvable", False),
            "answer_type": classification.get("answer_type", "unknown"),
            "ground_truth_quality": classification.get("ground_truth_quality", "unknown"),
            "format_clarity": classification.get("format_clarity", "unknown"),
            "evaluation_difficulty": classification.get("evaluation_difficulty", "unknown"),
            "problem_difficulty": classification.get("problem_difficulty", "unknown"),
            
            # For analysis
            "recommended": classification.get("recommend_for_evaluation", False),
            "has_reasoning": bool(task.get("reasoning")),
            "original_verdict": task.get("original_verdict", "unknown"),
        }
        
        enriched.append(enriched_task)
    
    return enriched


def analyze_dataset(tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate statistics about the dataset.
    
    Args:
        tasks: List of tasks with metadata
        
    Returns:
        Dictionary of dataset statistics
    """
    stats = {
        "total_tasks": len(tasks),
        "by_capability": defaultdict(int),
        "by_answer_type": defaultdict(int),
        "by_problem_difficulty": defaultdict(int),
        "by_evaluation_difficulty": defaultdict(int),
        "by_ground_truth_quality": defaultdict(int),
        "tool_solvable_count": 0,
        "recommended_count": 0,
    }
    
    for task in tasks:
        metadata = task.get("dataset_metadata", {})
        
        stats["by_capability"][metadata.get("source_capability", "unknown")] += 1
        stats["by_answer_type"][metadata.get("answer_type", "unknown")] += 1
        stats["by_problem_difficulty"][metadata.get("problem_difficulty", "unknown")] += 1
        stats["by_evaluation_difficulty"][metadata.get("evaluation_difficulty", "unknown")] += 1
        stats["by_ground_truth_quality"][metadata.get("ground_truth_quality", "unknown")] += 1
        
        if metadata.get("tool_solvable"):
            stats["tool_solvable_count"] += 1
        if metadata.get("recommended"):
            stats["recommended_count"] += 1
    
    # Convert defaultdicts to regular dicts for JSON serialization
    stats["by_capability"] = dict(stats["by_capability"])
    stats["by_answer_type"] = dict(stats["by_answer_type"])
    stats["by_problem_difficulty"] = dict(stats["by_problem_difficulty"])
    stats["by_evaluation_difficulty"] = dict(stats["by_evaluation_difficulty"])
    stats["by_ground_truth_quality"] = dict(stats["by_ground_truth_quality"])
    
    return stats


def save_dataset(
    tasks: List[Dict[str, Any]],
    output_file: Path,
    include_stats: bool = True
) -> None:
    """Save dataset to JSON file.
    
    Args:
        tasks: List of tasks to save
        output_file: Output file path
        include_stats: Whether to include statistics in output
    """
    output_data = {
        "metadata": {
            "dataset_name": "filtered_math_tasks",
            "version": "1.0",
            "description": "Curated math tasks for tool-assisted evaluation",
            "num_tasks": len(tasks),
        },
        "tasks": tasks,
    }
    
    if include_stats:
        output_data["statistics"] = analyze_dataset(tasks)
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    log.info(f"Saved dataset to {output_file}")


def print_dataset_summary(stats: Dict[str, Any]) -> None:
    """Print dataset statistics summary."""
    print("\n" + "="*70)
    print("DATASET SUMMARY")
    print("="*70)
    
    print(f"\nTotal tasks: {stats['total_tasks']}")
    print(f"Tool-solvable: {stats['tool_solvable_count']} ({stats['tool_solvable_count']/stats['total_tasks']*100:.1f}%)")
    print(f"Recommended: {stats['recommended_count']} ({stats['recommended_count']/stats['total_tasks']*100:.1f}%)")
    
    print("\nDistribution by capability:")
    for cap, count in sorted(stats['by_capability'].items(), key=lambda x: x[1], reverse=True):
        pct = count / stats['total_tasks'] * 100
        print(f"  {cap}: {count} ({pct:.1f}%)")
    
    print("\nDistribution by answer type:")
    for atype, count in sorted(stats['by_answer_type'].items()):
        pct = count / stats['total_tasks'] * 100
        print(f"  {atype}: {count} ({pct:.1f}%)")
    
    print("\nDistribution by problem difficulty:")
    for diff, count in sorted(stats['by_problem_difficulty'].items()):
        pct = count / stats['total_tasks'] * 100
        print(f"  {diff}: {count} ({pct:.1f}%)")
    
    print("\nDistribution by evaluation difficulty:")
    for diff, count in sorted(stats['by_evaluation_difficulty'].items()):
        pct = count / stats['total_tasks'] * 100
        print(f"  {diff}: {count} ({pct:.1f}%)")
    
    print("\nDistribution by ground truth quality:")
    for quality, count in sorted(stats['by_ground_truth_quality'].items()):
        pct = count / stats['total_tasks'] * 100
        print(f"  {quality}: {count} ({pct:.1f}%)")
    
    print("="*70)


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Generate filtered dataset with metadata",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 50-task dataset from filtered classifications
  python generate_filtered_dataset.py --input classified_tasks_filtered.json --output filtered_dataset_50.json
  
  # Generate 100-task dataset with random sampling (no stratification)
  python generate_filtered_dataset.py --input classified_tasks_filtered.json --num-tasks 100 --no-stratify
  
  # Use custom seed for reproducibility
  python generate_filtered_dataset.py --input classified_tasks_filtered.json --seed 123
        """
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input file with classified/filtered tasks"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("filtered_dataset_50.json"),
        help="Output file for generated dataset"
    )
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=50,
        help="Number of tasks to sample (default: 50)"
    )
    parser.add_argument(
        "--no-stratify",
        action="store_true",
        help="Disable stratified sampling by capability"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    args = parser.parse_args()
    
    # Load classified tasks
    log.info(f"Loading classified tasks from {args.input}")
    classified_tasks = load_classified_tasks(args.input)
    
    # Generate filtered dataset
    log.info(f"Generating dataset with {args.num_tasks} tasks...")
    sampled_tasks = generate_filtered_dataset(
        classified_tasks,
        num_tasks=args.num_tasks,
        stratify_by_capability=not args.no_stratify,
        seed=args.seed
    )
    
    # Enrich with metadata
    log.info("Enriching tasks with metadata...")
    enriched_tasks = enrich_metadata(sampled_tasks)
    
    # Generate statistics
    stats = analyze_dataset(enriched_tasks)
    
    # Save dataset
    save_dataset(enriched_tasks, args.output, include_stats=True)
    
    # Print summary
    print_dataset_summary(stats)
    
    log.info(f"\nDataset generation complete!")
    log.info(f"Output file: {args.output}")


if __name__ == "__main__":
    main()
