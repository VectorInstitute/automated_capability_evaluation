#!/usr/bin/env python3
"""
Unified Task Management Script

Consolidates task discovery, selection, and extraction into one tool.

Usage:
    # List all failed tasks
    python manage_tasks.py list
    
    # Extract tasks using default selections
    python manage_tasks.py extract
    
    # Extract specific tasks
    python manage_tasks.py extract --tasks computational_linear_algebra:2,3,4 numerical_methods_ode:2,4,5
    
    # Extract N random failed tasks per capability
    python manage_tasks.py extract --random 3
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict


# Default task selections (can be overridden via CLI)
DEFAULT_TASK_SELECTIONS = {
    'computational_linear_algebra': [2, 3, 4],
    'numerical_simulation_dynamical_systems': [1, 2, 3],
    'numerical_methods_ode': [2, 4, 5],
    'complex_analysis_problems': [3, 4, 5],
    'advanced_integration_techniques': [2, 6, 9],
}


class TaskManager:
    """Manages task discovery, selection, and extraction."""
    
    def __init__(self, math_dir: Path = Path('math')):
        self.math_dir = math_dir
        
    def load_capability_file(self, capability_name: str) -> Dict[str, Any]:
        """Load a capability JSON file."""
        cap_dir = self.math_dir / capability_name
        
        if not cap_dir.exists():
            raise FileNotFoundError(f"Capability directory not found: {cap_dir}")
        
        cap_file = cap_dir / 'capability.json'
        if not cap_file.exists():
            raise FileNotFoundError(f"Capability file not found: {cap_file}")
        
        with open(cap_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def find_failed_tasks(self, capability_name: str) -> List[Dict[str, Any]]:
        """Find all tasks with verdict='no' in a capability."""
        try:
            capability_data = self.load_capability_file(capability_name)
        except FileNotFoundError:
            return []
        
        failed_tasks = []
        failed_data = capability_data.get('capability_failed_data', [])
        
        for task_data in failed_data:
            verification = task_data.get('verification', {})
            verdict = verification.get('verdict', '')
            
            if verdict == 'no':
                failed_tasks.append({
                    'task_id': task_data.get('id', ''),
                    'problem': task_data.get('problem', ''),
                    'answer': task_data.get('answer', ''),
                    'reasoning': task_data.get('reasoning', ''),
                    'verification': verification,
                })
        
        return failed_tasks
    
    def list_all_failed_tasks(self) -> Dict[str, List[Dict[str, Any]]]:
        """Find all failed tasks across all capabilities."""
        all_failed = {}
        
        if not self.math_dir.exists():
            print(f"Error: Math directory not found: {self.math_dir}")
            return all_failed
        
        capability_dirs = sorted([d for d in self.math_dir.iterdir() if d.is_dir()])
        
        for cap_dir in capability_dirs:
            capability_name = cap_dir.name
            failed_tasks = self.find_failed_tasks(capability_name)
            
            if failed_tasks:
                all_failed[capability_name] = failed_tasks
        
        return all_failed
    
    def extract_specific_tasks(
        self, 
        task_selections: Dict[str, List[int]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Extract specific tasks by ID."""
        extracted_tasks = {}
        
        for capability_name, task_ids in task_selections.items():
            try:
                capability_data = self.load_capability_file(capability_name)
                failed_data = capability_data.get('capability_failed_data', [])
                
                tasks = []
                for task_id in task_ids:
                    # Find the task with matching ID
                    for task in failed_data:
                        if str(task.get('id', '')) == str(task_id):
                            tasks.append({
                                'task_id': task.get('id', ''),
                                'problem': task.get('problem', ''),
                                'answer': task.get('answer', ''),
                                'reasoning': task.get('reasoning', ''),
                                'verification': task.get('verification', {}),
                            })
                            break
                
                if tasks:
                    extracted_tasks[capability_name] = tasks
                    
            except FileNotFoundError as e:
                print(f"Warning: {e}")
                continue
        
        return extracted_tasks
    
    def extract_random_tasks(self, num_per_capability: int) -> Dict[str, List[Dict[str, Any]]]:
        """Extract N random failed tasks per capability."""
        all_failed = self.list_all_failed_tasks()
        extracted_tasks = {}
        
        for capability_name, failed_tasks in all_failed.items():
            if len(failed_tasks) >= num_per_capability:
                selected = random.sample(failed_tasks, num_per_capability)
                extracted_tasks[capability_name] = selected
            elif failed_tasks:
                print(f"Warning: {capability_name} has only {len(failed_tasks)} failed tasks (requested {num_per_capability})")
                extracted_tasks[capability_name] = failed_tasks
        
        return extracted_tasks
    
    def save_tasks(self, tasks: Dict[str, List[Dict[str, Any]]], output_file: Path):
        """Save extracted tasks to JSON file."""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(tasks, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Saved {sum(len(t) for t in tasks.values())} tasks to: {output_file}")


def print_list_summary(all_failed: Dict[str, List[Dict[str, Any]]]):
    """Print summary of all failed tasks."""
    print("\n" + "="*80)
    print("FAILED TASKS SUMMARY")
    print("="*80)
    
    total_capabilities = len(all_failed)
    total_tasks = sum(len(tasks) for tasks in all_failed.values())
    
    print(f"\nTotal capabilities with failed tasks: {total_capabilities}")
    print(f"Total failed tasks: {total_tasks}\n")
    
    print(f"{'Capability':<50} {'Failed Tasks':<15} {'Task IDs'}")
    print("-"*80)
    
    for cap_name, tasks in sorted(all_failed.items()):
        task_ids = [str(t['task_id']) for t in tasks]
        task_ids_str = ', '.join(task_ids[:10])
        if len(task_ids) > 10:
            task_ids_str += f", ... ({len(task_ids) - 10} more)"
        
        print(f"{cap_name:<50} {len(tasks):<15} {task_ids_str}")
    
    print("="*80)


def print_extract_summary(tasks: Dict[str, List[Dict[str, Any]]]):
    """Print summary of extracted tasks."""
    print("\n" + "="*80)
    print("EXTRACTION SUMMARY")
    print("="*80)
    
    total_tasks = sum(len(task_list) for task_list in tasks.values())
    
    print(f"\nTotal tasks extracted: {total_tasks}")
    print(f"Total capabilities: {len(tasks)}\n")
    
    print(f"{'Capability':<50} {'Tasks':<10} {'Task IDs'}")
    print("-"*80)
    
    for cap_name, task_list in sorted(tasks.items()):
        task_ids = [str(t['task_id']) for t in task_list]
        task_ids_str = ', '.join(task_ids)
        print(f"{cap_name:<50} {len(task_list):<10} {task_ids_str}")
    
    # Verify all are failures
    print("\n" + "="*80)
    print("VERIFICATION CHECK")
    print("="*80)
    
    all_failed = True
    for cap_name, task_list in tasks.items():
        for task in task_list:
            verdict = task['verification'].get('verdict', '')
            if verdict != 'no':
                print(f"⚠️  {cap_name} Task {task['task_id']}: verdict='{verdict}' (NOT A FAILURE)")
                all_failed = False
    
    if all_failed:
        print("✓ All extracted tasks have verdict='no' (confirmed failures)")
    else:
        print("✗ Some tasks are not failures - review selections")
    
    print("="*80)


def parse_task_selections(selections: List[str]) -> Dict[str, List[int]]:
    """Parse task selections from CLI format.
    
    Example: ["cap1:1,2,3", "cap2:4,5,6"] -> {"cap1": [1,2,3], "cap2": [4,5,6]}
    """
    parsed = {}
    for selection in selections:
        if ':' not in selection:
            print(f"Warning: Invalid format '{selection}'. Expected 'capability:id1,id2,id3'")
            continue
        
        cap_name, ids_str = selection.split(':', 1)
        task_ids = [int(id.strip()) for id in ids_str.split(',')]
        parsed[cap_name] = task_ids
    
    return parsed


def cmd_list(args):
    """List all failed tasks."""
    manager = TaskManager()
    all_failed = manager.list_all_failed_tasks()
    
    if not all_failed:
        print("No failed tasks found.")
        return
    
    print_list_summary(all_failed)
    
    if args.verbose:
        print("\n" + "="*80)
        print("DETAILED TASK INFORMATION")
        print("="*80)
        
        for cap_name, tasks in sorted(all_failed.items()):
            print(f"\n{'='*80}")
            print(f"Capability: {cap_name}")
            print(f"{'='*80}")
            
            for task in tasks:
                print(f"\nTask ID: {task['task_id']}")
                print(f"Problem: {task['problem'][:200]}...")
                print(f"Answer: {task['answer'][:100]}...")
                print(f"Verdict: {task['verification'].get('verdict', 'unknown')}")
                print("-"*80)


def cmd_extract(args):
    """Extract tasks based on selections."""
    manager = TaskManager()
    
    if args.random:
        print(f"Extracting {args.random} random failed tasks per capability...")
        random.seed(args.seed)
        extracted_tasks = manager.extract_random_tasks(args.random)
    elif args.tasks:
        print("Extracting specified tasks...")
        task_selections = parse_task_selections(args.tasks)
        extracted_tasks = manager.extract_specific_tasks(task_selections)
    else:
        print("Extracting default task selections...")
        extracted_tasks = manager.extract_specific_tasks(DEFAULT_TASK_SELECTIONS)
    
    if not extracted_tasks:
        print("No tasks extracted.")
        return
    
    # Save tasks
    output_file = Path(args.output)
    manager.save_tasks(extracted_tasks, output_file)
    
    # Print summary
    print_extract_summary(extracted_tasks)
    
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print(f"1. Review {args.output}")
    print("2. Run experiments: python run_experiments.py")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Unified task management for capability evaluation experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all failed tasks
  python manage_tasks.py list
  
  # List with detailed information
  python manage_tasks.py list --verbose
  
  # Extract using default selections
  python manage_tasks.py extract
  
  # Extract specific tasks
  python manage_tasks.py extract --tasks computational_linear_algebra:2,3,4 numerical_methods_ode:2,4,5
  
  # Extract 3 random failed tasks per capability
  python manage_tasks.py extract --random 3
  
  # Extract to custom output file
  python manage_tasks.py extract --output my_tasks.json
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List all failed tasks')
    list_parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show detailed task information'
    )
    
    # Extract command
    extract_parser = subparsers.add_parser('extract', help='Extract tasks to JSON file')
    extract_parser.add_argument(
        '--output', '-o',
        default='selected_tasks.json',
        help='Output file path (default: selected_tasks.json)'
    )
    extract_group = extract_parser.add_mutually_exclusive_group()
    extract_group.add_argument(
        '--tasks', '-t',
        nargs='+',
        help='Specific tasks to extract (format: capability:id1,id2,id3)'
    )
    extract_group.add_argument(
        '--random', '-r',
        type=int,
        metavar='N',
        help='Extract N random failed tasks per capability'
    )
    extract_parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for --random option (default: 42)'
    )
    
    args = parser.parse_args()
    
    if args.command == 'list':
        cmd_list(args)
    elif args.command == 'extract':
        cmd_extract(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
