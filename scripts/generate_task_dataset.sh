#!/bin/bash
# Workflow for generating filtered task dataset with tool-assisted evaluation
# This script runs the complete pipeline from classification to filtered dataset generation

set -e  # Exit on error

echo "=========================================================================="
echo "Task Classification and Dataset Generation Workflow"
echo "=========================================================================="

# Default values
MATH_DIR="math"
NUM_TASKS=50
MODEL="gemini-3-flash-preview"
SEED=42
OUTPUT_DIR="filtered_tasks_output_100"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --math-dir)
            MATH_DIR="$2"
            shift 2
            ;;
        --num-tasks)
            NUM_TASKS="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --math-dir DIR       Directory with math tasks (default: math)"
            echo "  --num-tasks N        Number of tasks to generate (default: 50)"
            echo "  --model MODEL        Model for classification (default: gemini-3-flash-preview)"
            echo "  --seed SEED          Random seed (default: 42)"
            echo "  --output-dir DIR     Output directory (default: filtered_tasks_output)"
            echo "  --help               Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo ""
echo "Configuration:"
echo "  Math directory: $MATH_DIR"
echo "  Number of tasks: $NUM_TASKS"
echo "  Model: $MODEL"
echo "  Seed: $SEED"
echo "  Output directory: $OUTPUT_DIR"
echo ""

# Step 1: Classify tasks (stratified sampling until target found)
echo "=========================================================================="
echo "Step 1: Classifying tasks with stratified sampling"
echo "=========================================================================="
echo ""

python classify_tasks.py \
    --math-dir "$MATH_DIR" \
    --output "$OUTPUT_DIR/classified_tasks.json" \
    --model "$MODEL" \
    --num-tasks "$NUM_TASKS" \
    --seed "$SEED"

if [ $? -ne 0 ]; then
if [ $? -ne 0 ]; then
    echo "Error: Task classification failed"
    exit 1
fi

echo ""
echo "Classification complete!"
echo "Output: $OUTPUT_DIR/classified_tasks.json"
echo ""

# Step 2: Generate filtered dataset with metadata
echo "=========================================================================="
echo "Step 2: Generating filtered dataset with metadata"
echo "=========================================================================="
echo ""

python generate_filtered_dataset.py \
    --input "$OUTPUT_DIR/classified_tasks.json" \
    --output "$OUTPUT_DIR/filtered_dataset_${NUM_TASKS}.json" \
    --num-tasks "$NUM_TASKS" \
    --seed "$SEED"

if [ $? -ne 0 ]; then
    echo "Error: Dataset generation failed"
    exit 1
fi

echo ""
echo "Dataset generation complete!"
echo "Output: $OUTPUT_DIR/filtered_dataset_${NUM_TASKS}.json"
echo ""

# Step 3: Create experiment-ready format (compatible with run_experiments.py)
echo "=========================================================================="
echo "Step 3: Creating experiment-ready format"
echo "=========================================================================="
echo ""

python -c "
import json
from pathlib import Path

# Load filtered dataset
with open('$OUTPUT_DIR/filtered_dataset_${NUM_TASKS}.json', 'r') as f:
    dataset = json.load(f)

# Convert to experiment format (grouped by capability)
tasks_by_capability = {}
for task in dataset['tasks']:
    cap_name = task.get('capability_name', 'unknown')
    if cap_name not in tasks_by_capability:
        tasks_by_capability[cap_name] = []
    
    tasks_by_capability[cap_name].append({
        'task_id': task.get('task_id', task.get('dataset_metadata', {}).get('dataset_id', '')),
        'problem': task.get('problem', ''),
        'answer': task.get('answer', ''),
        'reasoning': task.get('reasoning', ''),
        'metadata': task.get('dataset_metadata', {})
    })

# Save in experiment format
output_file = Path('$OUTPUT_DIR/experiment_tasks_${NUM_TASKS}.json')
with open(output_file, 'w') as f:
    json.dump(tasks_by_capability, f, indent=2, ensure_ascii=False)

print(f'Created experiment-ready format: {output_file}')
print(f'Capabilities: {len(tasks_by_capability)}')
print(f'Total tasks: {sum(len(tasks) for tasks in tasks_by_capability.values())}')
"

if [ $? -ne 0 ]; then
    echo "Error: Format conversion failed"
    exit 1
fi

echo ""
echo "=========================================================================="
echo "Workflow complete!"
echo "=========================================================================="
echo ""
echo "Generated files:"
echo "  1. $OUTPUT_DIR/classified_tasks.json - Recommended tasks with classifications"
echo "  2. $OUTPUT_DIR/filtered_dataset_${NUM_TASKS}.json - Final dataset with metadata"
echo "  3. $OUTPUT_DIR/experiment_tasks_${NUM_TASKS}.json - Experiment-ready format"
echo ""
echo "Next steps:"
echo "  1. Review the filtered dataset and statistics"
echo "  2. Run experiments using: python run_experiments.py --num-tasks N"
echo "  3. Specify custom task file by copying experiment_tasks_${NUM_TASKS}.json to selected_tasks.json"
echo ""
