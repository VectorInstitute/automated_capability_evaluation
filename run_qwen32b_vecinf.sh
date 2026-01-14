#!/bin/bash
#SBATCH --job-name=qwen32b_eval
#SBATCH --output=qwen32b_eval_%j.out
#SBATCH --error=qwen32b_eval_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# Load necessary modules (adjust based on your cluster setup)
# module load python/3.9  # Uncomment if needed

# Navigate to project directory (adjust path as needed)
cd $HOME/automated_capability_evaluation || cd /path/to/automated_capability_evaluation

# Activate virtual environment if you have one
# source venv/bin/activate  # Uncomment if needed

echo "=========================================="
echo "Checking available Vector Inference models..."
echo "=========================================="
vec-inf list --json-mode | python3 -m json.tool 2>/dev/null || vec-inf list

echo ""
echo "=========================================="
echo "Starting Qwen 32B evaluation..."
echo "=========================================="

# Run the evaluation
# Vector Inference will be automatically used for Qwen 32B
# The code will check if the model name matches available models
python3 -m src.task_solve_models.run_single_agent \
  --batch-file evaluation_batch.json \
  --model Qwen2.5-32B-Instruct

echo ""
echo "=========================================="
echo "Evaluation completed!"
echo "=========================================="

