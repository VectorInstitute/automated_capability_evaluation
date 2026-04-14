#!/bin/bash

set -euo pipefail

cd /fs01/projects/DeepLesion/projects/new_ace/automated_capability_evaluation

# Ensure scripts are executable
chmod +x scripts/static_benchmarks/*_eval.sh || true

sbatch scripts/static_benchmarks/finance_math_eval.sh
sbatch scripts/static_benchmarks/finance_tasks_eval.sh
sbatch scripts/static_benchmarks/xfinbench_eval.sh
sbatch scripts/static_benchmarks/bizbench_eval.sh
