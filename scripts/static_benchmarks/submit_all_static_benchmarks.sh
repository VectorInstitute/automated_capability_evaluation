#!/bin/bash

set -euo pipefail

cd /fs01/projects/DeepLesion/projects/new_ace/automated_capability_evaluation

# Ensure scripts are executable
chmod +x scripts/static_benchmarks/*_eval.sh || true

sbatch scripts/static_benchmarks/math500_eval.sh
sbatch scripts/static_benchmarks/hardmath_eval.sh
sbatch scripts/static_benchmarks/wemath_eval.sh
sbatch scripts/static_benchmarks/stateval_eval.sh
sbatch scripts/static_benchmarks/orca_math_eval.sh
sbatch scripts/static_benchmarks/proofnet_eval.sh
sbatch scripts/static_benchmarks/harp_eval.sh
sbatch scripts/static_benchmarks/finance_math_eval.sh
sbatch scripts/static_benchmarks/finance_tasks_eval.sh
sbatch scripts/static_benchmarks/xfinbench_eval.sh
sbatch scripts/static_benchmarks/bizbench_eval.sh
sbatch scripts/static_benchmarks/omni_math_eval.sh
sbatch scripts/static_benchmarks/minif2f_eval.sh
