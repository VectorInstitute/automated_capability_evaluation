#!/bin/bash
#SBATCH --job-name=gemma_xfinbench_test_local_array
#SBATCH --output=/projects/DeepLesion/projects/new_ace/automated_capability_evaluation/logs/xfinbench_test_local_array_%A_%a.out
#SBATCH --error=/projects/DeepLesion/projects/new_ace/automated_capability_evaluation/logs/xfinbench_test_local_array_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --array=0-7%8

set -euo pipefail

cd /projects/DeepLesion/projects/new_ace/automated_capability_evaluation

# shellcheck disable=SC1091
source /projects/DeepLesion/py311_env/bin/activate

# shellcheck disable=SC1091
source "scripts/static_benchmarks/env_slurm_inspect.sh"

# Allow direct execution without sbatch by defaulting to shard 0.
: "${SLURM_ARRAY_TASK_ID:=0}"

NUM_SHARDS=8

# Count only text-only XFinBench test rows since the adapter skips rows with figures.
TOTAL=$(
python - <<'PY'
from datasets import load_dataset

ds = load_dataset(
    "Zhihan/XFinBench",
    data_files={"validation": "validation_set.csv", "test": "test_set.csv"},
)["test"]

print(sum(1 for row in ds if row.get("figure") is None))
PY
)

CHUNK=$(((TOTAL + NUM_SHARDS - 1) / NUM_SHARDS))
OFFSET=$((SLURM_ARRAY_TASK_ID * CHUNK))
TAG="_XFINBENCH_TEST_GEMMA_3"

if [ "$OFFSET" -ge "$TOTAL" ]; then
  echo "No work for shard ${SLURM_ARRAY_TASK_ID} (OFFSET=$OFFSET >= TOTAL=$TOTAL). Exiting."
  exit 0
fi

echo "TOTAL=$TOTAL NUM_SHARDS=$NUM_SHARDS CHUNK=$CHUNK OFFSET=$OFFSET TAG=$TAG"

# Stage 0_static: build dataset shard from the XFinBench test split.
python -m src.run_eval_pipeline \
  stage=0_static \
  validation_tag="$TAG" \
  +static_benchmark_cfg.benchmark_id=Zhihan/XFinBench \
  +static_benchmark_cfg.split=test \
  +static_benchmark_cfg.offset="$OFFSET" \
  +static_benchmark_cfg.limit="$CHUNK" \
  +static_benchmark_cfg.domain=finance \
  +static_benchmark_cfg.capability_id=xfinbench_test \
  +static_benchmark_cfg.capability_name=XFinBenchTest

# Stage 1_local: evaluate local subject model(s) from run_cfg.yaml.
python -m src.run_eval_pipeline \
  stage=1_local \
  validation_tag="$TAG" \
  eval_tag="$TAG"

# Stage 2: aggregate per-shard scores.
python -m src.run_eval_pipeline \
  stage=2 \
  eval_tag="$TAG"

echo "Stage 0_static datasets: base_output/test_exp/eval/datasets/$TAG"
echo "Stage 1_local results:  base_output/test_exp/eval/results/$TAG"
echo "Stage 2 scores:         base_output/test_exp/eval/scores/$TAG"
