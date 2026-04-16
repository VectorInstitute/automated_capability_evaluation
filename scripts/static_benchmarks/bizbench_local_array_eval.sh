#!/bin/bash
#SBATCH --job-name=gemma_bizbench_local_array
#SBATCH --output=/projects/DeepLesion/projects/new_ace/automated_capability_evaluation/logs/bizbench_local_array_%A_%a.out
#SBATCH --error=/projects/DeepLesion/projects/new_ace/automated_capability_evaluation/logs/bizbench_local_array_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a40:1
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

# Count only FinKnow rows that survive adapter filtering.
TOTAL=$(
python - <<'PY'
from datasets import load_dataset

ds = load_dataset("kensho/bizbench", split="test")

def is_valid(row):
    question = str(row.get("question", "")).strip()
    task = str(row.get("task", "") or "").lower()
    answer = row.get("answer")
    if answer is None:
        answer_text = ""
    elif isinstance(answer, dict):
        for key in ("answer", "label", "text", "value"):
            if key in answer and answer[key] is not None:
                answer_text = str(answer[key]).strip()
                break
        else:
            answer_text = str(answer).strip()
    else:
        answer_text = str(answer).strip()
    # Adapter default is `finknow_only=true`, so we shard based on the same subset.
    return bool("finknow" in task and question and answer_text)

print(sum(1 for row in ds if is_valid(row)))
PY
)

CHUNK=$(((TOTAL + NUM_SHARDS - 1) / NUM_SHARDS))
OFFSET=$((SLURM_ARRAY_TASK_ID * CHUNK))
TAG="_BIZBENCH_TEST_GEMMA_3"

if [ "$OFFSET" -ge "$TOTAL" ]; then
  echo "No work for shard ${SLURM_ARRAY_TASK_ID} (OFFSET=$OFFSET >= TOTAL=$TOTAL). Exiting."
  exit 0
fi

echo "TOTAL=$TOTAL NUM_SHARDS=$NUM_SHARDS CHUNK=$CHUNK OFFSET=$OFFSET TAG=$TAG"

# Stage 0_static: build dataset shard from BizBench test split.
python -m src.run_eval_pipeline \
  stage=0_static \
  validation_tag="$TAG" \
  +static_benchmark_cfg.benchmark_id=kensho/bizbench \
  +static_benchmark_cfg.split=test \
  +static_benchmark_cfg.offset="$OFFSET" \
  +static_benchmark_cfg.limit="$CHUNK"

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
