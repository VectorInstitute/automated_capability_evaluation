#!/bin/bash
#SBATCH --job-name=topic_cls_qwen3_32b
#SBATCH --output=logs/topic_cls_qwen3_32b_%j.out
#SBATCH --error=logs/topic_cls_qwen3_32b_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1

set -euo pipefail

cd /projects/DeepLesion/projects/new_ace/automated_capability_evaluation

# shellcheck disable=SC1091
source /projects/DeepLesion/py311_env/bin/activate

# shellcheck disable=SC1091
source "scripts/static_benchmarks/env_slurm_inspect.sh"

# Example defaults: classify XFinBench test split.
BENCHMARK_ID="${BENCHMARK_ID:-Zhihan/XFinBench}"
SPLIT="${SPLIT:-test}"
OUTPUT_JSONL="${OUTPUT_JSONL:-base_output/topic_classification/xfinbench_qwen3_32b.jsonl}"
MODEL_PATH="${MODEL_PATH:-/model-weights/Qwen3-32B}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MAX_TOKENS="${MAX_TOKENS:-32}"

python scripts/static_benchmarks/classify_static_benchmark_topics_vllm.py \
  --benchmark-id "$BENCHMARK_ID" \
  --split "$SPLIT" \
  --topic-csv "topic.csv" \
  --output-jsonl "$OUTPUT_JSONL" \
  --resume \
  --model-path "$MODEL_PATH" \
  --batch-size "$BATCH_SIZE" \
  --max-tokens "$MAX_TOKENS" \
  --trust-remote-code

echo "Saved classifications to $OUTPUT_JSONL"
