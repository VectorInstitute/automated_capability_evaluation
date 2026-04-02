#!/bin/bash
#SBATCH --job-name=xfinbench_test_eval
#SBATCH --output=logs/xfinbench_test_eval_%A_%a.out
#SBATCH --error=logs/xfinbench_test_eval_%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-9

set -euo pipefail

cd /projects/DeepLesion/projects/new_ace/automated_capability_evaluation

# shellcheck disable=SC1091
source "scripts/static_benchmarks/env_slurm_inspect.sh"

# Allow running via sbatch (with SLURM_ARRAY_TASK_ID) or directly (defaults to 0).
: "${SLURM_ARRAY_TASK_ID:=0}"

# 10 chunks over ~2828 filtered test examples → ~300 per chunk
CHUNK=300
OFFSET=$((SLURM_ARRAY_TASK_ID * CHUNK))
VALIDATION_TAG="_XFINBENCH_TEST_${SLURM_ARRAY_TASK_ID}_SundayNight"

# Stage 0_static: build datasets from Zhihan/XFinBench (test split, CSV-backed HF repo)
python -m src.run_eval_pipeline \
  stage=0_static \
  validation_tag="$VALIDATION_TAG" \
  +static_benchmark_cfg.benchmark_id=Zhihan/XFinBench \
  +static_benchmark_cfg.split=test \
  +static_benchmark_cfg.offset="$OFFSET" \
  +static_benchmark_cfg.limit="$CHUNK" \
  +static_benchmark_cfg.domain=finance \
  +static_benchmark_cfg.capability_id=xfinbench_test \
  +static_benchmark_cfg.capability_name="XFinBench Test"

# Stage 1: run subject models on the static datasets
python -m src.run_eval_pipeline \
  stage=1 \
  validation_tag="$VALIDATION_TAG" \
  eval_tag="$VALIDATION_TAG"

# Stage 2: aggregate scores
python -m src.run_eval_pipeline \
  stage=2 \
  eval_tag="$VALIDATION_TAG"

echo "Stage 0_static datasets (test split): base_output/test_exp/eval/datasets/$VALIDATION_TAG"
echo "Stage 1 results (Inspect logs): base_output/test_exp/eval/results/$VALIDATION_TAG"
echo "Stage 2 scores: base_output/test_exp/eval/scores/$VALIDATION_TAG"

# Optional: generate flattened JSONL views of Inspect logs for easier reading
RESULTS_DIR="base_output/test_exp/eval/results/$VALIDATION_TAG"
if [ -d "$RESULTS_DIR" ]; then
  echo "Flattening Inspect logs under $RESULTS_DIR ..."
  for model_dir in "$RESULTS_DIR"/*/; do
    [ -d "$model_dir" ] || continue
    model_name="$(basename "$model_dir")"
    for area_dir in "$model_dir"*/; do
      [ -d "$area_dir" ] || continue
      for cap_dir in "$area_dir"*/; do
        [ -d "$cap_dir" ] || continue
        cap_name="$(basename "$cap_dir")"
        log_file="$(ls "$cap_dir"/*_task_*.json 2>/dev/null | head -n 1 || true)"
        if [ -n "$log_file" ]; then
          out_file="$cap_dir/flat_${cap_name}.jsonl"
          python scripts/flatten_inspect_logs.py \
            --log_path "$log_file" \
            --out_path "$out_file"
          echo "  Wrote flattened log for $model_name/$cap_name to $out_file"
        fi
      done
    done
  done
fi

