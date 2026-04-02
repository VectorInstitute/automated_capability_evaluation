#!/bin/bash
#SBATCH --job-name=finance_tasks_eval
#SBATCH --output=logs/finance_tasks_eval_%j.out
#SBATCH --error=logs/finance_tasks_eval_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

cd /fs01/projects/DeepLesion/projects/new_ace/automated_capability_evaluation

# shellcheck disable=SC1091
source "scripts/static_benchmarks/env_slurm_inspect.sh"

VALIDATION_TAG="_FINANCE_TASKS_$(date +%Y%m%d_%H%M%S)"

# Stage 0_static: build datasets from local finance_tasks.json
python -m src.run_eval_pipeline \
  stage=0_static \
  validation_tag="$VALIDATION_TAG" \
  +static_benchmark_cfg.benchmark_id=finance_tasks.json \
  +static_benchmark_cfg.split=na \
  +static_benchmark_cfg.domain=finance \
  +static_benchmark_cfg.capability_id=finance_tasks \
  +static_benchmark_cfg.capability_name="Finance Tasks"
  # +static_benchmark_cfg.limit=30 \

# Stage 1: run subject models on the static datasets
python -m src.run_eval_pipeline \
  stage=1 \
  validation_tag="$VALIDATION_TAG" \
  eval_tag="$VALIDATION_TAG"

# Stage 2: aggregate scores
python -m src.run_eval_pipeline \
  stage=2 \
  eval_tag="$VALIDATION_TAG"

echo "Stage 0_static datasets: base_output/test_exp/eval/datasets/$VALIDATION_TAG"
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

