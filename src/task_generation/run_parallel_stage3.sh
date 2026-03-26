#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

WORKER_COUNT="${WORKER_COUNT:-5}"
TASKS_TAG="${1:-}"
CAPABILITIES_TAG="${2:-}"

if [[ -z "$CAPABILITIES_TAG" ]]; then
  echo "Usage: bash src/task_generation/run_parallel_stage3.sh [tasks_tag] <capabilities_tag>" >&2
  echo "Example (fresh run): bash src/task_generation/run_parallel_stage3.sh '' placeholder" >&2
  echo "Example (resume): bash src/task_generation/run_parallel_stage3.sh _20260326_120000 placeholder" >&2
  exit 1
fi

if [[ -z "$TASKS_TAG" ]]; then
  TASKS_TAG="$(date +"_%Y%m%d_%H%M%S")"
fi

echo "Running agentic Stage 3 in parallel via src.run_base_pipeline"
echo "  tasks_tag: $TASKS_TAG"
echo "  worker_count: $WORKER_COUNT"
echo "  capabilities_tag: $CAPABILITIES_TAG"

PIDS=()
for ((i=0; i<WORKER_COUNT; i++)); do
  CMD=(
    python3 -m src.run_base_pipeline
    stage=3
    task_generation_cfg.mode=agentic
    tasks_tag="$TASKS_TAG"
    capabilities_tag="$CAPABILITIES_TAG"
    task_generation_cfg.worker_index="$i"
    task_generation_cfg.worker_count="$WORKER_COUNT"
  )

  echo "Starting worker $i/$WORKER_COUNT: ${CMD[*]}"
  "${CMD[@]}" &
  PIDS+=("$!")
done

FAIL=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    FAIL=1
  fi
done

if [[ "$FAIL" -ne 0 ]]; then
  echo "One or more workers failed for tasks_tag=$TASKS_TAG" >&2
  exit 1
fi

echo "All workers finished successfully for tasks_tag=$TASKS_TAG"
echo "Reuse this same tasks_tag to resume:"
echo "  $TASKS_TAG"
