#!/bin/bash
#SBATCH --job-name=gemma_book3_book4_local_array
#SBATCH --output=/projects/DeepLesion/projects/new_ace/automated_capability_evaluation/logs/gemma_book3_book4_local_array_%A_%a.out
#SBATCH --error=/projects/DeepLesion/projects/new_ace/automated_capability_evaluation/logs/gemma_book3_book4_local_array_%A_%a.err
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
ROOT_DIR="${ROOT_DIR:-/projects/DeepLesion/projects/new_ace/automated_capability_evaluation/Finance_Book3_Book4}"

export ROOT_DIR NUM_SHARDS SLURM_ARRAY_TASK_ID

mapfile -t ASSIGNED_FILES < <(
python - <<'PY'
from pathlib import Path
import os

root = Path(os.environ["ROOT_DIR"])
num_shards = int(os.environ["NUM_SHARDS"])
shard_id = int(os.environ["SLURM_ARRAY_TASK_ID"])

files = sorted(root.glob("**/tasks.json"))
for idx, path in enumerate(files):
    if idx % num_shards == shard_id:
        print(path)
PY
)

TOTAL_FILES=$(
python - <<'PY'
from pathlib import Path
import os

root = Path(os.environ["ROOT_DIR"])
print(len(sorted(root.glob("**/tasks.json"))))
PY
)

SHARD_FILES="${#ASSIGNED_FILES[@]}"
echo "ROOT_DIR=$ROOT_DIR TOTAL_FILES=$TOTAL_FILES NUM_SHARDS=$NUM_SHARDS SHARD=$SLURM_ARRAY_TASK_ID ASSIGNED_FILES=$SHARD_FILES"

if [ "$SHARD_FILES" -eq 0 ]; then
  echo "No tasks.json files assigned to shard ${SLURM_ARRAY_TASK_ID}. Exiting."
  exit 0
fi

for TASKS_JSON in "${ASSIGNED_FILES[@]}"; do
  export TASKS_JSON

  mapfile -t META < <(
  python - <<'PY'
import json
import os
import re
from pathlib import Path

path = Path(os.environ["TASKS_JSON"])
payload = json.loads(path.read_text(encoding="utf-8"))
tasks = payload.get("tasks", [])
first = tasks[0] if tasks and isinstance(tasks[0], dict) else {}

def clean(value: str, fallback: str) -> str:
    value = str(value or "").strip()
    if not value:
        value = fallback
    return value

def slug(value: str, fallback: str) -> str:
    value = clean(value, fallback)
    return re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower() or fallback

area_dir = path.parent.parent.name
cap_dir = path.parent.name

area_id = clean(first.get("area_id"), area_dir)
capability_id = clean(first.get("capability_id"), cap_dir)
area_name = clean(first.get("area_name"), area_id)
capability_name = clean(first.get("capability_name"), capability_id)
tag_suffix = slug(path.parent.relative_to(Path(os.environ["ROOT_DIR"])).as_posix(), capability_id)

print(area_id)
print(capability_id)
print(area_name)
print(capability_name)
print(tag_suffix)
PY
  )

  AREA_ID="${META[0]}"
  CAPABILITY_ID="${META[1]}"
  AREA_NAME="${META[2]}"
  CAPABILITY_NAME="${META[3]}"
  TAG_SUFFIX="${META[4]}"
  TAG="_FINANCE_BOOK3_BOOK4_GEMMA_3_${TAG_SUFFIX}"

  echo "Evaluating $TASKS_JSON"
  echo "  AREA_ID=$AREA_ID"
  echo "  CAPABILITY_ID=$CAPABILITY_ID"
  echo "  TAG=$TAG"

  # Stage 0_static: ingest one local tasks.json export using the local JSON adapter.
  python -m src.run_eval_pipeline \
    stage=0_static \
    validation_tag="$TAG" \
    +static_benchmark_cfg.benchmark_id="$TASKS_JSON" \
    +static_benchmark_cfg.area_id="$AREA_ID" \
    +static_benchmark_cfg.capability_id="$CAPABILITY_ID" \
    +static_benchmark_cfg.capability_name="$CAPABILITY_NAME" \
    +static_benchmark_cfg.domain=finance

  # Stage 1_local: evaluate local subject model(s) from run_cfg.yaml.
  python -m src.run_eval_pipeline \
    stage=1_local \
    validation_tag="$TAG" \
    eval_tag="$TAG"

  # Stage 2: aggregate scores for this tasks.json bundle.
  python -m src.run_eval_pipeline \
    stage=2 \
    eval_tag="$TAG"

  echo "Finished $TASKS_JSON"
  echo "  Stage 0_static datasets: base_output/test_exp/eval/datasets/$TAG"
  echo "  Stage 1_local results:  base_output/test_exp/eval/results/$TAG"
  echo "  Stage 2 scores:         base_output/test_exp/eval/scores/$TAG"
done
