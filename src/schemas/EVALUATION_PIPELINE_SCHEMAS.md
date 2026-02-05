# ACE Evaluation Pipeline Standardized Schemas

The **evaluation pipeline** takes the validated tasks and solutions from the generation pipeline and evaluates subject LLMs on them using Inspect AI. It produces capability scores that measure how well each subject LLM performs on each capability.

This document defines the standardized input and output formats for each stage of the evaluation pipeline. These schemas ensure consistency across different implementations and enable interoperability between pipeline stages.

## Pipeline Stages

The evaluation pipeline consists of three stages:

0. **Setup and Dataset Preparation**: Validate inputs, convert tasks to Inspect format (no LLM calls)
1. **Evaluation Execution**: Run Inspect evaluation with subject LLMs (creates `eval_tag`)
2. **Score Aggregation**: Compute capability scores from raw results (no LLM calls)

---

## Implementation Files

- [`src/run_eval_pipeline.py`](../run_eval_pipeline.py) (pipeline entrypoint)
- [`src/eval_stages/stage0_setup_and_dataset.py`](../eval_stages/stage0_setup_and_dataset.py)
- [`src/eval_stages/stage1_eval_execution.py`](../eval_stages/stage1_eval_execution.py)
- [`src/eval_stages/stage2_score_aggregation.py`](../eval_stages/stage2_score_aggregation.py)
- [`src/schemas/eval_schemas.py`](eval_schemas.py)
- [`src/schemas/eval_io_utils.py`](eval_io_utils.py)

---

## Implementation Approach

**Pipeline Pattern:**
- **Stage 0**: Deterministic data transformation (no LLM, no tag needed)
- **Stage 1**: LLM-dependent evaluation (creates `eval_tag` for results)
- **Stage 2**: Deterministic aggregation (uses `eval_tag` from Stage 1)

**Shared Config:**
The evaluation pipeline uses the **same configuration file** as the generation pipeline
([`src/cfg/run_cfg.yaml`](../cfg/run_cfg.yaml)), with an evaluation-specific section
(`eval_cfg`).

**Resumability:**
- **Stage 0**: Idempotent - skips datasets that already exist
- **Stage 1**: Creates a fresh `eval_tag` by default; skips any completed evaluations
  only if you re-run Stage 1 with the same `eval_tag` programmatically

---

## Configuration

```yaml
eval_cfg:
  # Subject LLMs to evaluate (required)
  subject_llms:
    - name: "gpt-4o"
      provider: "openai"
    - name: "claude-3-sonnet"
      provider: "anthropic"

  # Judge LLM for scoring (required)
  judge_llm:
    name: "gpt-4o-mini"
    provider: "openai"
```

---

## Naming Conventions

See [`src/schemas/GENERATION_PIPELINE_SCHEMAS.md`](GENERATION_PIPELINE_SCHEMAS.md) for
naming conventions. Tags follow the same format: `_YYYYMMDD_HHMMSS`.

---

## Directory Structure

Evaluation outputs are stored in an `eval/` subdirectory within the experiment directory
(see [`src/schemas/GENERATION_PIPELINE_SCHEMAS.md`](GENERATION_PIPELINE_SCHEMAS.md) for
generation structure):

```
<experiment_id>/
  eval/
    datasets/                              # Stage 0 output
      <validation_tag>/                    # Tied to validation source
        <area_id>/
          <capability_id>/
            dataset.json                   # EvalDataset

    results/                               # Stage 1 output
      <eval_tag>/
        eval_config.json                   # EvalConfig saved here
        <subject_llm>/
          <area_id>/
            <capability_id>/               # Inspect logs
              *.json                       # Inspect log files (per run)

    scores/                                # Stage 2 output
      <eval_tag>/
        <subject_llm>/
          capability_scores.json           # List[CapabilityScore]
```

**Example:**
```
r0_10x10/
  eval/
    datasets/
      _20251017_091500/
        area_000/
          cap_000/dataset.json
          cap_001/dataset.json
    results/
      _20251020_143000/
        eval_config.json
        gpt-4o/
          area_000/
            cap_000/
            cap_001/
        claude-3-sonnet/
          area_000/
            cap_000/
            cap_001/
    scores/
      _20251020_143000/
        gpt-4o/capability_scores.json
        claude-3-sonnet/capability_scores.json
```

---

## Dataclasses

The evaluation pipeline uses 3 dataclasses, plus reuses `PipelineMetadata` from the
generation pipeline (see [`src/schemas/GENERATION_PIPELINE_SCHEMAS.md`](GENERATION_PIPELINE_SCHEMAS.md#pipelinemetadata)).

**File:** [`src/schemas/eval_schemas.py`](eval_schemas.py)

### EvalConfig

Configuration for the evaluation run.

**Fields:**
- `experiment_id`: String (required)
- `eval_tag`: String (set in Stage 1)
- `subject_llms`: List[Dict] (required, each dict has "name" and "provider")
- `judge_llm`: Dict (required, has "name" and "provider")
- `validation_tag`: String (required, tag from generation Stage 5)

### EvalDataset

Dataset prepared for Inspect evaluation. Contains all info for one capability.

**Fields:**
- `area_id`: String (required)
- `capability_id`: String (required)
- `capability_name`: String (required)
- `domain`: String (required)
- `tasks`: List[Dict] (required, each dict has "id", "input", "target")
- `num_tasks`: Integer (required)
- `prompt_template`: String (required)

### CapabilityScore

Score for a single capability from evaluation.

**Fields:**
- `area_id`: String (required)
- `capability_id`: String (required)
- `capability_name`: String (required)
- `subject_llm`: String (required)
- `mean`: Float (required, 0.0 to 1.0)
- `std_err`: Float (required)
- `num_tasks`: Integer (required)

---

## Eval Stage 0: Setup and Dataset Preparation

### Purpose
Validate inputs and convert validated tasks to Inspect-compatible format.

### Input
- **validation_tag**: String - Tag from generation Stage 5 (required)
- **Configuration**: `eval_cfg` section from config YAML (subject_llms and judge_llm required)

### Validation Checks
1. Generation `experiment.json` exists
2. Validation outputs exist at `validation/<validation_tag>/`
3. `subject_llms` and `judge_llm` are configured

### Output: `dataset.json` (per capability)

**Stage Output:** EvalDataset dataclass
**Save Function:** `save_eval_dataset(dataset: EvalDataset, output_path: Path)`

**Also Saved:** `eval_config.json` (EvalConfig + PipelineMetadata)
**Path:** `<output_dir>/<experiment_id>/eval/datasets/<validation_tag>/eval_config.json`

**File Path:** `<output_dir>/<experiment_id>/eval/datasets/<validation_tag>/<area_id>/<capability_id>/dataset.json`

```json
{
  "area_id": "area_000",
  "capability_id": "cap_000",
  "capability_name": "compound_interest",
  "domain": "personal_finance",
  "tasks": [
    {"id": "task_000", "input": "What is the future value of $1000...", "target": "1647.01"},
    {"id": "task_001", "input": "Calculate the present value of $5000...", "target": "3402.92"}
  ],
  "num_tasks": 10,
  "prompt_template": "..."
}
```

**Returns:** EvalConfig object (for use in Stage 1)

---

## Eval Stage 1: Evaluation Execution

### Purpose
Run Inspect evaluation for each capability with each subject LLM.

### Input
- **eval_config**: EvalConfig from Stage 0

### Tag Handling
- **Creates**: `eval_tag` for this evaluation run (generated by `timestamp_tag()` in
  [`src/utils/timestamp_utils.py`](../utils/timestamp_utils.py))
- **Resume behavior**: Stage 1 always creates a fresh `eval_tag` via
  `timestamp_tag()`. The implementation skips any `(subject_llm, area_id, capability_id)`
  that already has log files under the **current** `eval_tag` directory. This is only
  relevant if you re-run Stage 1 using the same `eval_tag` programmatically.

### Output: Inspect logs + `eval_config.json`

**Stage Output:** Raw Inspect AI logs (stored by Inspect directly)

**File Path:** `<output_dir>/<experiment_id>/eval/results/<eval_tag>/<subject_llm>/<area_id>/<capability_id>/`

The `eval_config.json` is saved to
`<output_dir>/<experiment_id>/eval/results/<eval_tag>/eval_config.json` for reference.

**Returns:** `eval_tag` string

### Scoring Details (Per-Task)
- Each task in `EvalDataset.tasks` becomes an Inspect `Sample` with `id=task["id"]`
  (see [`src/eval_stages/stage1_eval_execution.py`](../eval_stages/stage1_eval_execution.py)).
- The judge model scores each sample via `model_graded_fact` during Stage 1.
- Per-task scores live **only** in the Inspect log JSON files, under
  `samples[].scores`. These scores are aggregated in Stage 2; there is no separate
  per-task score file written by this pipeline.

---

## Eval Stage 2: Score Aggregation

### Purpose
Compute final capability scores from raw Inspect results.

### Input
- **eval_tag**: Tag from Stage 1

### Output: `capability_scores.json` (per subject LLM)

**Stage Output:** List[CapabilityScore]
**Save Function:** `save_capability_scores(scores: List[CapabilityScore], output_path: Path)`

**File Path:** `<output_dir>/<experiment_id>/eval/scores/<eval_tag>/<subject_llm>/capability_scores.json`

**Aggregation Note:** Stage 2 reads all Inspect log JSON files under
`<output_dir>/<experiment_id>/eval/results/<eval_tag>/<subject_llm>/<area_id>/<capability_id>/`
and aggregates `samples[].scores` into `mean`, `std_err`, and `num_tasks`.

```json
[
  {
    "area_id": "area_000",
    "capability_id": "cap_000",
    "capability_name": "compound_interest",
    "subject_llm": "gpt-4o",
    "mean": 0.90,
    "std_err": 0.03,
    "num_tasks": 10
  }
]
```

**Returns:** `eval_tag` string

---

## Usage

### Run Full Evaluation

```bash
# Basic usage - evaluate all capabilities
python -m src.run_eval_pipeline validation_tag=_20251017_091500
```

### Run Specific Stages

```bash
# Run only Stage 0 (setup + dataset preparation)
python -m src.run_eval_pipeline stage=0 validation_tag=_20251017_091500

# Run Stage 0 + Stage 1 (setup, datasets, and evaluation)
python -m src.run_eval_pipeline stage=1 validation_tag=_20251017_091500

# Run only Stage 2 (score aggregation) - requires eval_tag from Stage 1
python -m src.run_eval_pipeline stage=2 eval_tag=_20251020_143000
```

---

## IO Utilities

The following functions are provided in [`src/schemas/eval_io_utils.py`](eval_io_utils.py):

### Save Functions
- `save_eval_config(config: EvalConfig, metadata: PipelineMetadata, path: Path)`
- `save_eval_dataset(dataset: EvalDataset, path: Path)`
- `save_capability_scores(scores: List[CapabilityScore], path: Path)`

### Load Functions
- `load_eval_config(path: Path) -> Tuple[EvalConfig, PipelineMetadata]`
- `load_eval_dataset(path: Path) -> EvalDataset`
- `load_capability_scores(path: Path) -> List[CapabilityScore]`

### Helper Functions
- Use `timestamp_tag()` from [`src/utils/timestamp_utils.py`](../utils/timestamp_utils.py)
  to generate tags
- `get_experiment_dir(output_base_dir: str, experiment_id: str) -> Path`

---

## Relationship to Generation Pipeline

The evaluation pipeline depends on the generation pipeline outputs:

| Eval Stage | Depends On | Generation Stage |
|------------|------------|------------------|
| Eval Stage 0 | `experiment.json` | Stage 0 |
| Eval Stage 0 | `validation/<validation_tag>/` | Stage 5 |

---

## Legacy: LBO Support

The previous version of the repository included **Latent Bayesian Optimization (LBO)** for intelligent capability selection during evaluation. This functionality has been moved to the `legacy/` directory for reference.

See `legacy/README.md` for details on the LBO implementation and how it was used.

---
