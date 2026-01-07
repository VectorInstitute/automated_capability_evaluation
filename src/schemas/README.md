# ACE Pipeline Schemas

This directory contains standardized schemas for all ACE pipeline stages, ensuring consistent data formats across different implementations.

## Structure

### Generation Pipeline

- **[`GENERATION_PIPELINE_SCHEMAS.md`](GENERATION_PIPELINE_SCHEMAS.md)** - Documentation for generation pipeline stages
- **Python Dataclasses** - Type-safe data structures:
  - [`experiment_schemas.py`](experiment_schemas.py) - Experiment (Stage 0)
  - [`domain_schemas.py`](domain_schemas.py) - Domain (Stage 0)
  - [`metadata_schemas.py`](metadata_schemas.py) - Common metadata (PipelineMetadata)
  - [`area_schemas.py`](area_schemas.py) - Area generation (Stage 1)
  - [`capability_schemas.py`](capability_schemas.py) - Capability generation (Stage 2)
  - [`task_schemas.py`](task_schemas.py) - Task generation (Stage 3)
  - [`solution_schemas.py`](solution_schemas.py) - Solution generation (Stage 4)
  - [`validation_schemas.py`](validation_schemas.py) - Validation (Stage 5)
- **I/O Utilities**:
  - [`io_utils.py`](io_utils.py) - Save/load functions for generation pipeline outputs

### Evaluation Pipeline

- **[`EVALUATION_PIPELINE_SCHEMAS.md`](EVALUATION_PIPELINE_SCHEMAS.md)** - Documentation for evaluation pipeline stages
- **Python Dataclasses**:
  - [`eval_schemas.py`](eval_schemas.py) - EvalConfig, EvalDataset, CapabilityScore
- **I/O Utilities**:
  - [`eval_io_utils.py`](eval_io_utils.py) - Save/load functions for evaluation pipeline outputs

## Usage

### Using Python Dataclasses

```python
from src.schemas import (
    Domain,
    Experiment,
    PipelineMetadata,
    Area,
    Capability,
    Task,
    TaskSolution,
    ValidationResult,
)

# Create area
domain = Domain(name="Personal Finance", domain_id="domain_000")
area = Area(
    name="Cash Flow & Budget Management",
    area_id="area_000",
    description="Design and monitor budgets...",
    domain=domain,
    # generation_metadata is optional
)

# Convert to dict for JSON serialization
data = area.to_dict()

# Load from dict
area = Area.from_dict(data)
```

### Using Save/Load Functions

```python
from pathlib import Path
from src.schemas import (
    save_areas,
    load_areas,
    PipelineMetadata,
    Area,
)

# Save areas
areas = [Area(...), Area(...)]
metadata = PipelineMetadata(
    experiment_id="r0_10x10",
    output_base_dir="agentic_outputs",
    timestamp="2025-11-06T12:00:00Z",
    output_stage_tag="_20251009_122040"
)
save_areas(areas, metadata, Path("output/areas.json"))

# Load areas
areas, metadata = load_areas(Path("output/areas.json"))
```

## Pipeline Stages

### Generation Pipeline

0. **Experiment Setup** → `Experiment`, `Domain`
1. **Area Generation** → `Area`
2. **Capability Generation** → `Capability`
3. **Task Generation** → `Task`
4. **Solution Generation** → `TaskSolution`
5. **Validation** → `ValidationResult`

See [`GENERATION_PIPELINE_SCHEMAS.md`](GENERATION_PIPELINE_SCHEMAS.md) for detailed specifications.

### Evaluation Pipeline

0. **Setup and Dataset Preparation** → `EvalConfig`, `EvalDataset`
1. **Evaluation Execution** → Inspect AI logs (creates `eval_tag`)
2. **Score Aggregation** → `CapabilityScore`

See [`EVALUATION_PIPELINE_SCHEMAS.md`](EVALUATION_PIPELINE_SCHEMAS.md) for detailed specifications.
