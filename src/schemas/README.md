# ACE Pipeline Schemas

This directory contains standardized schemas for all ACE pipeline stages, ensuring consistent data formats across different implementations.

## Structure

- **[`PIPELINE_SCHEMAS.md`](PIPELINE_SCHEMAS.md)** - Complete documentation of input/output formats for each stage
- **Python Dataclasses** - Type-safe data structures for each stage:
  - [`experiment_schemas.py`](experiment_schemas.py) - Experiment and Domain (Stage 0)
  - [`metadata_schemas.py`](metadata_schemas.py) - Common metadata (PipelineMetadata)
  - [`area_schemas.py`](area_schemas.py) - Area generation (Stage 1)
  - [`capability_schemas.py`](capability_schemas.py) - Capability generation (Stage 2)
  - [`task_schemas.py`](task_schemas.py) - Task generation (Stage 3)
  - [`solution_schemas.py`](solution_schemas.py) - Solution generation (Stage 4)
  - [`validation_schemas.py`](validation_schemas.py) - Validation (Stage 5)
- **I/O Utilities** - Save and load functions:
  - [`io_utils.py`](io_utils.py) - Functions to save/load all stage outputs (save/load functions for all 7 stage outputs)

## Usage

### Using Python Dataclasses

```python
from src.schemas import (
    Experiment,
    Domain,
    PipelineMetadata,
    Area,
    Capability,
    Task,
    TaskSolution,
    ValidationResult,
)

# Create area
area = Area(
    name="Cash Flow & Budget Management",
    area_id="area_000",
    description="Design and monitor budgets...",
    domain="personal finance",
    domain_id="domain_000",
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
    save_areas_output,
    load_areas_output,
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
save_areas_output(areas, metadata, Path("output/areas.json"))

# Load areas
areas, metadata = load_areas_output(Path("output/areas.json"))
```

## Pipeline Stages

0. **Experiment Setup** → `Experiment`, `Domain`
1. **Area Generation** → `Area`
2. **Capability Generation** → `Capability`
3. **Task Generation** → `Task`
4. **Solution Generation** → `TaskSolution`
5. **Validation** → `ValidationResult`

See [`PIPELINE_SCHEMAS.md`](PIPELINE_SCHEMAS.md) for detailed specifications.
