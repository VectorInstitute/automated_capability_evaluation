# ACE Pipeline Standardized Schemas

This document defines the standardized input and output formats for each stage of the ACE pipeline. These schemas ensure consistency across different implementations and enable interoperability between pipeline stages.

## Pipeline Stages

The ACE pipeline consists of multiple stages, where each stage consumes the output from the previous stage:

0. **Experiment Setup** - Initialize experiment and create domain metadata
1. **Area Generation** - Generate domain areas
2. **Capability Generation** - Generate capabilities for each area
3. **Task Generation** - Generate tasks for each capability
4. **Solution Generation** - Generate solutions for each task
5. **Validation** - Validate solutions against tasks

**Note:** Experiment configuration must remain consistent throughout the pipeline. Once set during experiment setup, it should not be changed to avoid inconsistencies.

## Implementation Approach

**Pipeline Pattern:**
Each stage follows a consistent pattern:
1. **Consumes Previous Stage Output**: Each stage (except Stage 0) loads data from the previous stage's output files using provided load functions
2. **Stage Implementation**: Produces dataclass objects (or lists of dataclasses) + metadata
3. **Save Function**: Takes dataclass objects + metadata → saves to JSON file using provided save functions

**Important:** All stage implementations must follow this pattern to ensure the pipeline is clean, consistent, and maintainable. This enables interoperability between different implementations, resumability of failed runs, and clear traceability through the pipeline.

**Note:** The dataclasses, save functions (`save_<stage>(data, metadata, output_path)`), and load functions (`load_<stage>(file_path) -> <OutputDataclass>`) for each stage will be provided and must be used. Do not implement custom serialization or data structures - use the standardized schemas to ensure consistency across the pipeline. Dataclasses provide type safety, validation, and clear structure. JSON is the serialization format.

**Iteration Note:** Some stages operate on subsets (one area, capability, or task at a time) and require an outer orchestrator/loop script to iterate over all items:
- **Stage 2 (Capability Generation)**: Operates on one area at a time - orchestrator loops over all areas from Stage 1
- **Stage 3 (Task Generation)**: Operates on one capability at a time - orchestrator loops over all capabilities from Stage 2
- **Stage 4 (Solution Generation)**: Operates on one task at a time - orchestrator loops over all tasks from Stage 3
- **Stage 5 (Validation)**: Operates on one task at a time - orchestrator loops over all solutions from Stage 4

The stage implementation itself handles a single item, and the orchestrator manages the iteration across all items.

---

## Naming Conventions

All identifiers and tags in the pipeline follow standardized formats:

### Tags
- **Format**: `_YYYYMMDD_HHMMSS` (e.g., `_20251009_122040`)
- **Usage**: Used for versioning outputs in Stages 1-5
- **Generation**: Automatically generated when a new run is created (timestamp-based)

### Domain IDs
- **Format**: `domain_` + zero-padded 3-digit number (e.g., `domain_000`)
- **Assignment**: Sequential starting from `domain_000`
- **Scope**: Unique within an experiment (typically only one domain per experiment)

### Area IDs
- **Format**: `area_` + zero-padded 3-digit number (e.g., `area_000`, `area_001`)
- **Assignment**: Sequential starting from `area_000` when areas are generated
- **Scope**: Unique within an experiment

### Capability IDs
- **Format**: `cap_` + zero-padded 3-digit number (e.g., `cap_000`, `cap_001`)
- **Assignment**: Sequential starting from `cap_000` within each area when capabilities are generated
- **Scope**: Unique within an area (but can repeat across areas, e.g., `area_000/cap_000/` and `area_001/cap_000/`)

### Task IDs
- **Format**: `task_` + zero-padded 3-digit number (e.g., `task_000`, `task_001`)
- **Assignment**: Sequential starting from `task_000` within each capability when tasks are generated
- **Scope**: Unique within a capability


---

## Directory Structure

All outputs are stored in the following flat directory structure, with each stage having its own top-level directory and tags for different generation runs:

```
<output_dir>/
  <experiment_id>/
    experiment.json                  # Experiment metadata (all configuration)
    domain/
      domain.json                    # Domain metadata (contains domain_id)
    areas/
      <area_tag>/                   # Tag from area generation (e.g., _20251009_122040)
        areas.json                  # All areas for this area generation run (output from Stage 1)
    capabilities/
      <cap_tag>/                    # Tag from capability generation (e.g., _20251009_131252)
        <area_id>/                 # One directory per area (e.g., area_000, area_001)
          capabilities.json         # All capabilities for this area in this generation run
    tasks/
      <task_tag>/                   # Tag from task generation (e.g., _20251014_114358)
        <area_id>/                  # One directory per area (e.g., area_000, area_001)
          <capability_id>/          # One directory per capability (e.g., cap_000, cap_001)
            tasks.json              # All tasks for this capability in this generation run
    solutions/
      <solution_tag>/               # Tag from solution generation (e.g., _20251016_182128)
        <area_id>/                  # One directory per area (e.g., area_000, area_001)
          <capability_id>/          # One directory per capability (e.g., cap_000, cap_001)
            <task_id>/              # One directory per task (e.g., task_000, task_001)
              solution.json         # Solution for this task
    validation/
      <validation_tag>/             # Tag from validation run (e.g., _20251017_091500)
        <area_id>/                  # One directory per area (e.g., area_000, area_001)
          <capability_id>/          # One directory per capability (e.g., cap_000, cap_001)
            <task_id>/              # One directory per task (e.g., task_000, task_001)
              validation.json       # Validation result for this task
```

**Example:**
```
agentic_outputs/
  r0_10x10/
    experiment.json                  # Experiment configuration
    domain/
      domain.json                    # Domain metadata
    areas/
      _20251009_122040/              # Tag from first area generation
        areas.json                   # All areas from this generation
      _20251010_143022/              # Tag from second area generation (different set of areas)
        areas.json                   # All areas from this generation
    capabilities/
      _20251009_131252/              # Tag from first capability generation
        area_000/
          capabilities.json          # Capabilities for area_000
        area_001/
          capabilities.json          # Capabilities for area_001
      _20251011_091500/              # Tag from second capability generation
        area_000/
          capabilities.json          # Capabilities for area_000
    tasks/
      _20251014_114358/              # Tag from first task generation
        area_000/
          cap_000/
            tasks.json               # Tasks for area_000/cap_000
          cap_001/
            tasks.json               # Tasks for area_000/cap_001
        area_001/
          cap_000/
            tasks.json               # Tasks for area_001/cap_000
      _20251015_120000/              # Tag from second task generation
        area_000/
          cap_000/
            tasks.json               # Tasks for area_000/cap_000
    solutions/
      _20251016_182128/              # Tag from solution generation
        area_000/
          cap_000/
            task_000/
              solution.json
            task_001/
              solution.json
        area_001/
          cap_000/
            task_000/
              solution.json
    validation/
      _20251017_091500/              # Tag from validation run
        area_000/
          cap_000/
            task_000/
              validation.json
            task_001/
              validation.json
        area_001/
          cap_000/
            task_000/
              validation.json
```

**File Naming:**
- Experiment: `experiment.json` (no versioning, one file per experiment, contains all configuration)
- Domain: `domain.json` (no versioning, one file per experiment)
- Areas: `areas.json` (versioned by tag: `areas/<area_tag>/areas.json`)
- Capabilities: `capabilities.json` (versioned by tag: `capabilities/<cap_tag>/<area_id>/capabilities.json`)
- Tasks: `tasks.json` (versioned by tag: `tasks/<task_tag>/<area_id>/<capability_id>/tasks.json`)
- Solutions: `solution.json` (versioned by tag: `solutions/<solution_tag>/<area_id>/<capability_id>/<task_id>/solution.json`)
- Validation: `validation.json` (versioned by tag: `validation/<validation_tag>/<area_id>/<capability_id>/<task_id>/validation.json`)

**Resumability Benefits:**
- Each area has its own directory - easy to see which areas are processed
- Each capability has its own directory - easy to see which capabilities are complete
- Missing files are immediately visible in the directory structure
- Can resume from any area or capability by checking if directory/files exist
- Tags allow multiple runs/versions to coexist
- Can check latest tag to determine most recent run

**Versioning Strategy:**
- Each stage generates a new tag when run (see Tags in Naming Conventions section)
- Tags are independent per stage (areas can have different tag than capabilities)
- **Input tags**: Each stage requires tag(s) from previous stage(s) to load input data
  - Stage 1 (Areas): No input tag (uses domain.json)
  - Stage 2 (Capabilities): Requires `areas_tag` from Stage 1
  - Stage 3 (Tasks): Requires `capabilities_tag` from Stage 2
  - Stage 4 (Solutions): Requires `tasks_tag` from Stage 3
  - Stage 5 (Validation): Requires `solutions_tag` from Stage 4 (task information is included in solution files)
- **Resume tags**: Optional - If provided, stage loads existing output and continues incomplete generation
  - Checks for existing files with resume tag
  - Identifies which items are incomplete (e.g., missing capabilities, tasks, solutions)
  - Continues generation only for incomplete items
  - Preserves existing completed items
- **New tags**: If no resume tag provided, generates new tag and creates fresh output

---

## Dataclasses

All dataclasses used across pipeline stages are defined below. Stage implementations must use these standardized dataclasses.

**Note:** All ID and tag formats (domain_id, area_id, capability_id, task_id, tags) are defined in the [Naming Conventions](#naming-conventions) section. Individual field descriptions below do not repeat these format definitions.

### PipelineMetadata

All pipeline outputs include a `metadata` object (represented by the `PipelineMetadata` dataclass) that provides pipeline execution context and traceability.

**Required Fields:**
- `experiment_id`: String (required, experiment identifier)
- `output_base_dir`: String (required, base output directory for all pipeline outputs)
- `timestamp`: String (required, ISO 8601 format, e.g., "2025-11-06T12:00:00Z")
- `input_stage_tag`: String (optional, tag of the input data used from previous stage) - Present when stage uses input from previous stage, null for Stage 0
- `output_stage_tag`: String (optional, tag for this output) - Present for versioned stages (Stages 1-5), null for Stage 0 (not versioned)
- `resume`: Boolean (required, indicates if this run was resumed from a previous checkpoint)

**Optional Fields:**
- Additional optional fields may be added as needed for pipeline-specific metadata

**Note:**
- Stage-specific identifiers (domain_id, area_id, capability_id, task_id) are stored in the actual data objects (Domain, Area, Capability, Task), NOT in PipelineMetadata
- PipelineMetadata focuses on pipeline execution context, not the content being processed

### Experiment

**Fields:**
- `experiment_id`: String (required, experiment identifier)
- `domain`: String (required, human-readable domain name)
- `domain_id`: String (required)
- `pipeline_type`: String (optional, e.g., "agentic", "diverse_task") - identifies the pipeline variant
- `configuration`: Dict[str, Any] (required, complete configuration used for this experiment - structure varies by pipeline type)

### Domain

**Fields:**
- `name`: String (required, human-readable domain name)
- `domain_id`: String (required)
- `description`: String (optional, domain description)

### Area

**Fields:**
- `name`: String (required, human-readable area name)
- `area_id`: String (required)
- `description`: String (optional, area description)
- `domain`: String (required, domain name)
- `domain_id`: String (required)
- `generation_metadata`: Dict (optional, nested dictionary containing process-specific information)
  - This field can contain any generation-specific data (e.g., generation method, parameters, intermediate steps)
  - Structure is flexible and depends on the generation method

### Capability

**Fields:**
- `name`: String (required, capability name)
- `capability_id`: String (required)
- `description`: String (optional, capability description)
- `area`: String (required, area name)
- `area_id`: String (required)
- `domain`: String (required, domain name)
- `domain_id`: String (required)
- `generation_metadata`: Dict (optional, nested dictionary containing process-specific information)
  - This field can contain any generation-specific data (e.g., generation method, parameters, intermediate steps)
  - Structure is flexible and depends on the generation method

### Task

**Fields:**
- `task_id`: String (required, unique within capability)
- `task`: String (required, the task/problem text)
- `capability_id`: String (required)
- `capability`: String (required, capability name)
- `area`: String (required, area name)
- `area_id`: String (required)
- `domain`: String (required, domain name)
- `domain_id`: String (required)

### TaskSolution

**Fields:**
- `task_id`: String (required)
- `task`: String (required, the task/problem text from Stage 3)
- `capability`: String (required, capability name)
- `capability_id`: String (required)
- `area`: String (required, area name)
- `area_id`: String (required)
- `domain`: String (required, domain name)
- `domain_id`: String (required)
- `solution`: String (required, the final solution)
- `reasoning`: String (required, explanation of the solution)
- `numerical_answer`: String (optional, JSON string with numerical results)
- `generation_metadata`: Dict (optional, nested dictionary containing process-specific information)
  - This field can contain any generation-specific data (e.g., debate rounds, agent interactions, pipeline type)
  - Structure is flexible and depends on the generation method (agentic, single-agent, etc.)

### ValidationResult

**Fields:**
- `task_id`: String (required)
- `task`: String (required, the task/problem text from Stage 3)
- `capability`: String (required, capability name)
- `capability_id`: String (required)
- `area`: String (required, area name)
- `area_id`: String (required)
- `domain`: String (required, domain name)
- `domain_id`: String (required)
- `verification`: Boolean (required, overall validation status - whether the solution is verified/valid)
- `feedback`: String (required, detailed feedback on the validation)
- `score`: Float (optional, validation score, typically 0.0 to 1.0)
- `generation_metadata`: Dict (optional, nested dictionary containing process-specific information)
  - This field can contain any validation-specific data (e.g., validation method, criteria details, error details)
  - Structure is flexible and depends on the validation method

---

## Stage 0: Experiment Setup

### Input
All inputs come from a configuration YAML file (e.g., `src/cfg/agentic_config.yaml`). Important fields include:
- **Experiment ID**: String - The experiment identifier (e.g., "r0_10x10")
- **Domain Name**: String - The domain name (e.g., "personal finance", "mathematics")
- **Description**: String (optional) - Domain description
- **Output Base Directory**: String - Base output directory for all pipeline outputs (e.g., `global_cfg.output_dir` in agentic pipeline)

**Note:** The `experiment_id` and `output_base_dir` from the config YAML file are consistent across all stages. All stage-specific configurations (e.g., `num_areas`, `num_capabilities_per_area`, `num_tasks_per_capability`) also come from this same config YAML file.

### Tag Handling
- **No input tag required** (first stage)
- **No resume tag** - Not applicable (single domain JSON, always creates new files)

### Outputs

This stage creates two files:
1. `experiment.json` - Experiment metadata
2. `domain.json` - Domain metadata

#### Output 1: `experiment.json`

**Stage Output:** Experiment dataclass + PipelineMetadata
**Save Function:** `save_experiment(experiment: Experiment, metadata: PipelineMetadata, output_path: Path)`

**File Path:** `<output_dir>/<experiment_id>/experiment.json`

```json
{
  "metadata": {
    "experiment_id": "r0_10x10",
    "output_base_dir": "agentic_outputs",
    "timestamp": "2025-11-06T12:00:00Z",
    "input_stage_tag": null,
    "output_stage_tag": null,
    "resume": false
  },
  "experiment": {
    "experiment_id": "r0_10x10",
    "domain": "personal finance",
    "domain_id": "domain_000",
    "pipeline_type": "agentic",
    "configuration": {
      ...
    }
  }
}
```

**Schema:** See `Experiment` and `PipelineMetadata` dataclasses in the Dataclasses section above.

#### Output 2: `domain.json`

**Stage Output:** Domain dataclass object + PipelineMetadata
**Save Function:** `save_domain(domain: Domain, metadata: PipelineMetadata, output_path: Path)`

**File Path:** `<output_dir>/<experiment_id>/domain/domain.json`

```json
{
  "metadata": {
    "experiment_id": "r0_10x10",
    "output_base_dir": "agentic_outputs",
    "timestamp": "2025-11-06T12:00:00Z",
    "input_stage_tag": null,
    "output_stage_tag": null,
    "resume": false
  },
  "domain": {
    "name": "personal finance",
    "domain_id": "domain_000",
    "description": "Personal finance domain covering budgeting, investing, retirement planning, etc."
  }
}
```

---

## Stage 1: Area Generation

### Input
- **Domain**: Domain object (from Stage 0) - Loaded from `domain/domain.json`
- **Configuration**: Dict - Stage-specific configuration from config YAML file (e.g., `num_areas`)

### Tag Handling
- **Input tag**: Not applicable (uses `domain/domain.json` from Stage 0, which has no tag)
- **Resume tag**: Not applicable (single `areas.json` file with all areas, always creates new files)
- **New tag**: Generates new tag and creates `areas/<new_tag>/areas.json`

### Output: `areas.json`

**Stage Output:** List[Area] dataclasses + PipelineMetadata
**Save Function:** `save_areas(areas: List[Area], metadata: PipelineMetadata, output_path: Path)`

**File Path:** `<output_dir>/<experiment_id>/areas/<tag>/areas.json`
```json
{
  "metadata": {
    "experiment_id": "r0_10x10",
    "output_base_dir": "agentic_outputs",
    "timestamp": "2025-11-06T12:00:00Z",
    "input_stage_tag": null,
    "output_stage_tag": "_20251009_122040",
    "resume": false
  },
  "areas": [
    {
      "name": "Cash Flow & Budget Management",
      "area_id": "area_000",
      "description": "Design and monitor budgets using various methodologies...",
      "domain": "personal finance",
      "domain_id": "domain_000"
    }
  ]
}
```

---

## Stage 2: Capability Generation

### Input
- **Areas tag**: String - Tag from Stage 1 output (e.g., `_20251009_122040`)
  - Loads areas from `areas/<areas_tag>/areas.json`
- **Configuration**: Dict - Stage-specific configuration from config YAML file (e.g., `num_capabilities_per_area`)

### Tag Handling
- **Resume tag**: Optional - If provided, goes to `capabilities/<resume_tag>/` directory
  - For each area_id, checks if `capabilities/<resume_tag>/<area_id>/capabilities.json` exists
  - If file exists, capabilities for that area were already generated successfully, so skip it
  - If file doesn't exist, creates `<area_id>/` subdirectory and generates capabilities for that area
- **New tag**: If no resume tag provided, generates new tag (cap_tag) for this capability generation run
  - For each area, creates `capabilities/<cap_tag>/<area_id>/capabilities.json`

### Output: `capabilities.json` (one per area)

**Stage Output:** List[Capability] dataclasses + PipelineMetadata
**Save Function:** `save_capabilities(capabilities: List[Capability], metadata: PipelineMetadata, output_path: Path)`

**File Path:** `<output_dir>/<experiment_id>/capabilities/<cap_tag>/<area_id>/capabilities.json`

```json
{
  "metadata": {
    "experiment_id": "r0_10x10",
    "output_base_dir": "agentic_outputs",
    "timestamp": "2025-11-06T12:30:00Z",
    "input_stage_tag": "_20251009_122040",
    "output_stage_tag": "_20251009_131252",
    "resume": false
  },
  "capabilities": [
    {
      "name": "budget_policy_and_structure",
      "capability_id": "cap_000",
      "description": "Define the strategic framework and methodology for budgeting...",
      "area": "Cash Flow & Budget Management",
      "area_id": "area_000",
      "domain": "personal finance",
      "domain_id": "domain_000"
    }
  ]
}
```

---

## Stage 3: Task Generation

### Input
- **Capabilities tag**: String - Tag from Stage 2 output (e.g., `_20251009_131252`)
  - Loads capabilities from `capabilities/<capabilities_tag>/<area_id>/capabilities.json` for each area
- **Configuration**: Dict - Stage-specific configuration from config YAML file (e.g., `num_final_problems_per_capability`)

### Tag Handling
- **Resume tag**: Optional - If provided, goes to `tasks/<resume_tag>/` directory
  - For each `<area_id>` and `<capability_id>`, checks if `tasks/<resume_tag>/<area_id>/<capability_id>/tasks.json` exists
  - If file exists, tasks for that capability were already generated successfully, so skip it
  - If file doesn't exist, creates `<area_id>/<capability_id>/` subdirectories and generates tasks for that capability
- **New tag**: If no resume tag provided, generates new tag (task_tag) for this task generation run
  - For each capability, creates `tasks/<task_tag>/<area_id>/<capability_id>/tasks.json`

### Output: `tasks.json` (one per capability)

**Stage Output:** List[Task] dataclasses + PipelineMetadata
**Save Function:** `save_tasks(tasks: List[Task], metadata: PipelineMetadata, output_path: Path)`

**File Path:** `<output_dir>/<experiment_id>/tasks/<task_tag>/<area_id>/<capability_id>/tasks.json`

```json
{
  "metadata": {
    "experiment_id": "r0_10x10",
    "output_base_dir": "agentic_outputs",
    "timestamp": "2025-11-06T13:00:00Z",
    "input_stage_tag": "_20251009_131252",
    "output_stage_tag": "_20251014_114358",
    "resume": false
  },
  "tasks": [
    {
      "task_id": "task_000",
      "task": "You are advising a client who wants to set up a zero-based budget...",
      "capability_id": "cap_000",
      "capability": "budget_policy_and_structure",
      "area": "Cash Flow & Budget Management",
      "area_id": "area_000",
      "domain": "personal finance",
      "domain_id": "domain_000"
    },
    {
      "task_id": "task_001",
      "task": "A family of four needs to restructure their budget...",
      "capability_id": "cap_000",
      "capability": "budget_policy_and_structure",
      "area": "Cash Flow & Budget Management",
      "area_id": "area_000",
      "domain": "personal finance",
      "domain_id": "domain_000"
    }
  ]
}
```

---

## Stage 4: Solution Generation

### Input
- **Tasks tag**: String - Tag from Stage 3 output (e.g., `_20251014_114358`)
  - For each area and capability, loads tasks from `tasks/<tasks_tag>/<area_id>/<capability_id>/tasks.json`
- **Configuration**: Dict - Stage-specific configuration from config YAML file (e.g., `max_rounds`)

### Tag Handling
- **Resume tag**: Optional - If provided, goes to `solutions/<resume_tag>/` directory
  - For each area_id, capability_id, and task_id combination, checks if `solutions/<resume_tag>/<area_id>/<capability_id>/<task_id>/solution.json` exists
  - If file exists, solution for that task was already generated successfully, so skip it
  - If file doesn't exist, creates `<area_id>/<capability_id>/<task_id>/` subdirectories and generates solution for that task
- **New tag**: If no resume tag provided, generates new tag (solution_tag) for this solution generation run
  - For each task, creates `solutions/<solution_tag>/<area_id>/<capability_id>/<task_id>/solution.json`

### Output: `solution.json` (one per task)

**Stage Output:** TaskSolution dataclass + PipelineMetadata
**Save Function:** `save_solution(task_solution: TaskSolution, metadata: PipelineMetadata, output_path: Path)`

**File Path:** `<output_dir>/<experiment_id>/solutions/<solution_tag>/<area_id>/<capability_id>/<task_id>/solution.json`

```json
{
  "metadata": {
    "experiment_id": "r0_10x10",
    "output_base_dir": "agentic_outputs",
    "timestamp": "2025-11-06T13:30:00Z",
    "input_stage_tag": "_20251014_114358",
    "output_stage_tag": "_20251016_182128",
    "resume": false
  },
  "task_id": "task_000",
  "task": "You are advising a client who wants to set up a zero-based budget...",
  "capability": "budget_policy_and_structure",
  "capability_id": "cap_000",
  "area": "Cash Flow & Budget Management",
  "area_id": "area_000",
  "domain": "personal finance",
  "domain_id": "domain_000",
  "solution": "The optimal approach is to use a zero-based budgeting methodology...",
  "reasoning": "Both agents agreed on the zero-based approach because...",
  "numerical_answer": "{\"budget_allocation\": {...}}",
  "generation_metadata": {
    "pipeline_type": "agentic",
    "consensus_reached": true,
    "total_rounds": 2,
    "agents": [
      {
        "agent_id": "A",
        "thought": "I need to analyze the client's financial situation...",
        "final_answer": "{\"recommendation\": {...}}",
        "round_number": 0
      },
      {
        "agent_id": "B",
        "thought": "The client's income and expenses suggest...",
        "final_answer": "{\"recommendation\": {...}}",
        "round_number": 0
      }
    ]
  }
}
```

---

## Stage 5: Validation

### Input
- **Solutions tag**: String - Tag from Stage 4 output (e.g., `_20251016_182128`)
  - For each area, capability, and task, loads solutions from `solutions/<solutions_tag>/<area_id>/<capability_id>/<task_id>/solution.json`
  - Task information is included in the solution files, so no separate tasks tag is needed
- **Configuration**: Dict - Stage-specific configuration from config YAML file (e.g., validation criteria)

### Tag Handling
- **Input tag**: Required - `solutions_tag` from Stage 4
  - For each area, capability, and task, loads solutions from `solutions/<solutions_tag>/<area_id>/<capability_id>/<task_id>/solution.json`
  - Task information is included in the solution files
- **Resume tag**: Optional - If provided, goes to `validation/<resume_tag>/` directory
  - For each `<area_id>/<capability_id>/<task_id>/solution.json` in `solutions/<solutions_tag>`, checks if `validation/<resume_tag>/<area_id>/<capability_id>/<task_id>/validation.json` exists
  - If file exists, validation for that task was already completed successfully, so skip it
  - If file doesn't exist, creates `<area_id>/<capability_id>/<task_id>/` subdirectories and generates validation for that task
- **New tag**: If no resume tag provided, generates new tag (validation_tag) for this validation run
  - For each task, creates `validation/<validation_tag>/<area_id>/<capability_id>/<task_id>/validation.json`

### Output: `validation.json` (one per task)

**Stage Output:** ValidationResult dataclass + PipelineMetadata
**Save Function:** `save_validation(validation_result: ValidationResult, metadata: PipelineMetadata, output_path: Path)`

**File Path:** `<output_dir>/<experiment_id>/validation/<validation_tag>/<area_id>/<capability_id>/<task_id>/validation.json`

```json
{
  "metadata": {
    "experiment_id": "r0_10x10",
    "output_base_dir": "agentic_outputs",
    "timestamp": "2025-11-06T14:00:00Z",
    "input_stage_tag": "_20251016_182128",
    "output_stage_tag": "_20251017_091500",
    "resume": false
  },
  "task_id": "task_000",
  "task": "You are advising a client who wants to set up a zero-based budget...",
  "capability": "budget_policy_and_structure",
  "capability_id": "cap_000",
  "area": "Cash Flow & Budget Management",
  "area_id": "area_000",
  "domain": "personal finance",
  "domain_id": "domain_000",
  "verification": true,
  "feedback": "Solution addresses all aspects of the task...",
  "score": 0.95,
  "generation_metadata": {
    "validation_method": "llm_based",
    "criteria": {
      "solution_completeness": true,
      "solution_accuracy": true,
      "reasoning_quality": true
    }
  }
}
```


---
