# ACE Pipeline Standardized Schemas

This document defines the standardized input and output formats for each stage of the ACE pipeline. These schemas ensure consistency across different implementations and enable interoperability between pipeline stages.

## Implementation Approach

**Pipeline Pattern:**
Each stage follows a consistent pattern:
1. **Stage Implementation**: Produces dataclass objects (or lists of dataclasses) + metadata
2. **Save Function**: Takes dataclass objects + metadata → saves to JSON file

**Functions (to be provided):**
- **Save functions**: `save_<stage>_output(data, metadata, output_path)` - Handle JSON serialization, file writing, directory creation
- **Load functions**: `load_<stage>_output(file_path) -> <OutputDataclass>` - Load dataclass objects from JSON files

Dataclasses provide type safety, validation, and clear structure. JSON is the serialization format.

## Pipeline Stages

0. **Experiment Setup** - Initialize experiment and create domain metadata
1. **Area Generation** - Generate domain areas
2. **Capability Generation** - Generate capabilities for each area
3. **Task Generation** - Generate tasks for each capability
4. **Solution Generation** - Generate solutions for each task
5. **Validation** - Validate solutions against tasks

**Note:** Experiment configuration must remain consistent throughout the pipeline. Once set during experiment setup, it should not be changed to avoid inconsistencies.

---

## Directory Structure

All outputs are stored in the following directory structure, organized hierarchically by area and capability for easy resumability, with versioning support:

```
<output_dir>/
  <experiment_id>/
    experiment.json                  # Experiment metadata (all configuration)
    domain.json                      # Domain metadata (contains domain_id)
    areas/
      <tag>/
        areas.json                   # All areas for this experiment run
    <area_id>/
      capabilities/
        <tag>/
          capabilities.json          # All capabilities for this area run
      <capability_id>/
        tasks/
          <tag>/
            tasks.json               # All tasks for this capability run
        solutions/
          <tag>/
            <task_id>_solution.json  # Individual solution files (e.g., task_000_solution.json)
        validation/
          <tag>/
            <task_id>_validation.json  # Validation results per task (e.g., task_000_validation.json)
```

**Example:**
```
agentic_outputs/
  r0_10x10/
    experiment.json                  # Experiment configuration
    domain.json
    areas/
      _20251009_122040/
        areas.json
      _20251010_143022/
        areas.json
    area_000/                           # area_id = "area_000" (first area)
      capabilities/
        _20251009_131252/
          capabilities.json
      cap_000/                          # capability_id = "cap_000" (first capability in area_000)
        tasks/
          _20251014_114358/
            tasks.json
        solutions/
          _20251016_182128/
            task_000_solution.json
            task_001_solution.json
            task_002_solution.json
        validation/
          _20251017_091500/
            task_000_validation.json
            task_001_validation.json
            task_002_validation.json
      cap_001/                          # capability_id = "cap_001" (second capability in area_000)
        tasks/
          _20251014_114358/
            tasks.json
    area_001/                           # area_id = "area_001" (second area)
      capabilities/
        _20251009_131252/
          capabilities.json
      cap_000/                          # capability_id = "cap_000" (first capability in area_001)
        tasks/
          _20251014_114358/
            tasks.json
```

**Directory Naming Rules:**
- `<output_dir>`: Base output directory (e.g., `agentic_outputs`)
- `<experiment_id>`: Experiment identifier (e.g., `r0_10x10`)
- `<tag>`: Timestamp tag in format `_YYYYMMDD_HHMMSS` (e.g., `_20251009_122040`)
  - Generated automatically when a stage is run
  - Allows multiple versions/runs of the same stage
  - Each stage has its own tag (independent versioning)
- `<area_id>`: String identifier in format `area_` + zero-padded 3-digit number (e.g., `area_000`, `area_001`)
  - Format: `area_` prefix + zero-padded 3-digit number (000, 001, 002, ...)
  - Assigned sequentially starting from 000 when areas are generated
  - Example: First area → `area_000`, Second area → `area_001`, etc.
  - Unique within an experiment
  - Used in directory paths for clean, explicit paths
  - Human-readable name stored in `areas.json`
- `<capability_id>`: String identifier in format `cap_` + zero-padded 3-digit number (e.g., `cap_000`, `cap_001`)
  - Format: `cap_` prefix + zero-padded 3-digit number (000, 001, 002, ...)
  - Assigned sequentially starting from 000 within each area when capabilities are generated
  - Example: First capability in area_000 → `cap_000`, Second capability → `cap_001`, etc.
  - Unique within an area (but can repeat across areas, e.g., `area_000/cap_000/` and `area_001/cap_000/`)
  - Used in directory paths for clean, explicit paths
  - Human-readable name stored in `capabilities.json`
- `<task_id>`: String identifier in format `task_` + zero-padded 3-digit number (e.g., `task_000`, `task_001`)
  - Format: `task_` prefix + zero-padded 3-digit number (000, 001, 002, ...)
  - Assigned sequentially starting from 000 within each capability when tasks are generated
  - Example: First task in cap_000 → `task_000`, Second task → `task_001`, etc.
  - Unique within a capability


**File Naming:**
- Experiment: `experiment.json` (no versioning, one file per experiment, contains all configuration)
- Domain: `domain.json` (no versioning, one file per experiment)
- Areas: `areas.json` (versioned by tag: `areas/<tag>/areas.json`)
- Capabilities: `capabilities.json` (versioned by tag: `<area_id>/capabilities/<tag>/capabilities.json`)
- Tasks: `tasks.json` (versioned by tag: `<area_id>/<capability_id>/tasks/<tag>/tasks.json`)
- Solutions: `<task_id>_solution.json` (versioned by tag: `<area_id>/<capability_id>/solutions/<tag>/<task_id>_solution.json`, e.g., `task_000_solution.json`)
- Validation: `<task_id>_validation.json` (versioned by tag: `<area_id>/<capability_id>/validation/<tag>/<task_id>_validation.json`, e.g., `task_000_validation.json`)

**Resumability Benefits:**
- Each area has its own directory - easy to see which areas are processed
- Each capability has its own directory - easy to see which capabilities are complete
- Missing files are immediately visible in the directory structure
- Can resume from any area or capability by checking if directory/files exist
- Tags allow multiple runs/versions to coexist
- Can check latest tag to determine most recent run

**Versioning Strategy:**
- Each stage generates a new tag when run (format: `_YYYYMMDD_HHMMSS`)
- Tags are independent per stage (areas can have different tag than capabilities)
- **Input tags**: Each stage requires tag(s) from previous stage(s) to load input data
  - Stage 1 (Areas): No input tag (uses domain.json)
  - Stage 2 (Capabilities): Requires `areas_tag` from Stage 1
  - Stage 3 (Tasks): Requires `capabilities_tag` from Stage 2
  - Stage 4 (Solutions): Requires `tasks_tag` from Stage 3
  - Stage 5 (Validation): Requires both `tasks_tag` (Stage 3) and `solutions_tag` (Stage 4)
- **Resume tags**: Optional - If provided, stage loads existing output and continues incomplete generation
  - Checks for existing files with resume tag
  - Identifies which items are incomplete (e.g., missing capabilities, tasks, solutions)
  - Continues generation only for incomplete items
  - Preserves existing completed items
- **New tags**: If no resume tag provided, generates new tag and creates fresh output

---

## Stage 0: Experiment Setup

### Input
All inputs come from the configuration file. Important fields:
- **Experiment ID**: String - The experiment identifier (e.g., "r0_10x10")
- **Domain Name**: String - The domain name (e.g., "personal finance", "mathematics")
- **Description**: String (optional) - Domain description
- **Configuration**: Dict - Complete experiment configuration (all config sections: `global_cfg`, `debate_cfg`, `agents`, `area_generation`, `capability_generation`, `task_generation`, `task_solver`, `exp_cfg`, etc.)

### Tag Handling
- **No input tag required** (first stage)
- **No resume tag** - Always creates new files (overwrites if exists)

### Outputs

This stage creates two files:
1. `experiment.json` - Experiment metadata and complete configuration
2. `domain.json` - Domain metadata

#### Output 1: `experiment.json`

**Stage Output:** Experiment dataclass + PipelineMetadata
**Save Function:** `save_experiment_output(experiment: Experiment, metadata: PipelineMetadata, output_path: Path)`

**Implementation:**
- Stage creates `Experiment` dataclass object with experiment information and configuration
- Stage creates `PipelineMetadata` dataclass object with metadata
- Pass both to `save_experiment_output(experiment, metadata, output_path)` which creates `ExperimentMetadata` dataclass, serializes to JSON, and writes to file

**File Path:** `<output_dir>/<experiment_id>/experiment.json`

```json
{
  "metadata": {
    "experiment_id": "r0_10x10",
    "stage": "experiment_setup",
    "timestamp": "2025-11-06T12:00:00Z"
  },
  "experiment": {
    "experiment_id": "r0_10x10",
    "domain": "personal finance",
    "domain_id": "personal_finance",
    "configuration": {
      "global_cfg": {
        "domain": "personal finance",
        "output_dir": "agentic_outputs"
      },
      "debate_cfg": {
        "max_round": 5
      },
      "agents": {
        "scientist_a": {
          "model_name": "gpt-5",
          "seed": 8
        },
        "scientist_b": {
          "model_name": "gemini-2.5-pro",
          "seed": 88
        },
        "moderator": {
          "model_name": "claude-opus-4-1-20250805",
          "seed": 888
        }
      },
      "area_generation": {
        "num_areas": 10
      },
      "capability_generation": {
        "num_capabilities_per_area": 5
      },
      "task_generation": {
        "num_final_problems_per_capability": 3,
        "buffer_param": 2,
        "max_rounds": 3
      },
      "task_solver": {
        "max_tasks": 0,
        "max_rounds": 1
      },
      "exp_cfg": {
        "exp_id": "r0_10x10"
      }
    }
  }
}
```

**Schema (JSON representation of ExperimentMetadata dataclass):**
- `metadata`: Object containing pipeline metadata
  - `experiment_id`: String (required, experiment identifier)
  - `stage`: String (required, value: "experiment_setup")
  - `timestamp`: String (required, ISO 8601 format)
- `experiment`: Object containing experiment information
  - `experiment_id`: String (required, experiment identifier)
  - `domain`: String (required, human-readable domain name)
  - `domain_id`: String (required, slugified domain identifier)
  - `configuration`: Object (required, all configuration used for this experiment)
    - Contains all config sections: `global_cfg`, `debate_cfg`, `agents`, `area_generation`, `capability_generation`, `task_generation`, `task_solver`, `exp_cfg`, etc.
    - Structure matches the input configuration format exactly

#### Output 2: `domain.json`

**Stage Output:** Domain dataclass object + PipelineMetadata
**Save Function:** `save_domain_output(domain: Domain, metadata: PipelineMetadata, output_path: Path)`

**Implementation:**
- Stage creates `Domain` dataclass object with domain information
- Stage creates `PipelineMetadata` dataclass object with metadata
- Pass both to `save_domain_output(domain, metadata, output_path)` which serializes to JSON and writes to file

**File Path:** `<output_dir>/<experiment_id>/domain.json`

```json
{
  "metadata": {
    "domain": "personal finance",
    "domain_id": "personal_finance",
    "stage": "experiment_setup",
    "timestamp": "2025-11-06T12:00:00Z"
  },
  "domain": {
    "name": "personal finance",
    "domain_id": "personal_finance",
    "description": "Personal finance domain covering budgeting, investing, retirement planning, etc."
  }
}
```

**Schema (JSON representation of Domain dataclass):**
- `metadata`: Object containing pipeline metadata
  - `domain`: String (required, human-readable domain name)
  - `domain_id`: String (required, slugified domain identifier)
  - `stage`: String (required, value: "experiment_setup")
  - `timestamp`: String (required, ISO 8601 format)
- `domain`: Object containing domain information
  - `name`: String (required, human-readable domain name)
  - `domain_id`: String (required, slugified identifier, filesystem-safe)
  - `description`: String (optional, domain description)

---

## Stage 1: Area Generation

### Input
- **Domain**: Domain object (from Stage 0) - Loaded from `domain.json`
- **Configuration**: Dict - Stage-specific configuration (e.g., `num_areas`)

### Tag Handling
- **Input tag**: Not applicable (uses domain.json which has no tag)
- **Resume tag**: Optional - If provided, loads from `areas/<resume_tag>/areas.json` and continues incomplete area generation
- **New tag**: If no resume tag provided, generates new tag (format: `_YYYYMMDD_HHMMSS`) and creates `areas/<new_tag>/areas.json`

### Output: `areas.json`

**Stage Output:** List[Area] dataclasses + PipelineMetadata
**Save Function:** `save_areas_output(areas: List[Area], metadata: PipelineMetadata, output_path: Path)`

**Implementation:**
- Stage generates list of `Area` dataclass objects
- Stage creates `PipelineMetadata` dataclass object with metadata
- Pass both to `save_areas_output(areas, metadata, output_path)` which creates `AreaGenerationOutput` dataclass, serializes to JSON, and writes to file

**File Path:** `<output_dir>/<experiment_id>/areas/<tag>/areas.json`
```json
{
  "metadata": {
    "domain": "personal finance",
    "domain_id": "personal_finance",
    "stage": "area_generation",
    "tag": "_20251009_122040",
    "timestamp": "2025-11-06T12:00:00Z"
  },
  "areas": [
    {
      "name": "Cash Flow & Budget Management",
      "area_id": "area_000",
      "description": "Design and monitor budgets using various methodologies..."
    }
  ]
}
```

**Schema (JSON representation of AreaGenerationOutput dataclass):**
- `metadata`: Object containing pipeline metadata
  - `domain`: String (required, human-readable domain name)
  - `domain_id`: String (required, slugified domain identifier)
  - `stage`: String (required, value: "area_generation")
  - `tag`: String (required, format `_YYYYMMDD_HHMMSS`, the tag used for this run's output)
  - `timestamp`: String (required, ISO 8601 format)
  - Note: No `input_tag` field (Stage 1 uses `domain.json` which has no tag)
- `areas`: Array of Area objects
  - `name`: String (required, human-readable name, unique within domain)
  - `area_id`: String (required, format `area_` + zero-padded 3-digit number, unique within experiment)
  - `description`: String (required, detailed description)

---

## Stage 2: Capability Generation

### Input
- **Areas**: Array of Area objects (from Stage 1) - Loaded from `areas/<areas_tag>/areas.json`
- **Areas tag**: String - Tag from Stage 1 output (e.g., `_20251009_122040`)
- **Configuration**: Dict - Stage-specific configuration (e.g., `num_capabilities_per_area`)

### Tag Handling
- **Input tag**: Required - `areas_tag` from Stage 1 output (e.g., `_20251009_122040`)
  - Loads areas from `areas/<areas_tag>/areas.json`
- **Resume tag**: Optional - If provided, loads from `<area_id>/capabilities/<resume_tag>/capabilities.json` for each area and continues incomplete capability generation
- **New tag**: If no resume tag provided, generates new tag (format: `_YYYYMMDD_HHMMSS`) and creates `<area_id>/capabilities/<new_tag>/capabilities.json` for each area

### Output: `capabilities.json` (one per area)

**Stage Output:** List[Capability] dataclasses + PipelineMetadata
**Save Function:** `save_capabilities_output(capabilities: List[Capability], metadata: PipelineMetadata, output_path: Path)`

**Implementation:**
- Stage generates list of `Capability` dataclass objects for an area
- Stage creates `PipelineMetadata` dataclass object with metadata (includes area_id)
- Pass both to `save_capabilities_output(capabilities, metadata, output_path)` which creates `CapabilityGenerationOutput` dataclass, serializes to JSON, and writes to file

**File Path:** `<output_dir>/<experiment_id>/<area_id>/capabilities/<tag>/capabilities.json`
Where `<area_id>` is a string in format `area_` + zero-padded 3-digit number (e.g., `area_000`, `area_001`)
```json
{
  "metadata": {
    "domain": "personal finance",
    "domain_id": "personal_finance",
    "area": "Cash Flow & Budget Management",
    "area_id": "area_000",
    "stage": "capability_generation",
    "input_tag": "_20251009_122040",
    "tag": "_20251009_131252",
    "timestamp": "2025-11-06T12:30:00Z"
  },
  "capabilities": [
    {
      "name": "budget_policy_and_structure",
      "capability_id": "cap_000",
      "description": "Define the strategic framework and methodology for budgeting...",
      "area": "Cash Flow & Budget Management",
      "area_id": "area_000"
    }
  ]
}
```

**Schema (JSON representation of CapabilityGenerationOutput dataclass):**
- `metadata`: Object containing pipeline metadata
  - `domain`: String (required, human-readable domain name)
  - `domain_id`: String (required, slugified domain identifier)
  - `area`: String (required, human-readable area name, must match an area name from Stage 1)
  - `area_id`: String (required, format `area_` + zero-padded 3-digit number, must match an area_id from Stage 1)
  - `stage`: String (required, value: "capability_generation")
  - `input_tag`: String (required, format `_YYYYMMDD_HHMMSS`, the areas tag from Stage 1 used as input)
  - `tag`: String (required, format `_YYYYMMDD_HHMMSS`, the tag used for this run's output)
  - `timestamp`: String (required, ISO 8601 format)
- `capabilities`: Array of Capability objects
  - `name`: String (required, human-readable name, unique within area)
  - `capability_id`: String (required, format `cap_` + zero-padded 3-digit number starting from 000, unique within area)
  - `description`: String (required, detailed description)
  - `area`: String (required, human-readable area name, must match parent area name)
  - `area_id`: String (required, format `area_` + zero-padded 3-digit number, must match parent area_id)

---

## Stage 3: Task Generation

### Input
- **Capabilities**: Array of Capability objects (from Stage 2) - Loaded from `<area_id>/capabilities/<capabilities_tag>/capabilities.json`
- **Capabilities tag**: String - Tag from Stage 2 output (e.g., `_20251009_131252`)
- **Configuration**: Dict - Stage-specific configuration (e.g., `num_final_problems_per_capability`)

### Tag Handling
- **Input tag**: Required - `capabilities_tag` from Stage 2 output (e.g., `_20251009_131252`)
  - Loads capabilities from `<area_id>/capabilities/<capabilities_tag>/capabilities.json` for each area
- **Resume tag**: Optional - If provided, loads from `<area_id>/<capability_id>/tasks/<resume_tag>/tasks.json` for each capability and continues incomplete task generation
- **New tag**: If no resume tag provided, generates new tag (format: `_YYYYMMDD_HHMMSS`) and creates `<area_id>/<capability_id>/tasks/<new_tag>/tasks.json` for each capability

### Output: `tasks.json` (one per capability)

**Stage Output:** Dict[str, Task] (mapping task_id to Task dataclass) + PipelineMetadata
**Save Function:** `save_tasks_output(tasks: Dict[str, Task], metadata: PipelineMetadata, output_path: Path)`

**Implementation:**
- Stage generates dictionary mapping `task_id` strings to `Task` dataclass objects
- Stage creates `PipelineMetadata` dataclass object with metadata (includes area_id, capability_id)
- Pass both to `save_tasks_output(tasks, metadata, output_path)` which creates `TaskGenerationOutput` dataclass, serializes to JSON, and writes to file

**File Path:** `<output_dir>/<experiment_id>/<area_id>/<capability_id>/tasks/<tag>/tasks.json`
Where `<area_id>` is format `area_` + zero-padded 3-digit number, `<capability_id>` is format `cap_` + zero-padded 3-digit number (e.g., `area_000/cap_000/`)
```json
{
  "metadata": {
    "domain": "personal finance",
    "domain_id": "personal_finance",
    "area": "Cash Flow & Budget Management",
    "area_id": "area_000",
    "capability": "budget_policy_and_structure",
    "capability_id": "cap_000",
    "stage": "task_generation",
    "input_tag": "_20251009_131252",
    "tag": "_20251014_114358",
    "timestamp": "2025-11-06T13:00:00Z"
  },
  "tasks": {
    "task_000": {
      "task": "You are advising a client who wants to set up a zero-based budget...",
      "capability_id": "cap_000"
    },
    "task_001": {
      "task": "A family of four needs to restructure their budget...",
      "capability_id": "cap_000"
    }
  }
}
```

**Schema (JSON representation of TaskGenerationOutput dataclass):**
- `metadata`: Object containing pipeline metadata
  - `domain`: String (required, human-readable domain name)
  - `domain_id`: String (required, slugified domain identifier)
  - `area`: String (required, human-readable area name, must match an area name from Stage 1)
  - `area_id`: String (required, format `area_` + zero-padded 3-digit number, must match an area_id from Stage 1)
  - `capability`: String (required, human-readable capability name, must match a capability name from Stage 2)
  - `capability_id`: String (required, format `cap_` + zero-padded 3-digit number, must match a capability_id from Stage 2)
  - `stage`: String (required, value: "task_generation")
  - `input_tag`: String (required, format `_YYYYMMDD_HHMMSS`, the capabilities tag from Stage 2 used as input)
  - `tag`: String (required, format `_YYYYMMDD_HHMMSS`, the tag used for this run's output)
  - `timestamp`: String (required, ISO 8601 format)
- `tasks`: Object mapping task_id to Task object
  - `task_id`: String (required, format: `task_` + zero-padded 3-digit number, unique within capability)
  - `task`: String (required, the task/problem text)
  - `capability_id`: String (required, format `cap_` + zero-padded 3-digit number, must match parent capability_id)

---

## Stage 4: Solution Generation

### Input
- **Tasks**: Object mapping task_id to Task objects (from Stage 3) - Loaded from `<area_id>/<capability_id>/tasks/<tasks_tag>/tasks.json`
- **Tasks tag**: String - Tag from Stage 3 output (e.g., `_20251014_114358`)
- **Configuration**: Dict - Stage-specific configuration (e.g., `max_rounds`)

### Tag Handling
- **Input tag**: Required - `tasks_tag` from Stage 3 output (e.g., `_20251014_114358`)
  - Loads tasks from `<area_id>/<capability_id>/tasks/<tasks_tag>/tasks.json` for each capability
- **Resume tag**: Optional - If provided, checks for existing solutions in `<area_id>/<capability_id>/solutions/<resume_tag>/<task_id>_solution.json` and continues incomplete solution generation
- **New tag**: If no resume tag provided, generates new tag (format: `_YYYYMMDD_HHMMSS`) and creates `<area_id>/<capability_id>/solutions/<new_tag>/<task_id>_solution.json` for each task

### Output: `<task_id>_solution.json` (one per task)

**Stage Output:** TaskSolution dataclass + List[AgentSolution] dataclasses + PipelineMetadata
**Save Function:** `save_solution_output(task_solution: TaskSolution, all_solutions: List[AgentSolution], metadata: PipelineMetadata, output_path: Path)`

**Implementation:**
- Stage generates `TaskSolution` dataclass object with solution information
- Stage generates list of `AgentSolution` dataclass objects
- Stage creates `PipelineMetadata` dataclass object with metadata (includes area_id, capability_id, task_id)
- Pass all to `save_solution_output(task_solution, all_solutions, metadata, output_path)` which creates `SolutionGenerationOutput` dataclass, serializes to JSON, and writes to file

**File Path:** `<output_dir>/<experiment_id>/<area_id>/<capability_id>/solutions/<tag>/<task_id>_solution.json`
Where `<area_id>` is format `area_` + zero-padded 3-digit number, `<capability_id>` is format `cap_` + zero-padded 3-digit number, `<task_id>` is format `task_` + zero-padded 3-digit number (e.g., `task_000_solution.json`)
```json
{
  "metadata": {
    "domain": "personal finance",
    "domain_id": "personal_finance",
    "area": "Cash Flow & Budget Management",
    "area_id": "area_000",
    "capability_name": "budget_policy_and_structure",
    "capability_id": "cap_000",
    "task_id": "task_000",
    "stage": "solution_generation",
    "input_tag": "_20251014_114358",
    "tag": "_20251016_182128",
    "timestamp": "2025-11-06T13:30:00Z"
  },
  "task_id": "task_000",
  "capability_name": "budget_policy_and_structure",
  "capability_id": "cap_000",
  "area_name": "Cash Flow & Budget Management",
  "area_id": "area_000",
  "problem": "You are advising a client who wants to set up a zero-based budget...",
  "solution": "The optimal approach is to use a zero-based budgeting methodology...",
  "numerical_answer": "{\"budget_allocation\": {...}}",
  "reasoning": "Both agents agreed on the zero-based approach because...",
  "consensus_reached": true,
  "total_rounds": 2,
  "all_solutions": [
    {
      "agent_id": "A",
      "task_id": "task_000",
      "thought": "I need to analyze the client's financial situation...",
      "final_answer": "{\"recommendation\": {...}}",
      "numerical_answer": "null",
      "round_number": "0"
    },
    {
      "agent_id": "B",
      "task_id": "task_000",
      "thought": "The client's income and expenses suggest...",
      "final_answer": "{\"recommendation\": {...}}",
      "numerical_answer": "null",
      "round_number": "0"
    }
  ]
}
```

**Schema (JSON representation of SolutionGenerationOutput dataclass):**
- `metadata`: Object containing pipeline metadata
  - `domain`: String (required, human-readable domain name)
  - `domain_id`: String (required, slugified domain identifier)
  - `area`: String (required, human-readable area name, must match an area name from Stage 1)
  - `area_id`: String (required, format `area_` + zero-padded 3-digit number, must match an area_id from Stage 1)
  - `capability_name`: String (required, human-readable capability name, must match a capability name from Stage 2)
  - `capability_id`: String (required, format `cap_` + zero-padded 3-digit number, must match a capability_id from Stage 2)
  - `task_id`: String (required, must match a task_id from Stage 3)
  - `stage`: String (required, value: "solution_generation")
  - `input_tag`: String (required, format `_YYYYMMDD_HHMMSS`, the tasks tag from Stage 3 used as input)
  - `tag`: String (required, format `_YYYYMMDD_HHMMSS`, the tag used for this run's output)
  - `timestamp`: String (required, ISO 8601 format)
- `task_id`: String (required, must match metadata.task_id)
- `capability_name`: String (required, human-readable capability name, must match metadata.capability_name)
- `capability_id`: String (required, format `cap_` + zero-padded 3-digit number, must match metadata.capability_id)
- `area_name`: String (required, human-readable area name, must match metadata.area)
- `area_id`: String (required, format `area_` + zero-padded 3-digit number, must match metadata.area_id)
- `problem`: String (required, the task text from Stage 3)
- `solution`: String (required, the final consensus solution)
- `numerical_answer`: String (optional, JSON string with numerical results)
- `reasoning`: String (required, explanation of consensus or disagreement)
- `consensus_reached`: Boolean (required, whether agents reached consensus)
- `total_rounds`: Integer (required, number of debate rounds)
- `all_solutions`: Array of AgentSolution objects
  - `agent_id`: String (required, "A" or "B")
  - `task_id`: String (required, must match parent task_id)
  - `thought`: String (required, agent's reasoning)
  - `final_answer`: String (required, JSON string with agent's solution)
  - `numerical_answer`: String (optional, JSON string or "null")
  - `round_number`: String (required, round number as string)

---

## Stage 5: Validation

### Input
- **Tasks**: Object mapping task_id to Task objects (from Stage 3) - Loaded from `<area_id>/<capability_id>/tasks/<tasks_tag>/tasks.json`
- **Tasks tag**: String - Tag from Stage 3 output (e.g., `_20251014_114358`)
- **Solutions**: Object mapping task_id to TaskSolution objects (from Stage 4) - Loaded from `<area_id>/<capability_id>/solutions/<solutions_tag>/<task_id>_solution.json`
- **Solutions tag**: String - Tag from Stage 4 output (e.g., `_20251016_182128`)
- **Configuration**: Dict - Validation criteria

### Tag Handling
- **Input tags**: Required - Both `tasks_tag` (from Stage 3) and `solutions_tag` (from Stage 4)
  - Loads tasks from `<area_id>/<capability_id>/tasks/<tasks_tag>/tasks.json`
  - Loads solutions from `<area_id>/<capability_id>/solutions/<solutions_tag>/<task_id>_solution.json`
- **Resume tag**: Optional - If provided, checks for existing validations in `<area_id>/<capability_id>/validation/<resume_tag>/<task_id>_validation.json` and continues incomplete validation
- **New tag**: If no resume tag provided, generates new tag (format: `_YYYYMMDD_HHMMSS`) and creates `<area_id>/<capability_id>/validation/<new_tag>/<task_id>_validation.json` for each task

### Output: `<task_id>_validation.json` (one per task)

**Stage Output:** ValidationResult dataclass + ValidationCriteria dataclass + PipelineMetadata
**Save Function:** `save_validation_output(validation_result: ValidationResult, criteria: ValidationCriteria, metadata: PipelineMetadata, output_path: Path)`

**Implementation:**
- Stage generates `ValidationResult` dataclass object with validation information
- Stage generates `ValidationCriteria` dataclass object with criteria results
- Stage creates `PipelineMetadata` dataclass object with metadata (includes area_id, capability_id, task_id)
- Pass all to `save_validation_output(validation_result, criteria, metadata, output_path)` which creates `ValidationOutput` dataclass, serializes to JSON, and writes to file

**File Path:** `<output_dir>/<experiment_id>/<area_id>/<capability_id>/validation/<tag>/<task_id>_validation.json`
Where `<area_id>` is format `area_` + zero-padded 3-digit number, `<capability_id>` is format `cap_` + zero-padded 3-digit number, `<task_id>` is format `task_` + zero-padded 3-digit number (e.g., `task_000_validation.json`)
```json
{
  "metadata": {
    "domain": "personal finance",
    "domain_id": "personal_finance",
    "area": "Cash Flow & Budget Management",
    "area_id": "area_000",
    "capability": "budget_policy_and_structure",
    "capability_id": "cap_000",
    "task_id": "task_000",
    "stage": "validation",
    "input_tags": {
      "tasks_tag": "_20251014_114358",
      "solutions_tag": "_20251016_182128"
    },
    "tag": "_20251017_091500",
    "timestamp": "2025-11-06T14:00:00Z"
  },
  "task_id": "task_000",
  "capability_name": "budget_policy_and_structure",
  "capability_id": "cap_000",
  "is_valid": true,
  "validation_score": 0.95,
  "criteria": {
    "solution_completeness": true,
    "solution_accuracy": true,
    "reasoning_quality": true,
    "consensus_quality": true
  },
  "feedback": "Solution addresses all aspects of the task...",
  "errors": []
}
```

**Schema (JSON representation of ValidationOutput dataclass):**
- `metadata`: Object containing pipeline metadata
  - `domain`: String (required, human-readable domain name)
  - `domain_id`: String (required, slugified domain identifier)
  - `area`: String (required, human-readable area name, must match an area name from Stage 1)
  - `area_id`: String (required, format `area_` + zero-padded 3-digit number, must match an area_id from Stage 1)
  - `capability`: String (required, human-readable capability name, must match a capability name from Stage 2)
  - `capability_id`: String (required, format `cap_` + zero-padded 3-digit number, must match a capability_id from Stage 2)
  - `task_id`: String (required, format `task_` + zero-padded 3-digit number, must match a task_id from Stage 3)
  - `stage`: String (required, value: "validation")
  - `input_tags`: Object (required, contains the input tags used)
    - `tasks_tag`: String (required, format `_YYYYMMDD_HHMMSS`, the tasks tag from Stage 3 used as input)
    - `solutions_tag`: String (required, format `_YYYYMMDD_HHMMSS`, the solutions tag from Stage 4 used as input)
  - `tag`: String (required, format `_YYYYMMDD_HHMMSS`, the tag used for this run's output)
  - `timestamp`: String (required, ISO 8601 format)
- `task_id`: String (required, must match metadata.task_id)
- `capability_name`: String (required, human-readable capability name, must match metadata.capability)
- `capability_id`: String (required, format `cap_` + zero-padded 3-digit number, must match metadata.capability_id)
- `is_valid`: Boolean (required, overall validation status)
- `validation_score`: Float (required, 0.0 to 1.0)
- `criteria`: Object with boolean criteria (ValidationCriteria dataclass)
  - `solution_completeness`: Boolean (required)
  - `solution_accuracy`: Boolean (required)
  - `reasoning_quality`: Boolean (required)
  - `consensus_quality`: Boolean (required)
- `feedback`: String (required, detailed feedback)
- `errors`: Array of strings (required, list of errors if any)

---

## ID Assignment Rules

All IDs are string identifiers with explicit prefixes and sequential numbering:

- **Area IDs**: Format `area_` + zero-padded 3-digit number (e.g., `area_000`, `area_001`)
  - Assigned sequentially starting from `area_000` when areas are generated
  - Unique within an experiment

- **Capability IDs**: Format `cap_` + zero-padded 3-digit number (e.g., `cap_000`, `cap_001`)
  - Assigned sequentially starting from `cap_000` within each area when capabilities are generated
  - Unique within an area (but can repeat across areas, e.g., `area_000/cap_000/` and `area_001/cap_000/`)

- **Task IDs**: Format `task_` + zero-padded 3-digit number (e.g., `task_000`, `task_001`)
  - Assigned sequentially starting from `task_000` within each capability when tasks are generated
  - Unique within a capability

**ID Properties:**
- String type with explicit prefixes (`area_`, `cap_`, `task_`)
- Sequential assignment (000, 001, 002, ...)
- Zero-padded 3-digit numbers ensure proper sorting
- Stable once assigned (don't change if items are reordered)
- Human-readable names are stored alongside IDs in JSON files
