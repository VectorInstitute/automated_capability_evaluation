# ACE

ACE (Active learning for Capability Evaluation) is a novel framework that uses active learning and powerful language models to automate fine-grained evaluation of foundation models. It enables scalable, adaptive testing that uncovers strengths and weaknesses beyond static benchmarks.

## Installing dependencies

The development environment can be set up using
[poetry](https://python-poetry.org/docs/#installation). Hence, make sure it is
installed and then run:

```bash
python3 -m poetry install
source $(poetry env info --path)/bin/activate
```

In order to install dependencies for testing (codestyle, unit tests, integration tests),
run:

```bash
python3 -m poetry install --with test
```

#### [Optional] Google Cloud Authentication

The capability evaluation logs (evaluated using [Inspect](https://inspect.aisi.org.uk/)) are stored in a GCP bucket. Use the following command to log in using your GCP account:

```bash
gcloud auth application-default login
```

## Run pipeline

### Configuration

1. Set environment variables:

- OPENAI_API_KEY
- GOOGLE_API_KEY - To use LLMs provided by Google
- ANTHROPIC_API_KEY - To use LLMs provided by Anthropic
- Rate limit vars (default values given):
    - RATE_LIMIT_CALLS=5
    - RATE_LIMIT_PERIOD=60
- LangSmith tracing vars:
    - LANGSMITH_TRACING=true
    - LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
    - LANGSMITH_API_KEY=<langsmith_api_key>
    - LANGSMITH_PROJECT="automated_capability_evaluation"
- GCP env vars:
    - GOOGLE_CLOUD_PROJECT=<project_id>

2. Modify `src/cfg/run_cfg.yaml`, if required.

### Base Pipeline

The base (non-agentic) pipeline consists of multiple stages that can be run sequentially or individually:

- **Stage 0**: Experiment and domain setup
- **Stage 1**: Area generation
- **Stage 2**: Capability generation and filtering
- **Stage 3**: Task generation (questions with options)
- **Stage 4**: Solution generation (determine correct answers)
- **Stage 5**: Task validation

#### Run All Stages

```bash
python -m src.run_base_pipeline stage=all
```

#### Run Individual Stages

```bash
# Stage 0: Setup
python -m src.run_base_pipeline stage=0

# Stage 1: Generate areas
python -m src.run_base_pipeline stage=1

# Stage 2: Generate capabilities (requires areas_tag from Stage 1)
python -m src.run_base_pipeline stage=2 areas_tag=_YYYYMMDD_HHMMSS

# Stage 3: Generate tasks (requires capabilities_tag from Stage 2)
python -m src.run_base_pipeline stage=3 capabilities_tag=_YYYYMMDD_HHMMSS

# Stage 4: Generate solutions (requires tasks_tag from Stage 3)
python -m src.run_base_pipeline stage=4 tasks_tag=_YYYYMMDD_HHMMSS

# Stage 5: Validate tasks (requires solution_tag from Stage 4)
python -m src.run_base_pipeline stage=5 solution_tag=_YYYYMMDD_HHMMSS
```

#### Resume from Existing Runs

```bash
# Resume Stage 2 from existing capabilities_tag
python -m src.run_base_pipeline stage=2 areas_tag=_YYYYMMDD_HHMMSS capabilities_tag=_YYYYMMDD_HHMMSS

# Resume Stage 3 from existing tasks_tag
python -m src.run_base_pipeline stage=3 capabilities_tag=_YYYYMMDD_HHMMSS tasks_tag=_YYYYMMDD_HHMMSS

# Resume Stage 4 from existing solution_tag
python -m src.run_base_pipeline stage=4 tasks_tag=_YYYYMMDD_HHMMSS solution_tag=_YYYYMMDD_HHMMSS

# Resume Stage 5 from existing validation_tag
python -m src.run_base_pipeline stage=5 solution_tag=_YYYYMMDD_HHMMSS validation_tag=_YYYYMMDD_HHMMSS
```

### Evaluation of subject LLM on generated capabilities

Evaluates the subject LLM on the generated capabilities and calculates a score for each.

```bash
python -m src.run_evaluation
```

### Capability selection/generation using active learning

Utilize the capability and the corresponding subject LLM score to select or generate a new capability.

```bash
python -m src.run_lbo
```
### Agentic Generation Scripts

These scripts implement the multi-agent debate workflow for automated generation of areas, capabilities, tasks, and solutions.
All configurable parameters are defined in `src/cfg/agentic_config.yaml`.

#### Understanding Pipeline Tags

The pipeline uses **auto-generated tags** to organize outputs from each step. Understanding how tags work is essential for running the pipeline:

- **Tag Format**: Tags are automatically generated timestamps in the format `_YYYYMMDD_HHMMSS` (e.g., `_20251104_143022`)
- **Auto-Generation**: When you run a step (e.g., Generate Areas), the script automatically creates a tag and includes it in the output path
- **Finding Tags**: After running a step, check the console output or the output directory to see the generated tag. The tag appears in the file path where outputs are saved
- **Using Tags**: To run the next step in the pipeline, you need to specify the tag from the previous step's output:
  - Step 2 (Generate Capabilities) needs `areas_tag` from Step 1
  - Step 3 (Generate Tasks) needs `capabilities_tag` from Step 2
  - Step 4 (Generate Solutions) needs `tasks_tag` from Step 3

**Example Workflow**:
1. Run `python -m src.agentic_area_generator` → outputs to `.../areas/_20251104_143022/areas.json`
2. Use the tag `_20251104_143022` in the next step:
   ```bash
   python -m src.agentic_capability_generator pipeline_tags.areas_tag=_20251104_143022
   ```
3. The capability generator outputs to `.../capabilities/_20251104_150315/...`
4. Use this new tag for the next step, and so on.

---

#### 1. Generate Areas
Generate domain areas using the scientist–moderator debate system:
```bash
python -m src.agentic_area_generator
```

This step auto-generates a tag (e.g., `_20251104_143022`) and outputs the results to:

**Output location:**
```
~/<output_dir>/<domain>/<exp_id>/areas/<areas_tag>/areas.json
```
Where:
- `<output_dir>` comes from `global_cfg.output_dir`
- `<domain>` comes from `global_cfg.domain` (spaces replaced with underscores)
- `<exp_id>` comes from `exp_cfg.exp_id`
- `<areas_tag>` is the auto-generated tag for this run (use this tag in Step 2)

#### 2. Generate Capabilities
Generate capabilities for each area:
```bash
# Use the areas_tag from Step 1 (Generate Areas) output
python -m src.agentic_capability_generator pipeline_tags.areas_tag=_YYYYMMDD_HHMMSS pipeline_tags.resume_capabilities_tag=_YYYYMMDD_HHMMSS
```

**Options:**
- `pipeline_tags.areas_tag` specifies which set of areas to use when generating capabilities. This should be the `<areas_tag>` from the output of Step 1 (Generate Areas).
- `pipeline_tags.resume_capabilities_tag` (optional) resumes a previous capability generation run.

This step auto-generates a new tag for the capabilities output.

**Output location:**
```
~/<output_dir>/<domain>/<exp_id>/capabilities/<capabilities_tag>/<area>/capabilities.json
```
Where:
- `<capabilities_tag>` is the auto-generated tag for this run (use this tag in Step 3)


#### 3. Generate Tasks
Generate evaluation tasks for a specific capabilities tag:
```bash
# Use the capabilities_tag from Step 2 (Generate Capabilities) output
python -m src.agentic_task_generator pipeline_tags.capabilities_tag=_YYYYMMDD_HHMMSS pipeline_tags.resume_tasks_tag=_YYYYMMDD_HHMMSS
```

**Options:**
- `pipeline_tags.capabilities_tag` specifies which set of capabilities to use when generating tasks. This should be the `<capabilities_tag>` from the output of Step 2 (Generate Capabilities).
- `pipeline_tags.resume_tasks_tag` (optional) resumes a previous task generation run.

This step auto-generates a new tag for the tasks output.

**Output location:**
```
~/<output_dir>/<domain>/<exp_id>/tasks/<tasks_tag>/[<area>]-[<capability>]/tasks.json
```
Where:
- `<tasks_tag>` is the auto-generated tag for this run (use this tag in Step 4)

#### 4. Generate Solutions
Solve generated tasks using the multi-agent debate system:
```bash
# Use the tasks_tag from Step 3 (Generate Tasks) output
python -m src.agentic_task_solver pipeline_tags.tasks_tag=_YYYYMMDD_HHMMSS pipeline_tags.resume_solutions_tag=_YYYYMMDD_HHMMSS
```

**Options:**
- `pipeline_tags.tasks_tag` specifies which set of tasks to solve. This should be the `<tasks_tag>` from the output of Step 3 (Generate Tasks).
- `pipeline_tags.resume_solutions_tag` (optional) resumes a previous solution generation run.

This step auto-generates a new tag for the solutions output.

**Output location:**
```
~/<output_dir>/<domain>/<exp_id>/task_solutions/<solutions_tag>/[<area>]-[<capability>]/<task_id>_solution.json
```
Where:
- `<solutions_tag>` is the auto-generated tag for this run

### Wikipedia-Based Analysis Tools

Tools for extracting, processing, and matching mathematical capabilities from Wikipedia. All prompts are centralized in `wikipedia/prompts.py`.

#### Wikipedia Glossary Scraper

Scrapes Wikipedia's "Glossary of areas of mathematics", extracts capability descriptions, and generates summaries with LLM-powered categorization.

```bash
cd wikipedia
python wikipedia_scraper.py
```

Outputs JSON files to `wikipedia/pages/` containing `capability_name`, `description`, `summary`, `area`, `url`, and `timestamp`.

#### Wikipedia-Generated Capability Matcher

Matches Wikipedia capabilities with generated capabilities using LLM-based similarity analysis. Supports bidirectional matching.

Configure `wikipedia/cfg/wiki_vs_generated.yaml`:
- `data_cfg.wikipedia_pages_dir`: Wikipedia pages directory
- `data_cfg.generated_dir`: Generated capabilities directory
- `processing_cfg.match_direction`: `generated_to_wikipedia` or `wikipedia_to_generated`

```bash
cd wikipedia
python wiki_vs_generated.py
```

#### Dataset Question Categorizer

Categorizes questions from GSM8K or MATH datasets into mathematical areas using generated or Wikipedia taxonomies. Supports checkpoint-based resume.

Configure `wikipedia/cfg/static_vs_generated.yaml`:
- `data_cfg.dataset_name`: `gsm8k` or `math`
- `data_cfg.dataset_path`: Dataset file (GSM8K) or directory (MATH)
- `categorization_cfg.extraction_method`: `generated` or `wikipedia`

```bash
cd wikipedia
python static_vs_generated.py
```


## Development Guidelines

When implementing new features or modifying existing pipeline stages:

1. **Follow Schema Guidelines**: All data objects must use the schema classes defined in `src/schemas/`:
   - Use `Domain`, `Area`, `Capability`, `Task`, `TaskSolution`, `ValidationResult` objects
   - Load/save using schema IO functions from `src/schemas/io_utils.py` (e.g., `load_solution()`, `save_validation()`)
   - See `src/schemas/GENERATION_PIPELINE_SCHEMAS.md` for detailed schema documentation

2. **Use Model Call Utilities**: All LLM interactions must use the standardized model client utilities:
   - Import from `src.utils.model_client_utils`
   - Use `get_standard_model_client()` to initialize clients
   - Use `async_call_model()` with appropriate `ModelCallMode` (e.g., `JSON_PARSE`, `TEXT`)
