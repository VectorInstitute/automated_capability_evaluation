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

### Capability Generation using the scientist LLM

Generates capability names and descriptions in the first step. In the second step, for each capability, it generates tasks, solves them, and verifies the solutions.

```bash
python -m src.run_capability_generation
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
