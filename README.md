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

---

#### 1. Generate Areas
Generate domain areas using the scientist–moderator debate system:
```bash
python -m src.agentic_area_generator
```

Output location:
```
~/<output_dir>/<domain>/<exp_id>/areas/<areas_tag>/areas.json
```
Where:
- <output_dir> comes from `global_cfg.output_dir`
- <domain> comes from `global_cfg.domain` (spaces replaced with underscores)
- <exp_id> comes from `exp_cfg.exp_id`
- <areas_tag> is the tag used for the generated areas

#### 2. Generate Capabilities
Generate capabilities for each area:
```bash
python -m src.agentic_capability_generator pipeline_tags.areas_tag=_YYYYMMDD_HHMMSS pipeline_tags.resume_capabilities_tag=_YYYYMMDD_HHMMSS
```

**Options:**
- `pipeline_tags.areas_tag` specifies which set of areas to use when generating capabilities.
- `pipeline_tags.resume_capabilities_tag` (optional) resumes a previous capability generation run.

**Output location:**
```
~/<output_dir>/<domain>/<exp_id>/capabilities/<capabilities_tag>/<area>/capabilities.json
```
Where:
- <capabilities_tag> is the tag used for the generated capabilities (either resumed or auto-generated)


#### 3. Generate Tasks
Generate evaluation tasks for a specific capabilities tag:
```bash
python -m src.agentic_task_generator pipeline_tags.capabilities_tag=_YYYYMMDD_HHMMSS pipeline_tags.resume_tasks_tag=_YYYYMMDD_HHMMSS
```

**Options:**
- `pipeline_tags.capabilities_tag` specifies which set of capabilities to use when generating tasks.
- `pipeline_tags.resume_tasks_tag` (optional) resumes a previous task generation run.

**Output location:**
```
~/<output_dir>/<domain>/<exp_id>/tasks/<tasks_tag>/[<area>]-[<capability>]/tasks.json
```
Where:
- <tasks_tag> is the tag used for the generated tasks (either resumed or auto-generated)

#### 4. Generate Solutions
Solve generated tasks using the multi-agent debate system:
```bash
python -m src.agentic_task_solver pipeline_tags.tasks_tag=_YYYYMMDD_HHMMSS pipeline_tags.resume_solutions_tag=_YYYYMMDD_HHMMSS
```

**Options:**
- `pipeline_tags.tasks_tag` specifies which set of tasks to solve.
- `pipeline_tags.resume_solutions_tag` (optional) resumes a previous solution generation run.

**Output location:**
```
~/<output_dir>/<domain>/<exp_id>/task_solutions/<solutions_tag>/[<area>]-[<capability>]/<task_id>_solution.json
```
Where:
- <solutions_tag> is the tag used for the generated solutions (either resumed or auto-generated)
