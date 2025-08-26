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

Generate areas, capabilities, and tasks using multi-agent debate systems. Configure parameters in `src/cfg/agentic_config.yaml`.

```bash
# Generate capability areas
python -m src.agentic_area_generator

# Generate capabilities for each area
python -m src.agentic_capability_generator

# Generate tasks for each capability
python -m src.agentic_task_generator
```
