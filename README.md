## 🧑🏿‍💻 Developing

### Installing dependencies

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

### [Optional] Google Cloud Authentication

The capability evaluation logs (evaluated using [Inspect](https://inspect.aisi.org.uk/)) are stored in a GCP bucket. Use the following command to log in using your GCP account:

```bash
gcloud auth application-default login
```

### Run pipeline

#### Configuration

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
    - LANGSMITH_PROJECT=<langsmith_project_id>
- GCP env vars:
    - GOOGLE_CLOUD_PROJECT=<gcp_project_id>

2. Modify `src/cfg/run_cfg.yaml`, if required.

#### Capability Generation using the scientist LLM

```bash
python3 src/run_capability_generation.py
```

#### Evaluation of subject LLM on generated capabilities

```bash
python3 src/run_evaluation.py
```

#### Run active learning pipeline

```bash
python3 src/run_lbo.py
```
