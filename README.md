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

### Run pipeline with default config

Note: Please set the following env vars before running the command: OPENAI_API_KEY, RATE_LIMIT_CALLS, RATE_LIMIT_PERIOD.

```bash
python3 src/run.py
```
