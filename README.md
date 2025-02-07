## 🧑🏿‍💻 Developing

### Installing dependencies

```bash
python3 -m venv env
source env/bin/activate
python3 -m pip install src/requirements.txt
```

### Run pipeline with default config

Note: Please set the following env vars before running the command: OPENAI_API_KEY, RATE_LIMIT_CALLS, RATE_LIMIT_PERIOD.

```bash
python3 src/run.py
```
