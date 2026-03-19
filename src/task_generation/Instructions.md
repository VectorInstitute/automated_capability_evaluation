# Agentic Task Generation Pipeline

This README describes how to run the **agentic task-generation pipeline**.

The config files can be found under:

- `src/cfg/task_generation/agent_config.yaml`
- `src/cfg/task_generation/pipeline_config.yaml`

The chapter text files should be under `src/task_generation/<domain_name_dir>/<book_name_dir>`.

---

## 0) What will run

For every chapter file under:

- `src/task_generation/<domain_name_dir>/<book_name_dir>/*.txt`

and for every blueprint combo in:

- `src/task_generation/blueprints/<blueprints_file>`

the runner does:

1. Designer draft generation (JSON batch, retries on JSON/shape failures)
2. Per question:
   - Step 2: include clarification info (*designer_agent*)
   - Step 3: remove redundancy (*designer_agent*)
   - Step 4: remove source references (*designer_agent*)
   - Step 5: soundness check (*designer_agent*)
   - Step 6: MCQ integrity check + revise options (question unchanged) (*verifier_agent*)
   - Step 7: final verification (structured JSON verdict) (*verifier_agent*)
   - Step 8: fix-bug loop if verification fails (retry up to `max_retries`) (*designer_agent*)
3. Save passing tasks
4. Optional chapter-level dedup + report + discarded tasks

---

## 1) Create & activate a Python environment

### Option A: venv (recommended)
```bash
cd /path/to/your/repo
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
```

### Option B: conda
```bash
cd /path/to/your/repo
conda create -n ace python=3.10 -y
conda activate ace
python -m pip install -U pip
```

---

## 2) Install dependencies

The pipeline requires packages equivalent to:
```bash
pip install pyyaml python-dotenv autogen-agentchat autogen-ext openai
```

> If you see import errors for `src.schemas.*`, make sure you are running from repo root (and `src/` is importable).

---

## 3) Set environment variables (API keys)

### If you use OpenAI for verifier (current config)
```bash
export OPENAI_API_KEY="YOUR_OPENAI_KEY"
```

### If you use Gemini for designer (your current config references Google)
```bash
export GOOGLE_API_KEY="YOUR_GOOGLE_KEY"
```

### If you switch to local vLLM (optional)
If you uncomment the `local_vllm` config and use it via AutoGen:
```bash
export VLLM_API_KEY="EMPTY"
```

> You can also put these into a `.env` file at repo root because `runner.py` calls `load_dotenv()`:
```bash
# .env (repo root)
OPENAI_API_KEY=...
GOOGLE_API_KEY=...
VLLM_API_KEY=EMPTY
```

---

## 4) Verify/prepare input files

### 4.1 Chapter text files
If the `src/cfg/task_generation/pipeline_config.yaml` have:
```yaml
book_chapter_dir: "<domain_name_dir>", e.g., 'math_books'
```

The runner will look under:
```text
src/task_generation/<domain_name_dir>/<book_name_dir>/**/*.txt, e.g., src/task_generation/math_books/all_of_statistics_chapters/all_of_statistics_chapter_000.txt
```

✅ Make sure the `.txt` chapters exist there.

### 4.2 Blueprints file
In `pipeline_config.yaml`:
```yaml
blueprints_file: "blueprints.json"
```

The runner will load:
```text
src/task_generation/blueprints/blueprints.json
```

---

## 5) Configure models (Designer / Verifier)

Open:
- `src/cfg/task_generation/agent_config.yaml`

### Contains
- **Designer**: Google Gemini (`gemini-3-pro-preview`)
- **Verifier**: OpenAI (`gpt-5.2-chat-latest`)
- **Dedup**: enabled (embedding model `text-embedding-3-small`)

#### If you want to run everything via OpenAI only
Change the designer config under:
```yaml
agents:
  designer:
    model: "openai_gpt"
    model_config:
      config_list:
        - api_type: "openai"
          model: "gpt-5.2-chat-latest"
          api_key: "${OPENAI_API_KEY}"
```

#### If you want to run via local vLLM
1) Make sure the vLLM server is running and reachable at:
```yaml
base_url: "http://0.0.0.0:8000/v1"
```

2) Uncomment the `designer`/`verifier` blocks you already have for `local_vllm` and export `VLLM_API_KEY=EMPTY`.

---

## 6) Run the pipeline

All the prompts are provided under:
```text
src/task_generation/prompts.py
```

From **repo root**, run:
```bash
python -m src.task_generation.runner
```

or equivalently:
```bash
python src/task_generation/runner.py
```

### After running the `runner` script
- A log file will be created at:
```text
src/task_generation/logs/task_gen_<timestamp>.log
```

- The console will show:
  - discovered chapter files
  - combos
  - per-step progress (draft attempts, per-question attempts)
  - dedup summary (if enabled)
  - save locations

---

## 7) Output structure

In `pipeline_config.yaml`:
```yaml
experiment_id: "ag_ext_src"
output_base_dir: "src/task_generation/agentic_outputs"
```

Runner writes to a timestamp tag like `_YYYYMMDD_HHMMSS`, producing:

```text
src/task_generation/agentic_outputs/
  ag_ext_src/
    tasks/
      _YYYYMMDD_HHMMSS/
        <book_name>/
          <chapter_id>/
            tasks.json
            verification_stats.json
            dedup_report.json              (if dedup enabled)
            discarded_tasks.json           (if dedup enabled + save_discarded true)
            embedding_cache.json           (if dedup enabled + cache_embeddings true)
```

### What each file contains
- `tasks.json`: final kept tasks (after dedup if enabled)
- `verification_stats.json`: per-attempt verifier summaries
- `dedup_report.json`: similarity clusters, kept/discarded metadata
- `discarded_tasks.json`: tasks removed by dedup (tagged as discarded)

---

## 8) Resume runs (skips chapters that are already done)

In `pipeline_config.yaml`:
```yaml
resume_tag: null
```

To resume a previously started run, set it to the same tag folder name:
```yaml
resume_tag: "_20260205_164212"
```

Then rerun:
```bash
python -m src.task_generation.runner
```

The runner will skip any chapter where:
```text
.../<tag>/<book>/<chapter>/tasks.json
```
already exists.

---

## 9) Common issues & fixes

### A) `ModuleNotFoundError: No module named 'src'`
You’re not running from repo root, or `PYTHONPATH` is not set.

Fix (run from repo root):
```bash
cd /path/to/your/repo
python -m src.task_generation.runner
```

If needed:
```bash
export PYTHONPATH="$(pwd)"
python -m src.task_generation.runner
```

### B) API key not found / empty
Because `agent_config.yaml` expands env vars, missing keys become empty strings.

Fix:
```bash
export OPENAI_API_KEY="..."
export GOOGLE_API_KEY="..."
```

### C) Blueprints file not found
The runner expects:
```text
src/task_generation/blueprints/<blueprints_file>
```
Make sure `pipeline_config.yaml` matches actual filename.

### D) Chapter files not found
The runner expects:
```text
src/task_generation/<book_chapter_dir>/**/*.txt
```
Make sure your `book_chapter_dir` exists under `src/task_generation/`.

### E) vLLM / local endpoint not used
If you intend to use a local OpenAI-compatible endpoint, confirm your client supports passing `base_url`.
If requests still go to OpenAI, you’ll need to ensure `OpenAIChatCompletionClient` is constructed with `base_url` (depending on your `autogen_ext` version).

---

## 10) One-command run (example)

```bash
cd /path/to/your/repo
source .venv/bin/activate

export OPENAI_API_KEY="YOUR_OPENAI_KEY"
export GOOGLE_API_KEY="YOUR_GOOGLE_KEY"

python -m src.task_generation.runner
```

---

## 11) Verify outputs quickly

After a run, list newest tag:
```bash
ls -lt src/task_generation/agentic_outputs/ag_ext_src/tasks | head
```

Inspect a chapter’s saved tasks:
```bash
cat src/task_generation/agentic_outputs/ag_ext_src/tasks/_YYYYMMDD_HHMMSS/<book>/<chapter>/tasks.json | head -n 50
```
