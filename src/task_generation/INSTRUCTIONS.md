# Task Generation (Agentic) README

This folder contains the agentic Stage-3 task generation pipeline.

It generates MCQ tasks from chapter text files using:
- `DesignerAgent` (drafting and repair)
- `VerifierAgent` (format/integrity checks)
- `run_task_generation_loop` (multi-step generation + retries)


## 1) What This Pipeline Does

For each chapter text file:
1. Summarizes chapter knowledge.
2. Generates one candidate MCQ at a time.
3. Runs iterative refinement steps (clarify, integrity check, redundancy/source cleanup, soundness).
4. Verifies pass/fail and retries with targeted repair prompts.
5. Saves passing tasks to Stage-3 output format.
6. Optionally deduplicates tasks and writes dedup reports.


## 2) Important Files

- `runner.py`: Main entrypoint for this agentic pipeline.
- `agentic_pipeline.py`: Core generation loop and per-step retry logic.
- `designer_agent.py`: Designer model wrapper and JSON extraction.
- `verifier_agent.py`: Verifier model wrapper and report parsing.
- `prompts.py`: All prompt templates used in each step.
- `blueprints/blueprints.json`: Difficulty/blooms/blueprint combinations.
- `init_math_book_chapter_text_files/`: Example chapter corpus location.
- `logs/`: Runtime logs (`task_gen_YYYYMMDD_HHMMSS.log`).

Config files:
- `src/cfg/task_generation/pipeline_config.yaml`
- `src/cfg/task_generation/agent_config.yaml`


## 3) Prerequisites

From repository root:

1. Install dependencies (your normal project setup).
2. Set environment variables used by models:
   - `GOOGLE_API_KEY` (designer by default)
   - `ANTHROPIC_API_KEY` (verifier by default)
   - `OPENAI_API_KEY` (used by dedup embeddings if enabled)
3. Ensure chapter corpus directory exists:
   - `src/task_generation/<book_chapter_dir>` where `book_chapter_dir` is set in `pipeline_config.yaml`.


## 4) Configure Before Run

### `pipeline_config.yaml`

Key fields:
- `pipeline.experiment_id`: output experiment folder name.
- `pipeline.output_base_dir`: root output directory.
- `pipeline.book_chapter_dir`: chapter corpus folder under `src/task_generation/`.
- `pipeline.blueprints_file`: blueprint JSON file under `src/task_generation/blueprints/`.
- `pipeline.max_retries`: retries per question in the repair loop.
- `pipeline.num_tasks`: tasks per chapter capability.
- `pipeline.resume_tag`: optional `_YYYYMMDD_HHMMSS` tag for resume.

Notes:
- `capability_source_mode: from_stage2` is currently not wired in `runner.py`.
- Use `capability_source_mode: placeholder` for now.

### `agent_config.yaml`

Controls:
- Designer/verifier model provider/model/API key mapping.
- Dedup config (`dedup.enabled`, threshold, embedding model, etc).


## 5) How To Run

### A) Run agentic task generation pipeline directly

From repo root:

```bash
python -m src.task_generation.runner
```

This uses:
- `src/cfg/task_generation/pipeline_config.yaml`
- `src/cfg/task_generation/agent_config.yaml`

### B) Run via base pipeline Stage 3

Set in `src/cfg/run_cfg.yaml`:
- `task_generation_cfg.mode: agentic`
- `stage: 3` (or `"all"` with prior stages configured)

Then run your normal base pipeline entrypoint (for example `src/run_base_pipeline.py`).


## 6) Output Structure

Outputs are written under:

`<output_base_dir>/<experiment_id>/tasks/<tasks_tag>/<area_id>/<capability_id>/`

Typical files:
- `tasks.json`: kept tasks (Stage-3 schema).
- `verification_stats.json`: verifier call summaries/logs.
- `dedup_report.json`: dedup details (if enabled).
- `discarded_tasks.json`: dedup-discarded tasks (if enabled).
- `checkpoints/passed_tasks_checkpoint.json`: resume checkpoint.


## 7) Resume and Checkpoint Behavior

- `resume_tag` in `pipeline_config.yaml`:
  - If set, runner reuses that output tag and skips chapters with existing `tasks.json`.
- Checkpoint resume:
  - `agentic_pipeline.py` restores accepted tasks from checkpoint when enabled.
  - It also rebuilds `previous_questions` anti-dup context from checkpointed tasks.

### Important Resume Gotcha

Checkpoint loading is tag-scoped. The checkpoint path is resolved under:

`<output_base_dir>/<experiment_id>/tasks/<out_tag>/<area_id>/<capability_id>/checkpoints/passed_tasks_checkpoint.json`

Where `out_tag` is selected as:
1. `tasks_tag` override passed from Stage-3/base pipeline, else
2. `pipeline.resume_tag`, else
3. a newly generated timestamp tag.

If your saved checkpoint is under `_20260227_155419` but you run with a different tag (or `resume_tag: null`), the pipeline will start from Q1 because it is looking at a different checkpoint path.

Recommended:
1. Set `pipeline.resume_tag: _YYYYMMDD_HHMMSS` when running `python -m src.task_generation.runner`, or
2. Pass `tasks_tag=_YYYYMMDD_HHMMSS` when running Stage 3 through `run_base_pipeline.py`.


## 8) Logs and Debugging

Logs are written to:
- `src/task_generation/logs/task_gen_YYYYMMDD_HHMMSS.log`

Watch for warnings such as:
- `generator returned non-JSON`
- `Step 3 produced non-MCQ payload; keeping prior candidate.`
- `System Error: empty response` / `json parse failed`

Interpretation:
- The LLM returned malformed/empty content or non-MCQ JSON.
- Pipeline now keeps last valid candidate state instead of propagating invalid payloads.


## 9) Common Failure Modes

1. **Empty model responses / malformed function-call filtering**
- Symptom: content null, parse fallback, fail report payload.
- Mitigation: retry in Step 1; guarded qcore propagation across steps.

2. **Verifier report accidentally passed as candidate question**
- Symptom: `Candidate_question` contains `overall_verdict/json_format_valid/...`.
- Mitigation: payload-shape guard in `agentic_pipeline.py`.

3. **Fenced JSON not parsed**
- Mitigation: `_ensure_json_string` now handles fenced ```json blocks and object extraction.

4. **Missing API keys**
- Symptom: model client init or API auth errors.
- Mitigation: verify environment variables before run.

5. **No chapter files found**
- Symptom: `No chapter .txt files found`.
- Mitigation: check `book_chapter_dir` path and `.txt` availability.


## 10) Practical Run Checklist

Before run:
1. Set API keys.
2. Verify chapter corpus path and blueprint file.
3. Set `num_tasks`, `max_retries`, and `resume_tag` policy.
4. Confirm model choices in `agent_config.yaml`.

After run:
1. Check `tasks.json` exists for each chapter.
2. Review `verification_stats.json` for high failure rates.
3. If needed, inspect logs and prompt payloads in `logs/`.
