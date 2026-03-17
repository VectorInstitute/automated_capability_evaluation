# Task Generation (Agentic) README

This folder contains the agentic Stage-3 task generation pipeline.

It generates MCQ tasks from chapter text files using:
- `DesignerAgent` (drafting and repair)
- `VerifierAgent` (format/integrity checks)
- `run_task_generation_loop` (multi-step generation + retries)


## 1) What This Pipeline Does

For each generation unit:
1. Resolves chapter context from either chapter-derived placeholders or Stage-2 capabilities mapped to chapter files.
2. Summarizes chapter knowledge once for that unit.
3. Generates one seed MCQ at a time.
4. Iteratively hardens that seed across multiple rounds, where each round starts from the previously hardened candidate.
5. Runs refinement steps (clarify, integrity check, redundancy/source cleanup, soundness).
6. Verifies pass/fail and retries with targeted repair prompts.
7. Saves passing tasks to Stage-3 output format.
8. Optionally deduplicates tasks and writes dedup reports.

Current status:
- This Stage-3 agentic pipeline is experimental and chapter-driven.
- Exact Stage 1 -> Stage 2 -> Stage 3 area/capability matching is not guaranteed in the current chapter-based flow.
- When Stage-2 capabilities are unavailable, the pipeline uses placeholder area/capability lineage so outputs remain schema-compatible.
- The supported workflow for this experimental path is to run task generation from Stage 3, not to rely on a full Stage 1 -> Stage 5 base-pipeline run.


## 2) Important Files

- `runner.py`: Main entrypoint for this agentic pipeline.
- `agentic_pipeline.py`: Core generation loop and per-step retry logic.
- `designer_agent.py`: Designer model wrapper and JSON extraction.
- `verifier_agent.py`: Verifier model wrapper and report parsing.
- `prompts.py`: All prompt templates used in each step.
- `blueprints/blueprints.json`: Difficulty/blooms/blueprint combinations.
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
- `pipeline.num_tasks`: number of seed generations per generation unit.
- `pipeline.hardening_rounds`: number of chained hardening rounds per seed generation.
- `pipeline.capability_source_mode`: `placeholder` or `from_stage2`.
- `pipeline.checkpoint.*`: checkpoint enable/resume/interval/path settings.

Notes:
- `capability_source_mode: from_stage2` is supported only when Stage-2 capability artifacts can actually be loaded.
- If Stage-2 capabilities are missing, the runner falls back to chapter-derived placeholder area/capability lineage.
- The current runner logs the number of blueprint combinations found, but task generation currently uses only the first blueprint combination.

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
- `stage: 3`

Then run your normal base pipeline entrypoint (for example `src/run_base_pipeline.py`).

```bash
python -m src.run_base_pipeline stage=3 capabilities_tag=placeholder
```

## 6) Output Structure

Outputs are written under:

`<output_base_dir>/<experiment_id>/tasks/<tasks_tag>/<area_id>/<capability_id>/`

Typical files:
- `tasks.json`: kept tasks (Stage-3 schema).
- `verification_stats.json`: verifier call summaries/logs.
- `dedup_report.json`: dedup details (if enabled).
- `discarded_tasks.json`: dedup-discarded tasks (if enabled).
- `checkpoints/passed_tasks_checkpoint.json`: resume checkpoint.

### Sample Output

Example output directory after one generation-unit run:

```text
base_output/
  test_exp/
    tasks/
      _20260309_190626/
        area_ch_finance_the_economics_of_money_banking_and_financial_markets_007_806a69f8/
          cap_ch_finance_the_economics_of_money_banking_and_financial_markets_007_806a69f8/
            tasks.json
            verification_stats.json
            checkpoints/
              passed_tasks_checkpoint.json
```

Example `tasks.json` shape:

```json
{
  "metadata": {
    "experiment_id": "test_exp",
    "output_base_dir": "base_output",
    "timestamp": "2026-03-11T15:58:33.936893Z",
    "resume": true,
    "input_stage_tag": "placeholder",
    "output_stage_tag": "_20260309_190626"
  },
  "tasks": [
    {
      "task_id": "task_000",
      "task_statement": "Nebula Tech just paid an annual dividend ($D_0$) of $2.50 ... What is the immediate change in Nebula Tech's stock price?\n\nOptions:\nA. It increases by $7.49.\nB. It decreases by $15.96.\nC. It decreases by $11.57.\nD. It increases by $15.96.\nE. None of the above",
      "task_type": "multiple_choice",
      "solution_type": "multiple_choice",
      "difficulty": "Hard",
      "bloom_level": "Apply",
      "choices": [
        { "label": "A", "solution": "It increases by $7.49." },
        { "label": "B", "solution": "It decreases by $15.96." },
        { "label": "C", "solution": "It decreases by $11.57." },
        { "label": "D", "solution": "It increases by $15.96." },
        { "label": "E", "solution": "None of the above" }
      ],
      "capability_name": "placeholder_capability_Finance_The_Economics_of_Money_Banking_and_Financial_Markets_007",
      "capability_id": "cap_000",
      "area_name": "placeholder_area_Finance_The_Economics_of_Money_Banking_and_Financial_Markets_007",
      "area_id": "area_000",
      "domain_name": "Finance",
      "domain_id": "domain_000",
      "generation_metadata": {
        "chapter_id": "Finance_The_Economics_of_Money_Banking_and_Financial_Markets_007",
        "chapter_relpath": "Finance_The_Economics_of_Money_Banking_and_Financial_Markets/Finance_The_Economics_of_Money_Banking_and_Financial_Markets_007.txt",
        "capability_source_mode": "placeholder",
        "blueprint_key": "Hard_Apply",
        "correct_answer": "B",
        "chapter_question_id": "Finance_The_Economics_of_Money_Banking_and_Financial_Markets_007_q_000",
        "solution_graph": { "...": "omitted for brevity" },
        "complete_solution": "Step 1: Determine the market price prior to the announcement ...",
        "hardening_round_candidate_index": 1,
        "hardening_round_candidate_total": 5,
        "seed_generation_index": 1,
        "seed_generation_target": 50
      }
    }
  ]
}
```

Example `verification_stats.json` shape:

```json
{
  "chapter_id": "Finance_The_Economics_of_Money_Banking_and_Financial_Markets_007",
  "chapter_relpath": "Finance_The_Economics_of_Money_Banking_and_Financial_Markets/Finance_The_Economics_of_Money_Banking_and_Financial_Markets_007.txt",
  "book_name": "Finance_The_Economics_of_Money_Banking_and_Financial_Markets",
  "num_verifier_calls": 315,
  "verification_logs": [
    {
      "task_batch_id": "batch_97f316",
      "attempt_human": "1/4",
      "blueprint_key": "Hard_Apply",
      "difficulty": "Hard",
      "blooms_level": "Apply",
      "seed_generation_index": 1,
      "hardening_round_candidate_index": 1,
      "summary": {
        "overall_verdict": "Pass",
        "json_format_valid": "Yes",
        "mcq_integrity": "Yes",
        "constraint_compliance": "Yes"
      }
    }
  ]
}
```

Notes:
- In the current experimental chapter-based flow, `area_id` and `capability_id` may be placeholder lineage values.
- `generation_metadata` includes useful traceability fields such as chapter origin, correct answer, hardening-round index, and seed-generation index.
- If dedup is enabled, you may also see `dedup_report.json` and `discarded_tasks.json` in the same directory.


## 7) Resume and Checkpoint Behavior

There are two different resume mechanisms:

- Output-tag resume at the runner level:
  - The runner resumes only when an existing `tasks_tag` is passed in.
  - Direct `python -m src.task_generation.runner` runs do not read a `pipeline.resume_tag` from config.
  - Stage-3/base-pipeline runs can resume by passing an existing `tasks_tag`.
  - When resuming, the runner skips any generation unit that already has `tasks.json`.

- Checkpoint resume inside the agentic loop:
  - If checkpointing is enabled and a checkpoint file exists for the current output tag/unit, accepted tasks and verification logs are restored.
  - The pipeline also restores generation progress so seed-generation counting continues from the checkpointed state.
  - Prompt anti-dup memory is rebuilt from checkpointed tasks using one representative per seed generation, not every accepted hardened task.

### Important Resume Gotcha

Checkpoint loading is tag-scoped. The checkpoint path is resolved under:

`<output_base_dir>/<experiment_id>/tasks/<out_tag>/<area_id>/<capability_id>/checkpoints/passed_tasks_checkpoint.json`

Where `out_tag` is selected as:
1. `tasks_tag` override passed from Stage-3/base pipeline, else
2. a newly generated timestamp tag.

If your saved checkpoint is under `_20260227_155419` but you run with a different `tasks_tag`, the pipeline will start fresh because it is looking at a different checkpoint path.

Recommended:
1. Reuse the same `tasks_tag` when resuming through Stage 3, or
2. Keep the same output tag path if you are resuming from an existing checkpointed run.

### Checkpoint Save Behavior

- Checkpoints are written under `checkpoints/passed_tasks_checkpoint.json`.
- The loop saves checkpoints periodically based on `pipeline.checkpoint.every`.
- It also saves checkpoints after skipped/failed seed-generation attempts when checkpointing is enabled.
- Checkpoints store accepted tasks, verification logs, and generation progress state.


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

6. **Expecting full Stage-1-to-Stage-3 lineage in the experimental chapter-based flow**
- Symptom: area/capability ids look placeholder-like or do not match an earlier full base-pipeline run exactly.
- Mitigation: treat the current agentic pipeline as Stage-3-first and chapter-driven; use Stage-2 capability artifacts only when they are available and intentionally mapped.


## 10) Practical Run Checklist

Before run:
1. Set API keys.
2. Verify chapter corpus path and blueprint file.
3. Set `num_tasks`, `hardening_rounds`, `max_retries`, and checkpoint policy.
4. Confirm model choices in `agent_config.yaml`.
5. Decide whether you are using placeholder lineage or loading Stage-2 capabilities.

After run:
1. Check `tasks.json` exists for each chapter.
2. Review `verification_stats.json` for high failure rates.
3. If dedup is enabled, review `dedup_report.json` and `discarded_tasks.json`.
4. If needed, inspect logs and prompt payloads in `logs/`.
