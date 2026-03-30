# Task Generation (Agentic) README

This folder contains the agentic Stage-3 task-generation pipeline.

It generates MCQ tasks from chapter text files using:
- `DesignerAgent` for draft generation and repair
- `VerifierAgent` for MCQ integrity and final verification
- `run_task_generation_loop(...)` for the multi-step generation loop
- `runner.py` for chapter discovery, sharding, resume, and artifact writing

> [!WARNING]
> This is an experimental Stage-3 path. It is chapter-driven and compatible with the Stage-3 output layout, but it should still be treated as a specialized task-generation path rather than a fully validated replacement for every existing base-pipeline behavior.

The default Stage-3 mode remains `task_generation_cfg.mode: base` in `src/cfg/run_cfg.yaml`.

To use this path, explicitly set:

`task_generation_cfg.mode: agentic`

> [!WARNING]
> Experimental feature: this agentic task-generation pipeline is an interim Stage-3 implementation and is not a full drop-in replacement for the standard Stage 1→5 pipeline.

The default pipeline mode remains `task_generation_cfg.mode: base` in `src/cfg/run_cfg.yaml`.

To run this chapter-based agentic Stage-3 path, explicitly set:
`task_generation_cfg.mode: agentic`

When `mode=base`, Stage 3 uses the standard capability-based generation flow

## 1) What This Pipeline Does

For each generation unit, the pipeline:
1. Resolves a generation unit from chapter text:
   - `placeholder` mode: one chapter file becomes one generation unit
   - `from_stage2` mode: one Stage-2 capability becomes one generation unit, optionally mapped to one or more chapters
2. Summarizes chapter knowledge once per generation unit and caches it.
3. Iterates over every configured blueprint combination in `blueprints.json`.
4. For each combination, runs seed generation one question at a time.
5. Optionally runs hardening rounds for each seed candidate.
6. Sends the seed candidate and any hardened candidates through the same downstream refinement steps.
7. Runs:
   - clarification
   - MCQ integrity check/revision
   - redundant-info cleanup
   - source-reference cleanup
   - soundness cleanup
   - final verification
8. Retries failed candidates with targeted repair prompts.
9. Saves passing tasks in the normal Stage-3 layout:
   `tasks/<tasks_tag>/<area_id>/<capability_id>/tasks.json`
10. Writes chapter-level artifacts such as verification stats, token stats, chapter summary, checkpoints, and optional dedup outputs.

Important current behavior:
- If `hardening_rounds: 0`, the pipeline runs in seed-only mode.
- All configured blueprint combinations are used.
- `choices` are preserved in the task schema, and the option text is also appended into `task_statement`.
- Token usage is recorded across the full agentic generation path and saved separately in `token_stats.json`.
- Resume is controlled by `tasks_tag`.
- In `placeholder` mode, chapter lineage is synthesized but remains schema-compatible.
- In `from_stage2` mode, Stage-2 capability IDs and lineage are preserved when capability artifacts are available.


## 2) Important Files

- `runner.py`: main entrypoint, chapter discovery, sharding, resume, output writing
- `agentic_pipeline.py`: core generation loop and retry logic
- `designer_agent.py`: designer model wrapper and JSON extraction
- `verifier_agent.py`: verifier model wrapper and verification/report parsing
- `prompts.py`: prompt templates
- `blueprints/blueprints.json`: difficulty/Bloom's combinations and per-combo task counts
- `run_parallel_stage3.sh`: helper script for launching multiple Stage-3 workers
- `logs/`: runtime logs (`task_gen_YYYYMMDD_HHMMSS.log`)

Config files:
- `src/cfg/task_generation/pipeline_config.yaml`
- `src/cfg/task_generation/agent_config.yaml`
- `src/cfg/run_cfg.yaml`


## 3) Prerequisites

From repository root:

1. Install project dependencies as usual.
2. Set environment variables used by your configured models.
3. Ensure the chapter corpus directory exists:
   - `src/task_generation/<book_chapter_dir>`
4. Ensure your blueprints file exists under:
   - `src/task_generation/blueprints/<blueprints_file>`


## 4) Configuration

### `src/cfg/task_generation/pipeline_config.yaml`

Key fields:
- `pipeline.experiment_id`: experiment folder name
- `pipeline.output_base_dir`: output root
- `pipeline.book_chapter_dir`: chapter corpus root under `src/task_generation/`
- `pipeline.blueprints_file`: blueprint JSON file name
- `pipeline.capability_source_mode`: `placeholder` or `from_stage2`
- `pipeline.max_retries`: retry count for failed candidates
- `pipeline.hardening_rounds`: number of hardening rounds per seed generation
- `pipeline.num_tasks_per_combo`: default seed count per blueprint combination
- `pipeline.checkpoint.*`: checkpoint behavior

Notes:
- `hardening_rounds: 0` means seed-only generation.
- In `placeholder` mode, one discovered chapter file becomes one generation unit.
- In `from_stage2` mode, one capability becomes one generation unit.
- If a capability-to-chapter mapping file is omitted in `from_stage2` mode:
  - one discovered chapter => that chapter is used
  - multiple discovered chapters => all discovered chapters are bundled as context for that capability

### `src/cfg/run_cfg.yaml`

Relevant Stage-3 fields:
- `task_generation_cfg.mode`
- `task_generation_cfg.worker_index`
- `task_generation_cfg.worker_count`
- `capabilities_tag`
- `tasks_tag`

These are used when launching the agentic pipeline through:

`python -m src.run_base_pipeline`

### `src/cfg/task_generation/agent_config.yaml`

Controls:
- designer and verifier model config
- provider and API key mapping
- dedup settings


## 5) How Many Tasks Are Generated

For one generation unit:
- each blueprint combination contributes `seed_num_tasks * (hardening_rounds + 1)` maximum passing-task slots
- with `hardening_rounds: 0`, this becomes exactly `seed_num_tasks`
- total tasks per generation unit are the sum across all configured blueprint combinations, before dedup

Example:
- 4 combinations
- each with `num_tasks: 13`
- `hardening_rounds: 0`

Then the pipeline targets:

`4 * 13 = 52`

maximum passing tasks for that generation unit before dedup.


## 6) How Chapters Become Generation Units

### Placeholder Mode

When `capability_source_mode: placeholder`:
- all `*.txt` files under `book_chapter_dir` are discovered recursively
- each chapter file becomes one generation unit
- placeholder area/capability IDs are generated deterministically from chapter identity

If you have one book folder with many chapter files:
- one chapter file = one generation unit

If you later add multiple book folders under the same corpus root:
- all chapter files across all books are discovered
- each chapter still becomes its own generation unit
- the pipeline can process chapters from different books in the same run

### From Stage-2 Mode

When `capability_source_mode: from_stage2`:
- Stage-2 capability files are loaded from the provided `capabilities_tag`
- one capability becomes one generation unit
- capability IDs and lineage are preserved
- chapter context may come from one or more chapter files depending on the mapping


## 7) How To Run

### A) Run agentic Stage 3 through the base pipeline

Single-process run:

```bash
python -m src.run_base_pipeline stage=3 capabilities_tag=placeholder task_generation_cfg.mode=agentic
```

Resume a single-process run:

```bash
python -m src.run_base_pipeline stage=3 capabilities_tag=placeholder tasks_tag=_YOUR_TASKS_TAG task_generation_cfg.mode=agentic
```

### B) Run agentic Stage 3 directly through the runner

```bash
python -m src.task_generation.runner
```

Optional direct-runner args:
- `--tasks-tag`
- `--capabilities-tag`
- `--worker-index`
- `--worker-count`

### C) Run parallel workers with the helper script

Fresh 4-worker run:

```bash
bash src/task_generation/run_parallel_stage3.sh '' placeholder
```

Resume the same run:

```bash
bash src/task_generation/run_parallel_stage3.sh _YOUR_TASKS_TAG placeholder
```

The script launches multiple workers through:

`python -m src.run_base_pipeline stage=3 ...`


## 8) Parallel Sharding Behavior

Chapter-level sharding is deterministic.

The runner:
1. discovers all generation units
2. sorts them in a stable order
3. assigns each unit by modulo:

`unit_index % worker_count == worker_index`

Example with `worker_count=4`:
- worker 0 gets indices `0, 4, 8, ...`
- worker 1 gets `1, 5, 9, ...`
- worker 2 gets `2, 6, 10, ...`
- worker 3 gets `3, 7, 11, ...`

Important:
- all workers in one parallel run must use the same `tasks_tag`
- workers differ only by `worker_index`
- workers do not run all their assigned chapters simultaneously
- each worker processes its owned chapters sequentially
- so with 4 workers, you get up to 4 generation units active at once


## 9) Output Structure

Outputs are written under:

`<output_base_dir>/<experiment_id>/tasks/<tasks_tag>/<area_id>/<capability_id>/`

Typical files:
- `tasks.json`: final kept tasks
- `verification_stats.json`: verification call logs
- `token_stats.json`: full token-accounting logs and summaries
- `chapter_summary.json`: cached chapter summary and summary-call usage
- `dedup_report.json`: dedup report if enabled
- `discarded_tasks.json`: dedup-discarded tasks if enabled
- `checkpoints/<combo>_passed_tasks_checkpoint.json`: combo-specific checkpoint files

Notes:
- Stage-3 output format remains:
  `tasks/<tasks_tag>/<area_id>/<capability_id>/tasks.json`
- `verification_stats.json` is verification-focused
- `token_stats.json` is the source of truth for token accounting


## 10) `token_stats.json`

`token_stats.json` stores token usage across the full agentic generation path, not just final verification.

It includes:
- one record per model call in `token_usage_logs`
- chapter-level totals
- breakdowns by stage
- breakdowns by model role
- counts of calls with missing provider usage metadata

Important:
- `total_input_tokens` is computed directly from saved per-call logs
- `total_output_tokens` is computed directly from saved per-call logs
- `total_tokens = total_input_tokens + total_output_tokens`
- this makes later benchmark-cost calculation reproducible


## 11) Resume and Checkpoint Behavior

Resume is controlled by `tasks_tag`.

### Runner-Level Resume

A run is treated as resume only when an existing `tasks_tag` is explicitly provided.

That means:
- if `tasks_tag` is provided, Stage 3 is treated as resume
- if `tasks_tag` is not provided, Stage 3 creates a fresh tag and starts a new run

At the generation-unit level:
- if final `tasks.json` already exists for that unit, the runner skips it
- otherwise the runner continues into chapter-summary reuse and checkpoint loading

### Chapter Summary Resume

The runner caches chapter summaries in:

`chapter_summary.json`

On resume:
- if `chapter_summary.json` exists, the summary is reused
- the chapter summary call is not repeated

### Combo Checkpoint Resume

Checkpoint files are combo-specific and live under:

`checkpoints/<combo>_passed_tasks_checkpoint.json`

These checkpoints store:
- accepted tasks
- verification logs
- token usage logs
- generation progress state

On resume:
- accepted tasks are restored
- verification logs are restored
- token logs are restored
- prompt anti-dup memory is rebuilt from restored tasks

Important resume rule:
- reuse the same `tasks_tag`
- if running in parallel, also reuse the same `worker_count` and the same worker indices


## 12) Common Failure Modes

1. **Missing API keys**
- Symptom: model client init/auth errors
- Mitigation: verify environment variables first

2. **No chapter files found**
- Symptom: `No chapter .txt files found`
- Mitigation: verify `book_chapter_dir`

3. **Malformed model output**
- Symptom: non-JSON, empty content, invalid MCQ payload
- Mitigation: the pipeline retries and keeps the last valid candidate state where possible

4. **Using a different `tasks_tag` when resuming**
- Symptom: pipeline appears to start fresh
- Mitigation: resume with the same `tasks_tag`

5. **Changing shard layout on resume**
- Symptom: chapters may be reassigned unexpectedly across workers
- Mitigation: keep the same `worker_count` and worker indices

6. **Assuming each worker runs all its chapters simultaneously**
- Reality: each worker processes its assigned generation units sequentially

6. **Expecting full Stage-1-to-Stage-3 lineage in the experimental chapter-based flow**
- Symptom: area/capability ids look placeholder-like or do not match an earlier full base-pipeline run exactly.
- Mitigation: treat the current agentic pipeline as Stage-3-first and chapter-driven; use Stage-2 capability artifacts only when they are available and intentionally mapped.


## 13) Practical Checklist

Before run:
1. Set model API keys
2. Verify chapter corpus path
3. Verify blueprint file
4. Set `hardening_rounds`, `max_retries`, and checkpoint policy
5. Decide between `placeholder` and `from_stage2`
6. Decide whether to run single-process or sharded parallel workers

Before a parallel run:
1. choose one shared `tasks_tag`
2. launch all workers with the same `tasks_tag`
3. vary only `worker_index`

After run:
1. check `tasks.json`
2. review `verification_stats.json`
3. review `token_stats.json`
4. inspect `dedup_report.json` if dedup is enabled
5. inspect logs in `src/task_generation/logs/` if something looks off
