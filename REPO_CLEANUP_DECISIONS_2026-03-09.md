# Repo Cleanup Decisions (2026-03-09)

## Scope

Cleanup of obsolete or unused code/tests with a conservative policy:
- Remove code that is confirmed dead.
- Archive code that may still be useful for reference.

## Decision Summary

### Deleted (Agentic Pipeline)

Reason: team confirmed the current agentic pipeline is no longer needed (including legacy retention).

Removed from active tree:
- `src/agentic_area_generator.py`
- `src/agentic_capability_generator.py`
- `src/agentic_task_generator.py`
- `src/agentic_task_solver.py`
- `src/area_generation/`
- `src/capability_generation/`
- `src/task_generation/`
- `src/task_solver/`
- `src/utils/agentic_prompts.py`
- `src/cfg/agentic_config.yaml`

### Archived to `legacy/pre_schema_pipeline/`

Reason: scripts were tied to older capability/LBO workflows and are no longer part of active base/eval pipelines, but may still provide historical context.

Moved from `src/`:
- `create_seed_capabilities.py`
- `generate_tasks.py`
- `get_seed_capability_results.py`
- `run_embedding_eval.py`
- `run_evaluation.py`

Moved from `example_scripts/`:
- `plot_llm_capability_scores.py`

### Deleted (Generated Test Artifacts)

Reason: generated outputs should not be versioned.

Removed:
- `tests/src/visualizations/*.pdf`

### Updated

- `README.md`: removed active references to deleted/archived pipelines; documented active evaluation entrypoint and legacy locations.
- `legacy/README.md`: updated legacy index to match current retained legacy folders.
- `example_scripts/README.md`: points to archived plotting script.
- `src/schemas/GENERATION_PIPELINE_SCHEMAS.md`: updated stale config path reference.
- `tests/src/test_dim_reduction_and_visualization.py`: now writes plots to `tmp_path` and asserts file creation, removing dependency on committed PDFs.

## Team Clarifications Needed Before Additional Cleanup

1. Should we also archive the remaining capability-centric stack (`src/capability.py`, `src/model.py`, `src/utils/capability_*`, related tests/examples), or keep it as supported functionality?
2. Should legacy Python tests under `legacy/tests/` remain in-repo as reference, or be removed from the default code tree entirely?
3. Do you want `legacy/` excluded from future lint/format/test runs, or kept as lint-clean reference code?

## Notes for PR Description

- Include two explicit lists:
  - "Deleted": agentic pipeline files + generated test PDF artifacts.
  - "Archived to legacy": pre-schema scripts moved to `legacy/pre_schema_pipeline/`.
