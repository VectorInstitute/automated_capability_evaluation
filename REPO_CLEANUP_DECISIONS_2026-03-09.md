# Repo Cleanup Decisions (2026-03-09)

## Scope

Cleanup of obsolete or unused code/tests:
- Remove code that is confirmed dead.
- Archive code that may still be useful for reference under `legacy/`.

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

### Deleted (Legacy Tests)

Reason: team confirmed legacy tests are no longer needed.

Removed:
- `legacy/tests/`
- legacy-oriented tests and fixtures previously moved out of `tests/src/`:
  - `test_capability_class.py`
  - `test_capability_embedding.py`
  - `test_dim_reduction_and_visualization.py`
  - `test_embedding_and_filtering.py`
  - `test_model_class.py`
  - `capabilities_t2/`
  - `seed_capabilities/`
  - `seed_capabilities_scores/`
  - `resources/manual_capabilities.json`

### Updated

- `README.md`: removed active references to deleted/archived pipelines; documented active evaluation entrypoint and legacy locations.
- `legacy/README.md`: updated legacy index to match current retained legacy folders.
- `example_scripts/README.md`: points to archived plotting script.
- `src/schemas/GENERATION_PIPELINE_SCHEMAS.md`: updated stale config path reference.

## Clarifications Resolved

1. Agentic pipeline is fully removed and not retained in legacy.
2. Legacy tests are fully removed.
3. Legacy-only capability stack is retained under `legacy/` while active pipeline keeps only the currently used utility surface.

## Notes for PR Description

- Include two explicit lists:
  - "Deleted": agentic pipeline files + legacy tests + generated artifacts.
  - "Archived to legacy": pre-schema scripts and legacy-only capability modules.
