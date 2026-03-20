"""Tests for evaluation pipeline stages and I/O helpers."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from src.eval_stages import stage1_eval_execution as stage1
from src.eval_stages import stage2_score_aggregation as stage2
from src.schemas.eval_io_utils import get_eval_dir
from src.schemas.eval_schemas import EvalConfig, EvalDataset


def _sample(sample_id: str, scorer_values: dict[str, object]) -> SimpleNamespace:
    return SimpleNamespace(
        id=sample_id,
        scores={
            scorer: SimpleNamespace(value=value)
            for scorer, value in scorer_values.items()
        },
    )


def test_parse_inspect_logs_uses_best_single_log(tmp_path, monkeypatch):
    """Stage 2 should avoid double-counting multiple logs and scorers."""
    result_dir = tmp_path / "area_000" / "cap_000"
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "a.json").write_text("{}")
    (result_dir / "b.json").write_text("{}")

    logs = {
        "a.json": SimpleNamespace(
            samples=[
                _sample("task_000", {"scorer_a": 1, "scorer_b": 0}),
                _sample("task_001", {"scorer_a": 0, "scorer_b": 1}),
            ]
        ),
        "b.json": SimpleNamespace(samples=[_sample("task_000", {"scorer_a": 1})]),
    }

    monkeypatch.setattr(
        stage2,
        "read_eval_log",
        lambda path: logs[Path(path).name],
    )

    parsed = stage2._parse_inspect_logs(result_dir, {"task_000", "task_001"})

    assert parsed["num_tasks"] == 2
    assert parsed["mean"] == pytest.approx(0.5)


def test_check_eval_completed_requires_full_task_coverage(tmp_path, monkeypatch):
    """Stage 1 resume skip should require complete sample coverage."""
    result_dir = tmp_path / "llm" / "area_000" / "cap_000"
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "partial.json").write_text("{}")
    (result_dir / "complete.json").write_text("{}")

    scored_ids = {
        "partial.json": {"task_000", "task_001"},
        "complete.json": {"task_000", "task_001", "task_002"},
    }
    monkeypatch.setattr(
        stage1,
        "_scored_sample_ids",
        lambda log_file: scored_ids[Path(log_file).name],
    )

    assert stage1._check_eval_completed(
        tmp_path, "llm", "area_000", "cap_000", {"task_000", "task_001", "task_002"}
    )
    assert not stage1._check_eval_completed(
        tmp_path,
        "llm",
        "area_000",
        "cap_000",
        {"task_000", "task_001", "task_002", "task_003"},
    )


def test_check_eval_completed_fails_on_extra_task_ids(tmp_path, monkeypatch):
    """Stage 1 completion should require exact task ID match."""
    result_dir = tmp_path / "llm" / "area_000" / "cap_000"
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "log.json").write_text("{}")

    monkeypatch.setattr(
        stage1,
        "_scored_sample_ids",
        lambda _log_file: {"task_000", "task_001", "task_extra"},
    )

    assert not stage1._check_eval_completed(
        tmp_path, "llm", "area_000", "cap_000", {"task_000", "task_001"}
    )


def test_find_retry_log_prefers_failed_incomplete_log(tmp_path, monkeypatch):
    """Resume helper should choose the best failed/incomplete log candidate."""
    result_dir = tmp_path / "llm" / "area_000" / "cap_000"
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "a.json").write_text("{}")
    (result_dir / "b.json").write_text("{}")
    (result_dir / "c.json").write_text("{}")

    logs = {
        "a.json": SimpleNamespace(
            status="error",
            invalidated=False,
            samples=[_sample("task_000", {"scorer": 1})],
        ),
        "b.json": SimpleNamespace(
            status="cancelled",
            invalidated=False,
            samples=[
                _sample("task_000", {"scorer": 1}),
                _sample("task_001", {"scorer": 0}),
            ],
        ),
        "c.json": SimpleNamespace(
            status="success",
            invalidated=False,
            samples=[
                _sample("task_000", {"scorer": 1}),
                _sample("task_001", {"scorer": 0}),
            ],
        ),
    }
    monkeypatch.setattr(stage1, "read_eval_log", lambda path: logs[Path(path).name])

    retry_log = stage1._find_retry_log(
        result_dir,
        {"task_000", "task_001", "task_002"},
    )
    assert retry_log is not None
    assert retry_log.name == "b.json"


def test_run_eval_stage1_honors_provided_eval_tag(tmp_path, monkeypatch):
    """Stage 1 should write outputs under an explicitly provided eval_tag."""
    output_base_dir = tmp_path / "outputs"
    exp_id = "exp_001"
    validation_tag = "_20260101_010101"
    datasets_dir = output_base_dir / exp_id / "eval" / "datasets" / validation_tag
    datasets_dir.mkdir(parents=True, exist_ok=True)
    (datasets_dir / "eval_config.json").write_text("{}")

    cfg = OmegaConf.create(
        {
            "exp_cfg": {"exp_id": exp_id},
            "global_cfg": {"output_dir": str(output_base_dir)},
        }
    )

    eval_config = EvalConfig(
        experiment_id=exp_id,
        eval_tag="",
        subject_llms=[{"name": "gpt-4o", "provider": "openai"}],
        judge_llm={"name": "gpt-4o-mini", "provider": "openai"},
        validation_tag=validation_tag,
    )

    dataset = EvalDataset(
        area_id="area_000",
        capability_id="cap_000",
        capability_name="compound_interest",
        domain="personal_finance",
        tasks=[
            {"id": "task_000", "input": "q1", "target": "a1"},
            {"id": "task_001", "input": "q2", "target": "a2"},
        ],
        num_tasks=2,
        prompt_template="template",
    )

    dataset_path = datasets_dir / "area_000" / "cap_000" / "dataset.json"
    calls: dict[str, object] = {}
    eval_calls: list[tuple[str, str, Path]] = []

    monkeypatch.setattr(stage1, "load_eval_config", lambda _p: (eval_config, None))
    monkeypatch.setattr(stage1, "_find_datasets", lambda _p: [dataset_path])
    monkeypatch.setattr(stage1, "load_eval_dataset", lambda _p: dataset)
    monkeypatch.setattr(stage1, "_check_eval_completed", lambda *args, **kwargs: False)

    def _fake_save_eval_config(config, metadata, output_path):
        calls["config"] = config
        calls["metadata"] = metadata
        calls["output_path"] = output_path

    def _fake_run_inspect_eval(dataset, subject_llm, judge_llm, output_dir):
        eval_calls.append((dataset.capability_id, subject_llm, output_dir))
        return True

    monkeypatch.setattr(stage1, "save_eval_config", _fake_save_eval_config)
    monkeypatch.setattr(stage1, "_run_inspect_eval", _fake_run_inspect_eval)

    eval_tag = stage1.run_eval_stage1(
        cfg, validation_tag=validation_tag, eval_tag="_existing_eval_tag"
    )

    assert eval_tag == "_existing_eval_tag"
    assert eval_config.eval_tag == "_existing_eval_tag"
    assert calls["output_path"] == (
        output_base_dir
        / exp_id
        / "eval"
        / "results"
        / "_existing_eval_tag"
        / "eval_config.json"
    )
    assert calls["metadata"].resume
    assert len(eval_calls) == 1


def test_run_eval_stage1_uses_inspect_retry_for_failed_log(tmp_path, monkeypatch):
    """Resume mode should use inspect eval_retry when a failed log exists."""
    output_base_dir = tmp_path / "outputs"
    exp_id = "exp_001"
    validation_tag = "_20260101_010101"
    datasets_dir = output_base_dir / exp_id / "eval" / "datasets" / validation_tag
    datasets_dir.mkdir(parents=True, exist_ok=True)
    (datasets_dir / "eval_config.json").write_text("{}")

    cfg = OmegaConf.create(
        {
            "exp_cfg": {"exp_id": exp_id},
            "global_cfg": {"output_dir": str(output_base_dir)},
        }
    )

    eval_config = EvalConfig(
        experiment_id=exp_id,
        eval_tag="",
        subject_llms=[{"name": "gpt-4o", "provider": "openai"}],
        judge_llm={"name": "gpt-4o-mini", "provider": "openai"},
        validation_tag=validation_tag,
    )
    dataset = EvalDataset(
        area_id="area_000",
        capability_id="cap_000",
        capability_name="compound_interest",
        domain="personal_finance",
        tasks=[
            {"id": "task_000", "input": "q1", "target": "a1"},
            {"id": "task_001", "input": "q2", "target": "a2"},
        ],
        num_tasks=2,
        prompt_template="template",
    )
    dataset_path = datasets_dir / "area_000" / "cap_000" / "dataset.json"

    retry_log = (
        output_base_dir
        / exp_id
        / "eval"
        / "results"
        / "_existing_eval_tag"
        / "gpt-4o"
        / "area_000"
        / "cap_000"
        / "failed.json"
    )
    retry_log.parent.mkdir(parents=True, exist_ok=True)
    retry_log.write_text("{}")

    check_calls = {"count": 0}
    retried: list[tuple[Path, Path]] = []

    monkeypatch.setattr(stage1, "load_eval_config", lambda _p: (eval_config, None))
    monkeypatch.setattr(stage1, "_find_datasets", lambda _p: [dataset_path])
    monkeypatch.setattr(stage1, "load_eval_dataset", lambda _p: dataset)
    monkeypatch.setattr(stage1, "save_eval_config", lambda *args, **kwargs: None)
    monkeypatch.setattr(stage1, "_find_retry_log", lambda *_args, **_kwargs: retry_log)

    def _fake_check(*_args, **_kwargs):
        check_calls["count"] += 1
        return check_calls["count"] >= 2

    def _fake_retry(retry_log_path: Path, output_dir: Path):
        retried.append((retry_log_path, output_dir))
        return True

    def _fail_fresh_eval(*_args, **_kwargs):
        raise AssertionError("Expected retry path, not fresh eval path")

    monkeypatch.setattr(stage1, "_check_eval_completed", _fake_check)
    monkeypatch.setattr(stage1, "_run_inspect_retry", _fake_retry)
    monkeypatch.setattr(stage1, "_run_inspect_eval", _fail_fresh_eval)

    eval_tag = stage1.run_eval_stage1(
        cfg, validation_tag=validation_tag, eval_tag="_existing_eval_tag"
    )

    assert eval_tag == "_existing_eval_tag"
    assert len(retried) == 1
    assert retried[0][0] == retry_log


def test_get_eval_dir_points_to_results_subdirectory():
    """Eval I/O helper should resolve to eval results path."""
    experiment_dir = Path("/tmp/example_exp")
    assert get_eval_dir(experiment_dir, "_20260101_010101") == (
        experiment_dir / "eval" / "results" / "_20260101_010101"
    )


def test_find_inspect_logs_json_only(tmp_path):
    """Stage helpers should discover only Inspect JSON logs."""
    result_dir = tmp_path / "area_000" / "cap_000"
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "a.json").write_text("{}")
    (result_dir / "b.eval").write_text("PK\x03\x04")

    stage1_logs = stage1._find_inspect_logs(result_dir)
    stage2_logs = stage2._find_inspect_logs(result_dir)

    assert [p.suffix for p in stage1_logs] == [".json"]
    assert [p.suffix for p in stage2_logs] == [".json"]
