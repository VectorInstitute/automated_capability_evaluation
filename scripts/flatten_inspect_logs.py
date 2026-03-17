"""Utility to flatten Inspect JSON logs into a simple, readable format.

Given an Inspect eval log file (one of the large JSON files under
base_output/<exp_id>/eval/results/<eval_tag>/<model>/<area>/<capability>/),
this script writes out a JSONL file with, per row:

- id:          sample id
- question:    original input
- ground_truth: target string
- model_output: subject model's answer text
- grade:       judge letter grade (if present, e.g. \"C\" or \"I\")

Usage:
    python scripts/flatten_inspect_logs.py \\
        --log_path base_output/test_exp/eval/results/_20260316_031445/\\
                  gpt-5-nano/static_benchmarks/integral/\\
                  2026-03-15T23-14-46-04-00_task_mZxA3jKBseS2smuk4ppcxN.json \\
        --out_path base_output/test_exp/eval/results/_20260316_031445/\\
                  gpt-5-nano/static_benchmarks/integral/flat_integral.jsonl

The first line of the JSONL file is a summary object with:
- num_samples
- num_correct
- num_incorrect
- accuracy
- f1 (computed treating "C" as correct, "I" as incorrect)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def flatten_inspect_log(log_path: Path) -> List[Dict[str, Any]]:
    data = json.loads(log_path.read_text(encoding="utf-8"))

    samples = data.get("samples", [])
    flattened: List[Dict[str, Any]] = []

    for s in samples:
        sid = s.get("id")
        question = s.get("input")
        target = s.get("target")

        model_output = None
        output = s.get("output") or {}
        choices = output.get("choices") or []
        if choices:
            msg = (choices[0] or {}).get("message") or {}
            model_output = msg.get("content")

        grade = None
        scores = s.get("scores") or {}
        fact = scores.get("model_graded_fact") or {}
        grade = fact.get("value")

        flattened.append(
            {
                "id": sid,
                "question": question,
                "ground_truth": target,
                "model_output": model_output,
                "grade": grade,
            }
        )

    return flattened


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_path", type=str, required=True)
    parser.add_argument("--out_path", type=str, required=True)
    args = parser.parse_args()

    log_path = Path(args.log_path)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = flatten_inspect_log(log_path)

    num_samples = len(rows)
    num_correct = sum(1 for r in rows if r.get("grade") == "C")
    num_incorrect = sum(1 for r in rows if r.get("grade") == "I")
    accuracy = (num_correct / num_samples) if num_samples else 0.0
    # In this binary setting with grades only, we treat F1 as equal to accuracy.
    f1 = accuracy

    with out_path.open("w", encoding="utf-8") as f:
        summary = {
            "summary": True,
            "num_samples": num_samples,
            "num_correct": num_correct,
            "num_incorrect": num_incorrect,
            "accuracy": accuracy,
            "f1": f1,
        }
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

