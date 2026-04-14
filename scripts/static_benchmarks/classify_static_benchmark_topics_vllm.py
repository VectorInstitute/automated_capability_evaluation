#!/usr/bin/env python3
"""Classify static benchmark questions into finance capabilities with vLLM.

This script:
1) Loads one static benchmark through existing Stage-0 adapters.
2) Reads `topic.csv` and builds a high-level-area -> capabilities taxonomy.
3) Prompts a local model (e.g., Qwen3-32B) to return only one capability.
4) Saves outputs incrementally after every processed batch.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set

from src.eval_stages.stage0_static_benchmarks import _build_datasets_from_spec
from src.eval_stages.static_benchmarks.specs import StaticBenchmarkSpec


@dataclass(frozen=True)
class Taxonomy:
    """Prompt-ready taxonomy container."""

    high_level_areas: List[str]
    capabilities: List[str]
    area_to_capabilities: Dict[str, List[str]]
    prompt_block: str


def _read_topic_taxonomy(topic_csv_path: Path) -> Taxonomy:
    """Read high-level areas + capabilities from topic.csv."""
    area_to_caps_set: Dict[str, Set[str]] = {}

    # Be robust to UTF-8 BOM and leading blank/comment lines before header.
    with topic_csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))

    required = {"High Level Area", "Capability"}
    header_idx = None
    header: List[str] = []
    for idx, row in enumerate(rows):
        normalized = [str(cell).strip() for cell in row]
        if not any(normalized):
            continue
        if required.issubset(set(normalized)):
            header_idx = idx
            header = normalized
            break

    if header_idx is None:
        raise ValueError(
            f"Missing expected columns in {topic_csv_path}: {sorted(required)}"
        )

    for row in rows[header_idx + 1 :]:
        if not row or not any(str(cell).strip() for cell in row):
            continue
        rec = {
            header[i]: str(row[i]).strip() if i < len(row) else ""
            for i in range(len(header))
        }
        area = rec.get("High Level Area", "").strip()
        capability = rec.get("Capability", "").strip()
        if not area or not capability:
            continue
        area_to_caps_set.setdefault(area, set()).add(capability)

    if not area_to_caps_set:
        raise ValueError(f"No usable area/capability rows found in {topic_csv_path}")

    area_to_capabilities: Dict[str, List[str]] = {
        area: sorted(caps) for area, caps in sorted(area_to_caps_set.items())
    }
    high_level_areas = list(area_to_capabilities.keys())
    capabilities = sorted(
        {cap for caps in area_to_capabilities.values() for cap in caps}
    )

    lines: List[str] = []
    lines.append("High-level areas and their capabilities:")
    for area in high_level_areas:
        lines.append(f"- {area}:")
        for cap in area_to_capabilities[area]:
            lines.append(f"  - {cap}")
    prompt_block = "\n".join(lines)

    return Taxonomy(
        high_level_areas=high_level_areas,
        capabilities=capabilities,
        area_to_capabilities=area_to_capabilities,
        prompt_block=prompt_block,
    )


def _load_tasks_from_static_benchmark(
    *,
    benchmark_id: str,
    split: str,
    offset: int | None,
    limit: int | None,
) -> List[Dict[str, str]]:
    """Load tasks via the same adapters used by stage=0_static."""
    spec = StaticBenchmarkSpec(
        benchmark_id=benchmark_id,
        split=split,
        offset=offset,
        limit=limit,
        area_id="static_benchmarks",
        domain="finance",
    )
    datasets = _build_datasets_from_spec(spec)

    tasks: List[Dict[str, str]] = []
    for ds in datasets:
        for task in ds.tasks:
            tid = str(task.get("id", "")).strip()
            q = str(task.get("input", "")).strip()
            if tid and q:
                tasks.append({"id": tid, "question": q})
    return tasks


def _build_prompt(question: str, taxonomy: Taxonomy) -> str:
    """Build short-thinking classification prompt."""
    return (
        "You are classifying a finance question into ONE capability.\n"
        "Think very briefly.\n"
        "Choose exactly one capability from the provided list.\n"
        "Do not explain.\n"
        "Return exactly one line in this format:\n"
        "CAPABILITY: <exact capability name>\n\n"
        f"{taxonomy.prompt_block}\n\n"
        "Question:\n"
        f"{question}\n\n"
        "Only output the final capability line."
    )


def _extract_capability(raw_text: str, allowed_capabilities: Sequence[str]) -> str:
    """Parse model output and map to one allowed capability if possible."""
    text = (raw_text or "").strip()
    allowed_map = {cap.lower(): cap for cap in allowed_capabilities}

    m = re.search(r"(?im)^\s*CAPABILITY\s*:\s*(.+?)\s*$", text)
    if m:
        value = m.group(1).strip()
        if value.lower() in allowed_map:
            return allowed_map[value.lower()]
        text = value

    text_norm = re.sub(r"\s+", " ", text).strip().lower()
    if text_norm in allowed_map:
        return allowed_map[text_norm]

    # Fallback: find capability mention in output.
    # Longest-first reduces accidental partial matches.
    for cap in sorted(allowed_capabilities, key=len, reverse=True):
        if cap.lower() in text.lower():
            return cap

    return ""


def _batched(items: Sequence[Dict[str, str]], batch_size: int) -> Iterable[List[Dict[str, str]]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for i in range(0, len(items), batch_size):
        yield list(items[i : i + batch_size])


def _load_done_ids(output_jsonl: Path) -> Set[str]:
    """Read already-processed task IDs for resume support."""
    done: Set[str] = set()
    if not output_jsonl.exists():
        return done
    with output_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            tid = str(row.get("id", "")).strip()
            if tid:
                done.add(tid)
    return done


def run(args: argparse.Namespace) -> None:
    # Safer default in many cluster environments.
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    try:
        from transformers import PreTrainedTokenizerBase
        from vllm import LLM, SamplingParams
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("This script requires transformers and vllm.") from exc

    # Compatibility shim for some tokenizer/vLLM combinations.
    if not hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
        PreTrainedTokenizerBase.all_special_tokens_extended = property(  # type: ignore[attr-defined]
            lambda self: list(self.all_special_tokens)
        )

    topic_csv_path = Path(args.topic_csv).resolve()
    output_jsonl = Path(args.output_jsonl).resolve()
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    taxonomy = _read_topic_taxonomy(topic_csv_path)
    tasks = _load_tasks_from_static_benchmark(
        benchmark_id=args.benchmark_id,
        split=args.split,
        offset=args.offset,
        limit=args.limit,
    )
    if not tasks:
        raise ValueError("No tasks loaded from benchmark.")

    done_ids = _load_done_ids(output_jsonl) if args.resume else set()
    pending_tasks = [task for task in tasks if task["id"] not in done_ids]

    print(
        f"Loaded {len(tasks)} tasks; pending={len(pending_tasks)}; "
        f"already_done={len(done_ids)}"
    )
    print(f"Model: {args.model_path}")
    print(f"Benchmark: {args.benchmark_id} (split={args.split})")
    print(f"Output: {output_jsonl}")

    if not pending_tasks:
        print("Nothing to do.")
        return

    llm = LLM(
        model=args.model_path,
        tokenizer=args.model_path,
        trust_remote_code=args.trust_remote_code,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
    )
    tokenizer = llm.get_tokenizer() if hasattr(llm, "get_tokenizer") else None

    sampling = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.max_tokens,
        repetition_penalty=1.0,
    )

    total = len(pending_tasks)
    processed = 0
    with output_jsonl.open("a", encoding="utf-8") as out_f:
        for batch_idx, batch in enumerate(_batched(pending_tasks, args.batch_size), start=1):
            prompts: List[str] = []
            for row in batch:
                user_prompt = _build_prompt(row["question"], taxonomy)
                if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
                    text_prompt = tokenizer.apply_chat_template(
                        [{"role": "user", "content": user_prompt}],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                else:
                    text_prompt = user_prompt
                prompts.append(text_prompt)

            outputs = llm.generate(prompts, sampling)
            for row, output in zip(batch, outputs, strict=True):
                raw = output.outputs[0].text.strip() if output.outputs else ""
                predicted = _extract_capability(raw, taxonomy.capabilities)
                record = {
                    "id": row["id"],
                    "question": row["question"],
                    "predicted_capability": predicted,
                    "model_output_raw": raw,
                    "model_name": "Qwen3-32B",
                    "benchmark_id": args.benchmark_id,
                    "split": args.split,
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()

            processed += len(batch)
            print(
                f"[batch {batch_idx}] wrote {len(batch)} rows | "
                f"progress {processed}/{total}"
            )

    print("Done.")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Classify one static benchmark into finance capabilities with vLLM."
    )
    p.add_argument("--benchmark-id", required=True, help="Static benchmark ID or local JSON path.")
    p.add_argument("--split", default="test", help="Benchmark split (default: test).")
    p.add_argument("--offset", type=int, default=None, help="Optional benchmark offset.")
    p.add_argument("--limit", type=int, default=None, help="Optional benchmark limit.")
    p.add_argument(
        "--topic-csv",
        default="topic.csv",
        help="Path to topic.csv with High Level Area and Capability columns.",
    )
    p.add_argument(
        "--output-jsonl",
        required=True,
        help="Where to append classification rows (JSONL).",
    )
    p.add_argument("--resume", action="store_true", help="Skip IDs already in output JSONL.")

    # Model / vLLM args
    p.add_argument("--model-path", default="/model-weights/Qwen3-32B")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-tokens", type=int, default=32)
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--dtype", default="auto")
    p.add_argument("--max-model-len", type=int, default=8192)
    return p


if __name__ == "__main__":
    parser = build_arg_parser()
    run(parser.parse_args())
