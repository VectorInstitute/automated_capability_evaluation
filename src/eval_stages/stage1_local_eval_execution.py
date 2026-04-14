"""Eval Stage 1_local: direct evaluation without Inspect.

This stage runs subject models directly, including local HuggingFace models
loaded from disk via `provider: hf_local`. Local HF models can run through
`transformers` or `vllm`, then each response is judged and written to the final
`flat_<capability>.jsonl` output expected by downstream workflows.
"""

from __future__ import annotations

import json
import logging
import os
import gc
import asyncio
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

from omegaconf import DictConfig
import torch
from tqdm.auto import tqdm
import re

from src.model import Model
from src.schemas.eval_io_utils import load_eval_config, load_eval_dataset, save_eval_config
from src.schemas.eval_schemas import EvalDataset
from src.schemas.metadata_schemas import PipelineMetadata
from src.utils.inspect_eval_utils import LLM_JUDGE_PROMPT, parse_submission
from src.utils.timestamp_utils import iso_timestamp, timestamp_tag

logger = logging.getLogger(__name__)


def _find_datasets(datasets_dir: Path) -> List[Path]:
    """Return all Stage 0 dataset files."""
    if not datasets_dir.exists():
        return []
    return sorted(datasets_dir.rglob("dataset.json"))


def _flat_result_path(output_dir: Path, capability_id: str) -> Path:
    return output_dir / f"flat_{capability_id}.jsonl"


def _read_flat_rows(flat_path: Path) -> List[Dict[str, Any]]:
    """Read non-summary rows from a flat jsonl file."""
    if not flat_path.exists():
        return []

    rows: List[Dict[str, Any]] = []
    with open(flat_path, "r", encoding="utf-8") as f:
        # Skip summary line
        try:
            next(f)
        except StopIteration:
            return []
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _check_flat_completed(flat_path: Path, expected_task_ids: Set[str]) -> bool:
    """Return True if flat file has exactly the expected task IDs."""
    if not flat_path.exists() or not expected_task_ids:
        return False
    rows = _read_flat_rows(flat_path)
    row_ids = {str(row.get("id", "")) for row in rows if row.get("id") is not None}
    return row_ids == expected_task_ids


def _write_flat_results(output_path: Path, rows: List[Dict[str, Any]]) -> None:
    """Write rows in the same schema as flatten_inspect_logs.py."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    num_samples = len(rows)
    num_correct = sum(1 for row in rows if row.get("grade") == "C")
    num_incorrect = sum(1 for row in rows if row.get("grade") == "I")
    accuracy = (num_correct / num_samples) if num_samples else 0.0
    f1 = accuracy

    with open(output_path, "w", encoding="utf-8") as f:
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


def _format_prompt(dataset: EvalDataset, task: Dict[str, str]) -> str:
    """Render the Stage 0 prompt template for a task."""
    template = dataset.prompt_template or "{input}"
    try:
        prompt = template.format(input=task["input"])
    except Exception:  # noqa: BLE001
        prompt = str(task["input"])

    is_mcq = bool(re.search(r"(?im)^\s*options\s*:\s*$", str(task.get("input", ""))))
    if is_mcq:
        answer_instruction = (
            "\n\nReason briefly and do not repeat yourself. Stop immediately after the final "
            "answer line.\n\nThis is a multiple-choice question. On the last line, return ONLY "
            "the option letter in machine-readable form as `ANSWER: <LETTER>` "
            "(e.g., `ANSWER: B`). Do NOT return a number, currency amount, or explanation "
            "on the final answer line."
        )
    else:
        answer_instruction = (
            "\n\nReason briefly and do not repeat yourself. Stop immediately after the final "
            "answer line.\n\nReturn your final answer in a machine-readable form on the last "
            "line as `ANSWER: <final answer>`."
        )
    return prompt + answer_instruction


def _build_model(model_config: Dict[str, Any]) -> Model:
    """Instantiate a repo Model from eval subject/judge config."""
    model_kwargs = {
        key: value
        for key, value in model_config.items()
        if key not in {"name", "provider", "generation_cfg"}
    }
    return Model(
        model_name=str(model_config["name"]),
        model_provider=str(model_config.get("provider", "openai")),
        **model_kwargs,
    )


def _is_hf_local_provider(provider: str) -> bool:
    """Return True for direct HuggingFace local model providers."""
    return provider in {"hf_local", "local_hf", "transformers"}


def _uses_vllm_backend(model_config: Dict[str, Any]) -> bool:
    """Return True when a local HF model should run via vLLM."""
    backend = str(model_config.get("inference_backend", "transformers")).lower()
    return _is_hf_local_provider(str(model_config.get("provider", ""))) and (
        backend == "vllm"
    )


def _build_messages(sys_prompt: str, user_prompt: str) -> List[Dict[str, str]]:
    """Build chat-style messages for subject generation."""
    messages: List[Dict[str, str]] = []
    if sys_prompt.strip():
        messages.append({"role": "system", "content": sys_prompt})
    messages.append({"role": "user", "content": user_prompt})
    return messages


def _render_text_prompt(tokenizer: Any, *, sys_prompt: str, user_prompt: str) -> str:
    """Render a text prompt, using chat templates when available."""
    messages = _build_messages(sys_prompt, user_prompt)
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    if sys_prompt.strip():
        return f"{sys_prompt.strip()}\n\n{user_prompt}".strip()
    return user_prompt


def _load_hf_local_model(
    model_config: Dict[str, Any],
) -> Tuple[Any, Any]:
    """Load a local HuggingFace causal LM and tokenizer."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "transformers is required for provider=hf_local in stage=1_local"
        ) from exc

    model_path = model_config.get("model_path")
    if not model_path:
        raise ValueError(
            "provider=hf_local requires `model_path` in subject_llms config"
        )

    trust_remote_code = bool(model_config.get("trust_remote_code", True))
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
    )

    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    if torch.cuda.is_available():
        torch_dtype = torch.bfloat16
        device_map = model_config.get("device_map", "auto")
    else:
        torch_dtype = torch.float32
        device_map = model_config.get("device_map", None)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )
    model.eval()
    return tokenizer, model


def _load_vllm_model(model_config: Dict[str, Any]) -> Any:
    """Load a local vLLM engine from disk."""
    try:
        from transformers import PreTrainedTokenizerBase
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "transformers is required for inference_backend=vllm in stage=1_local"
        ) from exc

    # vLLM 0.8.x still expects this tokenizer property, but it is missing in
    # newer transformers builds used by Qwen tokenizers in this environment.
    if not hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
        PreTrainedTokenizerBase.all_special_tokens_extended = property(  # type: ignore[attr-defined]
            lambda self: list(self.all_special_tokens)
        )

    # vLLM can crash when CUDA is initialized from forked worker processes.
    # Default to the safer spawn mode unless the user explicitly overrides it.
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    try:
        from vllm import LLM
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "vllm is required for inference_backend=vllm in stage=1_local"
        ) from exc

    model_path = model_config.get("model_path")
    if not model_path:
        raise ValueError(
            "inference_backend=vllm requires `model_path` in subject_llms config"
        )

    llm_kwargs: Dict[str, Any] = {
        "model": model_path,
        "tokenizer": model_path,
        "trust_remote_code": bool(model_config.get("trust_remote_code", True)),
        "tensor_parallel_size": int(model_config.get("tensor_parallel_size", 1)),
        "gpu_memory_utilization": float(
            model_config.get("gpu_memory_utilization", 0.9)
        ),
        "dtype": model_config.get("dtype", "auto"),
    }

    if "max_model_len" in model_config:
        llm_kwargs["max_model_len"] = int(model_config["max_model_len"])
    if "enforce_eager" in model_config:
        llm_kwargs["enforce_eager"] = bool(model_config["enforce_eager"])

    return LLM(**llm_kwargs)


def _wait_for_vllm_startup_memory(
    gpu_memory_utilization: float, timeout_seconds: float = 90.0
) -> None:
    """Wait until enough free GPU memory is available for vLLM startup."""
    if not torch.cuda.is_available():
        return

    # Clamp to sensible bounds in case config has bad values.
    target_util = min(max(float(gpu_memory_utilization), 0.0), 1.0)
    deadline = time.monotonic() + timeout_seconds
    required_gib = None
    latest_free_gib = None
    total_gib = None

    while time.monotonic() < deadline:
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        latest_free_gib = free_bytes / (1024**3)
        total_gib = total_bytes / (1024**3)
        required_gib = target_util * total_gib
        if latest_free_gib >= required_gib:
            return
        time.sleep(2.0)

    if required_gib is not None and latest_free_gib is not None and total_gib is not None:
        logger.warning(
            (
                "Proceeding with vLLM load before target free GPU memory recovered "
                "(free=%.2f GiB, total=%.2f GiB, required=%.2f GiB, utilization=%.2f)."
            ),
            latest_free_gib,
            total_gib,
            required_gib,
            target_util,
        )


def _teardown_vllm_engine(vllm_engine: Any, model_name: str) -> None:
    """Shut down a vLLM ``LLM`` instance and free its GPU memory.

    The ``LLM`` object holds ``llm_engine`` (an ``LLMEngine``), which in turn
    holds ``engine_core`` (a ``SyncMPClient`` / ``MPClient``).  The EngineCore
    runs in a **separate process** that owns the actual GPU tensors, so we must
    call ``engine_core.shutdown()`` to terminate that process — ``del`` alone
    is not enough.
    """
    # 1. Graceful shutdown via the engine_core subprocess manager.
    try:
        llm_engine = getattr(vllm_engine, "llm_engine", None)
        if llm_engine is not None:
            engine_core = getattr(llm_engine, "engine_core", None)
            if engine_core is not None and hasattr(engine_core, "shutdown"):
                logger.info("  Calling engine_core.shutdown() for %s", model_name)
                engine_core.shutdown()
    except Exception as exc:  # noqa: BLE001
        logger.warning("engine_core.shutdown() failed for %s: %s", model_name, exc)

    # 2. Delete Python references so the GC can collect any remaining C++ handles.
    try:
        del vllm_engine
    except Exception:  # noqa: BLE001
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 3. Give the EngineCore subprocess time to exit and release GPU memory.
    time.sleep(5.0)


def _normalize_text(text: str) -> str:
    return " ".join(text.strip().split())


def _last_sentence(text: str) -> str:
    """
    Extract the last "sentence-like" fragment from model output.

    For this project we approximate "sentence" as the last non-empty line,
    because many answers end in LaTeX blocks (e.g., `\\boxed{...}`) that may
    not be punctuation-terminated.
    """
    if not text:
        return ""
    lines = [line.strip() for line in str(text).splitlines() if line.strip()]
    if not lines:
        return ""
    last = lines[-1]
    # Common LaTeX terminators sometimes end up as the last line alone.
    if last in {"$$", "$"} and len(lines) >= 2:
        last = lines[-2]
    return last


def _parse_mcq_options(question: str) -> Dict[str, str]:
    """Parse MCQ options from a question string into {letter: option_text}."""
    if not question:
        return {}
    lines = question.splitlines()
    in_options = False
    options: Dict[str, str] = {}
    for line in lines:
        if re.match(r"(?im)^\s*options\s*:\s*$", line):
            in_options = True
            continue
        if not in_options:
            continue
        m = re.match(r"^\s*([A-Z])\s*[.)]\s*(.+?)\s*$", line.strip())
        if not m:
            # Stop when we leave the options block (blank line or non-option text)
            if options and not line.strip():
                break
            continue
        options[m.group(1).upper()] = m.group(2).strip()
    return options


def _extract_number(text: str) -> Optional[float]:
    """Extract a numeric value from text (handles commas and currency)."""
    if not text:
        return None
    m = re.search(r"[-+]?\d[\d,]*(?:\.\d+)?", text)
    if not m:
        return None
    try:
        return float(m.group(0).replace(",", ""))
    except ValueError:
        return None


def _map_numeric_answer_to_option_letter(
    *,
    submission: str,
    question: str,
    target: str,
    rel_tol: float = 1e-3,
) -> Optional[str]:
    """If target is a letter MCQ, map numeric submission to the closest matching option."""
    target_letter = target.strip().upper()
    if not re.fullmatch(r"[A-Z]", target_letter):
        return None

    options = _parse_mcq_options(question)
    if not options:
        return None

    sub_val = _extract_number(submission)
    if sub_val is None:
        return None

    best_letter: Optional[str] = None
    for letter, opt_text in options.items():
        opt_val = _extract_number(opt_text)
        if opt_val is None:
            continue
        denom = max(1.0, abs(opt_val))
        if abs(sub_val - opt_val) / denom <= rel_tol:
            best_letter = letter
            break
    return best_letter


def _generate_batch_with_hf_local(
    tokenizer: Any,
    model: Any,
    *,
    prompts: List[str],
    generation_config: Dict[str, Any],
) -> List[str]:
    """Generate a batch of responses with a local HF causal LM."""
    if not prompts:
        return []

    max_new_tokens = int(generation_config.get("max_tokens", 512))
    temperature = float(generation_config.get("temperature", 0.0) or 0.0)
    top_p = float(generation_config.get("top_p", 1.0) or 1.0)
    repetition_penalty = float(generation_config.get("repetition_penalty", 1.0) or 1.0)
    do_sample = temperature > 0

    encoded = tokenizer(prompts, return_tensors="pt", padding=True)
    input_ids = encoded["input_ids"]
    attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids))

    model_device = next(model.parameters()).device
    input_ids = input_ids.to(model_device)
    attention_mask = attention_mask.to(model_device)

    generate_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "repetition_penalty": repetition_penalty,
    }
    if do_sample:
        generate_kwargs["temperature"] = temperature
        generate_kwargs["top_p"] = top_p

    with torch.inference_mode():
        generated = model.generate(**generate_kwargs)

    prompt_token_count = input_ids.shape[-1]
    generated_texts: List[str] = []
    for row_tokens in generated:
        generated_tokens = row_tokens[prompt_token_count:]
        output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        generated_texts.append(output_text.strip())
    return generated_texts


def _generate_batch_with_vllm(
    llm: Any,
    *,
    prompts: List[str],
    generation_config: Dict[str, Any],
) -> List[str]:
    """Generate a batch of responses with vLLM."""
    try:
        from vllm import SamplingParams
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "vllm is required for inference_backend=vllm in stage=1_local"
        ) from exc

    sampling_params = SamplingParams(
        max_tokens=int(generation_config.get("max_tokens", 512)),
        temperature=float(generation_config.get("temperature", 0.0) or 0.0),
        top_p=float(generation_config.get("top_p", 1.0) or 1.0),
        repetition_penalty=float(
            generation_config.get("repetition_penalty", 1.0) or 1.0
        ),
    )
    outputs = llm.generate(prompts, sampling_params)

    generated_texts: List[str] = []
    for output in outputs:
        if output.outputs:
            generated_texts.append((output.outputs[0].text or "").strip())
        else:
            generated_texts.append("")
    return generated_texts


def _batched(
    items: List[Dict[str, Any]], batch_size: int
) -> Iterator[List[Dict[str, Any]]]:
    """Yield fixed-size batches from a list."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def _build_judge_prompt(submission: str, target: str) -> str:
    """Render the judge prompt for one submission/target pair."""
    return LLM_JUDGE_PROMPT.format(submission=submission, target=target)


def _judge_outputs_to_grades(outputs: List[str]) -> List[str]:
    """Convert judge outputs to C/I grades."""
    return [
        "C" if output and output.strip().lower().startswith("yes") else "I"
        for output in outputs
    ]


def _score_existing_row_ids(
    flat_path: Path, expected_task_ids: Set[str]
) -> Dict[str, Dict[str, Any]]:
    """Load previously scored rows and keep only expected task IDs."""
    row_by_id: Dict[str, Dict[str, Any]] = {}
    for row in _read_flat_rows(flat_path):
        row_id = str(row.get("id", ""))
        if (
            row_id
            and row_id in expected_task_ids
            and row.get("grade") in {"C", "I"}
            and row_id not in row_by_id
        ):
            row_by_id[row_id] = row
    return row_by_id


def _ordered_rows(
    tasks: List[Dict[str, Any]], row_by_id: Dict[str, Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Order scored rows to match the original dataset task order."""
    return [
        row_by_id[str(task["id"])]
        for task in tasks
        if str(task["id"]) in row_by_id
    ]


def _log_running_performance(
    *,
    llm_name: str,
    capability_id: str,
    row_by_id: Dict[str, Dict[str, Any]],
    total_tasks: int,
) -> None:
    """Log running completion and accuracy for current model/capability."""
    done = len(row_by_id)
    if done == 0:
        acc = 0.0
    else:
        correct = sum(1 for row in row_by_id.values() if row.get("grade") == "C")
        acc = correct / done
    logger.info(
        "  Progress %s/%s: %d/%d scored | running_accuracy=%.4f",
        llm_name,
        capability_id,
        done,
        total_tasks,
        acc,
    )


def _judge_batch(
    rows: List[Dict[str, Any]],
    *,
    judge_generation_cfg: Dict[str, Any],
    judge_model: Optional[Model] = None,
    judge_tokenizer: Any = None,
    judge_hf_model: Any = None,
    judge_vllm_model: Any = None,
    judge_vllm_tokenizer: Any = None,
    max_concurrent_requests: int = 8,
) -> List[Dict[str, Any]]:
    """Judge a batch of rows, using exact-match shortcuts when possible."""
    if not rows:
        return []

    scored_rows: List[Optional[Dict[str, Any]]] = [None] * len(rows)
    unresolved_indices: List[int] = []
    unresolved_prompts: List[str] = []
    unresolved_task_ids: List[str] = []

    for index, row in enumerate(rows):
        raw_output = str(row["model_output"])
        parsed_submission = parse_submission(raw_output) or raw_output
        judge_submission = _last_sentence(raw_output) or parsed_submission
        target = str(row["ground_truth"])
        # If this is an MCQ with a letter target, allow mapping a numeric final answer
        # back to an option letter based on the question's options.
        mapped_letter = _map_numeric_answer_to_option_letter(
            submission=parsed_submission,
            question=str(row.get("question", "")),
            target=target,
        )
        if mapped_letter is not None:
            parsed_submission = mapped_letter
        if _normalize_text(parsed_submission).lower() == _normalize_text(target).lower():
            scored_rows[index] = {**row, "grade": "C"}
            continue
        unresolved_indices.append(index)
        unresolved_task_ids.append(str(row.get("id", "")))
        # Give the judge only the model's final fragment to reduce noise.
        judge_prompt = _build_judge_prompt(judge_submission, target)
        unresolved_prompts.append(judge_prompt)

    if unresolved_prompts:
        if judge_vllm_model is not None:
            prompts = [
                _render_text_prompt(
                    judge_vllm_tokenizer,
                    sys_prompt="You are a careful, non-pedantic grading assistant.",
                    user_prompt=prompt,
                )
                for prompt in unresolved_prompts
            ]
            judge_outputs = _generate_batch_with_vllm(
                judge_vllm_model,
                prompts=prompts,
                generation_config=judge_generation_cfg,
            )
        elif judge_hf_model is not None:
            prompts = [
                _render_text_prompt(
                    judge_tokenizer,
                    sys_prompt="You are a careful, non-pedantic grading assistant.",
                    user_prompt=prompt,
                )
                for prompt in unresolved_prompts
            ]
            judge_outputs = _generate_batch_with_hf_local(
                judge_tokenizer,
                judge_hf_model,
                prompts=prompts,
                generation_config=judge_generation_cfg,
            )
        else:
            if judge_model is None:
                raise ValueError("judge_model is required when no local judge backend is set")
            async def _run_async_judge(prompts: List[str]) -> List[str]:
                sem = asyncio.Semaphore(max(1, int(max_concurrent_requests)))

                async def _one(p: str) -> str:
                    async with sem:
                        txt, _ = await judge_model.async_generate(
                            sys_prompt="You are a careful, non-pedantic grading assistant.",
                            user_prompt=p,
                            generation_config=judge_generation_cfg,
                        )
                        return txt or ""

                return list(await asyncio.gather(*(_one(p) for p in prompts)))

            try:
                judge_outputs = asyncio.run(_run_async_judge(unresolved_prompts))
            except Exception:
                # Fallback to synchronous calls if async event loop issues occur.
                judge_outputs = []
                for prompt in unresolved_prompts:
                    judge_text, _ = judge_model.generate(
                        sys_prompt="You are a careful, non-pedantic grading assistant.",
                        user_prompt=prompt,
                        generation_config=judge_generation_cfg,
                    )
                    judge_outputs.append(judge_text or "")

        for index, grade in zip(
            unresolved_indices,
            _judge_outputs_to_grades(judge_outputs),
            strict=True,
        ):
            scored_rows[index] = {**rows[index], "grade": grade}
    return [row for row in scored_rows if row is not None]


def run_eval_stage1_local(
    cfg: DictConfig,
    validation_tag: str,
    eval_tag: Optional[str] = None,
) -> str:
    """Run local/direct Stage 1 evals and return eval_tag."""
    exp_id = cfg.exp_cfg.exp_id
    output_base_dir = Path(cfg.global_cfg.output_dir)
    experiment_dir = output_base_dir / exp_id

    datasets_dir = experiment_dir / "eval" / "datasets" / validation_tag
    eval_config_path = datasets_dir / "eval_config.json"
    if not eval_config_path.exists():
        raise ValueError(
            f"eval_config.json not found at {eval_config_path}. Run Stage 0 first."
        )
    eval_config, _ = load_eval_config(eval_config_path)

    is_resume = eval_tag is not None
    if eval_tag is None:
        eval_tag = timestamp_tag()

    logger.info(
        "Eval Stage 1_local: Running direct evaluations (eval_tag=%s, resume=%s)",
        eval_tag,
        is_resume,
    )

    dataset_paths = _find_datasets(datasets_dir)
    logger.info("Found %d datasets", len(dataset_paths))
    if not dataset_paths:
        raise ValueError(f"No datasets found in {datasets_dir}. Run Stage 0 first.")

    datasets = [load_eval_dataset(p) for p in dataset_paths]

    eval_dir = experiment_dir / "eval" / "results" / eval_tag
    results_dir = eval_dir

    eval_config.eval_tag = eval_tag
    metadata = PipelineMetadata(
        experiment_id=exp_id,
        output_base_dir=str(output_base_dir),
        timestamp=iso_timestamp(),
        input_stage_tag=validation_tag,
        output_stage_tag=eval_tag,
        resume=is_resume,
    )
    results_config_path = eval_dir / "eval_config.json"
    save_eval_config(eval_config, metadata, results_config_path)
    logger.info("Saved eval_config.json to %s", results_config_path)

    subject_llms = eval_config.subject_llms
    judge_llm_cfg = dict(eval_config.judge_llm)
    judge_generation_cfg = dict(judge_llm_cfg.get("generation_cfg", {}))
    if "max_tokens" not in judge_generation_cfg:
        judge_generation_cfg["max_tokens"] = 16
    if "temperature" not in judge_generation_cfg:
        judge_generation_cfg["temperature"] = 0
    judge_provider = str(judge_llm_cfg.get("provider", "openai"))
    judge_batch_size = int(judge_llm_cfg.get("batch_size", 32))
    judge_using_vllm = _uses_vllm_backend(judge_llm_cfg)
    judge_model: Optional[Model] = None
    judge_tokenizer: Any = None
    judge_hf_model: Any = None
    # IMPORTANT: if judge is vLLM, we load it lazily per combination to avoid
    # having subject-vLLM and judge-vLLM resident at the same time.
    judge_vllm_model: Any = None
    judge_vllm_tokenizer: Any = None
    if _is_hf_local_provider(judge_provider) and not judge_using_vllm:
        logger.info("Loading local HF judge %s", judge_llm_cfg["name"])
        judge_tokenizer, judge_hf_model = _load_hf_local_model(judge_llm_cfg)
    elif not judge_using_vllm:
        judge_model = _build_model(judge_llm_cfg)

    model_instances: Dict[Tuple[str, str], Model] = {}
    hf_model_instances: Dict[Tuple[str, str], Tuple[Any, Any]] = {}
    vllm_model_instances: Dict[Tuple[str, str], Any] = {}

    num_completed_this_run = 0
    num_skipped_completed = 0
    num_failed = 0
    num_incomplete = 0
    total_combinations = len(datasets) * len(subject_llms)

    combination_index = 0
    for dataset in datasets:
        expected_task_ids = {str(task["id"]) for task in dataset.tasks}
        for llm_config in subject_llms:
            combination_index += 1
            llm_name = str(llm_config["name"])
            llm_provider = str(llm_config.get("provider", "openai"))
            using_vllm = _uses_vllm_backend(dict(llm_config))
            logger.info(
                "Combination %d/%d: Evaluating %s/%s with %s/%s%s",
                combination_index,
                total_combinations,
                dataset.area_id,
                dataset.capability_id,
                llm_provider,
                llm_name,
                " [vllm]" if using_vllm else "",
            )

            output_dir = results_dir / llm_name / dataset.area_id / dataset.capability_id
            flat_path = _flat_result_path(output_dir, dataset.capability_id)

            if _check_flat_completed(flat_path, expected_task_ids):
                logger.info(
                    "  Skipping %s/%s with %s (already completed)",
                    dataset.area_id,
                    dataset.capability_id,
                    llm_name,
                )
                num_skipped_completed += 1
                continue

            model_key = (llm_provider, llm_name)
            subject_generation_cfg = dict(llm_config.get("generation_cfg", {}))
            total_tasks = len(dataset.tasks)
            batch_size = int(llm_config.get("batch_size", 16))
            row_by_id = _score_existing_row_ids(flat_path, expected_task_ids)
            pending_tasks = [
                task for task in dataset.tasks if str(task["id"]) not in row_by_id
            ]

            if row_by_id:
                logger.info(
                    "  Resuming %s/%s with %d/%d tasks already scored",
                    dataset.area_id,
                    dataset.capability_id,
                    len(row_by_id),
                    total_tasks,
                )
                _write_flat_results(flat_path, _ordered_rows(dataset.tasks, row_by_id))

            if not pending_tasks:
                logger.info(
                    "  Skipping %s/%s with %s (all tasks already scored)",
                    dataset.area_id,
                    dataset.capability_id,
                    llm_name,
                )
                num_skipped_completed += 1
                continue

            if using_vllm:
                if model_key not in vllm_model_instances:
                    for old_key in list(vllm_model_instances):
                        if old_key != model_key:
                            logger.info(
                                "  Tearing down previous vLLM engine %s before loading %s",
                                old_key[1], llm_name,
                            )
                            old_engine = vllm_model_instances.pop(old_key, None)
                            if old_engine is not None:
                                _teardown_vllm_engine(old_engine, old_key[1])
                    _wait_for_vllm_startup_memory(
                        float(llm_config.get("gpu_memory_utilization", 0.9))
                    )
                    logger.info("  Loading vLLM engine for %s", llm_name)
                    vllm_model_instances[model_key] = _load_vllm_model(dict(llm_config))
                vllm_model = vllm_model_instances[model_key]
            elif _is_hf_local_provider(llm_provider):
                if model_key not in hf_model_instances:
                    hf_model_instances[model_key] = _load_hf_local_model(
                        dict(llm_config)
                    )
                tokenizer, hf_model = hf_model_instances[model_key]
            else:
                if model_key not in model_instances:
                    model_instances[model_key] = _build_model(dict(llm_config))
                subject_model = model_instances[model_key]

            success = True
            failed_task_id = None
            try:
                logger.info(
                    "  Processing %d pending tasks (subject_batch_size=%d, judge_batch_size=%d)",
                    len(pending_tasks),
                    batch_size,
                    judge_batch_size,
                )
                subject_tokenizer = None
                if using_vllm and hasattr(vllm_model, "get_tokenizer"):
                    subject_tokenizer = vllm_model.get_tokenizer()

                # If BOTH subject and judge are vLLM, avoid dual-engine residency:
                # - If they point to the same model_path, reuse the subject engine for judging.
                # - Otherwise, generate everything first, free subject engine, then start judge.
                judge_needs_serialization = bool(judge_using_vllm and using_vllm)
                can_reuse_subject_as_judge = False
                if judge_needs_serialization:
                    subj_path = str(dict(llm_config).get("model_path", ""))
                    judge_path = str(judge_llm_cfg.get("model_path", ""))
                    can_reuse_subject_as_judge = bool(subj_path and judge_path and subj_path == judge_path)

                if judge_needs_serialization and not can_reuse_subject_as_judge:
                    # Phase A: generate all pending outputs (no judging yet)
                    all_generated: List[Dict[str, Any]] = []
                    with tqdm(
                        total=len(pending_tasks),
                        desc=f"Generate {llm_name}/{dataset.capability_id}",
                        dynamic_ncols=True,
                    ) as gen_bar:
                        for task_batch in _batched(pending_tasks, batch_size):
                            failed_task_id = task_batch[0].get("id")
                            prompts = [
                                _render_text_prompt(
                                    subject_tokenizer,
                                    sys_prompt="",
                                    user_prompt=_format_prompt(dataset, task),
                                )
                                for task in task_batch
                            ]
                            generated_texts = _generate_batch_with_vllm(
                                vllm_model,
                                prompts=prompts,
                                generation_config=subject_generation_cfg,
                            )
                            for task, generated_text in zip(task_batch, generated_texts, strict=True):
                                all_generated.append(
                                    {
                                        "id": task["id"],
                                        "question": task["input"],
                                        "ground_truth": task["target"],
                                        "model_output": generated_text,
                                    }
                                )
                            gen_bar.update(len(task_batch))

                    # Tear down subject vLLM before starting judge vLLM
                    subject_engine = vllm_model_instances.pop(model_key, None)
                    if subject_engine is not None:
                        _teardown_vllm_engine(subject_engine, llm_name)
                    _wait_for_vllm_startup_memory(
                        float(judge_llm_cfg.get("gpu_memory_utilization", 0.9))
                    )

                    # Phase B: start judge vLLM and judge in batches
                    logger.info("  Loading vLLM judge (after subject generation teardown)")
                    judge_vllm_model = _load_vllm_model(judge_llm_cfg)
                    judge_vllm_tokenizer = (
                        judge_vllm_model.get_tokenizer()
                        if hasattr(judge_vllm_model, "get_tokenizer")
                        else None
                    )
                    with tqdm(
                        total=len(all_generated),
                        desc=f"Judge {llm_name}/{dataset.capability_id}",
                        dynamic_ncols=True,
                    ) as judge_bar:
                        for judge_batch in _batched(all_generated, judge_batch_size):
                            failed_task_id = judge_batch[0].get("id")
                            scored_batch = _judge_batch(
                                judge_batch,
                                judge_generation_cfg=judge_generation_cfg,
                                judge_model=judge_model,
                                judge_tokenizer=judge_tokenizer,
                                judge_hf_model=judge_hf_model,
                                judge_vllm_model=judge_vllm_model,
                                judge_vllm_tokenizer=judge_vllm_tokenizer,
                                max_concurrent_requests=judge_batch_size,
                            )
                            for scored_row in scored_batch:
                                row_by_id[str(scored_row["id"])] = scored_row
                            _write_flat_results(flat_path, _ordered_rows(dataset.tasks, row_by_id))
                            _log_running_performance(
                                llm_name=llm_name,
                                capability_id=dataset.capability_id,
                                row_by_id=row_by_id,
                                total_tasks=total_tasks,
                            )
                            judge_bar.update(len(judge_batch))

                    # Tear down judge vLLM too
                    _teardown_vllm_engine(judge_vllm_model, str(judge_llm_cfg.get("name", "judge")))
                    judge_vllm_model = None
                    judge_vllm_tokenizer = None
                else:
                    # Default fast path: generate + judge streaming (can reuse subject engine as judge if same model)
                    if judge_using_vllm and using_vllm and can_reuse_subject_as_judge:
                        judge_vllm_model = vllm_model
                        judge_vllm_tokenizer = subject_tokenizer
                    elif judge_using_vllm and judge_vllm_model is None:
                        logger.info("  Loading vLLM judge %s", judge_llm_cfg["name"])
                        judge_vllm_model = _load_vllm_model(judge_llm_cfg)
                        judge_vllm_tokenizer = (
                            judge_vllm_model.get_tokenizer()
                            if hasattr(judge_vllm_model, "get_tokenizer")
                            else None
                        )

                    with tqdm(
                        total=total_tasks,
                        initial=len(row_by_id),
                        desc=f"Eval {llm_name}/{dataset.capability_id}",
                        dynamic_ncols=True,
                    ) as eval_bar:
                        for task_batch in _batched(pending_tasks, batch_size):
                            failed_task_id = task_batch[0].get("id")
                            if using_vllm:
                                prompts = [
                                    _render_text_prompt(
                                        subject_tokenizer,
                                        sys_prompt="",
                                        user_prompt=_format_prompt(dataset, task),
                                    )
                                    for task in task_batch
                                ]
                                generated_texts = _generate_batch_with_vllm(
                                    vllm_model,
                                    prompts=prompts,
                                    generation_config=subject_generation_cfg,
                                )
                            elif _is_hf_local_provider(llm_provider):
                                prompts = [
                                    _render_text_prompt(
                                        tokenizer,
                                        sys_prompt="",
                                        user_prompt=_format_prompt(dataset, task),
                                    )
                                    for task in task_batch
                                ]
                                generated_texts = _generate_batch_with_hf_local(
                                    tokenizer,
                                    hf_model,
                                    prompts=prompts,
                                    generation_config=subject_generation_cfg,
                                )
                            else:
                                generated_texts = []
                                for task in task_batch:
                                    failed_task_id = task.get("id")
                                    prompt = _format_prompt(dataset, task)
                                    generated_text, _ = subject_model.generate(
                                        sys_prompt="",
                                        user_prompt=prompt,
                                        generation_config=subject_generation_cfg,
                                    )
                                    generated_texts.append(generated_text or "")

                            generated_rows = [
                                {
                                    "id": task["id"],
                                    "question": task["input"],
                                    "ground_truth": task["target"],
                                    "model_output": generated_text,
                                }
                                for task, generated_text in zip(
                                    task_batch, generated_texts, strict=True
                                )
                            ]

                            for jb in _batched(generated_rows, judge_batch_size):
                                failed_task_id = jb[0].get("id")
                                scored_batch = _judge_batch(
                                    jb,
                                    judge_generation_cfg=judge_generation_cfg,
                                    judge_model=judge_model,
                                    judge_tokenizer=judge_tokenizer,
                                    judge_hf_model=judge_hf_model,
                                    judge_vllm_model=judge_vllm_model,
                                    judge_vllm_tokenizer=judge_vllm_tokenizer,
                                    max_concurrent_requests=judge_batch_size,
                                )
                                for scored_row in scored_batch:
                                    row_by_id[str(scored_row["id"])] = scored_row

                            _write_flat_results(flat_path, _ordered_rows(dataset.tasks, row_by_id))
                            _log_running_performance(
                                llm_name=llm_name,
                                capability_id=dataset.capability_id,
                                row_by_id=row_by_id,
                                total_tasks=total_tasks,
                            )
                            eval_bar.update(len(task_batch))
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "  Direct evaluation failed for %s/%s task %s with %s/%s: %s",
                    dataset.area_id,
                    dataset.capability_id,
                    failed_task_id,
                    llm_provider,
                    llm_name,
                    exc,
                )
                success = False

            rows = _ordered_rows(dataset.tasks, row_by_id)
            _write_flat_results(flat_path, rows)

            if success:
                if _check_flat_completed(flat_path, expected_task_ids):
                    num_completed_this_run += 1
                else:
                    logger.warning(
                        "  Incomplete flat output for %s/%s with %s "
                        "(task IDs mismatch: missing or extra scored tasks)",
                        dataset.area_id,
                        dataset.capability_id,
                        llm_name,
                    )
                    num_incomplete += 1
            else:
                num_failed += 1

    logger.info(
        "Eval Stage 1_local summary: completed_this_run=%d skipped_completed=%d "
        "failed=%d incomplete=%d total=%d",
        num_completed_this_run,
        num_skipped_completed,
        num_failed,
        num_incomplete,
        total_combinations,
    )

    return eval_tag
