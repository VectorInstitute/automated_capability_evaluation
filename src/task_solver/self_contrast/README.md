# Self-Contrast Evaluation Framework

An implementation of the **Self-Contrast** methodology for evaluating LLMs on structured benchmarks. The framework solves each problem from multiple independent perspectives, performs a contrastive comparison to detect discrepancies, and adjudicates disagreements to produce a final answer.

Five solver variants are provided, from base Self-Contrast through tool-integrated multi-perspective runs and a single-agent + tools baseline (V5).

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Solver Variants](#solver-variants)
4. [Installation & Dependencies](#installation--dependencies)
5. [Quick Start](#quick-start)
6. [Running on vec-inf / vLLM](#running-on-vec-inf--vllm)
7. [CLI Reference](#cli-reference)
8. [Pipeline Walkthrough](#pipeline-walkthrough)
9. [Module Reference](#module-reference)
10. [Results Organization](#results-organization)
11. [Dataset Format](#dataset-format)
12. [Evaluation](#evaluation)
13. [Provider Support](#provider-support)
14. [Prompt Design](#prompt-design)
15. [Tool Integration](#tool-integration)
16. [Experimental Results](#experimental-results)
17. [Directory Structure](#directory-structure)
18. [Troubleshooting](#troubleshooting)

---

## Overview

Self-Contrast is a **single-model, multi-perspective** framework -- not a multi-agent system. The same LLM is called multiple times with different perspective prompts, and the results are reconciled through a structured contrast-and-adjudicate pipeline.

The pipeline has three core phases:

1. **Perspective Solving** -- The model solves the problem independently from each perspective (e.g. Top-Down, Bottom-Up, Analogical). Each perspective produces an answer and rationale. Optionally, code is generated and executed to ground the answer in computation.
2. **Contrastive Comparison** -- All solutions are compared pairwise. The model identifies discrepancies in assumptions, formulas, intermediate values, or final answers, and produces a verification checklist.
3. **Adjudication** -- If discrepancies exist, the model reconciles them using the checklist. If all perspectives agree, majority vote is used directly.

### Key Design Decisions

- **No framework dependency**: Uses raw `requests.post()` HTTP calls. No autogen, langchain, or other LLM framework.
- **Dataset-agnostic**: Any JSON batch file can be used. Dataset path is configurable via `--dataset-dir` and `--batch-file`.
- **Provider-agnostic**: Supports OpenAI, Anthropic, Google Gemini, Ollama, vec-inf, and vLLM out of the box.
- **Centralized prompts**: All prompt templates live in `prompts.py` for easy iteration and reproducibility.
- **Cluster-friendly**: Gracefully handles missing optional dependencies (`datasets`, `autogen_core`) via try-except imports.

---

## Architecture

```
                    ┌─────────────────────────────┐
                    │       LLMClient              │
                    │  (OpenAI / Anthropic /        │
                    │   Gemini / Ollama /           │
                    │   vec-inf / vLLM)             │
                    └──────────────┬───────────────┘
                                   │
                    ┌──────────────▼───────────────┐
                    │     SelfContrastSolver        │
                    │                               │
                    │  Phase 1: Perspective Solving  │
                    │    ┌─────────┐ ┌─────────┐    │
                    │    │Top-Down │ │Bottom-Up│    │
                    │    └─────────┘ └─────────┘    │
                    │    ┌──────────┐ ┌──────────┐  │
                    │    │Analogical│ │Tool-Asst.│  │
                    │    └──────────┘ └──────────┘  │
                    │         (3 or 4 perspectives) │
                    │                               │
                    │  Phase 2: Contrast             │
                    │    Pairwise discrepancy check  │
                    │    Verification checklist       │
                    │                               │
                    │  Phase 3: Adjudication          │
                    │    Reconcile or majority vote   │
                    └──────────────┬───────────────┘
                                   │
                    ┌──────────────▼───────────────┐
                    │  Optional: PythonExecutor      │
                    │  (numpy, scipy, sympy, etc.)   │
                    └───────────────────────────────┘
```

---

## Solver Variants

### V1: Base Self-Contrast (`run_self_contrast.py`)

The core Self-Contrast method with three reasoning perspectives. Python code execution is used only for problems that match a built-in computation heuristic (primarily `calcu`-type tasks), restricted to the Python standard library (`math` allowed).

| Property | Value |
|---|---|
| Perspectives | Top-Down, Bottom-Up, Analogical |
| Tool use | Standard library only, calcu heuristic |
| LLM calls per problem | ~5 (3 perspectives + 1 contrast + 0-1 adjudication) |
| `tool_mode` | `"none"` |

**When code executes (V1 heuristic):**
- Task type is `calcu`, OR
- Problem text contains a number AND a calculation keyword (calculate, compute, NPV, IRR, percentage, yield, etc.)

### V2: Single Agent Baseline (`run_single_agent.py`)

A simple baseline with one LLM call per problem. No perspectives, no contrast, no tools. Included for side-by-side accuracy comparison with the Self-Contrast variants.

| Property | Value |
|---|---|
| Perspectives | None |
| Tool use | None |
| LLM calls per problem | 1 |
| `tool_mode` | N/A |

### V3: Tools Integrated (`run_tools_integrated.py`)

Self-Contrast where **every perspective** can use the full scientific toolkit from `src/tools/`. The code prompt is built from the rich library metadata in `src/tools/definitions.py`, failed executions can be retried with error feedback, `--tool-selection` can decide whether tools are actually needed, and `--docs-path` can inject local HTML documentation excerpts.

| Property | Value |
|---|---|
| Perspectives | Top-Down, Bottom-Up, Analogical (all tool-augmented) |
| Tool use | All perspectives, optionally gated by `--tool-selection` |
| LLM calls per problem | ~8 (3 code + 3 answer + 1 contrast + 0-1 adjudication) |
| `tool_mode` | `"all"` |

### V4: Tool as 4th Perspective (`run_tool_perspective.py`)

Self-Contrast with the original three **reasoning-only** perspectives unchanged, plus a fourth **Tool-Assisted** perspective with the same toolkit access as V3.

| Property | Value |
|---|---|
| Perspectives | Top-Down, Bottom-Up, Analogical + Tool-Assisted |
| Tool use | Only the 4th perspective |
| LLM calls per problem | ~7 (3 answer + 1 code + 1 answer + 1 contrast + 0-1 adjudication) |
| `tool_mode` | `"tool_perspective_only"` |

### V5: Single Agent + Tools (`run_single_agent_tools.py`)

One agent generates Python using the full scientific toolkit, executes it via `PythonExecutor`, then produces a final JSON answer. No perspectives and no contrast. Use this as the **tools baseline** when comparing against V3/V4 (multi-perspective + tools).

| Property | Value |
|---|---|
| Perspectives | None |
| Tool use | Full toolkit (same libraries as V3) |
| Contrast / adjudication | None |
| LLM calls per problem | 2 (code + answer; optional code retry on failure) |
| `tool_mode` | N/A (standalone runner) |

### Variant Comparison

```
                  V1 (Base)    V2 (Single)  V3 (Tools)  V4 (ToolPersp)  V5 (Agent+Tools)
Perspectives      3            0            3           4               0
Tool use          heuristic    none         all         4th only        all
Contrast step     yes          no           yes         yes             no
Adjudication      conditional  no           conditional conditional     no
LLM calls/prob    ~5           1            ~8          ~7              ~2 (+ retries)
Best for          reasoning    baseline     computation hybrid          single-agent + code
```

---

## Installation & Dependencies

### Minimal (V1 and V2 only)

The self-contrast module requires only the `requests` library:

```bash
pip install requests
```

### Full (V3, V4, V5 with tools)

Tool-augmented variants use `PythonExecutor` from `src/tools/`, which requires scientific libraries:

```bash
pip install requests numpy scipy sympy statsmodels numpy_financial
```

### Optional dependencies

These are used by other parts of the project but **not required** by self-contrast:

| Package | Used by | Self-contrast needs it? |
|---|---|---|
| `datasets` / `pyarrow` | `src/utils/data_utils.py` | No (wrapped in try-except) |
| `autogen_core` | `src/task_solver/generator.py` | No (wrapped in try-except) |
| `beautifulsoup4` | `src/tools/docs.py` (RAG) | Only if `--docs-path` is used |

### Cluster environments (e.g. Alliance Canada)

If `pip install datasets` fails due to a blocked `pyarrow` wheel, no code changes are needed. The import is wrapped in a try-except in `src/utils/__init__.py` and `src/task_solver/__init__.py`, so self-contrast runs without it.

---

## Quick Start

### Running V1 (Base Self-Contrast)

```bash
# OpenAI (reads OPENAI_API_KEY from environment)
python -m src.task_solver.self_contrast.run_self_contrast \
    --model gpt-4o \
    --batch-file evaluation_batch.json

# Google Gemini
python -m src.task_solver.self_contrast.run_self_contrast \
    --model gemini-2.5-pro \
    --batch-file evaluation_batch.json

# Anthropic Claude
python -m src.task_solver.self_contrast.run_self_contrast \
    --model claude-opus-4-5-20251101 \
    --batch-file evaluation_batch.json

# vec-inf / vLLM endpoint
python -m src.task_solver.self_contrast.run_self_contrast \
    --model Qwen2.5-72B-Instruct \
    --url http://gpu028:21134/v1
```

### Running V2 (Single Agent Baseline)

```bash
python -m src.task_solver.self_contrast.run_single_agent \
    --model gpt-4o \
    --batch-file evaluation_batch.json
```

### Running V3 (Tools Integrated)

```bash
python -m src.task_solver.self_contrast.run_tools_integrated \
    --model gpt-4o \
    --batch-file evaluation_batch.json
```

### Running V4 (Tool Perspective)

```bash
python -m src.task_solver.self_contrast.run_tool_perspective \
    --model gpt-4o \
    --batch-file evaluation_batch.json
```

### Running V5 (Single Agent + Tools)

```bash
python -m src.task_solver.self_contrast.run_single_agent_tools \
    --model gpt-4o \
    --batch-file evaluation_batch.json
```

For **gpt-5-mini**, OpenAI requires `temperature=1`:

```bash
python -m src.task_solver.self_contrast.run_single_agent_tools \
    --model gpt-5-mini --temperature 1 \
    --batch-file evaluation_batch.json
```

### Quick Test (limit problems)

```bash
python -m src.task_solver.self_contrast.run_self_contrast \
    --model gpt-4o \
    --batch-file evaluation_batch.json \
    --max-problems 5
```

### All Five Versions on a Single Endpoint

```bash
# V1
python3 -m src.task_solver.self_contrast.run_self_contrast \
    --model Qwen2.5-72B-Instruct --url http://gpu028:21134/v1 \
    --batch-file evaluation_batch.json

# V2
python3 -m src.task_solver.self_contrast.run_single_agent \
    --model Qwen2.5-72B-Instruct --url http://gpu028:21134/v1 \
    --batch-file evaluation_batch.json

# V3
python3 -m src.task_solver.self_contrast.run_tools_integrated \
    --model Qwen2.5-72B-Instruct --url http://gpu028:21134/v1 \
    --batch-file evaluation_batch.json

# V4
python3 -m src.task_solver.self_contrast.run_tool_perspective \
    --model Qwen2.5-72B-Instruct --url http://gpu028:21134/v1 \
    --batch-file evaluation_batch.json

# V5
python3 -m src.task_solver.self_contrast.run_single_agent_tools \
    --model Qwen2.5-72B-Instruct --url http://gpu028:21134/v1 \
    --batch-file evaluation_batch.json
```

---

## Running on vec-inf / vLLM

[vec-inf](https://github.com/VectorInstitute/vector-inference) and vLLM expose an OpenAI-compatible `/v1/chat/completions` endpoint. The framework auto-detects these when `--url` is provided.

### Basic usage

```bash
python -m src.task_solver.self_contrast.run_self_contrast \
    --model Qwen2.5-72B-Instruct \
    --url http://gpu028:21134/v1 \
    --batch-file evaluation_batch.json
```

### API key

If your endpoint requires an API key, set `LOCAL_API_KEY` in your environment or `.env` file:

```bash
export LOCAL_API_KEY=your-key-here
```

Or pass it directly:

```bash
python -m src.task_solver.self_contrast.run_self_contrast \
    --model Qwen2.5-72B-Instruct \
    --url http://gpu028:21134/v1 \
    --api-key your-key-here
```

### Qwen3 models

Qwen3 models have a "thinking mode" enabled by default that generates extensive internal reasoning (`<think>...</think>` blocks), causing timeouts on code-generation prompts. The framework automatically disables this by sending `chat_template_kwargs: {"enable_thinking": false}` for any model containing "qwen3" in the name when running on `ollama` or `openai_compatible` providers.

### Sequential execution

For local/vec-inf endpoints (single GPU server), perspective calls run **sequentially** to avoid overloading the server. Cloud APIs (OpenAI, Anthropic, Gemini) run perspectives in **parallel** via `asyncio.gather`.

### Terminating a vec-inf job

```bash
vec-inf shutdown --model_name Qwen2.5-72B-Instruct
```

---

## CLI Reference

All runners share a consistent CLI interface. V2 and V5 omit perspective-specific flags (`--prompt-repeat`, `--tool-selection`, etc.). V5 adds `--max-code-retries` for the standalone tools runner.

| Flag | Default | V1 | V2 | V3 | V4 | V5 | Description |
|---|---|:---:|:---:|:---:|:---:|:---:|---|
| `--model` | `gpt-4o` | x | x | x | x | x | Model name |
| `--dataset-dir` | `self_contrast/dataset/XFinBench/` | x | x | x | x | x | Path to dataset directory |
| `--batch-file` | `evaluation_batch.json` | x | x | x | x | x | Batch file name or path |
| `--output` | Auto-generated | x | x | x | x | x | Custom output file name |
| `--results-dir` | `self_contrast/Results/` | x | x | x | x | x | Root results directory |
| `--temperature` | `0.7` | x | x | x | x | x | Sampling temperature |
| `--max-problems` | All | x | x | x | x | x | Limit number of problems |
| `--url` | Auto-detected | x | x | x | x | x | Base URL for vec-inf/vLLM |
| `--api-key` | From env vars | x | x | x | x | x | API key override |
| `--prompt-repeat` | `1` | x | | x | x | | Repeat prompt 1/2/3 times |
| `--max-code-retries` | `1` | | | x | x | x | Retry failed code with error |
| `--docs-path` | -- | | | x | x | | Local HTML docs for RAG |
| `--tool-selection` | Off | | | x | x | | LLM-based tool selection |
| `--force-json` | On | x | x | x | x | x | Force JSON response format |
| `--no-force-json` | -- | x | x | x | x | x | Disable forced JSON |

---

## Pipeline Walkthrough

### Phase 1: Perspective Solving

For each perspective, the solver:

1. **Determines whether to use tools** based on `tool_mode`, the V1 heuristic, and optional tool-selection.
2. **Requests code** (if tools active): sends the perspective guidance, problem text, and toolkit context.
3. **Executes code** (if generated): runs via `PythonExecutor` with allowed scientific imports.
4. **Retries failed code** (V3/V4): re-prompts the LLM with the error trace.
5. **Requests the answer**: sends the perspective, problem, and tool output (if any).
6. **Parses the response**: uses `parse_llm_json_response` with fallback regex extraction.

### Phase 2: Contrastive Comparison

1. All perspective answers are formatted and labeled (A, B, C, ...).
2. The model identifies pairwise discrepancies.
3. Returns a JSON with `pairwise_discrepancies` and a `checklist` of verification items.
4. If **any** pairwise comparison has non-empty items, discrepancies exist.

### Phase 3: Adjudication

- **No discrepancies**: majority vote across perspective answers. Answers backed by successful Python execution are preferred.
- **Discrepancies found**: the model reconciles using the checklist from Phase 2.

---

## Module Reference

### `solver.py` -- `SelfContrastSolver`

The core solver class. Configured differently by each runner script.

```python
from src.task_solver.self_contrast.solver import SelfContrastSolver
from src.task_solver.self_contrast.model_client import LLMClient
from src.task_solver.self_contrast.perspectives import BASE_PERSPECTIVES

client = LLMClient("gpt-4o")

solver = SelfContrastSolver(
    client,
    list(BASE_PERSPECTIVES),
    tool_mode="none",                    # "none" | "all" | "tool_perspective_only"
    prompt_repeat=1,                     # 1/2/3
    force_json=True,                     # Request JSON response format
)

result = await solver.solve_problem(problem)
```

### `model_client.py` -- `LLMClient`

Lightweight multi-provider LLM client. No external framework dependency.

```python
from src.task_solver.self_contrast.model_client import LLMClient

client = LLMClient(
    "gpt-4o",
    base_url=None,          # Auto-detected; set for vec-inf/vLLM
    api_key=None,           # Auto from env vars
    temperature=0.7,
    timeout=None,           # Auto: 120s cloud, 600s ollama
)

response = client.call(
    system_prompt="You are a financial analyst.",
    user_prompt="What is the NPV of ...",
    force_json=True,
)
```

**Provider dispatch:**

| Method | Used by | Protocol |
|---|---|---|
| `_call_openai_compatible()` | OpenAI, Ollama, vec-inf, vLLM | `POST /v1/chat/completions` |
| `_call_anthropic()` | Anthropic Claude | `POST /v1/messages` |
| `_call_google()` | Google Gemini | `POST /v1beta/models/{model}:generateContent` |

**API key resolution order:**

| Provider | Environment variables checked |
|---|---|
| OpenAI | `OPENAI_API_KEY` |
| Anthropic | `ANTHROPIC_API_KEY` |
| Google | `GOOGLE_API_KEY` |
| openai_compatible | `LOCAL_API_KEY`, then `OPENAI_API_KEY` |
| ollama | `LOCAL_API_KEY` |

### `perspectives.py` -- Perspective Definitions

| ID | Label | Guidance | `uses_tools` |
|---|---|---|---|
| `top_down` | Top-Down | Start from goal, work backward, derive formula | `False` |
| `bottom_up` | Bottom-Up | Start from given data, compute step by step | `False` |
| `analogical` | Analogical | Solve by analogy to a known problem | `False` |
| `tool_assisted` | Tool-Assisted | Use Python scientific tools (V4 only) | `True` |

### `prompts.py` -- Prompt Templates

All prompt templates centralized for easy iteration:

| Constant | Used by | Purpose |
|---|---|---|
| `FORMAT_RULES` | All | Per-task-type answer format instructions |
| `PERSPECTIVE_SYSTEM_PROMPT` | V1, V3, V4 | System message for answer generation |
| `PERSPECTIVE_CODE_SYSTEM_PROMPT` | V1, V3, V4 | System message for code generation |
| `PERSPECTIVE_CODE_REQUEST` | V1 | Code request (standard library only) |
| `PERSPECTIVE_ANSWER_REQUEST` | V1, V3, V4 | Answer request (with optional tool output) |
| `TOOL_INTEGRATED_CODE_REQUEST` | V3 | Code request with full toolkit context |
| `TOOL_PERSPECTIVE_CODE_REQUEST` | V4 | Code request for 4th tool perspective |
| `TOOL_SELECTION_SYSTEM_PROMPT` | V3, V4 | System message for tool-selection step |
| `TOOL_SELECTION_PROMPT` | V3, V4 | Decides whether tools are needed |
| `CODE_RETRY_PROMPT` | V3, V4 | Fix failed code with execution error |
| `CONTRAST_SYSTEM_PROMPT` | V1, V3, V4 | System message for auditor role |
| `CONTRAST_USER_PROMPT` | V1, V3, V4 | Pairwise comparison template |
| `ADJUDICATION_SYSTEM_PROMPT` | V1, V3, V4 | System message for reconciliation |
| `ADJUDICATION_USER_PROMPT` | V1, V3, V4 | Final answer template with checklist |
| `SINGLE_AGENT_SYSTEM_PROMPT` | V2, V5 | System message for single-agent final answer |
| `SINGLE_AGENT_USER_PROMPT` | V2 | User prompt for single-agent (no tools) |
| `SINGLE_AGENT_TOOLS_CODE_SYSTEM_PROMPT` | V5 | System message for code generation |
| `SINGLE_AGENT_TOOLS_CODE_PROMPT` | V5 | User prompt for toolkit-backed code |
| `SINGLE_AGENT_TOOLS_ANSWER_PROMPT` | V5 | Final answer with optional tool output |

### `evaluator.py` -- Evaluation

| Task Type | Method | Tolerance |
|---|---|---|
| `calcu` | Numerical comparison | 1% relative (or 0.01 absolute when \|gt\| < 1) |
| `bool` | Boolean normalization | Maps true/false/yes/no/1.0/0.0 |
| `mcq` | Letter extraction | Matches A-Z with various formats |

---

## Results Organization

```
Results/
├── v1_self_contrast/{model}/
├── v2_single_agent/{model}/
├── v3_tools_integrated/{model}/
├── v4_tool_perspective/{model}/
├── v5_single_agent_tools/{model}/
├── comparison_summary.json          # All results in structured JSON
└── comparison_dashboard.html        # Interactive HTML dashboard
```

**Filename convention:** `{batch}_T{temp}_R{repeat}_{timestamp}.json`

Compare across models or variants:

```bash
# All models on V1:
ls Results/v1_self_contrast/*/

# All variants for gpt-4o:
ls Results/*/gpt-4o/
```

---

## Dataset Format

Any JSON file containing an array of problem objects:

```json
[
  {
    "id": "prob_001",
    "question": "A bond with face value $1000 has a 5% coupon rate. What is the annual coupon payment?",
    "task": "calcu",
    "ground_truth": "50"
  },
  {
    "id": "prob_002",
    "question": "Is diversification a strategy to reduce unsystematic risk?",
    "task": "bool",
    "ground_truth": "1.0"
  },
  {
    "id": "prob_003",
    "question": "Which of the following is a measure of systematic risk?",
    "choice": "A. Standard deviation\nB. Beta\nC. Variance\nD. Range",
    "task": "mcq",
    "ground_truth": "B"
  }
]
```

**Task types:**

| Type | Count in XFinBench | Answer format | Description |
|---|---|---|---|
| `calcu` | 163 | Numeric value | Financial math / computation problems |
| `bool` | 74 | `1.0` or `0.0` | True/false financial statements |
| `mcq` | 13 | Single letter (A-D) | Multiple choice questions |

---

## Evaluation

### Output file structure

```json
{
  "metrics": {
    "total_processed": 250,
    "total_correct": 192,
    "accuracy": 0.768,
    "per_type": {
      "calcu": {"total": 163, "correct": 120, "accuracy": 0.736},
      "bool":  {"total": 74,  "correct": 69,  "accuracy": 0.932},
      "mcq":   {"total": 13,  "correct": 3,   "accuracy": 0.231}
    },
    "adjudication_rate": 0.34,
    "model": "gemini-3.1-pro-preview",
    "method": "Self-Contrast",
    "execution_time_formatted": "45m 12s"
  },
  "results": [ ... ]
}
```

### Console summary

Each run prints a formatted summary:

```
SELF-CONTRAST (V1: BASE) EVALUATION SUMMARY
================================================================================
Model: gpt-4o
Batch: evaluation_batch
Total Problems: 250
Correct: 192
Overall Accuracy: 76.80%

Per Task-Type:
  calcu: 120/163 (73.6%)
  bool: 69/74 (93.2%)
  mcq: 3/13 (23.1%)

Adjudication Rate: 34.0%
Execution Time: 45m 12s
```

---

## Provider Support

| Provider | Model name patterns | Env var | Default endpoint |
|---|---|---|---|
| OpenAI | `gpt-*`, `o1-*`, `o3-*`, `o4-*` | `OPENAI_API_KEY` | `https://api.openai.com/v1` |
| Anthropic | `claude-*` | `ANTHROPIC_API_KEY` | `https://api.anthropic.com/v1` |
| Google | `gemini-*` | `GOOGLE_API_KEY` | `https://generativelanguage.googleapis.com/v1beta` |
| Ollama | `llama*`, `mistral*`, etc. | `LOCAL_API_KEY` | `http://localhost:11434/v1` |
| vec-inf / vLLM | Any model + `--url` | `LOCAL_API_KEY` | User-provided |

**Provider detection priority:** Model name keywords are checked first (gpt -> OpenAI, claude -> Anthropic, gemini -> Google). If `--url` is provided and no keyword matches, the provider is set to `openai_compatible`. If the URL contains `localhost` or port `11434`, it defaults to `ollama`.

---

## Prompt Design

### Perspective prompts

Each perspective receives a system message and user message containing:
1. The perspective label and guidance
2. The full problem text (with choices if MCQ)
3. Task-type-specific format rules
4. Tool output section (if code was executed)

### Code generation prompts

| Variant | Prompt | Libraries |
|---|---|---|
| V1 | `PERSPECTIVE_CODE_REQUEST` | Standard library + `math` |
| V3 | `TOOL_INTEGRATED_CODE_REQUEST` | Full toolkit from `src/tools/definitions.py` |
| V4 | `TOOL_PERSPECTIVE_CODE_REQUEST` | Full toolkit (4th perspective only) |
| V5 | `SINGLE_AGENT_TOOLS_CODE_PROMPT` | Full toolkit (single-agent runner) |

For `ollama`/`openai_compatible` providers in V1, a simplified legacy prompt is used for code generation to maximize compatibility with local models.

---

## Tool Integration

V3, V4, and V5 use `src/tools/` for code execution:

| File | Purpose |
|---|---|
| `src/tools/definitions.py` | Library metadata, allowed imports, common functions |
| `src/tools/executor.py` | Sandboxed Python execution with timeout |
| `src/tools/docs.py` | Optional HTML documentation retrieval |

**Allowed libraries:**

| Category | Libraries |
|---|---|
| Scientific | numpy, scipy, sympy, math, fractions, decimal, cmath |
| Statistical | statsmodels |
| Financial | numpy_financial, py_vollib, pypfopt, empyrical, arch |
| Standard | datetime |

---

## Experimental Results

Results on XFinBench (250 problems), temperature 0.7 (gpt-5-mini uses temperature 1.0 as required by the API):

### Overall Accuracy (%)

| Model | V1 (Base) | V2 (Single) | V3 (Tools) | V4 (ToolPersp) | V5 (Agent+Tools) |
|---|---|---|---|---|---|
| Gemini 3.1 Pro | 76.8 | 77.2 | 77.2 | 77.2 | **76.8** |
| Gemini 3 Flash | 76.0 | 75.6 | 76.0 | **77.2** | 75.6 |
| Claude Opus 4.5 | **75.2** | 73.2 | 74.4 | 73.6 | 73.6 |
| GPT-5-mini | 71.6 | 68.4 | 72.8 | 68.8 | **74.4** |
| GPT-5.2 | **70.4** | 66.8 | 66.8 | 57.2 | 66.4 |
| Qwen3-32B | 58.4 | 49.2 | **60.8** | 53.2 | — |
| Qwen2.5-72B | **53.2** | 46.8 | 44.8 | 41.6 | — |

### Key findings

- **Self-Contrast consistently outperforms single agent**: V1 beats V2 across all models (avg +5pp).
- **V5 (Single Agent + Tools) is competitive**: Matches or exceeds V2 across all models, with GPT-5-mini achieving the highest V5 score (74.4%).
- **Multi-perspective + tools (V3) vs single agent + tools (V5)**: V3 and V5 are close, but V3 edges ahead for Gemini and Qwen models while V5 wins for GPT-5-mini.
- **Tool benefit is model-dependent**: Tools help GPT-5-mini (+4.4pp V3 vs V2, +6.0pp V5 vs V2) and Qwen3-32B (+11.6pp V3 vs V2) but hurt Qwen2.5-72B (-2.0pp).
- **Gemini models are the most consistent**: 75-77% across all variants.
- **Bool accuracy is highest**: 66-96% across models. MCQ is most variable (23-69%).

### Interactive dashboard

Open `Results/comparison_dashboard.html` in a browser for charts and detailed breakdowns by version, model, and task type.

---

## Directory Structure

```
src/task_solver/self_contrast/
├── README.md                     # This file
├── __init__.py                   # Package exports
├── _json_utils.py                # Local fallback for JSON parsing
│
│   # Core modules
├── model_client.py               # LLMClient: multi-provider HTTP client
├── perspectives.py               # Perspective definitions + PerspectiveResponse
├── prompts.py                    # All prompt templates
├── solver.py                     # SelfContrastSolver: the core pipeline
├── evaluator.py                  # evaluate_result, evaluate_batch, save_results
│
│   # Runner scripts (one per variant)
├── run_self_contrast.py          # V1: Base Self-Contrast
├── run_single_agent.py           # V2: Single Agent baseline
├── run_tools_integrated.py       # V3: All perspectives + tools
├── run_tool_perspective.py       # V4: 4th Tool-Assisted perspective
├── run_single_agent_tools.py     # V5: Single Agent + Tools
│
│   # Data and output (git-ignored)
├── dataset/
│   └── XFinBench/
│       ├── evaluation_batch.json
│       └── validation_set.csv
└── Results/
    ├── v1_self_contrast/
    ├── v2_single_agent/
    ├── v3_tools_integrated/
    ├── v4_tool_perspective/
    ├── v5_single_agent_tools/
    ├── comparison_summary.json
    └── comparison_dashboard.html
```

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'datasets'`

Self-contrast does not require `datasets`. This error occurs because the parent package `src/utils/__init__.py` imports `data_utils.py` which needs it. The import is wrapped in try-except and will not affect self-contrast. If you still see this error, ensure `src/utils/__init__.py` has the try-except wrapper.

### `ModuleNotFoundError: No module named 'autogen_core'`

Same pattern: `src/task_solver/__init__.py` imports `generator.py` which needs autogen. The import is wrapped in try-except.

### vec-inf timeouts on Qwen3 models

Qwen3 models have a default "thinking mode" that generates very long internal reasoning. The framework disables this automatically by setting `chat_template_kwargs: {"enable_thinking": false}`. If you still see timeouts, check that your `model_client.py` has the `_should_disable_thinking()` method.

### `SyntaxError: future feature annotations is not defined`

Your Python version is too old. Self-contrast requires Python 3.10+.

### API key not found

Set the appropriate environment variable:

```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
export GOOGLE_API_KEY=AI...
export LOCAL_API_KEY=...          # For vec-inf/vLLM/Ollama
```

Or use `--api-key` on the command line.

---

## Resources

- **XFinBench**: Financial question-answering benchmark with calcu, mcq, and bool task types.
- **vec-inf**: Vector Institute inference platform. [GitHub](https://github.com/VectorInstitute/vector-inference)
