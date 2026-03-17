# Scientific Tools Module

A comprehensive toolkit for integrating scientific computing capabilities into LLM-based research pipelines. The module provides safe Python code execution, intelligent tool selection, and documentation-aware context retrieval for mathematical, scientific, and financial computations.

## Overview

The tools module consists of four main components:

1. **`toolkit.py`**: High-level orchestration layer that coordinates tool selection and code execution
2. **`executor.py`**: Safe Python code execution with import restrictions and error handling
3. **`docs.py`**: Structure-aware documentation retrieval from HTML files
4. **`definitions.py`**: Library configurations and metadata for scientific computing packages

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  ScientificToolKit                      │
│  ┌─────────────────────┐      ┌────────────────────┐    │
│  │  Tool Selection     │      │  Code Execution    │    │
│  │  (LLM-based)        │      │  (PythonExecutor)  │    │
│  └─────────────────────┘      └────────────────────┘    │
│           ↓                             ↑               │
│  ┌─────────────────────────────────────────────────┐    │
│  │   ScientificDocRetriever (Optional RAG)         │    │
│  └─────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

### Operating Modes

The toolkit supports two modes optimized for different use cases:

**1. RAG Mode (enable_rag=True)**
- Retrieves full function signatures and documentation from local HTML files
- Provides precise, deterministic context to the LLM
- Best for: Accuracy-critical tasks, evaluation benchmarks
- Trade-off: Slightly higher latency and token usage

**2. Parametric Mode (enable_rag=False)**
- Relies on model's built-in knowledge of scientific libraries
- Fast, lightweight context generation
- Best for: High-throughput generation, rapid prototyping
- Trade-off: May hallucinate less common function signatures

## Quick Start

### Basic Usage

```python
from autogen_core.models import ChatCompletionClient
from src.tools.toolkit import ScientificToolKit

# Initialize with your LLM client
model_client = ChatCompletionClient(...)

# Create toolkit instance (uses parametric mode by default)
toolkit = ScientificToolKit(model_client=model_client)

# Execute Python code directly
result = toolkit.execute_code("""
import numpy as np

# Coefficient matrix
A = np.array([[2, 3], [1, -1]])
b = np.array([7, 1])

# Solve system
solution = np.linalg.solve(A, b)
print(f"x = {solution[0]}, y = {solution[1]}")
""")

if result["success"]:
    print(result["output"])  # x = 2.0, y = 1.0
else:
    print(f"Error: {result['error']}")
```

### Integration with Task Solver

The toolkit integrates seamlessly with the agentic task solver pipeline:

```python
from src.task_solver.tool_assisted_scientist import ToolAssistedScientist
from src.tools.toolkit import ScientificToolKit

# Initialize toolkit (defaults to parametric mode)
toolkit = ScientificToolKit(model_client=model_client)

# Create scientist agent with toolkit
scientist = ToolAssistedScientist(
    scientist_id="scientist_1",
    model_client=model_client,
    toolkit=toolkit,  # Inject toolkit dependency
    moderator_topic_type="task_solver",
)

# The scientist will automatically:
# 1. Generate code using the LLM's parametric knowledge
# 2. Execute code using toolkit.execute_code()
# 3. Handle errors and retry with feedback
```

## Module Reference

### ScientificToolKit

Primary interface for tool management and orchestration.

#### Initialization

```python
ScientificToolKit(
    model_client: ChatCompletionClient,
    docs_path: Path = Path("materials"),
    enable_tool_selection: bool = False,
    enable_rag: bool = False,
)
```

**Parameters:**
- `model_client`: LLM client (used only if enable_tool_selection=True)
- `docs_path`: Path to HTML documentation (used only if enable_rag=True)
- `enable_tool_selection`: Enable LLM-based necessity detection (default: False)
- `enable_rag`: **EXPERIMENTAL/WIP** - Enable HTML doc retrieval (default: False)

#### Methods

##### `async prepare_tools(problem_text: str) -> Dict[str, Any]`

**Note:** Only needed if `enable_tool_selection=True`. Most users can skip this and execute code directly.

Analyzes a problem to determine if computational tools are needed (requires an additional LLM call).

**Returns:**
```python
{
    "needs_tools": bool,           # Whether computation is required
    "selected_modules": List[Dict], # Selected library modules (if tool_selection enabled)
    "documentation": str,          # Context for code generation
    "reasoning": str,              # LLM's reasoning
}
```

**Example:**
```python
# Only if enable_tool_selection=True
toolkit = ScientificToolKit(model_client=client, enable_tool_selection=True)
context = await toolkit.prepare_tools(
    "Calculate the eigenvalues of the matrix [[1, 2], [3, 4]]"
)
# Returns: {"needs_tools": True, "reasoning": "...", ...}
```

##### `execute_code(code: str) -> Dict[str, Any]`

Executes Python code with import restrictions and error handling.

**Returns:**
```python
{
    "success": bool,    # Whether execution succeeded
    "output": str,      # Standard output from code
    "error": str|None,  # Error message if failed
}
```

**Example:**
```python
result = toolkit.execute_code("print(2 ** 10)")
# Returns: {"success": True, "output": "1024\n", "error": None}
```

##### `format_tool_context(tool_context: Dict[str, Any]) -> str`

Formats tool context for inclusion in LLM prompts.

**Note:** Only used with `enable_tool_selection=True`. Formats tool context for LLM prompts.

**Example:**
```python
# Only if enable_tool_selection=True
context = await toolkit.prepare_tools("Calculate sqrt(16)")
formatted = toolkit.format_tool_context(context)
# Returns formatted string for prompt injection
```

### PythonExecutor

Low-level Python code execution with security controls.

#### Initialization

```python
PythonExecutor(
    allowed_imports: Optional[List[str]] = None,
    timeout: int = 30,
)
```

**Default allowed imports:**
- Scientific: `numpy`, `scipy`, `sympy`, `math`, `fractions`, `decimal`, `cmath`
- Statistical: `statsmodels`
- Financial: `numpy_financial`, `py_vollib`, `pypfopt`, `empyrical`, `arch`

#### Methods

##### `execute(code: str) -> CodeExecutionResult`

Executes Python code in a controlled environment.

**Example:**
```python
from src.tools.executor import PythonExecutor

executor = PythonExecutor()
result = executor.execute("""
import sympy as sp
x = sp.symbols('x')
derivative = sp.diff(x**2, x)
print(derivative)
""")

print(result.success)  # True
print(result.output)   # "2*x\n"
```

##### `validate_syntax(code: str) -> tuple[bool, Optional[str]]`

Validates Python syntax without execution.

**Example:**
```python
is_valid, error = executor.validate_syntax("print('hello')")
# Returns: (True, None)

is_valid, error = executor.validate_syntax("print('unclosed")
# Returns: (False, "Syntax error: unterminated string literal...")
```

### ScientificDocRetriever (WIP)

Retrieves function signatures and documentation from HTML files.

#### Structure

Exploits naming conventions in scientific library documentation:
- **NumPy/SciPy**: `reference/generated/numpy.module.function.html`
- **SymPy**: `modules/topic/file.html`

## Testing

The module includes comprehensive test coverage:

```bash
# Run all tools tests
pytest tests/tools/ -v

# Run specific test suite
pytest tests/tools/test_toolkit.py -v
pytest tests/tools/test_executor.py -v

# Integration testing
RUN_INTEGRATION_TESTS=1 pytest -v -s tests/tools/test_toolkit_integration.py
```
