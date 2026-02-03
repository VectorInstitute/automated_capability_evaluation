"""Scientific computing toolkit for research code.

Simplified, flat structure for easy experimentation and prompt iteration.

Core Components:
---------------
- definitions.py: Tool and library definitions
- docs.py: Documentation retrieval from HTML files
- executor.py: Python code execution engine
- toolkit.py: Main ScientificToolKit class (combines selection + execution)

Usage:
------
    from src.tools.toolkit import ScientificToolKit
    
    # Create toolkit
    toolkit = ScientificToolKit(
        model_client=my_client,
        enable_tool_selection=True
    )
    
    # Use in scientist
    scientist = ToolAssistedScientist(..., toolkit=toolkit)
"""

from src.tools.definitions import PYTHON_SCIENTIFIC_TOOL, LibraryConfig, ToolDefinition
from src.tools.docs import ScientificDocRetriever
from src.tools.executor import CodeExecutionResult, PythonExecutor
from src.tools.toolkit import ScientificToolKit

__all__ = [
    "ScientificToolKit",
    "ScientificDocRetriever",
    "PythonExecutor",
    "CodeExecutionResult",
    "ToolDefinition",
    "LibraryConfig",
    "PYTHON_SCIENTIFIC_TOOL",
]
