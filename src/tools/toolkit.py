"""Scientific Toolkit - Unified tool management for research code.

All prompts, logic, and execution in one place for easy iteration.
Structure-aware RAG implementation for scientific computing.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List

from autogen_core.models import ChatCompletionClient, SystemMessage, UserMessage

from src.tools.docs import ScientificDocRetriever
from src.tools.executor import PythonExecutor
from src.utils.json_utils import parse_llm_json_response

log = logging.getLogger("tools.toolkit")


# ==============================================================================
# PROMPTS - Edit these directly to iterate on tool selection behavior
# ==============================================================================

TOOL_SELECTION_PROMPT = """Analyze the following problem and identify the specific Library MODULES required to solve it.

PROBLEM:
{problem_text}

AVAILABLE LIBRARIES:
{tools_description}

Respond with a JSON object:
{{
    "tool_necessity": true/false,  // Set to true if ANY computation is required
    "reasoning": "Brief explanation",
    "selected_modules": [
        {{
            "library": "library_name", // e.g., "numpy"
            "module": "module_name"    // e.g., "linalg"
        }}
    ]
}}

Return only valid JSON, no markdown formatting."""


# Static description for Parametric Mode (saves tokens, relies on model knowledge)
PARAMETRIC_OVERVIEW = """
- **NumPy**: Linear algebra, Fourier transforms, random number generation, matrices.
- **SciPy**: Optimization, integration, interpolation, eigenvalue problems, signal processing, statistical distributions.
- **SymPy**: Symbolic mathematics, equation solving, calculus, matrices, physics.
- **Standard Python**: Math, decimal, fractions.
"""


# ==============================================================================
# Main Toolkit Class
# ==============================================================================

class ScientificToolKit:
    """Unified toolkit for scientific computing with Python.
    
    Supports two modes:
    - RAG mode (enable_rag=True): Retrieves full documentation from HTML files
    - Parametric mode (enable_rag=False): Uses model's built-in knowledge for fast tagging
    """
    
    def __init__(
        self,
        model_client: ChatCompletionClient,
        docs_path: Path = Path("materials"),
        enable_tool_selection: bool = True,
        enable_rag: bool = True,
    ):
        """Initialize toolkit.
        
        Parameters
        ----------
        model_client : ChatCompletionClient
            LLM for tool selection prompts.
        docs_path : Path
            Path to documentation files.
        enable_tool_selection : bool
            If False, skips tool necessity check (always uses tools).
        enable_rag : bool
            If False, skips HTML parsing and uses parametric knowledge (fast, for generation).
        """
        self.model_client = model_client
        self.enable_tool_selection = enable_tool_selection
        self.enable_rag = enable_rag
        
        # Only initialize the heavy retriever if RAG is enabled
        self.doc_retriever = ScientificDocRetriever(docs_path) if enable_rag else None
        
        # Setup code executor (always available)
        self.executor = PythonExecutor(
            allowed_imports=["numpy", "scipy", "sympy", "math", "fractions", "decimal"]
        )
        
        log.info(
            f"Initialized ScientificToolKit - "
            f"Selection: {'enabled' if enable_tool_selection else 'disabled'}, "
            f"RAG: {'enabled' if enable_rag else 'disabled (parametric)'}"
        )
    
    async def prepare_tools(self, problem_text: str) -> Dict[str, Any]:
        """Analyze problem and prepare tool context.
        
        RAG mode: Retrieves full documentation from HTML files.
        Parametric mode: Uses model's built-in knowledge (fast, for tagging).
        
        Parameters
        ----------
        problem_text : str
            Problem statement to analyze.
            
        Returns
        -------
        Dict[str, Any]
            Tool context with keys:
            - needs_tools: bool
            - selected_modules: List[Dict]
            - documentation: str (full module context or parametric note)
            - reasoning: str
        """
        # 1. Get Library Overview
        if self.enable_rag and self.doc_retriever:
            # High precision: exact list of files on disk
            overview = self.doc_retriever.get_library_overview()
        else:
            # High speed: generic description of capabilities
            overview = PARAMETRIC_OVERVIEW
        
        # 2. Module Selection via LLM (Same for both modes)
        response = await self.model_client.create(
            [
                SystemMessage(content="You are an expert scientific programmer."),
                UserMessage(
                    content=TOOL_SELECTION_PROMPT.format(
                        problem_text=problem_text,
                        tools_description=overview
                    ),
                    source="user"
                )
            ],
            json_output=True
        )
        
        selection = parse_llm_json_response(response.content)
        
        # 3. Check Necessity (if enabled)
        necessity = str(selection.get("tool_necessity", "false")).lower() in ("true", "1", "yes")
        
        if self.enable_tool_selection and not necessity:
            log.info("Tool necessity check: No tools needed")
            return {
                "needs_tools": False,
                "selected_modules": [],
                "documentation": "",
                "reasoning": selection.get("reasoning", "No computational tools required")
            }
        
        # 4. Context Construction
        doc_context = []
        selected_modules = selection.get("selected_modules", [])
        
        log.info(f"Selected modules: {selected_modules}")
        
        # BRANCH: Only fetch HTML content if RAG is enabled
        if self.enable_rag and self.doc_retriever:
            for item in selected_modules:
                lib = item.get("library")
                mod = item.get("module")
                if lib and mod:
                    # Retrieve actual signatures from HTML
                    module_docs = self.doc_retriever.get_full_module_context(lib, mod)
                    doc_context.append(module_docs)
        else:
            # Parametric Mode: Add a lightweight reminder instead of full docs
            doc_context.append("Tool selection based on parametric knowledge. Write code assuming standard library versions.")
        
        full_documentation = "\n\n".join(doc_context)
        
        log.info(f"Generated documentation context: {len(full_documentation)} chars")
        
        return {
            "needs_tools": True,
            "selected_modules": selected_modules,
            "documentation": full_documentation,
            "reasoning": selection.get("reasoning", "Computational tools selected")
        }
    
    def execute_code(self, code: str) -> Dict[str, Any]:
        """Execute Python code using the executor.
        
        Parameters
        ----------
        code : str
            Python code to execute.
            
        Returns
        -------
        Dict[str, Any]
            Result with keys: success, output, error
        """
        result = self.executor.execute(code)
        return result.to_dict()
    
    def format_tool_context(self, tool_context: Dict[str, Any]) -> str:
        """Format tool context for inclusion in scientist prompt.
        
        Parameters
        ----------
        tool_context : Dict[str, Any]
            Result from prepare_tools().
            
        Returns
        -------
        str
            Formatted string for prompt.
        """
        if not tool_context["needs_tools"]:
            return f"Note: {tool_context['reasoning']}"
        
        parts = [
            "**AVAILABLE TOOLS:**",
            f"Reasoning: {tool_context['reasoning']}",
            "",
            "**MODULE DOCUMENTATION:**",
            tool_context["documentation"]
        ]
        
        return "\n".join(parts)
