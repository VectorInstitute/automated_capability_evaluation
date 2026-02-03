"""Python code executor with restricted imports."""

import ast
import sys
import traceback
from dataclasses import dataclass, field
from io import StringIO
from typing import Any, Dict, List, Optional


@dataclass
class CodeExecutionResult:
    """Result of code execution."""
    success: bool
    output: str
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "metadata": self.metadata,
        }


class PythonExecutor:
    """Execute Python code with restricted imports for safe execution."
    
    Attributes
    ----------
    allowed_imports : List[str]
        Set of allowed module names for import.
    timeout : int
        Maximum execution time in seconds (not enforced in basic implementation).
    """
    
    def __init__(
        self,
        allowed_imports: Optional[List[str]] = None,
        timeout: int = 30
    ):
        """Initialize the Python executor.
        
        Parameters
        ----------
        allowed_imports : Optional[List[str]]
            List of allowed module names. If None, uses default scientific libs.
        timeout : int
            Maximum execution time in seconds.
        """
        if allowed_imports is None:
            allowed_imports = [
                'sympy',
                'numpy',
                'scipy',
                'math',
                'fractions',
                'decimal',
                'cmath',
            ]
        
        self._allowed_imports = set(allowed_imports)
        self._timeout = timeout
    
    @property
    def supported_libraries(self) -> List[str]:
        """List of supported libraries/modules."""
        return list(self._allowed_imports)
    
    def validate_syntax(self, code: str) -> tuple[bool, Optional[str]]:
        """Validate Python syntax.
        
        Parameters
        ----------
        code : str
            Python code to validate.
            
        Returns
        -------
        tuple[bool, Optional[str]]
            (is_valid, error_message)
        """
        try:
            ast.parse(code)
            return True, None
        except SyntaxError as e:
            return False, f"Syntax error at line {e.lineno}: {e.msg}"
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    def _validate_imports(self, code: str) -> tuple[bool, Optional[str]]:
        """Validate that only allowed imports are used.
        
        Parameters
        ----------
        code : str
            Python code to validate.
            
        Returns
        -------
        tuple[bool, Optional[str]]
            (is_valid, error_message)
        """
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    if isinstance(node, ast.ImportFrom):
                        module_name = node.module
                    else:
                        module_name = node.names[0].name if node.names else None
                    
                    if module_name:
                        base_module = module_name.split('.')[0]
                        if base_module not in self._allowed_imports:
                            return False, (
                                f"Import of '{module_name}' is not allowed. "
                                f"Allowed modules: {', '.join(sorted(self._allowed_imports))}"
                            )
            return True, None
        except Exception as e:
            return False, f"Import validation error: {str(e)}"
    
    def _create_restricted_globals(self) -> Dict[str, Any]:
        """Create a restricted global namespace for code execution.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary with restricted builtins and allowed imports.
        """
        # Restricted builtins - only safe operations
        safe_builtins = {
            '__import__': __import__,
            'print': print,
            'range': range,
            'len': len,
            'abs': abs,
            'min': min,
            'max': max,
            'sum': sum,
            'round': round,
            'float': float,
            'int': int,
            'str': str,
            'bool': bool,
            'list': list,
            'dict': dict,
            'tuple': tuple,
            'set': set,
            'enumerate': enumerate,
            'zip': zip,
            'sorted': sorted,
            'reversed': reversed,
            'map': map,
            'filter': filter,
            'any': any,
            'all': all,
            'pow': pow,
            'divmod': divmod,
            'isinstance': isinstance,
            'type': type,
            'True': True,
            'False': False,
            'None': None,
        }
        
        return {
            '__builtins__': safe_builtins,
        }
    
    def execute(self, code: str, context: Optional[Dict[str, Any]] = None) -> CodeExecutionResult:
        """Execute Python code with safety restrictions.
        
        Parameters
        ----------
        code : str
            Python code to execute.
        context : Optional[Dict[str, Any]]
            Additional context (currently unused).
            
        Returns
        -------
        CodeExecutionResult
            Result object containing success status, output, and any errors.
        """
        # Validate syntax first
        is_valid, error_msg = self.validate_syntax(code)
        if not is_valid:
            return CodeExecutionResult(
                success=False,
                output="",
                error=error_msg
            )
        
        # Validate imports
        is_valid, error_msg = self._validate_imports(code)
        if not is_valid:
            return CodeExecutionResult(
                success=False,
                output="",
                error=error_msg
            )
        
        # Create restricted execution environment
        restricted_globals = self._create_restricted_globals()
        
        # Pre-import allowed modules to make them available
        for module_name in self._allowed_imports:
            try:
                restricted_globals[module_name] = __import__(module_name)
            except ImportError:
                pass  # Module not available in environment
        
        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()
        
        try:
            # Execute the code
            exec(code, restricted_globals)
            output = captured_output.getvalue()
            
            return CodeExecutionResult(
                success=True,
                output=output,
                error=None,
                metadata={"executor": "python"}
            )
            
        except Exception as e:
            # Capture any execution errors
            error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            
            return CodeExecutionResult(
                success=False,
                output=captured_output.getvalue(),
                error=error_msg,
                metadata={"executor": "python"}
            )
            
        finally:
            # Restore stdout
            sys.stdout = old_stdout


def execute_python_code(
    code: str,
    allowed_imports: Optional[List[str]] = None,
    timeout: int = 30
) -> Dict[str, Any]:
    """Convenience function to execute Python code.
    
    Parameters
    ----------
    code : str
        Python code to execute.
    allowed_imports : Optional[List[str]]
        List of allowed imports.
    timeout : int
        Maximum execution time in seconds.
        
    Returns
    -------
    Dict[str, Any]
        Dictionary with 'success', 'output', 'error', and 'metadata' keys.
    """
    executor = PythonExecutor(allowed_imports=allowed_imports, timeout=timeout)
    result = executor.execute(code)
    return result.to_dict()
