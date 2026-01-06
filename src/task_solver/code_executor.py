"""Python code execution module for tool-assisted task solving."""

import ast
import sys
import traceback
from io import StringIO
from typing import Any


class CodeExecutionResult:
    """Result of code execution.
    
    Attributes
    ----------
    success : bool
        Whether the code executed successfully.
    output : str
        Standard output from the code execution.
    error : str | None
        Error message if execution failed, None otherwise.
    """
    
    def __init__(self, success: bool, output: str, error: str | None = None):
        self.success = success
        self.output = output
        self.error = error
    
    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "success": self.success,
            "output": self.output,
            "error": self.error,
        }


class PythonCodeExecutor:
    """Execute Python code with restricted imports for mathematical computations.
    
    This executor allows SymPy, NumPy, and SciPy for mathematical tasks while
    restricting potentially dangerous operations.
    
    Attributes
    ----------
    allowed_imports : set[str]
        Set of allowed module names for import.
    timeout : int
        Maximum execution time in seconds (not enforced in basic implementation).
    """
    
    def __init__(self, timeout: int = 30):
        """Initialize the code executor.
        
        Parameters
        ----------
        timeout : int, optional
            Maximum execution time in seconds, by default 30.
            Note: Timeout enforcement requires additional implementation.
        """
        self.allowed_imports = {
            'sympy',
            'numpy',
            'scipy',
            'math',
            'fractions',
            'decimal',
            'cmath',
        }
        self.timeout = timeout
    
    def _validate_syntax(self, code: str) -> tuple[bool, str | None]:
        """Validate Python syntax.
        
        Parameters
        ----------
        code : str
            Python code to validate.
            
        Returns
        -------
        tuple[bool, str | None]
            (is_valid, error_message)
        """
        try:
            ast.parse(code)
            return True, None
        except SyntaxError as e:
            return False, f"Syntax error at line {e.lineno}: {e.msg}"
    
    def _create_restricted_globals(self) -> dict[str, Any]:
        """Create a restricted global namespace for code execution.
        
        Returns
        -------
        dict[str, Any]
            Dictionary with restricted builtins and allowed imports.
        """
        # Restricted builtins - only safe operations
        safe_builtins = {
            '__import__': __import__,  # Required for import statements to work
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
    
    def execute(self, code: str) -> CodeExecutionResult:
        """Execute Python code with safety restrictions.
        
        Parameters
        ----------
        code : str
            Python code to execute.
            
        Returns
        -------
        CodeExecutionResult
            Result object containing success status, output, and any errors.
        """
        # Validate syntax first
        is_valid, error_msg = self._validate_syntax(code)
        if not is_valid:
            return CodeExecutionResult(
                success=False,
                output="",
                error=error_msg
            )
        
        # Validate imports using AST
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
                        if base_module not in self.allowed_imports:
                            return CodeExecutionResult(
                                success=False,
                                output="",
                                error=(
                                    f"Import of '{module_name}' is not allowed. "
                                    f"Allowed modules: {', '.join(sorted(self.allowed_imports))}"
                                )
                            )
        except Exception as e:
            return CodeExecutionResult(
                success=False,
                output="",
                error=f"Import validation error: {str(e)}"
            )
        
        # Create restricted execution environment
        restricted_globals = self._create_restricted_globals()
        
        # Pre-import allowed modules to make them available
        for module_name in self.allowed_imports:
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
                error=None
            )
            
        except Exception as e:
            # Capture any execution errors
            error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            
            return CodeExecutionResult(
                success=False,
                output=captured_output.getvalue(),
                error=error_msg
            )
            
        finally:
            # Restore stdout
            sys.stdout = old_stdout


def execute_python_code(code: str, timeout: int = 30) -> dict[str, Any]:
    """Convenience function to execute Python code.
    
    Parameters
    ----------
    code : str
        Python code to execute.
    timeout : int, optional
        Maximum execution time in seconds, by default 30.
        
    Returns
    -------
    dict[str, Any]
        Dictionary with 'success', 'output', and 'error' keys.
    """
    executor = PythonCodeExecutor(timeout=timeout)
    result = executor.execute(code)
    return result.to_dict()
