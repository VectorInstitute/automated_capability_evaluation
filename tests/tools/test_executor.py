"""Unit tests for Python executor."""

import pytest

from src.tools.executor import PythonExecutor, execute_python_code


class TestPythonExecutor:
    """Test suite for PythonExecutor."""
    
    def test_simple_execution(self):
        """Test executing simple Python code."""
        executor = PythonExecutor()
        
        code = "print('Hello, World!')"
        result = executor.execute(code)
        
        assert result.success
        assert "Hello, World!" in result.output
        assert result.error is None
    
    def test_math_computation(self):
        """Test mathematical computation."""
        executor = PythonExecutor()
        
        code = """
import math
result = math.sqrt(16)
print(f"Result: {result}")
"""
        result = executor.execute(code)
        
        assert result.success
        assert "Result: 4.0" in result.output
    
    def test_numpy_execution(self):
        """Test NumPy code execution."""
        executor = PythonExecutor()
        
        code = """
import numpy as np
arr = np.array([1, 2, 3, 4])
print(f"Sum: {arr.sum()}")
"""
        result = executor.execute(code)
        
        assert result.success
        assert "Sum: 10" in result.output
    
    def test_syntax_error(self):
        """Test handling syntax errors."""
        executor = PythonExecutor()
        
        code = "print('unclosed string"
        result = executor.execute(code)
        
        assert not result.success
        assert result.error is not None
        assert "Syntax error" in result.error
    
    def test_runtime_error(self):
        """Test handling runtime errors."""
        executor = PythonExecutor()
        
        code = "x = 1 / 0"
        result = executor.execute(code)
        
        assert not result.success
        assert result.error is not None
        assert "ZeroDivisionError" in result.error
    
    def test_restricted_import(self):
        """Test that restricted imports are blocked."""
        executor = PythonExecutor()
        
        code = "import os"
        result = executor.execute(code)
        
        assert not result.success
        assert "not allowed" in result.error
    
    def test_sympy_execution(self):
        """Test SymPy code execution."""
        executor = PythonExecutor()
        
        code = """
from sympy import symbols, diff
x = symbols('x')
expr = x**2
derivative = diff(expr, x)
print(f"Derivative: {derivative}")
"""
        result = executor.execute(code)
        
        assert result.success
        assert "Derivative:" in result.output
    
    def test_validate_syntax(self):
        """Test syntax validation."""
        executor = PythonExecutor()
        
        # Valid syntax
        is_valid, error = executor.validate_syntax("print('hello')")
        assert is_valid
        assert error is None
        
        # Invalid syntax
        is_valid, error = executor.validate_syntax("print('unclosed")
        assert not is_valid
        assert error is not None
    
    def test_convenience_function(self):
        """Test convenience function."""
        code = "print('test')"
        result = execute_python_code(code)
        
        assert result["success"]
        assert "test" in result["output"]
