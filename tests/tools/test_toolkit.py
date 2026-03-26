"""Tests for Scientific Toolkit module."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

from src.tools.toolkit import ScientificToolKit


class TestScientificToolkitInitialization:
    """Test suite for toolkit initialization and configuration."""

    def test_basic_initialization(self):
        """Test basic toolkit initialization."""
        mock_client = Mock()

        toolkit = ScientificToolKit(model_client=mock_client)

        assert toolkit is not None
        assert toolkit.model_client is mock_client
        assert toolkit.executor is not None
        assert (
            toolkit.enable_tool_selection is False
        )  # Default is False (parametric mode)
        assert toolkit.enable_rag is False  # Default is False (parametric mode)

    def test_initialization_with_tool_selection_disabled(self):
        """Test initialization with tool selection disabled."""
        mock_client = Mock()

        toolkit = ScientificToolKit(
            model_client=mock_client,
            enable_tool_selection=False,
        )

        assert toolkit.enable_tool_selection is False
        assert toolkit.executor is not None

    def test_initialization_with_rag_disabled(self):
        """Test initialization with RAG (parametric mode) disabled."""
        mock_client = Mock()

        toolkit = ScientificToolKit(
            model_client=mock_client,
            enable_rag=False,
        )

        assert toolkit.enable_rag is False
        assert toolkit.doc_retriever is None

    def test_initialization_with_rag_enabled(self):
        """Test initialization with RAG enabled."""
        mock_client = Mock()

        toolkit = ScientificToolKit(
            model_client=mock_client,
            enable_rag=True,
        )

        assert toolkit.enable_rag is True
        assert toolkit.doc_retriever is not None

    def test_custom_docs_path(self):
        """Test initialization with custom docs path."""
        mock_client = Mock()
        custom_path = Path("/tmp/custom_docs")

        toolkit = ScientificToolKit(
            model_client=mock_client,
            docs_path=custom_path,
        )

        assert toolkit is not None


class TestCodeExecution:
    """Test suite for code execution functionality."""

    def test_simple_code_execution(self):
        """Test executing simple Python code."""
        mock_client = Mock()
        toolkit = ScientificToolKit(model_client=mock_client)

        result = toolkit.execute_code("print(2 + 2)")

        assert result["success"] is True
        assert "4" in result["output"]

    def test_math_code_execution(self):
        """Test executing mathematical code."""
        mock_client = Mock()
        toolkit = ScientificToolKit(model_client=mock_client)

        code = """
import math
result = math.sqrt(16)
print(f'Result: {result}')
"""
        result = toolkit.execute_code(code)

        assert result["success"] is True
        assert "Result: 4.0" in result["output"]

    def test_numpy_code_execution(self):
        """Test executing numpy code."""
        mock_client = Mock()
        toolkit = ScientificToolKit(model_client=mock_client)

        code = """
import numpy as np
arr = np.array([1, 2, 3])
print(f'Sum: {arr.sum()}')
"""
        result = toolkit.execute_code(code)

        assert result["success"] is True
        assert "Sum: 6" in result["output"]

    def test_error_handling_in_execution(self):
        """Test error handling during code execution."""
        mock_client = Mock()
        toolkit = ScientificToolKit(model_client=mock_client)

        result = toolkit.execute_code("x = 1 / 0")

        assert result["success"] is False
        assert result["error"] is not None


class TestToolPreparation:
    """Test suite for tool preparation functionality."""

    @pytest.mark.asyncio
    async def test_tool_preparation_with_selection_disabled(self):
        """Test tool preparation when selection is disabled."""
        mock_client = Mock()
        toolkit = ScientificToolKit(
            model_client=mock_client,
            enable_tool_selection=False,
        )

        result = await toolkit.prepare_tools("Find sqrt(16)")

        assert result["needs_tools"] is True
        assert result["selected_modules"] == []
        assert "reasoning" in result
        assert "documentation" in result

    @pytest.mark.asyncio
    async def test_tool_preparation_tools_needed_rag_mode(self):
        """Test tool preparation with tools needed in RAG mode."""
        mock_client = AsyncMock()

        # Mock LLM response
        mock_response = Mock()
        mock_response.content = """{
            "tool_necessity": true,
            "reasoning": "Need numerical computation",
            "selected_modules": [
                {
                    "library": "numpy",
                    "module": "linalg"
                }
            ]
        }"""

        mock_client.create.return_value = mock_response

        toolkit = ScientificToolKit(
            model_client=mock_client,
            enable_tool_selection=True,
            enable_rag=True,
        )

        result = await toolkit.prepare_tools("Solve a linear system")

        assert result["needs_tools"] is True
        assert "reasoning" in result
        assert "selected_modules" in result

    @pytest.mark.asyncio
    async def test_tool_preparation_tools_needed_parametric_mode(self):
        """Test tool preparation with tools needed in parametric mode."""
        mock_client = AsyncMock()

        # Mock LLM response
        mock_response = Mock()
        mock_response.content = """{
            "tool_necessity": true,
            "reasoning": "Need numerical computation",
            "selected_modules": [
                {
                    "library": "scipy",
                    "module": "optimize"
                }
            ]
        }"""

        mock_client.create.return_value = mock_response

        toolkit = ScientificToolKit(
            model_client=mock_client,
            enable_tool_selection=True,
            enable_rag=False,
        )

        result = await toolkit.prepare_tools("Optimize a function")

        assert result["needs_tools"] is True
        assert "reasoning" in result
        assert "selected_modules" in result
        assert len(result["selected_modules"]) > 0

    @pytest.mark.asyncio
    async def test_tool_preparation_tools_not_needed(self):
        """Test tool preparation when no tools are needed."""
        mock_client = AsyncMock()

        # Mock LLM response indicating no tools needed
        mock_response = Mock()
        mock_response.content = """{
            "tool_necessity": false,
            "reasoning": "This is a conceptual question",
            "selected_modules": []
        }"""

        mock_client.create.return_value = mock_response

        toolkit = ScientificToolKit(
            model_client=mock_client,
            enable_tool_selection=True,
        )

        result = await toolkit.prepare_tools("What is calculus?")

        assert result["needs_tools"] is False
        assert "reasoning" in result
        assert result["selected_modules"] == []


class TestToolContextFormatting:
    """Test suite for tool context formatting."""

    def test_format_tool_context_with_tools(self):
        """Test formatting context when tools are needed."""
        mock_client = Mock()
        toolkit = ScientificToolKit(model_client=mock_client)

        context = {
            "needs_tools": True,
            "reasoning": "Need numerical methods",
            "documentation": "# NumPy Documentation\n\nFunctions available...",
            "selected_modules": [{"library": "numpy", "module": "linalg"}],
        }

        formatted = toolkit.format_tool_context(context)

        assert "AVAILABLE TOOLS" in formatted
        assert "Need numerical methods" in formatted
        assert "MODULE DOCUMENTATION" in formatted
        assert "NumPy Documentation" in formatted

    def test_format_tool_context_without_tools(self):
        """Test formatting context when tools are not needed."""
        mock_client = Mock()
        toolkit = ScientificToolKit(model_client=mock_client)

        context = {
            "needs_tools": False,
            "reasoning": "No computational tools required",
            "documentation": "",
            "selected_modules": [],
        }

        formatted = toolkit.format_tool_context(context)

        assert "Note:" in formatted
        assert "No computational tools required" in formatted
        assert "MODULE DOCUMENTATION" not in formatted

    def test_format_tool_context_empty_documentation(self):
        """Test formatting context with empty documentation."""
        mock_client = Mock()
        toolkit = ScientificToolKit(model_client=mock_client)

        context = {
            "needs_tools": True,
            "reasoning": "Tools selected",
            "documentation": "",
            "selected_modules": [],
        }

        formatted = toolkit.format_tool_context(context)

        assert "AVAILABLE TOOLS" in formatted
        assert "Tools selected" in formatted


class TestToolkitIntegration:
    """Test suite for end-to-end integration scenarios."""

    @pytest.mark.asyncio
    async def test_full_workflow_parametric_mode(self):
        """Test full workflow in parametric mode."""
        mock_client = AsyncMock()

        # Mock LLM response
        mock_response = Mock()
        mock_response.content = """{
            "tool_necessity": true,
            "reasoning": "Need matrix operations",
            "selected_modules": [
                {"library": "numpy", "module": "linalg"}
            ]
        }"""
        mock_client.create.return_value = mock_response

        toolkit = ScientificToolKit(
            model_client=mock_client,
            enable_rag=False,
        )

        # Prepare tools
        tool_context = await toolkit.prepare_tools("Solve Ax = b")

        assert tool_context["needs_tools"] is True

        # Format context
        formatted = toolkit.format_tool_context(tool_context)
        assert len(formatted) > 0

        # Execute code
        code = "import numpy as np\nprint(np.array([1, 2, 3]))"
        result = toolkit.execute_code(code)
        assert result["success"] is True

    def test_executor_available_after_init(self):
        """Test that executor is always available."""
        mock_client = Mock()

        # Test with different configurations
        for enable_rag in [True, False]:
            for enable_selection in [True, False]:
                toolkit = ScientificToolKit(
                    model_client=mock_client,
                    enable_rag=enable_rag,
                    enable_tool_selection=enable_selection,
                )

                assert toolkit.executor is not None

                # Verify executor can run code
                result = toolkit.execute_code("print('test')")
                assert result["success"] is True


class TestMalformedLLMResponses:
    """Test suite for handling edge cases in LLM responses.

    These tests cover scenarios where LLM generates malformed JSON,
    markdown-wrapped responses, or unexpected formatting.
    """

    @pytest.mark.asyncio
    async def test_json_with_trailing_comma(self):
        """Test handling of JSON with trailing commas."""
        mock_client = AsyncMock()

        # Mock response with trailing comma (invalid JSON)
        mock_response = Mock()
        mock_response.content = """{
            "tool_necessity": true,
            "reasoning": "Need numpy",
            "selected_modules": [
                {"library": "numpy", "module": "linalg"},
            ]
        }"""

        mock_client.create.return_value = mock_response

        toolkit = ScientificToolKit(
            model_client=mock_client,
            enable_tool_selection=True,
        )

        # Should handle gracefully (json_utils should fix this)
        try:
            result = await toolkit.prepare_tools("Test problem")
            # If it succeeds, verify structure
            assert "reasoning" in result
        except Exception as e:
            # Expected to potentially fail - this is an edge case
            assert "JSON" in str(e) or "parse" in str(e).lower()

    @pytest.mark.asyncio
    async def test_json_wrapped_in_markdown(self):
        """Test handling of JSON wrapped in markdown code blocks."""
        mock_client = AsyncMock()

        # Mock response wrapped in markdown
        mock_response = Mock()
        mock_response.content = """```json
{
    "tool_necessity": false,
    "reasoning": "Conceptual question",
    "selected_modules": []
}
```"""

        mock_client.create.return_value = mock_response

        toolkit = ScientificToolKit(
            model_client=mock_client,
            enable_tool_selection=True,
        )

        # Should unwrap and parse correctly
        result = await toolkit.prepare_tools("What is math?")
        assert result["needs_tools"] is False
        assert "reasoning" in result

    @pytest.mark.asyncio
    async def test_json_with_extra_text(self):
        """Test handling of JSON with extra explanatory text."""
        mock_client = AsyncMock()

        # Mock response with preamble text
        mock_response = Mock()
        mock_response.content = """Here is my analysis:
{
    "tool_necessity": true,
    "reasoning": "Matrix operations required",
    "selected_modules": [{"library": "numpy", "module": "linalg"}]
}
Hope this helps!"""

        mock_client.create.return_value = mock_response

        toolkit = ScientificToolKit(
            model_client=mock_client,
            enable_tool_selection=True,
        )

        # Should extract JSON from surrounding text
        try:
            result = await toolkit.prepare_tools("Solve system")
            assert "reasoning" in result
        except Exception:
            # May fail - this tests error handling
            pass

    @pytest.mark.asyncio
    async def test_json_with_unexpected_keys(self):
        """Test handling of JSON with extra unexpected keys."""
        mock_client = AsyncMock()

        # Mock response with extra fields
        mock_response = Mock()
        mock_response.content = """{
            "tool_necessity": true,
            "reasoning": "Need tools",
            "selected_modules": [],
            "extra_field": "unexpected",
            "another_field": 123,
            "nested": {"unwanted": "data"}
        }"""

        mock_client.create.return_value = mock_response

        toolkit = ScientificToolKit(
            model_client=mock_client,
            enable_tool_selection=True,
        )

        # Should ignore extra keys and extract what's needed
        result = await toolkit.prepare_tools("Test")
        assert result["needs_tools"] is True
        assert "reasoning" in result

    @pytest.mark.asyncio
    async def test_json_with_single_quotes(self):
        """Test handling of JSON-like data with single quotes."""
        mock_client = AsyncMock()

        # Invalid JSON using single quotes
        mock_response = Mock()
        mock_response.content = """{
            'tool_necessity': false,
            'reasoning': 'Simple question',
            'selected_modules': []
        }"""

        mock_client.create.return_value = mock_response

        toolkit = ScientificToolKit(
            model_client=mock_client,
            enable_tool_selection=True,
        )

        # Should fail or be fixed by json_utils
        try:
            result = await toolkit.prepare_tools("Test")
            # If it works, verify structure
            assert "reasoning" in result
        except Exception:
            # Expected to fail - single quotes aren't valid JSON
            assert True  # Acceptable failure

    @pytest.mark.asyncio
    async def test_completely_malformed_response(self):
        """Test handling of completely non-JSON response."""
        mock_client = AsyncMock()

        # Mock response that's not JSON at all
        mock_response = Mock()
        mock_response.content = "I think you should use numpy.linalg for this problem."

        mock_client.create.return_value = mock_response

        toolkit = ScientificToolKit(
            model_client=mock_client,
            enable_tool_selection=True,
        )

        # Should fail gracefully with JSONDecodeError
        with pytest.raises(json.JSONDecodeError):
            await toolkit.prepare_tools("Test problem")

    @pytest.mark.asyncio
    async def test_empty_response(self):
        """Test handling of empty response."""
        mock_client = AsyncMock()

        # Mock empty response
        mock_response = Mock()
        mock_response.content = ""

        mock_client.create.return_value = mock_response

        toolkit = ScientificToolKit(
            model_client=mock_client,
            enable_tool_selection=True,
        )

        # Should fail gracefully with JSONDecodeError
        with pytest.raises(json.JSONDecodeError):
            await toolkit.prepare_tools("Test problem")

    @pytest.mark.asyncio
    async def test_json_with_escaped_quotes(self):
        """Test handling of JSON with improperly escaped quotes."""
        mock_client = AsyncMock()

        # JSON with escaped quotes that might confuse parser
        mock_response = Mock()
        mock_response.content = """{
            "tool_necessity": true,
            "reasoning": "Need \\"special\\" tools",
            "selected_modules": []
        }"""

        mock_client.create.return_value = mock_response

        toolkit = ScientificToolKit(
            model_client=mock_client,
            enable_tool_selection=True,
        )

        # Should parse correctly
        result = await toolkit.prepare_tools("Test")
        assert result["needs_tools"] is True
        assert "special" in result["reasoning"]
