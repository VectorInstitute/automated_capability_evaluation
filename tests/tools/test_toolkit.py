"""Tests for simplified toolkit."""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, Mock

from src.tools.toolkit import ScientificToolKit


class TestScientificToolKit:
    """Test suite for ScientificToolKit."""
    
    def test_toolkit_initialization(self):
        """Test toolkit initializes with direct instantiation."""
        mock_client = Mock()
        
        toolkit = ScientificToolKit(
            model_client=mock_client,
            enable_tool_selection=True,
        )
        
        assert toolkit is not None
        assert toolkit.executor is not None
        assert toolkit.tool is not None
    
    def test_code_execution(self):
        """Test code execution works."""
        mock_client = Mock()
        toolkit = ScientificToolKit(model_client=mock_client)
        
        result = toolkit.execute_code("print(2 + 2)")
        
        assert result["success"] is True
        assert "4" in result["output"]
    
    async def test_tool_preparation_disabled(self):
        """Test tool preparation with selection disabled."""
        mock_client = Mock()
        toolkit = ScientificToolKit(
            model_client=mock_client,
            enable_tool_selection=False
        )
        
        result = await toolkit.prepare_tools("Find sqrt(16)")
        
        assert result["needs_tools"] is True
        assert len(result["selected_libraries"]) > 0
    
    async def test_tool_preparation_enabled(self):
        """Test tool preparation with selection enabled."""
        mock_client = AsyncMock()
        
        # Mock LLM responses
        stage1_response = Mock()
        stage1_response.content = '''
        {
            "needs_tools": true,
            "reasoning": "Need numerical computation",
            "selected_libraries": [
                {
                    "library_name": "numpy",
                    "rationale": "For numerical operations",
                    "confidence": "high"
                }
            ],
            "confidence": "high"
        }
        '''
        
        stage2_response = Mock()
        stage2_response.content = '''
        {
            "selected_modules": [
                {
                    "library": "numpy",
                    "module": "sqrt",
                    "purpose": "Calculate square root"
                }
            ],
            "approach": "Use np.sqrt"
        }
        '''
        
        mock_client.create.side_effect = [stage1_response, stage2_response]
        
        toolkit = ScientificToolKit(
            model_client=mock_client,
            enable_tool_selection=True
        )
        
        result = await toolkit.prepare_tools("Find sqrt(16)")
        
        assert result["needs_tools"] is True
        assert "numpy" in result["selected_libraries"]
    
    def test_format_tool_context(self):
        """Test formatting of tool context."""
        mock_client = Mock()
        toolkit = ScientificToolKit(model_client=mock_client)
        
        context = {
            "needs_tools": True,
            "selected_libraries": ["numpy", "scipy"],
            "approach": "Use numerical methods",
            "selected_modules": []
        }
        
        formatted = toolkit.format_tool_context(context)
        
        assert "numpy" in formatted
        assert "scipy" in formatted
        assert "Use numerical methods" in formatted
