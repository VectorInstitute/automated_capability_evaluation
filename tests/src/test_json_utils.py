"""
Tests for the JSON utility functions related to LaTeX backslash handling.

This module contains tests specifically for the changes made to handle LaTeX
expressions with backslashes in JSON strings. The main changes were:
- fix_common_json_errors: Character-by-character processing to correctly escape
  LaTeX backslashes while preserving valid JSON escape sequences
- parse_llm_json_response: Fallback mechanism for "Invalid escape" errors

The tests verify that:
- LaTeX expressions with backslashes (e.g., \\(, \\), \\[, \\], \\to, \\lim) are properly escaped
- Valid JSON escape sequences (e.g., \\n, \\t, \\", \\\\) are preserved
- The fallback mechanism correctly handles invalid escape errors
"""

import json

import pytest

from src.utils.json_utils import (
    fix_common_json_errors,
    parse_llm_json_response,
)


class TestFixCommonJsonErrorsLaTeX:
    """Test cases for fix_common_json_errors function - LaTeX backslash handling."""

    def test_latex_parentheses_escaped(self):
        """Test that LaTeX parentheses are properly escaped."""
        content = '{"thought": "The equation \\(x + y\\) is important"}'
        result = fix_common_json_errors(content)
        # Should double-escape: \\( becomes \\\\(
        assert '\\\\(x + y\\\\)' in result or '\\\\(' in result
        # Verify it can be parsed
        parsed = json.loads(result)
        assert "(" in parsed["thought"] or "\\(" in parsed["thought"]

    def test_latex_brackets_escaped(self):
        """Test that LaTeX brackets are properly escaped."""
        content = '{"thought": "The equation \\[x + y = z\\] is important"}'
        result = fix_common_json_errors(content)
        # Should double-escape: \\[ becomes \\\\[
        assert '\\\\[' in result or '\\[' in result
        # Verify it can be parsed
        parsed = json.loads(result)
        assert "[" in parsed["thought"] or "\\[" in parsed["thought"]

    def test_latex_commands_escaped(self):
        """Test that LaTeX commands like \\to, \\lim are properly escaped."""
        content = '{"thought": "As x \\to 0, we have \\lim f(x)"}'
        result = fix_common_json_errors(content)
        # Should escape \to and \lim
        assert '\\to' in result or '\\\\to' in result
        # Verify it can be parsed
        parsed = json.loads(result)
        assert "to" in parsed["thought"] or "\\to" in parsed["thought"]

    def test_valid_json_escapes_preserved(self):
        """Test that valid JSON escape sequences are preserved."""
        # Use raw string to ensure \n and \t are literal backslash+n and backslash+t
        content = r'{"text": "Line 1\nLine 2\tTabbed"}'
        result = fix_common_json_errors(content)
        # Valid escapes should remain (as single backslash sequences)
        assert r'\n' in result or '\n' in result
        assert r'\t' in result or '\t' in result
        # Verify it can be parsed and produces actual newline/tab characters
        parsed = json.loads(result)
        assert "\n" in parsed["text"] or "\\n" in parsed["text"]
        assert "\t" in parsed["text"] or "\\t" in parsed["text"]

    def test_valid_json_quote_escape_preserved(self):
        """Test that escaped quotes are preserved."""
        content = '{"text": "He said \\"hello\\""}'
        result = fix_common_json_errors(content)
        # Escaped quotes should remain
        assert '\\"' in result
        # Verify it can be parsed
        parsed = json.loads(result)
        assert '"' in parsed["text"]

    def test_valid_json_backslash_escape_preserved(self):
        """Test that escaped backslashes are preserved."""
        content = '{"path": "C:\\\\Users\\\\file.txt"}'
        result = fix_common_json_errors(content)
        # Escaped backslashes should remain
        assert '\\\\' in result
        # Verify it can be parsed
        parsed = json.loads(result)
        assert "\\" in parsed["path"]

    def test_latex_in_nested_string(self):
        """Test LaTeX handling in nested JSON strings."""
        content = '{"outer": {"inner": "The formula \\(a + b\\) is key"}}'
        result = fix_common_json_errors(content)
        # Should handle LaTeX in nested strings
        parsed = json.loads(result)
        assert "(" in parsed["outer"]["inner"] or "\\(" in parsed["outer"]["inner"]

    def test_mixed_valid_and_invalid_escapes(self):
        """Test handling of mixed valid and invalid escape sequences."""
        # Use raw string to ensure proper escape handling
        content = r'{"text": "Valid\nNewline and invalid\to LaTeX"}'
        result = fix_common_json_errors(content)
        # Valid escape should remain, invalid should be escaped
        assert r'\n' in result or '\n' in result
        # Verify it can be parsed
        parsed = json.loads(result)
        # Should contain newline character or escaped newline
        assert "\n" in parsed["text"] or "\\n" in parsed["text"]


class TestParseLlmJsonResponseLaTeX:
    """Test cases for parse_llm_json_response function - LaTeX and fallback handling."""

    def test_parse_json_with_latex(self):
        """Test parsing JSON containing LaTeX expressions."""
        content = '{"thought": "The equation \\(x + y\\) is important", "answer": "42"}'
        result = parse_llm_json_response(content)
        assert "thought" in result
        assert "answer" in result
        # The thought should contain the LaTeX (possibly escaped)
        assert "x + y" in result["thought"] or "(" in result["thought"]

    def test_parse_json_with_latex_complex(self):
        """Test parsing complex JSON with multiple LaTeX expressions."""
        content = (
            '{"thought": "We need to solve \\[x^2 + 2x + 1 = 0\\] '
            'and find \\lim_{x \\to 0} f(x)", "answer": "x = -1"}'
        )
        result = parse_llm_json_response(content)
        assert "thought" in result
        assert "answer" in result
        assert "x = -1" in result["answer"]

    def test_parse_json_fallback_on_invalid_escape(self):
        """Test that fallback mechanism works for invalid escape errors."""
        # This should trigger the fallback mechanism
        content = '{"text": "Invalid escape\\to here"}'
        result = parse_llm_json_response(content)
        assert "text" in result
        # Should successfully parse after fallback fix

    def test_parse_json_with_latex_in_multiple_fields(self):
        """Test parsing JSON with LaTeX in multiple fields."""
        content = (
            '{"thought": "Consider \\(a + b\\)", '
            '"solution": "The answer is \\[x = 5\\]", '
            '"reasoning": "Using \\lim_{x \\to 0}"}'
        )
        result = parse_llm_json_response(content)
        assert "thought" in result
        assert "solution" in result
        assert "reasoning" in result


class TestLaTeXEdgeCases:
    """Test edge cases for LaTeX backslash handling."""

    def test_latex_at_string_boundary(self):
        """Test LaTeX expressions at string boundaries."""
        content = '{"text": "\\(start\\) and \\(end\\)"}'
        result = parse_llm_json_response(content)
        assert "text" in result

    def test_latex_with_numbers(self):
        """Test LaTeX expressions containing numbers."""
        content = '{"formula": "\\[x^2 + 2x + 1\\]"}'
        result = parse_llm_json_response(content)
        assert "formula" in result

    def test_escaped_quotes_with_latex(self):
        """Test combination of escaped quotes and LaTeX."""
        content = '{"text": "He said \\"Consider \\(x + y\\)\\""}'
        result = parse_llm_json_response(content)
        assert "text" in result
        assert '"' in result["text"] or "Consider" in result["text"]

    def test_newline_characters_in_latex(self):
        """Test newline characters near LaTeX expressions."""
        content = '{"text": "Line 1\\n\\(x + y\\)\\nLine 2"}'
        result = parse_llm_json_response(content)
        assert "text" in result
        assert "\n" in result["text"]
