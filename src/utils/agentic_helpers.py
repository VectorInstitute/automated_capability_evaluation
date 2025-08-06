"""Helper functions for multi-agent debate-based capability generation."""

import json
import logging
import re
from typing import Any, Dict, Optional

from autogen import AssistantAgent


log = logging.getLogger("agentic_helpers")


def _to_autogen_cfg(
    node: Dict[str, Any], extra: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Convert node configuration to AutoGen format."""
    cfg = {
        "config_list": [{"model": node["name"]}],
        "seed": node.get("seed", 42),
    }

    if extra:
        cfg.update(extra)

    return cfg


def _make_scientist(label: str, cfg: Dict[str, Any], domain: str) -> AssistantAgent:
    """Create a scientist agent with domain-specific configuration."""
    system_message = f"""You are {label}, an expert scientist specializing in {domain}.
Your role is to collaborate with other scientists to design comprehensive capability hierarchies.
Focus on creating well-structured, practical capabilities that can be effectively evaluated.
Always provide detailed, thoughtful responses and consider multiple perspectives."""

    return AssistantAgent(
        name=label,
        system_message=system_message,
        llm_config=cfg,
    )


def _make_moderator(cfg: Dict[str, Any]) -> AssistantAgent:
    """Create a moderator agent to facilitate debate and reach consensus."""
    system_message = """You are the Moderator, responsible for facilitating productive debate between scientists.
Your role is to:
1. Ensure all perspectives are considered
2. Guide discussions toward consensus
3. Synthesize final outputs from debates
4. Maintain focus on the core objectives
Be fair, thorough, and decisive in your moderation."""

    return AssistantAgent(
        name="Moderator",
        system_message=system_message,
        llm_config=cfg,
    )


def _extract_json(text: str) -> Dict[str, Any]:
    """Extract JSON object from text with robust error handling."""
    # Try direct JSON parsing first
    try:
        result = json.loads(text)
        return result if isinstance(result, dict) else {}
    except json.JSONDecodeError:
        pass

    # Look for JSON code blocks
    code_block_pattern = r"```(?:json)?\s*(\{.*?\})\s*```"
    match = re.search(code_block_pattern, text, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group(1))
            return result if isinstance(result, dict) else {}
        except json.JSONDecodeError:
            pass

    # Try to find JSON object with balanced braces
    brace_count = 0
    start_idx = -1

    for i, char in enumerate(text):
        if char == "{":
            if brace_count == 0:
                start_idx = i
            brace_count += 1
        elif char == "}":
            brace_count -= 1
            if brace_count == 0 and start_idx != -1:
                try:
                    result = json.loads(text[start_idx : i + 1])
                    return result if isinstance(result, dict) else {}
                except json.JSONDecodeError:
                    pass

    log.error("Failed to extract JSON from text")
    return {}


def _ask(agent: AssistantAgent, prompt: str) -> str:
    """Single-turn chat wrapper for AutoGen agents.

    Handles different AutoGen versions and response formats.
    """
    try:
        if hasattr(agent, "generate_reply"):
            # AutoGen 0.2.x
            result = agent.generate_reply(
                messages=[{"role": "user", "content": prompt}],
                sender=None,
            )
            if isinstance(result, dict):
                content = result.get("content", str(result))
                return str(content)
            return str(result)
        # AutoGen 0.1.x
        result = agent.generate_reply(
            messages=[{"role": "user", "content": prompt}],
            sender=None,
        )
        return str(result)
    except Exception as e:
        log.error("Error getting response from %s: %s", agent.name, e)
        raise


def _debate_once(
    scientist_a: AssistantAgent,
    scientist_b: AssistantAgent,
    moderator: AssistantAgent,
    prompt: str,
    max_rounds: int = 3,
) -> Dict[str, Any]:
    """Run a single debate round between scientists moderated by moderator."""
    log.info("Starting debate round")

    # Initial responses from both scientists
    response_a = _ask(scientist_a, prompt)
    response_b = _ask(scientist_b, prompt)

    # Moderator synthesizes and asks for refinement
    synthesis_prompt = f"""Review the following responses from two scientists:

Scientist A: {response_a}

Scientist B: {response_b}

Synthesize these perspectives and identify areas of agreement and disagreement.
Provide a final consensus output that incorporates the best elements from both responses.
Focus on creating a comprehensive, well-structured result."""

    final_response = _ask(moderator, synthesis_prompt)

    # Extract structured output
    try:
        return _extract_json(final_response)
    except Exception:
        log.error("Failed to extract final result - returning empty structure")
        return {"finalized": False}


def _stub_class(name: str, desc: str) -> str:
    """Generate Python class string for Capability.from_dict."""
    return f'''class Capability:
    """Autogenerated capability for {name}. {desc}"""

    @staticmethod
    def repr_tasks():
        return {{}}

    @staticmethod
    def get_instructions(t):
        return f"Solve the following problem: {{t['problem']}}"

    @staticmethod
    def score(answer, target):
        return answer.strip().lower() == target['answer'].strip().lower()
'''
