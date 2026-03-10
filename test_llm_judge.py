#!/usr/bin/env python3
"""Quick test for LLM-as-a-Judge verifier"""

import asyncio
import sys


sys.path.insert(0, "src")

from dotenv import load_dotenv


load_dotenv()

from autogen_core.models import SystemMessage, UserMessage

from utils.model_client_utils import get_model_client


async def test_llm_judge(expected, prediction, should_pass):
    """Test a single case"""
    judge_prompt = f"""You are an expert financial auditor grading an AI's mathematical execution. Your goal is to determine if the AI produced the mathematically exact final answer.

In corporate finance and derivatives pricing, precision is paramount. Your grading criteria must strictly reward exact computation and penalize manual estimation.

The exact expected answer is: {expected}
The AI's prediction is: {prediction}

GRADING RULES:
1. STRICT COMPUTATION: You must PENALIZE answers that suffer from intermediate rounding errors, accumulated floating-point drift, or arithmetic approximations. If the prediction differs from the expected answer due to these errors (e.g., 1023654.52 vs expected 1023654.55, or 1.259 vs expected 1.2592), mark it as FALSE.
2. FORGIVE FORMATTING: You must IGNORE superficial formatting differences. For example, treat "$1.41", "1.41", and "1.4100" as identical. Treat "15.58%" and "0.1558" as identical.

Based on these rules, did the AI predict the exact correct number?

Output ONLY the word "TRUE" or "FALSE". Do not provide any other explanation."""

    client = get_model_client("gemini-3-flash-preview")
    response = await client.create(
        [
            SystemMessage(content="You are a precise numerical evaluator."),
            UserMessage(content=judge_prompt, source="user"),
        ]
    )

    judgment = response.content.strip().upper()
    result = "TRUE" in judgment

    status = "✓ PASS" if result == should_pass else "✗ FAIL"
    print(
        f"{status}: Expected={expected}, Prediction={prediction}, Judge={judgment} (expected {should_pass})"
    )
    return result == should_pass


async def main():
    """Run test cases"""
    print("\n" + "=" * 70)
    print("Testing LLM-as-a-Judge Verifier")
    print("=" * 70 + "\n")

    test_cases = [
        # (expected, prediction, should_pass, description)
        ("1.41", "$1.41", True, "Formatting: $ symbol"),
        ("1.41", "1.4100", True, "Formatting: trailing zeros"),
        ("0.1558", "15.58%", True, "Formatting: % to decimal"),
        ("1023654.55", "1023654.52", False, "Precision: rounding error"),
        ("1.2592", "1.259", False, "Precision: truncation error"),
        ("100.00", "100", True, "Formatting: decimal places"),
        ("5000", "$5,000", True, "Formatting: commas"),
        ("0.05", "5%", True, "Formatting: percentage"),
    ]

    results = []
    for expected, prediction, should_pass, description in test_cases:
        print(f"\nTest: {description}")
        result = await test_llm_judge(expected, prediction, should_pass)
        results.append(result)

    print("\n" + "=" * 70)
    print(f"Results: {sum(results)}/{len(results)} tests passed")
    print("=" * 70 + "\n")

    return all(results)


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
