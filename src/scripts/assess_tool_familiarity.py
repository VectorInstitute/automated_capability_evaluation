#!/usr/bin/env python3
"""
Tool Familiarity Probe

Assesses LLM familiarity with financial computing libraries by testing basic usage.
If a tool scores < 4/5 on syntax-free execution, it's deemed "unfamiliar"
and should be given few-shot examples in TOOL_ASSISTED_ROUND_1_PROMPT.
"""

import asyncio
import logging
from typing import Dict

from dotenv import load_dotenv


load_dotenv()

from autogen_core.models import SystemMessage, UserMessage

from src.tools.executor import PythonExecutor
from src.utils.model_client_utils import get_model_client
from src.utils.tool_assisted_prompts import (
    TOOL_ASSISTED_ROUND_1_PROMPT,
    TOOL_ASSISTED_SYSTEM_MESSAGE,
)


logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


# 5 basic questions per library to probe familiarity
PROBES = {
    "numpy_financial": [
        "Calculate the Net Present Value (NPV) of cash flows: [-1000, 200, 300, 400, 500] with a discount rate of 0.10 (10%).",
        "Calculate the Internal Rate of Return (IRR) for cash flows: [-1000, 300, 400, 400, 400].",
        "Calculate the PMT (payment) for a loan: present value = 100000, annual rate = 0.05, nper = 360 months.",
        "Calculate the Future Value (FV) for an investment: rate = 0.06, nper = 10 years, pmt = -100, pv = -1000.",
        "Calculate the number of periods (NPER) needed: rate = 0.05, pmt = -200, pv = 5000, fv = 0.",
    ],
    "py_vollib": [
        "Calculate the call option price using Black-Scholes: S=100, K=105, T=1, r=0.05, sigma=0.20.",
        "Calculate the put option price using Black-Scholes: S=100, K=95, T=1, r=0.05, sigma=0.20.",
        "Calculate the delta (first Greek) for a call option: S=100, K=100, T=1, r=0.05, sigma=0.25.",
        "Calculate implied volatility for a call option with price 7.0: S=100, K=105, T=1, r=0.05.",
        "Calculate gamma (second Greek) for an option: S=100, K=100, T=1, r=0.05, sigma=0.20.",
    ],
    "pypfopt": [
        "Given returns matrix [[0.1, 0.2], [0.15, 0.1]] and covariance matrix [[0.01, 0.005], [0.005, 0.02]], calculate the minimum volatility portfolio.",
        "Calculate the Sharpe ratio for a portfolio with annual return 0.12, risk-free rate 0.02, and volatility 0.15.",
        "Create an EfficientFrontier object with expected_returns [0.1, 0.12, 0.15] and a sample covariance matrix.",
        "Optimize for maximum Sharpe ratio given expected returns and sample covariance matrix.",
        "Calculate the maximum Sharpe ratio portfolio weights for 3 assets with given returns and covariance.",
    ],
    "empyrical": [
        "Calculate the Sharpe ratio for returns [0.01, 0.02, -0.01, 0.03, 0.02] with risk-free rate 0.0.",
        "Calculate the Sortino ratio for returns [0.01, 0.02, -0.01, 0.03, 0.02] with minimum acceptable return 0.0.",
        "Calculate the maximum drawdown for returns [0.05, -0.10, 0.08, 0.12, -0.05].",
        "Calculate the annual return for daily returns [0.001, 0.002, -0.001, 0.003, 0.002].",
        "Calculate alpha and beta for asset returns [0.01, 0.02, -0.01, 0.03, 0.02] against benchmark [0.005, 0.01, 0.005, 0.015, 0.01].",
    ],
    "arch": [
        "Fit a GARCH(1,1) model to simulated returns data with 100 observations.",
        "Fit an EGARCH model to returns data and generate 5-step ahead forecasts using simulation method.",
        "Create a ConstantMean model with GARCH volatility and fit it to return data.",
        "Generate 5-step ahead forecasts from a fitted GARCH model using simulation method.",
        "Fit a HARX model using the arch.univariate.HARX class to time series data.",
    ],
}


async def assess_tool_familiarity(
    model_name: str = "gemini-2.5-pro", threshold: int = 4
) -> Dict[str, Dict[str, any]]:
    """Assess LLM familiarity with financial computing libraries.

    Args:
        model_name: LLM model to use for code generation
        threshold: Success count threshold (out of 5) to deem tool "familiar"

    Returns
    -------
        Dictionary with results for each tool including score and recommendations
    """
    client = get_model_client(model_name)
    executor = PythonExecutor(
        allowed_imports=[
            "numpy_financial",
            "py_vollib",
            "pypfopt",
            "empyrical",
            "arch",
            "numpy",
            "pandas",
            "scipy",
            "datetime",
            "math",
            "statsmodels",
        ]
    )

    results = {}

    for tool, prompts in PROBES.items():
        log.info(f"\n{'=' * 70}")
        log.info(f"Testing {tool} familiarity...")
        log.info(f"{'=' * 70}")

        success_count = 0
        failures = []

        for i, prompt in enumerate(prompts, 1):
            log.info(f"\n[Question {i}/5] {prompt[:80]}...")

            try:
                # Generate code using the same prompt structure as the actual benchmark
                tool_context_str = f"""You have access to financial computing libraries.

Available libraries: {tool}

Use these libraries to solve the problem."""

                formatted_prompt = TOOL_ASSISTED_ROUND_1_PROMPT.format(
                    problem_text=prompt, tool_context=tool_context_str
                )

                resp = await client.create(
                    [
                        SystemMessage(content=TOOL_ASSISTED_SYSTEM_MESSAGE),
                        UserMessage(content=formatted_prompt, source="user"),
                    ]
                )

                # Parse JSON response
                import json

                try:
                    response_data = json.loads(resp.content)
                    code = response_data.get("code")

                    if code is None:
                        log.warning("  ✗ No code provided in response")
                        failures.append(
                            {
                                "question": i,
                                "error": "No code provided in JSON response",
                            }
                        )
                        continue

                except json.JSONDecodeError as e:
                    log.error(f"  ✗ Failed to parse JSON response: {str(e)[:100]}")
                    failures.append(
                        {"question": i, "error": f"JSON parse error: {str(e)[:200]}"}
                    )
                    continue

                # Execute code
                exec_result = executor.execute(code)

                if exec_result.success:
                    log.info("  ✓ Code executed successfully")
                    success_count += 1
                else:
                    log.warning(
                        f"  ✗ Execution failed: {exec_result.error[:100] if exec_result.error else 'Unknown error'}"
                    )
                    failures.append(
                        {
                            "question": i,
                            "error": exec_result.error[:200]
                            if exec_result.error
                            else "No error details",
                        }
                    )
            except Exception as e:
                log.error(f"  ✗ Exception: {str(e)[:100]}")
                failures.append({"question": i, "error": str(e)[:200]})

        # Determine recommendation
        is_familiar = success_count >= threshold
        recommendation = (
            "PASS"
            if is_familiar
            else "WARN: Add few-shot examples to TOOL_ASSISTED_ROUND_1_PROMPT"
        )

        results[tool] = {
            "score": f"{success_count}/5",
            "success_count": success_count,
            "is_familiar": is_familiar,
            "recommendation": recommendation,
            "failures": failures if failures else [],
        }

        log.info(f"\n{tool} Score: {success_count}/5")
        log.info(f"Recommendation: {recommendation}")

    # Print summary
    log.info(f"\n{'=' * 70}")
    log.info("SUMMARY")
    log.info(f"{'=' * 70}")

    unfamiliar_tools = [
        tool for tool, result in results.items() if not result["is_familiar"]
    ]

    if unfamiliar_tools:
        log.warning(f"\n⚠️  UNFAMILIAR TOOLS (score < {threshold}/5):")
        for tool in unfamiliar_tools:
            log.warning(f"  - {tool}: {results[tool]['score']}")
        log.warning(
            "\nACTION REQUIRED: Add few-shot examples for these tools to TOOL_ASSISTED_ROUND_1_PROMPT"
        )
        log.warning("in src/utils/tool_assisted_prompts.py")
    else:
        log.info(f"\n✓ All tools meet the familiarity threshold ({threshold}/5)")
        log.info("  No action required.")

    return results


async def main():
    """Main execution function."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Assess LLM familiarity with financial computing libraries"
    )
    parser.add_argument(
        "--model",
        default="gemini-2.5-pro",
        help="LLM model to use (default: gemini-2.5-pro)",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=4,
        help="Success count threshold to deem tool 'familiar' (default: 4/5)",
    )

    args = parser.parse_args()

    results = await assess_tool_familiarity(
        model_name=args.model, threshold=args.threshold
    )

    # Save results
    import json
    from datetime import datetime
    from pathlib import Path

    output_file = Path("tool_familiarity_results.json")
    with open(output_file, "w") as f:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "model": args.model,
                "threshold": args.threshold,
                "results": results,
            },
            f,
            indent=2,
        )

    log.info(f"\n✓ Results saved to: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
