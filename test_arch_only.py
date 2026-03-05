#!/usr/bin/env python3
"""Quick test for arch library familiarity only"""

import asyncio
import sys
import json
sys.path.insert(0, 'src')

from scripts.assess_tool_familiarity import PROBES
from utils.model_client_utils import get_model_client
from utils.tool_assisted_prompts import TOOL_ASSISTED_SYSTEM_MESSAGE, TOOL_ASSISTED_ROUND_1_PROMPT
from tools.executor import PythonExecutor
from autogen_core.models import SystemMessage, UserMessage

async def test_arch(model_name="gemini-3-flash-preview"):
    client = get_model_client(model_name)
    executor = PythonExecutor(
        allowed_imports=["arch", "numpy", "pandas", "scipy", "datetime", "math", "statsmodels"]
    )
    
    print(f"\n{'='*70}")
    print(f"Testing arch familiarity with {model_name}...")
    print(f"{'='*70}\n")
    
    arch_probes = PROBES["arch"]
    success_count = 0
    
    for i, prompt in enumerate(arch_probes, 1):
        print(f"[Q{i}/5] {prompt[:60]}...")
        
        try:
            tool_context_str = f"""You have access to financial computing libraries.

Available libraries: arch

Use these libraries to solve the problem."""
            
            formatted_prompt = TOOL_ASSISTED_ROUND_1_PROMPT.format(
                problem_text=prompt,
                tool_context=tool_context_str
            )
            
            resp = await client.create([
                SystemMessage(content=TOOL_ASSISTED_SYSTEM_MESSAGE),
                UserMessage(content=formatted_prompt, source="user")
            ])
            
            # Parse JSON response
            try:
                response_data = json.loads(resp.content)
                code = response_data.get("code")
                
                if code is None:
                    print(f"  ✗ FAIL: No code provided\n")
                    continue
                    
            except json.JSONDecodeError as e:
                print(f"  ✗ FAIL: JSON parse error: {str(e)[:100]}\n")
                continue
            exec_result = executor.execute(code)
            
            if exec_result.success:
                print(f"  ✓ PASS\n")
                success_count += 1
            else:
                print(f"  ✗ FAIL: {exec_result.error[:150]}\n")
        except Exception as e:
            print(f"  ✗ ERROR: {str(e)[:150]}\n")
    
    print(f"{'='*70}")
    print(f"Result: {success_count}/5")
    print(f"Status: {'✓ PASS (≥4/5)' if success_count >= 4 else '✗ FAIL (<4/5)'}")
    print(f"{'='*70}\n")
    
    return success_count >= 4

if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "gemini-3-flash-preview"
    passed = asyncio.run(test_arch(model))
    sys.exit(0 if passed else 1)
