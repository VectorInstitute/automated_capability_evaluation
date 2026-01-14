
import json
import re
import logging
from typing import Dict, Any, Optional

from autogen_core.models import (
    ChatCompletionClient,
    SystemMessage,
    UserMessage
)

log = logging.getLogger(__name__)

class FinancialProblemSolver:
    def __init__(self, model_client: ChatCompletionClient):
        self.model_client = model_client

    async def solve_problem(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Solves a single financial problem using the LLM.
        """
        task_type = problem.get("task")
        question = problem.get("question")
        choices = problem.get("choice", "")
        
        # Construct the prompt based on task type
        prompt = f"Question: {question}\n"
        
        if choices:
            prompt += f"Choices:\n{choices}\n"
            
        prompt += "\nProvide your answer in the following JSON format:\n"
        prompt += "{\n"
        prompt += '  "reasoning": "Your step-by-step reasoning here",\n'
        
        if task_type == "mcq":
            prompt += '  "answer": "The option letter (e.g., A, B, C)"\n'
        elif task_type == "bool":
            prompt += '  "answer": "1.0 for True/Yes, 0.0 for False/No"\n'
        elif task_type == "calcu":
             prompt += '  "answer": "The numerical value only (no units)"\n'
        else:
            prompt += '  "answer": "Your final answer"\n'
            
        prompt += "}\n"
        prompt += "IMPORTANT: Return only valid JSON. Do not include markdown formatting like ```json ... ```."

        system_message = SystemMessage(
            content="You are a financial expert. Solve the given problem accurately. Follow the specific format for the answer."
        )
        user_message = UserMessage(content=prompt, source="user")

        try:
            response = await self.model_client.create([system_message, user_message])
            content = str(response.content).strip()
            
            # Robust Extraction Logic
            prediction = None
            reasoning = "No reasoning provided"
            
            # Attempt 1: Regex Extraction (More robust than pure JSON parsing)
            # Look for JSON-like patterns or explicit labels
            
            # Try to find a JSON block first
            json_match = re.search(r"\{.*\}", content, re.DOTALL)
            if json_match:
                try:
                    # Clean up the potential JSON string
                    json_str = json_match.group(0)
                    # Handle common LLM JSON errors like unescaped newlines
                    json_str = re.sub(r'"\s*\n\s*"', '", "', json_str)
                    
                    parsed = json.loads(json_str)
                    prediction = parsed.get("answer")
                    reasoning = parsed.get("reasoning", reasoning)
                except Exception:
                    pass # Fallback to regex if JSON fails
            
            # If JSON parsing failed or didn't find the answer, try direct regex
            if prediction is None:
                # Look for "answer": "..." or "answer": 123
                ans_match = re.search(r'"answer":\s*["\']?(.*?)["\']?[\s,}]', content, re.IGNORECASE)
                if ans_match:
                    prediction = ans_match.group(1).strip()
                
                # Look for reasoning
                reas_match = re.search(r'"reasoning":\s*["\']?(.*?)["\']?[\s,}]', content, re.IGNORECASE | re.DOTALL)
                if reas_match:
                    reasoning = reas_match.group(1).strip()
                else:
                    # Use the whole content if reasoning not found separately
                    reasoning = content

            # Final Cleanup
            if prediction is None and not json_match:
                # If everything failed, the model might have just outputted the answer directly
                # (especially smaller models)
                prediction = content
                reasoning = "Raw model response"

            return {
                "id": problem.get("id"),
                "prediction": prediction,
                "reasoning": reasoning,
                "ground_truth": problem.get("ground_truth"),
                "task_type": task_type,
                "raw_response": content
            }

        except Exception as e:
            log.error(f"Error calling model for problem {problem.get('id')}: {e}")
            return {
                "id": problem.get("id"),
                "prediction": None,
                "reasoning": f"Model Error: {str(e)}",
                "ground_truth": problem.get("ground_truth"),
                "task_type": task_type
            }

