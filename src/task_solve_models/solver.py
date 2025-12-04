
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
            content = str(response.content)
            
            # Simple cleanup for markdown code blocks if present
            content = re.sub(r"```json\s*", "", content)
            content = re.sub(r"```\s*$", "", content)
            content = content.strip()
            
            try:
                parsed_response = json.loads(content)
                return {
                    "id": problem.get("id"),
                    "prediction": parsed_response.get("answer"),
                    "reasoning": parsed_response.get("reasoning"),
                    "ground_truth": problem.get("ground_truth"),
                    "task_type": task_type
                }
            except json.JSONDecodeError:
                log.error(f"Failed to parse JSON response for problem {problem.get('id')}")
                return {
                    "id": problem.get("id"),
                    "prediction": None,
                    "reasoning": "JSON Parse Error",
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

