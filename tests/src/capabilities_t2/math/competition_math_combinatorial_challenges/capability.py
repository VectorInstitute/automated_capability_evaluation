class Capability:
    @staticmethod
    def repr_tasks() -> dict[str, dict]:
        return {
            "1": {
                "problem": "In how many ways can you arrange the letters of the word 'COMBINATORICS' such that no two 'C's are adjacent?",
                "answer": "86400",
            },
            "2": {
                "problem": "A committee of 5 people is to be formed from a group of 10 people. If two specific people refuse to work together, how many different committees can be formed?",
                "answer": "126",
            },
            "3": {
                "problem": "How many ways can you distribute 10 indistinguishable balls into 4 distinguishable boxes such that each box contains at least 1 ball?",
                "answer": "210",
            },
        }

    @staticmethod
    def get_instructions(t: dict) -> str:
        return f"""Solve the following combinatorial problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.\n\nProblem: {t["problem"]}\n\nRemember to put your answer on its own line at the end in the form "ANSWER:$ANSWER" (without quotes) where $ANSWER is the answer to the problem."""

    @staticmethod
    def score(t: dict, submission: str) -> float | None:
        from .utils import evaluate_with_llm_judge, parse_submission

        answer = parse_submission(submission)
        correct = evaluate_with_llm_judge(answer, t["answer"])
        return 1.0 if correct else 0.0
