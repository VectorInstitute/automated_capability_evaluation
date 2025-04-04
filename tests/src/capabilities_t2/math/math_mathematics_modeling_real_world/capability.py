class Capability:
    @staticmethod
    def repr_tasks() -> dict[str, dict]:
        return {
            "1": {
                "problem": "A company wants to determine the optimal number of units to produce in order to maximize profit. The profit function is given by P(x) = -2x^2 + 40x - 150, where x is the number of units produced. How many units should the company produce to maximize profit?",
                "answer": "10",
            },
            "2": {
                "problem": "A farmer has 240 meters of fencing to create a rectangular pen. What dimensions will maximize the area of the pen?",
                "answer": "60 by 60",
            },
            "3": {
                "problem": "A car rental company charges a flat fee of $50 plus $20 per day. If a customer has a budget of $200, how many days can they rent a car?",
                "answer": "7",
            },
        }

    @staticmethod
    def get_instructions(t: dict) -> str:
        return f"""Solve the following real-world modeling problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.\n\nProblem: {t["problem"]}\n\nRemember to put your answer on its own line at the end in the form "ANSWER:$ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command."""

    @staticmethod
    def score(t: dict, submission: str) -> float | None:
        return 1.0 if submission == t["answer"] else 0.0
