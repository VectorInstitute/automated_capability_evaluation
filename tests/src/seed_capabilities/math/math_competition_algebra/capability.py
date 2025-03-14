class Capability:
    @staticmethod
    def repr_samples() -> dict[str, dict]:
        return {
            "1": {
                "problem": "Let $a$ and $b$ be real numbers. The function $h(x)=ax+b$ satisfies $h(1)=5$ and $h(-1)=1$.  What is $h(6)$?",
                "answer": "15",
            },
            "2": {
                "problem": "Let \\[f(x) = \\left\\{\n\\begin{array}{cl}\n\\frac{x}{21} & \\text{ if }x\\text{ is a multiple of 3 and 7}, \\\\\n3x & \\text{ if }x\\text{ is only a multiple of 7}, \\\\\n7x & \\text{ if }x\\text{ is only a multiple of 3}, \\\\\nx+3 & \\text{ if }x\\text{ is not a multiple of 3 or 7}.\n\\end{array}\n\\right.\\]If $f^a(x)$ means the function is nested $a$ times (for example, $f^2(x)=f(f(x))$), what is the smallest value of $a$ greater than 1 that satisfies $f(2)=f^a(2)$?",
                "answer": "7",
            },
            "3": {
                "problem": "Let $A$ and $B$ be real numbers such that $\\frac{A}{x-5}+B(x+1)=\\frac{-3x^2+12x+22}{x-5}$. What is $A+B$?",
                "answer": "4",
            },
        }

    @staticmethod
    def get_instructions(t: dict) -> str:
        return f"""Solve the following algebra math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.\n\nProblem: {t["problem"]}\n\nRemember to put your answer on its own line at the end in the form "ANSWER:$ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command."""

    @staticmethod
    def score(t: dict, submission: str) -> float | None:
        return 1.0 if submission == t["answer"] else 0.0
