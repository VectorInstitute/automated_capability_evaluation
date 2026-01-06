"""Prompts for tool-assisted task solving with code execution capabilities."""

# =============================================================================
# TOOL-ASSISTED TASK SOLVING PROMPTS
# =============================================================================

TOOL_ASSISTED_SYSTEM_MESSAGE = """You are an expert problem solver with access to Python code execution capabilities. You can use the following libraries to assist in solving mathematical problems:

**Available Libraries:**
- **SymPy**: For symbolic mathematics (symbolic integration, differentiation, equation solving, etc.)
- **NumPy**: For numerical computations (arrays, linear algebra, numerical methods)
- **SciPy**: For scientific computing (numerical integration, optimization, ODE solving)
- **Math/Fractions/Decimal**: Standard Python mathematical libraries

**When to Use Code:**
- To verify analytical solutions programmatically
- To perform complex symbolic manipulations
- To compute numerical results with high precision
- To check edge cases and special conditions
- To solve problems that are computationally intensive

**Best Practices:**
- Always explain your mathematical reasoning first
- Use code to verify or compute, not as a substitute for understanding
- Show intermediate steps in your code with print statements
- Verify that code outputs match your analytical reasoning"""

TOOL_ASSISTED_ROUND_1_PROMPT = """Can you solve the following problem? You have access to Python code execution with SymPy, NumPy, and SciPy.

PROBLEM: {problem_text}

**Instructions:**
1. First, explain your mathematical approach and reasoning
2. If helpful, write Python code to verify your solution or perform computations
3. Use the code output to inform or verify your final answer
4. Provide a clear final answer

**Code Execution Format:**
If you want to execute code, include it in your response. The code will be executed and the output will be provided back to you.

**Available Tools:**
```python
# SymPy - Symbolic Mathematics
from sympy import symbols, Function, Eq, dsolve, diff, integrate, simplify, solve, limit, series, sqrt, exp, log, sin, cos, tan, pi, E, I, oo
from sympy import Matrix, eye, zeros, ones, det, eigenvals, eigenvects
from sympy.abc import x, y, z, t, a, b, c, n

# NumPy - Numerical Computing
import numpy as np
# Use: np.array, np.linalg.eig, np.linalg.det, np.linalg.inv, np.linalg.solve, etc.

# SciPy - Scientific Computing
from scipy import integrate, optimize, linalg
from scipy.integrate import odeint, solve_ivp, quad, dblquad
# Use: integrate.quad, optimize.fsolve, linalg.lu, linalg.qr, linalg.svd, etc.

# Standard Math
import math
from fractions import Fraction
from decimal import Decimal
```

IMPORTANT: Return your response as raw JSON only. Do not wrap it in markdown code blocks or add any formatting. Do not include any prefixes or prose. The JSON should be directly parseable.

CRITICAL: You MUST escape all backslashes in LaTeX expressions within JSON strings. Use double backslashes (\\\\) for single backslashes (\\). This is crucial for valid JSON parsing. For example:
- Write \\\\(x^2\\\\) instead of \\(x^2\\)
- Write \\\\[equation\\\\] instead of \\[equation\\]
- Write \\\\times instead of \\times
- For any newlines within a string value, use `\\n` instead of an actual newline.

**IMPORTANT - Separating Code from Documentation:**
- Your "code" field should contain ONLY executable Python code
- DO NOT include LaTeX in Python code or comments unless properly escaped as raw strings
- Keep LaTeX explanations in "thought" and "final_answer" fields, NOT in code
- If you need to print mathematical expressions from code, use sympy's printing functions or just print the Python representation

**CRITICAL - Code Field Format:**
When providing Python code in the JSON "code" field, write it as a single-line string with \\n for line breaks.
AVOID using print statements with \\n inside strings - just use simple print statements on separate lines instead.

**CODE GENERATION BEST PRACTICES:**
1. **Use triple-quoted strings** for multi-line output: print('''output''')
2. **Separate print statements** instead of embedding \\n in strings
3. **NO LaTeX in code** - Keep LaTeX only in "thought" and "final_answer" fields
4. **Simple, focused code** - Only numerical computation, no decorative output
5. **All imports at top** - Import before using any library
6. **Test mentally** - Walk through the code logic before including it

**Example of CORRECT code:**
"code": "import numpy as np\\nA = np.array([[1, 1], [1, -1]])\\nb = np.array([2, 0])\\nQ, R = np.linalg.qr(A)\\nx = np.linalg.solve(R, Q.T @ b)\\nprint('Solution:')\\nprint(x)"

**Example of INCORRECT code:**
"code": "# Solving \\\\(Ax = b\\\\)\\nA = np.array([[1, 1], [1, -1]])"  <- BAD! LaTeX in comments causes syntax errors

**Example of BETTER output formatting:**
"code": "result = [1, 1]\\nprint('x =', result)"  <- GOOD! Simple and clear

Provide your solution in JSON format with the following structure:
{{
    "thought": "Your detailed mathematical reasoning and approach. ENSURE ALL BACKSLASHES IN LATEX ARE ESCAPED WITH DOUBLE BACKSLASHES (\\\\).",
    "code": "Python code to execute (if applicable, otherwise null). NO LATEX IN CODE!",
    "code_output": "Will be filled in after code execution (leave as null)",
    "final_answer": "Your complete answer informed by reasoning and computation",
    "numerical_answer": "The final numerical result (if applicable, otherwise null)"
}}

Example with code:
{{
    "thought": "This is a first-order ODE. I'll solve it analytically using separation of variables, then verify with SymPy.",
    "code": "from sympy import *\\nx = symbols('x')\\ny = Function('y')\\node = Eq(y(x).diff(x), x*y(x))\\nsolution = dsolve(ode, y(x))\\nprint('General solution:', solution)\\nprint('Verification:', simplify(solution.lhs - solution.rhs))",
    "code_output": null,
    "final_answer": "The solution is y = C*exp(x^2/2)",
    "numerical_answer": null
}}

Example without code:
{{
    "thought": "This is a simple algebraic equation. I can solve it directly: 2x + 3 = 11, so 2x = 8, therefore x = 4.",
    "code": null,
    "code_output": null,
    "final_answer": "x = 4",
    "numerical_answer": 4
}}

Respond with valid JSON only."""

TOOL_ASSISTED_SUBSEQUENT_ROUNDS_PROMPT = """These are the reasoning and solutions to the problem from other agents:

{other_solutions}

Using the solutions from other agents as additional information, can you provide your answer to the problem? You have access to Python code execution with SymPy, NumPy, and SciPy.

The original problem is: {problem_text}

**Instructions:**
1. Review the other agents' approaches and identify strengths/weaknesses
2. Explain your mathematical reasoning, considering the other solutions
3. If helpful, use Python code to verify or compute
4. Provide your final answer, which may agree with, disagree with, or synthesize the other solutions

**Available Tools:**
```python
# SymPy - Symbolic Mathematics
from sympy import symbols, Function, Eq, dsolve, diff, integrate, simplify, solve, limit, series, sqrt, exp, log, sin, cos, tan, pi, E, I, oo
from sympy import Matrix, eye, zeros, ones, det, eigenvals, eigenvects
from sympy.abc import x, y, z, t, a, b, c, n

# NumPy - Numerical Computing
import numpy as np

# SciPy - Scientific Computing
from scipy import integrate, optimize, linalg
from scipy.integrate import odeint, solve_ivp, quad, dblquad
```

IMPORTANT: Return your response as raw JSON only. Do not wrap it in markdown code blocks or add any formatting. Do not include any prefixes or prose. The JSON should be directly parseable.

CRITICAL: You MUST escape all backslashes in LaTeX expressions within JSON strings. Use double backslashes (\\\\) for single backslashes (\\). This is crucial for valid JSON parsing. For example:
- Write \\\\(x^2\\\\) instead of \\(x^2\\)
- Write \\\\[equation\\\\] instead of \\[equation\\]
- Write \\\\times instead of \\times
- For any newlines within a string value, use `\\n` instead of an actual newline.

**IMPORTANT - Separating Code from Documentation:**
- Your "code" field should contain ONLY executable Python code
- DO NOT include LaTeX in Python code or comments unless properly escaped as raw strings
- Keep LaTeX explanations in "thought" and "final_answer" fields, NOT in code
- If you need to print mathematical expressions from code, use sympy's printing functions

Provide your solution in JSON format:
{{
    "thought": "Your reasoning considering other agents' solutions. ENSURE ALL BACKSLASHES IN LATEX ARE ESCAPED WITH DOUBLE BACKSLASHES (\\\\).",
    "code": "Python code to execute (if applicable, otherwise null). NO LATEX IN CODE!",
    "code_output": "Will be filled in after code execution (leave as null)",
    "final_answer": "Your complete answer",
    "numerical_answer": "The final numerical result (if applicable, otherwise null)"
}}

Respond with valid JSON only."""

# =============================================================================
# CODE EXECUTION EXAMPLES FOR DIFFERENT CAPABILITIES
# =============================================================================

CODE_EXAMPLES_ODE = """
# Example: Solving a first-order ODE
from sympy import *
x = symbols('x')
y = Function('y')

# Define ODE: dy/dx = xy with y(0) = 1
ode = Eq(y(x).diff(x), x*y(x))
general_solution = dsolve(ode, y(x))
print("General solution:", general_solution)

# Apply initial condition
C1 = symbols('C1')
particular = general_solution.rhs.subs(x, 0)
C_value = solve(particular - 1, C1)[0]
print("C value:", C_value)

final_solution = general_solution.rhs.subs(C1, C_value)
print("Particular solution:", final_solution)

# Verify by substitution
dy_dx = diff(final_solution, x)
rhs = x * final_solution
print("Verification:", simplify(dy_dx - rhs) == 0)
"""

CODE_EXAMPLES_LINEAR_ALGEBRA = """
# Example: Matrix operations and eigenvalues
import numpy as np
from scipy import linalg

# Define matrix
A = np.array([[4, 2], [1, 3]])
print("Matrix A:")
print(A)

# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)
print("\\nEigenvalues:", eigenvalues)
print("Eigenvectors:")
print(eigenvectors)

# Verify: A*v = λ*v
for i in range(len(eigenvalues)):
    v = eigenvectors[:, i]
    lam = eigenvalues[i]
    print(f"\\nVerification for λ={lam}:")
    print("A*v =", A @ v)
    print("λ*v =", lam * v)
    print("Equal?", np.allclose(A @ v, lam * v))
"""

CODE_EXAMPLES_INTEGRATION = """
# Example: Symbolic and numerical integration
from sympy import *
from scipy import integrate as scipy_integrate

x = symbols('x')

# Symbolic integration
expr = sin(x) * exp(-x)
symbolic_result = integrate(expr, (x, 0, oo))
print("Symbolic result:", symbolic_result)
print("Numerical value:", float(symbolic_result))

# Numerical integration for comparison
def integrand(x):
    return np.sin(x) * np.exp(-x)

numerical_result, error = scipy_integrate.quad(integrand, 0, np.inf)
print("\\nNumerical integration:", numerical_result)
print("Error estimate:", error)
"""

CODE_EXAMPLES_DYNAMICAL_SYSTEMS = """
# Example: Numerical simulation of ODE system
import numpy as np
from scipy.integrate import odeint

# Define system: Lotka-Volterra equations
def lotka_volterra(state, t, alpha, beta, gamma, delta):
    x, y = state
    dx_dt = alpha * x - beta * x * y
    dy_dt = delta * x * y - gamma * y
    return [dx_dt, dy_dt]

# Parameters
alpha, beta, gamma, delta = 1.0, 0.1, 1.5, 0.075

# Initial conditions
x0, y0 = 10, 5
initial_state = [x0, y0]

# Time points
t = np.linspace(0, 50, 1000)

# Solve ODE
solution = odeint(lotka_volterra, initial_state, t, args=(alpha, beta, gamma, delta))

print("Initial state:", initial_state)
print("Final state:", solution[-1])
print("Max prey population:", np.max(solution[:, 0]))
print("Max predator population:", np.max(solution[:, 1]))
"""

CODE_EXAMPLES_COMPLEX_ANALYSIS = """
# Example: Complex number operations
from sympy import *

# Define complex numbers
z1 = 3 + 4*I
z2 = 1 - 2*I

print("z1 =", z1)
print("z2 =", z2)

# Operations
print("\\nz1 + z2 =", z1 + z2)
print("z1 * z2 =", expand(z1 * z2))
print("z1 / z2 =", simplify(z1 / z2))

# Magnitude and argument
print("\\n|z1| =", abs(z1))
print("arg(z1) =", arg(z1))

# Complex conjugate
print("\\nconjugate(z1) =", conjugate(z1))

# Euler's formula
theta = symbols('theta', real=True)
euler = exp(I * theta)
print("\\ne^(iθ) =", euler)
print("Expanded:", expand(euler, complex=True))
"""
