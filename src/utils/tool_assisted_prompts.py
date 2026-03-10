"""Prompts for tool-assisted task solving with code execution capabilities."""

# =============================================================================
# TOOL-ASSISTED TASK SOLVING PROMPTS
# =============================================================================

TOOL_ASSISTED_SYSTEM_MESSAGE = """You are an expert problem solver with access to Python code execution capabilities. You can use the following libraries to assist in solving mathematical and financial problems:

**Available Libraries:**
- **SymPy**: For symbolic mathematics (symbolic integration, differentiation, equation solving, etc.)
- **NumPy**: For numerical computations (arrays, linear algebra, numerical methods)
- **SciPy**: For scientific computing (numerical integration, optimization, ODE solving)
- **Math/Fractions/Decimal**: Standard Python mathematical libraries
- **Financial Libraries**: numpy_financial, py_vollib, pypfopt, empyrical, arch (for finance-specific computations)

Always use the SIMPLEST tool that solves the problem correctly:
1. If basic arithmetic works → use standard Python operators
2. If you need standard math functions → use math library
3. If you need arrays/matrices → use NumPy
4. If you need symbolic manipulation → use SymPy
5. ONLY use financial libraries (numpy_financial, py_vollib, etc.) when the problem EXPLICITLY involves:
   - Time value of money calculations (NPV, IRR, annuities)
   - Option pricing models (Black-Scholes, Greeks)
   - Portfolio optimization or risk metrics
   - Bond pricing with yield curves

**When to Use Code:**
- For complex calculations that benefit from computational tools
- When the problem requires numerical precision
- For problems involving symbolic manipulation or calculus

**Response Format:**
You will respond in two stages:
1. First, generate code to solve the problem
2. After seeing the code output, format the final answer appropriately

This ensures you can properly format answers for different question types (boolean, multiple choice, or numerical)."""

# =============================================================================
# STAGE 1: CODE GENERATION PROMPT
# =============================================================================

TOOL_ASSISTED_CODE_GENERATION_PROMPT = """PROBLEM: {problem_text}

{tool_context}

**CRITICAL - SIMPLICITY PRINCIPLE:**
Use the simplest approach that correctly solves the problem. Do NOT use specialized libraries unless the problem explicitly requires them:
- For basic arithmetic (percentages, ratios, simple formulas): Use Python operators or math library
- For algebra/calculus: Use SymPy
- For numerical arrays/linear algebra: Use NumPy
- For statistical calculations: Use SciPy
- For financial derivatives pricing (options, swaps): Use py_vollib or numpy_financial ONLY if the problem explicitly involves these instruments

**Stage 1 Instructions - Generate Code:**
1. Analyze the problem and identify the simplest appropriate approach
2. Write Python code to solve the problem
3. Make sure your code prints the key results clearly

DO NOT provide final_answer or numerical_answer yet - you will do that in Stage 2 after seeing the code output.

**Available Tools:**
```python
# Standard Math (USE FIRST for basic calculations)
import math
from fractions import Fraction
from decimal import Decimal
from datetime import datetime, timedelta

# SymPy - Symbolic Mathematics (for algebra/calculus)
from sympy import symbols, Function, Eq, dsolve, diff, integrate, simplify, solve, limit, series, sqrt, exp, log, sin, cos, tan, pi, E, I, oo
from sympy import Matrix, eye, zeros, ones, det, eigenvals, eigenvects
from sympy.abc import x, y, z, t, a, b, c, n

# NumPy - Numerical Computing (for arrays/linear algebra)
import numpy as np
# Use: np.array, np.linalg.eig, np.linalg.det, np.linalg.inv, np.linalg.solve, etc.

# SciPy - Scientific Computing (for optimization/integration)
from scipy import integrate, optimize, linalg
from scipy.integrate import odeint, solve_ivp, quad, dblquad
from scipy.stats import norm
# Use: integrate.quad, optimize.fsolve, scipy.stats.norm, etc.

# Financial Computing Libraries (USE ONLY for derivative pricing problems)
import numpy_financial as npf
# Use ONLY for: Time value of money with given formulas (NPV, IRR, PMT)
# Example: npf.npv(rate, cashflows), npf.irr(cashflows)

import py_vollib.black_scholes as bs
from py_vollib.black_scholes.greeks.analytical import delta, gamma, theta, vega
# Use ONLY for: Black-Scholes option pricing (calls/puts with volatility)
# Example: bs.black_scholes('c', S, K, t, r, sigma)

from arch import arch_model
from arch.univariate import HARX
# Use ONLY for: GARCH/volatility modeling problems
# For multi-step forecasts: use method='simulation'

import statsmodels.api as sm
# Use ONLY for: Regression, time series analysis
```

IMPORTANT: Return your response as raw JSON only. Do not wrap it in markdown code blocks.

CRITICAL: Escape all backslashes in code with double backslashes (\\\\).

**Stage 1 Response Format (thought + code only):**
{{
    "thought": "Your detailed reasoning about the approach",
    "code": "Python code to solve the problem"
}}

Respond with valid JSON only."""

# =============================================================================
# STAGE 2: ANSWER FORMATTING PROMPT
# =============================================================================

TOOL_ASSISTED_ANSWER_FORMATTING_PROMPT = """Your code has been executed. Here is the output:

```
{code_output}
```

**Stage 2 Instructions - Format Final Answer:**
Based on the code output above, provide the final answer in the format appropriate for this question type:
- For **boolean questions**: Answer with Yes/No, True/False, or 1/0 as appropriate
- For **multiple choice questions**: Provide the letter (A/B/C/D) and/or the full text of the correct option
- For **numerical questions**: Provide the number with appropriate units and context

**Question:** {problem_text}

**Response Format:**
{{
    "final_answer": "Your complete answer formatted appropriately for the question type",
    "numerical_answer": "The final numerical result if applicable, otherwise null"
}}

IMPORTANT: Return raw JSON only. Do not wrap in markdown code blocks.

Respond with valid JSON only."""

# =============================================================================
# LEGACY PROMPTS (kept for backward compatibility with subsequent rounds)
# =============================================================================

TOOL_ASSISTED_ROUND_1_PROMPT_LEGACY = """PROBLEM: {problem_text}

{tool_context}

**CRITICAL - SIMPLICITY PRINCIPLE:**
Use the simplest approach that correctly solves the problem. Do NOT use specialized libraries unless the problem explicitly requires them:
- For basic arithmetic (percentages, ratios, simple formulas): Use Python operators or math library
- For algebra/calculus: Use SymPy
- For numerical arrays/linear algebra: Use NumPy
- For statistical calculations: Use SciPy
- For financial derivatives pricing (options, swaps): Use py_vollib or numpy_financial ONLY if the problem explicitly involves these instruments

**Instructions:**
1. Explain your mathematical reasoning and approach
2. Write Python code using the SIMPLEST appropriate tools
3. Use the code output to inform your final answer
4. Provide a clear final numerical answer

**Available Tools:**
```python
# Standard Math (USE FIRST for basic calculations)
import math
from fractions import Fraction
from decimal import Decimal
from datetime import datetime, timedelta

# SymPy - Symbolic Mathematics (for algebra/calculus)
from sympy import symbols, Function, Eq, dsolve, diff, integrate, simplify, solve, limit, series, sqrt, exp, log, sin, cos, tan, pi, E, I, oo
from sympy import Matrix, eye, zeros, ones, det, eigenvals, eigenvects
from sympy.abc import x, y, z, t, a, b, c, n

# NumPy - Numerical Computing (for arrays/linear algebra)
import numpy as np
# Use: np.array, np.linalg.eig, np.linalg.det, np.linalg.inv, np.linalg.solve, etc.

# SciPy - Scientific Computing (for optimization/integration)
from scipy import integrate, optimize, linalg
from scipy.integrate import odeint, solve_ivp, quad, dblquad
from scipy.stats import norm
# Use: integrate.quad, optimize.fsolve, scipy.stats.norm, etc.

# Financial Computing Libraries (USE ONLY for derivative pricing problems)
import numpy_financial as npf
# Use ONLY for: Time value of money with given formulas (NPV, IRR, PMT)
# Example: npf.npv(rate, cashflows), npf.irr(cashflows)

import py_vollib.black_scholes as bs
from py_vollib.black_scholes.greeks.analytical import delta, gamma, theta, vega
# Use ONLY for: Black-Scholes option pricing (calls/puts with volatility)
# Example: bs.black_scholes('c', S, K, t, r, sigma)

from arch import arch_model
from arch.univariate import HARX
# Use ONLY for: GARCH/volatility modeling problems
# For multi-step forecasts: use method='simulation'

import statsmodels.api as sm
# Use ONLY for: Regression, time series analysis
```

IMPORTANT: Return your response as raw JSON only. Do not wrap it in markdown code blocks or add any formatting.

CRITICAL: You MUST escape all backslashes in LaTeX expressions within JSON strings. Use double backslashes (\\\\).

Provide your solution in JSON format:
{{
    "thought": "Your detailed mathematical reasoning. Identify which tool is most appropriate and why.",
    "code": "Python code using simplest appropriate tools",
    "code_output": null,
    "final_answer": "Your complete answer",
    "numerical_answer": "The final numerical result (if applicable, otherwise null)"
}}

Respond with valid JSON only."""

TOOL_ASSISTED_SUBSEQUENT_ROUNDS_PROMPT = """These are the reasoning and solutions to the problem from other agents:

{other_solutions}

Using the solutions from other agents as additional information, can you provide your answer to the problem?

The original problem is: {problem_text}

**CRITICAL - SIMPLICITY PRINCIPLE:**
Use the simplest approach that correctly solves the problem. Review other agents' approaches - if they used complex libraries unnecessarily, prefer the simpler solution.

**Instructions:**
1. Classify the problem type and identify the simplest correct approach
2. Review other agents' solutions - identify which tools they used and whether they were appropriate
3. Provide your solution using the simplest effective approach
4. Your answer may agree with, disagree with, or synthesize the other solutions

**Available Tools:**
```python
# Standard Math (USE FIRST for basic calculations)
import math
from fractions import Fraction
from decimal import Decimal

# SymPy (for algebra/calculus)
from sympy import symbols, solve, simplify, diff, integrate

# NumPy (for arrays/linear algebra)
import numpy as np

# SciPy (for optimization/integration)
from scipy import optimize, integrate, stats

# Financial Libraries (USE ONLY for derivative pricing)
import numpy_financial as npf
import py_vollib.black_scholes as bs
from arch import arch_model
```

IMPORTANT: Return your response as raw JSON only. Use the simplicity principle - prefer basic Python over specialized libraries unless truly needed.

CRITICAL: Escape all backslashes in LaTeX with double backslashes (\\\\).

Provide your solution in JSON format:
{{
    "thought": "Your detailed reasoning. Review what other agents tried and choose the simplest effective approach.",
    "code": "Python code using simplest appropriate tools",
    "code_output": null,
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

# =============================================================================
# BACKWARD COMPATIBILITY ALIAS
# =============================================================================
# For existing code that imports TOOL_ASSISTED_ROUND_1_PROMPT
# This now uses the new two-stage architecture (Code Generation -> Answer Formatting)
TOOL_ASSISTED_ROUND_1_PROMPT = TOOL_ASSISTED_CODE_GENERATION_PROMPT
