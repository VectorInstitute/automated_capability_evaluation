"""Tool definitions for scientific computing.

Simple dataclasses without over-abstraction. Easy to read and modify.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class LibraryConfig:
    """Configuration for a library/module."""
    name: str
    import_name: str
    description: str
    docs_path: Optional[str] = None
    common_functions: List[str] = field(default_factory=list)
    use_cases: List[str] = field(default_factory=list)


@dataclass
class ToolDefinition:
    """Complete definition of a tool."""
    tool_id: str
    name: str
    description: str
    libraries: List[LibraryConfig]
    allowed_imports: List[str]
    use_cases: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# Library Configurations
NUMPY_CONFIG = LibraryConfig(
    name="numpy",
    import_name="numpy as np",
    description="Fundamental package for numerical computing. Arrays, matrices, and mathematical functions.",
    docs_path="numpy-html-1.17.0",
    common_functions=[
        "np.array", "np.linspace", "np.zeros", "np.ones", "np.eye",
        "np.linalg.eig", "np.linalg.det", "np.linalg.inv", "np.linalg.solve",
        "np.dot", "np.cross", "np.sin", "np.cos", "np.exp", "np.log", "np.sqrt"
    ],
    use_cases=[
        "Array operations and linear algebra",
        "Matrix computations",
        "Statistical computations"
    ]
)

SCIPY_CONFIG = LibraryConfig(
    name="scipy",
    import_name="scipy",
    description="Scientific computing library. Integration, optimization, linear algebra, statistics.",
    docs_path="scipy-html-1.17.0",
    common_functions=[
        "scipy.integrate.quad", "scipy.integrate.odeint", "scipy.integrate.solve_ivp",
        "scipy.optimize.minimize", "scipy.optimize.fsolve",
        "scipy.linalg.lu", "scipy.linalg.qr", "scipy.linalg.svd",
        "scipy.stats.norm", "scipy.interpolate.interp1d"
    ],
    use_cases=[
        "Numerical integration",
        "Optimization and root finding",
        "Advanced linear algebra"
    ]
)

SYMPY_CONFIG = LibraryConfig(
    name="sympy",
    import_name="sympy",
    description="Symbolic mathematics library. Algebraic manipulation, calculus, equation solving.",
    docs_path="sympy-docs-html-1.14.0",
    common_functions=[
        "sympy.symbols", "sympy.simplify", "sympy.expand", "sympy.factor",
        "sympy.diff", "sympy.integrate", "sympy.solve", "sympy.dsolve",
        "sympy.Matrix", "sympy.sin", "sympy.cos", "sympy.pi"
    ],
    use_cases=[
        "Symbolic differentiation and integration",
        "Solving equations",
        "Symbolic simplification"
    ]
)

MATH_CONFIG = LibraryConfig(
    name="math",
    import_name="math",
    description="Python's standard math library.",
    docs_path=None,
    common_functions=["math.sin", "math.cos", "math.sqrt", "math.exp", "math.log", "math.pi"],
    use_cases=["Basic mathematical operations"]
)

FRACTIONS_CONFIG = LibraryConfig(
    name="fractions",
    import_name="fractions",
    description="Rational number arithmetic.",
    docs_path=None,
    common_functions=["Fraction"],
    use_cases=["Exact rational arithmetic"]
)

DECIMAL_CONFIG = LibraryConfig(
    name="decimal",
    import_name="decimal",
    description="High-precision decimal arithmetic.",
    docs_path=None,
    common_functions=["Decimal"],
    use_cases=["High-precision calculations"]
)

CMATH_CONFIG = LibraryConfig(
    name="cmath",
    import_name="cmath",
    description="Complex number operations.",
    docs_path=None,
    common_functions=["cmath.sqrt", "cmath.exp", "cmath.log"],
    use_cases=["Complex number operations"]
)

# The Tool Definition
PYTHON_SCIENTIFIC_TOOL = ToolDefinition(
    tool_id="python_code_execution",
    name="Python Code Execution",
    description="Execute Python code with SymPy, NumPy, SciPy for symbolic math, numerical computing, and scientific analysis.",
    libraries=[SYMPY_CONFIG, NUMPY_CONFIG, SCIPY_CONFIG, MATH_CONFIG, FRACTIONS_CONFIG, DECIMAL_CONFIG, CMATH_CONFIG],
    allowed_imports=["sympy", "numpy", "scipy", "math", "fractions", "decimal", "cmath"],
    use_cases=[
        "Solving differential equations",
        "Matrix operations and linear algebra",
        "Numerical integration",
        "Symbolic manipulation",
        "Optimization problems"
    ],
    metadata={"timeout": 30}
)
