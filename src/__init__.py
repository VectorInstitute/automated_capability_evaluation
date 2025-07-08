"""
The __init__.py file for the src module in the automatic_benchmark_generation project.

It initializes the src module, making it easier to import and use the utilities
provided by this module in other parts of the project.
"""

# Import and re-export functions that are commonly used across the package
from .generate_capabilities import (
    generate_capabilities,
    generate_capabilities_using_llm,
    generate_capability_areas,
)
from .utils.capability_management_utils import (
    get_previous_capabilities,
)


# Export the functions so they can be imported from the package level
__all__ = [
    "generate_capability_areas",
    "generate_capabilities",
    "generate_capabilities_using_llm",
    "get_previous_capabilities",
]
