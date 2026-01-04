"""
causalGEM: Bivariate causal discovery via the Generative Exposure Model.

This package implements the Generative Exposure Model (GEM) framework for
determining causal direction between pairs of continuous random variables.
The core statistic C_{X→Y} = H(X) - H(Y) exploits entropy asymmetries that
arise under causal relationships.

Main Features
-------------
- Entropy difference estimation with cross-fitting for debiased inference
- Confidence intervals and hypothesis testing for causal direction
- Diagnostic tools for checking GEM model assumptions
- Simulation utilities for validation studies

The Generative Exposure Model
-----------------------------
In the GEM framework, we consider Y = g(X) + ε where:
- X is the cause (exposure)
- g is a smooth generative function
- ε is independent noise

The key insight is that H(Y) - H(X) = E[log|g'(X)|], which creates
detectable asymmetries for non-linear generative functions.

Quick Start
-----------
>>> from causalgem import estimate_causal_direction
>>> import numpy as np
>>> 
>>> # Generate data with X → Y
>>> np.random.seed(42)
>>> x = np.random.uniform(0, 1, 500)
>>> y = x**2 + np.random.normal(0, 0.1, 500)
>>> 
>>> # Estimate causal direction
>>> result = estimate_causal_direction(x, y)
>>> print(result)

References
----------
.. [1] Your paper citation here

License
-------
MIT License
"""

__version__ = "0.1.0"
__author__ = "Your Name"

# Core estimation functions
from causalgem.estimator import (
    CausalGEMResult,
    estimate_causal_direction,
    estimate_entropy_difference,
    estimate_with_decision,
    estimate_stratified,
)

# Diagnostic tools
from causalgem.diagnostics import (
    OrthogonalityResult,
    DynamicsResult,
    check_orthogonality,
    analyze_dynamics,
    run_diagnostics,
)

# Utility functions
from causalgem.utils import (
    affine_transform,
    rank_transform,
    standardize,
    preprocess_pair,
    validate_inputs,
)

# Simulation tools
from causalgem.simulation import (
    generate_gem_data,
    generate_causal_pair,
    generate_copula_data,
    GENERATIVE_FUNCTIONS,
)

__all__ = [
    # Version info
    "__version__",
    "__author__",
    # Core
    "CausalGEMResult",
    "estimate_causal_direction",
    "estimate_entropy_difference",
    "estimate_with_decision",
    "estimate_stratified",
    # Diagnostics
    "OrthogonalityResult",
    "DynamicsResult",
    "check_orthogonality",
    "analyze_dynamics",
    "run_diagnostics",
    # Utils
    "affine_transform",
    "rank_transform",
    "standardize",
    "preprocess_pair",
    "validate_inputs",
    # Simulation
    "generate_gem_data",
    "generate_causal_pair",
    "generate_copula_data",
    "GENERATIVE_FUNCTIONS",
]
