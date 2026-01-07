"""
causalGEM: Bivariate causal discovery via the Generative Exposure Model.

This package implements the Generative Exposure Model (GEM) framework for
determining causal direction between pairs of continuous random variables.

The Generative Exposure Model
-----------------------------
In the GEM framework, we consider Y = g(X) + ε where:

- **X** is the cause (exposure)
- **g** is a smooth generative function
- **ε** is independent noise

The key insight is that H(Y) - H(X) = E[log|g'(X)|], which creates
detectable asymmetries for non-linear generative functions.

Decision Logic
--------------
Causal direction is determined by combining:

1. **Orthogonality check**: Validates that E[log|g'(X)|] ≈ ∫log|g'(x)|dx
2. **Dynamics analysis**: Determines if g is contracting or expanding
3. **Sign of C = H(X) - H(Y)**: The entropy asymmetry statistic

Decision rules (when orthogonality holds):

- Contracting dynamics (|g'| < 1) + C > 0 → **X → Y**
- Expanding dynamics (|g'| > 1) + C < 0 → **X → Y**
- Contracting dynamics + C < 0 → **Y → X**
- Expanding dynamics + C > 0 → **Y → X**

Quick Start
-----------
>>> import numpy as np
>>> from causalgem import estimate_causal_direction
>>> 
>>> # Generate data: Y = X² + noise (contracting function on [0,1])
>>> np.random.seed(42)
>>> x = np.random.uniform(0, 1, 500)
>>> y = x**2 + np.random.normal(0, 0.1, 500)
>>> 
>>> # Estimate causal direction
>>> result = estimate_causal_direction(x, y)
>>> print(result)
>>> print(f"Direction: {result.direction_str}")
>>> print(f"Reason: {result.decision_reason}")

For quick estimation without assumption checks (not recommended for 
final analysis):

>>> from causalgem import estimate_entropy_difference
>>> result = estimate_entropy_difference(x, y)

Main Functions
--------------
- `estimate_causal_direction`: Full GEM inference with assumption checks
- `estimate_entropy_difference`: Entropy difference only (no diagnostics)
- `estimate_stratified`: For stratified/grouped data
- `run_diagnostics`: Run all diagnostic checks separately
- `generate_causal_pair`: Generate synthetic data for testing

License
-------
MIT License

Authors
-------
Soumik Purkayastha (soumik@pitt.edu)
Peter X.-K. Song (pxsong@umich.edu)
"""

__version__ = "0.1.0"
__author__ = "Soumik Purkayastha, Peter X.-K. Song"

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
    DiagnosticSummary,
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
    generate_benchmark_pair,
    GENERATIVE_FUNCTIONS,
)

__all__ = [
    # Version info
    "__version__",
    "__author__",
    # Core estimation
    "CausalGEMResult",
    "estimate_causal_direction",
    "estimate_entropy_difference",
    "estimate_with_decision",
    "estimate_stratified",
    # Diagnostics
    "OrthogonalityResult",
    "DynamicsResult",
    "DiagnosticSummary",
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
    "generate_benchmark_pair",
    "GENERATIVE_FUNCTIONS",
]
