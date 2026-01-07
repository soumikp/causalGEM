# causalGEM

[![PyPI version](https://badge.fury.io/py/causalGEM.svg)](https://pypi.org/project/causalGEM/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**Bivariate causal discovery via the Generative Exposure Model (GEM).**

This package implements the GEM framework for determining causal direction between pairs of continuous random variables. The method exploits entropy asymmetries that arise under causal relationships of the form Y = g(X) + ε.

## The Generative Exposure Model

In the GEM framework, we model causal relationships as:

```
Y = g(X) + ε
```

where **X** is the cause (exposure), **g** is a smooth generative function, **ε** is independent noise, and **Y** is the effect (outcome).

The key theoretical result is that under this model:

```
H(Y) - H(X) = E[log|g'(X)|]
```

This creates detectable asymmetries in marginal entropies that can be used for causal discovery.

## Installation

```bash
git clone https://github.com/soumikp/causalGEM.git
cd causalGEM
pip install -e ".[dev]"
```

## Quick Start

```python
import numpy as np
from causalgem import estimate_causal_direction

# Generate data with causal relationship X → Y
np.random.seed(42)
x = np.random.uniform(0, 1, 500)
y = x**2 + np.random.normal(0, 0.1, 500)  # Contracting function

# Estimate causal direction (runs full diagnostics)
result = estimate_causal_direction(x, y)
print(result)
```

Output:
```
CausalGEMResult(
  C_{X→Y} = 0.2847 [0.1523, 0.4172],
  decision = 'X → Y',
  orthogonality = True,
  dynamics = 'contracting',
  reason = 'Contracting dynamics + positive C',
  H(X) = 0.0012, H(Y) = -0.2835,
  n = 500
)
```

## Decision Logic

The GEM framework determines causal direction through a three-step process:

### Step 1: Check Orthogonality Assumption

The method requires that E[log|g'(X)|] ≈ ∫log|g'(x)|dx (functional orthogonality). If this assumption is violated, the result is **inconclusive**.

### Step 2: Analyze Function Dynamics

Determine whether the generative function g is:
- **Contracting**: |g'(x)| < 1 on average → compresses variance
- **Expanding**: |g'(x)| > 1 on average → amplifies variance

### Step 3: Combine with Entropy Statistic

The core statistic is **C = H(X) - H(Y)**:

| Dynamics | Sign of C | Interpretation |
|----------|-----------|----------------|
| Contracting | C > 0 (significant) | **X → Y** |
| Expanding | C < 0 (significant) | **X → Y** |
| Contracting | C < 0 (significant) | **Y → X** |
| Expanding | C > 0 (significant) | **Y → X** |
| Any | CI includes 0 | Inconclusive |
| — | Orthogonality violated | Inconclusive |

### Why This Logic?

Under Y = g(X) + ε:
- If g is **contracting** (|g'| < 1), then log|g'| < 0, so E[log|g'|] < 0, meaning H(Y) < H(X), thus **C > 0**
- If g is **expanding** (|g'| > 1), then log|g'| > 0, so E[log|g'|] > 0, meaning H(Y) > H(X), thus **C < 0**

When the observed sign of C matches the dynamics, we have evidence for X → Y. When they conflict, the data suggests the reverse direction Y → X.

## Features

### Full Causal Direction Estimation

```python
from causalgem import estimate_causal_direction

result = estimate_causal_direction(x, y, alpha=0.05)

print(f"Direction: {result.direction_str}")
print(f"Reason: {result.decision_reason}")
print(f"Orthogonality holds: {result.orthogonality_holds}")
print(f"Dynamics: {result.dynamics}")
print(f"Point estimate: {result.delta:.4f}")
print(f"95% CI: [{result.ci_lower:.4f}, {result.ci_upper:.4f}]")
```

### Quick Entropy Difference (No Diagnostics)

For exploratory analysis without full diagnostic checks:

```python
from causalgem import estimate_entropy_difference

# Faster but doesn't verify assumptions
result = estimate_entropy_difference(x, y)
```

### Run Diagnostics Separately

```python
from causalgem import run_diagnostics

diagnostics = run_diagnostics(x, y, n_bootstrap=500)

print(f"Orthogonality holds: {diagnostics.orthogonality.assumption_holds}")
print(f"Deviation score: {diagnostics.orthogonality.deviation_score:.4f}")
print(f"Function dynamics: {diagnostics.dynamics.conclusion}")
print(f"Avg log gradient: {diagnostics.dynamics.avg_log_gradient:.4f}")
```

### Stratified Analysis

For data with natural groupings (e.g., by sex or study site):

```python
from causalgem import estimate_stratified

# strata: 0 = male, 1 = female
result = estimate_stratified(x, y, strata=sex)
print(result)
```

### Simulation Tools

Generate synthetic data for validation:

```python
from causalgem import generate_causal_pair, generate_gem_data

# Simple causal pair: Y = X² + noise
x, y, y_true = generate_causal_pair(
    n=500, 
    func='square',
    noise_sd=0.1
)

# Full GEM simulation with correlated errors
data = generate_gem_data(
    n=500,
    func='exp',
    rho=0.2,  # Correlation between X and error
    noise_sd=0.1
)
```

**Available generative functions:** `'linear'`, `'square'`, `'cube'`, `'sqrt'`, `'cbrt'`, `'exp'`, `'log'`, `'sin'`, `'tanh'`, `'sigmoid'`

## Method Details

### Estimation Procedure

1. **Normalization**: Data transformed to [0, 1] via affine transformation
2. **Sample Splitting**: Cross-fitting for debiased estimation
3. **Density Estimation**: Kernel density estimation via fastKDE
4. **Entropy Calculation**: H(X) = -E[log f_X(X)]
5. **Variance Estimation**: Delta method using covariance of log-densities
6. **Diagnostics**: Bootstrap-based orthogonality and dynamics checks
7. **Decision**: Combine diagnostics with entropy statistic sign

### Key Assumptions

1. **GEM Structure**: Y = g(X) + ε with independent ε
2. **Functional Orthogonality**: E[log|g'(X)|] equals its population average
3. **Smooth Densities**: Both marginal densities are sufficiently smooth
4. **Bounded Support**: Best performance when data is in bounded range

Use `run_diagnostics()` to verify assumptions before interpreting results.

## API Reference

### Core Functions

| Function | Description |
|----------|-------------|
| `estimate_causal_direction(x, y)` | Full estimation with diagnostics |
| `estimate_entropy_difference(x, y)` | Entropy difference only |
| `estimate_stratified(x, y, strata)` | Stratified estimation |

### Diagnostics

| Function | Description |
|----------|-------------|
| `check_orthogonality(x, y)` | Check functional orthogonality |
| `analyze_dynamics(x, y)` | Determine contracting/expanding |
| `run_diagnostics(x, y)` | Run all diagnostic checks |

### Simulation

| Function | Description |
|----------|-------------|
| `generate_gem_data(n, func, rho, ...)` | Full GEM simulation |
| `generate_causal_pair(n, func, ...)` | Simple Y = f(X) + ε |
| `generate_benchmark_pair(pair_type, n)` | Standard benchmark pairs |

## Examples

### Understanding Contracting vs Expanding

```python
import numpy as np
from causalgem import estimate_causal_direction, generate_causal_pair

# Contracting function: sqrt (derivative < 1 for x > 0.25)
x, y, _ = generate_causal_pair(500, func='sqrt', noise_sd=0.05, seed=42)
result = estimate_causal_direction(x, y)
print(f"sqrt: {result.dynamics}, C={result.delta:.3f} → {result.direction_str}")
# Expected: contracting, C > 0 → X → Y

# Expanding function: square (derivative > 1 for x > 0.5)
x, y, _ = generate_causal_pair(500, func='square', noise_sd=0.05, seed=42)
result = estimate_causal_direction(x, y)
print(f"square: {result.dynamics}, C={result.delta:.3f} → {result.direction_str}")
# Note: x² on [0,1] is actually contracting (derivative 2x < 2)
```

### Real Data Analysis Workflow

```python
import pandas as pd
from causalgem import estimate_causal_direction, run_diagnostics

# Load data
df = pd.read_csv('your_data.csv')
x, y = df['exposure'].values, df['outcome'].values

# Step 1: Check assumptions
diag = run_diagnostics(x, y, n_bootstrap=500)
print(f"Orthogonality: {diag.orthogonality.assumption_holds}")
print(f"Dynamics: {diag.dynamics.conclusion}")

# Step 2: Run estimation (includes diagnostics)
result = estimate_causal_direction(x, y)
print(result)

# Interpret
if result.decision == 0:
    print(f"Cannot determine direction: {result.decision_reason}")
else:
    print(f"Evidence for {result.direction_str}: {result.decision_reason}")
```

## Citation

If you use this package in your research, please cite:

```bibtex
@article{purkayastha2025gem,
  title={Quantification and cross-fitting inference of asymmetric relations under generative exposure mapping models},
  author={Purkayastha, Soumik and Song, Peter X-K},
  journal={Statistica Sinica},
  year={2025},
  publisher={the International Chinese Statistical Association}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
