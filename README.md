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

where:
- **X** is the cause (exposure variable)
- **g** is a smooth generative function  
- **ε** is independent noise
- **Y** is the effect (outcome variable)

The key theoretical result is that under this model:

```
H(Y) - H(X) = E[log|g'(X)|]
```

This creates detectable asymmetries in marginal entropies that can be used for causal discovery.

## Installation

```bash
pip install causalGEM
```

For development installation:
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
y = x**2 + np.random.normal(0, 0.1, 500)

# Estimate causal direction
result = estimate_causal_direction(x, y)
print(result)
```

Output:
```
CausalGEMResult(
  C_{X→Y} = 0.2847 [0.1523, 0.4172],
  decision = 'X → Y',
  H(X) = 0.0012, H(Y) = -0.2835,
  n = 500
)
```

## Core Statistic

The main statistic is **C_{X→Y} = H(X) - H(Y)**, where H(·) denotes differential entropy.

| Value of C | Interpretation |
|------------|----------------|
| C > 0 (significant) | Evidence for X → Y |
| C < 0 (significant) | Evidence for Y → X |
| C ≈ 0 (CI includes 0) | Inconclusive |

## Features

### Causal Direction Estimation

```python
from causalgem import estimate_causal_direction

result = estimate_causal_direction(x, y, alpha=0.05)

print(f"Point estimate: {result.delta:.4f}")
print(f"95% CI: [{result.ci_lower:.4f}, {result.ci_upper:.4f}]")
print(f"Direction: {result.direction_str}")
print(f"Significant: {result.is_significant}")
```

### Diagnostics for GEM Assumptions

Check model assumptions before applying the method:

```python
from causalgem import run_diagnostics

diagnostics = run_diagnostics(x, y, n_bootstrap=500)

# Check functional orthogonality assumption
print(f"Orthogonality holds: {diagnostics.orthogonality.assumption_holds}")

# Analyze function dynamics (contracting vs expanding)
print(f"Function type: {diagnostics.dynamics.conclusion}")
```

### Stratified Analysis

For data with multiple strata (e.g., by sex or study site):

```python
from causalgem import estimate_stratified

# sex: 0 = male, 1 = female
result = estimate_stratified(x, y, strata=sex)
print(result)
```

### Simulation Tools

Generate synthetic data for validation:

```python
from causalgem import generate_gem_data, generate_causal_pair

# Simple causal pair with additive noise
x, y, y_true = generate_causal_pair(
    n=500, 
    func='square',  # Y = X² + noise
    noise_sd=0.1
)

# Full GEM simulation with correlated errors
data = generate_gem_data(
    n=500,
    func='exp',
    rho=0.2,  # Correlation between X and error
    noise_sd=0.1
)
x, y = data['x'], data['y']
```

**Available generative functions:** `'linear'`, `'square'`, `'cube'`, `'sqrt'`, `'cbrt'`, `'exp'`, `'log'`, `'sin'`, `'tanh'`, `'sigmoid'`

## Method Details

### Estimation Procedure

1. **Sample Splitting**: Data is split into two halves for cross-fitting
2. **Density Estimation**: Kernel density estimation via fastKDE
3. **Entropy Calculation**: H(X) = -E[log f_X(X)]
4. **Variance Estimation**: Delta method using covariance of log-densities
5. **Inference**: Asymptotic confidence intervals and hypothesis testing

### Key Assumptions

1. **GEM Structure**: Y = g(X) + ε with independent ε
2. **Functional Orthogonality**: E[log|g'(X)|] equals its population average
3. **Smooth Densities**: Both marginal densities are sufficiently smooth
4. **Bounded Support**: Best performance when data is in bounded range

Use `run_diagnostics()` to check these assumptions.

## API Reference

### Core Functions

| Function | Description |
|----------|-------------|
| `estimate_causal_direction(x, y)` | Main estimation function |
| `estimate_entropy_difference(x, y)` | Alias for main function |
| `estimate_stratified(x, y, strata)` | Stratified estimation |
| `estimate_with_decision(x, y)` | Legacy tuple-returning interface |

### Diagnostics

| Function | Description |
|----------|-------------|
| `check_orthogonality(x, y)` | Check functional orthogonality |
| `analyze_dynamics(x, y)` | Determine expanding/contracting |
| `run_diagnostics(x, y)` | Run all diagnostic checks |

### Simulation

| Function | Description |
|----------|-------------|
| `generate_gem_data(n, func, rho, ...)` | Full GEM simulation |
| `generate_causal_pair(n, func, ...)` | Simple Y = f(X) + ε |
| `generate_copula_data(n, ...)` | Arbitrary copula data |

## Examples

### Benchmark on Tuebingen Dataset

```python
from causalgem import estimate_causal_direction
from cdt.data import load_dataset  # pip install cdt

data, labels = load_dataset('tuebingen')

correct = 0
for i in range(len(data)):
    pair = data.iloc[i]
    result = estimate_causal_direction(pair['A'], pair['B'])
    if result.decision == int(labels.iloc[i]):
        correct += 1

print(f"Accuracy: {correct/len(data):.1%}")
```

### Real Data Analysis with Diagnostics

```python
import pandas as pd
from causalgem import estimate_causal_direction, run_diagnostics

# Load your data
df = pd.read_csv('your_data.csv')
x, y = df['exposure'].values, df['outcome'].values

# Check assumptions first
diag = run_diagnostics(x, y)
print(f"Assumptions satisfied: {diag.all_passed}")
print(f"Function dynamics: {diag.dynamics.conclusion}")

# Run estimation
result = estimate_causal_direction(x, y)
print(result)
```

## Citation

If you use this package in your research, please cite:

```bibtex
@article{yourname2025gem,
  title={Your Paper Title},
  author={Your Name and Coauthors},
  journal={Journal Name},
  year={2025}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
