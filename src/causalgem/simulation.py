"""
Data generation functions for simulation studies.

Provides utilities for generating synthetic bivariate data with known
causal structures for validating the entropy-based causal discovery method.
"""

from typing import Callable, Dict, Optional, Tuple, Union
import numpy as np
from scipy.stats import rankdata, norm

__all__ = [
    "generate_causal_pair",
    "generate_gem_data",
    "generate_copula_data",
    "GENERATIVE_FUNCTIONS",
]


# Pre-defined generative functions
GENERATIVE_FUNCTIONS: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "linear": lambda x: x,
    "square": lambda x: x ** 2,
    "cube": lambda x: x ** 3,
    "sqrt": lambda x: np.sqrt(x),
    "cbrt": lambda x: np.cbrt(x),
    "exp": lambda x: np.exp(x),
    "log": lambda x: np.log(x + 0.01),  # Shift to avoid log(0)
    "sin": lambda x: np.sin(np.pi * x / 2),
    "tanh": lambda x: np.tanh(x),
    "sigmoid": lambda x: 1 / (1 + np.exp(-5 * (x - 0.5))),
}


def _affine_to_unit(x: np.ndarray) -> np.ndarray:
    """Transform to [0, 1] range."""
    x_min, x_max = np.min(x), np.max(x)
    if x_max - x_min < 1e-10:
        return np.zeros_like(x) + 0.5
    return (x - x_min) / (x_max - x_min)


def generate_causal_pair(
    n: int,
    func: Union[str, Callable[[np.ndarray], np.ndarray]] = "square",
    noise_sd: float = 0.1,
    x_dist: str = "uniform",
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a causal pair X → Y with Y = f(X) + ε.
    
    Parameters
    ----------
    n : int
        Number of samples to generate.
    func : str or callable, default='square'
        Generative function. If str, uses predefined function.
        Available: 'linear', 'square', 'cube', 'sqrt', 'cbrt', 
        'exp', 'log', 'sin', 'tanh', 'sigmoid'.
    noise_sd : float, default=0.1
        Standard deviation of additive Gaussian noise.
    x_dist : str, default='uniform'
        Distribution of X: 'uniform', 'normal', or 'beta'.
    seed : int, optional
        Random seed for reproducibility.
        
    Returns
    -------
    x : ndarray of shape (n,)
        Cause variable (in [0, 1]).
    y : ndarray of shape (n,)
        Effect variable (in [0, 1] before noise).
    y_true : ndarray of shape (n,)
        Effect without noise.
        
    Examples
    --------
    >>> x, y, y_true = generate_causal_pair(500, func='square', noise_sd=0.1)
    >>> x.shape, y.shape
    ((500,), (500,))
    
    >>> # Using custom function
    >>> x, y, _ = generate_causal_pair(500, func=lambda t: t**0.5 * np.sin(t))
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate X
    if x_dist == "uniform":
        x = np.random.uniform(0, 1, n)
    elif x_dist == "normal":
        x = norm.cdf(np.random.randn(n))  # Transform to [0, 1]
    elif x_dist == "beta":
        x = np.random.beta(2, 5, n)
    else:
        raise ValueError(f"Unknown x_dist: {x_dist}")
    
    # Get generative function
    if isinstance(func, str):
        if func not in GENERATIVE_FUNCTIONS:
            raise ValueError(f"Unknown func: {func}. Available: {list(GENERATIVE_FUNCTIONS.keys())}")
        g = GENERATIVE_FUNCTIONS[func]
    else:
        g = func
    
    # Generate Y = g(X)
    y_true = g(x)
    y_true = _affine_to_unit(y_true)
    
    # Add noise
    noise = np.random.normal(0, noise_sd, n)
    y = y_true + noise
    
    return x, y, y_true


def generate_gem_data(
    n: int,
    func: Union[str, Callable[[np.ndarray], np.ndarray]] = "square",
    rho: float = 0.0,
    noise_sd: float = 0.1,
    seed: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """Generate data from Generative Error Model (GEM).
    
    Generates X and ε with specified correlation (via Gaussian copula),
    then computes Y = f(X) + ε.
    
    Parameters
    ----------
    n : int
        Number of samples.
    func : str or callable, default='square'
        Generative function.
    rho : float, default=0.0
        Correlation between X and ε. rho=0 means independent errors.
    noise_sd : float, default=0.1
        Noise standard deviation.
    seed : int, optional
        Random seed.
        
    Returns
    -------
    dict
        Dictionary with keys: 'x', 'y', 'y_true', 'epsilon'.
        
    Notes
    -----
    This implements the simulation scheme from the paper where
    X and ε can be correlated (violating the standard ANM assumption).
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate correlated (X, ε) via Gaussian copula
    cov = [[1, rho], [rho, 1]]
    z = np.random.multivariate_normal([0, 0], cov, n)
    x_normal, e_normal = z[:, 0], z[:, 1]
    
    # Transform to uniform via rank
    x_uniform = rankdata(x_normal) / (n + 1)
    e_uniform = rankdata(e_normal) / (n + 1)
    
    # Transform epsilon to N(0, noise_sd^2)
    epsilon = norm.ppf(e_uniform) * noise_sd
    
    # Get generative function
    if isinstance(func, str):
        g = GENERATIVE_FUNCTIONS[func]
    else:
        g = func
    
    # Generate Y
    x = x_uniform.copy()
    y_true = g(x)
    y_true = _affine_to_unit(y_true)
    y = y_true + epsilon
    
    return {
        "x": x,
        "y": y,
        "y_true": y_true,
        "epsilon": epsilon,
    }


def generate_copula_data(
    n: int,
    x_params: Tuple[str, Tuple] = ("normal", (0, 1)),
    y_params: Tuple[str, Tuple] = ("normal", (0, 1)),
    rho: float = 0.25,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate bivariate data with Gaussian copula.
    
    Creates dependent data where dependence is specified via
    Gaussian copula but marginals can be arbitrary.
    
    Parameters
    ----------
    n : int
        Number of samples.
    x_params : tuple, default=('normal', (0, 1))
        Distribution and parameters for X.
        Format: (dist_name, (param1, param2, ...))
        Supported: 'normal', 'exponential', 'lognormal', 'uniform'.
    y_params : tuple, default=('normal', (0, 1))
        Distribution and parameters for Y.
    rho : float, default=0.25
        Copula correlation parameter.
    seed : int, optional
        Random seed.
        
    Returns
    -------
    x : ndarray of shape (n,)
        First variable.
    y : ndarray of shape (n,)
        Second variable.
        
    Examples
    --------
    >>> # Normal X, Exponential Y with correlation 0.5
    >>> x, y = generate_copula_data(
    ...     500,
    ...     x_params=('normal', (0, 1)),
    ...     y_params=('exponential', (1.0,)),
    ...     rho=0.5
    ... )
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate correlated uniforms via Gaussian copula
    cov = [[1, rho], [rho, 1]]
    z = np.random.multivariate_normal([0, 0], cov, n)
    u1 = norm.cdf(z[:, 0])  # Uniform marginal
    u2 = norm.cdf(z[:, 1])  # Uniform marginal
    
    # Transform to target marginals
    def transform_marginal(u, dist_name, params):
        if dist_name == "normal":
            mu, sigma = params
            return norm.ppf(u) * sigma + mu
        elif dist_name == "exponential":
            rate = params[0]
            from scipy.stats import expon
            return expon.ppf(u, scale=1/rate)
        elif dist_name == "lognormal":
            mu, sigma = params
            from scipy.stats import lognorm
            return lognorm.ppf(u, s=sigma, scale=np.exp(mu))
        elif dist_name == "uniform":
            a, b = params
            return a + u * (b - a)
        else:
            raise ValueError(f"Unknown distribution: {dist_name}")
    
    x = transform_marginal(u1, x_params[0], x_params[1])
    y = transform_marginal(u2, y_params[0], y_params[1])
    
    return x, y


def generate_benchmark_pair(
    pair_type: str,
    n: int = 500,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Generate standard benchmark pairs used in causal discovery literature.
    
    Parameters
    ----------
    pair_type : str
        Type of benchmark pair:
        - 'anm_square': Y = X^2 + ε
        - 'anm_cube': Y = X^3 + ε  
        - 'anm_exp': Y = exp(X) + ε
        - 'linear': Y = X + ε
        - 'nonlinear_mixed': More complex nonlinear relationship
    n : int, default=500
        Sample size.
    seed : int, optional
        Random seed.
        
    Returns
    -------
    x : ndarray
        Cause variable.
    y : ndarray
        Effect variable.
    true_direction : int
        1 if X→Y, -1 if Y→X.
    """
    if seed is not None:
        np.random.seed(seed)
    
    x = np.random.uniform(0, 1, n)
    
    if pair_type == "anm_square":
        y = x**2 + np.random.normal(0, 0.1, n)
        true_direction = 1
    elif pair_type == "anm_cube":
        y = x**3 + np.random.normal(0, 0.1, n)
        true_direction = 1
    elif pair_type == "anm_exp":
        y = np.exp(x) + np.random.normal(0, 0.1, n)
        true_direction = 1
    elif pair_type == "linear":
        y = x + np.random.normal(0, 0.2, n)
        true_direction = 0  # Linear is symmetric
    elif pair_type == "nonlinear_mixed":
        y = np.sin(2 * np.pi * x) + 0.5 * x**2 + np.random.normal(0, 0.1, n)
        true_direction = 1
    else:
        raise ValueError(f"Unknown pair_type: {pair_type}")
    
    return x, y, true_direction
