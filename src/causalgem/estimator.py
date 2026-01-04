"""
Core causal discovery estimators based on the Generative Exposure Model (GEM).

This module implements the differential entropy statistic C_{X→Y} = H(X) - H(Y)
for bivariate causal discovery under the GEM framework.

The Generative Exposure Model assumes Y = g(X) + ε where g is a smooth
generative function and ε is independent noise. Under this model:
    H(Y) - H(X) = E[log|g'(X)|]
which creates detectable asymmetries for causal inference.
"""

from dataclasses import dataclass
from typing import Literal, Optional, Tuple, Union
import numpy as np
from numpy.typing import ArrayLike
import scipy.stats as st
from fastkde import fastKDE

__all__ = [
    "CausalGEMResult",
    "estimate_causal_direction",
    "estimate_entropy_difference",
    "estimate_with_decision",
    "estimate_stratified",
]


@dataclass
class CausalGEMResult:
    """Results from GEM-based causal discovery.
    
    Attributes
    ----------
    delta : float
        Point estimate of C_{X→Y} = H(X) - H(Y).
    ci_lower : float
        Lower bound of (1-alpha) confidence interval.
    ci_upper : float
        Upper bound of (1-alpha) confidence interval.
    std_error : float
        Standard error of the estimate.
    h_x : float
        Estimated entropy H(X).
    h_y : float
        Estimated entropy H(Y).
    decision : int
        Causal direction decision: 1 (X→Y), -1 (Y→X), 0 (inconclusive).
    n_samples : int
        Number of samples used.
    alpha : float
        Significance level used for inference.
    """
    delta: float
    ci_lower: float
    ci_upper: float
    std_error: float
    h_x: float
    h_y: float
    decision: int
    n_samples: int
    alpha: float = 0.05
    
    def __repr__(self) -> str:
        direction = {1: "X → Y", -1: "Y → X", 0: "Inconclusive"}[self.decision]
        return (
            f"CausalGEMResult(\n"
            f"  C_{{X→Y}} = {self.delta:.4f} [{self.ci_lower:.4f}, {self.ci_upper:.4f}],\n"
            f"  decision = '{direction}',\n"
            f"  H(X) = {self.h_x:.4f}, H(Y) = {self.h_y:.4f},\n"
            f"  n = {self.n_samples}\n"
            f")"
        )
    
    @property
    def is_significant(self) -> bool:
        """Whether the result is statistically significant."""
        return self.decision != 0
    
    @property
    def direction_str(self) -> str:
        """String representation of causal direction."""
        return {1: "X → Y", -1: "Y → X", 0: "Inconclusive"}[self.decision]


def _affine_transform(x: np.ndarray) -> np.ndarray:
    """Transform data to [0, 1] range via affine transformation."""
    x_min, x_max = np.min(x), np.max(x)
    if x_max - x_min < 1e-10:
        return np.zeros_like(x) + 0.5
    return (x - x_min) / (x_max - x_min)


def _estimate_entropy_single_split(
    estim_data: np.ndarray,
    inf_data: np.ndarray,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Estimate entropies using one split configuration.
    
    Parameters
    ----------
    estim_data : ndarray of shape (n/2, 2)
        Data used for density estimation.
    inf_data : ndarray of shape (n/2, 2)
        Data used for inference (evaluation points).
        
    Returns
    -------
    h_x : float
        Entropy estimate for X.
    h_y : float
        Entropy estimate for Y.
    log_fx : ndarray
        Log density values for X.
    log_fy : ndarray
        Log density values for Y.
    """
    margin_x = fastKDE.pdf_at_points(
        var1=estim_data[:, 0], 
        list_of_points=list(inf_data[:, 0])
    )
    margin_y = fastKDE.pdf_at_points(
        var1=estim_data[:, 1], 
        list_of_points=list(inf_data[:, 1])
    )
    
    # Filter positive densities
    valid = np.logical_and(margin_x > 0, margin_y > 0)
    margin_x = margin_x[valid]
    margin_y = margin_y[valid]
    
    log_fx = np.log(margin_x)
    log_fy = np.log(margin_y)
    
    h_x = -np.mean(log_fx)
    h_y = -np.mean(log_fy)
    
    return h_x, h_y, log_fx, log_fy


def estimate_causal_direction(
    x: ArrayLike,
    y: ArrayLike,
    alpha: float = 0.05,
    normalize: bool = True,
) -> CausalGEMResult:
    """Estimate causal direction using the GEM framework.
    
    Computes the entropy asymmetry statistic C_{X→Y} = H(X) - H(Y) and
    performs inference using cross-fitting with kernel density estimation.
    
    Parameters
    ----------
    x : array-like of shape (n,)
        First variable (potential cause/exposure).
    y : array-like of shape (n,)
        Second variable (potential effect/outcome).
    alpha : float, default=0.05
        Significance level for confidence intervals.
    normalize : bool, default=True
        Whether to apply affine transformation to [0,1].
        
    Returns
    -------
    result : CausalGEMResult
        Object containing estimates and inference results.
        
    Examples
    --------
    >>> import numpy as np
    >>> from causalgem import estimate_causal_direction
    >>> np.random.seed(42)
    >>> x = np.random.uniform(0, 1, 500)
    >>> y = x**2 + np.random.normal(0, 0.1, 500)  # X causes Y
    >>> result = estimate_causal_direction(x, y)
    >>> print(result.direction_str)
    'X → Y'
    
    Notes
    -----
    Under the Generative Exposure Model Y = g(X) + ε:
    - C > 0 suggests X → Y when g is contracting (|g'| < 1 on average)
    - C < 0 suggests X → Y when g is expanding (|g'| > 1 on average)
    
    The estimator uses cross-fitting: the sample is split in half, with
    each half used alternately for density estimation and evaluation.
    This provides a debiased estimate with valid asymptotic inference.
    
    References
    ----------
    .. [1] Your paper reference here
    """
    return estimate_entropy_difference(x, y, alpha=alpha, normalize=normalize)


def estimate_entropy_difference(
    x: ArrayLike,
    y: ArrayLike,
    alpha: float = 0.05,
    normalize: bool = True,
) -> CausalGEMResult:
    """Estimate differential entropy asymmetry C_{X→Y} = H(X) - H(Y).
    
    Uses cross-fitting (sample splitting) for debiased estimation with
    kernel density estimation via fastKDE.
    
    Parameters
    ----------
    x : array-like of shape (n,)
        First variable (potential cause).
    y : array-like of shape (n,)
        Second variable (potential effect).
    alpha : float, default=0.05
        Significance level for confidence intervals.
    normalize : bool, default=True
        Whether to apply affine transformation to [0,1].
        
    Returns
    -------
    result : CausalGEMResult
        Object containing estimates and inference results.
        
    Examples
    --------
    >>> import numpy as np
    >>> from causalgem import estimate_entropy_difference
    >>> np.random.seed(42)
    >>> x = np.random.uniform(0, 1, 500)
    >>> y = x**2 + np.random.normal(0, 0.1, 500)
    >>> result = estimate_entropy_difference(x, y)
    >>> print(result)
    """
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    
    if len(x) != len(y):
        raise ValueError(f"x and y must have same length, got {len(x)} and {len(y)}")
    
    # Ensure even sample size
    n = len(x)
    if n % 2 != 0:
        x, y = x[:-1], y[:-1]
        n -= 1
    
    # Normalize to [0, 1]
    if normalize:
        x = _affine_transform(x)
        y = _affine_transform(y)
    
    data = np.column_stack((x, y))
    estim, inf = np.split(data, 2)
    
    # First split: estim for density, inf for evaluation
    h_x1, h_y1, log_fx1, log_fy1 = _estimate_entropy_single_split(estim, inf)
    cov1 = np.cov(log_fx1, log_fy1)
    delta_var1 = cov1[0, 0] + cov1[1, 1] - 2 * cov1[0, 1]
    
    # Second split: inf for density, estim for evaluation
    h_x2, h_y2, log_fx2, log_fy2 = _estimate_entropy_single_split(inf, estim)
    cov2 = np.cov(log_fx2, log_fy2)
    delta_var2 = cov2[0, 0] + cov2[1, 1] - 2 * cov2[0, 1]
    
    # Cross-fitted estimates
    h_x = (h_x1 + h_x2) / 2
    h_y = (h_y1 + h_y2) / 2
    delta = h_x - h_y
    
    # Variance estimation
    delta_var = (delta_var1 + delta_var2) / 2
    delta_sd = np.sqrt(delta_var)
    
    n_eff = len(log_fx1)  # Effective sample size
    se = delta_sd / np.sqrt(n_eff)
    
    z = st.norm.ppf(1 - alpha / 2)
    ci_lower = delta - z * se
    ci_upper = delta + z * se
    
    # Decision rule
    if ci_lower * ci_upper < 0:
        decision = 0  # Inconclusive
    else:
        decision = 1 if ci_lower > 0 else -1
    
    return CausalGEMResult(
        delta=delta,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        std_error=se,
        h_x=h_x,
        h_y=h_y,
        decision=decision,
        n_samples=n,
        alpha=alpha,
    )


def estimate_with_decision(
    x: ArrayLike,
    y: ArrayLike,
    alpha: float = 0.05,
) -> Tuple[int, float, float, float, float, float, float]:
    """Legacy interface returning tuple of results.
    
    Parameters
    ----------
    x : array-like
        First variable.
    y : array-like
        Second variable.
    alpha : float
        Significance level.
        
    Returns
    -------
    decision : int
        1 (X→Y), -1 (Y→X), or 0 (inconclusive).
    h_x : float
        Entropy of X.
    h_y : float
        Entropy of Y.
    ci_lower : float
        Lower CI bound.
    delta : float
        Point estimate.
    ci_upper : float
        Upper CI bound.
    std_error : float
        Standard error.
    """
    result = estimate_entropy_difference(x, y, alpha=alpha)
    return (
        result.decision,
        result.h_x,
        result.h_y,
        result.ci_lower,
        result.delta,
        result.ci_upper,
        result.std_error,
    )


def estimate_stratified(
    x: ArrayLike,
    y: ArrayLike,
    strata: ArrayLike,
    alpha: float = 0.05,
) -> CausalGEMResult:
    """Estimate entropy difference with stratified data.
    
    Combines estimates across strata using inverse-variance weighting.
    Useful when data has natural groupings (e.g., sex, site).
    
    Parameters
    ----------
    x : array-like
        First variable.
    y : array-like
        Second variable.
    strata : array-like
        Stratum labels for each observation.
    alpha : float
        Significance level.
        
    Returns
    -------
    result : CausalGEMResult
        Combined estimate across all strata.
        
    Examples
    --------
    >>> # Stratified by sex (0=male, 1=female)
    >>> result = estimate_stratified(x, y, strata=sex)
    """
    x = np.asarray(x)
    y = np.asarray(y)
    strata = np.asarray(strata)
    
    unique_strata = np.unique(strata)
    
    deltas, variances, ns = [], [], []
    
    for s in unique_strata:
        mask = strata == s
        if np.sum(mask) < 10:
            continue
        result = estimate_entropy_difference(x[mask], y[mask], alpha=alpha)
        deltas.append(result.delta)
        variances.append(result.std_error**2 * result.n_samples)
        ns.append(result.n_samples)
    
    if not deltas:
        raise ValueError("No strata with sufficient samples (need >= 10)")
    
    deltas = np.array(deltas)
    variances = np.array(variances)
    ns = np.array(ns)
    
    # Inverse-variance weighting
    weights = ns / ns.sum()
    delta = np.sum(weights * deltas)
    delta_var = np.sum(weights**2 * variances) / ns.sum()
    delta_se = np.sqrt(delta_var)
    
    z = st.norm.ppf(1 - alpha / 2)
    ci_lower = delta - z * delta_se
    ci_upper = delta + z * delta_se
    
    if ci_lower * ci_upper < 0:
        decision = 0
    else:
        decision = 1 if ci_lower > 0 else -1
    
    return CausalGEMResult(
        delta=delta,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        std_error=delta_se,
        h_x=np.nan,  # Not meaningful for stratified
        h_y=np.nan,
        decision=decision,
        n_samples=int(ns.sum()),
        alpha=alpha,
    )


# Alias for backward compatibility
EntropyCausalResult = CausalGEMResult
