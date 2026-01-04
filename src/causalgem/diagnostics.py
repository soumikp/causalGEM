"""
Diagnostic tools for checking GEM model assumptions.

This module provides functions to verify the key assumptions underlying
the Generative Exposure Model (GEM) framework:
1. Functional orthogonality between input and gradient
2. Classification of generative function dynamics (expanding/contracting)

These diagnostics help validate whether the GEM assumptions hold for
a given dataset before applying causal discovery.
"""

from dataclasses import dataclass
from typing import Literal, Optional, Tuple
import numpy as np
from numpy.typing import ArrayLike
from scipy.interpolate import UnivariateSpline
from scipy.integrate import quad

__all__ = [
    "OrthogonalityResult",
    "DynamicsResult",
    "check_orthogonality",
    "analyze_dynamics",
    "run_diagnostics",
]


@dataclass
class OrthogonalityResult:
    """Results from functional orthogonality diagnostic.
    
    Attributes
    ----------
    lhs_estimate : float
        Left-hand side: E[log|g'(X)|].
    rhs_estimate : float
        Right-hand side: integral of log|g'| over support.
    deviation_score : float
        LHS - RHS. Values near 0 indicate assumption holds.
    bootstrap_scores : ndarray, optional
        Bootstrap distribution of deviation scores.
    ci_lower : float, optional
        Lower confidence bound.
    ci_upper : float, optional
        Upper confidence bound.
    assumption_holds : bool
        Whether the 95% CI includes 0.
    """
    lhs_estimate: float
    rhs_estimate: float
    deviation_score: float
    bootstrap_scores: Optional[np.ndarray] = None
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None
    
    @property
    def assumption_holds(self) -> bool:
        if self.ci_lower is None or self.ci_upper is None:
            return abs(self.deviation_score) < 0.5  # Heuristic
        return self.ci_lower <= 0 <= self.ci_upper
    
    def __repr__(self) -> str:
        status = "✓ holds" if self.assumption_holds else "✗ violated"
        ci_str = ""
        if self.ci_lower is not None:
            ci_str = f", 95% CI: [{self.ci_lower:.4f}, {self.ci_upper:.4f}]"
        return (
            f"OrthogonalityResult(\n"
            f"  deviation={self.deviation_score:.4f}{ci_str},\n"
            f"  assumption {status}\n"
            f")"
        )


@dataclass 
class DynamicsResult:
    """Results from generative function dynamics analysis.
    
    Attributes
    ----------
    avg_log_gradient : float
        Average of log|g'(x)| over the domain.
    conclusion : str
        'contracting', 'expanding', or 'inconclusive'.
    ci_lower : float, optional
        Lower confidence bound.
    ci_upper : float, optional
        Upper confidence bound.
    bootstrap_results : ndarray, optional
        Bootstrap distribution.
    """
    avg_log_gradient: float
    conclusion: Literal["contracting", "expanding", "inconclusive"]
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None
    bootstrap_results: Optional[np.ndarray] = None
    
    def __repr__(self) -> str:
        ci_str = ""
        if self.ci_lower is not None:
            ci_str = f", 95% CI: [{self.ci_lower:.4f}, {self.ci_upper:.4f}]"
        return (
            f"DynamicsResult(\n"
            f"  avg_log_gradient={self.avg_log_gradient:.4f}{ci_str},\n"
            f"  conclusion='{self.conclusion}'\n"
            f")"
        )


def _estimate_avg_log_gradient(
    x: np.ndarray,
    y: np.ndarray,
    smoothing_factor: Optional[float] = None,
    eps: float = 1e-8,
) -> float:
    """Estimate average log-gradient via spline fitting.
    
    Parameters
    ----------
    x : ndarray
        Input variable.
    y : ndarray
        Output variable.
    smoothing_factor : float, optional
        Smoothing parameter for spline. None uses automatic.
    eps : float
        Small constant to avoid log(0).
        
    Returns
    -------
    float
        Estimated average of log|g'(x)|.
    """
    # Sort data for spline fitting
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]
    
    # Fit smoothing spline
    spline = UnivariateSpline(x_sorted, y_sorted, s=smoothing_factor)
    deriv = spline.derivative(n=1)
    
    # Numerical integration of log|g'|
    x_min, x_max = np.min(x), np.max(x)
    if x_max - x_min < eps:
        return np.nan
    
    integrand = lambda t: np.log(np.abs(deriv(t)) + eps)
    result, _ = quad(integrand, x_min, x_max, limit=1000)
    
    return result / (x_max - x_min)


def check_orthogonality(
    x: ArrayLike,
    y: ArrayLike,
    n_bootstrap: int = 500,
    smoothing_factor: Optional[float] = None,
    random_state: Optional[int] = None,
) -> OrthogonalityResult:
    """Check functional orthogonality assumption via bootstrap.
    
    Tests whether E[log|g'(X)|] ≈ (1/(b-a))∫log|g'(x)|dx, which is
    required for the validity of the entropy-based causal discovery.
    
    Parameters
    ----------
    x : array-like
        Input variable (potential cause).
    y : array-like
        Output variable (potential effect).
    n_bootstrap : int, default=500
        Number of bootstrap iterations.
    smoothing_factor : float, optional
        Spline smoothing parameter.
    random_state : int, optional
        Random seed for reproducibility.
        
    Returns
    -------
    OrthogonalityResult
        Diagnostic results including bootstrap confidence intervals.
        
    Examples
    --------
    >>> import numpy as np
    >>> from entropycausal.diagnostics import check_orthogonality
    >>> x = np.random.uniform(0, 1, 300)
    >>> y = x**2 + np.random.normal(0, 0.1, 300)
    >>> result = check_orthogonality(x, y)
    >>> print(result)
    """
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    
    if random_state is not None:
        np.random.seed(random_state)
    
    eps = 1e-8
    n = len(x)
    
    def compute_score(x_data, y_data):
        sort_idx = np.argsort(x_data)
        x_s = x_data[sort_idx]
        y_s = y_data[sort_idx]
        
        try:
            spline = UnivariateSpline(x_s, y_s, s=smoothing_factor)
            deriv = spline.derivative(n=1)
            
            # LHS: sample average
            grad_at_x = deriv(x_data)
            log_grad = np.log(np.abs(grad_at_x) + eps)
            lhs = np.mean(log_grad)
            
            # RHS: integral average
            x_min, x_max = np.min(x_data), np.max(x_data)
            integrand = lambda t: np.log(np.abs(deriv(t)) + eps)
            rhs, _ = quad(integrand, x_min, x_max, limit=1000)
            rhs /= (x_max - x_min)
            
            return lhs - rhs, lhs, rhs
        except Exception:
            return np.nan, np.nan, np.nan
    
    # Point estimate
    score, lhs, rhs = compute_score(x, y)
    
    # Bootstrap
    bootstrap_scores = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, size=n, replace=True)
        s, _, _ = compute_score(x[idx], y[idx])
        if not np.isnan(s):
            bootstrap_scores.append(s)
    
    bootstrap_scores = np.array(bootstrap_scores)
    ci_lower, ci_upper = np.percentile(bootstrap_scores, [2.5, 97.5])
    
    return OrthogonalityResult(
        lhs_estimate=lhs,
        rhs_estimate=rhs,
        deviation_score=score,
        bootstrap_scores=bootstrap_scores,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
    )


def analyze_dynamics(
    x: ArrayLike,
    y: ArrayLike,
    n_bootstrap: int = 500,
    smoothing_factor: Optional[float] = None,
    random_state: Optional[int] = None,
) -> DynamicsResult:
    """Determine if generative function is contracting or expanding.
    
    Analyzes the dynamics of the relationship Y = g(X) by estimating
    the average log-gradient. Contracting functions (avg < 0) compress
    variance, while expanding functions (avg > 0) amplify it.
    
    Parameters
    ----------
    x : array-like
        Input variable.
    y : array-like
        Output variable.
    n_bootstrap : int, default=500
        Number of bootstrap iterations.
    smoothing_factor : float, optional
        Spline smoothing parameter.
    random_state : int, optional
        Random seed.
        
    Returns
    -------
    DynamicsResult
        Analysis results including bootstrap inference.
        
    Examples
    --------
    >>> import numpy as np
    >>> from entropycausal.diagnostics import analyze_dynamics
    >>> x = np.random.uniform(0, 1, 300)
    >>> y = np.sqrt(x)  # Contracting function
    >>> result = analyze_dynamics(x, y)
    >>> print(result.conclusion)  # 'contracting'
    """
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    
    if random_state is not None:
        np.random.seed(random_state)
    
    n = len(x)
    
    # Point estimate
    avg_log_grad = _estimate_avg_log_gradient(x, y, smoothing_factor)
    
    # Bootstrap
    bootstrap_results = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, size=n, replace=True)
        grad = _estimate_avg_log_gradient(x[idx], y[idx], smoothing_factor)
        if not np.isnan(grad):
            bootstrap_results.append(grad)
    
    bootstrap_results = np.array(bootstrap_results)
    ci_lower, ci_upper = np.percentile(bootstrap_results, [2.5, 97.5])
    
    # Determine conclusion
    if ci_upper < 0:
        conclusion = "contracting"
    elif ci_lower > 0:
        conclusion = "expanding"
    else:
        conclusion = "inconclusive"
    
    return DynamicsResult(
        avg_log_gradient=avg_log_grad,
        conclusion=conclusion,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        bootstrap_results=bootstrap_results,
    )


@dataclass
class DiagnosticSummary:
    """Combined diagnostic results."""
    orthogonality: OrthogonalityResult
    dynamics: DynamicsResult
    
    @property
    def all_passed(self) -> bool:
        return self.orthogonality.assumption_holds
    
    def __repr__(self) -> str:
        return (
            f"DiagnosticSummary(\n"
            f"  orthogonality: {self.orthogonality.assumption_holds},\n"
            f"  dynamics: {self.dynamics.conclusion}\n"
            f")"
        )


def run_diagnostics(
    x: ArrayLike,
    y: ArrayLike,
    n_bootstrap: int = 500,
    random_state: Optional[int] = None,
) -> DiagnosticSummary:
    """Run all diagnostic checks.
    
    Parameters
    ----------
    x : array-like
        Input variable.
    y : array-like
        Output variable.
    n_bootstrap : int
        Bootstrap iterations.
    random_state : int, optional
        Random seed.
        
    Returns
    -------
    DiagnosticSummary
        Combined diagnostic results.
    """
    orth = check_orthogonality(x, y, n_bootstrap, random_state=random_state)
    dyn = analyze_dynamics(x, y, n_bootstrap, random_state=random_state)
    return DiagnosticSummary(orthogonality=orth, dynamics=dyn)
