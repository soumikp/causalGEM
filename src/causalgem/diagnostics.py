"""
Diagnostic tools for checking GEM model assumptions.

This module provides functions to verify the key assumptions underlying
the Generative Exposure Model (GEM) framework:

1. **Functional Orthogonality**: E[log|g'(X)|] ≈ ∫log|g'(x)|dx
2. **Function Dynamics**: Whether g is contracting (|g'| < 1) or expanding (|g'| > 1)

These diagnostics are essential for valid causal inference under GEM.
The main estimation function `estimate_causal_direction` calls these
automatically, but they can also be run separately for deeper analysis.
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
    "DiagnosticSummary",
    "check_orthogonality",
    "analyze_dynamics",
    "run_diagnostics",
]


@dataclass
class OrthogonalityResult:
    """Results from functional orthogonality diagnostic.
    
    The orthogonality assumption requires that E[log|g'(X)|] equals the
    integral average ∫log|g'(x)|dx / (b-a). This is tested via bootstrap.
    
    Attributes
    ----------
    lhs_estimate : float
        Left-hand side: sample average E[log|g'(X)|].
    rhs_estimate : float
        Right-hand side: integral average of log|g'| over support.
    deviation_score : float
        LHS - RHS. Values near 0 indicate assumption holds.
    bootstrap_scores : ndarray, optional
        Bootstrap distribution of deviation scores.
    ci_lower : float, optional
        Lower 95% confidence bound for deviation.
    ci_upper : float, optional
        Upper 95% confidence bound for deviation.
    """
    lhs_estimate: float
    rhs_estimate: float
    deviation_score: float
    bootstrap_scores: Optional[np.ndarray] = None
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None
    
    @property
    def assumption_holds(self) -> bool:
        """Whether the orthogonality assumption is satisfied.
        
        Returns True if the 95% CI for the deviation includes 0.
        """
        if self.ci_lower is None or self.ci_upper is None:
            return abs(self.deviation_score) < 0.5  # Heuristic fallback
        return self.ci_lower <= 0 <= self.ci_upper
    
    def __repr__(self) -> str:
        status = "✓ holds" if self.assumption_holds else "✗ violated"
        ci_str = ""
        if self.ci_lower is not None:
            ci_str = f", 95% CI: [{self.ci_lower:.4f}, {self.ci_upper:.4f}]"
        return (
            f"OrthogonalityResult(\n"
            f"  deviation = {self.deviation_score:.4f}{ci_str},\n"
            f"  assumption {status}\n"
            f")"
        )


@dataclass 
class DynamicsResult:
    """Results from generative function dynamics analysis.
    
    Determines whether the generative function g is contracting (|g'| < 1
    on average) or expanding (|g'| > 1 on average) by estimating the
    average of log|g'(x)| over the domain.
    
    Attributes
    ----------
    avg_log_gradient : float
        Average of log|g'(x)| over the domain.
        Negative → contracting, Positive → expanding.
    conclusion : str
        'contracting', 'expanding', or 'inconclusive'.
    ci_lower : float, optional
        Lower 95% confidence bound.
    ci_upper : float, optional
        Upper 95% confidence bound.
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
            f"  avg_log_gradient = {self.avg_log_gradient:.4f}{ci_str},\n"
            f"  conclusion = '{self.conclusion}'\n"
            f")"
        )


@dataclass
class DiagnosticSummary:
    """Combined diagnostic results from all assumption checks.
    
    Attributes
    ----------
    orthogonality : OrthogonalityResult
        Results from orthogonality check.
    dynamics : DynamicsResult
        Results from dynamics analysis.
    """
    orthogonality: OrthogonalityResult
    dynamics: DynamicsResult
    
    @property
    def all_passed(self) -> bool:
        """Whether all required assumptions are satisfied."""
        return self.orthogonality.assumption_holds
    
    @property
    def can_determine_direction(self) -> bool:
        """Whether we can determine causal direction.
        
        Requires orthogonality to hold and dynamics to be conclusive.
        """
        return (self.orthogonality.assumption_holds and 
                self.dynamics.conclusion != "inconclusive")
    
    def __repr__(self) -> str:
        return (
            f"DiagnosticSummary(\n"
            f"  orthogonality: {'✓' if self.orthogonality.assumption_holds else '✗'} "
            f"(deviation = {self.orthogonality.deviation_score:.4f}),\n"
            f"  dynamics: {self.dynamics.conclusion} "
            f"(avg_log_grad = {self.dynamics.avg_log_gradient:.4f}),\n"
            f"  can_determine_direction: {self.can_determine_direction}\n"
            f")"
        )


def _estimate_avg_log_gradient(
    x: np.ndarray,
    y: np.ndarray,
    smoothing_factor: Optional[float] = None,
    eps: float = 1e-8,
) -> float:
    """Estimate average log-gradient via spline fitting.
    
    Fits a smoothing spline to (x, y), computes the derivative,
    and returns the integral average of log|g'| over the domain.
    """
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]
    
    spline = UnivariateSpline(x_sorted, y_sorted, s=smoothing_factor)
    deriv = spline.derivative(n=1)
    
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
    
    The test fits a smoothing spline to estimate g, then compares the
    sample average of log|g'(X)| to the integral average.
    
    Parameters
    ----------
    x : array-like
        Input variable (potential cause).
    y : array-like
        Output variable (potential effect).
    n_bootstrap : int, default=500
        Number of bootstrap iterations for CI estimation.
    smoothing_factor : float, optional
        Spline smoothing parameter. None uses automatic selection.
    random_state : int, optional
        Random seed for reproducibility.
        
    Returns
    -------
    OrthogonalityResult
        Diagnostic results with bootstrap confidence intervals.
        Check `.assumption_holds` for the verdict.
        
    Examples
    --------
    >>> import numpy as np
    >>> from causalgem import check_orthogonality
    >>> np.random.seed(42)
    >>> x = np.random.uniform(0, 1, 300)
    >>> y = x**2 + np.random.normal(0, 0.1, 300)
    >>> result = check_orthogonality(x, y)
    >>> print(f"Orthogonality holds: {result.assumption_holds}")
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
            
            # LHS: sample average of log|g'(X)|
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
    
    # Bootstrap for confidence interval
    bootstrap_scores = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, size=n, replace=True)
        s, _, _ = compute_score(x[idx], y[idx])
        if not np.isnan(s):
            bootstrap_scores.append(s)
    
    bootstrap_scores = np.array(bootstrap_scores)
    
    if len(bootstrap_scores) > 0:
        ci_lower, ci_upper = np.percentile(bootstrap_scores, [2.5, 97.5])
    else:
        ci_lower, ci_upper = np.nan, np.nan
    
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
    
    Estimates the average of log|g'(x)| over the domain by fitting a
    smoothing spline. The sign determines the dynamics:
    
    - avg_log_gradient < 0 → **contracting** (|g'| < 1 on average)
    - avg_log_gradient > 0 → **expanding** (|g'| > 1 on average)
    
    The conclusion is based on whether the 95% CI excludes zero.
    
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
        Analysis results with conclusion and confidence intervals.
        
    Examples
    --------
    >>> import numpy as np
    >>> from causalgem import analyze_dynamics
    >>> np.random.seed(42)
    >>> x = np.random.uniform(0, 1, 300)
    >>> y = np.sqrt(x)  # Contracting: derivative 1/(2√x) < 1 for x > 0.25
    >>> result = analyze_dynamics(x, y)
    >>> print(f"Dynamics: {result.conclusion}")
    """
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    
    if random_state is not None:
        np.random.seed(random_state)
    
    n = len(x)
    
    # Point estimate
    avg_log_grad = _estimate_avg_log_gradient(x, y, smoothing_factor)
    
    # Bootstrap for confidence interval
    bootstrap_results = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, size=n, replace=True)
        grad = _estimate_avg_log_gradient(x[idx], y[idx], smoothing_factor)
        if not np.isnan(grad):
            bootstrap_results.append(grad)
    
    bootstrap_results = np.array(bootstrap_results)
    
    if len(bootstrap_results) > 0:
        ci_lower, ci_upper = np.percentile(bootstrap_results, [2.5, 97.5])
    else:
        ci_lower, ci_upper = np.nan, np.nan
    
    # Determine conclusion based on whether CI excludes zero
    if np.isnan(ci_lower) or np.isnan(ci_upper):
        conclusion = "inconclusive"
    elif ci_upper < 0:
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


def run_diagnostics(
    x: ArrayLike,
    y: ArrayLike,
    n_bootstrap: int = 500,
    smoothing_factor: Optional[float] = None,
    random_state: Optional[int] = None,
) -> DiagnosticSummary:
    """Run all diagnostic checks for GEM assumptions.
    
    This runs both the orthogonality check and dynamics analysis,
    returning a combined summary that indicates whether causal
    direction can be reliably determined.
    
    Parameters
    ----------
    x : array-like
        Input variable (potential cause).
    y : array-like
        Output variable (potential effect).
    n_bootstrap : int, default=500
        Bootstrap iterations for both tests.
    smoothing_factor : float, optional
        Spline smoothing parameter.
    random_state : int, optional
        Random seed for reproducibility.
        
    Returns
    -------
    DiagnosticSummary
        Combined results from all diagnostic checks.
        Check `.can_determine_direction` for overall verdict.
        
    Examples
    --------
    >>> import numpy as np
    >>> from causalgem import run_diagnostics
    >>> np.random.seed(42)
    >>> x = np.random.uniform(0, 1, 500)
    >>> y = x**2 + np.random.normal(0, 0.1, 500)
    >>> diag = run_diagnostics(x, y)
    >>> print(diag)
    >>> if diag.can_determine_direction:
    ...     print("Safe to interpret causal direction")
    """
    orth = check_orthogonality(
        x, y, 
        n_bootstrap=n_bootstrap,
        smoothing_factor=smoothing_factor,
        random_state=random_state
    )
    dyn = analyze_dynamics(
        x, y, 
        n_bootstrap=n_bootstrap,
        smoothing_factor=smoothing_factor,
        random_state=random_state
    )
    return DiagnosticSummary(orthogonality=orth, dynamics=dyn)
