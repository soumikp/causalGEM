"""
Core causal discovery estimators based on the Generative Exposure Model (GEM).

This module implements the differential entropy statistic C_{X→Y} = H(X) - H(Y)
for bivariate causal discovery under the GEM framework.

The Generative Exposure Model assumes Y = g(X) + ε where g is a smooth
generative function and ε is independent noise. Under this model:
    H(Y) - H(X) = E[log|g'(X)|]
which creates detectable asymmetries for causal inference.

Decision Logic
--------------
The causal direction is determined by combining three pieces of information:
1. Orthogonality assumption: Must hold for valid inference
2. Function dynamics: Whether g is contracting or expanding
3. Sign of C = H(X) - H(Y)

Decision rules:
- Contracting dynamics + C > 0 (significant) → X → Y
- Expanding dynamics + C < 0 (significant) → X → Y  
- Contracting dynamics + C < 0 (significant) → Y → X
- Expanding dynamics + C > 0 (significant) → Y → X
- Orthogonality violated or dynamics inconclusive → Inconclusive
"""

from dataclasses import dataclass, field
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
    orthogonality_holds : bool or None
        Whether the orthogonality assumption is satisfied.
    dynamics : str or None
        Function dynamics: 'contracting', 'expanding', or 'inconclusive'.
    decision_reason : str
        Explanation for the causal decision.
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
    orthogonality_holds: Optional[bool] = None
    dynamics: Optional[str] = None
    decision_reason: str = ""
    
    def __repr__(self) -> str:
        direction = {1: "X → Y", -1: "Y → X", 0: "Inconclusive"}[self.decision]
        parts = [
            f"CausalGEMResult(",
            f"  C_{{X→Y}} = {self.delta:.4f} [{self.ci_lower:.4f}, {self.ci_upper:.4f}],",
            f"  decision = '{direction}',",
        ]
        if self.orthogonality_holds is not None:
            parts.append(f"  orthogonality = {self.orthogonality_holds},")
        if self.dynamics is not None:
            parts.append(f"  dynamics = '{self.dynamics}',")
        if self.decision_reason:
            parts.append(f"  reason = '{self.decision_reason}',")
        parts.extend([
            f"  H(X) = {self.h_x:.4f}, H(Y) = {self.h_y:.4f},",
            f"  n = {self.n_samples}",
            f")"
        ])
        return "\n".join(parts)
    
    @property
    def is_significant(self) -> bool:
        """Whether the entropy difference is statistically significant."""
        return self.ci_lower > 0 or self.ci_upper < 0
    
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
    """Estimate entropies using one split configuration."""
    margin_x = fastKDE.pdf_at_points(
        var1=estim_data[:, 0], 
        list_of_points=list(inf_data[:, 0])
    )
    margin_y = fastKDE.pdf_at_points(
        var1=estim_data[:, 1], 
        list_of_points=list(inf_data[:, 1])
    )
    
    valid = np.logical_and(margin_x > 0, margin_y > 0)
    margin_x = margin_x[valid]
    margin_y = margin_y[valid]
    
    log_fx = np.log(margin_x)
    log_fy = np.log(margin_y)
    
    h_x = -np.mean(log_fx)
    h_y = -np.mean(log_fy)
    
    return h_x, h_y, log_fx, log_fy


def _compute_entropy_stats(
    x: np.ndarray,
    y: np.ndarray,
    alpha: float = 0.05,
) -> Tuple[float, float, float, float, float, float]:
    """Compute entropy difference statistics without decision logic."""
    n = len(x)
    if n % 2 != 0:
        x, y = x[:-1], y[:-1]
        n -= 1
    
    data = np.column_stack((x, y))
    estim, inf = np.split(data, 2)
    
    h_x1, h_y1, log_fx1, log_fy1 = _estimate_entropy_single_split(estim, inf)
    cov1 = np.cov(log_fx1, log_fy1)
    delta_var1 = cov1[0, 0] + cov1[1, 1] - 2 * cov1[0, 1]
    
    h_x2, h_y2, log_fx2, log_fy2 = _estimate_entropy_single_split(inf, estim)
    cov2 = np.cov(log_fx2, log_fy2)
    delta_var2 = cov2[0, 0] + cov2[1, 1] - 2 * cov2[0, 1]
    
    h_x = (h_x1 + h_x2) / 2
    h_y = (h_y1 + h_y2) / 2
    delta = h_x - h_y
    
    delta_var = (delta_var1 + delta_var2) / 2
    delta_sd = np.sqrt(delta_var)
    
    n_eff = len(log_fx1)
    se = delta_sd / np.sqrt(n_eff)
    
    z = st.norm.ppf(1 - alpha / 2)
    ci_lower = delta - z * se
    ci_upper = delta + z * se
    
    return delta, ci_lower, ci_upper, se, h_x, h_y


def estimate_causal_direction(
    x: ArrayLike,
    y: ArrayLike,
    alpha: float = 0.05,
    normalize: bool = True,
    check_assumptions: bool = True,
    n_bootstrap: int = 200,
    random_state: Optional[int] = None,
) -> CausalGEMResult:
    """Estimate causal direction using the GEM framework.
    
    This function implements the full GEM decision procedure:
    1. Check orthogonality assumption
    2. Analyze function dynamics (contracting vs expanding)
    3. Combine with entropy difference sign for causal direction
    
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
    check_assumptions : bool, default=True
        Whether to run diagnostic checks. If False, uses simple sign-based
        decision (not recommended for real analysis).
    n_bootstrap : int, default=200
        Number of bootstrap iterations for diagnostics.
    random_state : int, optional
        Random seed for reproducibility.
        
    Returns
    -------
    result : CausalGEMResult
        Object containing estimates, diagnostics, and decision.
        
    Examples
    --------
    >>> import numpy as np
    >>> from causalgem import estimate_causal_direction
    >>> np.random.seed(42)
    >>> x = np.random.uniform(0, 1, 500)
    >>> y = x**2 + np.random.normal(0, 0.1, 500)  # X causes Y via contracting g
    >>> result = estimate_causal_direction(x, y)
    >>> print(result.direction_str)
    'X → Y'
    
    Notes
    -----
    Decision logic under GEM (Y = g(X) + ε):
    
    - If g is **contracting** (|g'| < 1 on average): H(Y) < H(X), so C > 0
    - If g is **expanding** (|g'| > 1 on average): H(Y) > H(X), so C < 0
    
    Therefore:
    - Contracting + C > 0 → X → Y
    - Expanding + C < 0 → X → Y
    - Contracting + C < 0 → Y → X (X,Y roles reversed)
    - Expanding + C > 0 → Y → X (X,Y roles reversed)
    
    The orthogonality assumption must hold for these inferences to be valid.
    """
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    
    if len(x) != len(y):
        raise ValueError(f"x and y must have same length, got {len(x)} and {len(y)}")
    
    n = len(x)
    if n % 2 != 0:
        x, y = x[:-1], y[:-1]
        n -= 1
    
    # Normalize
    if normalize:
        x_norm = _affine_transform(x)
        y_norm = _affine_transform(y)
    else:
        x_norm, y_norm = x.copy(), y.copy()
    
    # Compute entropy statistics
    delta, ci_lower, ci_upper, se, h_x, h_y = _compute_entropy_stats(
        x_norm, y_norm, alpha
    )
    
    # Determine if statistically significant
    is_significant = ci_lower > 0 or ci_upper < 0
    c_positive = delta > 0
    
    # Default values
    orthogonality_holds = None
    dynamics = None
    decision = 0
    reason = ""
    
    if not check_assumptions:
        # Simple sign-based decision (legacy behavior)
        if not is_significant:
            decision = 0
            reason = "CI includes zero"
        else:
            decision = 1 if c_positive else -1
            reason = "Sign-based (assumptions not checked)"
    else:
        # Import diagnostics here to avoid circular imports
        from causalgem.diagnostics import check_orthogonality, analyze_dynamics
        
        # Step 1: Check orthogonality
        orth_result = check_orthogonality(
            x_norm, y_norm, 
            n_bootstrap=n_bootstrap,
            random_state=random_state
        )
        orthogonality_holds = orth_result.assumption_holds
        
        if not orthogonality_holds:
            decision = 0
            reason = "Orthogonality assumption violated"
        elif not is_significant:
            decision = 0
            reason = "CI includes zero (not significant)"
        else:
            # Step 2: Analyze dynamics
            dyn_result = analyze_dynamics(
                x_norm, y_norm,
                n_bootstrap=n_bootstrap,
                random_state=random_state
            )
            dynamics = dyn_result.conclusion
            
            if dynamics == "inconclusive":
                decision = 0
                reason = "Dynamics inconclusive"
            elif dynamics == "contracting":
                if c_positive:
                    decision = 1  # X → Y
                    reason = "Contracting dynamics + positive C"
                else:
                    decision = -1  # Y → X
                    reason = "Contracting dynamics + negative C (reversed)"
            elif dynamics == "expanding":
                if c_positive:
                    decision = -1  # Y → X
                    reason = "Expanding dynamics + positive C (reversed)"
                else:
                    decision = 1  # X → Y
                    reason = "Expanding dynamics + negative C"
    
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
        orthogonality_holds=orthogonality_holds,
        dynamics=dynamics,
        decision_reason=reason,
    )


def estimate_entropy_difference(
    x: ArrayLike,
    y: ArrayLike,
    alpha: float = 0.05,
    normalize: bool = True,
) -> CausalGEMResult:
    """Estimate entropy difference C_{X→Y} = H(X) - H(Y) without assumption checks.
    
    This is a lower-level function that computes the entropy difference
    without running diagnostic checks. For full causal inference with
    proper assumption verification, use `estimate_causal_direction()`.
    
    Parameters
    ----------
    x : array-like of shape (n,)
        First variable.
    y : array-like of shape (n,)
        Second variable.
    alpha : float, default=0.05
        Significance level for confidence intervals.
    normalize : bool, default=True
        Whether to apply affine transformation to [0,1].
        
    Returns
    -------
    result : CausalGEMResult
        Object with entropy estimates and confidence intervals.
        Note: Decision is based only on sign, not full GEM logic.
    """
    return estimate_causal_direction(
        x, y, alpha=alpha, normalize=normalize, check_assumptions=False
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
    result = estimate_causal_direction(x, y, alpha=alpha)
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
    check_assumptions: bool = True,
    n_bootstrap: int = 200,
) -> CausalGEMResult:
    """Estimate causal direction with stratified data.
    
    Combines estimates across strata using inverse-variance weighting.
    Diagnostics are run on pooled data.
    
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
    check_assumptions : bool
        Whether to run diagnostic checks.
    n_bootstrap : int
        Bootstrap iterations for diagnostics.
        
    Returns
    -------
    result : CausalGEMResult
        Combined estimate across all strata.
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
        # Use entropy_difference (no assumption check) for each stratum
        result = estimate_entropy_difference(x[mask], y[mask], alpha=alpha)
        deltas.append(result.delta)
        variances.append(result.std_error**2 * result.n_samples)
        ns.append(result.n_samples)
    
    if not deltas:
        raise ValueError("No strata with sufficient samples (need >= 10)")
    
    deltas = np.array(deltas)
    variances = np.array(variances)
    ns = np.array(ns)
    
    weights = ns / ns.sum()
    delta = np.sum(weights * deltas)
    delta_var = np.sum(weights**2 * variances) / ns.sum()
    delta_se = np.sqrt(delta_var)
    
    z = st.norm.ppf(1 - alpha / 2)
    ci_lower = delta - z * delta_se
    ci_upper = delta + z * delta_se
    
    is_significant = ci_lower > 0 or ci_upper < 0
    c_positive = delta > 0
    
    # Run diagnostics on pooled normalized data
    orthogonality_holds = None
    dynamics = None
    decision = 0
    reason = ""
    
    if check_assumptions:
        from causalgem.diagnostics import check_orthogonality, analyze_dynamics
        
        x_norm = _affine_transform(x)
        y_norm = _affine_transform(y)
        
        orth_result = check_orthogonality(x_norm, y_norm, n_bootstrap=n_bootstrap)
        orthogonality_holds = orth_result.assumption_holds
        
        if not orthogonality_holds:
            decision = 0
            reason = "Orthogonality assumption violated"
        elif not is_significant:
            decision = 0
            reason = "CI includes zero"
        else:
            dyn_result = analyze_dynamics(x_norm, y_norm, n_bootstrap=n_bootstrap)
            dynamics = dyn_result.conclusion
            
            if dynamics == "inconclusive":
                decision = 0
                reason = "Dynamics inconclusive"
            elif dynamics == "contracting":
                decision = 1 if c_positive else -1
                reason = f"Contracting + {'positive' if c_positive else 'negative'} C"
            else:
                decision = -1 if c_positive else 1
                reason = f"Expanding + {'positive' if c_positive else 'negative'} C"
    else:
        if not is_significant:
            decision = 0
            reason = "CI includes zero"
        else:
            decision = 1 if c_positive else -1
            reason = "Sign-based (assumptions not checked)"
    
    return CausalGEMResult(
        delta=delta,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        std_error=delta_se,
        h_x=np.nan,
        h_y=np.nan,
        decision=decision,
        n_samples=int(ns.sum()),
        alpha=alpha,
        orthogonality_holds=orthogonality_holds,
        dynamics=dynamics,
        decision_reason=reason,
    )


# Backward compatibility
EntropyCausalResult = CausalGEMResult
