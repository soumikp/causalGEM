"""
Utility functions for data preprocessing and transformations.
"""

from typing import Tuple, Union
import numpy as np
from numpy.typing import ArrayLike

__all__ = [
    "affine_transform",
    "rank_transform",
    "standardize",
    "preprocess_pair",
    "validate_inputs",
]


def affine_transform(
    x: ArrayLike,
    target_min: float = 0.0,
    target_max: float = 1.0,
) -> np.ndarray:
    """Transform data to specified range via affine transformation.
    
    Parameters
    ----------
    x : array-like
        Input data.
    target_min : float, default=0.0
        Minimum of target range.
    target_max : float, default=1.0
        Maximum of target range.
        
    Returns
    -------
    ndarray
        Transformed data in [target_min, target_max].
        
    Examples
    --------
    >>> x = np.array([1, 2, 3, 4, 5])
    >>> affine_transform(x)
    array([0.  , 0.25, 0.5 , 0.75, 1.  ])
    """
    x = np.asarray(x).flatten()
    x_min, x_max = np.min(x), np.max(x)
    
    if x_max - x_min < 1e-10:
        # Constant data
        return np.full_like(x, (target_min + target_max) / 2, dtype=float)
    
    scaled = (x - x_min) / (x_max - x_min)
    return target_min + scaled * (target_max - target_min)


def rank_transform(x: ArrayLike, ties: str = "average") -> np.ndarray:
    """Transform data to ranks, optionally normalized to (0, 1).
    
    Parameters
    ----------
    x : array-like
        Input data.
    ties : str, default='average'
        How to handle ties: 'average', 'min', 'max', 'dense', 'ordinal'.
        
    Returns
    -------
    ndarray
        Rank-transformed data in (0, 1).
        
    Examples
    --------
    >>> x = np.array([3, 1, 4, 1, 5])
    >>> rank_transform(x)
    array([0.5       , 0.25      , 0.66666667, 0.25      , 0.83333333])
    """
    from scipy.stats import rankdata
    x = np.asarray(x).flatten()
    ranks = rankdata(x, method=ties)
    return ranks / (len(x) + 1)


def standardize(
    x: ArrayLike,
    ddof: int = 1,
) -> np.ndarray:
    """Standardize data to zero mean and unit variance.
    
    Parameters
    ----------
    x : array-like
        Input data.
    ddof : int, default=1
        Delta degrees of freedom for std calculation.
        
    Returns
    -------
    ndarray
        Standardized data.
    """
    x = np.asarray(x).flatten()
    return (x - np.mean(x)) / np.std(x, ddof=ddof)


def preprocess_pair(
    x: ArrayLike,
    y: ArrayLike,
    method: str = "affine",
    dropna: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Preprocess a pair of variables for analysis.
    
    Parameters
    ----------
    x : array-like
        First variable.
    y : array-like
        Second variable.
    method : str, default='affine'
        Transformation method: 'affine', 'rank', 'standardize', or 'none'.
    dropna : bool, default=True
        Whether to remove observations with missing values.
        
    Returns
    -------
    x_processed : ndarray
        Preprocessed first variable.
    y_processed : ndarray
        Preprocessed second variable.
        
    Raises
    ------
    ValueError
        If inputs have different lengths or unknown method.
    """
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    
    if len(x) != len(y):
        raise ValueError(f"x and y must have same length: {len(x)} vs {len(y)}")
    
    # Handle missing values
    if dropna:
        valid = ~(np.isnan(x) | np.isnan(y))
        x, y = x[valid], y[valid]
    
    # Apply transformation
    transform_funcs = {
        "affine": affine_transform,
        "rank": rank_transform,
        "standardize": standardize,
        "none": lambda z: z,
    }
    
    if method not in transform_funcs:
        raise ValueError(f"Unknown method: {method}. Use one of {list(transform_funcs.keys())}")
    
    transform = transform_funcs[method]
    return transform(x), transform(y)


def validate_inputs(
    x: ArrayLike,
    y: ArrayLike,
    min_samples: int = 20,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Validate and prepare inputs for estimation.
    
    Parameters
    ----------
    x : array-like
        First variable.
    y : array-like
        Second variable.
    min_samples : int, default=20
        Minimum required sample size.
        
    Returns
    -------
    x : ndarray
        Validated first variable.
    y : ndarray
        Validated second variable.
    n : int
        Sample size (ensured to be even).
        
    Raises
    ------
    ValueError
        If inputs are invalid or sample size too small.
    """
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    
    if len(x) != len(y):
        raise ValueError(f"x and y must have same length: {len(x)} vs {len(y)}")
    
    # Remove NaN
    valid = ~(np.isnan(x) | np.isnan(y))
    x, y = x[valid], y[valid]
    
    n = len(x)
    if n < min_samples:
        raise ValueError(f"Sample size {n} < minimum {min_samples}")
    
    # Ensure even sample size
    if n % 2 != 0:
        x, y = x[:-1], y[:-1]
        n -= 1
    
    return x, y, n


def compute_bootstrap_ci(
    data: np.ndarray,
    alpha: float = 0.05,
    method: str = "percentile",
) -> Tuple[float, float]:
    """Compute bootstrap confidence interval.
    
    Parameters
    ----------
    data : ndarray
        Bootstrap samples.
    alpha : float, default=0.05
        Significance level.
    method : str, default='percentile'
        CI method: 'percentile' or 'bca'.
        
    Returns
    -------
    lower : float
        Lower confidence bound.
    upper : float
        Upper confidence bound.
    """
    if method == "percentile":
        lower = np.percentile(data, 100 * alpha / 2)
        upper = np.percentile(data, 100 * (1 - alpha / 2))
    else:
        raise NotImplementedError(f"Method {method} not implemented")
    
    return lower, upper
