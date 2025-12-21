"""
Statistics Module
=================

Utility functions for statistical analysis and error estimation.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


def stack_profiles(
    profiles: List[np.ndarray],
    weights: Optional[np.ndarray] = None,
    method: str = 'mean',
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Stack multiple profiles (e.g., density, mass).

    Parameters
    ----------
    profiles : list of ndarray
        List of profile arrays with the same shape.
    weights : ndarray, optional
        Weights for each profile. Default is equal weighting.
    method : str
        Stacking method: 'mean', 'median', 'weighted_mean'.

    Returns
    -------
    stacked : ndarray
        Stacked profile.
    scatter : ndarray
        Standard deviation or MAD.

    Examples
    --------
    >>> profiles = [rho_1, rho_2, rho_3]  # List of density profiles
    >>> stacked, scatter = stack_profiles(profiles, method='median')
    """
    if len(profiles) == 0:
        raise ValueError("Empty profile list")
    
    # Convert to 2D array
    profiles_arr = np.array(profiles)
    
    if method == 'mean':
        stacked = np.nanmean(profiles_arr, axis=0)
        scatter = np.nanstd(profiles_arr, axis=0)
    
    elif method == 'median':
        stacked = np.nanmedian(profiles_arr, axis=0)
        # Median absolute deviation
        scatter = np.nanmedian(np.abs(profiles_arr - stacked), axis=0) * 1.4826
    
    elif method == 'weighted_mean':
        if weights is None:
            weights = np.ones(len(profiles))
        
        weights = np.array(weights).reshape(-1, 1)
        stacked = np.nansum(profiles_arr * weights, axis=0) / np.nansum(weights)
        
        # Weighted standard deviation
        variance = np.nansum(weights * (profiles_arr - stacked)**2, axis=0) / np.nansum(weights)
        scatter = np.sqrt(variance)
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return stacked, scatter


def bootstrap_error(
    data: np.ndarray,
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_bootstrap: int = 1000,
    confidence: float = 0.68,
    random_state: Optional[int] = None,
) -> Tuple[float, float, float]:
    """
    Compute bootstrap error estimate.

    Parameters
    ----------
    data : ndarray
        Data array.
    statistic : callable
        Statistic to compute (default: mean).
    n_bootstrap : int
        Number of bootstrap samples.
    confidence : float
        Confidence interval width.
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    value : float
        Central statistic value.
    lower : float
        Lower confidence bound.
    upper : float
        Upper confidence bound.

    Examples
    --------
    >>> masses = np.random.lognormal(30, 0.5, 100)
    >>> mean, lower, upper = bootstrap_error(masses, np.mean, n_bootstrap=1000)
    >>> print(f"Mean: {mean:.2e} ({lower:.2e}, {upper:.2e})")
    """
    rng = np.random.default_rng(random_state)
    n = len(data)
    
    # Bootstrap samples
    bootstrap_stats = np.zeros(n_bootstrap)
    
    for i in range(n_bootstrap):
        sample = rng.choice(data, size=n, replace=True)
        bootstrap_stats[i] = statistic(sample)
    
    # Percentiles for confidence interval
    alpha = (1 - confidence) / 2
    lower_pct = alpha * 100
    upper_pct = (1 - alpha) * 100
    
    value = statistic(data)
    lower = np.percentile(bootstrap_stats, lower_pct)
    upper = np.percentile(bootstrap_stats, upper_pct)
    
    return value, lower, upper


def bootstrap_error_2d(
    data: np.ndarray,
    statistic: Callable[[np.ndarray], np.ndarray] = lambda x: np.mean(x, axis=0),
    n_bootstrap: int = 1000,
    confidence: float = 0.68,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute bootstrap errors for 2D data (e.g., profiles).

    Parameters
    ----------
    data : ndarray
        2D data array (n_samples, n_points).
    statistic : callable
        Statistic to compute along axis=0.
    n_bootstrap : int
        Number of bootstrap samples.
    confidence : float
        Confidence interval width.
    random_state : int, optional
        Random seed.

    Returns
    -------
    values : ndarray
        Central statistic values.
    lower : ndarray
        Lower confidence bounds.
    upper : ndarray
        Upper confidence bounds.
    """
    rng = np.random.default_rng(random_state)
    n_samples, n_points = data.shape
    
    # Bootstrap samples
    bootstrap_stats = np.zeros((n_bootstrap, n_points))
    
    for i in range(n_bootstrap):
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        bootstrap_stats[i] = statistic(data[indices])
    
    # Percentiles
    alpha = (1 - confidence) / 2
    lower_pct = alpha * 100
    upper_pct = (1 - alpha) * 100
    
    values = statistic(data)
    lower = np.percentile(bootstrap_stats, lower_pct, axis=0)
    upper = np.percentile(bootstrap_stats, upper_pct, axis=0)
    
    return values, lower, upper


def jackknife_error(
    data: np.ndarray,
    statistic: Callable[[np.ndarray], float] = np.mean,
) -> Tuple[float, float]:
    """
    Compute jackknife error estimate.

    Parameters
    ----------
    data : ndarray
        Data array.
    statistic : callable
        Statistic to compute.

    Returns
    -------
    value : float
        Central value.
    error : float
        Jackknife error estimate.
    """
    n = len(data)
    
    # Leave-one-out estimates
    jackknife_stats = np.zeros(n)
    
    for i in range(n):
        sample = np.delete(data, i)
        jackknife_stats[i] = statistic(sample)
    
    value = statistic(data)
    
    # Jackknife variance
    variance = (n - 1) / n * np.sum((jackknife_stats - jackknife_stats.mean())**2)
    error = np.sqrt(variance)
    
    return value, error


def weighted_mean_and_error(
    values: np.ndarray,
    errors: np.ndarray,
) -> Tuple[float, float]:
    """
    Compute weighted mean and propagated error.

    Parameters
    ----------
    values : ndarray
        Values to average.
    errors : ndarray
        Errors on values.

    Returns
    -------
    mean : float
        Weighted mean.
    error : float
        Error on weighted mean.
    """
    # Weights from inverse variance
    weights = 1.0 / errors**2
    
    mean = np.sum(weights * values) / np.sum(weights)
    error = 1.0 / np.sqrt(np.sum(weights))
    
    return mean, error


def compute_chi2(
    data: np.ndarray,
    model: np.ndarray,
    errors: np.ndarray,
) -> float:
    """
    Compute chi-squared.

    Parameters
    ----------
    data : ndarray
        Observed data.
    model : ndarray
        Model prediction.
    errors : ndarray
        Errors on data.

    Returns
    -------
    chi2 : float
        Chi-squared value.
    """
    return np.sum(((data - model) / errors)**2)


def compute_reduced_chi2(
    data: np.ndarray,
    model: np.ndarray,
    errors: np.ndarray,
    n_params: int = 0,
) -> float:
    """
    Compute reduced chi-squared.

    Parameters
    ----------
    data : ndarray
        Observed data.
    model : ndarray
        Model prediction.
    errors : ndarray
        Errors on data.
    n_params : int
        Number of fitted parameters.

    Returns
    -------
    chi2_red : float
        Reduced chi-squared.
    """
    n_dof = len(data) - n_params
    if n_dof <= 0:
        return np.inf
    
    chi2 = compute_chi2(data, model, errors)
    return chi2 / n_dof


def bin_data(
    x: np.ndarray,
    y: np.ndarray,
    bins: Union[int, np.ndarray] = 10,
    statistic: str = 'mean',
    errors: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """
    Bin data and compute statistics.

    Parameters
    ----------
    x : ndarray
        Independent variable.
    y : ndarray
        Dependent variable.
    bins : int or ndarray
        Number of bins or bin edges.
    statistic : str
        Statistic to compute: 'mean', 'median', 'sum', 'count'.
    errors : ndarray, optional
        Errors on y values.

    Returns
    -------
    result : dict
        Dictionary with 'x_centers', 'y_stat', 'y_err', 'counts'.
    """
    from scipy import stats as scipy_stats
    
    # Compute binned statistic
    if statistic == 'mean':
        stat, bin_edges, _ = scipy_stats.binned_statistic(x, y, statistic='mean', bins=bins)
        std, _, _ = scipy_stats.binned_statistic(x, y, statistic='std', bins=bins)
        count, _, _ = scipy_stats.binned_statistic(x, y, statistic='count', bins=bins)
        err = std / np.sqrt(count)
    
    elif statistic == 'median':
        stat, bin_edges, _ = scipy_stats.binned_statistic(x, y, statistic='median', bins=bins)
        
        def mad(arr):
            return np.nanmedian(np.abs(arr - np.nanmedian(arr))) * 1.4826
        
        std, _, _ = scipy_stats.binned_statistic(x, y, statistic=mad, bins=bins)
        count, _, _ = scipy_stats.binned_statistic(x, y, statistic='count', bins=bins)
        err = std / np.sqrt(count)
    
    elif statistic == 'sum':
        stat, bin_edges, _ = scipy_stats.binned_statistic(x, y, statistic='sum', bins=bins)
        count, _, _ = scipy_stats.binned_statistic(x, y, statistic='count', bins=bins)
        
        if errors is not None:
            # Propagate errors in quadrature
            err_sq, _, _ = scipy_stats.binned_statistic(x, errors**2, statistic='sum', bins=bins)
            err = np.sqrt(err_sq)
        else:
            err = np.sqrt(stat)  # Poisson error for counts
    
    else:
        stat, bin_edges, _ = scipy_stats.binned_statistic(x, y, statistic=statistic, bins=bins)
        count, _, _ = scipy_stats.binned_statistic(x, y, statistic='count', bins=bins)
        err = np.zeros_like(stat)
    
    # Bin centers
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    
    return {
        'x_centers': bin_centers,
        'y_stat': stat,
        'y_err': err,
        'counts': count,
        'bin_edges': bin_edges,
    }


def percentile_range(
    data: np.ndarray,
    lower: float = 16,
    upper: float = 84,
) -> Tuple[float, float, float]:
    """
    Compute median and percentile range.

    Parameters
    ----------
    data : ndarray
        Data array.
    lower : float
        Lower percentile.
    upper : float
        Upper percentile.

    Returns
    -------
    median : float
        Median value.
    lower_val : float
        Lower percentile value.
    upper_val : float
        Upper percentile value.
    """
    return (
        np.nanpercentile(data, 50),
        np.nanpercentile(data, lower),
        np.nanpercentile(data, upper),
    )


def running_statistic(
    x: np.ndarray,
    y: np.ndarray,
    window: int = 10,
    statistic: str = 'mean',
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute running statistic.

    Parameters
    ----------
    x : ndarray
        Independent variable.
    y : ndarray
        Dependent variable.
    window : int
        Window size.
    statistic : str
        'mean' or 'median'.

    Returns
    -------
    x_out : ndarray
        Windowed x values.
    y_out : ndarray
        Running statistic.
    """
    # Sort by x
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]
    
    n = len(x)
    n_out = n - window + 1
    
    x_out = np.zeros(n_out)
    y_out = np.zeros(n_out)
    
    for i in range(n_out):
        x_out[i] = np.mean(x_sorted[i:i+window])
        if statistic == 'mean':
            y_out[i] = np.mean(y_sorted[i:i+window])
        elif statistic == 'median':
            y_out[i] = np.median(y_sorted[i:i+window])
    
    return x_out, y_out


def correlation_coefficient(
    x: np.ndarray,
    y: np.ndarray,
) -> Tuple[float, float]:
    """
    Compute Pearson correlation coefficient and p-value.

    Parameters
    ----------
    x : ndarray
        First variable.
    y : ndarray
        Second variable.

    Returns
    -------
    r : float
        Correlation coefficient.
    p : float
        p-value.
    """
    from scipy import stats as scipy_stats
    
    # Remove NaN values
    mask = np.isfinite(x) & np.isfinite(y)
    return scipy_stats.pearsonr(x[mask], y[mask])
