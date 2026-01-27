"""
Utility functions for computing response fractions and derived metrics.
"""

import numpy as np


def compute_F_S(S_R, S_D, S_H, threshold=0.02):
    """
    Compute response fraction with significance masking.
    
    F_S = (S_Replace - S_DMO) / (S_Hydro - S_DMO)
    
    Parameters
    ----------
    S_R : np.ndarray
        Statistic for Replace model
    S_D : np.ndarray
        Statistic for DMO
    S_H : np.ndarray
        Statistic for Hydro
    threshold : float, optional
        Relative threshold for masking insignificant baryonic effects (default: 0.02)
    
    Returns
    -------
    np.ndarray
        Response fraction (values where |ΔS| < threshold are set to NaN)
    """
    Delta_S = S_H - S_D
    F = (S_R - S_D) / Delta_S
    
    # Mask where baryonic effect is insignificant
    significant = np.abs(Delta_S) > threshold * np.abs(S_D)
    F = F.astype(float)
    F[~significant] = np.nan
    
    return F


def compute_Delta_F(S_tile, S_D, S_H, threshold=0.02):
    """
    Compute tile contribution to response.
    
    ΔF_S = (S_tile - S_DMO) / (S_Hydro - S_DMO)
    
    Parameters
    ----------
    S_tile : np.ndarray
        Statistic for discrete tile model
    S_D : np.ndarray
        Statistic for DMO
    S_H : np.ndarray
        Statistic for Hydro
    threshold : float, optional
        Relative threshold for masking (default: 0.02)
    
    Returns
    -------
    np.ndarray
        Tile contribution
    """
    return compute_F_S(S_tile, S_D, S_H, threshold=threshold)


def compute_epsilon(F_tiles_sum, F_cumulative, threshold=0.02):
    """
    Compute non-additivity metric.
    
    ε = F_cumulative - Σ ΔF_tiles
    
    Parameters
    ----------
    F_tiles_sum : np.ndarray
        Sum of tile contributions
    F_cumulative : np.ndarray
        Cumulative response fraction
    threshold : float, optional
        Relative threshold for masking (default: 0.02)
    
    Returns
    -------
    np.ndarray
        Non-additivity (positive = super-additivity, negative = sub-additivity)
    """
    epsilon = F_cumulative - F_tiles_sum
    
    # Optionally mask where cumulative response is small
    # (epsilon is less meaningful when F_cumulative ≈ 0)
    mask = np.abs(F_cumulative) < threshold
    epsilon = epsilon.astype(float)
    epsilon[mask] = np.nan
    
    return epsilon


def compute_mean_F(F, x, weights=None, x_range=None):
    """
    Compute weighted average of response fraction.
    
    Parameters
    ----------
    F : np.ndarray
        Response fraction array
    x : np.ndarray
        Independent variable (e.g., k, ℓ, S/N)
    weights : np.ndarray, optional
        Weights for averaging (default: uniform)
    x_range : tuple, optional
        (x_min, x_max) range for averaging (default: all)
    
    Returns
    -------
    float
        Weighted mean response
    """
    # Filter by x_range if provided
    if x_range is not None:
        x_min, x_max = x_range
        mask = (x >= x_min) & (x <= x_max)
        F = F[mask]
        x = x[mask]
        if weights is not None:
            weights = weights[mask]
    
    # Remove NaNs
    valid = ~np.isnan(F)
    F = F[valid]
    if weights is not None:
        weights = weights[valid]
    
    if len(F) == 0:
        return np.nan
    
    # Compute weighted mean
    if weights is None:
        return np.mean(F)
    else:
        return np.average(F, weights=weights)


def bootstrap_F(R, D, H, n_bootstrap=1000, seed=42):
    """
    Compute response fraction and uncertainty via bootstrap over realizations.
    
    For statistics with shared cosmic variance (e.g., peak counts), bootstrap
    resampling accounts for correlations between DMO, Hydro, and Replace.
    
    Parameters
    ----------
    R : np.ndarray
        Replace statistic, shape (N_realizations, N_bins)
    D : np.ndarray
        DMO statistic, shape (N_realizations, N_bins)
    H : np.ndarray
        Hydro statistic, shape (N_realizations, N_bins)
    n_bootstrap : int, optional
        Number of bootstrap samples (default: 1000)
    seed : int, optional
        Random seed for reproducibility (default: 42)
    
    Returns
    -------
    F_mean : np.ndarray
        Mean response fraction, shape (N_bins,)
    F_std : np.ndarray
        Standard deviation, shape (N_bins,)
    """
    n_real, n_bins = R.shape
    
    rng = np.random.default_rng(seed)
    F_samples = np.zeros((n_bootstrap, n_bins))
    
    for i in range(n_bootstrap):
        # Resample realizations with replacement
        idx = rng.choice(n_real, size=n_real, replace=True)
        
        # Sum over realizations
        N_R = np.sum(R[idx], axis=0)
        N_D = np.sum(D[idx], axis=0)
        N_H = np.sum(H[idx], axis=0)
        
        # Compute response
        F_samples[i] = (N_R - N_D) / (N_H - N_D)
    
    return np.nanmean(F_samples, axis=0), np.nanstd(F_samples, axis=0)


def compute_summary_stats(F, x, x_ranges=None):
    """
    Compute summary statistics of response over scale ranges.
    
    Parameters
    ----------
    F : np.ndarray
        Response fraction
    x : np.ndarray
        Independent variable (e.g., k, ℓ)
    x_ranges : list of tuples, optional
        List of (x_min, x_max, label) tuples
    
    Returns
    -------
    dict
        Dictionary of {label: mean_F}
    """
    if x_ranges is None:
        # Default ranges for different statistics
        if np.max(x) > 100:  # Likely ℓ
            x_ranges = [
                (100, 1000, 'low_ell'),
                (1000, 5000, 'mid_ell'),
                (5000, 10000, 'high_ell')
            ]
        else:  # Likely k
            x_ranges = [
                (0.1, 1.0, 'low_k'),
                (1.0, 10.0, 'mid_k'),
                (10.0, 100.0, 'high_k')
            ]
    
    summary = {}
    for x_min, x_max, label in x_ranges:
        mean_F = compute_mean_F(F, x, x_range=(x_min, x_max))
        summary[label] = mean_F
    
    return summary


# ============================================================
# Miller Correction Functions
# ============================================================

def compute_mass_deficit(M_dmo, M_replace):
    """
    Compute fractional mass deficit.
    
    f = (M_DMO - M_Replace) / M_DMO
    
    f > 0 means Replace has less mass than DMO (mass deficit)
    f < 0 means Replace has more mass than DMO (mass excess)
    
    Parameters
    ----------
    M_dmo : float or np.ndarray
        DMO total mass
    M_replace : float or np.ndarray
        Replace total mass
    
    Returns
    -------
    float or np.ndarray
        Mass deficit fraction
    """
    return (M_dmo - M_replace) / M_dmo


def apply_miller_correction(Pk, f):
    """
    Apply Miller et al. (2023) correction to power spectrum.
    
    The mass deficit artificially boosts P(k) because of reduced mean density:
        δ = ρ/⟨ρ⟩ - 1
    
    If ⟨ρ⟩ is lower (mass deficit), δ is artificially inflated.
    
    Correction rescales the field: ρ_corr = ρ × (1-f)
    For power spectrum (∝ ρ²): P_corr(k) = P(k) × (1-f)²
    
    Parameters
    ----------
    Pk : np.ndarray
        Power spectrum (any shape)
    f : float or np.ndarray
        Mass deficit fraction. If array, should broadcast with Pk.
    
    Returns
    -------
    np.ndarray
        Corrected power spectrum
    """
    return Pk * (1 - f)**2


def compute_F_with_miller(Pk_R, Pk_D, Pk_H, f_R, threshold=0.02):
    """
    Compute response fraction with Miller correction applied.
    
    Parameters
    ----------
    Pk_R : np.ndarray
        Replace power spectrum
    Pk_D : np.ndarray
        DMO power spectrum
    Pk_H : np.ndarray
        Hydro power spectrum
    f_R : float or np.ndarray
        Mass deficit of Replace model
    threshold : float
        Relative threshold for masking insignificant baryonic effects
    
    Returns
    -------
    F_uncorr : np.ndarray
        Uncorrected response fraction
    F_corr : np.ndarray
        Miller-corrected response fraction
    """
    # Uncorrected response
    F_uncorr = compute_F_S(Pk_R, Pk_D, Pk_H, threshold=threshold)
    
    # Apply Miller correction to Replace P(k)
    Pk_R_corr = apply_miller_correction(Pk_R, f_R)
    F_corr = compute_F_S(Pk_R_corr, Pk_D, Pk_H, threshold=threshold)
    
    return F_uncorr, F_corr


def compute_mass_deficit_from_stats(model_mass, dmo_mass):
    """
    Compute mass deficit from stats.h5 mass arrays.
    
    Parameters
    ----------
    model_mass : np.ndarray
        Mass array from Replace model, shape (N_LP, N_planes)
    dmo_mass : np.ndarray
        Mass array from DMO model, shape (N_LP, N_planes)
    
    Returns
    -------
    f_mean : float
        Mean mass deficit across all lensplanes
    f_std : float
        Standard deviation of mass deficit
    f_all : np.ndarray
        Mass deficit for each lensplane
    """
    # Compute mass deficit for each lensplane
    f_all = compute_mass_deficit(dmo_mass, model_mass)
    
    # Mean and std across all lensplanes
    f_mean = np.nanmean(f_all)
    f_std = np.nanstd(f_all)
    
    return f_mean, f_std, f_all

