import numpy as np
import matplotlib.pyplot as plt
import h5py
import scienceplots
plt.style.use(['science', 'notebook'])


def compute_weighted_mean_F(F, F_err, axis=-1):
    """Compute inverse-variance weighted mean of F and its uncertainty.
    
    Uses weights w_i = 1/σ_i^2, giving:
        <F> = Σ(w_i * F_i) / Σ(w_i)
        σ(<F>) = 1 / sqrt(Σ(w_i))
    
    Parameters
    ----------
    F : ndarray
        F-statistic values, shape (..., n_bins)
    F_err : ndarray  
        Uncertainties on F, same shape as F
    axis : int
        Axis along which to compute the weighted mean (default: -1, the bin axis)
        
    Returns
    -------
    weighted_mean : ndarray
        Inverse-variance weighted mean, shape with `axis` dimension removed
    weighted_mean_err : ndarray
        Uncertainty on the weighted mean, same shape as weighted_mean
    """
    # Handle NaNs and zeros in errors
    with np.errstate(divide='ignore', invalid='ignore'):
        # Weights = 1/variance
        weights = np.where((F_err > 0) & np.isfinite(F_err) & np.isfinite(F), 
                          1.0 / (F_err**2), 0.0)
        
        # Weighted sum and sum of weights
        weighted_sum = np.nansum(weights * F, axis=axis)
        sum_weights = np.nansum(weights, axis=axis)
        
        # Weighted mean
        weighted_mean = np.where(sum_weights > 0, 
                                 weighted_sum / sum_weights, 
                                 np.nan)
        
        # Uncertainty on weighted mean: σ = 1/sqrt(Σw)
        weighted_mean_err = np.where(sum_weights > 0,
                                     1.0 / np.sqrt(sum_weights),
                                     np.nan)
    
    return weighted_mean, weighted_mean_err


def compute_integrated_scores(R, H, D, R_err=None, H_err=None, D_err=None, axis=-1):
    """Compute multiple integrated score metrics for comparing Replace to Hydro/DMO.
    
    These metrics avoid the bin-by-bin F issues by integrating over all bins.
    
    Parameters
    ----------
    R : ndarray
        Replace statistic values, shape (..., n_bins)
    H : ndarray
        Hydro statistic values, same shape as R
    D : ndarray
        DMO statistic values, same shape as R
    R_err, H_err, D_err : ndarray, optional
        Uncertainties on R, H, D. Same shape.
    axis : int
        Axis along which to integrate (default: -1, the bin axis)
        
    Returns
    -------
    scores : dict
        Dictionary with different integrated metrics:
        - 'IER': Integrated Effect Ratio = Σ(R-D) / Σ(H-D)
        - 'IER_err': Uncertainty on IER
        - 'signal_weighted_F': Σ|H-D|·F / Σ|H-D| (weights by baryonic signal)
        - 'signal_weighted_F_err': Uncertainty on signal-weighted F
        - 'frac_signal_weighted_F': Σ|(H-D)/D|·F / Σ|(H-D)/D| (weights by fractional signal)
        - 'snr_weighted_F': Σ|(H-D)/D|·(1/σ_F²)·F / Σ|(H-D)/D|·(1/σ_F²) (SNR weighting)
        - 'snr_weighted_F_err': Uncertainty on SNR-weighted F
        - 'rms_F': sqrt(mean(F²)) - magnitude without cancellation
        - 'chi2_frac': 1 - χ²(R,H)/χ²(D,H) - fraction of variance explained
        - 'mean_F': Simple unweighted mean (for comparison)
    """
    # Differences
    R_minus_D = R - D
    H_minus_D = H - D
    R_minus_H = R - H
    D_minus_H = D - H
    
    # Bin-by-bin F (for derived metrics)
    with np.errstate(divide='ignore', invalid='ignore'):
        F_binwise = np.where(np.abs(H_minus_D) > 1e-10, R_minus_D / H_minus_D, np.nan)
    
    scores = {}
    
    # 1. Integrated Effect Ratio (IER)
    # IER = Σ(R-D) / Σ(H-D) - treats the sum as a single measurement
    sum_R_minus_D = np.nansum(R_minus_D, axis=axis)
    sum_H_minus_D = np.nansum(H_minus_D, axis=axis)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        scores['IER'] = np.where(np.abs(sum_H_minus_D) > 1e-10,
                                  sum_R_minus_D / sum_H_minus_D, np.nan)
    
    # IER error via propagation
    if R_err is not None and H_err is not None and D_err is not None:
        # σ(Σ(R-D)) = sqrt(Σ(σ_R² + σ_D²))
        var_sum_R_minus_D = np.nansum(R_err**2 + D_err**2, axis=axis)
        var_sum_H_minus_D = np.nansum(H_err**2 + D_err**2, axis=axis)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            # Standard ratio error propagation
            rel_err_num = np.sqrt(var_sum_R_minus_D) / np.abs(sum_R_minus_D)
            rel_err_den = np.sqrt(var_sum_H_minus_D) / np.abs(sum_H_minus_D)
            scores['IER_err'] = np.abs(scores['IER']) * np.sqrt(rel_err_num**2 + rel_err_den**2)
    else:
        scores['IER_err'] = np.full_like(scores['IER'], np.nan)
    
    # 2. Signal-Weighted F (absolute signal)
    # <F>_signal = Σ(|H-D| · F) / Σ|H-D|
    abs_signal = np.abs(H_minus_D)
    weighted_sum = np.nansum(abs_signal * F_binwise, axis=axis)
    sum_weights = np.nansum(abs_signal, axis=axis)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        scores['signal_weighted_F'] = np.where(sum_weights > 1e-10,
                                                weighted_sum / sum_weights, np.nan)
    
    # Signal-weighted F error (approximate)
    if R_err is not None and H_err is not None and D_err is not None:
        # Each F_i has error F_err_i, weight w_i = |H-D|_i
        # σ(<F>_w) ≈ sqrt(Σ w_i² σ_F_i²) / Σw_i
        with np.errstate(divide='ignore', invalid='ignore'):
            F_err_binwise = np.abs(F_binwise) * np.sqrt(
                ((np.sqrt(R_err**2 + D_err**2)) / np.abs(R_minus_D + 1e-30))**2 +
                ((np.sqrt(H_err**2 + D_err**2)) / np.abs(H_minus_D + 1e-30))**2
            )
            weighted_var = np.nansum((abs_signal * F_err_binwise)**2, axis=axis)
            scores['signal_weighted_F_err'] = np.where(sum_weights > 1e-10,
                                                        np.sqrt(weighted_var) / sum_weights, np.nan)
    else:
        scores['signal_weighted_F_err'] = np.full_like(scores['signal_weighted_F'], np.nan)
    
    # 3. Fractional-Signal-Weighted F (weights by relative baryonic effect)
    # <F>_frac = Σ(|(H-D)/D| · F) / Σ|(H-D)/D|
    # This weights by where baryons have the largest *fractional* effect
    with np.errstate(divide='ignore', invalid='ignore'):
        frac_signal = np.abs(H_minus_D / D)
        frac_signal = np.where(np.isfinite(frac_signal), frac_signal, 0)
    
    frac_weighted_sum = np.nansum(frac_signal * F_binwise, axis=axis)
    sum_frac_weights = np.nansum(frac_signal, axis=axis)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        scores['frac_signal_weighted_F'] = np.where(sum_frac_weights > 1e-10,
                                                     frac_weighted_sum / sum_frac_weights, np.nan)
    
    # 3b. SNR-Weighted F (fractional signal × inverse variance)
    # <F>_SNR = Σ(|(H-D)/D| × 1/σ_F² × F) / Σ(|(H-D)/D| × 1/σ_F²)
    # This properly downweights noisy bins while still prioritizing where baryons matter
    if R_err is not None and H_err is not None and D_err is not None:
        # Compute bin-wise F error
        with np.errstate(divide='ignore', invalid='ignore'):
            F_err_binwise = np.abs(F_binwise) * np.sqrt(
                ((np.sqrt(R_err**2 + D_err**2)) / np.abs(R_minus_D + 1e-30))**2 +
                ((np.sqrt(H_err**2 + D_err**2)) / np.abs(H_minus_D + 1e-30))**2
            )
            # Inverse variance weights (handle zeros/infinities)
            inv_var = np.where((F_err_binwise > 0) & np.isfinite(F_err_binwise), 
                              1.0 / (F_err_binwise**2), 0)
        
        # Combined weight: fractional_signal × inverse_variance
        snr_weights = frac_signal * inv_var
        
        # Mask out invalid bins
        valid = np.isfinite(F_binwise) & np.isfinite(snr_weights) & (snr_weights > 0)
        snr_weights = np.where(valid, snr_weights, 0)
        F_masked = np.where(valid, F_binwise, 0)
        
        snr_weighted_sum = np.nansum(snr_weights * F_masked, axis=axis)
        sum_snr_weights = np.nansum(snr_weights, axis=axis)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            scores['snr_weighted_F'] = np.where(sum_snr_weights > 1e-10,
                                                 snr_weighted_sum / sum_snr_weights, np.nan)
            # Uncertainty: 1/sqrt(sum(weights))
            scores['snr_weighted_F_err'] = np.where(sum_snr_weights > 1e-10,
                                                     1.0 / np.sqrt(sum_snr_weights), np.nan)
    else:
        # Fall back to frac_signal_weighted_F if no errors available
        scores['snr_weighted_F'] = scores['frac_signal_weighted_F']
        scores['snr_weighted_F_err'] = np.full_like(scores['snr_weighted_F'], np.nan)
    
    # 4. RMS F - captures magnitude without cancellation
    F_squared = F_binwise**2
    n_valid = np.sum(np.isfinite(F_binwise), axis=axis)
    mean_F_squared = np.nansum(F_squared, axis=axis) / np.maximum(n_valid, 1)
    scores['rms_F'] = np.sqrt(mean_F_squared)
    
    # 5. Chi-squared improvement fraction
    # χ²_frac = 1 - Σ(R-H)² / Σ(D-H)²
    # If χ²_frac = 1, Replace perfectly matches Hydro
    # If χ²_frac = 0, Replace is no better than DMO
    chi2_replace = np.nansum(R_minus_H**2, axis=axis)
    chi2_dmo = np.nansum(D_minus_H**2, axis=axis)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        scores['chi2_frac'] = np.where(chi2_dmo > 1e-20,
                                        1.0 - chi2_replace / chi2_dmo, np.nan)
    
    # 6. Simple unweighted mean (for comparison)
    scores['mean_F'] = np.nanmean(F_binwise, axis=axis)
    
    return scores


def compute_F(cum, hydro, dmo, k_bins, k_nyquist=None, mask_all_small=False, 
              threshold=0.03, min_denom=1e-10, k_min=None,
              cum_err=None, hydro_err=None, dmo_err=None):
    """Compute response fraction F = (Replace - DMO) / (Hydro - DMO) with uncertainties.
    
    Parameters
    ----------
    cum : ndarray, shape (n_planes, n_k) or (n_k,)
        Replace power spectrum
    hydro : ndarray, shape (n_planes, n_k) or (n_k,)
        Hydro power spectrum  
    dmo : ndarray, shape (n_planes, n_k) or (n_k,)
        DMO power spectrum
    k_bins : ndarray, shape (n_k,)
        Wavenumber or multipole bins
    k_nyquist : float, optional
        Nyquist frequency. If None, uses K_NYQUIST for P(k).
    mask_all_small : bool, optional
        If True, mask ALL k bins where baryonic effect < threshold.
        If False (default), only mask k < k_min (first k where effect > threshold).
    threshold : float, optional
        Fractional difference threshold (default 0.03 = 3%).
    min_denom : float, optional
        Minimum absolute value for denominator (hydro - dmo) to avoid division
        by near-zero. Default 1e-10. Set to None to use relative threshold only.
    k_min: float, optional
        Minimum k value to include
    cum_err : ndarray, optional
        Standard error on cum. Same shape as cum.
    hydro_err : ndarray, optional
        Standard error on hydro. Same shape as hydro.
    dmo_err : ndarray, optional
        Standard error on dmo. Same shape as dmo.
        
    Returns
    -------
    F : ndarray
        Response fraction with invalid k ranges masked as NaN
    F_err : ndarray or None
        Uncertainty on F. None if no errors provided.
    """
    numerator = cum - dmo
    denom = hydro - dmo
    F_raw = numerator / denom
    
    # Compute error if all three uncertainties are provided
    F_err = None
    if cum_err is not None and hydro_err is not None and dmo_err is not None:
        # Error propagation for numerator: σ(A-B) = sqrt(σ_A^2 + σ_B^2)
        numerator_err = np.sqrt(cum_err**2 + dmo_err**2)
        
        # Error propagation for denominator: σ(A-B) = sqrt(σ_A^2 + σ_B^2)
        denominator_err = np.sqrt(hydro_err**2 + dmo_err**2)
        
        # Error propagation for ratio: σ(A/B) = |A/B| * sqrt((σ_A/A)^2 + (σ_B/B)^2)
        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            F_err = np.abs(F_raw) * np.sqrt(
                (numerator_err / numerator)**2 + (denominator_err / denom)**2
            )
    
    # Use provided Nyquist or default to K_NYQUIST
    if k_nyquist is None:
        k_nyquist = K_NYQUIST
    k_nyquist_idx = np.abs(k_bins - k_nyquist).argmin()
    
    # Fractional difference: |(hydro - dmo) / dmo|
    frac_diff = np.abs(denom / dmo)  # shape (n_planes, n_k)
    
    n_planes, n_k = F_raw.shape
    k_indices = np.arange(n_k)
    
    # Start with Nyquist mask
    invalid_mask = k_indices[None, :] >= k_nyquist_idx
    
    # Add small denominator mask (prevents infinity from division by ~0)
    if min_denom is not None:
        invalid_mask = invalid_mask | (np.abs(denom) < min_denom)
    if k_min:
        invalid_mask = k_indices < k_min
        F_raw[:, invalid_mask] = np.nan
        if F_err is not None:
            F_err[:, invalid_mask] = np.nan
    if mask_all_small:
        # Mask ALL bins where effect < threshold
        invalid_mask = invalid_mask | (frac_diff < threshold)
    else:
        # Only mask k < k_min (first k where effect > threshold) per plane
        k_min_idx = np.argmax(frac_diff > threshold, axis=-1)  # shape (n_planes,)
        invalid_mask = invalid_mask | (k_indices[None, :] < k_min_idx[:, None])
    
    F_raw[invalid_mask] = np.nan
    if F_err is not None:
        F_err[invalid_mask] = np.nan
    
    if F_err is not None:
        return F_raw, F_err
    else:
        return F_raw


import numpy as np

def compute_statistic_response(cum_stats, disc_stats, base_stats, 
                                CUM_MODELS, DISC_MODELS, 
                                stat_name='pdf', 
                                compute_F_func=None,
                                mask_all_small=True, 
                                threshold=0.03, k_nyquist=1e6):
    """
    Compute mean, std, stderr, and F-statistic for any statistic across models.
    
    Parameters
    ----------
    cum_stats : dict
        Dictionary of cumulative model statistics
    disc_stats : dict
        Dictionary of discrete model statistics
    base_stats : dict
        Dictionary of baseline (hydro, dmo) statistics
    CUM_MODELS : list
        List of cumulative model names
    DISC_MODELS : list
        List of discrete model names
    stat_name : str
        Name of statistic to compute (e.g., 'pdf', 'peaks', 'minima', 'V0', 'V1', 'V2')
    compute_F_func : callable
        Function to compute F-statistic. Should have signature:
        compute_F(stat, hydro, dmo, k_bins, mask_all_small, threshold, 
                  cum_err, hydro_err, dmo_err)
    mask_all_small : bool
        Whether to mask small values in F computation
    threshold : float
        Threshold for masking
        
    Returns
    -------
    results : dict
        Dictionary containing:
        - 'cum_mean': mean statistic for cumulative models
        - 'cum_std': standard deviation for cumulative models
        - 'cum_stderr': standard error for cumulative models
        - 'cum_F': F-statistic for cumulative models
        - 'cum_F_err': F-statistic errors for cumulative models
        - 'disc_mean': mean statistic for discrete models
        - 'disc_std': standard deviation for discrete models
        - 'disc_stderr': standard error for discrete models
        - 'disc_F': F-statistic for discrete models
        - 'disc_F_err': F-statistic errors for discrete models
        - 'base_mean': mean statistic for baseline models
        - 'base_std': standard deviation for baseline models
        - 'base_stderr': standard error for baseline models
        - 'bins': bin centers (e.g., SN for pdf, k for Cl, etc.)
        - 'mean_cum_F': ν-averaged F for cumulative models
        - 'mean_disc_F': ν-averaged F for discrete models
    """
    
    # ==================== MEANS ====================
    cum_mean = {
        model_name: cum_stats[model_name][stat_name].mean(axis=0) 
        for model_name in CUM_MODELS 
        if cum_stats[model_name] is not None and stat_name in cum_stats[model_name]
    }
    
    disc_mean = {
        model_name: disc_stats[model_name][stat_name].mean(axis=0) 
        for model_name in DISC_MODELS 
        if disc_stats[model_name] is not None and stat_name in disc_stats[model_name]
    }
    
    base_mean = {
        model_name: base_stats[model_name][stat_name].mean(axis=0) 
        for model_name in base_stats 
        if base_stats[model_name] is not None and stat_name in base_stats[model_name]
    }
    
    # ==================== STANDARD DEVIATIONS ====================
    cum_std = {
        model_name: cum_stats[model_name][stat_name].std(axis=0) 
        for model_name in CUM_MODELS 
        if cum_stats[model_name] is not None and stat_name in cum_stats[model_name]
    }
    
    disc_std = {
        model_name: disc_stats[model_name][stat_name].std(axis=0) 
        for model_name in DISC_MODELS 
        if disc_stats[model_name] is not None and stat_name in disc_stats[model_name]
    }
    
    base_std = {
        model_name: base_stats[model_name][stat_name].std(axis=0) 
        for model_name in base_stats 
        if base_stats[model_name] is not None and stat_name in base_stats[model_name]
    }
    
    # ==================== STANDARD ERRORS ====================
    cum_stderr = {
        model_name: np.abs(cum_stats[model_name][stat_name].std(axis=0) / 
                    np.sqrt(cum_stats[model_name][stat_name].shape[0]))
        for model_name in CUM_MODELS 
        if cum_stats[model_name] is not None and stat_name in cum_stats[model_name]
    }
    
    disc_stderr = {
        model_name: np.abs(disc_stats[model_name][stat_name].std(axis=0) / 
                    np.sqrt(disc_stats[model_name][stat_name].shape[0]))
        for model_name in DISC_MODELS 
        if disc_stats[model_name] is not None and stat_name in disc_stats[model_name]
    }
    
    base_stderr = {
        model_name: np.abs(base_stats[model_name][stat_name].std(axis=0) / 
                    np.sqrt(base_stats[model_name][stat_name].shape[0]))
        for model_name in base_stats 
        if base_stats[model_name] is not None and stat_name in base_stats[model_name]
    }
    
    # ==================== BIN CENTERS ====================
    # Determine which bin key to use based on statistic
    bin_key_map = {
        'pdf': 'sn_bin_edges',
        'peaks': 'sn_bin_edges', 
        'minima': 'sn_bin_edges',
        'Cl': 'ell_bins',
        'Pk': 'k_bins',
        'V0': 'mf_thresholds',
        'V1': 'mf_thresholds',
        'V2': 'mf_thresholds',
        'S0': 'scale_bin_edges',
        'S1': 'scale_bin_edges',
    }
    
    bin_key = bin_key_map.get(stat_name, 'sn_bin_edges')  # Default to SN bins
    
    # Statistics where bins are directly the values (not edges to convert to centers)
    direct_bin_stats = ['Cl', 'Pk', 'V0', 'V1', 'V2']
    
    if bin_key in base_stats['hydro']:
        if stat_name not in direct_bin_stats:
            bin_edges = base_stats['hydro'][bin_key]
            bins = bin_edges[1:] - 0.5 * np.median(np.diff(bin_edges))  # bin centers
        else:
            bins = base_stats['hydro'][bin_key]
    else:
        # Fallback: try to infer from data shape
        bins = np.arange(base_mean['hydro'].shape[-1])
    
    # ==================== F-STATISTICS ====================
    cum_F = {}
    cum_F_err = {}
    disc_F = {}
    disc_F_err = {}
    
    if compute_F_func is not None:
        # Compute F for cumulative models
        for model_name in cum_mean:
            result = compute_F_func(
                cum_mean[model_name], 
                base_mean['hydro'], 
                base_mean['dmo'],
                k_bins=bins, 
                mask_all_small=mask_all_small, 
                threshold=threshold,
                cum_err=cum_stderr[model_name], 
                hydro_err=base_stderr['hydro'], 
                dmo_err=base_stderr['dmo'], 
                k_nyquist=k_nyquist
            )
            
            if isinstance(result, tuple):
                cum_F[model_name], cum_F_err[model_name] = result
            else:
                cum_F[model_name] = result
                cum_F_err[model_name] = np.zeros_like(result)  # No errors provided
        
        # Compute F for discrete models
        for model_name in disc_mean:
            result = compute_F_func(
                disc_mean[model_name], 
                base_mean['hydro'], 
                base_mean['dmo'],
                k_bins=bins, 
                mask_all_small=mask_all_small, 
                threshold=threshold,
                cum_err=disc_stderr[model_name], 
                hydro_err=base_stderr['hydro'], 
                dmo_err=base_stderr['dmo'],
                k_nyquist=k_nyquist

            )
            
            if isinstance(result, tuple):
                disc_F[model_name], disc_F_err[model_name] = result
            else:
                disc_F[model_name] = result
                disc_F_err[model_name] = np.zeros_like(result)
    
    # ==================== MEAN F (averaged over bins) ====================
    # Simple (unweighted) mean for backward compatibility
    mean_cum_F = {
        model_name: np.nanmean(cum_F[model_name], axis=1) 
        for model_name in cum_F
    } if cum_F else {}
    
    mean_disc_F = {
        model_name: np.nanmean(disc_F[model_name], axis=1) 
        for model_name in disc_F
    } if disc_F else {}
    
    # ==================== WEIGHTED MEAN F (inverse-variance weighting) ====================
    # This down-weights noisy bins (e.g., minima tails) appropriately
    weighted_mean_cum_F = {}
    weighted_mean_cum_F_err = {}
    for model_name in cum_F:
        if model_name in cum_F_err and cum_F_err[model_name] is not None:
            wmean, werr = compute_weighted_mean_F(cum_F[model_name], cum_F_err[model_name], axis=1)
            weighted_mean_cum_F[model_name] = wmean
            weighted_mean_cum_F_err[model_name] = werr
        else:
            # Fall back to unweighted mean if no errors available
            weighted_mean_cum_F[model_name] = np.nanmean(cum_F[model_name], axis=1)
            weighted_mean_cum_F_err[model_name] = np.full_like(weighted_mean_cum_F[model_name], np.nan)
    
    weighted_mean_disc_F = {}
    weighted_mean_disc_F_err = {}
    for model_name in disc_F:
        if model_name in disc_F_err and disc_F_err[model_name] is not None:
            wmean, werr = compute_weighted_mean_F(disc_F[model_name], disc_F_err[model_name], axis=1)
            weighted_mean_disc_F[model_name] = wmean
            weighted_mean_disc_F_err[model_name] = werr
        else:
            # Fall back to unweighted mean if no errors available
            weighted_mean_disc_F[model_name] = np.nanmean(disc_F[model_name], axis=1)
            weighted_mean_disc_F_err[model_name] = np.full_like(weighted_mean_disc_F[model_name], np.nan)
    
    # ==================== INTEGRATED SCORES ====================
    # These alternative metrics avoid bin-by-bin F issues
    cum_integrated_scores = {}
    disc_integrated_scores = {}
    
    for model_name in cum_mean:
        cum_integrated_scores[model_name] = compute_integrated_scores(
            R=cum_mean[model_name],
            H=base_mean['hydro'],
            D=base_mean['dmo'],
            R_err=cum_stderr.get(model_name),
            H_err=base_stderr.get('hydro'),
            D_err=base_stderr.get('dmo'),
            axis=1  # average over bins (axis 1), keep planes (axis 0)
        )
    
    for model_name in disc_mean:
        disc_integrated_scores[model_name] = compute_integrated_scores(
            R=disc_mean[model_name],
            H=base_mean['hydro'],
            D=base_mean['dmo'],
            R_err=disc_stderr.get(model_name),
            H_err=base_stderr.get('hydro'),
            D_err=base_stderr.get('dmo'),
            axis=1
        )
    
    # ==================== RETURN RESULTS ====================
    return {
        # Means
        'cum_mean': cum_mean,
        'disc_mean': disc_mean,
        'base_mean': base_mean,
        
        # Standard deviations
        'cum_std': cum_std,
        'disc_std': disc_std,
        'base_std': base_std,
        
        # Standard errors
        'cum_stderr': cum_stderr,
        'disc_stderr': disc_stderr,
        'base_stderr': base_stderr,
        
        # F-statistics
        'cum_F': cum_F,
        'cum_F_err': cum_F_err,
        'disc_F': disc_F,
        'disc_F_err': disc_F_err,
        
        # Mean F (unweighted - for backward compatibility)
        'mean_cum_F': mean_cum_F,
        'mean_disc_F': mean_disc_F,
        
        # Weighted mean F (inverse-variance weighting)
        'weighted_mean_cum_F': weighted_mean_cum_F,
        'weighted_mean_cum_F_err': weighted_mean_cum_F_err,
        'weighted_mean_disc_F': weighted_mean_disc_F,
        'weighted_mean_disc_F_err': weighted_mean_disc_F_err,
        
        # Integrated scores (alternative to mean F)
        'cum_integrated_scores': cum_integrated_scores,
        'disc_integrated_scores': disc_integrated_scores,
        
        # Bins
        'bins': bins,
        
        # Metadata
        'stat_name': stat_name
    }
def compute_all_statistics(cum_stats, disc_stats, base_stats,
                           CUM_MODELS, DISC_MODELS,
                           statistics=None,
                           compute_F_func=None,
                           mask_all_small=True,
                           threshold=0.03,
                           verbose=False):
    """
    Compute response for all statistics at once.
    
    Parameters
    ----------
    cum_stats : dict
        Dictionary of cumulative model statistics
    disc_stats : dict
        Dictionary of discrete model statistics
    base_stats : dict
        Dictionary of baseline (hydro, dmo) statistics
    CUM_MODELS : list
        List of cumulative model names
    DISC_MODELS : list
        List of discrete model names
    statistics : list of str, optional
        List of statistic names. If None, uses default list.
    compute_F_func : callable
        Function to compute F-statistic
    mask_all_small : bool
        Whether to mask small values in F computation
    threshold : float
        Threshold for masking
    verbose : bool
        Print progress and debugging info
    
    Returns
    -------
    results : dict
        Dictionary with keys = statistic names, values = result dicts from compute_statistic_response()
    """
    if statistics is None:
        statistics = ['Cl', 'pdf', 'peaks', 'minima', 'V0', 'V1', 'V2']
    
    results = {}
    
    for stat in statistics:
        if verbose:
            print(f"Computing {stat}...")
        
        try:
            results[stat] = compute_statistic_response(
                cum_stats, disc_stats, base_stats,
                CUM_MODELS, DISC_MODELS,
                stat_name=stat,
                compute_F_func=compute_F_func,
                mask_all_small=mask_all_small,
                threshold=threshold,
            )
        except Exception as e:
            print(f"  ERROR: Failed to compute {stat}: {e}")
            if verbose:
                import traceback
                traceback.print_exc()
            continue
    
    return results


############ PLOTS ######################
def plot_model_response(results, stat_name='pdf', z_idx=23, z_val=1.0, 
                       model_type='cumulative', savename=None):
    """
    Unified plotting function for both cumulative and discrete models.
    
    Parameters
    ----------
    results : dict
        Output from compute_statistic_response()
    stat_name : str
        Name of statistic (for labels)
    z_idx : int
        Redshift index
    z_val : float
        Redshift value
    model_type : str
        Either 'cumulative' or 'discrete'
    savename : str, optional
        Save filename. If None, uses default based on stat_name and model_type
    """
    if model_type == 'cumulative':
        plot_cumulative_response(results, stat_name, z_idx, z_val, savename)
    elif model_type == 'discrete':
        plot_discrete_response(results, stat_name, z_idx, z_val, savename)
    else:
        raise ValueError(f"model_type must be 'cumulative' or 'discrete', got {model_type}")

def plot_cumulative_response(results, stat_name='pdf', z_idx=23, z_val=1.0, savename=None):
    """Plot cumulative model response (your previous function)."""
    mass_labels = ['1.00e12', '3.16e12', '1.00e13', '3.16e13']
    radius_labels = ['0.5', '1.0', '3.0', '5.0']
    colors = plt.cm.jet(np.linspace(0.0, 1.0, 4))
    
    cum_mean = results['cum_mean']
    cum_stderr = results['cum_stderr']
    cum_F = results['cum_F']
    cum_F_err = results['cum_F_err']
    base_mean = results['base_mean']
    base_stderr = results['base_stderr']
    bins = results['bins']
    
    fig, axes = plt.subplots(2, 4, figsize=(24, 12), sharex=True, gridspec_kw={'hspace':0.05, 'wspace':0.05})
    
    hydro_data = base_mean['hydro']
    dmo_data = base_mean['dmo']
    hydro_err = base_stderr['hydro']
    dmo_err = base_stderr['dmo']
    
    # Set errorevery based on statistic type
    if stat_name == 'Pk':
        errorevery = 10000  # Pk has ~3000 bins
    elif stat_name == 'Cl':
        errorevery = 10  # Cl has ~700 bins
    else:
        errorevery = 1

    # Top row: Ratios
    for i, radius in enumerate(radius_labels):
        ax = axes[0, i]
        
        ratio = hydro_data[z_idx] / dmo_data[z_idx]
        ratio_err = ratio * np.sqrt((hydro_err[z_idx]/hydro_data[z_idx])**2 + 
                                     (dmo_err[z_idx]/dmo_data[z_idx])**2)
        # ratio = np.where(bins >= -1.5, ratio, np.nan)
        # ratio_err = np.where(bins >= -1.5, ratio_err, np.nan)
        
        ax.errorbar(bins, ratio, yerr=np.abs(ratio_err), color='black', 
                   label='Hydro' if i==0 else None, ls='--', 
                   capsize=2, alpha=0.6, errorevery=errorevery)
        
        if i > 0:
            ax.tick_params(labelleft=False)
            
        for j, mass in enumerate(mass_labels):
            model_name = f"hydro_replace_Ml_{mass}_Mu_1.00e15_Ri_0.0_Ro_{radius}"
            if model_name in cum_mean:
                P = cum_mean[model_name]
                P_err = cum_stderr[model_name]
                
                ratio = P[z_idx] / dmo_data[z_idx]
                ratio_err = ratio * np.sqrt((P_err[z_idx]/P[z_idx])**2 + 
                                           (dmo_err[z_idx]/dmo_data[z_idx])**2)
                
                # ratio = np.where(bins >= -5, ratio, np.nan)
                # ratio_err = np.where(bins >= -5, ratio_err, np.nan)
                
                ax.errorbar(bins, ratio, yerr=np.abs(ratio_err), color=colors[j], 
                           label=rf'$M>10^{{{np.log10(float(mass)):.1f}}}$',
                           capsize=2, alpha=0.6, errorevery=errorevery)
                ax.grid(alpha=0.3)

                
                ax.set_ylim(0.5, 1.5)
                ax.set_title(rf'$\alpha = {radius}$')
                ax.grid(alpha=0.3)


    # Bottom row: F-statistic
    for i, radius in enumerate(radius_labels):
        ax = axes[1, i]
        if i > 0:
            ax.tick_params(labelleft=False)
            
        for j, mass in enumerate(mass_labels):
            model_name = f"hydro_replace_Ml_{mass}_Mu_1.00e15_Ri_0.0_Ro_{radius}"
            if model_name in cum_F:
                F = cum_F[model_name]
                F_err = cum_F_err[model_name]
                
                ax.errorbar(bins, F[z_idx], yerr=np.abs(F_err[z_idx]), 
                           color=colors[j],
                           label=rf'$M>10^{{{np.log10(float(mass)):.1f}}}$',
                           capsize=2, alpha=0.6, errorevery=errorevery)
                ax.grid(alpha=0.3)

        
        if stat_name in ['pdf', 'peaks', 'minima', 'V0', 'V1', 'V2']:
            ax.set_xlabel(r'$\nu$ (S/N)')
            ax.set_xlim(-5, 10)
        elif stat_name == 'Cl':
            ax.set_xlabel(r'$\ell$')
            ax.set_xscale('log')
            ax.set_xlim(1e3, 3e5)
        elif stat_name == 'Pk':
            ax.set_xlabel(r'$k\,[h\,{\rm Mpc}^{-1}]$')
            ax.set_xscale('log')
            ax.set_xlim(0.1, 63)
        else:
            ax.set_xlabel('Scale')
            
        ax.set_ylim(-0.5, 1.5)
        ax.axhline(0, color='k', ls='--', lw=1.)
        ax.axhline(1, color='k', ls='--', lw=1.)

    axes[1, 0].set_ylabel(rf'$F_{{\rm {stat_name}}}(\nu)$')
    axes[0, 0].set_ylabel(f'{stat_name} / {stat_name}' + r'$_{\rm dmo}$')
    axes[0, 0].legend(loc='upper left', ncols=2, title=r'$M_{\rm min}\,[h^{-1}\,M_\odot]$')
    
    # plt.tight_layout()
    
    if savename is None:
        savename = f'figures/{stat_name}_cumulative_by_radius.pdf'
    plt.savefig(savename)
    plt.show()

def plot_discrete_response(results, stat_name='pdf', z_idx=23, z_val=1.0, 
                          savename=None):
    """
    Plot discrete (tile-based) statistic response: ratio and F-statistic by radius and mass bins.
    
    Parameters
    ----------
    results : dict
        Output from compute_statistic_response()
    stat_name : str
        Name of statistic (for labels)
    z_idx : int
        Redshift index
    z_val : float
        Redshift value
    savename : str, optional
        Save filename. If None, uses default based on stat_name
    """
    # Discrete model mass/radius combinations
    disc_mass_lower = ['1.00e12', '3.16e12', '1.00e13', '3.16e13']
    disc_mass_upper = ['3.16e12', '1.00e13', '3.16e13', '1.00e15']
    disc_radius_inner = ['0.0', '0.5', '1.0', '3.0']
    disc_radius_outer = ['0.5', '1.0', '3.0', '5.0']
    
    mass_bin_labels = [r'$M\in [10^{12}, 10^{12.5}]$', r'$M\in [10^{12.5}, 10^{13}]$', 
                        r'$M\in [10^{13}, 10^{13.5}]$', r'$M\in [10^{13.5}, \infty]$']
    radius_bin_labels = ['[0, 0.5]', '[0.5, 1.0]', '[1.0, 3.0]', '[3.0, 5.0]']
    colors = plt.cm.jet(np.linspace(0.0, 1.0, 4))
    
    # Extract data from results
    disc_mean = results['disc_mean']
    disc_stderr = results['disc_stderr']
    disc_F = results['disc_F']
    disc_F_err = results['disc_F_err']
    base_mean = results['base_mean']
    base_stderr = results['base_stderr']
    bins = results['bins']
    
    fig, axes = plt.subplots(2, 4, figsize=(24, 12), sharex=True,gridspec_kw={'hspace':0.05, 'wspace':0.05})
    
    hydro_data = base_mean['hydro']
    dmo_data = base_mean['dmo']
    hydro_err = base_stderr['hydro']
    dmo_err = base_stderr['dmo']
    
    # Set errorevery based on statistic type
    if stat_name == 'Pk':
        errorevery = 10000  # Pk has ~3000 bins
    elif stat_name == 'Cl':
        errorevery = 10  # Cl has ~700 bins
    else:
        errorevery = 1

    # Top row: Ratios (one subplot per radial bin)
    for i, (Ri, Ro) in enumerate(zip(disc_radius_inner, disc_radius_outer)):
        ax = axes[0, i]
        
        # Hydro/DMO ratio with error propagation
        ratio = hydro_data[z_idx] / dmo_data[z_idx]
        ratio_err = ratio * np.sqrt((hydro_err[z_idx]/hydro_data[z_idx])**2 + 
                                     (dmo_err[z_idx]/dmo_data[z_idx])**2)
        # ratio = np.where(bins >= -5, ratio, np.nan)
        # ratio_err = np.where(bins >= -5, ratio_err, np.nan)
        
        ax.errorbar(bins, ratio, yerr=np.abs(ratio_err), color='black', 
                   label='Hydro' if i==0 else None, ls='--', 
                   capsize=2, alpha=0.6, errorevery=errorevery)
        
        if i > 0:
            ax.tick_params(labelleft=False)
        
        # Plot each mass bin as a different line
        for j, (Ml, Mu) in enumerate(zip(disc_mass_lower, disc_mass_upper)):
            model_name = f"hydro_replace_Ml_{Ml}_Mu_{Mu}_Ri_{Ri}_Ro_{Ro}"
            if model_name in disc_mean:
                P = disc_mean[model_name]
                P_err = disc_stderr[model_name]
                
                ratio = P[z_idx] / dmo_data[z_idx]
                ratio_err = ratio * np.sqrt((P_err[z_idx]/P[z_idx])**2 + 
                                           (dmo_err[z_idx]/dmo_data[z_idx])**2)
                
                # ratio = np.where(bins >= -5, ratio, np.nan)
                # ratio_err = np.where(bins >= -5, ratio_err, np.nan)
                
                ax.errorbar(bins, ratio, yerr=np.abs(ratio_err), color=colors[j], 
                           label=mass_bin_labels[j],
                           capsize=2, alpha=0.6, errorevery=errorevery)
                ax.grid(alpha=0.3)

                
                ax.set_ylim(0.5, 1.5)
                ax.set_title(rf'$r \in [{Ri}, {Ro}] R_{{200}}$')
                ax.grid(alpha=0.3)


    # Bottom row: F-statistic with errors
    for i, (Ri, Ro) in enumerate(zip(disc_radius_inner, disc_radius_outer)):
        ax = axes[1, i]
        if i > 0:
            ax.tick_params(labelleft=False)
        
        for j, (Ml, Mu) in enumerate(zip(disc_mass_lower, disc_mass_upper)):
            model_name = f"hydro_replace_Ml_{Ml}_Mu_{Mu}_Ri_{Ri}_Ro_{Ro}"
            if model_name in disc_F:
                F = disc_F[model_name]
                F_err = disc_F_err[model_name]
                
                ax.errorbar(bins, F[z_idx], yerr=np.abs(F_err[z_idx]), 
                           color=colors[j],
                           label=mass_bin_labels[j],
                           capsize=2, alpha=0.6, errorevery=errorevery)
                ax.grid(alpha=0.3)

        
        # Adjust x-axis label based on statistic type
        if stat_name in ['pdf', 'peaks', 'minima', 'V0', 'V1', 'V2']:
            ax.set_xlabel(r'$\nu$ (S/N)')
            ax.set_xlim(-5, 10)
        elif stat_name == 'Cl':
            ax.set_xlabel(r'$\ell$')
            ax.set_xscale('log')
            ax.set_xlim(1e3, 3e5)
        elif stat_name == 'Pk':
            ax.set_xlabel(r'$k\,[h\,{\rm Mpc}^{-1}]$')
            ax.set_xscale('log')
            ax.set_xlim(0.1, 63)
        else:
            ax.set_xlabel('Scale')
        
        ax.set_ylim(-0.5, 1.5)
        ax.axhline(0, color='k', ls='--', lw=1.)
        ax.axhline(1, color='k', ls='--', lw=1.)

    axes[1, 0].set_ylabel(rf'$F_{{\rm {stat_name}}}(\nu)$')
    axes[0, 0].set_ylabel(f'{stat_name} / {stat_name}' + r'$_{\rm dmo}$')
    axes[1, 1].legend(loc='upper center', ncols=1, title=r'Mass bin [$h^{-1}M_\odot$]')
    
    plt.tight_layout()
    
    if savename is None:
        savename = f'figures/{stat_name}_discrete_by_radius.pdf'
    plt.savefig(savename)
    plt.show()

def compute_discrete_sum(Ml_cum, Ro_cum, disc_F_dict, z_idx, bins):
    """
    Sum discrete tiles that contribute to a cumulative model.
    
    Parameters
    ----------
    Ml_cum : str or float
        Lower mass threshold for cumulative model
    Ro_cum : str or float
        Outer radius threshold for cumulative model
    disc_F_dict : dict
        Dictionary of discrete F-statistics
    z_idx : int
        Redshift index
    bins : array
        Bin centers (e.g., SN, k, etc.)
    
    Returns
    -------
    disc_sum : array
        Sum of discrete F contributions
    """
    disc_mass_lower = ['1.00e12', '3.16e12', '1.00e13', '3.16e13']
    disc_mass_upper = ['3.16e12', '1.00e13', '3.16e13', '1.00e15']
    disc_radius_inner = ['0.0', '0.5', '1.0', '3.0']
    disc_radius_outer = ['0.5', '1.0', '3.0', '5.0']
    
    Ml_cum_val = float(Ml_cum)
    Ro_cum_val = float(Ro_cum)
    
    disc_sum = np.zeros_like(bins)
    
    for Ml, Mu in zip(disc_mass_lower, disc_mass_upper):
        for Ri, Ro in zip(disc_radius_inner, disc_radius_outer):
            Ml_val = float(Ml)
            Mu_val = float(Mu)
            Ri_val = float(Ri)
            Ro_val = float(Ro)
            
            # Check if this tile contributes to cumulative model
            # Mass: tile overlaps with M > Ml_cum if Mu > Ml_cum
            # Radius: tile overlaps with R < Ro_cum if Ri < Ro_cum
            if Mu_val > Ml_cum_val and Ri_val < Ro_cum_val:
                model_name = f"hydro_replace_Ml_{Ml}_Mu_{Mu}_Ri_{Ri}_Ro_{Ro}"
                if model_name in disc_F_dict:
                    disc_sum += disc_F_dict[model_name][z_idx]
    
    return disc_sum


def plot_cumulative_discrete_residuals(results, stat_name='pdf', z_idx=23, z_val=1.0, 
                                       savename=None):
    """
    Plot residuals between cumulative and discrete tile sum: ε = F_cum - Σ(ΔF_disc).
    
    This validates whether discrete tiles properly sum to cumulative models.
    
    Parameters
    ----------
    results : dict
        Output from compute_statistic_response()
    stat_name : str
        Name of statistic (for labels)
    z_idx : int
        Redshift index
    z_val : float
        Redshift value
    savename : str, optional
        Save filename. If None, uses default based on stat_name
    """
    mass_labels = ['1.00e12', '3.16e12', '1.00e13', '3.16e13']
    radius_labels = ['0.5', '1.0', '3.0', '5.0']
    colors = plt.cm.jet(np.linspace(0.0, 1.0, 4))
    
    # Extract data from results
    cum_F = results['cum_F']
    disc_F = results['disc_F']
    bins = results['bins']
    
    fig, axes = plt.subplots(2, 4, figsize=(24, 12), sharex=True,gridspec_kw={'hspace':0.05, 'wspace':0.05})
    
    # Top row: Residuals ε = F_cum - Σ(ΔF_disc)
    for i, radius in enumerate(radius_labels):
        ax = axes[0, i]
        
        for j, mass in enumerate(mass_labels):
            cum_model_name = f"hydro_replace_Ml_{mass}_Mu_1.00e15_Ri_0.0_Ro_{radius}"
            
            if cum_model_name in cum_F:
                F_cum = cum_F[cum_model_name][z_idx]
                F_disc_sum = compute_discrete_sum(mass, radius, disc_F, z_idx, bins)
                
                # Residual
                residual = F_cum - F_disc_sum
                
                ax.plot(bins, residual, color=colors[j], 
                       label=rf'$M > 10^{{{np.log10(float(mass)):.1f}}}$',
                       lw=2, alpha=0.8)
        
        ax.axhline(1, color='k', ls='--', lw=1.)
        ax.set_ylim(-0.5, 0.5)
        ax.set_title(rf'$\alpha = {radius}$ $R_{{200}}$')
        ax.grid(alpha=0.3)
        
        if i == 0:
            ax.set_ylabel(r'$\epsilon(\nu) = F_{\rm cum} - \sum \Delta F_{\rm disc}$', 
                         )
            ax.legend(loc='best', ncols=2, title=r'$M_{\rm min}\,[h^{-1}\,M_\odot]$')
        else:
            ax.tick_params(labelleft=False)
    
    # Bottom row: Show F_cum and Σ(ΔF_disc) overlaid
    for i, radius in enumerate(radius_labels):
        ax = axes[1, i]
        
        for j, mass in enumerate(mass_labels):
            cum_model_name = f"hydro_replace_Ml_{mass}_Mu_1.00e15_Ri_0.0_Ro_{radius}"
            
            if cum_model_name in cum_F:
                F_cum = cum_F[cum_model_name][z_idx]
                F_disc_sum = compute_discrete_sum(mass, radius, disc_F, z_idx, bins)
                
                # Plot both
                ax.plot(bins, F_cum, color=colors[j], ls='-', lw=2,
                       label=rf'$F_{{\rm cum}}$' if j == 0 else None,
                       alpha=0.8)
                ax.plot(bins, F_disc_sum, color=colors[j], ls=':', lw=2.5,
                       label=rf'$\sum \Delta F_{{\rm disc}}$' if j == 0 else None,
                       alpha=0.8)
        
        ax.axhline(0, color='k', ls='--', lw=1., alpha=0.5)
        ax.axhline(1, color='k', ls='--', lw=1., alpha=0.5)
        ax.set_ylim(-0.5, 1.5)
        
        # Adjust x-axis based on statistic type
        if stat_name in ['pdf', 'peaks', 'minima', 'V0', 'V1', 'V2']:
            ax.set_xlabel(r'$\nu$ (S/N)')
            ax.set_xlim(-5, 10)
        elif stat_name == 'Cl':
            ax.set_xlabel(r'$\ell$')
            ax.set_xscale('log')
            ax.set_xlim(1e3, 3e5)
        elif stat_name == 'Pk':
            ax.set_xlabel(r'$k\,[h\,{\rm Mpc}^{-1}]$')
            ax.set_xscale('log')
            ax.set_xlim(0.1, 60)
        else:
            ax.set_xlabel('Scale')
        
        ax.grid(alpha=0.3)
        
        if i == 0:
            ax.set_ylabel(r'$F(\nu)$')
            ax.legend(loc='best')
        else:
            ax.tick_params(labelleft=False)
    

    plt.tight_layout()
    
    if savename is None:
        savename = f'figures/{stat_name}_cumulative_vs_discrete_residuals.pdf'
    plt.savefig(savename, dpi=150)
    plt.show()


def plot_cumulative_discrete_residuals_with_stats(results, stat_name='pdf', 
                                                  z_idx=23, z_val=1.0, 
                                                  savename=None):
    """
    Plot residuals with quantitative statistics (RMS, max absolute error).
    """
    mass_labels = ['1.00e12', '3.16e12', '1.00e13', '3.16e13']
    radius_labels = ['0.5', '1.0', '3.0', '5.0']
    colors = plt.cm.jet(np.linspace(0.0, 1.0, 4))
    
    cum_F = results['cum_F']
    disc_F = results['disc_F']
    bins = results['bins']
    
    fig, axes = plt.subplots(2, 4, figsize=(24, 12), sharex=True,gridspec_kw={'hspace':0.05, 'wspace':0.05})
    
    # Store residual statistics
    residual_stats = {}
    
    # Top row: Residuals
    for i, radius in enumerate(radius_labels):
        ax = axes[0, i]
        
        for j, mass in enumerate(mass_labels):
            cum_model_name = f"hydro_replace_Ml_{mass}_Mu_1.00e15_Ri_0.0_Ro_{radius}"
            
            if cum_model_name in cum_F:
                F_cum = cum_F[cum_model_name][z_idx]
                F_disc_sum = compute_discrete_sum(mass, radius, disc_F, z_idx, bins)
                
                residual = F_cum - F_disc_sum
                
                # Compute statistics
                rms = np.sqrt(np.nanmean(residual**2))
                max_abs = np.nanmax(np.abs(residual))
                residual_stats[cum_model_name] = {'rms': rms, 'max_abs': max_abs}
                
                ax.plot(bins, residual, color=colors[j], 
                       label=rf'$M > 10^{{{np.log10(float(mass)):.1f}}}$ (RMS={rms:.3f})',
                       lw=2, alpha=0.8)
        
        ax.axhline(0, color='gray', ls='--', lw=1)
        ax.set_ylim(-0.5, 0.5)
        ax.set_title(rf'$\alpha = {radius}$ $R_{{200}}$')
        ax.grid(alpha=0.3)
        
        if i == 0:
            ax.set_ylabel(r'$\epsilon(\nu) = F_{\rm cum} - \sum \Delta F_{\rm disc}$', 
                         )
            ax.legend(loc='best', ncols=2, title=r'$M_{\rm min}\,[h^{-1}\,M_\odot]$')
        else:
            ax.tick_params(labelleft=False)
    
    # Bottom row: Overlaid F_cum and sum
    for i, radius in enumerate(radius_labels):
        ax = axes[1, i]
        
        for j, mass in enumerate(mass_labels):
            cum_model_name = f"hydro_replace_Ml_{mass}_Mu_1.00e15_Ri_0.0_Ro_{radius}"
            
            if cum_model_name in cum_F:
                F_cum = cum_F[cum_model_name][z_idx]
                F_disc_sum = compute_discrete_sum(mass, radius, disc_F, z_idx, bins)
                
                ax.plot(bins, F_cum, color=colors[j], ls='-', lw=2,
                       label=rf'$F_{{\rm cum}}$' if j == 0 else None,
                       alpha=0.8)
                ax.plot(bins, F_disc_sum, color=colors[j], ls=':', lw=2.5,
                       label=rf'$\sum \Delta F_{{\rm disc}}$' if j == 0 else None,
                       alpha=0.8)
        
        ax.axhline(0, color='k', ls='--', lw=1.)
        ax.axhline(1, color='k', ls='--', lw=1.)
        ax.set_ylim(-0.5, 1.5)
        
        if stat_name in ['pdf', 'peaks', 'minima', 'V0', 'V1', 'V2']:
            ax.set_xlabel(r'$\nu$ (S/N)')
            ax.set_xlim(-5, 10)
        elif stat_name == 'Cl':
            ax.set_xlabel(r'$\ell$')
            ax.set_xscale('log')
            ax.set_xlim(1e3, 3e5)
        elif stat_name == 'Pk':
            ax.set_xlabel(r'$k\,[h\,{\rm Mpc}^{-1}]$')
            ax.set_xscale('log')
            ax.set_xlim(0.1, 63)
        else:
            ax.set_xlabel('Scale')
        
        ax.grid(alpha=0.3)
        
        if i == 0:
            ax.set_ylabel(r'$F(\nu)$')
            ax.legend(loc='best', ncols=2, title=r'$M_{\rm min}\,[h^{-1}\,M_\odot]$')
        else:
            ax.tick_params(labelleft=False)
    
    # Print summary statistics
    print(f"\n{stat_name.upper()} Residual Statistics at z={z_val}:")
    print("=" * 60)
    for model_name, stats in residual_stats.items():
        print(f"{model_name:50s}: RMS={stats['rms']:.4f}, Max|ε|={stats['max_abs']:.4f}")

    plt.tight_layout()
    
    if savename is None:
        savename = f'figures/{stat_name}_cumulative_vs_discrete_residuals.pdf'
    plt.savefig(savename, dpi=150)
    plt.show()
    
    return residual_stats

def compute_residual_evolution(results, stat_name='pdf'):
    """
    Compute RMS residuals across all redshifts.
    
    Returns
    -------
    residual_rms : dict
        Dictionary with model names as keys, arrays of RMS vs redshift as values
    """
    mass_labels = ['1.00e12', '3.16e12', '1.00e13', '3.16e13']
    radius_labels = ['0.5', '1.0', '3.0', '5.0']
    
    cum_F = results['cum_F']
    disc_F = results['disc_F']
    bins = results['bins']
    
    # Determine number of redshifts
    first_model = list(cum_F.keys())[0]
    n_z = cum_F[first_model].shape[0]
    
    residual_rms = {}
    
    for radius in radius_labels:
        for mass in mass_labels:
            cum_model_name = f"hydro_replace_Ml_{mass}_Mu_1.00e15_Ri_0.0_Ro_{radius}"
            
            if cum_model_name in cum_F:
                rms_vs_z = []
                
                for z_idx in range(n_z):
                    F_cum = cum_F[cum_model_name][z_idx]
                    F_disc_sum = compute_discrete_sum(mass, radius, disc_F, z_idx, bins)
                    
                    residual = F_cum - F_disc_sum
                    rms = np.sqrt(np.nanmean(residual**2))
                    rms_vs_z.append(rms)
                
                residual_rms[cum_model_name] = np.array(rms_vs_z)
    
    return residual_rms


def plot_residual_evolution(residual_rms, stat_name='pdf', 
                            redshifts=None, savename=None):
    """
    Plot RMS residual evolution with redshift.
    """
    if redshifts is None:
        redshifts = np.arange(len(list(residual_rms.values())[0]))
    
    fig, ax = plt.subplots(figsize=(24, 12),gridspec_kw={'hspace':0.05, 'wspace':0.05})
    
    for model_name, rms_array in residual_rms.items():
        ax.plot(redshifts, rms_array, '-', label=model_name, alpha=0.7)
    
    ax.set_xlabel('Redshift')
    ax.set_ylabel(r'RMS($\epsilon$)')
    ax.set_title(f'{stat_name.upper()}: Residual RMS vs Redshift', weight='bold')
    ax.grid(alpha=0.3)
    ax.legend(loc='best', ncols=2)
    
    plt.tight_layout()
    
    if savename is None:
        savename = f'figures/{stat_name}_residual_evolution.pdf'
    plt.savefig(savename, dpi=150)
    plt.show()

# Usage


def plot_redshift_evolution(results, stat_name='pdf', model_type='cumulative',
                           kappa_redshifts=None, savename=None, weighting='frac_signal'):
    """
    Plot mean F-statistic evolution with redshift.
    
    Parameters
    ----------
    results : dict
        Output from compute_statistic_response()
    stat_name : str
        Name of statistic (for labels)
    model_type : str
        Either 'cumulative' or 'discrete'
    kappa_redshifts : dict, optional
        Dictionary mapping indices to redshift values
    savename : str, optional
        Save filename
    weighting : str
        Weighting method for averaging F over bins:
        - 'snr': Use snr_weighted_F (fractional signal × inverse variance).
        - 'frac_signal' (default): Use frac_signal_weighted_F from integrated_scores.
        - 'signal': Use signal_weighted_F.
        - 'inverse_variance': Use weighted_mean_cum_F.
        - 'uniform': Use mean_cum_F (simple unweighted mean).
    """
    # Default redshift mapping
    if kappa_redshifts is None:
        kappa_redshifts = {
            1: 0.034, 2: 0.070, 3: 0.105, 4: 0.142, 5: 0.179,
            6: 0.216, 7: 0.255, 8: 0.294, 9: 0.335, 10: 0.376,
            11: 0.418, 12: 0.462, 13: 0.506, 14: 0.552, 15: 0.599,
            16: 0.648, 17: 0.698, 18: 0.749, 19: 0.803, 20: 0.858,
            21: 0.914, 22: 0.973, 23: 1.034, 24: 1.097, 25: 1.163,
            26: 1.231, 27: 1.302, 28: 1.375, 29: 1.452, 30: 1.532,
            31: 1.615, 32: 1.703, 33: 1.794, 34: 1.889, 35: 1.989,
            36: 2.094, 37: 2.203, 38: 2.319, 39: 2.440, 40: 2.568
        }
    
    redshifts_all = list(kappa_redshifts.values())
    z_indices_all = np.arange(len(redshifts_all))
    
    colors = plt.cm.jet(np.linspace(0.0, 1.0, 4))
    
    if model_type == 'cumulative':
        _plot_cumulative_redshift_evolution(
            results, stat_name, redshifts_all, z_indices_all, colors, savename, weighting
        )
    elif model_type == 'discrete':
        _plot_discrete_redshift_evolution(
            results, stat_name, redshifts_all, z_indices_all, colors, savename, weighting
        )
    else:
        raise ValueError(f"model_type must be 'cumulative' or 'discrete', got {model_type}")


def _plot_cumulative_redshift_evolution(results, stat_name, redshifts_all, 
                                        z_indices_all, colors, savename, weighting='frac_signal'):
    """Plot cumulative model redshift evolution using specified weighting."""
    mass_labels = ['1.00e12', '3.16e12', '1.00e13', '3.16e13']
    radius_labels = ['0.5', '1.0', '3.0', '5.0']
    mass_display = [r'$M > 10^{12}$', r'$M > 10^{12.5}$', 
                    r'$M > 10^{13}$', r'$M > 10^{13.5}$']
    radius_display = [r'$0.5$', r'$1.0$', r'$3.0$', r'$5.0$']
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharex=True, sharey=True,gridspec_kw={'hspace':0.05, 'wspace':0.05})
    
    # F vs z for each radius (lines = mass)
    for i, radius in enumerate(radius_labels):
        ax = axes[i]
        
        for j, mass in enumerate(mass_labels):
            model_name = f"hydro_replace_Ml_{mass}_Mu_1.00e15_Ri_0.0_Ro_{radius}"
            
            # Collect F values across all redshifts using helper
            F_vs_z = []
            F_err_vs_z = []
            
            for z_idx in z_indices_all:
                F_val, F_err = _get_F_value(results, model_name, z_idx, weighting, 'cum')
                F_vs_z.append(F_val)
                F_err_vs_z.append(F_err if not np.isnan(F_err) else 0)
            
            # Plot with errorbars (only if errors available)
            if any(e > 0 for e in F_err_vs_z):
                ax.errorbar(redshifts_all, F_vs_z, yerr=np.abs(F_err_vs_z), 
                           fmt='-', color=colors[j], 
                           label=mass_display[j], linewidth=2, markersize=6,
                           capsize=3, alpha=0.7)
            else:
                ax.plot(redshifts_all, F_vs_z, '-', color=colors[j], 
                       label=mass_display[j], linewidth=2, markersize=6, alpha=0.7)
        
        ax.set_xscale('log')
        ax.set_title(r'$\alpha = $' + radius_display[i] + r' $R_{200}$')
        ax.set_xlabel(r'$z$')
        ax.axhline(1, color='k', ls='--', lw=1.)
        ax.axhline(0, color='k', ls='--', lw=1.)
        ax.set_ylim(-0.2, 1.5)
        ax.grid(alpha=0.3)
        
        if i == 0:
            ax.set_ylabel(rf'$\langle F \rangle^{{\rm signal}}_{{\rm {stat_name}}}$')
            ax.legend(loc='best', ncols=2, 
                     title=r'$M_{\rm min}\,[h^{-1}\,M_\odot]$')

    plt.tight_layout()
    
    if savename is None:
        savename = f'figures/F{stat_name}_cumulative_vs_redshift_{weighting}.pdf'
    plt.savefig(savename, dpi=150)
    plt.show()


def _plot_discrete_redshift_evolution(results, stat_name, redshifts_all, 
                                      z_indices_all, colors, savename, weighting='frac_signal'):
    """Plot discrete model redshift evolution using specified weighting."""
    disc_mass_lower = ['1.00e12', '3.16e12', '1.00e13', '3.16e13']
    disc_mass_upper = ['3.16e12', '1.00e13', '3.16e13', '1.00e15']
    disc_radius_inner = ['0.0', '0.5', '1.0', '3.0']
    disc_radius_outer = ['0.5', '1.0', '3.0', '5.0']
    
    mass_bin_labels = [r'$M\in [10^{12}, 10^{12.5}]$', r'$M\in [10^{12.5}, 10^{13}]$', 
                       r'$M\in [10^{13}, 10^{13.5}]$', r'$M\in [10^{13.5}, 10^{15}]$']
    radius_bin_display = [r'[0, 0.5]', r'[0.5, 1]', r'[1, 3]', r'[3, 5]']
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharex=True, sharey=True,gridspec_kw={'hspace':0.05, 'wspace':0.05})
    
    # F vs z for each radial bin (lines = mass bins)
    for i, (Ri, Ro) in enumerate(zip(disc_radius_inner, disc_radius_outer)):
        ax = axes[i]
        
        for j, (Ml, Mu) in enumerate(zip(disc_mass_lower, disc_mass_upper)):
            model_name = f"hydro_replace_Ml_{Ml}_Mu_{Mu}_Ri_{Ri}_Ro_{Ro}"
            
            # Collect F and errors across all redshifts using helper function
            F_vs_z = []
            F_err_vs_z = []
            
            for z_idx in z_indices_all:
                F_val, F_err = _get_F_value(results, model_name, z_idx, weighting, 'disc')
                F_vs_z.append(F_val)
                F_err_vs_z.append(F_err if not np.isnan(F_err) else 0)
            
            # Plot with errorbars
            ax.errorbar(redshifts_all, F_vs_z, yerr=np.abs(F_err_vs_z), 
                       fmt='-', color=colors[j], 
                       label=mass_bin_labels[j], linewidth=2, markersize=6,
                       capsize=3, alpha=0.7)
        
        ax.set_xscale('log')
        ax.set_title(r'$r \in $' + radius_bin_display[i] + r' $R_{200}$')
        ax.set_xlabel(r'$z$')
        ax.axhline(1, color='k', ls='--', lw=1.)
        ax.axhline(0, color='k', ls='--', lw=1.)
        ax.set_ylim(-0.2, 1.5)
        ax.grid(alpha=0.3)
        
        if i == 0:
            ax.set_ylabel(rf'$\langle F\rangle^{{\rm signal}}_{{\rm {stat_name}}}$')
            ax.legend(loc='best', 
                     title=r'Mass bin [$h^{-1}M_\odot$]')
    

    plt.tight_layout()
    
    if savename is None:
        savename = f'figures/F{stat_name}_discrete_vs_redshift_{weighting}.pdf'
    plt.savefig(savename, dpi=150)
    plt.show()


def plot_single_stat_heatmap(results, stat_name='Pk', z_idx=23, z_val=1.0,
                             vmin=0.0, vmax=1.0, cmap='Reds',cmap2='bwr',
                             savename=None, weighting='frac_signal', show_errors=False):
    """
    Plot cumulative and discrete heatmaps side-by-side for a single statistic.
    
    Parameters
    ----------
    results : dict
        Results dictionary from compute_statistic_response() for a single statistic
    stat_name : str
        Name of the statistic (for labels/title)
    z_idx : int
        Redshift index
    z_val : float
        Redshift value for title
    vmin, vmax : float
        Color scale limits
    cmap : str
        Colormap name
    savename : str, optional
        Save filename
    weighting : str
        Weighting method: 'snr', 'frac_signal', 'signal', 'inverse_variance', 'uniform'
    show_errors : bool
        If True, display uncertainties in cell annotations
    """
    # Model definitions
    cum_mass_labels = ['1.00e12', '3.16e12', '1.00e13', '3.16e13']
    cum_radius_labels = ['0.5', '1.0', '3.0', '5.0']
    
    disc_mass_lower = ['1.00e12', '3.16e12', '1.00e13', '3.16e13']
    disc_mass_upper = ['3.16e12', '1.00e13', '3.16e13', '1.00e15']
    disc_radius_inner = ['0.0', '0.5', '1.0', '3.0']
    disc_radius_outer = ['0.5', '1.0', '3.0', '5.0']
    
    cum_mass_display = [r'$10^{12}$', r'$10^{12.5}$', r'$10^{13}$', r'$10^{13.5}$']
    cum_radius_display = [r'$0.5$', r'$1.0$', r'$3.0$', r'$5.0$']
    
    disc_mass_display = [r'$[10^{12}, 10^{12.5}]$', r'$[10^{12.5}, 10^{13}]$', 
                         r'$[10^{13}, 10^{13.5}]$', r'$[10^{13.5}, 10^{15}]$']
    disc_radius_display = [r'[0, 0.5]', r'[0.5, 1]', r'[1, 3]', r'[3, 5]']
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # ============= LEFT: CUMULATIVE =============
    ax_cum = axes[0]
    
    F_cum_matrix = np.full((4, 4), np.nan)
    F_cum_err_matrix = np.full((4, 4), np.nan)
    
    for i, mass in enumerate(cum_mass_labels):
        for j, radius in enumerate(cum_radius_labels):
            model_name = f"hydro_replace_Ml_{mass}_Mu_1.00e15_Ri_0.0_Ro_{radius}"
            F_val, F_err = _get_F_value(results, model_name, z_idx, weighting, 'cum')
            F_cum_matrix[i, j] = F_val
            F_cum_err_matrix[i, j] = F_err
    
    im = ax_cum.imshow(F_cum_matrix, cmap=cmap, vmin=0, vmax=1, 
                       aspect='auto', origin='lower')
    
    # Add text annotations
    for i in range(4):
        for j in range(4):
            val = F_cum_matrix[i, j]
            err = F_cum_err_matrix[i, j]
            if not np.isnan(val):
                color = 'black'
                if show_errors and not np.isnan(err) and err > 0:
                    text = f'{val:.2f}\n±{err:.2f}'
                else:
                    text = f'{val:.2f}'
                ax_cum.text(j, i, text, ha='center', va='center', 
                           color=color, weight='bold')
    
    ax_cum.set_xticks(range(4))
    ax_cum.set_yticks(range(4))
    ax_cum.set_xticklabels(cum_radius_display)
    ax_cum.set_yticklabels(cum_mass_display)
    ax_cum.set_xlabel(r'$\alpha\, [R_{200}]$')
    ax_cum.set_ylabel(r'$M_{\rm min}\, [h^{-1}M_\odot]$')
    ax_cum.set_title('Cumulative')
    
    # ============= RIGHT: DISCRETE =============
    ax_disc = axes[1]
    
    F_disc_matrix = np.full((4, 4), np.nan)
    F_disc_err_matrix = np.full((4, 4), np.nan)
    
    for i, (Ml, Mu) in enumerate(zip(disc_mass_lower, disc_mass_upper)):
        for j, (Ri, Ro) in enumerate(zip(disc_radius_inner, disc_radius_outer)):
            model_name = f"hydro_replace_Ml_{Ml}_Mu_{Mu}_Ri_{Ri}_Ro_{Ro}"
            F_val, F_err = _get_F_value(results, model_name, z_idx, weighting, 'disc')
            F_disc_matrix[i, j] = F_val
            F_disc_err_matrix[i, j] = F_err
    
    im = ax_disc.imshow(F_disc_matrix, cmap=cmap2, vmin=-0.5, vmax=0.5, 
                        aspect='auto', origin='lower')
    
    # Add text annotations
    for i in range(4):
        for j in range(4):
            val = F_disc_matrix[i, j]
            err = F_disc_err_matrix[i, j]
            if not np.isnan(val):
                color = 'black'
                if show_errors and not np.isnan(err) and err > 0:
                    text = f'{val:.2f}\n±{err:.2f}'
                else:
                    text = f'{val:.2f}'
                ax_disc.text(j, i, text, ha='center', va='center', 
                            color=color, weight='bold')
    
    ax_disc.set_xticks(range(4))
    ax_disc.set_yticks(range(4))
    ax_disc.set_xticklabels(disc_radius_display, rotation=45, ha='right')
    ax_disc.set_yticklabels(disc_mass_display, rotation=45)
    ax_disc.set_xlabel(r'Radial bin $[R_{200}]$')
    ax_disc.set_ylabel(r'Mass bin $[h^{-1}M_\odot]$')
    ax_disc.set_title('Discrete')
    
    # Single colorbar
    weight_labels = {
        'frac_signal': r'$\langle F \rangle_{\rm signal}$',
        'signal': r'$F_{\rm signal}$',
        'inverse_variance': r'$\langle F \rangle_{\rm w}$',
        'uniform': r'$\langle F \rangle$'
    }
    label = weight_labels.get(weighting, r'$\langle F \rangle$')
    # fig.colorbar(im, ax=axes.ravel().tolist(), label=label, shrink=0.8, pad=0.02)
    
    
    plt.tight_layout()
    
    if savename is None:
        savename = f'figures/{stat_name}_cumulative_discrete_heatmap_{weighting}.pdf'
    plt.savefig(savename, dpi=150, bbox_inches='tight')
    plt.show()
    
    return fig, axes


def plot_unified_stat_heatmap(results, stat_name='Pk', z_idx=23, z_val=1.0,
                               vmin=0.0, vmax=1.0, cmap='Reds',
                               savename=None, weighting='snr', show_errors=False,
                               show_region_boundaries=True, show_annotations=True,
                               figsize=(10, 8)):
    """
    Plot a unified 8×8 heatmap combining discrete and cumulative models.
    
    Layout matches off_diagonal_response_v2.ipynb cell 14:
    - Rows 0-3: Discrete mass bins (HIGH mass first: 10^13.5-∞, ..., 10^12-10^12.5)
    - Rows 4-7: Cumulative mass (corner tile repeated, then M≥10^13, M≥10^12.5, M≥10^12)
    - Cols 0-3: Discrete radius bins [0-0.5, 0.5-1, 1-3, 3-5] R₂₀₀
    - Cols 4-7: Cumulative radius (corner repeated, then r<1, r<3, r<5) R₂₀₀
    
    The corner tile (M≥10^13.5, r<0.5) appears in both discrete and cumulative
    regions as they are identical.
    
    Parameters
    ----------
    results : dict
        Results dictionary from compute_statistic_response() for a single statistic
    stat_name : str
        Name of the statistic (for labels/title)
    z_idx : int
        Redshift index
    z_val : float
        Redshift value for title
    vmin, vmax : float
        Color scale limits
    cmap : str
        Colormap name
    savename : str, optional
        Save filename. If None, uses default naming.
    weighting : str
        Weighting method: 'snr', 'frac_signal', 'signal', 'inverse_variance', 'uniform'
    show_errors : bool
        If True, display uncertainties in cell annotations
    show_region_boundaries : bool
        If True, draw dashed lines separating discrete/cumulative regions
    show_annotations : bool
        If True, show F values as text in each cell
    figsize : tuple
        Figure size (width, height)
        
    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    # Define the 8×8 grid structure following off_diagonal_response_v2.ipynb cell 14
    # Mass in DECREASING order (high mass at top, i.e., row 0)
    # Using origin='upper' so row 0 is at top
    
    # Mass bins ordered (rows 0-7):
    # Discrete (exclusive) - HIGH mass first
    # (13.5, 15.0) -> row 0, (13.0, 13.5) -> row 1, (12.5, 13.0) -> row 2, (12.0, 12.5) -> row 3
    # Cumulative (includes corner tile repeated)
    # (13.5, 15.0) -> row 4 (corner), (13.0, 15.0) -> row 5, (12.5, 15.0) -> row 6, (12.0, 15.0) -> row 7
    
    mass_bins_ordered = [
        # Discrete (exclusive) - HIGH mass first
        (13.5, 15.0),   # row 0 - [M4] - Also cumulative corner
        (13.0, 13.5),   # row 1 - [M3]
        (12.5, 13.0),   # row 2 - [M2]
        (12.0, 12.5),   # row 3 - [M1]
        # Cumulative (includes corner repeated)
        (13.5, 15.0),   # row 4 - corner tile REPEATED
        (13.0, 15.0),   # row 5 - M >= 10^13
        (12.5, 15.0),   # row 6 - M >= 10^12.5
        (12.0, 15.0),   # row 7 - M >= 10^12 (full)
    ]
    
    radius_bins_ordered = [
        # Discrete (exclusive) - includes corner at R=0-0.5
        (0.0, 0.5),     # col 0 - [R1] - Also cumulative corner
        (0.5, 1.0),     # col 1 - [R2]
        (1.0, 3.0),     # col 2 - [R3]
        (3.0, 5.0),     # col 3 - [R4]
        # Cumulative (includes corner repeated)
        (0.0, 0.5),     # col 4 - corner tile REPEATED
        (0.0, 1.0),     # col 5 - R < 1 R200
        (0.0, 3.0),     # col 6 - R < 3 R200
        (0.0, 5.0),     # col 7 - R < 5 R200 (full)
    ]
    
    # Display labels for the 8×8 grid
    mass_labels_display = [
        # Discrete (rows 0-3) - high mass first
        r'$10^{13.5}\!-\!\infty$',
        r'$10^{13}\!-\!10^{13.5}$',
        r'$10^{12.5}\!-\!10^{13}$', 
        r'$10^{12}\!-\!10^{12.5}$',
        # Cumulative (rows 4-7)
        r'$>\!10^{13.5}$',
        r'$>\!10^{13}$',
        r'$>\!10^{12.5}$',
        r'$>\!10^{12}$',
    ]
    
    radius_labels_display = [
        # Discrete (cols 0-3)
        r'$[0, 0.5]$',
        r'$[0.5, 1]$',
        r'$[1, 3]$',
        r'$[3, 5]$',
        # Cumulative (cols 4-7)
        r'$<\!0.5$',
        r'$<\!1$',
        r'$<\!3$',
        r'$<\!5$',
    ]
    
    # Helper to convert log mass bounds to model string format
    # Matches notebook's format: f"{10**M_lo:.2e}".replace('e+', 'e')
    def mass_to_str(log_mass):
        return f'{10**log_mass:.2e}'.replace('e+', 'e')
    
    # Helper to get model name from (M_lo, M_hi, R_in, R_out)
    def get_model_name(M_lo, M_hi, R_in, R_out):
        Ml = mass_to_str(M_lo)
        Mu = mass_to_str(M_hi)
        return f"hydro_replace_Ml_{Ml}_Mu_{Mu}_Ri_{R_in}_Ro_{R_out}"
    
    # Build the 8×8 F matrix
    F_matrix = np.full((8, 8), np.nan)
    F_err_matrix = np.full((8, 8), np.nan)
    
    # Fill the matrix
    for i, (M_lo, M_hi) in enumerate(mass_bins_ordered):
        for j, (R_in, R_out) in enumerate(radius_bins_ordered):
            model_name = get_model_name(M_lo, M_hi, R_in, R_out)
            
            # Determine if this is a discrete or cumulative model for lookup
            is_mass_discrete = i < 4
            is_radius_discrete = j < 4
            
            # For the corner tile and cumulative models, try 'cum' lookup
            # For discrete models, try 'disc' lookup
            if is_mass_discrete and is_radius_discrete:
                # Pure discrete
                F_val, F_err = _get_F_value(results, model_name, z_idx, weighting, 'disc')
            elif not is_mass_discrete and not is_radius_discrete:
                # Pure cumulative
                F_val, F_err = _get_F_value(results, model_name, z_idx, weighting, 'cum')
            else:
                # Mixed/extended - try both
                F_val, F_err = _get_F_value(results, model_name, z_idx, weighting, 'disc')
                if np.isnan(F_val):
                    F_val, F_err = _get_F_value(results, model_name, z_idx, weighting, 'cum')
            
            F_matrix[i, j] = F_val
            F_err_matrix[i, j] = F_err
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot heatmap with origin='upper' so row 0 is at top (high mass)
    im = ax.imshow(F_matrix, cmap=cmap, vmin=vmin, vmax=vmax,
                   aspect='auto', origin='upper')
    
    # Add text annotations
    if show_annotations:
        for i in range(8):
            for j in range(8):
                val = F_matrix[i, j]
                err = F_err_matrix[i, j]
                if not np.isnan(val):
                    # Determine text color based on value
                    normalized_val = (val - vmin) / (vmax - vmin) if vmax != vmin else 0.5
                    color = 'white' if normalized_val > 0.6 else 'black'
                    
                    if show_errors and not np.isnan(err) and err > 0:
                        text = f'{val:.2f}\n±{err:.2f}'
                    else:
                        text = f'{val:.2f}'
                    
                    ax.text(j, i, text, ha='center', va='center',
                           color=color, fontsize=8, weight='bold')
    
    # Set ticks and labels
    ax.set_xticks(range(8))
    ax.set_yticks(range(8))
    ax.set_xticklabels(radius_labels_display, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(mass_labels_display, fontsize=9)
    
    ax.set_xlabel(r'$\alpha$ [$R_{200}$]', fontsize=11)
    ax.set_ylabel(r'Mass [$h^{-1}M_\odot$]', fontsize=11)
    
    # Draw region boundaries
    if show_region_boundaries:
        # Horizontal line separating discrete (rows 0-3) from cumulative (rows 4-7) mass
        ax.axhline(3.5, color='k', lw=1.5, ls='--', alpha=0.7)
        # Vertical line separating discrete (cols 0-3) from cumulative (cols 4-7) radius
        ax.axvline(3.5, color='k', lw=1.5, ls='--', alpha=0.7)
    
    # Title
    weight_labels = {
        'snr': 'SNR-weighted',
        'frac_signal': 'Signal-weighted',
        'signal': 'Abs. signal-weighted',
        'inverse_variance': 'Inv-var weighted',
        'uniform': 'Unweighted'
    }
    weight_label = weight_labels.get(weighting, weighting)
    ax.set_title(f'{stat_name} Response at $z_s = {z_val:.2f}$ ({weight_label})', 
                 fontsize=12, pad=10)
    
    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label(r'$\langle F \rangle$', fontsize=11)
    
    plt.tight_layout()
    
    if savename is not None:
        plt.savefig(savename, dpi=150, bbox_inches='tight')
    plt.show()
    
    return fig, ax


def plot_unified_stat_heatmap_from_F_all(F_all, stat_name='Pk', z_val=1.0,
                                          vmin=0.0, vmax=1.0, cmap='Reds',
                                          savename=None, 
                                          show_region_boundaries=True, 
                                          show_annotations=True,
                                          figsize=(10, 8)):
    """
    Plot a unified 8×8 heatmap using pre-computed F_all dictionary.
    
    This version works directly with F_all = {model_name: {stat: F_value, ...}, ...}
    as computed in off_diagonal_response_v2.ipynb.
    
    Layout:
    - Rows 0-3: Discrete mass bins (HIGH mass first: 10^13.5-∞, ..., 10^12-10^12.5)
    - Rows 4-7: Cumulative mass (corner tile repeated, then M≥10^13, M≥10^12.5, M≥10^12)
    - Cols 0-3: Discrete radius bins [0-0.5, 0.5-1, 1-3, 3-5] R₂₀₀
    - Cols 4-7: Cumulative radius (corner repeated, then r<1, r<3, r<5) R₂₀₀
    
    Parameters
    ----------
    F_all : dict
        Dictionary mapping model_name -> {stat_name: F_value, ...}
    stat_name : str
        Name of the statistic to plot
    z_val : float
        Redshift value for title (unused if show_title=False)
    vmin, vmax : float
        Color scale limits
    cmap : str
        Colormap name
    savename : str, optional
        Save filename
    show_region_boundaries : bool
        If True, draw dashed lines separating discrete/cumulative regions
    show_annotations : bool
        If True, show F values as text in each cell
    figsize : tuple
        Figure size (width, height)
        
    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    # Mass bins ordered (rows 0-7) - HIGH mass first
    mass_bins_ordered = [
        (13.5, 15.0),   # row 0 - discrete M4 / cumulative corner
        (13.0, 13.5),   # row 1 - discrete M3
        (12.5, 13.0),   # row 2 - discrete M2
        (12.0, 12.5),   # row 3 - discrete M1
        (13.5, 15.0),   # row 4 - cumulative corner (repeated)
        (13.0, 15.0),   # row 5 - M >= 10^13
        (12.5, 15.0),   # row 6 - M >= 10^12.5
        (12.0, 15.0),   # row 7 - M >= 10^12 (full)
    ]
    
    radius_bins_ordered = [
        (0.0, 0.5),     # col 0 - discrete R1 / cumulative corner
        (0.5, 1.0),     # col 1 - discrete R2
        (1.0, 3.0),     # col 2 - discrete R3
        (3.0, 5.0),     # col 3 - discrete R4
        (0.0, 0.5),     # col 4 - cumulative corner (repeated)
        (0.0, 1.0),     # col 5 - R < 1 R200
        (0.0, 3.0),     # col 6 - R < 3 R200
        (0.0, 5.0),     # col 7 - R < 5 R200 (full)
    ]
    
    # Display labels - matching cell 15 style
    mass_labels_display = [
        f"$10^{{{M_lo:.1f}}}$-$\\infty$" if M_hi == 15.0 else f"$10^{{{M_lo:.1f}}}$-$10^{{{M_hi:.1f}}}$"
        for M_lo, M_hi in mass_bins_ordered
    ]
    
    radius_labels_display = [f"{R_in}-{R_out}" for R_in, R_out in radius_bins_ordered]
    
    # Model name generator (matching notebook format)
    def get_model_name(M_lo, M_hi, R_in, R_out):
        M_lo_str = f"{10**M_lo:.2e}".replace('e+', 'e')
        M_hi_str = f"{10**M_hi:.2e}".replace('e+', 'e')
        return f"hydro_replace_Ml_{M_lo_str}_Mu_{M_hi_str}_Ri_{R_in}_Ro_{R_out}"
    
    # Build the 8×8 F matrix
    F_matrix = np.full((8, 8), np.nan)
    
    for i, (M_lo, M_hi) in enumerate(mass_bins_ordered):
        for j, (R_in, R_out) in enumerate(radius_bins_ordered):
            model_name = get_model_name(M_lo, M_hi, R_in, R_out)
            if model_name in F_all:
                F_matrix[i, j] = F_all[model_name].get(stat_name, np.nan)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot heatmap with origin='upper' so row 0 is at top (high mass)
    im = ax.imshow(F_matrix, cmap=cmap, vmin=vmin, vmax=vmax,
                   aspect='auto', origin='upper')
    
    # Add text annotations (all black)
    if show_annotations:
        for i in range(8):
            for j in range(8):
                val = F_matrix[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                           color='black', weight='bold')
    
    # Set ticks and labels
    ax.set_xticks(range(8))
    ax.set_yticks(range(8))
    ax.set_xticklabels(radius_labels_display, rotation=45, ha='right')
    ax.set_yticklabels(mass_labels_display, rotation=0)
    
    ax.set_xlabel(r'$\alpha$ [$r/R_{200}$]')
    ax.set_ylabel(r'M [$h^{-1}\,M_\odot$]')
    
    # Draw region boundaries (matching cell 15 style)
    if show_region_boundaries:
        ax.axhline(3.5, color='k', lw=1.5, ls='--', alpha=0.7)
        ax.axvline(3.5, color='k', lw=1.5, ls='--', alpha=0.7)
    
    # Colorbar - full height of y-axis, label includes statistic name
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    # Map stat_name to LaTeX symbol
    stat_symbols = {
        'Pk': 'P', 'Cl': 'C_\\ell', 'peaks': '\\mathrm{peaks}', 
        'minima': '\\mathrm{minima}', 'MFs_V0': 'V_0', 'MFs_V1': 'V_1', 
        'MFs_V2': 'V_2', 'pdf': '\\mathrm{pdf}'
    }
    stat_sym = stat_symbols.get(stat_name, stat_name)
    cbar.set_label(rf'$\langle F_{{{stat_sym}}} \rangle$')
    
    plt.tight_layout()
    
    if savename is not None:
        plt.savefig(savename, dpi=200, bbox_inches='tight')
    plt.show()
    
    return fig, ax


def plot_unified_stat_heatmap_with_margins(F_all, stat_name='Pk',
                                            vmin=0.0, vmax=1.0, cmap='Reds',
                                            savename=None, 
                                            show_region_boundaries=True, 
                                            show_annotations=True,
                                            figsize=(12, 10)):
    """
    Plot unified 8×8 heatmap with marginal difference plots showing nonlinearity.
    
    The marginal plots show ΔF = F_cumulative - Σ(F_discrete):
    - Top margin: For each discrete radius column, difference between cumulative-in-mass
                  value (row 7) and sum of discrete mass rows (0-3)
    - Right margin: For each discrete mass row, difference between cumulative-in-radius
                    value (col 7) and sum of discrete radius cols (0-3)
    
    Positive ΔF = super-additive (nonlinear enhancement)
    Negative ΔF = sub-additive (nonlinear suppression)
    
    Parameters
    ----------
    F_all : dict
        Dictionary mapping model_name -> {stat_name: F_value, ...}
    stat_name : str
        Name of the statistic to plot
    vmin, vmax : float
        Color scale limits for heatmap
    cmap : str
        Colormap name
    savename : str, optional
        Save filename
    show_region_boundaries : bool
        If True, draw dashed lines separating discrete/cumulative regions
    show_annotations : bool
        If True, show F values as text in each cell
    figsize : tuple
        Figure size (width, height)
        
    Returns
    -------
    fig, axes : matplotlib Figure and dict of Axes
    """
    # Mass bins ordered (rows 0-7) - HIGH mass first
    mass_bins_ordered = [
        (13.5, 15.0),   # row 0 - discrete M4
        (13.0, 13.5),   # row 1 - discrete M3
        (12.5, 13.0),   # row 2 - discrete M2
        (12.0, 12.5),   # row 3 - discrete M1
        (13.5, 15.0),   # row 4 - cumulative corner
        (13.0, 15.0),   # row 5 - M >= 10^13
        (12.5, 15.0),   # row 6 - M >= 10^12.5
        (12.0, 15.0),   # row 7 - M >= 10^12 (full)
    ]
    
    radius_bins_ordered = [
        (0.0, 0.5),     # col 0 - discrete R1
        (0.5, 1.0),     # col 1 - discrete R2
        (1.0, 3.0),     # col 2 - discrete R3
        (3.0, 5.0),     # col 3 - discrete R4
        (0.0, 0.5),     # col 4 - cumulative corner
        (0.0, 1.0),     # col 5 - R < 1 R200
        (0.0, 3.0),     # col 6 - R < 3 R200
        (0.0, 5.0),     # col 7 - R < 5 R200 (full)
    ]
    
    # Display labels
    mass_labels_display = [
        f"$10^{{{M_lo:.1f}}}$-$\\infty$" if M_hi == 15.0 else f"$10^{{{M_lo:.1f}}}$-$10^{{{M_hi:.1f}}}$"
        for M_lo, M_hi in mass_bins_ordered
    ]
    radius_labels_display = [f"{R_in}-{R_out}" for R_in, R_out in radius_bins_ordered]
    
    # Model name generator
    def get_model_name(M_lo, M_hi, R_in, R_out):
        M_lo_str = f"{10**M_lo:.2e}".replace('e+', 'e')
        M_hi_str = f"{10**M_hi:.2e}".replace('e+', 'e')
        return f"hydro_replace_Ml_{M_lo_str}_Mu_{M_hi_str}_Ri_{R_in}_Ro_{R_out}"
    
    # Build the 8×8 F matrix
    F_matrix = np.full((8, 8), np.nan)
    for i, (M_lo, M_hi) in enumerate(mass_bins_ordered):
        for j, (R_in, R_out) in enumerate(radius_bins_ordered):
            model_name = get_model_name(M_lo, M_hi, R_in, R_out)
            if model_name in F_all:
                F_matrix[i, j] = F_all[model_name].get(stat_name, np.nan)
    
    # Compute marginal differences (nonlinearity) - 8 bars each
    # Top margin: For each column j, compare cumulative-in-mass (row 7) to sum of discrete mass rows (0-3)
    delta_top = np.zeros(8)
    for j in range(8):
        F_cumulative = F_matrix[7, j]  # M > 10^12 at column j
        F_sum_discrete = np.nansum(F_matrix[0:4, j])  # Sum of discrete mass bins at column j
        delta_top[j] = F_cumulative - F_sum_discrete
    
    # Right margin: For each row i, compare cumulative-in-radius (col 7) to sum of discrete radius cols (0-3)
    delta_right = np.zeros(8)
    for i in range(8):
        F_cumulative = F_matrix[i, 7]  # Row i at r < 5 R200
        F_sum_discrete = np.nansum(F_matrix[i, 0:4])  # Sum of discrete radius bins at row i
        delta_right[i] = F_cumulative - F_sum_discrete
    
    # Create figure with GridSpec layout (no colorbar)
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, width_ratios=[8, 1.5], height_ratios=[1.5, 8],
                          wspace=0.05, hspace=0.05)
    
    ax_main = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)
    # Top-right corner empty
    ax_corner = fig.add_subplot(gs[0, 1])
    ax_corner.axis('off')
    
    # Main heatmap
    im = ax_main.imshow(F_matrix, cmap=cmap, vmin=vmin, vmax=vmax,
                        aspect='auto', origin='upper')
    
    # Add text annotations (all black)
    if show_annotations:
        for i in range(8):
            for j in range(8):
                val = F_matrix[i, j]
                if not np.isnan(val):
                    ax_main.text(j, i, f'{val:.2f}', ha='center', va='center',
                                color='black', weight='bold', fontsize=9)
    
    # Set main axis ticks and labels
    ax_main.set_xticks(range(8))
    ax_main.set_yticks(range(8))
    ax_main.set_xticklabels(radius_labels_display, rotation=45, ha='right')
    ax_main.set_yticklabels(mass_labels_display, rotation=0)
    ax_main.set_xlabel(r'$\alpha$ [$r/R_{200}$]')
    ax_main.set_ylabel(r'M [$h^{-1}\,M_\odot$]')
    
    # Draw region boundaries
    if show_region_boundaries:
        ax_main.axhline(3.5, color='k', lw=1.5, ls='--', alpha=0.7)
        ax_main.axvline(3.5, color='k', lw=1.5, ls='--', alpha=0.7)
    
    # Top marginal: ΔF for all 8 columns (cumulative in mass)
    bar_colors_top = ['steelblue' if d >= 0 else 'coral' for d in delta_top]
    ax_top.bar(range(8), delta_top, color=bar_colors_top, edgecolor='k', width=0.7)
    ax_top.axhline(0, color='k', lw=0.5, ls='-')
    ax_top.axvline(3.5, color='k', lw=1.5, ls='--', alpha=0.7)  # Region boundary
    ax_top.set_xlim(-0.5, 7.5)
    ax_top.set_ylabel(r'$\Delta F_M$', fontsize=10)
    ax_top.tick_params(labelbottom=False)
    # Add value annotations
    for j, d in enumerate(delta_top):
        if not np.isnan(d):
            ax_top.text(j, d + 0.01 * np.sign(d), f'{d:.2f}', ha='center', 
                       va='bottom' if d >= 0 else 'top', fontsize=7, color='black')
    ax_top.set_title(r'$F_{M>\mathrm{min}} - \sum_i F_{M_i}$', fontsize=10)
    
    # Right marginal: ΔF for all 8 rows (cumulative in radius)
    bar_colors_right = ['steelblue' if d >= 0 else 'coral' for d in delta_right]
    ax_right.barh(range(8), delta_right, color=bar_colors_right, edgecolor='k', height=0.7)
    ax_right.axvline(0, color='k', lw=0.5, ls='-')
    ax_right.axhline(3.5, color='k', lw=1.5, ls='--', alpha=0.7)  # Region boundary
    ax_right.set_ylim(7.5, -0.5)  # Inverted to match heatmap orientation
    ax_right.set_xlabel(r'$\Delta F_\alpha$', fontsize=10)
    ax_right.tick_params(labelleft=False)
    # Add value annotations
    for i, d in enumerate(delta_right):
        if not np.isnan(d):
            ax_right.text(d + 0.01 * np.sign(d), i, f'{d:.2f}', va='center',
                         ha='left' if d >= 0 else 'right', fontsize=7, color='black')
    ax_right.set_title(r'$F_{\alpha<\mathrm{max}} - \sum_i F_{\alpha_i}$', fontsize=10)
    
    plt.tight_layout()
    
    if savename is not None:
        plt.savefig(savename, dpi=200, bbox_inches='tight')
    plt.show()
    
    return fig, {'main': ax_main, 'top': ax_top, 'right': ax_right}


# Usage examples
def plot_combined_heatmap(all_results, model_type='discrete', z_idx=23, z_val=1.0,
                         statistics=None, vmin=0.0, vmax=1.0, cmap='RdYlGn',
                         savename=None, weighting='frac_signal', show_errors=False):
    """
    Create multi-panel heatmap showing all statistics.
    
    Parameters
    ----------
    all_results : dict
        Dictionary with keys = statistic names, values = result dicts from compute_statistic_response()
    model_type : str
        Either 'cumulative', 'discrete', or 'both'
    z_idx : int
        Redshift index
    z_val : float
        Redshift value for title
    statistics : list of str, optional
        List of statistics to plot. If None, uses all available
    vmin, vmax : float
        Color scale limits
    cmap : str
        Colormap name
    savename : str, optional
        Save filename
    weighting : str
        Weighting method for averaging F over bins:
        - 'snr': Use snr_weighted_F (fractional signal × inverse variance).
          Combines fractional baryonic signal with inverse-variance weighting.
        - 'frac_signal' (default): Use frac_signal_weighted_F from integrated_scores.
          Weights by |(H-D)/D|, emphasizing where baryons have largest fractional effect.
        - 'signal': Use signal_weighted_F. Weights by |H-D|, emphasizing largest absolute effect.
        - 'inverse_variance': Use weighted_mean_cum_F (inverse-variance weighting).
        - 'uniform': Use mean_cum_F (simple unweighted mean).
    show_errors : bool
        If True, display uncertainties in cell annotations (only for inverse_variance weighting).
    """
    if statistics is None:
        statistics = list(all_results.keys())
    
    n_stats = len(statistics)
    
    if model_type == 'discrete':
        # Determine grid layout
        if n_stats <= 6:
            nrows, ncols = 1, n_stats
        elif n_stats <= 8:
            nrows, ncols = 2, 4
        elif n_stats <= 12:
            nrows, ncols = 3, 4
        else:
            nrows = int(np.ceil(n_stats / 4))
            ncols = 4
        
        fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 5*nrows))
        if n_stats == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        _plot_discrete_heatmaps(fig, all_results, statistics, axes, z_idx, z_val, 
                               vmin, vmax, cmap, savename, weighting, show_errors)
                               
    elif model_type == 'cumulative':
        # Determine grid layout
        if n_stats <= 6:
            nrows, ncols = 1, n_stats
        elif n_stats <= 8:
            nrows, ncols = 2, 4
        elif n_stats <= 12:
            nrows, ncols = 3, 4
        else:
            nrows = int(np.ceil(n_stats / 4))
            ncols = 4
        
        fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 5*nrows))
        if n_stats == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        _plot_cumulative_heatmaps(fig, all_results, statistics, axes, z_idx, z_val,
                                 vmin, vmax, cmap, savename, weighting, show_errors)
                                 
    elif model_type == 'both':
        # 2 rows × n_stats columns
        # Top row: cumulative for all statistics
        # Bottom row: discrete for all statistics
        fig, axes = plt.subplots(2, n_stats, figsize=(4*n_stats, 8))
        
        if n_stats == 1:
            axes = axes.reshape(2, 1)
        
        _plot_both_heatmaps(fig, all_results, statistics, axes, z_idx, z_val,
                           vmin, vmax, cmap, savename, weighting, show_errors)
    else:
        raise ValueError(f"model_type must be 'cumulative', 'discrete', or 'both', got {model_type}")


def _get_F_value(results, model_name, z_idx, weighting='frac_signal', model_type='cum'):
    """
    Extract F value based on weighting method.
    
    Parameters
    ----------
    results : dict
        Results dictionary from compute_statistic_response()
    model_name : str
        Model name
    z_idx : int
        Redshift index
    weighting : str
        Weighting method: 'snr', 'frac_signal', 'signal', 'inverse_variance', or 'uniform'
    model_type : str
        'cum' for cumulative or 'disc' for discrete
        
    Returns
    -------
    F_val : float
        F value (or NaN if not found)
    F_err : float
        F uncertainty (or NaN if not available)
    """
    prefix = 'cum' if model_type == 'cum' else 'disc'
    
    if weighting == 'snr':
        scores_key = f'{prefix}_integrated_scores'
        if scores_key in results and model_name in results[scores_key]:
            F_val = results[scores_key][model_name]['snr_weighted_F'][z_idx]
            F_err = results[scores_key][model_name].get('snr_weighted_F_err', [np.nan] * (z_idx + 1))[z_idx]
            return F_val, F_err
    elif weighting == 'frac_signal':
        scores_key = f'{prefix}_integrated_scores'
        if scores_key in results and model_name in results[scores_key]:
            F_val = results[scores_key][model_name]['frac_signal_weighted_F'][z_idx]
            # No error estimate for frac_signal_weighted currently
            return F_val, np.nan
    elif weighting == 'signal':
        scores_key = f'{prefix}_integrated_scores'
        if scores_key in results and model_name in results[scores_key]:
            F_val = results[scores_key][model_name]['signal_weighted_F'][z_idx]
            F_err = results[scores_key][model_name].get('signal_weighted_F_err', [np.nan] * (z_idx + 1))[z_idx]
            return F_val, F_err
    elif weighting == 'inverse_variance':
        mean_key = f'weighted_mean_{prefix}_F'
        err_key = f'weighted_mean_{prefix}_F_err'
        if mean_key in results and model_name in results[mean_key]:
            F_val = results[mean_key][model_name][z_idx]
            F_err = results.get(err_key, {}).get(model_name, [np.nan] * (z_idx + 1))[z_idx]
            return F_val, F_err
    elif weighting == 'uniform':
        mean_key = f'mean_{prefix}_F'
        if mean_key in results and model_name in results[mean_key]:
            F_val = results[mean_key][model_name][z_idx]
            return F_val, np.nan
    
    return np.nan, np.nan


def _plot_both_heatmaps(fig, all_results, statistics, axes, z_idx, z_val,
                       vmin, vmax, cmap, savename, weighting='frac_signal', show_errors=False):
    """Plot both cumulative (top row) and discrete (bottom row) heatmaps.
    
    Parameters
    ----------
    weighting : str
        Weighting method: 'snr', 'frac_signal', 'signal', 'inverse_variance', or 'uniform'
    show_errors : bool
        If True and errors available, display ±σ in cell annotations.
    """
    
    disc_mass_lower = ['1.00e12', '3.16e12', '1.00e13', '3.16e13']
    disc_mass_upper = ['3.16e12', '1.00e13', '3.16e13', '1.00e15']
    disc_radius_inner = ['0.0', '0.5', '1.0', '3.0']
    disc_radius_outer = ['0.5', '1.0', '3.0', '5.0']
    
    cum_mass_labels = ['1.00e12', '3.16e12', '1.00e13', '3.16e13']
    cum_radius_labels = ['0.5', '1.0', '3.0', '5.0']
    
    disc_mass_display = [r'$[10^{12}, 10^{12.5}]$', r'$[10^{12.5}, 10^{13}]$', 
                         r'$[10^{13}, 10^{13.5}]$', r'$[10^{13.5}, 10^{15}]$']
    disc_radius_display = [r'[0, 0.5]', r'[0.5, 1]', r'[1, 3]', r'[3, 5]']
    
    cum_mass_display = [r'$10^{12}$', r'$10^{12.5}$', 
                        r'$10^{13}$', r'$10^{13.5}$']
    cum_radius_display = [r'$0.5$', r'$1.0$', r'$3.0$', r'$5.0$']
    
    n_stats = len(statistics)
    im = None
    
    for col_idx, stat_name in enumerate(statistics):
        results = all_results[stat_name]
        
        # ============= ROW 0: CUMULATIVE =============
        ax_cum = axes[0, col_idx]
        
        F_cum_matrix = np.full((4, 4), np.nan)
        F_cum_err_matrix = np.full((4, 4), np.nan)
        
        for i, mass in enumerate(cum_mass_labels):
            for j, radius in enumerate(cum_radius_labels):
                model_name = f"hydro_replace_Ml_{mass}_Mu_1.00e15_Ri_0.0_Ro_{radius}"
                F_val, F_err = _get_F_value(results, model_name, z_idx, weighting, 'cum')
                F_cum_matrix[i, j] = F_val
                F_cum_err_matrix[i, j] = F_err
        
        im = ax_cum.imshow(F_cum_matrix, cmap=cmap, vmin=vmin, vmax=vmax, 
                          aspect='auto', origin='lower')
        
        # Add text annotations with optional error
        for i in range(4):
            for j in range(4):
                val = F_cum_matrix[i, j]
                err = F_cum_err_matrix[i, j]
                if not np.isnan(val):
                    color = 'black'
                    if show_errors and not np.isnan(err) and err > 0:
                        text = f'{val:.2f}\n±{err:.2f}'
                        
                    else:
                        text = f'{val:.2f}'
                        
                    ax_cum.text(j, i, text, ha='center', va='center', 
                               color=color, weight='bold')
        
        ax_cum.set_xticks(range(4))
        ax_cum.set_yticks(range(4))
        
        # X-axis labels showing cumulative radial bins
        ax_cum.set_xticklabels(cum_radius_display, ha='right')
        
        # Y-axis labels only on leftmost column
        if col_idx == 0:
            ax_cum.set_yticklabels(cum_mass_display)
            ax_cum.set_ylabel(r'$M_{\rm min}\, [h^{-1}M_\odot]$', 
                             )
        else:
            ax_cum.set_yticklabels([])
        
        # Title on top row
        ax_cum.set_title(stat_name.upper(), pad=10)
        
        # ============= ROW 1: DISCRETE =============
        ax_disc = axes[1, col_idx]
        
        F_disc_matrix = np.full((4, 4), np.nan)
        F_disc_err_matrix = np.full((4, 4), np.nan)
        
        for i, (Ml, Mu) in enumerate(zip(disc_mass_lower, disc_mass_upper)):
            for j, (Ri, Ro) in enumerate(zip(disc_radius_inner, disc_radius_outer)):
                model_name = f"hydro_replace_Ml_{Ml}_Mu_{Mu}_Ri_{Ri}_Ro_{Ro}"
                F_val, F_err = _get_F_value(results, model_name, z_idx, weighting, 'disc')
                F_disc_matrix[i, j] = F_val
                F_disc_err_matrix[i, j] = F_err
        
        im = ax_disc.imshow(F_disc_matrix, cmap=cmap, vmin=vmin, vmax=vmax, 
                           aspect='auto', origin='lower')
        
        # Add text annotations with optional error
        for i in range(4):
            for j in range(4):
                val = F_disc_matrix[i, j]
                err = F_disc_err_matrix[i, j]
                if not np.isnan(val):
                    color = 'black'
                    if show_errors and not np.isnan(err) and err > 0:
                        text = f'{val:.2f}\n±{err:.2f}'
                        
                    else:
                        text = f'{val:.2f}'
                        
                    ax_disc.text(j, i, text, ha='center', va='center', 
                                color=color, weight='bold')
        
        ax_disc.set_xticks(range(4))
        ax_disc.set_yticks(range(4))
        
        # X-axis labels on bottom row showing discrete radial bins
        ax_disc.set_xticklabels(disc_radius_display, ha='right', rotation=45)
        ax_disc.set_xlabel(r'$\alpha \,[R_{200}]$')
        
        # Y-axis labels only on leftmost column
        if col_idx == 0:
            ax_disc.set_yticklabels(disc_mass_display)
            ax_disc.set_ylabel(r'Mass [$h^{-1}M_\odot$]', 
                               )
        else:
            ax_disc.set_yticklabels([])
    
    # Add single colorbar with appropriate label
    weight_labels = {
        'frac_signal': r'$\langle F \rangle_{\rm signal}$',
        'signal': r'$F_{\rm signal}$',
        'inverse_variance': r'$\langle F \rangle_{\rm w}$',
        'uniform': r'$\langle F \rangle$'
    }
    label = weight_labels.get(weighting, r'$\langle F \rangle$')
    fig.colorbar(im, ax=axes.ravel().tolist(), label=label, 
                shrink=1.0, pad=0.02)

    
    if savename is None:
        savename = f'figures/all_statistics_both_heatmap_{weighting}.pdf'
    plt.savefig(savename, dpi=150, bbox_inches='tight')
    plt.show()


def _plot_discrete_heatmaps(fig, all_results, statistics, axes, z_idx, z_val,
                            vmin, vmax, cmap, savename, weighting='frac_signal', show_errors=False):
    """Plot discrete model heatmaps with configurable weighting method."""
    disc_mass_lower = ['1.00e12', '3.16e12', '1.00e13', '3.16e13']
    disc_mass_upper = ['3.16e12', '1.00e13', '3.16e13', '1.00e15']
    disc_radius_inner = ['0.0', '0.5', '1.0', '3.0']
    disc_radius_outer = ['0.5', '1.0', '3.0', '5.0']
    
    mass_bin_display = [r'$[10^{12}, 10^{12.5}]$', r'$[10^{12.5}, 10^{13}]$', 
                        r'$[10^{13}, 10^{13.5}]$', r'$[10^{13.5}, \infty]$']
    radius_bin_display = [r'[0, 0.5]', r'[0.5, 1]', r'[1, 3]', r'[3, 5]']
    
    n_stats = len(statistics)
    im = None
    
    for idx, stat_name in enumerate(statistics):
        ax = axes[idx]
        
        results = all_results[stat_name]
        
        # Build F_matrix and error matrix using helper function
        F_matrix = np.full((4, 4), np.nan)
        F_err_matrix = np.full((4, 4), np.nan)
        for i, (Ml, Mu) in enumerate(zip(disc_mass_lower, disc_mass_upper)):
            for j, (Ri, Ro) in enumerate(zip(disc_radius_inner, disc_radius_outer)):
                model_name = f"hydro_replace_Ml_{Ml}_Mu_{Mu}_Ri_{Ri}_Ro_{Ro}"
                F_val, F_err = _get_F_value(results, model_name, z_idx, weighting, 'disc')
                F_matrix[i, j] = F_val
                F_err_matrix[i, j] = F_err
        
        im = ax.imshow(F_matrix, cmap=cmap, vmin=vmin, vmax=vmax, 
                      aspect='auto', origin='lower')
        
        # Add text annotations with optional error
        for i in range(4):
            for j in range(4):
                val = F_matrix[i, j]
                err = F_err_matrix[i, j]
                if not np.isnan(val):
                    # Determine text color based on value
                    if cmap in ['RdYlGn', 'RdBu_r']:
                        color = 'white' if val < 0.3 or val > 0.7 else 'black'
                    else:
                        color = 'white' if val < (vmin + vmax) / 2 else 'black'
                    
                    if show_errors and not np.isnan(err) and err > 0:
                        text = f'{val:.2f}\n±{err:.2f}'
                        
                    else:
                        text = f'{val:.2f}'
                        
                    ax.text(j, i, text, ha='center', va='center', 
                           color=color, weight='bold')
        
        # Set ticks and labels
        ax.set_xticks(range(4))
        ax.set_yticks(range(4))
        
        # Determine grid layout for labels
        nrows = int(np.ceil(n_stats / 4))
        bottom_row_start = (nrows - 1) * 4
        
        # X-axis labels (only bottom row)
        if idx >= bottom_row_start:
            ax.set_xticklabels(radius_bin_display, rotation=45, ha='right')
            ax.set_xlabel(r'Radial bin [$R_{200}$]')
        else:
            ax.set_xticklabels([])
        
        # Y-axis labels (only leftmost column)
        if idx % 4 == 0:
            ax.set_yticklabels(mass_bin_display)
            ax.set_ylabel(r'Mass bin [$h^{-1}M_\odot$]')
        else:
            ax.set_yticklabels([])
        
        ax.set_title(stat_name.upper(), weight='bold')
    
    # Hide unused subplots
    for idx in range(n_stats, len(axes)):
        axes[idx].axis('off')
    
    # Single colorbar for all
    label = r'$\langle F \rangle_w$' if use_weighted else r'$\langle F \rangle$'
    fig.colorbar(im, ax=axes[:n_stats].tolist(), label=label, 
                shrink=1.0, pad=0.01)

    
    if savename is None:
        savename = 'figures/all_statistics_discrete_heatmap.pdf'
    plt.savefig(savename, dpi=150, bbox_inches='tight')
    plt.show()


def _plot_cumulative_heatmaps(fig, all_results, statistics, axes, z_idx, z_val,
                              vmin, vmax, cmap, savename, weighting='frac_signal', show_errors=False):
    """Plot cumulative model heatmaps with configurable weighting method."""
    mass_labels = ['1.00e12', '3.16e12', '1.00e13', '3.16e13']
    radius_labels = ['0.5', '1.0', '3.0', '5.0']
    mass_display = [r'$M > 10^{12}$', r'$M > 10^{12.5}$', 
                    r'$M > 10^{13}$', r'$M > 10^{13.5}$']
    radius_display = [r'$0.5$', r'$1.0$', r'$3.0$', r'$5.0$']
    
    n_stats = len(statistics)
    im = None
    
    for idx, stat_name in enumerate(statistics):
        ax = axes[idx]
        results = all_results[stat_name]
        
        # Build F_matrix and error matrix using helper function
        F_matrix = np.full((4, 4), np.nan)
        F_err_matrix = np.full((4, 4), np.nan)
        for i, mass in enumerate(mass_labels):
            for j, radius in enumerate(radius_labels):
                model_name = f"hydro_replace_Ml_{mass}_Mu_1.00e15_Ri_0.0_Ro_{radius}"
                F_val, F_err = _get_F_value(results, model_name, z_idx, weighting, 'cum')
                F_matrix[i, j] = F_val
                F_err_matrix[i, j] = F_err
        
        im = ax.imshow(F_matrix, cmap=cmap, vmin=vmin, vmax=vmax, 
                      aspect='auto', origin='lower')
        
        # Add text annotations with optional error
        for i in range(4):
            for j in range(4):
                val = F_matrix[i, j]
                err = F_err_matrix[i, j]
                if not np.isnan(val):
                    if cmap in ['RdYlGn', 'RdBu_r']:
                        color = 'white' if val < 0.3 or val > 0.7 else 'black'
                    else:
                        color = 'white' if val < (vmin + vmax) / 2 else 'black'
                    
                    if show_errors and not np.isnan(err) and err > 0:
                        text = f'{val:.2f}\n±{err:.2f}'
                        
                    else:
                        text = f'{val:.2f}'
                        
                    ax.text(j, i, text, ha='center', va='center', 
                           color=color, weight='bold')
        
        # Set ticks and labels
        ax.set_xticks(range(4))
        ax.set_yticks(range(4))
        
        # Determine grid layout for labels
        nrows = int(np.ceil(n_stats / 4))
        bottom_row_start = (nrows - 1) * 4
        
        # X-axis labels (only bottom row)
        if idx >= bottom_row_start:
            ax.set_xticklabels(radius_display)
            ax.set_xlabel(r'Radius factor [$R_{200}$]')
        else:
            ax.set_xticklabels([])
        
        # Y-axis labels (only leftmost column)
        if idx % 4 == 0:
            ax.set_yticklabels(mass_display)
            ax.set_ylabel(r'Mass threshold [$h^{-1}M_\odot$]')
        else:
            ax.set_yticklabels([])
        
        ax.set_title(stat_name.upper(), weight='bold')
    
    # Hide unused subplots
    for idx in range(n_stats, len(axes)):
        axes[idx].axis('off')
    
    # Single colorbar with appropriate label
    weight_labels = {
        'frac_signal': r'$\langle F\rangle_{\rm signal}$',
        'signal': r'$F_{\rm signal}$',
        'inverse_variance': r'$\langle F \rangle_{\rm w}$',
        'uniform': r'$\langle F \rangle$'
    }
    label = weight_labels.get(weighting, r'$\langle F \rangle$')
    fig.colorbar(im, ax=axes[:n_stats].tolist(), label=label, 
                shrink=1.0, pad=0.01)
    

    
    if savename is None:
        savename = f'figures/all_statistics_cumulative_heatmap_{weighting}.pdf'
    plt.savefig(savename, dpi=150, bbox_inches='tight')
    plt.show()


# Usage:
def plot_cumulative_vs_discrete_heatmap(all_results, stat_name='pdf', 
                                        z_idx=23, z_val=1.0, savename=None):
    """
    Plot cumulative and discrete heatmaps side-by-side for comparison.
    """
    fig, axes = plt.subplots(1, 2, figsize=(24, 12))
    
    results = all_results[stat_name]
    
    # Cumulative
    ax = axes[0]
    mass_labels = ['1.00e12', '3.16e12', '1.00e13', '3.16e13']
    radius_labels = ['0.5', '1.0', '3.0', '5.0']
    mass_display = [r'$M > 10^{12}$', r'$M > 10^{12.5}$', 
                    r'$M > 10^{13}$', r'$M > 10^{13.5}$']
    radius_display = [r'$0.5$', r'$1.0$', r'$3.0$', r'$5.0$']
    
    mean_cum_F = results['mean_cum_F']
    F_matrix_cum = np.full((4, 4), np.nan)
    for i, mass in enumerate(mass_labels):
        for j, radius in enumerate(radius_labels):
            model_name = f"hydro_replace_Ml_{mass}_Mu_1.00e15_Ri_0.0_Ro_{radius}"
            if model_name in mean_cum_F:
                F_matrix_cum[i, j] = mean_cum_F[model_name][z_idx]
    
    im = ax.imshow(F_matrix_cum, cmap='RdYlGn', vmin=0.0, vmax=1.0, 
                   aspect='auto', origin='lower')
    
    for i in range(4):
        for j in range(4):
            val = F_matrix_cum[i, j]
            if not np.isnan(val):
                color = 'white' if val < 0.3 or val > 0.7 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center', 
                       color=color, weight='bold')
    
    ax.set_xticks(range(4))
    ax.set_xticklabels(radius_display)
    ax.set_yticks(range(4))
    ax.set_yticklabels(mass_display)
    ax.set_xlabel(r'Radius factor [$R_{200}$]')
    ax.set_ylabel(r'Mass threshold [$h^{-1}M_\odot$]')
    ax.set_title('Cumulative Models', weight='bold')
    
    # Discrete
    ax = axes[1]
    disc_mass_lower = ['1.00e12', '3.16e12', '1.00e13', '3.16e13']
    disc_mass_upper = ['3.16e12', '1.00e13', '3.16e13', '1.00e15']
    disc_radius_inner = ['0.0', '0.5', '1.0', '3.0']
    disc_radius_outer = ['0.5', '1.0', '3.0', '5.0']
    mass_bin_display = [r'$[10^{12}, 10^{12.5}]$', r'$[10^{12.5}, 10^{13}]$', 
                        r'$[10^{13}, 10^{13.5}]$', r'$[10^{13.5}, 10^{15}]$']
    radius_bin_display = [r'[0, 0.5]', r'[0.5, 1]', r'[1, 3]', r'[3, 5]']
    
    mean_disc_F = results['mean_disc_F']
    F_matrix_disc = np.full((4, 4), np.nan)
    for i, (Ml, Mu) in enumerate(zip(disc_mass_lower, disc_mass_upper)):
        for j, (Ri, Ro) in enumerate(zip(disc_radius_inner, disc_radius_outer)):
            model_name = f"hydro_replace_Ml_{Ml}_Mu_{Mu}_Ri_{Ri}_Ro_{Ro}"
            if model_name in mean_disc_F:
                F_matrix_disc[i, j] = mean_disc_F[model_name][z_idx]
    
    im = ax.imshow(F_matrix_disc, cmap='RdYlGn', vmin=0.0, vmax=1.0, 
                   aspect='auto', origin='lower')
    
    for i in range(4):
        for j in range(4):
            val = F_matrix_disc[i, j]
            if not np.isnan(val):
                color = 'white' if val < 0.3 or val > 0.7 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center', 
                       color=color, weight='bold')
    
    ax.set_xticks(range(4))
    ax.set_xticklabels(radius_bin_display, rotation=45, ha='right')
    ax.set_yticks(range(4))
    ax.set_yticklabels(mass_bin_display)
    ax.set_xlabel(r'Radial bin [$R_{200}$]')
    ax.set_ylabel(r'Mass bin [$h^{-1}M_\odot$]')
    ax.set_title('Discrete Models', weight='bold')
    
    fig.colorbar(im, ax=axes, label=r'$\langle F \rangle$', shrink=1.0)

    # plt.tight_layout()
    
    if savename is None:
        savename = f'figures/{stat_name}_cumulative_vs_discrete_comparison.pdf'
    plt.savefig(savename, dpi=150)
    plt.show()

# Usage
def plot_combined_response(results, stat_name='pdf', z_idx=23, z_val=1.0, savename=None):
    """
    Combined 4-row plot:
    Row 1: S_hydro/S_DMO ratio
    Row 2: Cumulative F
    Row 3: Discrete F  
    Row 4: ε = F_cum - Σ(ΔF_disc)
    """
    mass_labels = ['1.00e12', '3.16e12', '1.00e13', '3.16e13']
    radius_labels = ['0.5', '1.0', '3.0', '5.0']
    
    # Discrete model definitions
    disc_mass_lower = ['1.00e12', '3.16e12', '1.00e13', '3.16e13']
    disc_mass_upper = ['3.16e12', '1.00e13', '3.16e13', '1.00e15']
    disc_radius_inner = ['0.0', '0.5', '1.0', '3.0']
    disc_radius_outer = ['0.5', '1.0', '3.0', '5.0']
    
    # Mass values for colorbar
    mass_values = np.array([12.0, 12.5, 13.0, 13.5])  # log10(M)
    
    # Continuous colormap for cumulative (rows 0, 1, 3)
    cum_cmap = plt.cm.get_cmap('jet', 4)
    cum_colors = [cum_cmap(m) for m in range(4)]
    
    # Discrete colormap for discrete row (row 2)
    disc_cmap = plt.cm.get_cmap('jet', 4)
    disc_colors = [disc_cmap(i) for i in range(4)]
    
    # Extract data from results
    cum_mean = results['cum_mean']
    cum_stderr = results['cum_stderr']
    cum_F = results['cum_F']
    cum_F_err = results['cum_F_err']
    disc_mean = results['disc_mean']
    disc_stderr = results['disc_stderr']
    disc_F = results['disc_F']
    disc_F_err = results['disc_F_err']
    base_mean = results['base_mean']
    base_stderr = results['base_stderr']
    bins = results['bins']
    
    fig, axes = plt.subplots(4, 4, figsize=(14, 14), sharex=True, 
                             gridspec_kw={'hspace':0.085, 'wspace':0.13, 'right':0.88})
    
    hydro_data = base_mean['hydro']
    dmo_data = base_mean['dmo']
    hydro_err = base_stderr['hydro']
    dmo_err = base_stderr['dmo']
    
    # Set errorevery based on statistic type
    if stat_name == 'Pk':
        errorevery = 10000
    elif stat_name == 'Cl':
        errorevery = 10
    else:
        errorevery = 1

    # ==================== ROW 0: S_hydro/S_DMO ratio ====================
    for i, radius in enumerate(radius_labels):
        ax = axes[0, i]
        
        ratio = hydro_data[z_idx] / dmo_data[z_idx]
        ratio_err = ratio * np.sqrt((hydro_err[z_idx]/hydro_data[z_idx])**2 + 
                                     (dmo_err[z_idx]/dmo_data[z_idx])**2)
        
        ax.errorbar(bins, ratio, yerr=np.abs(ratio_err), color='black', 
                   ls='--', capsize=2, alpha=0.6, errorevery=errorevery)
        
        if i > 0:
            ax.tick_params(labelleft=False)
            
        for j, mass in enumerate(mass_labels):
            model_name = f"hydro_replace_Ml_{mass}_Mu_1.00e15_Ri_0.0_Ro_{radius}"
            if model_name in cum_mean:
                P = cum_mean[model_name]
                P_err = cum_stderr[model_name]
                
                ratio = P[z_idx] / dmo_data[z_idx]
                ratio_err = ratio * np.sqrt((P_err[z_idx]/P[z_idx])**2 + 
                                           (dmo_err[z_idx]/dmo_data[z_idx])**2)
                
                ax.errorbar(bins, ratio, yerr=np.abs(ratio_err), color=cum_colors[j], 
                           capsize=2, alpha=0.6, errorevery=errorevery)
                ax.grid(alpha=0.3)
                
        ax.set_ylim(0.5, 1.5)
        ax.set_title(rf'$\alpha = {radius}$')
        ax.grid(alpha=0.3)

    axes[0, 0].set_ylabel(f'{stat_name} / {stat_name}' + r'$_{\rm DMO}$')

    # ==================== ROW 1: Cumulative F ====================
    for i, radius in enumerate(radius_labels):
        ax = axes[1, i]
        if i > 0:
            ax.tick_params(labelleft=False)
            
        for j, mass in enumerate(mass_labels):
            model_name = f"hydro_replace_Ml_{mass}_Mu_1.00e15_Ri_0.0_Ro_{radius}"
            if model_name in cum_F:
                F = cum_F[model_name]
                F_err = cum_F_err[model_name]
                
                ax.errorbar(bins, F[z_idx], yerr=np.abs(F_err[z_idx]), 
                           color=cum_colors[j],
                           capsize=2, alpha=0.6, errorevery=errorevery)
                ax.grid(alpha=0.3)
        
        ax.set_ylim(-0.6, 1.6)
        ax.axhline(0, color='k', ls='--', lw=1.)
        ax.axhline(1, color='k', ls='--', lw=1.)

    axes[1, 0].set_ylabel(rf'$F_{{\rm {stat_name}}}^{{\rm cum}}$')

    # ==================== ROW 2: Discrete F ====================
    for i, (Ri, Ro) in enumerate(zip(disc_radius_inner, disc_radius_outer)):
        ax = axes[2, i]
        if i > 0:
            ax.tick_params(labelleft=False)
        
        for j, (Ml, Mu) in enumerate(zip(disc_mass_lower, disc_mass_upper)):
            model_name = f"hydro_replace_Ml_{Ml}_Mu_{Mu}_Ri_{Ri}_Ro_{Ro}"
            if model_name in disc_F:
                F = disc_F[model_name]
                F_err = disc_F_err[model_name]
                
                ax.errorbar(bins, F[z_idx], yerr=np.abs(F_err[z_idx]), 
                           color=disc_colors[j],
                           capsize=2, alpha=0.6, errorevery=errorevery)
                ax.grid(alpha=0.3)
        
        ax.set_ylim(-1.0, 1.0)
        ax.axhline(0, color='k', ls='--', lw=1.)
        # ax.axhline(1, color='k', ls='--', lw=1.)

    axes[2, 0].set_ylabel(rf'$\Delta F_{{\rm {stat_name}}}^{{\rm disc}}$')

    # ==================== ROW 3: Residuals ε = F_cum - Σ(ΔF_disc) ====================
    for i, radius in enumerate(radius_labels):
        ax = axes[3, i]
        
        for j, mass in enumerate(mass_labels):
            cum_model_name = f"hydro_replace_Ml_{mass}_Mu_1.00e15_Ri_0.0_Ro_{radius}"
            
            if cum_model_name in cum_F:
                F_cum = cum_F[cum_model_name][z_idx]
                F_disc_sum = compute_discrete_sum(mass, radius, disc_F, z_idx, bins)
                
                # Residual
                residual = (F_cum - F_disc_sum)/F_cum
                
                ax.plot(bins, residual, color=cum_colors[j], lw=2, alpha=0.8)
        
        ax.axhline(0, color='k', ls='--', lw=1.)
        ax.set_ylim(-0.31, 0.31)
        ax.grid(alpha=0.3)
        
        if i !=0:
            ax.tick_params(labelleft=False)
        
        # X-axis labels only on bottom row
        if stat_name in ['pdf', 'peaks', 'minima', 'V0', 'V1', 'V2', 'N_p', 'N_m']:
            ax.set_xlabel(r'$\nu$ (S/N)')
            ax.set_xlim(-5, 10)
        elif stat_name == 'Cl':
            ax.set_xlabel(r'$\ell$')
            ax.set_xscale('log')
            ax.set_xlim(1e3, 3e5)
        elif stat_name == 'Pk':
            ax.set_xlabel(r'$k\,[h\,{\rm Mpc}^{-1}]$')
            ax.set_xscale('log')
            ax.set_xlim(0.1, 63)
        else:
            ax.set_xlabel('Scale')

    axes[3, 0].set_ylabel(r'$\epsilon$')
    
    # Create ScalarMappable objects for colorbars
    disc_bounds = [0, 1, 2, 3, 4]
    disc_norm_cb = plt.matplotlib.colors.BoundaryNorm(disc_bounds, disc_cmap.N)
    sm_cum = plt.cm.ScalarMappable(cmap=cum_cmap, norm=disc_norm_cb)
    sm_cum.set_array([])
    
    disc_bounds = [0, 1, 2, 3, 4]
    disc_norm_cb = plt.matplotlib.colors.BoundaryNorm(disc_bounds, disc_cmap.N)
    sm_disc = plt.cm.ScalarMappable(cmap=disc_cmap, norm=disc_norm_cb)
    sm_disc.set_array([])
    
    # Get positions from actual subplot axes to align colorbars
    for row_idx in range(4):
        # Get the rightmost subplot position for this row
        pos = axes[row_idx, 3].get_position()
        
        # Create colorbar aligned with this row
        cax = fig.add_axes([0.90, pos.y0, 0.015, pos.height])
        
        if row_idx == 2:  # Discrete row
            cbar = fig.colorbar(sm_disc, cax=cax, ticks=[1.0, 2.0, 3.0, 4.0])
            cbar.ax.set_yticklabels([r'$[12, 12.5]$', r'$[12.5, 13]$', 
                                      r'$[13, 13.5]$', r'$[13.5, \infty]$'], rotation=45)
            cbar.set_label(r'$\log_{10}(M_{\rm bin}/[{\rm M}_\odot])$')
        else:  # Cumulative rows (0, 1, 3)
            # cbar = fig.colorbar(sm_cum, cax=cax)
            # cbar.set_label(r'$\log_{10}(M_{\rm min}/[h^{-1}M_\odot])$')
            # cbar.set_ticks([12.0, 12.5, 13.0, 13.5])
            cbar = fig.colorbar(sm_cum, cax=cax, ticks=[0.5, 1.5, 2.5, 3.5])
            cbar.ax.set_yticklabels([r'$12$', r'$12.5$', 
                                      r'$13$', r'$13.5$'])
            cbar.set_label(r'$\log_{10}(M_{\rm min}/[{\rm M}_\odot])$')
        
    if savename is None:
        savename = f'figures/{stat_name}_combined_response.pdf'
    # plt.savefig(savename, dpi=150, bbox_inches='tight')
    # plt.show()
    
    return fig, axes

def plot_family_cumulative_F(results_dict, family_name, stat_names, stat_labels, 
                             z_idx=23, z_val=1.0, savename=None):
    """
    Plot cumulative F for a family of statistics.
    
    Parameters
    ----------
    results_dict : dict
        Dictionary mapping stat_name -> results from compute_statistic_response()
    family_name : str
        Name of the family (e.g., 'S/N Statistics', 'Minkowski Functionals', 'Power Spectra')
    stat_names : list
        List of statistic names (keys in results_dict)
    stat_labels : list
        Display labels for each statistic (for y-axis)
    z_idx : int
        Redshift index
    z_val : float
        Redshift value for title
    savename : str, optional
        Output filename
    """
    mass_labels = ['1.00e12', '3.16e12', '1.00e13', '3.16e13']
    radius_labels = ['0.5', '1.0', '3.0', '5.0']
    colors = plt.cm.jet(np.linspace(0.0, 1.0, 4))
    
    n_stats = len(stat_names)
    fig, axes = plt.subplots(n_stats, 4, figsize=(24, 6*n_stats), sharex='col',
                             gridspec_kw={'hspace':0.08, 'wspace':0.05})
    
    # Ensure axes is 2D even for single row
    if n_stats == 1:
        axes = axes[np.newaxis, :]
    
    for row, (stat_name, stat_label) in enumerate(zip(stat_names, stat_labels)):
        results = results_dict[stat_name]
        cum_F = results['cum_F']
        cum_F_err = results['cum_F_err']
        bins = results['bins']
        
        # Determine errorevery and x-axis settings
        if stat_name == 'Pk':
            errorevery = 100000
            xlabel = r'$k\,[h\,{\rm Mpc}^{-1}]$'
            xscale = 'log'
            xlim = (0.1, 63)
        elif stat_name == 'Cl':
            errorevery = 10
            xlabel = r'$\ell$'
            xscale = 'log'
            xlim = (1e3, 1e6)
        else:
            errorevery = 1
            xlabel = r'$\nu$ (S/N)'
            xscale = 'linear'
            xlim = (-5, 10)
        
        for col, radius in enumerate(radius_labels):
            ax = axes[row, col]
            
            if col > 0:
                ax.tick_params(labelleft=False)
            
            for j, mass in enumerate(mass_labels):
                model_name = f"hydro_replace_Ml_{mass}_Mu_1.00e15_Ri_0.0_Ro_{radius}"
                if model_name in cum_F:
                    F = cum_F[model_name]
                    F_err = cum_F_err[model_name]
                    
                    ax.errorbar(bins, F[z_idx], yerr=np.abs(F_err[z_idx]), 
                               color=colors[j],
                               label=rf'$M>10^{{{np.log10(float(mass)):.1f}}}$' if row == 0 else None,
                               capsize=2, alpha=0.6, errorevery=errorevery)
            
            ax.axhline(0, color='k', ls='--', lw=1.)
            ax.axhline(1, color='k', ls='--', lw=1.)
            ax.set_ylim(-0.5, 1.5)
            ax.grid(alpha=0.3)
            ax.set_xscale(xscale)
            ax.set_xlim(xlim)
            
            # Column titles on top row only
            if row == 0:
                ax.set_title(rf'$\alpha = {radius}$')
            
            # X-axis labels on bottom row only
            if row == n_stats - 1:
                ax.set_xlabel(xlabel)
        
        # Y-axis label for this row
        axes[row, 0].set_ylabel(rf'$F_{{\rm {stat_label}}}$')
    
    # Legend on first subplot
    axes[0, 0].legend(loc='upper left', ncols=2, title=r'$M_{\rm min}\,[h^{-1}\,M_\odot]$')
    
    
    if savename is None:
        savename = f'figures/{family_name.replace(" ", "_").lower()}_cumulative_F.pdf'
    plt.savefig(savename, dpi=150, bbox_inches='tight')
    plt.show()
    
    return fig, axes


def plot_family_discrete_F(results_dict, family_name, stat_names, stat_labels, 
                           z_idx=23, z_val=1.0, savename=None):
    """
    Plot discrete ΔF for a family of statistics.
    """
    disc_mass_lower = ['1.00e12', '3.16e12', '1.00e13', '3.16e13']
    disc_mass_upper = ['3.16e12', '1.00e13', '3.16e13', '1.00e15']
    disc_radius_inner = ['0.0', '0.5', '1.0', '3.0']
    disc_radius_outer = ['0.5', '1.0', '3.0', '5.0']
    mass_bin_labels = [r'$[10^{12}, 10^{12.5}]$', r'$[10^{12.5}, 10^{13}]$', 
                       r'$[10^{13}, 10^{13.5}]$', r'$[10^{13.5}, \infty]$']
    colors = plt.cm.jet(np.linspace(0.0, 1.0, 4))
    
    n_stats = len(stat_names)
    fig, axes = plt.subplots(n_stats, 4, figsize=(24, 6*n_stats), sharex='col',
                             gridspec_kw={'hspace':0.08, 'wspace':0.05})
    
    if n_stats == 1:
        axes = axes[np.newaxis, :]
    
    for row, (stat_name, stat_label) in enumerate(zip(stat_names, stat_labels)):
        results = results_dict[stat_name]
        disc_F = results['disc_F']
        disc_F_err = results['disc_F_err']
        bins = results['bins']
        
        if stat_name == 'Pk':
            errorevery, xlabel, xscale, xlim = 10000, r'$k\,[h\,{\rm Mpc}^{-1}]$', 'log', (0.1, 63)
        elif stat_name == 'Cl':
            errorevery, xlabel, xscale, xlim = 10, r'$\ell$', 'log', (1e3, 3e5)
        else:
            errorevery, xlabel, xscale, xlim = 1, r'$\nu$ (S/N)', 'linear', (-5, 10)
        
        for col, (Ri, Ro) in enumerate(zip(disc_radius_inner, disc_radius_outer)):
            ax = axes[row, col]
            
            if col > 0:
                ax.tick_params(labelleft=False)
            
            for j, (Ml, Mu) in enumerate(zip(disc_mass_lower, disc_mass_upper)):
                model_name = f"hydro_replace_Ml_{Ml}_Mu_{Mu}_Ri_{Ri}_Ro_{Ro}"
                if model_name in disc_F:
                    F = disc_F[model_name]
                    F_err = disc_F_err[model_name]
                    
                    ax.errorbar(bins, F[z_idx], yerr=np.abs(F_err[z_idx]), 
                               color=colors[j],
                               label=mass_bin_labels[j] if row == 0 else None,
                               capsize=2, alpha=0.6, errorevery=errorevery)
            
            ax.axhline(0, color='k', ls='--', lw=1.)
            ax.axhline(1, color='k', ls='--', lw=1.)
            ax.set_ylim(-0.5, 1.5)
            ax.grid(alpha=0.3)
            ax.set_xscale(xscale)
            ax.set_xlim(xlim)
            
            if row == 0:
                ax.set_title(rf'$r \in [{Ri}, {Ro}] R_{{200}}$')
            if row == n_stats - 1:
                ax.set_xlabel(xlabel)
        
        axes[row, 0].set_ylabel(rf'$\Delta F_{{\rm {stat_label}}}$')
    
    axes[0, 0].legend(loc='upper left', ncols=1, title=r'Mass bin [$h^{-1}M_\odot$]')
    
    if savename is None:
        savename = f'figures/{family_name.replace(" ", "_").lower()}_discrete_F.pdf'
    plt.savefig(savename, dpi=150, bbox_inches='tight')
    plt.show()
    
    return fig, axes


def plot_family_residuals(results_dict, family_name, stat_names, stat_labels, 
                          z_idx=23, z_val=1.0, savename=None):
    """
    Plot residuals ε = F_cum - Σ(ΔF_disc) for a family of statistics.
    """
    mass_labels = ['1.00e12', '3.16e12', '1.00e13', '3.16e13']
    radius_labels = ['0.5', '1.0', '3.0', '5.0']
    colors = plt.cm.jet(np.linspace(0.0, 1.0, 4))
    
    n_stats = len(stat_names)
    fig, axes = plt.subplots(n_stats, 4, figsize=(24, 5*n_stats), sharex='col',
                             gridspec_kw={'hspace':0.08, 'wspace':0.05})
    
    if n_stats == 1:
        axes = axes[np.newaxis, :]
    
    for row, (stat_name, stat_label) in enumerate(zip(stat_names, stat_labels)):
        results = results_dict[stat_name]
        cum_F = results['cum_F']
        disc_F = results['disc_F']
        bins = results['bins']
        
        if stat_name == 'Pk':
            xlabel, xscale, xlim = r'$k\,[h\,{\rm Mpc}^{-1}]$', 'log', (0.1, 63)
        elif stat_name == 'Cl':
            xlabel, xscale, xlim = r'$\ell$', 'log', (1e3, 3e5)
        else:
            xlabel, xscale, xlim = r'$\nu$ (S/N)', 'linear', (-5, 10)
        
        for col, radius in enumerate(radius_labels):
            ax = axes[row, col]
            
            if col > 0:
                ax.tick_params(labelleft=False)
            
            for j, mass in enumerate(mass_labels):
                cum_model_name = f"hydro_replace_Ml_{mass}_Mu_1.00e15_Ri_0.0_Ro_{radius}"
                
                if cum_model_name in cum_F:
                    F_cum = cum_F[cum_model_name][z_idx]
                    F_disc_sum = compute_discrete_sum(mass, radius, disc_F, z_idx, bins)
                    residual = F_cum - F_disc_sum
                    
                    ax.plot(bins, residual, color=colors[j], lw=2, alpha=0.8,
                           label=rf'$M > 10^{{{np.log10(float(mass)):.1f}}}$' if row == 0 else None)
            
            ax.axhline(0, color='k', ls='--', lw=1.)
            ax.set_ylim(-0.5, 0.5)
            ax.grid(alpha=0.3)
            ax.set_xscale(xscale)
            ax.set_xlim(xlim)
            
            if row == 0:
                ax.set_title(rf'$\alpha = {radius}$')
            if row == n_stats - 1:
                ax.set_xlabel(xlabel)
        
        axes[row, 0].set_ylabel(rf'$\epsilon_{{\rm {stat_label}}}$')
    
    axes[0, 0].legend(loc='upper left', ncols=2, title=r'$M_{\rm min}$')
    
    if savename is None:
        savename = f'figures/{family_name.replace(" ", "_").lower()}_residuals.pdf'
    plt.savefig(savename, dpi=150, bbox_inches='tight')
    plt.show()
    
    return fig, axes


def plot_family_redshift_evolution(results_dict, family_name, stat_names, stat_labels,
                                   model_type='cumulative', weighting='frac_signal',
                                   kappa_redshifts=None, savename=None):
    """
    Plot redshift evolution of mean F for a family of statistics.
    
    Parameters
    ----------
    results_dict : dict
        Dictionary mapping stat_name -> results from compute_statistic_response()
    family_name : str
        Name of the family (e.g., 'Power Spectra', 'S/N Statistics')
    stat_names : list
        List of statistic names (keys in results_dict)
    stat_labels : list
        Display labels for each statistic (for y-axis)
    model_type : str
        Either 'cumulative' or 'discrete'
    weighting : str
        Weighting method: 'snr', 'frac_signal', 'signal', 'inverse_variance', 'uniform'
    kappa_redshifts : dict, optional
        Dictionary mapping indices to redshift values
    savename : str, optional
        Output filename
    """
    # Default redshift mapping
    if kappa_redshifts is None:
        kappa_redshifts = {
            1: 0.034, 2: 0.070, 3: 0.105, 4: 0.142, 5: 0.179,
            6: 0.216, 7: 0.255, 8: 0.294, 9: 0.335, 10: 0.376,
            11: 0.418, 12: 0.462, 13: 0.506, 14: 0.552, 15: 0.599,
            16: 0.648, 17: 0.698, 18: 0.749, 19: 0.803, 20: 0.858,
            21: 0.914, 22: 0.973, 23: 1.034, 24: 1.097, 25: 1.163,
            26: 1.231, 27: 1.302, 28: 1.375, 29: 1.452, 30: 1.532,
            31: 1.615, 32: 1.703, 33: 1.794, 34: 1.889, 35: 1.989,
            36: 2.094, 37: 2.203, 38: 2.319, 39: 2.440, 40: 2.568
        }
    
    redshifts_all = list(kappa_redshifts.values())
    z_indices_all = np.arange(len(redshifts_all))
    
    colors = plt.cm.jet(np.linspace(0.0, 1.0, 4))
    
    if model_type == 'cumulative':
        return _plot_family_cumulative_redshift_evolution(
            results_dict, family_name, stat_names, stat_labels,
            redshifts_all, z_indices_all, colors, weighting, savename
        )
    elif model_type == 'discrete':
        return _plot_family_discrete_redshift_evolution(
            results_dict, family_name, stat_names, stat_labels,
            redshifts_all, z_indices_all, colors, weighting, savename
        )
    elif model_type == 'both':
        return _plot_family_both_redshift_evolution(
            results_dict, family_name, stat_names, stat_labels,
            redshifts_all, z_indices_all, colors, weighting, savename
        )
    else:
        raise ValueError(f"model_type must be 'cumulative', 'discrete', or 'both', got {model_type}")


def _plot_family_cumulative_redshift_evolution(results_dict, family_name, stat_names, stat_labels,
                                                redshifts_all, z_indices_all, colors, 
                                                weighting, savename):
    """Plot cumulative model redshift evolution for a family of statistics."""
    mass_labels = ['1.00e12', '3.16e12', '1.00e13', '3.16e13']
    radius_labels = ['0.5', '1.0', '3.0', '5.0']
    mass_display = [r'$M > 10^{12}$', r'$M > 10^{12.5}$', 
                    r'$M > 10^{13}$', r'$M > 10^{13.5}$']
    radius_display = [r'$0.5$', r'$1.0$', r'$3.0$', r'$5.0$']
    
    n_stats = len(stat_names)
    fig, axes = plt.subplots(n_stats, 4, figsize=(20, 5*n_stats), sharex=True, sharey=True,
                             gridspec_kw={'hspace':0.08, 'wspace':0.05})
    
    if n_stats == 1:
        axes = axes[np.newaxis, :]
    
    for row, (stat_name, stat_label) in enumerate(zip(stat_names, stat_labels)):
        results = results_dict[stat_name]
        
        for col, radius in enumerate(radius_labels):
            ax = axes[row, col]
            
            if col > 0:
                ax.tick_params(labelleft=False)
            
            for j, mass in enumerate(mass_labels):
                model_name = f"hydro_replace_Ml_{mass}_Mu_1.00e15_Ri_0.0_Ro_{radius}"
                
                # Collect F values across all redshifts
                F_vs_z = []
                F_err_vs_z = []
                
                for z_idx in z_indices_all:
                    F_val, F_err = _get_F_value(results, model_name, z_idx, weighting, 'cum')
                    F_vs_z.append(F_val)
                    F_err_vs_z.append(F_err if not np.isnan(F_err) else 0)
                
                # Plot with errorbars
                if any(e > 0 for e in F_err_vs_z):
                    ax.errorbar(redshifts_all, F_vs_z, yerr=np.abs(F_err_vs_z), 
                               fmt='-', color=colors[j], 
                               label=mass_display[j] if row == 0 else None, 
                               linewidth=2, markersize=6, capsize=3, alpha=0.7)
                else:
                    ax.plot(redshifts_all, F_vs_z, '-', color=colors[j], 
                           label=mass_display[j] if row == 0 else None, 
                           linewidth=2, markersize=6, alpha=0.7)
            
            ax.set_xscale('log')
            ax.axhline(1, color='k', ls='--', lw=1.)
            ax.axhline(0, color='k', ls='--', lw=1.)
            ax.set_ylim(-0.2, 1.5)
            ax.grid(alpha=0.3)
            
            # Column titles on top row only
            if row == 0:
                ax.set_title(r'$\alpha = $' + radius_display[col] + r' $R_{200}$')
            
            # X-axis labels on bottom row only
            if row == n_stats - 1:
                ax.set_xlabel(r'$z$')
        
        # Y-axis label for this row
        axes[row, 0].set_ylabel(rf'$\langle F \rangle_{{\rm {stat_label}}}$')
    
    # Legend on first subplot
    axes[0, 0].legend(loc='upper left', ncols=2, title=r'$M_{\rm min}\,[h^{-1}\,M_\odot]$')
    
    if savename is None:
        savename = f'figures/{family_name.replace(" ", "_").lower()}_cumulative_z_evolution.pdf'
    plt.savefig(savename, dpi=150, bbox_inches='tight')
    plt.show()
    
    return fig, axes


def _plot_family_discrete_redshift_evolution(results_dict, family_name, stat_names, stat_labels,
                                              redshifts_all, z_indices_all, colors, 
                                              weighting, savename):
    """Plot discrete model redshift evolution for a family of statistics."""
    disc_mass_lower = ['1.00e12', '3.16e12', '1.00e13', '3.16e13']
    disc_mass_upper = ['3.16e12', '1.00e13', '3.16e13', '1.00e15']
    disc_radius_inner = ['0.0', '0.5', '1.0', '3.0']
    disc_radius_outer = ['0.5', '1.0', '3.0', '5.0']
    
    mass_bin_labels = [r'$[10^{12}, 10^{12.5}]$', r'$[10^{12.5}, 10^{13}]$', 
                       r'$[10^{13}, 10^{13.5}]$', r'$[10^{13.5}, \infty]$']
    radius_bin_display = [r'[0, 0.5]', r'[0.5, 1]', r'[1, 3]', r'[3, 5]']
    
    n_stats = len(stat_names)
    fig, axes = plt.subplots(n_stats, 4, figsize=(20, 5*n_stats), sharex=True, sharey=True,
                             gridspec_kw={'hspace':0.08, 'wspace':0.05})
    
    if n_stats == 1:
        axes = axes[np.newaxis, :]
    
    for row, (stat_name, stat_label) in enumerate(zip(stat_names, stat_labels)):
        results = results_dict[stat_name]
        
        for col, (Ri, Ro) in enumerate(zip(disc_radius_inner, disc_radius_outer)):
            ax = axes[row, col]
            
            if col > 0:
                ax.tick_params(labelleft=False)
            
            for j, (Ml, Mu) in enumerate(zip(disc_mass_lower, disc_mass_upper)):
                model_name = f"hydro_replace_Ml_{Ml}_Mu_{Mu}_Ri_{Ri}_Ro_{Ro}"
                
                # Collect F values across all redshifts
                F_vs_z = []
                F_err_vs_z = []
                
                for z_idx in z_indices_all:
                    F_val, F_err = _get_F_value(results, model_name, z_idx, weighting, 'disc')
                    F_vs_z.append(F_val)
                    F_err_vs_z.append(F_err if not np.isnan(F_err) else 0)
                
                # Plot with errorbars
                ax.errorbar(redshifts_all, F_vs_z, yerr=np.abs(F_err_vs_z), 
                           fmt='-', color=colors[j], 
                           label=mass_bin_labels[j] if row == 0 else None,
                           linewidth=2, markersize=6, capsize=3, alpha=0.7)
            
            ax.set_xscale('log')
            ax.axhline(1, color='k', ls='--', lw=1.)
            ax.axhline(0, color='k', ls='--', lw=1.)
            ax.set_ylim(-0.2, 1.5)
            ax.grid(alpha=0.3)
            
            # Column titles on top row only
            if row == 0:
                ax.set_title(r'$r \in $' + radius_bin_display[col] + r' $R_{200}$')
            
            # X-axis labels on bottom row only
            if row == n_stats - 1:
                ax.set_xlabel(r'$z$')
        
        # Y-axis label for this row
        axes[row, 0].set_ylabel(rf'$\langle F \rangle_{{\rm {stat_label}}}$')
    
    # Legend on first subplot
    axes[0, 0].legend(loc='upper left', ncols=1, title=r'Mass bin [$h^{-1}M_\odot$]')
    
    if savename is None:
        savename = f'figures/{family_name.replace(" ", "_").lower()}_discrete_z_evolution.pdf'
    plt.savefig(savename, dpi=150, bbox_inches='tight')
    plt.show()
    
    return fig, axes


def _plot_family_both_redshift_evolution(results_dict, family_name, stat_names, stat_labels,
                                          redshifts_all, z_indices_all, colors, 
                                          weighting, savename):
    """Plot both cumulative (top) and discrete (bottom) redshift evolution for a family of statistics."""
    mass_labels = ['1.00e12', '3.16e12', '1.00e13', '3.16e13']
    radius_labels = ['0.5', '1.0', '3.0', '5.0']
    radius_display = [r'$0.5$', r'$1.0$', r'$3.0$', r'$5.0$']
    
    disc_mass_lower = ['1.00e12', '3.16e12', '1.00e13', '3.16e13']
    disc_mass_upper = ['3.16e12', '1.00e13', '3.16e13', '1.00e15']
    disc_radius_inner = ['0.0', '0.5', '1.0', '3.0']
    disc_radius_outer = ['0.5', '1.0', '3.0', '5.0']
    radius_bin_display = [r'[0, 0.5]', r'[0.5, 1]', r'[1, 3]', r'[3, 5]']
    
    # Mass values for colorbar
    mass_values = np.array([12.0, 12.5, 13.0, 13.5])  # log10(M)
    
    # Continuous colormap for cumulative (mapped to mass threshold)
    cum_cmap = plt.cm.get_cmap('jet', 4)
    cum_colors = [cum_cmap(m) for m in range(4)]
    
    # Discrete colormap for discrete bins (4 distinct colors)
    disc_cmap = plt.cm.get_cmap('jet', 4)
    disc_colors = [disc_cmap(i) for i in range(4)]
    
    n_stats = len(stat_names)
    # 2 rows: top = cumulative, bottom = discrete
    # Add extra space on right for colorbars
    fig, axes = plt.subplots(2, 4, figsize=(14, 6), sharex=True,
                             gridspec_kw={'hspace':0.15, 'wspace':0.05, 'right':0.88})
    
    # Use first statistic for single-stat case
    stat_name = stat_names[0]
    stat_label = stat_labels[0]
    results = results_dict[stat_name]
    
    # ============= ROW 0: CUMULATIVE =============
    for col, radius in enumerate(radius_labels):
        ax = axes[0, col]
        
        if col > 0:
            ax.tick_params(labelleft=False)
        
        for j, mass in enumerate(mass_labels):
            model_name = f"hydro_replace_Ml_{mass}_Mu_1.00e15_Ri_0.0_Ro_{radius}"
            
            # Collect F values across all redshifts
            F_vs_z = []
            F_err_vs_z = []
            
            for z_idx in z_indices_all:
                F_val, F_err = _get_F_value(results, model_name, z_idx, weighting, 'cum')
                F_vs_z.append(F_val)
                F_err_vs_z.append(F_err if not np.isnan(F_err) else 0)
            
            # Plot with errorbars using continuous colormap
            if any(e > 0 for e in F_err_vs_z):
                ax.errorbar(redshifts_all, F_vs_z, yerr=np.abs(F_err_vs_z), 
                           fmt='-', color=cum_colors[j], 
                           linewidth=2, markersize=6, capsize=3, alpha=0.8)
            else:
                ax.plot(redshifts_all, F_vs_z, '-', color=cum_colors[j], 
                       linewidth=2, markersize=6, alpha=0.8)
        
        ax.set_xscale('log')
        ax.axhline(1, color='k', ls='--', lw=1.)
        ax.axhline(0, color='k', ls='--', lw=1.)
        ax.set_ylim(-0.2, 1.5)
        ax.grid(alpha=0.3)
        
        # Column titles showing cumulative radius
        ax.set_title(r'$\alpha = $' + radius_display[col] + r' $R_{200}$')
    
    # Y-axis label for cumulative row
    axes[0, 0].set_ylabel(rf'$\langle F \rangle_{{\rm {stat_label}}}^{{\rm cum}}$')
    
    # Continuous colorbar for cumulative row
    cax_cum = fig.add_axes([0.90, 0.53, 0.015, 0.35])  # [left, bottom, width, height]
    cum_bounds = [0, 1, 2, 3, 4]
    cum_norm = plt.matplotlib.colors.BoundaryNorm(cum_bounds, cum_cmap.N)
    sm_cum = plt.cm.ScalarMappable(cmap=cum_cmap, norm=cum_norm)
    sm_cum.set_array([])
    cbar_cum = fig.colorbar(sm_cum, cax=cax_cum, ticks=[0.5, 1.5, 2.5, 3.5])
    cbar_cum.ax.set_yticklabels([r'$12$', r'$12.5$', 
                                  r'$13$', r'$13.5$'])
    cbar_cum.set_label(r'$\log_{10}(M_{\rm min}/[{\rm M}_\odot])$')
    
    # ============= ROW 1: DISCRETE =============
    for col, (Ri, Ro) in enumerate(zip(disc_radius_inner, disc_radius_outer)):
        ax = axes[1, col]
        
        if col > 0:
            ax.tick_params(labelleft=False)
        
        for j, (Ml, Mu) in enumerate(zip(disc_mass_lower, disc_mass_upper)):
            model_name = f"hydro_replace_Ml_{Ml}_Mu_{Mu}_Ri_{Ri}_Ro_{Ro}"
            
            # Collect F values across all redshifts
            F_vs_z = []
            F_err_vs_z = []
            
            for z_idx in z_indices_all:
                F_val, F_err = _get_F_value(results, model_name, z_idx, weighting, 'disc')
                F_vs_z.append(F_val)
                F_err_vs_z.append(F_err if not np.isnan(F_err) else 0)
            
            # Plot with errorbars using discrete colormap
            ax.errorbar(redshifts_all, F_vs_z, yerr=np.abs(F_err_vs_z), 
                       fmt='-', color=disc_colors[j],
                       linewidth=2, markersize=6, capsize=3, alpha=0.8)
        
        ax.set_xscale('log')
        ax.axhline(1, color='k', ls='--', lw=1.)
        ax.axhline(0, color='k', ls='--', lw=1.)
        ax.set_ylim(-1., 1.)
        ax.grid(alpha=0.3)
        
        # Column titles showing discrete radial bins
        ax.set_title(r'$r \in $' + radius_bin_display[col] + r' $R_{200}$')
        
        # X-axis labels on bottom row
        ax.set_xlabel(r'$z$')
    
    # Y-axis label for discrete row
    axes[1, 0].set_ylabel(rf'$\langle F \rangle_{{\rm {stat_label}}}^{{\rm disc}}$')
    
    # Discrete colorbar for discrete row
    cax_disc = fig.add_axes([0.90, 0.11, 0.015, 0.35])  # [left, bottom, width, height]
    disc_bounds = [0, 1, 2, 3, 4]
    disc_norm = plt.matplotlib.colors.BoundaryNorm(disc_bounds, disc_cmap.N)
    sm_disc = plt.cm.ScalarMappable(cmap=disc_cmap, norm=disc_norm)
    sm_disc.set_array([])
    cbar_disc = fig.colorbar(sm_disc, cax=cax_disc, ticks=[1., 2., 3., 4.])
    cbar_disc.ax.set_yticklabels([r'$[12, 12.5]$', r'$[12.5, 13]$', 
                                  r'$[13, 13.5]$', r'$[13.5, \infty]$'], rotation=45)
    cbar_disc.set_label(r'$\log_{10}(M_{\rm bin}/[{\rm M}_\odot])$')
    
    if savename is None:
        savename = f'figures/{family_name.replace(" ", "_").lower()}_both_z_evolution.pdf'
    # plt.show()
    
    return fig, axes