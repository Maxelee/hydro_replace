"""
baryonic_response.py

Halo-space response formalism for baryonic effects on cosmological statistics.

This module implements the mathematical framework for quantifying how
baryonic modifications in specific halo mass-radius tiles affect arbitrary
summary statistics (power spectra, weak lensing observables, peak counts, etc.).

Key equations implemented:
- Eq. (1): Hybrid density field rho(lambda)
- Eq. (4): Cumulative response fraction F_S(M_min, alpha)
- Eq. (9): Tile response fraction Delta F_S(a,i)
- Eq. (12): Additive approximation from tiles
- Eq. (13): Non-additivity epsilon_S
- Eq. (14): Halo-space multipoles mu_S(a,i)

Author: [Your name]
Date: December 2025
"""

from typing import Dict, Tuple, Callable, Optional, List, Union
import numpy as np
from scipy.interpolate import interp1d
import warnings

# For power spectrum computation
try:
    import Pylians3 as pyl
    PYLIANS_AVAILABLE = True
except ImportError:
    PYLIANS_AVAILABLE = False
    warnings.warn("Pylians3 not available. Power spectrum functions will be disabled.")


# ============================================================================
# Type Aliases
# ============================================================================
TileKey = Tuple[int, int]  # (mass_bin_index, radius_shell_index)
LambdaPattern = Dict[TileKey, float]
StatisticFn = Callable[[np.ndarray], np.ndarray]


# ============================================================================
# Reference Constants for Weak Lensing Statistics
# ============================================================================
# Computed from 500 DMO convergence maps at z_s ~ 1, 2 arcmin Gaussian smoothing
KAPPA_RMS_DMO = 0.0107  # Reference κ standard deviation for SNR normalization

# Default SNR bins for peaks and minima (based on where baryonic signal is detected)
PEAK_SNR_BINS = np.linspace(-2, 6, 17)    # 16 bins from SNR -2 to +6
MINIMA_SNR_BINS = np.linspace(-6, 2, 17)  # 16 bins from SNR -6 to +2


# ============================================================================
# Statistics Computation
# ============================================================================

def measure_power_spectrum(
    density_field: np.ndarray,
    BoxSize: float,
    axis: int = 0,
    MAS: str = 'CIC',
    threads: int = 1,
    verbose: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute 3D matter power spectrum using Pylians.
    
    Parameters
    ----------
    density_field : np.ndarray, shape (N, N, N)
        3D density field (can be overdensity delta = rho/<rho> - 1 or raw density).
    BoxSize : float
        Simulation box size in Mpc/h.
    axis : int, optional
        Axis along which to compute the power spectrum (default: 0).
    MAS : str, optional
        Mass assignment scheme used in the simulation ('NGP', 'CIC', 'TSC', 'PCS').
    threads : int, optional
        Number of OpenMP threads.
    verbose : bool, optional
        Print progress information.
        
    Returns
    -------
    k : np.ndarray, shape (Nbins,)
        Wavenumber bins in h/Mpc.
    Pk : np.ndarray, shape (Nbins,)
        Power spectrum P(k) in (Mpc/h)^3.
        
    Notes
    -----
    Uses Pylians3.PKL.Pk to compute the spherically averaged power spectrum.
    If input is raw density, it will be converted to overdensity internally.
    
    References
    ----------
    Pylians: https://pylians3.readthedocs.io/
    """
    if not PYLIANS_AVAILABLE:
        raise ImportError("Pylians3 is required for power spectrum computation. "
                         "Install with: pip install Pylians3")
    
    # Ensure density_field is float32 for Pylians
    delta = density_field.astype(np.float32, copy=False)
    
    # Convert to overdensity if needed (Pylians expects delta)
    mean_density = np.mean(delta)
    if mean_density > 0.1:  # Heuristic: if mean >> 0, likely raw density not overdensity
        if verbose:
            print(f"Converting raw density (mean={mean_density:.3e}) to overdensity delta")
        delta = delta / mean_density - 1.0
    
    # Compute power spectrum
    Pk_obj = pyl.PKL.Pk(delta, BoxSize, axis=axis, MAS=MAS, threads=threads, verbose=verbose)
    
    return Pk_obj.k3D, Pk_obj.Pk[:,0]  # k in h/Mpc, Pk in (Mpc/h)^3


def measure_convergence_power_spectrum(
    kappa_map: np.ndarray,
    pixel_size_arcmin: float,
    lmin: int = 10,
    lmax: int = 5000,
    nbins: int = 50
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute convergence power spectrum C_ell from a 2D kappa map.
    
    Parameters
    ----------
    kappa_map : np.ndarray, shape (Npix, Npix)
        2D convergence (kappa) map.
    pixel_size_arcmin : float
        Pixel size in arcminutes.
    lmin, lmax : int
        Minimum and maximum multipole ell.
    nbins : int
        Number of logarithmic bins in ell.
        
    Returns
    -------
    ell : np.ndarray, shape (nbins,)
        Multipole bin centers.
    C_ell : np.ndarray, shape (nbins,)
        Angular power spectrum.
        
    Notes
    -----
    Uses 2D FFT and azimuthal averaging in Fourier space.
    Assumes flat-sky approximation.
    
    Implements:
        C_ell = <|kappa_ell|^2> where kappa_ell = FFT(kappa_map)
    """
    # FFT
    kappa_fft = np.fft.fft2(kappa_map)
    kappa_fft = np.fft.fftshift(kappa_fft)
    
    # Power spectrum
    Pk_2d = np.abs(kappa_fft)**2
    
    # Pixel size in radians
    pixel_size_rad = pixel_size_arcmin / 60.0 * np.pi / 180.0
    map_size_rad = pixel_size_rad * kappa_map.shape[0]
    
    # Build ell grid
    Npix = kappa_map.shape[0]
    freq = np.fft.fftfreq(Npix, d=pixel_size_rad)
    freq = np.fft.fftshift(freq)
    kx, ky = np.meshgrid(freq, freq, indexing='ij')
    ell_map = 2.0 * np.pi * np.sqrt(kx**2 + ky**2)
    
    # Azimuthal average
    ell_bins = np.logspace(np.log10(lmin), np.log10(lmax), nbins+1)
    ell_centers = 0.5 * (ell_bins[:-1] + ell_bins[1:])
    
    C_ell = np.zeros(nbins)
    for i in range(nbins):
        mask = (ell_map >= ell_bins[i]) & (ell_map < ell_bins[i+1])
        if np.sum(mask) > 0:
            C_ell[i] = np.mean(Pk_2d[mask])
    
    # Normalize by map area
    C_ell *= (map_size_rad**2 / Npix**2)
    
    return ell_centers, C_ell


def measure_peak_counts(
    kappa_map: np.ndarray,
    nu_bins: np.ndarray = None,
    kappa_rms: float = None,
    sigma_noise: float = 0.0  # Deprecated, use kappa_rms
) -> np.ndarray:
    """
    Measure weak lensing peak counts in SNR (nu) bins.
    
    Parameters
    ----------
    kappa_map : np.ndarray, shape (Npix, Npix)
        2D convergence map (can be smoothed).
    nu_bins : np.ndarray, shape (Nbins+1,), optional
        Peak height (signal-to-noise) bin edges.
        Default: PEAK_SNR_BINS (np.linspace(-2, 6, 17))
    kappa_rms : float, optional
        Reference kappa RMS for SNR normalization. Should be computed from 
        a DMO ensemble to ensure consistent SNR bins across models.
        Default: KAPPA_RMS_DMO (0.0107, from 500 DMO maps at 2 arcmin smoothing)
    sigma_noise : float, optional
        DEPRECATED: Use kappa_rms instead. Kept for backward compatibility.
        
    Returns
    -------
    N_peaks : np.ndarray, shape (Nbins,)
        Number of peaks in each nu bin (normalized by map area if desired).
        
    Notes
    -----
    A peak is defined as a local maximum (pixel higher than all 8 neighbors).
    
    IMPORTANT: Using a constant reference kappa_rms (computed from DMO ensemble)
    ensures that SNR bins correspond to the same physical kappa thresholds 
    across all models (DMO, Hydro, Replace). This is critical for measuring
    baryonic response accurately.
    
    References
    ----------
    - Jain & Van Waerbeke 2000, ApJ, 530, L1
    - Dietrich & Hartlap 2010, MNRAS, 402, 1049
    """
    from scipy.ndimage import maximum_filter
    
    # Use defaults if not specified
    if nu_bins is None:
        nu_bins = PEAK_SNR_BINS
    
    # Determine sigma for SNR normalization
    if kappa_rms is not None:
        sigma = kappa_rms
    elif sigma_noise > 0:
        # Backward compatibility
        sigma = sigma_noise
    else:
        # Use module-level reference (recommended)
        sigma = KAPPA_RMS_DMO
    
    # Identify local maxima
    max_filtered = maximum_filter(kappa_map, size=3, mode='constant', cval=-np.inf)
    is_peak = (kappa_map == max_filtered) & (kappa_map > 0)
    
    # Convert to SNR
    nu_map = kappa_map / sigma
    peak_heights = nu_map[is_peak]
    
    # Histogram
    N_peaks, _ = np.histogram(peak_heights, bins=nu_bins)
    
    return N_peaks.astype(float)


def measure_minima_counts(
    kappa_map: np.ndarray,
    nu_bins: np.ndarray = None,
    kappa_rms: float = None,
    sigma_noise: float = 0.0  # Deprecated, use kappa_rms
) -> np.ndarray:
    """
    Measure weak lensing minima (valley) counts in SNR bins.
    
    Parameters
    ----------
    kappa_map : np.ndarray, shape (Npix, Npix)
        2D convergence map.
    nu_bins : np.ndarray, shape (Nbins+1,), optional
        Depth (nu) bin edges for minima.
        Default: MINIMA_SNR_BINS (np.linspace(-6, 2, 17))
    kappa_rms : float, optional
        Reference kappa RMS for SNR normalization. Should be computed from 
        a DMO ensemble to ensure consistent SNR bins across models.
        Default: KAPPA_RMS_DMO (0.0107, from 500 DMO maps at 2 arcmin smoothing)
    sigma_noise : float, optional
        DEPRECATED: Use kappa_rms instead. Kept for backward compatibility.
        
    Returns
    -------
    N_minima : np.ndarray, shape (Nbins,)
        Number of minima in each depth bin.
        
    Notes
    -----
    A minimum is a local minimum (pixel lower than all 8 neighbors).
    
    The SNR is computed as nu = kappa / sigma (NOT -kappa/sigma), so minima
    have negative SNR values. The nu_bins should span negative values 
    (e.g., -6 to 2) to capture the minima distribution.
    
    IMPORTANT: Using a constant reference kappa_rms (computed from DMO ensemble)
    ensures that SNR bins correspond to the same physical kappa thresholds 
    across all models (DMO, Hydro, Replace).
    """
    from scipy.ndimage import minimum_filter
    
    # Use defaults if not specified
    if nu_bins is None:
        nu_bins = MINIMA_SNR_BINS
    
    # Determine sigma for SNR normalization
    if kappa_rms is not None:
        sigma = kappa_rms
    elif sigma_noise > 0:
        sigma = sigma_noise
    else:
        sigma = KAPPA_RMS_DMO
    
    min_filtered = minimum_filter(kappa_map, size=3, mode='constant', cval=np.inf)
    is_minimum = (kappa_map == min_filtered) & (kappa_map < 0)
    
    # SNR with natural sign (minima have negative nu)
    nu_map = kappa_map / sigma
    minimum_depths = nu_map[is_minimum]
    
    N_minima, _ = np.histogram(minimum_depths, bins=nu_bins)
    
    return N_minima.astype(float)


# ============================================================================
# Core Formalism Functions
# ============================================================================

def compute_baseline_stats(
    rho_D: np.ndarray,
    rho_H: np.ndarray,
    stat_fn: StatisticFn,
    **stat_kwargs
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute baseline DMO, Hydro, and difference statistics.
    
    Implements Eq. (2) and (8):
        S_D = S[rho_D]
        S_H = S[rho_H]
        Delta S = S_H - S_D
    
    Parameters
    ----------
    rho_D : np.ndarray
        DMO density field (3D) or kappa map (2D).
    rho_H : np.ndarray
        Hydro density field or kappa map.
    stat_fn : callable
        Function that takes a field and returns a statistic array.
    **stat_kwargs
        Additional keyword arguments passed to stat_fn.
        
    Returns
    -------
    S_D : np.ndarray
        DMO statistic.
    S_H : np.ndarray
        Hydro statistic.
    Delta_S : np.ndarray
        S_H - S_D (total baryonic effect).
        
    Examples
    --------
    >>> def my_stat(field):
    ...     k, Pk = measure_power_spectrum(field, BoxSize=300.0)
    ...     return Pk
    >>> S_D, S_H, Delta_S = compute_baseline_stats(rho_dmo, rho_hydro, my_stat)
    """
    S_D = stat_fn(rho_D, **stat_kwargs)
    S_H = stat_fn(rho_H, **stat_kwargs)
    Delta_S = S_H - S_D
    return S_D, S_H, Delta_S


def cumulative_response_fraction(
    S_R: np.ndarray,
    S_D: np.ndarray,
    S_H: np.ndarray,
    mask_threshold: float = 0.0
) -> np.ndarray:
    """
    Compute cumulative response fraction F_S(M_min, alpha).
    
    Implements Eq. (4):
        F_S = (S_R - S_D) / (S_H - S_D)
    
    Parameters
    ----------
    S_R : np.ndarray
        Statistic from Replace field.
    S_D : np.ndarray
        DMO statistic.
    S_H : np.ndarray
        Hydro statistic.
    mask_threshold : float, optional
        Mask bins where |S_H - S_D| / |S_D| < mask_threshold.
        
    Returns
    -------
    F_S : np.ndarray
        Cumulative response fraction. NaN where denominator is small.
        
    Notes
    -----
    Values outside [0,1] indicate overshooting or undershooting,
    suggesting compensation from other tiles or non-linear effects.
    """
    denom = S_H - S_D
    
    # Mask small signals
    if mask_threshold > 0:
        small_signal = np.abs(denom / (np.abs(S_D) + 1e-30)) < mask_threshold
        denom = np.where(small_signal, np.nan, denom)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        F_S = (S_R - S_D) / denom
    
    return F_S


def tile_response_fraction(
    S_tile: np.ndarray,
    S_D: np.ndarray,
    Delta_S: np.ndarray,
    mask_threshold: float = 0.0
) -> np.ndarray:
    """
    Compute tile response fraction Delta F_S(a,i).
    
    Implements Eq. (9):
        Delta F_S(a,i) = (S_tile - S_D) / (S_H - S_D)
    
    Parameters
    ----------
    S_tile : np.ndarray
        Statistic from tile-only Replace field.
    S_D : np.ndarray
        DMO statistic.
    Delta_S : np.ndarray
        S_H - S_D.
    mask_threshold : float, optional
        Mask bins where |Delta_S| / |S_D| < mask_threshold.
        
    Returns
    -------
    Delta_F_S : np.ndarray
        Tile response fraction. Approximates R_S^(1)(a,i) / Delta S.
        
    Notes
    -----
    This is the finite-difference estimate of the first-order response
    coefficient for tile (a,i), normalized by the total baryonic effect.
    """
    if mask_threshold > 0:
        small_signal = np.abs(Delta_S / (np.abs(S_D) + 1e-30)) < mask_threshold
        Delta_S = np.where(small_signal, np.nan, Delta_S)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        Delta_F_S = (S_tile - S_D) / Delta_S
    
    return Delta_F_S


def lambda_pattern_from_Mmin_alpha(
    M_min: float,
    alpha_max: float,
    mass_bin_edges: np.ndarray,
    alpha_edges: np.ndarray
) -> LambdaPattern:
    """
    Build lambda_{a,i}(M_min, alpha) pattern for cumulative replacement.
    
    Implements Eq. (10):
        lambda_{a,i} = 1 if M_a >= M_min and alpha_{i+1} <= alpha_max, else 0
    
    Parameters
    ----------
    M_min : float
        Minimum halo mass threshold in M_sun/h.
    alpha_max : float
        Maximum radius factor (in units of R_200).
    mass_bin_edges : np.ndarray, shape (N_M + 1,)
        Halo mass bin edges in M_sun/h.
    alpha_edges : np.ndarray, shape (N_alpha + 1,)
        Radius factor bin edges (dimensionless).
        
    Returns
    -------
    lambda_pattern : dict[(int, int), float]
        Dictionary mapping (mass_bin_index, radius_shell_index) -> {0.0, 1.0}.
        
    Examples
    --------
    >>> mass_edges = np.array([1e12, 1e12.5, 1e13, 1e13.5, 1e14])
    >>> alpha_edges = np.array([0.0, 0.5, 1.0, 2.0, 3.0, 5.0])
    >>> lam = lambda_pattern_from_Mmin_alpha(1e13, 2.0, mass_edges, alpha_edges)
    >>> lam[(2, 1)]  # M >= 1e13, alpha shell [1.0, 2.0)
    1.0
    """
    lambda_pattern = {}
    N_M = len(mass_bin_edges) - 1
    N_alpha = len(alpha_edges) - 1
    
    for a in range(N_M):
        M_lo = mass_bin_edges[a]
        if M_lo < M_min:
            continue
        for i in range(N_alpha):
            alpha_hi = alpha_edges[i + 1]
            lam = 1.0 if alpha_hi <= alpha_max else 0.0
            lambda_pattern[(a, i)] = lam
    
    return lambda_pattern


def additive_response_from_tiles(
    lambda_pattern: LambdaPattern,
    tile_responses: Dict[TileKey, np.ndarray]
) -> np.ndarray:
    """
    Reconstruct cumulative response from tile responses (additive approximation).
    
    Implements Eq. (12):
        F_S^(lin)(M_min, alpha) = sum_{a,i} Delta F_S(a,i) * lambda_{a,i}
    
    Parameters
    ----------
    lambda_pattern : dict[(int, int), float]
        Tile activation pattern.
    tile_responses : dict[(int, int), np.ndarray]
        Delta F_S(a,i) for each tile.
        
    Returns
    -------
    F_lin : np.ndarray
        Additive approximation to cumulative response.
        
    Notes
    -----
    This is the first-order prediction. Deviations from the true F_S
    are encoded in the non-additivity epsilon_S (second-order effects).
    """
    # Initialize with zeros using template from any tile
    any_tile = next(iter(tile_responses.values()))
    F_lin = np.zeros_like(any_tile)
    
    for key, lam in lambda_pattern.items():
        if lam == 0.0 or key not in tile_responses:
            continue
        F_lin += lam * tile_responses[key]
    
    return F_lin


def non_additivity(
    F_true: np.ndarray,
    F_lin: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Compute fractional non-additivity epsilon_S.
    
    Implements Eq. (13):
        epsilon_S = (F_true - F_lin) / F_true
    
    Parameters
    ----------
    F_true : np.ndarray
        True cumulative response fraction from Replace field.
    F_lin : np.ndarray
        Additive approximation from tiles.
    mask : np.ndarray, optional
        Boolean mask for valid bins (e.g., where |F_true| > threshold).
        
    Returns
    -------
    epsilon_S : np.ndarray
        Fractional non-additivity. NaN where F_true ~ 0.
        
    Notes
    -----
    |epsilon_S| << 1: tiles contribute additively (first-order dominates)
    |epsilon_S| >> 1: strong non-linear coupling between tiles (second-order important)
    """
    if mask is not None:
        F_true = np.where(mask, F_true, np.nan)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        eps = (F_true - F_lin) / F_true
    
    return eps


def halo_space_multipoles(
    tile_responses: Dict[TileKey, np.ndarray],
    mask: Optional[np.ndarray] = None
) -> Dict[TileKey, np.ndarray]:
    """
    Compute normalized halo-space multipoles mu_S(a,i).
    
    Implements Eq. (14):
        mu_S(a,i) = Delta F_S(a,i) / sum_{b,j} Delta F_S(b,j)
    
    Parameters
    ----------
    tile_responses : dict[(int, int), np.ndarray]
        Delta F_S(a,i) for each tile.
    mask : np.ndarray, optional
        Boolean mask for statistic bins where normalization is valid.
        
    Returns
    -------
    mu_dict : dict[(int, int), np.ndarray]
        Normalized multipole coefficients, sum_{a,i} mu_S(a,i) ~ 1.
        
    Notes
    -----
    These are "baryonic multipole moments in halo space" - they describe
    how the total baryonic effect is distributed over (M, alpha) tiles.
    Compare mu_S across different statistics S to see which tiles matter for each.
    """
    keys = list(tile_responses.keys())
    tiles_array = np.stack([tile_responses[k] for k in keys], axis=0)
    
    if mask is not None:
        tiles_array = np.where(mask, tiles_array, 0.0)
    
    sum_tiles = np.sum(tiles_array, axis=0)
    
    mu_dict = {}
    for idx, key in enumerate(keys):
        mu = np.zeros_like(sum_tiles)
        valid = (np.abs(sum_tiles) > 1e-30)
        mu[valid] = tiles_array[idx][valid] / sum_tiles[valid]
        mu_dict[key] = mu
    
    return mu_dict


# ============================================================================
# High-Level Analysis Functions
# ============================================================================

class BaryonicResponseAnalysis:
    """
    High-level class for baryonic response analysis.
    
    This class manages:
    - Baseline DMO and Hydro fields/statistics
    - Tile-only Replace fields and responses
    - Cumulative Replace configurations
    - Additivity tests
    - Halo-space multipole computations
    
    Parameters
    ----------
    mass_bin_edges : np.ndarray
        Halo mass bin edges in M_sun/h (logarithmic spacing recommended).
    alpha_edges : np.ndarray
        Radius factor bin edges (e.g., [0, 0.5, 1, 2, 3, 5]).
    redshift : float
        Redshift of the analysis.
    stat_fn : callable
        Function to compute statistic from a field: stat_fn(field, **kwargs) -> array.
    stat_kwargs : dict, optional
        Keyword arguments passed to stat_fn.
        
    Attributes
    ----------
    S_D, S_H, Delta_S : np.ndarray
        Baseline DMO, Hydro, and difference statistics.
    tile_responses : dict
        Delta F_S(a,i) for each tile.
    cumulative_responses : dict
        F_S(M_min, alpha) for each cumulative configuration.
    """
    
    def __init__(
        self,
        mass_bin_edges: np.ndarray,
        alpha_edges: np.ndarray,
        redshift: float,
        stat_fn: StatisticFn,
        stat_kwargs: Optional[dict] = None
    ):
        self.mass_bin_edges = np.asarray(mass_bin_edges)
        self.alpha_edges = np.asarray(alpha_edges)
        self.redshift = redshift
        self.stat_fn = stat_fn
        self.stat_kwargs = stat_kwargs or {}
        
        # Will be populated
        self.S_D = None
        self.S_H = None
        self.Delta_S = None
        self.tile_responses = {}
        self.cumulative_responses = {}
        self.additivity_results = {}
        
    def set_baseline(
        self,
        rho_D: np.ndarray,
        rho_H: np.ndarray
    ):
        """
        Compute and store baseline DMO and Hydro statistics.
        
        Parameters
        ----------
        rho_D : np.ndarray
            DMO density field or kappa map.
        rho_H : np.ndarray
            Hydro density field or kappa map.
        """
        self.S_D, self.S_H, self.Delta_S = compute_baseline_stats(
            rho_D, rho_H, self.stat_fn, **self.stat_kwargs
        )
        print(f"[Baseline] S_D shape: {self.S_D.shape}, "
              f"mean |Delta S / S_D|: {np.nanmean(np.abs(self.Delta_S / self.S_D)):.3f}")
    
    def add_tile(
        self,
        tile_key: TileKey,
        rho_tile: np.ndarray
    ):
        """
        Compute and store tile response for a single tile.
        
        Parameters
        ----------
        tile_key : (int, int)
            (mass_bin_index, radius_shell_index)
        rho_tile : np.ndarray
            Replace field with only this tile activated.
        """
        if self.S_D is None:
            raise ValueError("Must call set_baseline() first.")
        
        S_tile = self.stat_fn(rho_tile, **self.stat_kwargs)
        Delta_F = tile_response_fraction(S_tile, self.S_D, self.Delta_S)
        self.tile_responses[tile_key] = Delta_F
        
        # Report
        a, i = tile_key
        M_lo = self.mass_bin_edges[a]
        M_hi = self.mass_bin_edges[a+1]
        alpha_lo = self.alpha_edges[i]
        alpha_hi = self.alpha_edges[i+1]
        print(f"[Tile {tile_key}] M=[{M_lo:.2e}, {M_hi:.2e}], "
              f"alpha=[{alpha_lo:.2f}, {alpha_hi:.2f}], "
              f"mean Delta F: {np.nanmean(Delta_F):.3f}")
    
    def add_cumulative(
        self,
        M_min: float,
        alpha_max: float,
        rho_R: np.ndarray,
        label: Optional[str] = None
    ):
        """
        Compute and store cumulative response for a given (M_min, alpha).
        
        Parameters
        ----------
        M_min : float
            Minimum halo mass in M_sun/h.
        alpha_max : float
            Maximum radius factor.
        rho_R : np.ndarray
            Replace field with all halos M >= M_min out to alpha_max.
        label : str, optional
            Custom label for storage (default: "M{M_min:.1e}_a{alpha_max:.1f}").
        """
        if self.S_D is None:
            raise ValueError("Must call set_baseline() first.")
        
        S_R = self.stat_fn(rho_R, **self.stat_kwargs)
        F_S = cumulative_response_fraction(S_R, self.S_D, self.S_H)
        
        if label is None:
            label = f"M{M_min:.1e}_a{alpha_max:.1f}"
        
        self.cumulative_responses[label] = {
            'M_min': M_min,
            'alpha_max': alpha_max,
            'F_S': F_S,
            'S_R': S_R
        }
        
        print(f"[Cumulative {label}] mean F_S: {np.nanmean(F_S):.3f}")
    
    def test_additivity(
        self,
        M_min: float,
        alpha_max: float,
        cumulative_label: Optional[str] = None
    ):
        """
        Test additivity: compare true F_S to sum of tiles.
        
        Parameters
        ----------
        M_min : float
            Minimum halo mass for cumulative configuration.
        alpha_max : float
            Maximum radius factor.
        cumulative_label : str, optional
            Label of cumulative response to test (must already exist).
            
        Returns
        -------
        epsilon_S : np.ndarray
            Fractional non-additivity.
        """
        if cumulative_label is None:
            cumulative_label = f"M{M_min:.1e}_a{alpha_max:.1f}"
        
        if cumulative_label not in self.cumulative_responses:
            raise ValueError(f"Cumulative response '{cumulative_label}' not found. "
                           f"Call add_cumulative() first.")
        
        F_true = self.cumulative_responses[cumulative_label]['F_S']
        
        # Build lambda pattern and additive prediction
        lam_pattern = lambda_pattern_from_Mmin_alpha(
            M_min, alpha_max, self.mass_bin_edges, self.alpha_edges
        )
        F_lin = additive_response_from_tiles(lam_pattern, self.tile_responses)
        
        eps = non_additivity(F_true, F_lin)
        
        self.additivity_results[cumulative_label] = {
            'F_true': F_true,
            'F_lin': F_lin,
            'epsilon': eps
        }
        
        print(f"[Additivity {cumulative_label}] mean |epsilon|: {np.nanmean(np.abs(eps)):.3f}")
        
        return eps
    
    def get_multipoles(self, mask: Optional[np.ndarray] = None) -> Dict[TileKey, np.ndarray]:
        """
        Compute normalized halo-space multipoles from tile responses.
        
        Parameters
        ----------
        mask : np.ndarray, optional
            Mask for valid statistic bins.
            
        Returns
        -------
        mu_dict : dict
            mu_S(a,i) for each tile.
        """
        return halo_space_multipoles(self.tile_responses, mask=mask)
    
    def summary(self):
        """Print summary of stored results."""
        print("\n=== Baryonic Response Analysis Summary ===")
        print(f"Redshift: {self.redshift}")
        print(f"Mass bins: {len(self.mass_bin_edges)-1}")
        print(f"Radius shells: {len(self.alpha_edges)-1}")
        print(f"Tiles stored: {len(self.tile_responses)}")
        print(f"Cumulative configs: {len(self.cumulative_responses)}")
        print(f"Additivity tests: {len(self.additivity_results)}")
        
        if self.S_D is not None:
            print(f"\nBaseline statistic shape: {self.S_D.shape}")
            print(f"Mean baryonic effect: {np.nanmean(self.Delta_S):.3e}")


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    """
    Example: Power spectrum analysis
    """
    # Dummy data for demonstration
    BoxSize = 300.0  # Mpc/h
    Ngrid = 256
    
    # Create dummy fields (replace with your actual data)
    rho_D = np.random.randn(Ngrid, Ngrid, Ngrid) * 0.1 + 1.0
    rho_H = rho_D + np.random.randn(Ngrid, Ngrid, Ngrid) * 0.05
    
    # Define mass and radius bins
    mass_edges = np.logspace(12, 14.5, 6)  # M_sun/h
    alpha_edges = np.array([0.0, 0.5, 1.0, 2.0, 3.0, 5.0])
    
    # Define statistic function
    def power_stat(field):
        k, Pk = measure_power_spectrum(field, BoxSize=BoxSize, threads=4)
        # Return only small-scale power for simplicity
        mask = (k >= 1.0) & (k <= 30.0)
        return Pk[mask]
    
    # Initialize analysis
    analysis = BaryonicResponseAnalysis(
        mass_bin_edges=mass_edges,
        alpha_edges=alpha_edges,
        redshift=0.0,
        stat_fn=power_stat
    )
    
    # Set baseline
    analysis.set_baseline(rho_D, rho_H)
    
    # Add tile (example: mass bin 2, radius shell 1)
    # In practice, load your precomputed tile-only Replace fields
    rho_tile_example = rho_D + 0.3 * (rho_H - rho_D)  # dummy
    analysis.add_tile((2, 1), rho_tile_example)
    
    # Add cumulative (M > 1e13, alpha = 2.0)
    rho_R_example = rho_D + 0.8 * (rho_H - rho_D)  # dummy
    analysis.add_cumulative(1e13, 2.0, rho_R_example)
    
    # Test additivity
    analysis.test_additivity(1e13, 2.0)
    
    # Get multipoles
    mu = analysis.get_multipoles()
    
    # Summary
    analysis.summary()
