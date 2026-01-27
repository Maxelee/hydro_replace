"""
Analysis Utilities for Hydro Replace Notebooks
=============================================

Unified data loading and analysis functions extracted from active notebooks.
Provides consistent interfaces for:
- Loading stats.h5 data
- Loading raw lensplanes  
- Building model names
- Computing response fractions
- Bootstrap error estimation

Usage:
    from analysis_utils import (
        load_stats, load_lensplane,
        get_cumulative_models, get_discrete_models,
        compute_F, compute_F_peaks, bootstrap_F,
        STATS_BASE, MASS_THRESHOLDS, RADII
    )
    
    # Load base statistics
    dmo = load_stats('dmo')
    hydro = load_stats('hydro')
    
    # Load Replace model
    model = get_cumulative_models()[0]  # e.g., M > 10^12, R < 0.5
    replace = load_stats(model)
    
    # Compute response fraction
    F = compute_F(replace['Pk'], dmo['Pk'], hydro['Pk'])
"""

import numpy as np
import h5py
from pathlib import Path
from typing import Optional, Dict, List, Union, Tuple
import warnings


# =============================================================================
# CONSTANTS & CONFIGURATION
# =============================================================================

# Data paths
STATS_BASE = Path('/mnt/home/mlee1/ceph/hydro_replace_stats')
LP_BASE = Path('/mnt/home/mlee1/ceph/hydro_replace_LP/L205n2500TNG')
RT_BASE = Path('/mnt/home/mlee1/ceph/hydro_replace_RT/L205n2500TNG')

# Box and grid parameters
BOX_SIZE = 205.0  # Mpc/h
GRID_LP = 4096    # Lensplane resolution
GRID_RT = 1024    # Ray-tracing output resolution
FOV_DEG = 5.0     # Field of view in degrees
N_LP = 20         # Lensplane orientations (LP_00 to LP_19)
N_RUNS = 100      # Ray-traced maps per LP (run001 to run100)

# Nyquist frequencies
K_NYQUIST = np.pi * GRID_LP / BOX_SIZE  # ~62.8 h/Mpc for density
ELL_NYQUIST = np.pi * GRID_RT / (FOV_DEG * np.pi / 180)  # ~36864 for kappa

# Mass thresholds (string format for model names)
MASS_THRESHOLDS = ['1.00e12', '3.16e12', '1.00e13', '3.16e13']
MASS_THRESHOLDS_FLOAT = [1.00e12, 3.16e12, 1.00e13, 3.16e13]
MASS_LABELS = [r'$10^{12.0}$', r'$10^{12.5}$', r'$10^{13.0}$', r'$10^{13.5}$']
LOG_MASS = [12.0, 12.5, 13.0, 13.5]

# Radius factors
RADII = ['0.5', '1.0', '3.0', '5.0']
RADII_FLOAT = [0.5, 1.0, 3.0, 5.0]
RADIUS_LABELS = [r'$\alpha=0.5$', r'$\alpha=1$', r'$\alpha=3$', r'$\alpha=5$']

# Discrete bins
DISCRETE_MASS_BINS = [
    ('1.00e12', '3.16e12'),
    ('3.16e12', '1.00e13'),
    ('1.00e13', '3.16e13'),
    ('3.16e13', '1.00e15')
]
DISCRETE_RADIUS_BINS = [
    ('0.0', '0.5'),
    ('0.5', '1.0'),
    ('1.0', '3.0'),
    ('3.0', '5.0')
]

# Source redshift mapping (kappa file index -> z_s)
KAPPA_REDSHIFTS = {
    13: 0.51, 20: 0.86, 25: 1.16, 30: 1.53, 35: 1.99, 40: 2.57
}


# =============================================================================
# MODEL NAME BUILDERS
# =============================================================================

def build_model_name(Ml: str, Mu: str, Ri: str, Ro: str) -> str:
    """Build model name from mass/radius bounds.
    
    Parameters
    ----------
    Ml : str
        Mass lower bound, e.g., '1.00e12'
    Mu : str
        Mass upper bound, e.g., '1.00e15' 
    Ri : str
        Radius inner bound, e.g., '0.0'
    Ro : str
        Radius outer bound, e.g., '3.0'
        
    Returns
    -------
    str
        Model name, e.g., 'hydro_replace_Ml_1.00e12_Mu_1.00e15_Ri_0.0_Ro_3.0'
    """
    return f"hydro_replace_Ml_{Ml}_Mu_{Mu}_Ri_{Ri}_Ro_{Ro}"


def get_cumulative_models() -> List[str]:
    """Get list of 16 cumulative Replace models.
    
    Cumulative models: M > M_min, r < alpha*R200
    Format: Ml={mass}, Mu=1.00e15, Ri=0.0, Ro={radius}
    
    Returns
    -------
    list
        16 model names ordered by (mass, radius)
    """
    models = []
    for mass in MASS_THRESHOLDS:
        for ro in RADII:
            models.append(build_model_name(mass, '1.00e15', '0.0', ro))
    return models


def get_discrete_models() -> List[str]:
    """Get list of 16 discrete tile models.
    
    Discrete models: M_min < M < M_max, r_min < r < r_max
    
    Returns
    -------
    list
        16 model names ordered by (mass_bin, radius_bin)
    """
    models = []
    for ml, mu in DISCRETE_MASS_BINS:
        for ri, ro in DISCRETE_RADIUS_BINS:
            models.append(build_model_name(ml, mu, ri, ro))
    return models


def get_all_models() -> List[str]:
    """Get list of all 102 models (dmo, hydro, 100 replace).
    
    Returns
    -------
    list
        All model names: dmo, hydro, then replace models
    """
    # Base models
    models = ['dmo', 'hydro']
    
    # All unique replace models (100 total from mass × radius grid)
    for i_m, ml in enumerate(MASS_THRESHOLDS + ['1.00e15'][:0]):  # Only use thresholds
        for j_m, mu in enumerate(['3.16e12', '1.00e13', '3.16e13', '1.00e15']):
            if float(mu) <= float(ml):
                continue
            for i_r, ri in enumerate(['0.0', '0.5', '1.0', '3.0']):
                for j_r, ro in enumerate(['0.5', '1.0', '3.0', '5.0']):
                    if float(ro) <= float(ri):
                        continue
                    models.append(build_model_name(ml, mu, ri, ro))
    
    return models


def parse_model_name(model_name: str) -> Optional[Dict]:
    """Parse a model name to extract parameters.
    
    Parameters
    ----------
    model_name : str
        Model name like 'hydro_replace_Ml_1.00e12_Mu_1.00e15_Ri_0.0_Ro_3.0'
        
    Returns
    -------
    dict or None
        {'Ml': float, 'Mu': float, 'Ri': float, 'Ro': float} or None if invalid
    """
    if not model_name.startswith('hydro_replace_Ml_'):
        return None
    try:
        parts = model_name.split('_')
        return {
            'Ml': float(parts[3]),
            'Mu': float(parts[5]),
            'Ri': float(parts[7]),
            'Ro': float(parts[9])
        }
    except (IndexError, ValueError):
        return None


# =============================================================================
# DATA LOADERS
# =============================================================================

def load_stats(
    model: str,
    keys: Optional[List[str]] = None,
    base_path: Path = STATS_BASE
) -> Optional[Dict[str, np.ndarray]]:
    """Load statistics from stats.h5 file.
    
    Parameters
    ----------
    model : str
        Model name, e.g., 'dmo', 'hydro', or 'hydro_replace_Ml_...'
    keys : list, optional
        Specific keys to load. If None, loads all.
    base_path : Path
        Base directory containing model folders
        
    Returns
    -------
    dict or None
        Dictionary of arrays, or None if file not found
        
    Examples
    --------
    >>> dmo = load_stats('dmo')
    >>> dmo['Pk'].shape
    (20, 20, 200)  # (LP, planes, k_bins)
    
    >>> peaks_only = load_stats('hydro', keys=['peaks', 'sn_bin_edges'])
    """
    path = base_path / model / 'stats.h5'
    if not path.exists():
        warnings.warn(f"Stats file not found: {path}")
        return None
    
    try:
        with h5py.File(path, 'r') as f:
            if keys is None:
                keys = list(f.keys())
            data = {key: f[key][()] for key in keys if key in f}
        return data if data else None
    except Exception as e:
        warnings.warn(f"Error loading {path}: {e}")
        return None


def load_all_stats(
    models: List[str],
    keys: Optional[List[str]] = None
) -> Dict[str, Dict[str, np.ndarray]]:
    """Load statistics for multiple models.
    
    Parameters
    ----------
    models : list
        List of model names
    keys : list, optional
        Specific keys to load
        
    Returns
    -------
    dict
        {model_name: stats_dict} for successfully loaded models
    """
    results = {}
    for model in models:
        stats = load_stats(model, keys=keys)
        if stats is not None:
            results[model] = stats
    return results


def load_lensplane(filepath: Union[str, Path]) -> Optional[np.ndarray]:
    """Load lux-format binary lensplane file.
    
    Parameters
    ----------
    filepath : str or Path
        Path to lenspot*.dat or lensplane_*.npz file
        
    Returns
    -------
    ndarray or None
        2D array (grid_size, grid_size)
        
    Examples
    --------
    >>> plane = load_lensplane(LP_BASE / 'dmo' / 'LP_00' / 'lenspot00.dat')
    >>> plane.shape
    (4096, 4096)
    """
    filepath = Path(filepath)
    if not filepath.exists():
        return None
    
    try:
        if filepath.suffix == '.npz':
            data = np.load(filepath)
            return data['plane'] if 'plane' in data else data['arr_0']
        else:
            # Binary format
            with open(filepath, 'rb') as f:
                grid_size = np.fromfile(f, dtype=np.int32, count=1)[0]
                data = np.fromfile(f, dtype=np.float64, count=grid_size**2)
            return data.reshape(grid_size, grid_size)
    except Exception as e:
        warnings.warn(f"Error loading lensplane {filepath}: {e}")
        return None


def load_kappa(
    model: str,
    lp_idx: int,
    run_idx: int,
    kappa_idx: int = 20,
    grid: int = 1024
) -> Optional[np.ndarray]:
    """Load convergence map from ray-tracing output.
    
    Parameters
    ----------
    model : str
        Model name
    lp_idx : int
        Lens plane index (0-19)
    run_idx : int
        Run index (1-100)
    kappa_idx : int
        Kappa file index (1-40). Default 20 ≈ z_s ~ 0.86
    grid : int
        Grid size (default 1024)
        
    Returns
    -------
    ndarray or None
        2D convergence map (grid, grid)
    """
    path = RT_BASE / model / f'LP_{lp_idx:02d}' / f'run{run_idx:03d}' / f'kappa{kappa_idx:02d}.dat'
    if not path.exists():
        return None
    
    try:
        with open(path, 'rb') as f:
            _ = np.fromfile(f, dtype="int32", count=1)  # dummy
            kappa = np.fromfile(f, dtype="float", count=grid*grid)
            _ = np.fromfile(f, dtype="int32", count=1)  # dummy
        return kappa.reshape(grid, grid)
    except Exception as e:
        warnings.warn(f"Error loading kappa {path}: {e}")
        return None


# =============================================================================
# RESPONSE FRACTION COMPUTATION
# =============================================================================

def compute_F(
    S_replace: np.ndarray,
    S_dmo: np.ndarray,
    S_hydro: np.ndarray,
    threshold: float = 0.02,
    min_denom: float = 1e-10
) -> np.ndarray:
    """Compute response fraction F = (Replace - DMO) / (Hydro - DMO).
    
    Parameters
    ----------
    S_replace : ndarray
        Statistic from Replace model
    S_dmo : ndarray
        Statistic from DMO simulation
    S_hydro : ndarray
        Statistic from Hydro simulation
    threshold : float
        Minimum |ΔS/S_dmo| to consider significant (default 0.02 = 2%)
    min_denom : float
        Minimum |S_hydro - S_dmo| to avoid division by zero
        
    Returns
    -------
    ndarray
        Response fraction F with NaN where insignificant
    """
    denom = S_hydro - S_dmo
    F = np.full_like(S_dmo, np.nan, dtype=float)
    
    # Significance mask
    significant = (np.abs(denom) > min_denom) & (np.abs(denom / S_dmo) > threshold)
    
    F[significant] = (S_replace[significant] - S_dmo[significant]) / denom[significant]
    return F


def compute_F_scale_dependent(
    S_replace: np.ndarray,
    S_dmo: np.ndarray,
    S_hydro: np.ndarray,
    x_bins: np.ndarray,
    x_nyquist: float,
    threshold: float = 0.02
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute scale-dependent response F(k) or F(ℓ) with Nyquist masking.
    
    Parameters
    ----------
    S_replace, S_dmo, S_hydro : ndarray
        Statistics with shape (N_samples, N_bins) or (N_bins,)
    x_bins : ndarray
        Scale bins (k or ℓ)
    x_nyquist : float
        Nyquist frequency (mask x > x_nyquist)
    threshold : float
        Significance threshold
        
    Returns
    -------
    F_mean : ndarray
        Mean response fraction
    F_std : ndarray
        Standard deviation across samples
    """
    # Ensure 2D
    if S_replace.ndim == 1:
        S_replace = S_replace[np.newaxis, :]
        S_dmo = S_dmo[np.newaxis, :]
        S_hydro = S_hydro[np.newaxis, :]
    
    # Compute F for each sample
    n_samples, n_bins = S_replace.shape
    F_all = np.full((n_samples, n_bins), np.nan)
    
    for i in range(n_samples):
        F_all[i] = compute_F(S_replace[i], S_dmo[i], S_hydro[i], threshold)
    
    # Mask beyond Nyquist
    F_all[:, x_bins >= x_nyquist] = np.nan
    
    # Compute mean and std
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        F_mean = np.nanmean(F_all, axis=0)
        F_std = np.nanstd(F_all, axis=0)
    
    return F_mean, F_std


def compute_cumulative_counts(counts: np.ndarray) -> np.ndarray:
    """Compute cumulative counts N(>x) from histogram counts.
    
    Parameters
    ----------
    counts : ndarray
        Histogram counts per bin
        
    Returns
    -------
    ndarray
        Cumulative counts from high to low
        
    Examples
    --------
    >>> counts = np.array([100, 50, 20, 5])  # bins at x=1,2,3,4
    >>> compute_cumulative_counts(counts)
    array([175,  75,  25,   5])  # N(>1), N(>2), N(>3), N(>4)
    """
    return np.cumsum(counts[::-1])[::-1]


def compute_F_peaks(
    peaks_replace: np.ndarray,
    peaks_dmo: np.ndarray,
    peaks_hydro: np.ndarray,
    z_idx: int,
    sn_threshold: float = 5.0,
    sn_edges: Optional[np.ndarray] = None,
    min_delta: int = 50
) -> float:
    """Compute response fraction for cumulative peak counts N(>ν).
    
    Parameters
    ----------
    peaks_replace, peaks_dmo, peaks_hydro : ndarray
        Peak count arrays, shape (N_real, N_z, N_bins)
    z_idx : int
        Redshift index
    sn_threshold : float
        S/N threshold ν
    sn_edges : ndarray, optional
        S/N bin edges (needed if sn_threshold not at bin center)
    min_delta : int
        Minimum |N_hydro - N_dmo| to consider significant
        
    Returns
    -------
    float
        Response fraction F, or NaN if insignificant
    """
    if sn_edges is not None:
        sn_centers = 0.5 * (sn_edges[:-1] + sn_edges[1:])
        idx = np.argmin(np.abs(sn_centers - sn_threshold))
    else:
        idx = int(sn_threshold)  # Assume bin index
    
    # Sum over realizations, compute cumulative
    N_dmo = compute_cumulative_counts(peaks_dmo[:, z_idx, :].sum(axis=0))[idx]
    N_hydro = compute_cumulative_counts(peaks_hydro[:, z_idx, :].sum(axis=0))[idx]
    N_replace = compute_cumulative_counts(peaks_replace[:, z_idx, :].sum(axis=0))[idx]
    
    delta = N_hydro - N_dmo
    if np.abs(delta) < min_delta:
        return np.nan
    
    return (N_replace - N_dmo) / delta


# =============================================================================
# BOOTSTRAP & STATISTICS
# =============================================================================

def bootstrap_F(
    S_replace: np.ndarray,
    S_dmo: np.ndarray,
    S_hydro: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute F and uncertainty via bootstrap over realizations.
    
    Parameters
    ----------
    S_replace, S_dmo, S_hydro : ndarray
        Statistics with shape (N_real, ...) where first axis is realizations
    n_bootstrap : int
        Number of bootstrap samples
    seed : int
        Random seed for reproducibility
        
    Returns
    -------
    F_mean : ndarray
        Mean response fraction
    F_std : ndarray
        Bootstrap standard deviation
    """
    n_real = S_replace.shape[0]
    other_shape = S_replace.shape[1:]
    
    rng = np.random.default_rng(seed)
    F_samples = []
    
    for _ in range(n_bootstrap):
        idx = rng.choice(n_real, size=n_real, replace=True)
        
        # Sum over resampled realizations
        sum_R = np.sum(S_replace[idx], axis=0)
        sum_D = np.sum(S_dmo[idx], axis=0)
        sum_H = np.sum(S_hydro[idx], axis=0)
        
        denom = sum_H - sum_D
        with np.errstate(divide='ignore', invalid='ignore'):
            F = (sum_R - sum_D) / denom
            F[np.abs(denom) < 1e-10] = np.nan
        F_samples.append(F)
    
    F_samples = np.array(F_samples)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(F_samples, axis=0), np.nanstd(F_samples, axis=0)


def weighted_mean(
    values: np.ndarray,
    x_bins: np.ndarray,
    x_range: Tuple[float, float] = None,
    weights: Optional[np.ndarray] = None
) -> Tuple[float, float]:
    """Compute weighted mean of values over x range.
    
    Parameters
    ----------
    values : ndarray
        Values to average
    x_bins : ndarray
        Corresponding x coordinates
    x_range : tuple, optional
        (x_min, x_max) to restrict averaging
    weights : ndarray, optional
        Weights for averaging (default: uniform)
        
    Returns
    -------
    mean : float
        Weighted mean
    std : float
        Weighted standard deviation
    """
    mask = ~np.isnan(values)
    if x_range is not None:
        mask &= (x_bins >= x_range[0]) & (x_bins <= x_range[1])
    
    if not np.any(mask):
        return np.nan, np.nan
    
    v = values[mask]
    w = weights[mask] if weights is not None else np.ones_like(v)
    
    mean = np.average(v, weights=w)
    variance = np.average((v - mean)**2, weights=w)
    return mean, np.sqrt(variance)


# =============================================================================
# PLOTTING UTILITIES
# =============================================================================

def setup_plot_style():
    """Set up publication-quality matplotlib style.
    
    Uses scienceplots if available, otherwise falls back to clean defaults.
    """
    import matplotlib.pyplot as plt
    
    try:
        import scienceplots
        plt.style.use(['science', 'notebook'])
    except ImportError:
        plt.rcParams.update({
            'figure.figsize': (8, 6),
            'figure.dpi': 100,
            'font.size': 11,
            'axes.labelsize': 12,
            'axes.titlesize': 12,
            'legend.fontsize': 10,
            'axes.grid': True,
            'grid.alpha': 0.3
        })


def get_mass_colors(n: int = 4) -> np.ndarray:
    """Get consistent color scheme for mass bins.
    
    Parameters
    ----------
    n : int
        Number of colors (default 4 for mass bins)
        
    Returns
    -------
    ndarray
        RGBA colors from viridis colormap
    """
    import matplotlib.pyplot as plt
    return plt.cm.viridis(np.linspace(0.15, 0.85, n))


def get_radius_colors(n: int = 4) -> np.ndarray:
    """Get consistent color scheme for radius bins.
    
    Parameters
    ----------
    n : int
        Number of colors (default 4 for radius bins)
        
    Returns
    -------
    ndarray
        RGBA colors from plasma colormap
    """
    import matplotlib.pyplot as plt
    return plt.cm.plasma(np.linspace(0.15, 0.85, n))


# =============================================================================
# QUICK ACCESS FUNCTIONS
# =============================================================================

def load_base_stats(keys: Optional[List[str]] = None) -> Dict[str, Dict]:
    """Quick loader for DMO and Hydro base statistics.
    
    Parameters
    ----------
    keys : list, optional
        Specific keys to load
        
    Returns
    -------
    dict
        {'dmo': {...}, 'hydro': {...}}
    """
    return {
        'dmo': load_stats('dmo', keys),
        'hydro': load_stats('hydro', keys)
    }


def load_cumulative_stats(keys: Optional[List[str]] = None) -> Dict[str, Dict]:
    """Quick loader for all 16 cumulative Replace models.
    
    Parameters
    ----------
    keys : list, optional
        Specific keys to load
        
    Returns
    -------
    dict
        {model_name: stats_dict}
    """
    return load_all_stats(get_cumulative_models(), keys)


def load_discrete_stats(keys: Optional[List[str]] = None) -> Dict[str, Dict]:
    """Quick loader for all 16 discrete tile models.
    
    Parameters
    ----------
    keys : list, optional
        Specific keys to load
        
    Returns
    -------
    dict
        {model_name: stats_dict}
    """
    return load_all_stats(get_discrete_models(), keys)


# =============================================================================
# MODULE INFO
# =============================================================================

__all__ = [
    # Constants
    'STATS_BASE', 'LP_BASE', 'RT_BASE',
    'BOX_SIZE', 'GRID_LP', 'GRID_RT', 'FOV_DEG',
    'K_NYQUIST', 'ELL_NYQUIST',
    'N_LP', 'N_RUNS',
    'MASS_THRESHOLDS', 'MASS_THRESHOLDS_FLOAT', 'MASS_LABELS', 'LOG_MASS',
    'RADII', 'RADII_FLOAT', 'RADIUS_LABELS',
    'DISCRETE_MASS_BINS', 'DISCRETE_RADIUS_BINS',
    'KAPPA_REDSHIFTS',
    
    # Model builders
    'build_model_name', 'parse_model_name',
    'get_cumulative_models', 'get_discrete_models', 'get_all_models',
    
    # Data loaders
    'load_stats', 'load_all_stats',
    'load_lensplane', 'load_kappa',
    'load_base_stats', 'load_cumulative_stats', 'load_discrete_stats',
    
    # Response functions
    'compute_F', 'compute_F_scale_dependent', 'compute_F_peaks',
    'compute_cumulative_counts',
    
    # Statistics
    'bootstrap_F', 'weighted_mean',
    
    # Plotting
    'setup_plot_style', 'get_mass_colors', 'get_radius_colors',
]


if __name__ == '__main__':
    # Quick test
    print("Analysis Utils Module")
    print("=" * 50)
    print(f"Stats base: {STATS_BASE}")
    print(f"K Nyquist: {K_NYQUIST:.1f} h/Mpc")
    print(f"ℓ Nyquist: {ELL_NYQUIST:.0f}")
    print(f"\nCumulative models: {len(get_cumulative_models())}")
    print(f"Discrete models: {len(get_discrete_models())}")
    print(f"\nExample cumulative: {get_cumulative_models()[0]}")
    print(f"Example discrete: {get_discrete_models()[0]}")
