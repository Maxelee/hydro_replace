"""
Constants and configuration for the hydro_replace statistics pipeline.
"""

import numpy as np

# Data paths — Replace models
LP_BASE = '/mnt/home/mlee1/ceph/hydro_replace_LP/L205n2500TNG'
RT_BASE = '/mnt/home/mlee1/ceph/hydro_replace_RT/L205n2500TNG'

# Data paths — BCM models
LP_BASE_BCM = '/mnt/home/mlee1/ceph/hydro_replace_LP_bcm/L205n2500TNG'
RT_BASE_BCM = '/mnt/home/mlee1/ceph/hydro_replace_RT_bcm/L205n2500TNG'

STATS_BASE = '/mnt/home/mlee1/ceph/hydro_replace_stats'

# Box and grid parameters
BOX_SIZE = 205.0  # Mpc/h
GRID_RES = 4096   # Lensplane resolution
RT_GRID = 1024    # Ray-tracing output resolution
FOV_DEG = 5.0     # Field of view in degrees
N_LP = 20         # Lensplane orientations (LP_00 to LP_19)
N_RUNS = 100      # Ray-traced maps per LP (run001 to run100)

# Weak lensing parameters
SMOOTHING_ARCMIN = 0.0  # No smoothing (was 2.5 arcmin)
PIXEL_SCALE_ARCMIN = FOV_DEG * 60.0 / RT_GRID  # ~0.293 arcmin/pixel
SN_BINS = np.linspace(-5, 10, 51)  # 51 edges → 50 bins for S/N histograms (-5σ to 10σ)

# Wavelet Scattering Transform parameters
WST_J = 5       # Number of octaves (probes scales from ~2 to ~64 pixels)
WST_Q = 1       # Scales per octave
WST_MOMENTS = (0.5, 1.0, 2.0)  # Moment orders for WST coefficients

# Bispectrum parameters
# Extended to ℓ=50000 to better match C_ℓ range (which goes to ~2×10^5)
# Note: bispectrum is O(N³) so we use fewer bins at high ℓ
BISPEC_ELL_BINS = np.logspace(np.log10(100), np.log10(50000), 20)  # 19 bins from ℓ=100 to 50000
BISPEC_CONFIGS = ['equilateral', 'squeezed', 'folded', 'isosceles']  # Triangle configurations

# Source redshift targets for convergence maps
# All 40 kappa files: kappa01.dat to kappa40.dat
# χ = kappa_num × 102.5 h⁻¹Mpc, z_s computed from χ(z) relation
KAPPA_TARGETS = {f'kappa{i:02d}': i for i in range(1, 41)}

# Exact source redshifts for all 40 kappa files
# Based on χ(z) relation: χ = kappa_num × 102.5 h⁻¹Mpc
KAPPA_REDSHIFTS = {
    1: 0.034, 2: 0.070, 3: 0.105, 4: 0.142, 5: 0.179,
    6: 0.216, 7: 0.255, 8: 0.294, 9: 0.335, 10: 0.376,
    11: 0.418, 12: 0.462, 13: 0.506, 14: 0.552, 15: 0.599,
    16: 0.648, 17: 0.698, 18: 0.749, 19: 0.803, 20: 0.858,
    21: 0.914, 22: 0.973, 23: 1.034, 24: 1.097, 25: 1.163,
    26: 1.231, 27: 1.302, 28: 1.375, 29: 1.452, 30: 1.532,
    31: 1.615, 32: 1.703, 33: 1.794, 34: 1.889, 35: 1.989,
    36: 2.094, 37: 2.203, 38: 2.319, 39: 2.440, 40: 2.568
}

# Snapshot configuration for density analysis (20 snapshots)
SNAPSHOT_ORDER = [96, 90, 85, 80, 76, 71, 67, 63, 59, 56, 
                  52, 49, 46, 43, 41, 38, 35, 33, 31, 29]
SNAPSHOT_REDSHIFTS = [0.04, 0.15, 0.27, 0.40, 0.50, 0.64, 0.78, 0.93, 1.07, 1.18,
                      1.36, 1.50, 1.65, 1.82, 1.93, 2.12, 2.32, 2.49, 2.68, 2.87]

# Number of lensplane files per LP orientation (lenspot00.dat to lenspot39.dat)
N_LENSPLANES = 40

# Mass thresholds for cumulative models (M⊙/h)
MASS_THRESHOLDS = ['1.00e12', '3.16e12', '1.00e13', '3.16e13']
MASS_THRESHOLDS_FLOAT = [1.00e12, 3.16e12, 1.00e13, 3.16e13]

# Radius factors (in units of R200)
RADII = [0.5, 1.0, 3.0, 5.0]

# Discrete mass bins (M⊙/h)
DISCRETE_MASS_BINS = [
    (1.00e12, 3.16e12),
    (3.16e12, 1.00e13),
    (1.00e13, 3.16e13),
    (3.16e13, 1.00e15)
]

# Discrete radius bins (in units of R200)
DISCRETE_RADIUS_BINS = [
    (0.0, 0.5),
    (0.5, 1.0),
    (1.0, 3.0),
    (3.0, 5.0)
]


def build_model_name(Ml, Mu, Ri, Ro):
    """
    Build model name from mass and radius bounds.
    
    Parameters
    ----------
    Ml : float
        Mass lower bound (M⊙/h)
    Mu : float
        Mass upper bound (M⊙/h)
    Ri : float
        Radius inner bound (R200 units)
    Ro : float
        Radius outer bound (R200 units)
    
    Returns
    -------
    str
        Model name in format: hydro_replace_Ml_{Ml}_Mu_{Mu}_Ri_{Ri}_Ro_{Ro}
    """
    # Format with .2e and remove '+' sign (e.g., 1.00e+12 -> 1.00e12)
    Ml_str = f"{Ml:.2e}".replace('e+', 'e')
    Mu_str = f"{Mu:.2e}".replace('e+', 'e')
    return f"hydro_replace_Ml_{Ml_str}_Mu_{Mu_str}_Ri_{Ri:.1f}_Ro_{Ro:.1f}"


def get_all_models():
    """
    Generate list of all model names.
    
    Returns
    -------
    list
        List of model names including 'dmo', 'hydro', 34 cumulative models, 
        and 64 discrete tile models (total 100 models)
    """
    models = ['dmo', 'hydro']
    
    # Cumulative models: 4 mass thresholds × 4 radii + 4 radii only + 4 mass only
    # Mass + Radius: M > threshold, r < radius
    for mass_str in MASS_THRESHOLDS:
        mass_float = float(mass_str)
        for radius in RADII:
            model_name = build_model_name(mass_float, 1.00e15, 0.0, radius)
            models.append(model_name)
    
    # Mass only: M > threshold, all radii
    for mass_str in MASS_THRESHOLDS:
        mass_float = float(mass_str)
        model_name = build_model_name(mass_float, 1.00e15, 0.0, 5.0)
        if model_name not in models:
            models.append(model_name)
    
    # Radius only: all masses, r < radius
    for radius in RADII:
        model_name = build_model_name(1.00e12, 1.00e15, 0.0, radius)
        if model_name not in models:
            models.append(model_name)
    
    # Discrete tile models: 4 mass bins × 4 radius shells
    for Ml, Mu in DISCRETE_MASS_BINS:
        for Ri, Ro in DISCRETE_RADIUS_BINS:
            model_name = build_model_name(Ml, Mu, Ri, Ro)
            models.append(model_name)
    
    return models


def get_cumulative_models():
    """Get only the 34 cumulative models (16 main + extras)."""
    models = []
    
    # Main grid: 4 mass × 4 radii = 16
    for mass_str in MASS_THRESHOLDS:
        mass_float = float(mass_str)
        for radius in RADII:
            model_name = build_model_name(mass_float, 1.00e15, 0.0, radius)
            models.append(model_name)
    
    return models


def get_discrete_models():
    """Get only the 16 discrete tile models."""
    models = []
    
    for Ml, Mu in DISCRETE_MASS_BINS:
        for Ri, Ro in DISCRETE_RADIUS_BINS:
            model_name = build_model_name(Ml, Mu, Ri, Ro)
            models.append(model_name)
    
    return models


# =============================================================================
# Survey noise parameters
# =============================================================================
SURVEY_PARAMS = {
    'LSST': {
        'sigma_e': 0.3,       # Intrinsic ellipticity dispersion (per component)
        'n_gal': 30.0,        # Galaxy number density (per arcmin^2)
        'area_deg2': 18000.0, # Survey area (deg^2) -- metadata only
    },
    'DES': {
        'sigma_e': 0.3,       # Intrinsic ellipticity dispersion (per component)
        'n_gal': 10.0,        # Galaxy number density (per arcmin^2)
        'area_deg2': 5000.0,  # Survey area (deg^2) -- metadata only
    },
}

# Smoothing scales to explore (theta_G in arcmin)
# The smoothing kernel is W(theta) = (1/(pi*theta_G^2)) * exp(-theta^2/theta_G^2)
# which is a Gaussian with sigma = theta_G / sqrt(2)
SMOOTHING_SCALES = [1.0, 2.0, 3.0]
