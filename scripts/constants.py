"""
Constants and configuration for the hydro_replace statistics pipeline.
"""

import numpy as np

# Data paths
LP_BASE = '/mnt/home/mlee1/ceph/hydro_replace_LP/L205n2500TNG'
RT_BASE = '/mnt/home/mlee1/ceph/hydro_replace_RT/L205n2500TNG'
STATS_BASE = '/mnt/home/mlee1/ceph/hydro_replace_stats'

# Box and grid parameters
BOX_SIZE = 205.0  # Mpc/h
GRID_RES = 4096   # Lensplane resolution
RT_GRID = 1024    # Ray-tracing output resolution
FOV_DEG = 5.0     # Field of view in degrees
N_LP = 20         # Lensplane orientations (LP_00 to LP_19)
N_RUNS = 100      # Ray-traced maps per LP (run001 to run100)

# Weak lensing parameters
SMOOTHING_ARCMIN = 2.5  # Gaussian smoothing scale
PIXEL_SCALE_ARCMIN = FOV_DEG * 60.0 / RT_GRID  # ~0.293 arcmin/pixel
SN_BINS = np.arange(-5, 11, 1)  # 16 edges → 15 bins for S/N histograms

# Source redshift targets for convergence maps
KAPPA_TARGETS = {
    'z0.5': 13,   # kappa13.dat → z_s ≈ 0.506
    'z1.0': 23,   # kappa23.dat → z_s ≈ 1.034
    'z2.0': 36,   # kappa36.dat → z_s ≈ 2.094
    'z2.5': 40    # kappa40.dat → z_s ≈ 2.568
}

# Exact source redshifts for kappa targets
KAPPA_REDSHIFTS = {
    13: 0.506,
    23: 1.034,
    36: 2.094,
    40: 2.568
}

# Snapshot configuration for density analysis (20 snapshots)
SNAPSHOT_ORDER = [96, 90, 85, 80, 76, 71, 67, 63, 59, 56, 
                  52, 49, 46, 43, 41, 38, 35, 33, 31, 29]
SNAPSHOT_REDSHIFTS = [0.04, 0.15, 0.27, 0.40, 0.50, 0.64, 0.78, 0.93, 1.07, 1.18,
                      1.36, 1.50, 1.65, 1.82, 1.93, 2.12, 2.32, 2.49, 2.68, 2.87]

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
    return f"hydro_replace_Ml_{Ml:.2e}_Mu_{Mu:.2e}_Ri_{Ri:.1f}_Ro_{Ro:.1f}"


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
