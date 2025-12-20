"""
Ray-Tracing Module
==================

Modules for convergence map generation and weak lensing analysis.
"""

from .raytrace_engine import (
    RayTraceConfig,
    LuxInterface,
    generate_lux_config,
    run_raytrace,
)
from .convergence_maps import (
    ConvergenceMap,
    smooth_map,
    compute_power_spectrum_2d,
)
from .peak_finding import (
    PeakCatalog,
    find_peaks,
    compute_peak_counts,
)

__all__ = [
    # Ray-tracing engine
    'RayTraceConfig',
    'LuxInterface',
    'generate_lux_config',
    'run_raytrace',
    # Convergence maps
    'ConvergenceMap',
    'smooth_map',
    'compute_power_spectrum_2d',
    # Peak finding
    'PeakCatalog',
    'find_peaks',
    'compute_peak_counts',
]
