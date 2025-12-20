"""
Analysis Module
===============

Tools for analyzing density profiles, power spectra, and mass conservation.
"""

from .profiles import (
    compute_density_profile,
    DensityProfile,
    ProfileAnalyzer,
)

from .power_spectrum import (
    compute_power_spectrum,
    PowerSpectrum,
    compute_suppression,
)

from .mass_conservation import (
    compute_mass_conservation,
    MassConservation,
    compute_mass_deficit,
)

from .statistics import (
    compute_mean_profile,
    compute_percentile_profile,
    bootstrap_error,
)

__all__ = [
    # Profiles
    "compute_density_profile",
    "DensityProfile",
    "ProfileAnalyzer",
    # Power spectrum
    "compute_power_spectrum",
    "PowerSpectrum",
    "compute_suppression",
    # Mass conservation
    "compute_mass_conservation",
    "MassConservation",
    "compute_mass_deficit",
    # Statistics
    "compute_mean_profile",
    "compute_percentile_profile",
    "bootstrap_error",
]
