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
    MassConservation,
    compute_mass_deficit,
    compute_enclosed_mass,
)

from .statistics import (
    stack_profiles,
    bootstrap_error,
    weighted_mean_and_error,
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
    "MassConservation",
    "compute_mass_deficit",
    "compute_enclosed_mass",
    # Statistics
    "stack_profiles",
    "bootstrap_error",
    "weighted_mean_and_error",
]
