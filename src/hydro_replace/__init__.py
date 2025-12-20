"""
Hydro Replace Package
=====================

A modular pipeline for analyzing baryonic correction models using halo replacement
techniques applied to cosmological simulations.

This package provides tools for:
- Bijective halo matching between DMO and hydro simulations
- Hydro particle replacement in DMO snapshots
- Baryonification (BCM) model application
- Density profile extraction and analysis
- Power spectrum computation
- Weak lensing ray-tracing and peak statistics

Author: Matthew Lee
"""

__version__ = "0.1.0"
__author__ = "Matthew Lee"

# Expose main classes and functions at package level
from .data.load_simulations import (
    load_simulation_config,
    SimulationData,
)

from .data.halo_catalogs import (
    HaloCatalog,
    load_halo_catalog,
)

from .data.bijective_matching import (
    BijectiveMatcher,
    MatchedCatalog,
)

from .data.particle_extraction import (
    ParticleExtractor,
    ExtractedHalo,
)

from .replacement.replace_core import (
    HaloReplacer,
    ReplacementResult,
)

from .analysis.profiles import (
    DensityProfile,
    compute_density_profile,
    ProfileAnalyzer,
)

from .analysis.power_spectrum import (
    PowerSpectrum,
    compute_power_spectrum,
    compute_suppression,
)

from .analysis.mass_conservation import (
    MassConservation,
    compute_enclosed_mass,
)

__all__ = [
    # Version
    "__version__",
    "__author__",
    # Configuration
    "load_simulation_config",
    # Data loading
    "HaloCatalog",
    "load_halo_catalog",
    "SimulationData",
    # Matching
    "BijectiveMatcher",
    "MatchedCatalog",
    # Extraction
    "ParticleExtractor",
    "ExtractedHalo",
    # Replacement
    "HaloReplacer",
    "ReplacementResult",
    # Profiles
    "DensityProfile",
    "compute_density_profile",
    "ProfileAnalyzer",
    # Power spectrum
    "PowerSpectrum",
    "compute_power_spectrum",
    "compute_suppression",
    # Mass conservation
    "MassConservation",
    "compute_enclosed_mass",
