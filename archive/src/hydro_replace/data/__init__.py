"""
Data Loading Module
===================

Functions for loading simulation data, halo catalogs, and configuration files.
"""

from .load_simulations import (
    load_simulation_config,
    SimulationData,
    load_snapshot_header,
    get_snapshot_redshift,
)

from .halo_catalogs import (
    load_halo_catalog,
    HaloCatalog,
    filter_by_mass,
)

from .bijective_matching import (
    bijective_halo_matching,
    BijectiveMatcher,
    MatchedCatalog,
)

from .particle_extraction import (
    extract_halo_particles,
    ParticleExtractor,
)

__all__ = [
    # Configuration
    "load_simulation_config",
    # Simulation data
    "SimulationData",
    "load_snapshot_header",
    "get_snapshot_redshift",
    # Halo catalogs
    "load_halo_catalog",
    "HaloCatalog",
    "filter_by_mass",
    # Matching
    "bijective_halo_matching",
    "BijectiveMatcher",
    "MatchedCatalog",
    # Particle extraction
    "extract_halo_particles",
    "ParticleExtractor",
]
