"""
Halo Replacement Module
=======================

Core functionality for replacing DMO halo particles with hydro particles.
"""

from .replace_core import (
    HaloReplacer,
    replace_halos,
    ReplacementResult,
)

from .mass_bins import (
    MassBinConfig,
    get_mass_bins,
    get_cumulative_bins,
)

from .validation import (
    validate_replacement,
    check_mass_conservation,
    ReplacementValidation,
)

__all__ = [
    # Core replacement
    "HaloReplacer",
    "replace_halos",
    "ReplacementResult",
    # Mass bins
    "MassBinConfig",
    "get_mass_bins",
    "get_cumulative_bins",
    # Validation
    "validate_replacement",
    "check_mass_conservation",
    "ReplacementValidation",
]
