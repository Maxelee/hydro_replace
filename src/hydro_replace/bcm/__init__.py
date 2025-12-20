"""
Baryon Correction Model Module
==============================

Interface to BaryonForge for BCM computations.
"""

from .arico_bcm import (
    AricoBCM,
    BCMParameters,
    compute_bcm_profiles,
    compute_bcm_power,
)

__all__ = [
    'AricoBCM',
    'BCMParameters',
    'compute_bcm_profiles',
    'compute_bcm_power',
]
