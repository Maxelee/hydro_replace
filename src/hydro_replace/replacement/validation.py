"""
Replacement Validation Module
=============================

Functions for validating replacement operations and checking mass conservation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .replace_core import ReplacementResult

logger = logging.getLogger(__name__)


@dataclass
class ReplacementValidation:
    """
    Results of replacement validation checks.

    Attributes
    ----------
    is_valid : bool
        Whether the replacement passed all checks.
    mass_conservation_error : float
        Relative mass change from replacement.
    particle_count_check : bool
        Whether particle counts are reasonable.
    mass_check : bool
        Whether mass conservation is within tolerance.
    messages : list of str
        Validation messages/warnings.
    """
    
    is_valid: bool
    mass_conservation_error: float
    particle_count_check: bool
    mass_check: bool
    messages: List[str]
    
    def __str__(self) -> str:
        status = "PASSED" if self.is_valid else "FAILED"
        lines = [
            f"Replacement Validation: {status}",
            f"  Mass conservation error: {self.mass_conservation_error:.2%}",
            f"  Particle count check: {'OK' if self.particle_count_check else 'FAIL'}",
            f"  Mass check: {'OK' if self.mass_check else 'FAIL'}",
        ]
        if self.messages:
            lines.append("  Messages:")
            for msg in self.messages:
                lines.append(f"    - {msg}")
        return "\n".join(lines)


def validate_replacement(
    result: ReplacementResult,
    original_dmo_mass: float,
    original_hydro_mass: float,
    mass_tolerance: float = 0.1,
    min_particles_per_halo: int = 100,
) -> ReplacementValidation:
    """
    Validate a replacement operation.

    Parameters
    ----------
    result : ReplacementResult
        Replacement operation result.
    original_dmo_mass : float
        Total mass of original DMO snapshot.
    original_hydro_mass : float
        Total mass of original hydro snapshot.
    mass_tolerance : float
        Maximum allowed relative mass change.
    min_particles_per_halo : int
        Minimum expected particles per replaced halo.

    Returns
    -------
    validation : ReplacementValidation
        Validation results.
    """
    messages = []
    
    # Check 1: Particle counts
    expected_min_particles = result.n_halos_replaced * min_particles_per_halo
    particle_count_check = result.n_hydro_replacement >= expected_min_particles
    
    if not particle_count_check:
        messages.append(
            f"Low hydro particle count: {result.n_hydro_replacement:,} "
            f"(expected >= {expected_min_particles:,})"
        )
    
    # Check 2: Mass conservation
    # In replacement, we remove DMO mass and add hydro mass
    # Total mass should be roughly conserved (within tolerance)
    
    total_result_mass = result.mass_dmo_background + result.mass_hydro_replacement
    
    # The expected total depends on the replacement mode
    # For now, check against a reasonable range
    mass_conservation_error = abs(total_result_mass - original_dmo_mass) / original_dmo_mass
    mass_check = mass_conservation_error < mass_tolerance
    
    if not mass_check:
        messages.append(
            f"Mass conservation error: {mass_conservation_error:.2%} "
            f"(tolerance: {mass_tolerance:.2%})"
        )
    
    # Check 3: No empty arrays (unless no halos replaced)
    if result.n_halos_replaced > 0:
        if result.n_hydro_replacement == 0:
            messages.append("No hydro particles extracted despite halos replaced")
            particle_count_check = False
    
    # Check 4: Coordinate bounds
    if len(result.combined_coords) > 0:
        coords = result.combined_coords
        if np.any(coords < 0) or np.any(np.isinf(coords)):
            messages.append("Invalid coordinates detected (negative or infinite)")
            particle_count_check = False
    
    is_valid = particle_count_check and mass_check
    
    return ReplacementValidation(
        is_valid=is_valid,
        mass_conservation_error=mass_conservation_error,
        particle_count_check=particle_count_check,
        mass_check=mass_check,
        messages=messages,
    )


def check_mass_conservation(
    result: ReplacementResult,
    original_total_mass: float,
    tolerance: float = 0.05,
) -> Tuple[bool, float, str]:
    """
    Simple mass conservation check.

    Parameters
    ----------
    result : ReplacementResult
        Replacement result.
    original_total_mass : float
        Original total mass.
    tolerance : float
        Relative tolerance.

    Returns
    -------
    passed : bool
        Whether check passed.
    relative_error : float
        Relative mass difference.
    message : str
        Status message.
    """
    result_mass = result.mass_dmo_background + result.mass_hydro_replacement
    relative_error = abs(result_mass - original_total_mass) / original_total_mass
    
    passed = relative_error < tolerance
    
    if passed:
        message = f"Mass conservation OK: {relative_error:.2%} error"
    else:
        message = f"Mass conservation FAILED: {relative_error:.2%} error (tolerance: {tolerance:.2%})"
    
    return passed, relative_error, message


def check_halo_coverage(
    result: ReplacementResult,
    expected_n_halos: int,
) -> Tuple[bool, str]:
    """
    Check that the expected number of halos were replaced.

    Parameters
    ----------
    result : ReplacementResult
        Replacement result.
    expected_n_halos : int
        Expected number of halos.

    Returns
    -------
    passed : bool
        Whether check passed.
    message : str
        Status message.
    """
    passed = result.n_halos_replaced == expected_n_halos
    
    if passed:
        message = f"Halo coverage OK: {result.n_halos_replaced} halos replaced"
    else:
        message = (
            f"Halo coverage mismatch: replaced {result.n_halos_replaced}, "
            f"expected {expected_n_halos}"
        )
    
    return passed, message


def compute_mass_deficit(
    dmo_coords: np.ndarray,
    dmo_masses: np.ndarray,
    hydro_coords: np.ndarray,
    hydro_masses: np.ndarray,
    center: np.ndarray,
    radii: List[float],
    box_size: float,
) -> Dict[str, float]:
    """
    Compute mass deficit M_hydro - M_dmo within various radii.

    Parameters
    ----------
    dmo_coords : ndarray
        DMO particle coordinates (N, 3).
    dmo_masses : ndarray
        DMO particle masses (N,).
    hydro_coords : ndarray
        Hydro particle coordinates (M, 3).
    hydro_masses : ndarray
        Hydro particle masses (M,).
    center : ndarray
        Center position (3,).
    radii : list of float
        Radii at which to compute enclosed mass.
    box_size : float
        Box size for periodic boundary handling.

    Returns
    -------
    deficits : dict
        Dictionary with keys like 'r_0.5', 'r_1.0' mapping to 
        (M_hydro - M_dmo) / M_dmo values.
    """
    deficits = {}
    
    # Compute distances with periodic boundaries
    def get_distances(coords, center):
        dx = coords - center
        dx = dx - np.round(dx / box_size) * box_size
        return np.linalg.norm(dx, axis=1)
    
    dmo_r = get_distances(dmo_coords, center)
    hydro_r = get_distances(hydro_coords, center)
    
    for r in radii:
        m_dmo = dmo_masses[dmo_r < r].sum()
        m_hydro = hydro_masses[hydro_r < r].sum()
        
        if m_dmo > 0:
            deficit = (m_hydro - m_dmo) / m_dmo
        else:
            deficit = np.nan
        
        deficits[f'r_{r:.1f}'] = deficit
    
    return deficits


def run_all_validations(
    result: ReplacementResult,
    original_dmo_mass: float,
    original_hydro_mass: float,
    expected_n_halos: int,
    verbose: bool = True,
) -> bool:
    """
    Run all validation checks and optionally print results.

    Parameters
    ----------
    result : ReplacementResult
        Replacement result.
    original_dmo_mass : float
        Original DMO total mass.
    original_hydro_mass : float
        Original hydro total mass.
    expected_n_halos : int
        Expected number of halos.
    verbose : bool
        Whether to print results.

    Returns
    -------
    all_passed : bool
        Whether all checks passed.
    """
    if verbose:
        print("=" * 60)
        print("REPLACEMENT VALIDATION REPORT")
        print("=" * 60)
    
    # Main validation
    validation = validate_replacement(
        result, original_dmo_mass, original_hydro_mass
    )
    
    if verbose:
        print(validation)
        print()
    
    # Halo coverage
    halo_passed, halo_msg = check_halo_coverage(result, expected_n_halos)
    
    if verbose:
        status = "OK" if halo_passed else "FAIL"
        print(f"Halo coverage: [{status}] {halo_msg}")
    
    # Summary
    all_passed = validation.is_valid and halo_passed
    
    if verbose:
        print()
        print("=" * 60)
        final_status = "ALL CHECKS PASSED" if all_passed else "SOME CHECKS FAILED"
        print(f"FINAL STATUS: {final_status}")
        print("=" * 60)
    
    return all_passed
