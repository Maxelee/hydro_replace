"""
Mass Conservation Module
========================

Functions for computing mass conservation metrics in halo replacement.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import h5py
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MassConservation:
    """
    Container for mass conservation analysis results.

    Attributes
    ----------
    halo_id : int
        Halo identifier.
    m200c : float
        Halo M_200c mass (Msun/h).
    r200c : float
        Halo R_200c radius (Mpc/h).
    radii : ndarray
        Radii at which mass is computed (Mpc/h).
    mass_hydro : ndarray
        Enclosed mass in hydro simulation at each radius.
    mass_dmo : ndarray
        Enclosed mass in DMO simulation at each radius.
    mass_replacement : ndarray
        Enclosed mass after replacement at each radius.
    """
    
    halo_id: int
    m200c: float
    r200c: float
    radii: np.ndarray
    mass_hydro: np.ndarray
    mass_dmo: np.ndarray
    mass_replacement: np.ndarray
    
    @property
    def n_radii(self) -> int:
        """Number of radii."""
        return len(self.radii)
    
    @property
    def radii_over_r200c(self) -> np.ndarray:
        """Radii normalized by R_200c."""
        return self.radii / self.r200c
    
    @property
    def mass_deficit_hydro(self) -> np.ndarray:
        """Mass deficit of hydro relative to DMO."""
        with np.errstate(divide='ignore', invalid='ignore'):
            return (self.mass_hydro - self.mass_dmo) / self.mass_dmo
    
    @property
    def mass_deficit_replacement(self) -> np.ndarray:
        """Mass deficit of replacement relative to DMO."""
        with np.errstate(divide='ignore', invalid='ignore'):
            return (self.mass_replacement - self.mass_dmo) / self.mass_dmo
    
    @property
    def replacement_accuracy(self) -> np.ndarray:
        """How well replacement matches hydro: M_rep / M_hydro."""
        with np.errstate(divide='ignore', invalid='ignore'):
            return self.mass_replacement / self.mass_hydro
    
    def get_deficit_at_radius(self, radius: float) -> Dict[str, float]:
        """Get mass deficits at a specific radius."""
        idx = np.argmin(np.abs(self.radii - radius))
        return {
            'radius': self.radii[idx],
            'deficit_hydro': self.mass_deficit_hydro[idx],
            'deficit_replacement': self.mass_deficit_replacement[idx],
            'accuracy': self.replacement_accuracy[idx],
        }
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'halo_id': self.halo_id,
            'm200c': self.m200c,
            'r200c': self.r200c,
            'radii': self.radii,
            'mass_hydro': self.mass_hydro,
            'mass_dmo': self.mass_dmo,
            'mass_replacement': self.mass_replacement,
        }


def compute_enclosed_mass(
    coords: np.ndarray,
    masses: np.ndarray,
    center: np.ndarray,
    radii: np.ndarray,
    box_size: Optional[float] = None,
) -> np.ndarray:
    """
    Compute enclosed mass at multiple radii.

    Parameters
    ----------
    coords : ndarray
        Particle coordinates (N, 3).
    masses : ndarray
        Particle masses (N,).
    center : ndarray
        Halo center (3,).
    radii : ndarray
        Radii at which to compute enclosed mass.
    box_size : float, optional
        Box size for periodic boundary conditions.

    Returns
    -------
    enclosed_mass : ndarray
        Enclosed mass at each radius.

    Examples
    --------
    >>> radii = np.logspace(-2, 1, 50)  # 0.01 to 10 Mpc/h
    >>> m_enc = compute_enclosed_mass(
    ...     coords, masses, halo_center, radii, box_size=205.0
    ... )
    """
    # Compute distances
    dr = coords - center
    
    # Apply periodic boundaries if needed
    if box_size is not None:
        dr = dr - box_size * np.round(dr / box_size)
    
    distances = np.sqrt(np.sum(dr**2, axis=1))
    
    # Compute enclosed mass at each radius
    enclosed_mass = np.zeros(len(radii))
    
    for i, r in enumerate(radii):
        mask = distances <= r
        enclosed_mass[i] = masses[mask].sum()
    
    return enclosed_mass


def compute_enclosed_mass_fast(
    coords: np.ndarray,
    masses: np.ndarray,
    center: np.ndarray,
    radii: np.ndarray,
    box_size: Optional[float] = None,
) -> np.ndarray:
    """
    Compute enclosed mass at multiple radii using cumulative sum.

    Faster than compute_enclosed_mass for many radii.

    Parameters
    ----------
    coords : ndarray
        Particle coordinates (N, 3).
    masses : ndarray
        Particle masses (N,).
    center : ndarray
        Halo center (3,).
    radii : ndarray
        Radii at which to compute enclosed mass (must be sorted).
    box_size : float, optional
        Box size for periodic boundary conditions.

    Returns
    -------
    enclosed_mass : ndarray
        Enclosed mass at each radius.
    """
    # Compute distances
    dr = coords - center
    
    if box_size is not None:
        dr = dr - box_size * np.round(dr / box_size)
    
    distances = np.sqrt(np.sum(dr**2, axis=1))
    
    # Sort by distance
    sort_idx = np.argsort(distances)
    sorted_dist = distances[sort_idx]
    sorted_mass = masses[sort_idx]
    
    # Cumulative mass
    cumulative_mass = np.cumsum(sorted_mass)
    
    # Interpolate to radii
    enclosed_mass = np.interp(
        radii,
        sorted_dist,
        cumulative_mass,
        left=0.0,
    )
    
    return enclosed_mass


def compute_mass_deficit(
    coords_hydro: np.ndarray,
    masses_hydro: np.ndarray,
    coords_dmo: np.ndarray,
    masses_dmo: np.ndarray,
    center: np.ndarray,
    radii: np.ndarray,
    box_size: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute mass deficit between hydro and DMO simulations.

    Parameters
    ----------
    coords_hydro : ndarray
        Hydro particle coordinates.
    masses_hydro : ndarray
        Hydro particle masses (baryons + DM).
    coords_dmo : ndarray
        DMO particle coordinates.
    masses_dmo : ndarray
        DMO particle masses.
    center : ndarray
        Halo center.
    radii : ndarray
        Radii at which to compute.
    box_size : float, optional
        Box size for periodic boundaries.

    Returns
    -------
    deficit : ndarray
        Fractional mass deficit (M_hydro - M_dmo) / M_dmo.
    deficit_abs : ndarray
        Absolute mass deficit M_hydro - M_dmo.
    """
    m_hydro = compute_enclosed_mass_fast(coords_hydro, masses_hydro, center, radii, box_size)
    m_dmo = compute_enclosed_mass_fast(coords_dmo, masses_dmo, center, radii, box_size)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        deficit = (m_hydro - m_dmo) / m_dmo
    
    deficit_abs = m_hydro - m_dmo
    
    return deficit, deficit_abs


@dataclass
class MassConservationAnalyzer:
    """
    Analyze mass conservation across multiple halos.

    Parameters
    ----------
    results : list
        List of MassConservation objects.
    """
    
    results: List[MassConservation] = field(default_factory=list)
    
    def add_result(self, result: MassConservation) -> None:
        """Add a mass conservation result."""
        self.results.append(result)
    
    @property
    def n_halos(self) -> int:
        """Number of halos analyzed."""
        return len(self.results)
    
    @property
    def halo_ids(self) -> np.ndarray:
        """Array of halo IDs."""
        return np.array([r.halo_id for r in self.results])
    
    @property
    def masses(self) -> np.ndarray:
        """Array of halo M_200c values."""
        return np.array([r.m200c for r in self.results])
    
    def get_stacked_deficit(
        self,
        r_over_r200c: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Compute stacked mass deficits at normalized radii.

        Parameters
        ----------
        r_over_r200c : ndarray
            Radii normalized by R_200c.

        Returns
        -------
        result : dict
            Dictionary with 'hydro_mean', 'hydro_std', 'replacement_mean', 
            'replacement_std', 'accuracy_mean', 'accuracy_std'.
        """
        n_radii = len(r_over_r200c)
        n_halos = len(self.results)
        
        deficit_hydro = np.zeros((n_halos, n_radii))
        deficit_replacement = np.zeros((n_halos, n_radii))
        accuracy = np.zeros((n_halos, n_radii))
        
        for i, res in enumerate(self.results):
            # Interpolate to common normalized radii
            deficit_hydro[i] = np.interp(
                r_over_r200c,
                res.radii_over_r200c,
                res.mass_deficit_hydro,
            )
            deficit_replacement[i] = np.interp(
                r_over_r200c,
                res.radii_over_r200c,
                res.mass_deficit_replacement,
            )
            accuracy[i] = np.interp(
                r_over_r200c,
                res.radii_over_r200c,
                res.replacement_accuracy,
            )
        
        return {
            'r_over_r200c': r_over_r200c,
            'hydro_mean': np.nanmean(deficit_hydro, axis=0),
            'hydro_std': np.nanstd(deficit_hydro, axis=0),
            'replacement_mean': np.nanmean(deficit_replacement, axis=0),
            'replacement_std': np.nanstd(deficit_replacement, axis=0),
            'accuracy_mean': np.nanmean(accuracy, axis=0),
            'accuracy_std': np.nanstd(accuracy, axis=0),
        }
    
    def get_deficit_by_mass_bin(
        self,
        mass_bins: np.ndarray,
        r_over_r200c: float = 1.0,
    ) -> Dict[str, np.ndarray]:
        """
        Compute mass deficits as function of halo mass.

        Parameters
        ----------
        mass_bins : ndarray
            Mass bin edges (Msun/h).
        r_over_r200c : float
            Normalized radius at which to evaluate.

        Returns
        -------
        result : dict
            Dictionary with bin centers and mean deficits.
        """
        bin_centers = np.sqrt(mass_bins[:-1] * mass_bins[1:])
        n_bins = len(bin_centers)
        
        deficit_hydro = np.full(n_bins, np.nan)
        deficit_replacement = np.full(n_bins, np.nan)
        accuracy = np.full(n_bins, np.nan)
        counts = np.zeros(n_bins, dtype=int)
        
        for res in self.results:
            # Find bin for this halo
            bin_idx = np.searchsorted(mass_bins[1:], res.m200c)
            if bin_idx >= n_bins:
                continue
            
            # Get deficit at specified radius
            deficit_at_r = res.get_deficit_at_radius(r_over_r200c * res.r200c)
            
            # Accumulate
            if np.isnan(deficit_hydro[bin_idx]):
                deficit_hydro[bin_idx] = deficit_at_r['deficit_hydro']
                deficit_replacement[bin_idx] = deficit_at_r['deficit_replacement']
                accuracy[bin_idx] = deficit_at_r['accuracy']
            else:
                deficit_hydro[bin_idx] += deficit_at_r['deficit_hydro']
                deficit_replacement[bin_idx] += deficit_at_r['deficit_replacement']
                accuracy[bin_idx] += deficit_at_r['accuracy']
            
            counts[bin_idx] += 1
        
        # Average
        mask = counts > 0
        deficit_hydro[mask] /= counts[mask]
        deficit_replacement[mask] /= counts[mask]
        accuracy[mask] /= counts[mask]
        
        return {
            'mass_bins': bin_centers,
            'deficit_hydro': deficit_hydro,
            'deficit_replacement': deficit_replacement,
            'accuracy': accuracy,
            'counts': counts,
        }
    
    def save_hdf5(self, filepath: Union[str, Path]) -> None:
        """Save all results to HDF5."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with h5py.File(filepath, 'w') as f:
            f.attrs['n_halos'] = self.n_halos
            
            for i, res in enumerate(self.results):
                grp = f.create_group(f'halo_{i:05d}')
                grp.attrs['halo_id'] = res.halo_id
                grp.attrs['m200c'] = res.m200c
                grp.attrs['r200c'] = res.r200c
                
                grp.create_dataset('radii', data=res.radii)
                grp.create_dataset('mass_hydro', data=res.mass_hydro)
                grp.create_dataset('mass_dmo', data=res.mass_dmo)
                grp.create_dataset('mass_replacement', data=res.mass_replacement)
    
    @classmethod
    def load_hdf5(cls, filepath: Union[str, Path]) -> 'MassConservationAnalyzer':
        """Load results from HDF5."""
        results = []
        
        with h5py.File(filepath, 'r') as f:
            n_halos = f.attrs['n_halos']
            
            for i in range(n_halos):
                grp = f[f'halo_{i:05d}']
                results.append(MassConservation(
                    halo_id=grp.attrs['halo_id'],
                    m200c=grp.attrs['m200c'],
                    r200c=grp.attrs['r200c'],
                    radii=grp['radii'][:],
                    mass_hydro=grp['mass_hydro'][:],
                    mass_dmo=grp['mass_dmo'][:],
                    mass_replacement=grp['mass_replacement'][:],
                ))
        
        return cls(results=results)
