"""
Density Profile Module
======================

Functions for computing and analyzing spherically-averaged density profiles.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
from scipy.stats import binned_statistic

logger = logging.getLogger(__name__)


@dataclass
class DensityProfile:
    """
    Container for a density profile.

    Attributes
    ----------
    r_bins : ndarray
        Radial bin edges (in units of R_200c).
    r_centers : ndarray
        Radial bin centers.
    density : ndarray
        Density in each bin (Msun/h / (Mpc/h)^3).
    mass_enclosed : ndarray
        Cumulative mass enclosed within each bin.
    halo_id : int
        Halo ID.
    halo_mass : float
        M_200c in Msun/h.
    halo_radius : float
        R_200c in Mpc/h.
    """
    
    r_bins: np.ndarray
    r_centers: np.ndarray
    density: np.ndarray
    mass_enclosed: np.ndarray
    halo_id: int = 0
    halo_mass: float = 0.0
    halo_radius: float = 0.0
    
    @property
    def n_bins(self) -> int:
        """Number of radial bins."""
        return len(self.r_centers)
    
    @property
    def log_density(self) -> np.ndarray:
        """Log10 of density (NaN for zero bins)."""
        with np.errstate(divide='ignore'):
            return np.log10(self.density)
    
    def get_density_at_r(self, r: float) -> float:
        """
        Interpolate density at a given radius.

        Parameters
        ----------
        r : float
            Radius in units of R_200c.

        Returns
        -------
        rho : float
            Interpolated density.
        """
        return np.interp(r, self.r_centers, self.density)
    
    def save_hdf5(self, filepath: Union[str, Path], group_name: str = "profile") -> None:
        """Save profile to HDF5 file."""
        filepath = Path(filepath)
        
        with h5py.File(filepath, 'a') as f:
            if group_name in f:
                del f[group_name]
            
            grp = f.create_group(group_name)
            grp.attrs['halo_id'] = self.halo_id
            grp.attrs['halo_mass'] = self.halo_mass
            grp.attrs['halo_radius'] = self.halo_radius
            
            grp.create_dataset('r_bins', data=self.r_bins)
            grp.create_dataset('r_centers', data=self.r_centers)
            grp.create_dataset('density', data=self.density)
            grp.create_dataset('mass_enclosed', data=self.mass_enclosed)
    
    @classmethod
    def load_hdf5(cls, filepath: Union[str, Path], group_name: str = "profile") -> 'DensityProfile':
        """Load profile from HDF5 file."""
        with h5py.File(filepath, 'r') as f:
            grp = f[group_name]
            return cls(
                r_bins=grp['r_bins'][:],
                r_centers=grp['r_centers'][:],
                density=grp['density'][:],
                mass_enclosed=grp['mass_enclosed'][:],
                halo_id=grp.attrs.get('halo_id', 0),
                halo_mass=grp.attrs.get('halo_mass', 0.0),
                halo_radius=grp.attrs.get('halo_radius', 0.0),
            )


def compute_density_profile(
    coords: np.ndarray,
    masses: np.ndarray,
    center: np.ndarray,
    radius: float,
    n_bins: int = 50,
    r_min: float = 0.01,
    r_max: float = 5.0,
    box_size: Optional[float] = None,
    halo_id: int = 0,
    halo_mass: float = 0.0,
) -> DensityProfile:
    """
    Compute spherically-averaged density profile.

    Parameters
    ----------
    coords : ndarray
        Particle coordinates (N, 3) in Mpc/h.
    masses : ndarray
        Particle masses (N,) in Msun/h.
    center : ndarray
        Profile center (3,) in Mpc/h.
    radius : float
        Characteristic radius (R_200c) in Mpc/h.
    n_bins : int
        Number of radial bins.
    r_min : float
        Minimum radius in units of R_200c.
    r_max : float
        Maximum radius in units of R_200c.
    box_size : float, optional
        Box size for periodic boundary handling.
    halo_id : int
        Halo ID for metadata.
    halo_mass : float
        Halo mass for metadata.

    Returns
    -------
    profile : DensityProfile
        Computed density profile.

    Examples
    --------
    >>> profile = compute_density_profile(
    ...     coords=particle_coords,
    ...     masses=particle_masses,
    ...     center=[100, 100, 100],
    ...     radius=0.5,
    ... )
    >>> print(f"Central density: {profile.density[0]:.2e}")
    """
    # Apply periodic boundary conditions
    dx = coords - center
    if box_size is not None:
        dx = dx - np.round(dx / box_size) * box_size
    
    # Compute radial distances (normalized by R_200c)
    r = np.linalg.norm(dx, axis=1) / radius
    
    # Create logarithmic bins
    r_bins = np.logspace(np.log10(r_min), np.log10(r_max), n_bins + 1)
    r_centers = np.sqrt(r_bins[:-1] * r_bins[1:])  # Geometric mean
    
    # Bin masses
    mass_in_bins, _, _ = binned_statistic(r, masses, statistic='sum', bins=r_bins)
    
    # Handle NaN from empty bins
    mass_in_bins = np.nan_to_num(mass_in_bins)
    
    # Compute shell volumes (in physical units: Mpc/h)^3
    volumes = 4/3 * np.pi * (r_bins[1:]**3 - r_bins[:-1]**3) * radius**3
    
    # Density = mass / volume
    density = mass_in_bins / volumes
    
    # Cumulative mass enclosed
    mass_enclosed = np.cumsum(mass_in_bins)
    
    return DensityProfile(
        r_bins=r_bins,
        r_centers=r_centers,
        density=density,
        mass_enclosed=mass_enclosed,
        halo_id=halo_id,
        halo_mass=halo_mass,
        halo_radius=radius,
    )


class ProfileAnalyzer:
    """
    Analyze and compare multiple density profiles.

    Parameters
    ----------
    profiles : dict
        Dictionary mapping labels to DensityProfile objects.

    Examples
    --------
    >>> analyzer = ProfileAnalyzer({
    ...     'dmo': dmo_profile,
    ...     'hydro': hydro_profile,
    ...     'bcm': bcm_profile,
    ... })
    >>> residuals = analyzer.compute_residuals('hydro', 'dmo')
    """
    
    def __init__(self, profiles: Dict[str, DensityProfile]):
        self.profiles = profiles
        
        # Verify all profiles have compatible bins
        reference = list(profiles.values())[0]
        for label, profile in profiles.items():
            if len(profile.r_centers) != len(reference.r_centers):
                raise ValueError(
                    f"Profile '{label}' has incompatible bins "
                    f"({len(profile.r_centers)} vs {len(reference.r_centers)})"
                )
    
    @property
    def r_centers(self) -> np.ndarray:
        """Radial bin centers (from first profile)."""
        return list(self.profiles.values())[0].r_centers
    
    @property
    def r_bins(self) -> np.ndarray:
        """Radial bin edges (from first profile)."""
        return list(self.profiles.values())[0].r_bins
    
    def compute_ratio(self, label1: str, label2: str) -> np.ndarray:
        """
        Compute ratio of two profiles.

        Parameters
        ----------
        label1, label2 : str
            Profile labels.

        Returns
        -------
        ratio : ndarray
            rho_1(r) / rho_2(r)
        """
        p1 = self.profiles[label1]
        p2 = self.profiles[label2]
        
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = p1.density / p2.density
        
        return ratio
    
    def compute_residuals(
        self,
        label1: str,
        label2: str,
        relative: bool = True,
    ) -> np.ndarray:
        """
        Compute residuals between two profiles.

        Parameters
        ----------
        label1, label2 : str
            Profile labels.
        relative : bool
            If True, compute (p1 - p2) / p2. Otherwise p1 - p2.

        Returns
        -------
        residuals : ndarray
            Profile residuals.
        """
        p1 = self.profiles[label1]
        p2 = self.profiles[label2]
        
        diff = p1.density - p2.density
        
        if relative:
            with np.errstate(divide='ignore', invalid='ignore'):
                residuals = diff / p2.density
        else:
            residuals = diff
        
        return residuals
    
    def compute_mass_ratio(self, label1: str, label2: str) -> np.ndarray:
        """
        Compute ratio of enclosed mass profiles.

        Parameters
        ----------
        label1, label2 : str
            Profile labels.

        Returns
        -------
        ratio : ndarray
            M_1(<r) / M_2(<r)
        """
        p1 = self.profiles[label1]
        p2 = self.profiles[label2]
        
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = p1.mass_enclosed / p2.mass_enclosed
        
        return ratio
    
    def get_stacked_profiles(self) -> np.ndarray:
        """
        Stack all profiles into a 2D array.

        Returns
        -------
        stacked : ndarray
            Array of shape (n_profiles, n_bins).
        """
        return np.array([p.density for p in self.profiles.values()])


def compute_stacked_profile(
    profiles: List[DensityProfile],
    statistic: str = 'median',
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute stacked profile from multiple halos.

    Parameters
    ----------
    profiles : list of DensityProfile
        Individual halo profiles.
    statistic : str
        'mean' or 'median'.

    Returns
    -------
    r_centers : ndarray
        Radial bin centers.
    stacked : ndarray
        Stacked profile.
    scatter : ndarray
        16-84 percentile scatter.
    """
    if len(profiles) == 0:
        raise ValueError("No profiles to stack")
    
    r_centers = profiles[0].r_centers
    
    # Stack densities
    densities = np.array([p.density for p in profiles])
    
    if statistic == 'median':
        stacked = np.nanmedian(densities, axis=0)
    else:
        stacked = np.nanmean(densities, axis=0)
    
    # Compute scatter
    p16 = np.nanpercentile(densities, 16, axis=0)
    p84 = np.nanpercentile(densities, 84, axis=0)
    scatter = np.array([stacked - p16, p84 - stacked])
    
    return r_centers, stacked, scatter


def compute_profile_comparison(
    coords_dict: Dict[str, np.ndarray],
    masses_dict: Dict[str, np.ndarray],
    center: np.ndarray,
    radius: float,
    box_size: Optional[float] = None,
    **kwargs,
) -> ProfileAnalyzer:
    """
    Compute and compare profiles for multiple particle types.

    Parameters
    ----------
    coords_dict : dict
        Dictionary mapping labels to coordinate arrays.
    masses_dict : dict
        Dictionary mapping labels to mass arrays.
    center : ndarray
        Profile center.
    radius : float
        Characteristic radius.
    box_size : float, optional
        Box size for periodic boundaries.
    **kwargs
        Additional arguments for compute_density_profile.

    Returns
    -------
    analyzer : ProfileAnalyzer
        Profile analyzer with computed profiles.

    Examples
    --------
    >>> analyzer = compute_profile_comparison(
    ...     coords_dict={'dmo': dmo_coords, 'hydro': hydro_coords},
    ...     masses_dict={'dmo': dmo_masses, 'hydro': hydro_masses},
    ...     center=[100, 100, 100],
    ...     radius=0.5,
    ... )
    >>> ratio = analyzer.compute_ratio('hydro', 'dmo')
    """
    profiles = {}
    
    for label in coords_dict:
        profiles[label] = compute_density_profile(
            coords=coords_dict[label],
            masses=masses_dict[label],
            center=center,
            radius=radius,
            box_size=box_size,
            **kwargs,
        )
    
    return ProfileAnalyzer(profiles)
