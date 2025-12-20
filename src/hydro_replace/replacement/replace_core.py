"""
Core Halo Replacement Module
============================

Main algorithm for replacing DMO particles with hydro particles.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
from scipy.spatial import cKDTree

# Try to import MPI
try:
    from mpi4py import MPI
    HAS_MPI = True
except ImportError:
    HAS_MPI = False

# Try to import MAS_library for pixelization
try:
    import MAS_library as MASL
    HAS_MAS = True
except ImportError:
    HAS_MAS = False

from ..data.bijective_matching import MatchedCatalog
from ..data.halo_catalogs import HaloCatalog

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ReplacementResult:
    """
    Container for replacement operation results.

    Attributes
    ----------
    dmo_background_coords : ndarray
        Coordinates of DMO particles NOT in replaced halos.
    dmo_background_masses : ndarray
        Masses of DMO background particles.
    hydro_replacement_coords : ndarray
        Coordinates of hydro particles in replaced halos.
    hydro_replacement_masses : ndarray
        Masses of hydro replacement particles.
    n_halos_replaced : int
        Number of halos replaced.
    mass_bin_label : str
        Label for the mass bin used.
    radius_mult : float
        Radius multiplier used.
    """
    
    dmo_background_coords: np.ndarray
    dmo_background_masses: np.ndarray
    hydro_replacement_coords: np.ndarray
    hydro_replacement_masses: np.ndarray
    n_halos_replaced: int
    mass_bin_label: str
    radius_mult: float
    
    @property
    def n_dmo_background(self) -> int:
        """Number of DMO background particles."""
        return len(self.dmo_background_coords)
    
    @property
    def n_hydro_replacement(self) -> int:
        """Number of hydro replacement particles."""
        return len(self.hydro_replacement_coords)
    
    @property
    def total_particles(self) -> int:
        """Total number of particles."""
        return self.n_dmo_background + self.n_hydro_replacement
    
    @property
    def combined_coords(self) -> np.ndarray:
        """Combined particle coordinates."""
        return np.concatenate([
            self.dmo_background_coords,
            self.hydro_replacement_coords
        ])
    
    @property
    def combined_masses(self) -> np.ndarray:
        """Combined particle masses."""
        return np.concatenate([
            self.dmo_background_masses,
            self.hydro_replacement_masses
        ])
    
    @property
    def mass_dmo_background(self) -> float:
        """Total mass of DMO background."""
        return float(self.dmo_background_masses.sum())
    
    @property
    def mass_hydro_replacement(self) -> float:
        """Total mass of hydro replacement."""
        return float(self.hydro_replacement_masses.sum())
    
    def save_hdf5(self, filepath: Union[str, Path]) -> None:
        """Save result to HDF5 file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with h5py.File(filepath, 'w') as f:
            # Metadata
            f.attrs['n_halos_replaced'] = self.n_halos_replaced
            f.attrs['mass_bin_label'] = self.mass_bin_label
            f.attrs['radius_mult'] = self.radius_mult
            f.attrs['n_dmo_background'] = self.n_dmo_background
            f.attrs['n_hydro_replacement'] = self.n_hydro_replacement
            
            # Data
            f.create_dataset('dmo_background_coords', 
                           data=self.dmo_background_coords,
                           compression='gzip', compression_opts=4)
            f.create_dataset('dmo_background_masses',
                           data=self.dmo_background_masses,
                           compression='gzip', compression_opts=4)
            f.create_dataset('hydro_replacement_coords',
                           data=self.hydro_replacement_coords,
                           compression='gzip', compression_opts=4)
            f.create_dataset('hydro_replacement_masses',
                           data=self.hydro_replacement_masses,
                           compression='gzip', compression_opts=4)
        
        logger.info(f"Saved replacement result to {filepath}")
    
    @classmethod
    def load_hdf5(cls, filepath: Union[str, Path]) -> 'ReplacementResult':
        """Load result from HDF5 file."""
        with h5py.File(filepath, 'r') as f:
            return cls(
                dmo_background_coords=f['dmo_background_coords'][:],
                dmo_background_masses=f['dmo_background_masses'][:],
                hydro_replacement_coords=f['hydro_replacement_coords'][:],
                hydro_replacement_masses=f['hydro_replacement_masses'][:],
                n_halos_replaced=f.attrs['n_halos_replaced'],
                mass_bin_label=f.attrs['mass_bin_label'],
                radius_mult=f.attrs['radius_mult'],
            )
    
    def pixelize_3d(
        self,
        box_size: float,
        grid_size: int = 1024,
        mas: str = 'CIC',
    ) -> np.ndarray:
        """
        Pixelize combined particle data to 3D grid.

        Parameters
        ----------
        box_size : float
            Box size in Mpc/h.
        grid_size : int
            Grid resolution.
        mas : str
            Mass assignment scheme ('CIC', 'NGP', 'TSC').

        Returns
        -------
        grid : ndarray
            3D density grid.
        """
        if not HAS_MAS:
            raise ImportError("MAS_library required for pixelization")
        
        grid = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)
        
        coords = self.combined_coords.astype(np.float32)
        masses = self.combined_masses.astype(np.float32)
        
        MASL.MA(coords, grid, box_size, MAS=mas, W=masses, verbose=False)
        
        return grid


# =============================================================================
# Halo Replacer Class
# =============================================================================

class HaloReplacer:
    """
    Replace DMO particles with hydro particles around matched halos.

    This class performs the core replacement operation, removing DMO
    particles within a specified radius of halo centers and inserting
    hydro particles from matched halos.

    Parameters
    ----------
    matched_catalog : MatchedCatalog
        Catalog of matched DMO-hydro halo pairs.
    dmo_coords : ndarray
        DMO particle coordinates (N, 3).
    dmo_masses : ndarray
        DMO particle masses (N,).
    hydro_coords : ndarray
        Hydro particle coordinates (M, 3).
    hydro_masses : ndarray
        Hydro particle masses (M,).
    box_size : float
        Simulation box size in Mpc/h.

    Examples
    --------
    >>> replacer = HaloReplacer(
    ...     matched_catalog=matched,
    ...     dmo_coords=dmo_coords,
    ...     dmo_masses=dmo_masses,
    ...     hydro_coords=hydro_coords,
    ...     hydro_masses=hydro_masses,
    ...     box_size=205.0,
    ... )
    >>> result = replacer.replace(
    ...     mass_min=1e13,
    ...     mass_max=1e14,
    ...     radius_mult=3.0,
    ... )
    """
    
    def __init__(
        self,
        matched_catalog: MatchedCatalog,
        dmo_coords: np.ndarray,
        dmo_masses: np.ndarray,
        hydro_coords: np.ndarray,
        hydro_masses: np.ndarray,
        box_size: float,
    ):
        self.matched_catalog = matched_catalog
        self.dmo_coords = dmo_coords.astype(np.float32)
        self.dmo_masses = dmo_masses.astype(np.float32)
        self.hydro_coords = hydro_coords.astype(np.float32)
        self.hydro_masses = hydro_masses.astype(np.float32)
        self.box_size = box_size
        
        # Build KD-trees
        logger.info("Building KD-trees for particle data...")
        self.dmo_tree = cKDTree(self.dmo_coords)
        self.hydro_tree = cKDTree(self.hydro_coords)
        
        logger.info(f"  DMO particles: {len(self.dmo_coords):,}")
        logger.info(f"  Hydro particles: {len(self.hydro_coords):,}")
    
    def replace(
        self,
        mass_min: float = 0.0,
        mass_max: float = np.inf,
        radius_mult: float = 3.0,
        mass_bin_label: str = "all",
        use_hydro_center: bool = True,
    ) -> ReplacementResult:
        """
        Perform halo replacement for a given mass bin.

        Parameters
        ----------
        mass_min : float
            Minimum halo mass in Msun/h.
        mass_max : float
            Maximum halo mass in Msun/h.
        radius_mult : float
            Radius multiplier for replacement (extraction_radius = radius_mult * R_200c).
        mass_bin_label : str
            Label for this mass bin configuration.
        use_hydro_center : bool
            If True, use hydro halo center for hydro particle extraction.
            If False, use DMO center for both.

        Returns
        -------
        result : ReplacementResult
            Replacement operation result.
        """
        # Filter halos by mass
        filtered = self.matched_catalog.filter_by_mass(mass_min, mass_max)
        n_halos = filtered.n_matches
        
        logger.info(f"Replacing {n_halos} halos in mass range [{mass_min:.2e}, {mass_max:.2e}]")
        logger.info(f"  Radius multiplier: {radius_mult}")
        
        if n_halos == 0:
            logger.warning("No halos in specified mass range")
            return ReplacementResult(
                dmo_background_coords=self.dmo_coords,
                dmo_background_masses=self.dmo_masses,
                hydro_replacement_coords=np.empty((0, 3), dtype=np.float32),
                hydro_replacement_masses=np.empty(0, dtype=np.float32),
                n_halos_replaced=0,
                mass_bin_label=mass_bin_label,
                radius_mult=radius_mult,
            )
        
        # Create masks for particle selection
        dmo_keep_mask = np.ones(len(self.dmo_coords), dtype=bool)
        hydro_extract_mask = np.zeros(len(self.hydro_coords), dtype=bool)
        
        # Process each halo
        for i in range(n_halos):
            dmo_center = filtered.dmo_positions[i]
            hydro_center = filtered.hydro_positions[i]
            dmo_radius = filtered.dmo_radii[i]
            hydro_radius = filtered.hydro_radii[i]
            
            # Calculate search radii
            dmo_search_radius = radius_mult * dmo_radius
            hydro_search_radius = radius_mult * hydro_radius
            
            # Find DMO particles to remove (within DMO halo)
            dmo_indices = self._query_periodic(
                self.dmo_tree, dmo_center, dmo_search_radius
            )
            dmo_keep_mask[dmo_indices] = False
            
            # Find hydro particles to insert (within hydro halo)
            center_for_hydro = hydro_center if use_hydro_center else dmo_center
            radius_for_hydro = hydro_search_radius if use_hydro_center else dmo_search_radius
            
            hydro_indices = self._query_periodic(
                self.hydro_tree, center_for_hydro, radius_for_hydro
            )
            hydro_extract_mask[hydro_indices] = True
        
        # Apply masks
        dmo_background_coords = self.dmo_coords[dmo_keep_mask]
        dmo_background_masses = self.dmo_masses[dmo_keep_mask]
        hydro_replacement_coords = self.hydro_coords[hydro_extract_mask]
        hydro_replacement_masses = self.hydro_masses[hydro_extract_mask]
        
        result = ReplacementResult(
            dmo_background_coords=dmo_background_coords,
            dmo_background_masses=dmo_background_masses,
            hydro_replacement_coords=hydro_replacement_coords,
            hydro_replacement_masses=hydro_replacement_masses,
            n_halos_replaced=n_halos,
            mass_bin_label=mass_bin_label,
            radius_mult=radius_mult,
        )
        
        logger.info(f"  DMO background: {result.n_dmo_background:,} particles")
        logger.info(f"  Hydro replacement: {result.n_hydro_replacement:,} particles")
        logger.info(f"  Total: {result.total_particles:,} particles")
        
        return result
    
    def _query_periodic(
        self,
        tree: cKDTree,
        center: np.ndarray,
        radius: float,
    ) -> np.ndarray:
        """Query KD-tree with periodic boundary handling."""
        indices = list(tree.query_ball_point(center, radius))
        
        # Check boundaries
        for dim in range(3):
            if center[dim] - radius < 0:
                shifted = center.copy()
                shifted[dim] += self.box_size
                indices.extend(tree.query_ball_point(shifted, radius))
            
            if center[dim] + radius > self.box_size:
                shifted = center.copy()
                shifted[dim] -= self.box_size
                indices.extend(tree.query_ball_point(shifted, radius))
        
        return np.unique(indices)
    
    def replace_multiple_configs(
        self,
        mass_bins: List[Tuple[float, float, str]],
        radius_mults: List[float],
    ) -> Dict[str, ReplacementResult]:
        """
        Run replacement for multiple mass bins and radius configurations.

        Parameters
        ----------
        mass_bins : list of tuples
            List of (mass_min, mass_max, label) tuples.
        radius_mults : list of float
            List of radius multipliers.

        Returns
        -------
        results : dict
            Dictionary mapping config labels to ReplacementResult objects.
        """
        results = {}
        
        n_configs = len(mass_bins) * len(radius_mults)
        logger.info(f"Running {n_configs} replacement configurations...")
        
        for mass_min, mass_max, mass_label in mass_bins:
            for radius_mult in radius_mults:
                config_label = f"{mass_label}_R{radius_mult:.0f}"
                logger.info(f"\n--- Configuration: {config_label} ---")
                
                result = self.replace(
                    mass_min=mass_min,
                    mass_max=mass_max,
                    radius_mult=radius_mult,
                    mass_bin_label=mass_label,
                )
                
                results[config_label] = result
        
        return results


# =============================================================================
# Convenience Functions
# =============================================================================

def replace_halos(
    matched_catalog: MatchedCatalog,
    dmo_coords: np.ndarray,
    dmo_masses: np.ndarray,
    hydro_coords: np.ndarray,
    hydro_masses: np.ndarray,
    box_size: float,
    mass_min: float = 0.0,
    mass_max: float = np.inf,
    radius_mult: float = 3.0,
    mass_bin_label: str = "all",
) -> ReplacementResult:
    """
    Replace DMO particles with hydro particles around matched halos.

    This is a convenience function that creates a HaloReplacer and
    performs a single replacement operation.

    Parameters
    ----------
    matched_catalog : MatchedCatalog
        Catalog of matched DMO-hydro halo pairs.
    dmo_coords : ndarray
        DMO particle coordinates (N, 3) in Mpc/h.
    dmo_masses : ndarray
        DMO particle masses (N,) in Msun/h.
    hydro_coords : ndarray
        Hydro particle coordinates (M, 3) in Mpc/h.
    hydro_masses : ndarray
        Hydro particle masses (M,) in Msun/h.
    box_size : float
        Simulation box size in Mpc/h.
    mass_min : float
        Minimum halo mass in Msun/h.
    mass_max : float
        Maximum halo mass in Msun/h.
    radius_mult : float
        Radius multiplier for replacement.
    mass_bin_label : str
        Label for this configuration.

    Returns
    -------
    result : ReplacementResult
        Replacement operation result.

    Examples
    --------
    >>> result = replace_halos(
    ...     matched_catalog=matched,
    ...     dmo_coords=dmo_coords,
    ...     dmo_masses=dmo_masses,
    ...     hydro_coords=hydro_coords,
    ...     hydro_masses=hydro_masses,
    ...     box_size=205.0,
    ...     mass_min=1e13,
    ...     radius_mult=3.0,
    ... )
    >>> print(f"Total particles: {result.total_particles:,}")
    """
    replacer = HaloReplacer(
        matched_catalog=matched_catalog,
        dmo_coords=dmo_coords,
        dmo_masses=dmo_masses,
        hydro_coords=hydro_coords,
        hydro_masses=hydro_masses,
        box_size=box_size,
    )
    
    return replacer.replace(
        mass_min=mass_min,
        mass_max=mass_max,
        radius_mult=radius_mult,
        mass_bin_label=mass_bin_label,
    )


def pixelize_snapshot(
    coords: np.ndarray,
    masses: np.ndarray,
    box_size: float,
    grid_size: int = 1024,
    mas: str = 'CIC',
) -> np.ndarray:
    """
    Pixelize particle data to 3D density grid.

    Parameters
    ----------
    coords : ndarray
        Particle coordinates (N, 3).
    masses : ndarray
        Particle masses (N,).
    box_size : float
        Box size in Mpc/h.
    grid_size : int
        Grid resolution.
    mas : str
        Mass assignment scheme.

    Returns
    -------
    grid : ndarray
        3D density grid.
    """
    if not HAS_MAS:
        raise ImportError("MAS_library required for pixelization")
    
    grid = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)
    
    MASL.MA(
        coords.astype(np.float32),
        grid,
        np.float32(box_size),
        MAS=mas,
        W=masses.astype(np.float32),
        verbose=False,
    )
    
    return grid
