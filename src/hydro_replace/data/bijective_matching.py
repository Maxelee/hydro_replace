"""
Bijective Halo Matching Module
==============================

Performs bijective (one-to-one) matching between hydro and DMO halo catalogs
using the most-bound particle method.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
import pandas as pd

# Try to import MPI
try:
    from mpi4py import MPI
    HAS_MPI = True
except ImportError:
    HAS_MPI = False

# Try to import illustris_python
try:
    from illustris_python import groupcat, snapshot
    HAS_ILLUSTRIS = True
except ImportError:
    HAS_ILLUSTRIS = False

from .halo_catalogs import HaloCatalog, load_halo_catalog

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class MatchedCatalog:
    """
    Container for matched halo pairs between DMO and hydro simulations.

    Attributes
    ----------
    dmo_indices : ndarray
        Indices of matched DMO halos.
    hydro_indices : ndarray
        Indices of matched hydro halos.
    overlap_fractions : ndarray
        Overlap fraction for each match (0-1).
    dmo_masses : ndarray
        DMO halo masses in Msun/h.
    hydro_masses : ndarray
        Hydro halo masses in Msun/h.
    dmo_positions : ndarray
        DMO halo positions (N, 3) in Mpc/h.
    hydro_positions : ndarray
        Hydro halo positions (N, 3) in Mpc/h.
    separations : ndarray
        Separations between matched pairs in Mpc/h.
    """
    
    dmo_indices: np.ndarray
    hydro_indices: np.ndarray
    overlap_fractions: np.ndarray
    dmo_masses: np.ndarray
    hydro_masses: np.ndarray
    dmo_positions: np.ndarray
    hydro_positions: np.ndarray
    dmo_radii: np.ndarray
    hydro_radii: np.ndarray
    separations: np.ndarray
    
    # Metadata
    n_bound_particles: int = 100
    min_overlap_fraction: float = 0.5
    min_halo_mass: float = 1e12
    
    @property
    def n_matches(self) -> int:
        """Number of matched pairs."""
        return len(self.dmo_indices)
    
    @property
    def mass_ratios(self) -> np.ndarray:
        """Mass ratio (hydro/DMO) for each match."""
        return self.hydro_masses / self.dmo_masses
    
    @property
    def statistics(self) -> Dict[str, float]:
        """Compute matching statistics."""
        return {
            'n_matches': self.n_matches,
            'mean_overlap': float(np.mean(self.overlap_fractions)),
            'median_overlap': float(np.median(self.overlap_fractions)),
            'min_overlap': float(np.min(self.overlap_fractions)),
            'max_overlap': float(np.max(self.overlap_fractions)),
            'mean_mass_ratio': float(np.mean(self.mass_ratios)),
            'std_mass_ratio': float(np.std(self.mass_ratios)),
            'mean_separation': float(np.mean(self.separations)),
            'max_separation': float(np.max(self.separations)),
        }
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert to pandas DataFrame.

        Returns
        -------
        df : DataFrame
            Matched catalog as DataFrame.
        """
        return pd.DataFrame({
            'dmo_id': self.dmo_indices,
            'hydro_id': self.hydro_indices,
            'overlap': self.overlap_fractions,
            'dmo_mass': self.dmo_masses,
            'hydro_mass': self.hydro_masses,
            'mass_ratio': self.mass_ratios,
            'dmo_x': self.dmo_positions[:, 0],
            'dmo_y': self.dmo_positions[:, 1],
            'dmo_z': self.dmo_positions[:, 2],
            'hydro_x': self.hydro_positions[:, 0],
            'hydro_y': self.hydro_positions[:, 1],
            'hydro_z': self.hydro_positions[:, 2],
            'dmo_radius': self.dmo_radii,
            'hydro_radius': self.hydro_radii,
            'separation': self.separations,
        })
    
    def filter_by_mass(
        self,
        mass_min: float = 0.0,
        mass_max: float = np.inf,
        use_dmo: bool = True,
    ) -> 'MatchedCatalog':
        """
        Filter matches by mass range.

        Parameters
        ----------
        mass_min, mass_max : float
            Mass range in Msun/h.
        use_dmo : bool
            If True, filter by DMO mass. Otherwise use hydro mass.

        Returns
        -------
        filtered : MatchedCatalog
            Filtered catalog.
        """
        masses = self.dmo_masses if use_dmo else self.hydro_masses
        mask = (masses >= mass_min) & (masses < mass_max)
        return self._apply_mask(mask)
    
    def _apply_mask(self, mask: np.ndarray) -> 'MatchedCatalog':
        """Apply boolean mask to all arrays."""
        return MatchedCatalog(
            dmo_indices=self.dmo_indices[mask],
            hydro_indices=self.hydro_indices[mask],
            overlap_fractions=self.overlap_fractions[mask],
            dmo_masses=self.dmo_masses[mask],
            hydro_masses=self.hydro_masses[mask],
            dmo_positions=self.dmo_positions[mask],
            hydro_positions=self.hydro_positions[mask],
            dmo_radii=self.dmo_radii[mask],
            hydro_radii=self.hydro_radii[mask],
            separations=self.separations[mask],
            n_bound_particles=self.n_bound_particles,
            min_overlap_fraction=self.min_overlap_fraction,
            min_halo_mass=self.min_halo_mass,
        )
    
    def save_hdf5(self, filepath: Union[str, Path]) -> None:
        """
        Save matched catalog to HDF5 file.

        Parameters
        ----------
        filepath : str or Path
            Output file path.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with h5py.File(filepath, 'w') as f:
            # Metadata
            f.attrs['n_matches'] = self.n_matches
            f.attrs['n_bound_particles'] = self.n_bound_particles
            f.attrs['min_overlap_fraction'] = self.min_overlap_fraction
            f.attrs['min_halo_mass'] = self.min_halo_mass
            
            # Statistics
            stats = self.statistics
            for key, value in stats.items():
                f.attrs[f'stat_{key}'] = value
            
            # Data
            f.create_dataset('dmo_indices', data=self.dmo_indices, compression='gzip')
            f.create_dataset('hydro_indices', data=self.hydro_indices, compression='gzip')
            f.create_dataset('overlap_fractions', data=self.overlap_fractions, compression='gzip')
            f.create_dataset('dmo_masses', data=self.dmo_masses, compression='gzip')
            f.create_dataset('hydro_masses', data=self.hydro_masses, compression='gzip')
            f.create_dataset('dmo_positions', data=self.dmo_positions, compression='gzip')
            f.create_dataset('hydro_positions', data=self.hydro_positions, compression='gzip')
            f.create_dataset('dmo_radii', data=self.dmo_radii, compression='gzip')
            f.create_dataset('hydro_radii', data=self.hydro_radii, compression='gzip')
            f.create_dataset('separations', data=self.separations, compression='gzip')
        
        logger.info(f"Saved matched catalog ({self.n_matches} pairs) to {filepath}")
    
    @classmethod
    def load_hdf5(cls, filepath: Union[str, Path]) -> 'MatchedCatalog':
        """
        Load matched catalog from HDF5 file.

        Parameters
        ----------
        filepath : str or Path
            Input file path.

        Returns
        -------
        catalog : MatchedCatalog
            Loaded matched catalog.
        """
        with h5py.File(filepath, 'r') as f:
            return cls(
                dmo_indices=f['dmo_indices'][:],
                hydro_indices=f['hydro_indices'][:],
                overlap_fractions=f['overlap_fractions'][:],
                dmo_masses=f['dmo_masses'][:],
                hydro_masses=f['hydro_masses'][:],
                dmo_positions=f['dmo_positions'][:],
                hydro_positions=f['hydro_positions'][:],
                dmo_radii=f['dmo_radii'][:],
                hydro_radii=f['hydro_radii'][:],
                separations=f['separations'][:],
                n_bound_particles=f.attrs.get('n_bound_particles', 100),
                min_overlap_fraction=f.attrs.get('min_overlap_fraction', 0.5),
                min_halo_mass=f.attrs.get('min_halo_mass', 1e12),
            )
    
    @classmethod
    def load_npz(cls, filepath: Union[str, Path]) -> 'MatchedCatalog':
        """
        Load matched catalog from NPZ file (legacy format).

        Parameters
        ----------
        filepath : str or Path
            Input file path.

        Returns
        -------
        catalog : MatchedCatalog
            Loaded matched catalog.
        """
        data = np.load(filepath)
        
        # Compute separations if not present
        if 'separations' not in data:
            dmo_pos = data['dmo_positions']
            hydro_pos = data['hydro_positions']
            separations = np.linalg.norm(dmo_pos - hydro_pos, axis=1)
        else:
            separations = data['separations']
        
        return cls(
            dmo_indices=data['dmo_indices'],
            hydro_indices=data['hydro_indices'],
            overlap_fractions=data['overlap_fractions'],
            dmo_masses=data['dmo_masses'],
            hydro_masses=data['hydro_masses'],
            dmo_positions=data['dmo_positions'],
            hydro_positions=data['hydro_positions'],
            dmo_radii=data.get('dmo_radii', np.zeros(len(data['dmo_indices']))),
            hydro_radii=data.get('hydro_radii', np.zeros(len(data['dmo_indices']))),
            separations=separations,
            n_bound_particles=int(data.get('n_bound_particles', 100)),
            min_overlap_fraction=float(data.get('min_overlap_fraction', 0.5)),
            min_halo_mass=float(data.get('min_halo_mass', 1e12)),
        )


# =============================================================================
# Matcher Class
# =============================================================================

class BijectiveMatcher:
    """
    Performs bijective matching between DMO and hydro simulations.

    This class uses the most-bound particle method to find one-to-one
    correspondences between halos in DMO and hydro simulations.

    Parameters
    ----------
    dmo_basePath : str
        Path to DMO simulation output.
    hydro_basePath : str
        Path to hydro simulation output.
    snapNum : int
        Snapshot number.
    n_bound_particles : int
        Number of most-bound particles to use for matching.
    min_overlap_fraction : float
        Minimum overlap fraction for valid match.
    min_halo_mass : float
        Minimum halo mass in Msun/h.
    min_particles : int
        Minimum particles per halo.

    Examples
    --------
    >>> matcher = BijectiveMatcher(
    ...     dmo_basePath='/path/to/TNG300-Dark',
    ...     hydro_basePath='/path/to/TNG300',
    ...     snapNum=99,
    ...     min_halo_mass=1e12,
    ... )
    >>> matched = matcher.run()
    >>> print(f"Found {matched.n_matches} matches")
    """
    
    def __init__(
        self,
        dmo_basePath: str,
        hydro_basePath: str,
        snapNum: int,
        n_bound_particles: int = 100,
        min_overlap_fraction: float = 0.5,
        min_halo_mass: float = 1e12,
        min_particles: int = 1000,
        mass_unit: float = 1e10,
        length_unit: float = 1e-3,
    ):
        self.dmo_basePath = dmo_basePath
        self.hydro_basePath = hydro_basePath
        self.snapNum = snapNum
        self.n_bound_particles = n_bound_particles
        self.min_overlap_fraction = min_overlap_fraction
        self.min_halo_mass = min_halo_mass
        self.min_particles = min_particles
        self.mass_unit = mass_unit
        self.length_unit = length_unit
        
        # MPI setup
        if HAS_MPI:
            self.comm = MPI.COMM_WORLD
            self.rank = self.comm.Get_rank()
            self.size = self.comm.Get_size()
        else:
            self.comm = None
            self.rank = 0
            self.size = 1
        
        # Data containers
        self._dmo_catalog: Optional[HaloCatalog] = None
        self._hydro_catalog: Optional[HaloCatalog] = None
        self._dmo_particles: Optional[Dict] = None
        self._hydro_particles: Optional[Dict] = None
        self._dmo_offsets: Optional[np.ndarray] = None
        self._hydro_offsets: Optional[np.ndarray] = None
    
    @property
    def is_root(self) -> bool:
        """Check if this is the root MPI rank."""
        return self.rank == 0
    
    def _log(self, msg: str, level: str = 'info') -> None:
        """Log message (only on root rank)."""
        if self.is_root:
            getattr(logger, level)(msg)
    
    def load_catalogs(self) -> None:
        """Load halo catalogs for both simulations."""
        self._log("Loading halo catalogs...")
        
        if not HAS_ILLUSTRIS:
            raise RuntimeError("illustris_python required for bijective matching")
        
        # Load DMO catalog
        fields = ['Group_M_Crit200', 'Group_R_Crit200', 'GroupPos', 
                  'GroupVel', 'GroupLenType']
        
        dmo_data = groupcat.loadHalos(self.dmo_basePath, self.snapNum, fields=fields)
        hydro_data = groupcat.loadHalos(self.hydro_basePath, self.snapNum, fields=fields)
        
        # Create HaloCatalog objects
        self._dmo_catalog = HaloCatalog(
            data=dmo_data,
            basePath=self.dmo_basePath,
            snapNum=self.snapNum,
            mass_unit=self.mass_unit,
            length_unit=self.length_unit,
        )
        
        self._hydro_catalog = HaloCatalog(
            data=hydro_data,
            basePath=self.hydro_basePath,
            snapNum=self.snapNum,
            mass_unit=self.mass_unit,
            length_unit=self.length_unit,
        )
        
        # Apply mass cuts
        self._dmo_catalog = self._dmo_catalog.filter_by_mass(self.min_halo_mass)
        self._hydro_catalog = self._hydro_catalog.filter_by_mass(self.min_halo_mass)
        
        self._log(f"  DMO halos above mass cut: {self._dmo_catalog.n_halos}")
        self._log(f"  Hydro halos above mass cut: {self._hydro_catalog.n_halos}")
    
    def load_particles(self) -> None:
        """Load all DM particle IDs (memory-intensive but fast)."""
        self._log("Loading DM particles...")
        
        # Load all DMO DM particles
        self._dmo_particles = snapshot.loadSubset(
            self.dmo_basePath, self.snapNum, 'dm', ['ParticleIDs']
        )
        
        # Load all hydro DM particles
        self._hydro_particles = snapshot.loadSubset(
            self.hydro_basePath, self.snapNum, 'dm', ['ParticleIDs']
        )
        
        # Build offset arrays for fast halo lookup
        dmo_lengths = groupcat.loadHalos(
            self.dmo_basePath, self.snapNum, fields=['GroupLenType']
        )['GroupLenType'][:, 1]  # DM is type 1
        
        hydro_lengths = groupcat.loadHalos(
            self.hydro_basePath, self.snapNum, fields=['GroupLenType']
        )['GroupLenType'][:, 1]
        
        self._dmo_offsets = np.zeros(len(dmo_lengths) + 1, dtype=np.int64)
        self._dmo_offsets[1:] = np.cumsum(dmo_lengths)
        
        self._hydro_offsets = np.zeros(len(hydro_lengths) + 1, dtype=np.int64)
        self._hydro_offsets[1:] = np.cumsum(hydro_lengths)
        
        self._log(f"  Loaded {len(self._dmo_particles)} DMO particles")
        self._log(f"  Loaded {len(self._hydro_particles)} hydro particles")
    
    def _get_halo_particles(
        self,
        halo_idx: int,
        is_dmo: bool = True,
    ) -> np.ndarray:
        """Get particle IDs for a specific halo."""
        if is_dmo:
            particles = self._dmo_particles
            offsets = self._dmo_offsets
        else:
            particles = self._hydro_particles
            offsets = self._hydro_offsets
        
        start = offsets[halo_idx]
        end = offsets[halo_idx + 1]
        
        return particles[start:end]
    
    def _find_best_match(
        self,
        source_ids: np.ndarray,
        target_particles: np.ndarray,
        target_offsets: np.ndarray,
    ) -> Tuple[Optional[int], int]:
        """
        Find which target halo has the most overlap with source_ids.

        Uses histogram binning for O(N) complexity.

        Returns
        -------
        best_halo : int or None
            Index of best matching halo, or None if no match.
        overlap : int
            Number of overlapping particles.
        """
        # Find indices in target that match source IDs
        matched_indices = np.nonzero(np.isin(target_particles, source_ids))[0]
        
        if len(matched_indices) == 0:
            return None, 0
        
        # Histogram bin matched particles into halos
        histogram, _ = np.histogram(matched_indices, bins=target_offsets)
        
        # Find halo with maximum overlap
        best_halo = int(np.argmax(histogram))
        max_overlap = int(histogram[best_halo])
        
        return best_halo, max_overlap
    
    def _match_single_halo(self, dmo_idx: int) -> Optional[Tuple[int, float]]:
        """
        Perform bijective matching for a single DMO halo.

        Returns
        -------
        result : tuple or None
            (hydro_idx, overlap_fraction) if match found, else None.
        """
        # Get DMO halo particles
        dmo_particles = self._get_halo_particles(dmo_idx, is_dmo=True)
        
        if len(dmo_particles) < self.min_particles:
            return None
        
        # Forward match: DMO -> Hydro
        hydro_idx, forward_overlap = self._find_best_match(
            dmo_particles,
            self._hydro_particles,
            self._hydro_offsets,
        )
        
        if hydro_idx is None:
            return None
        
        forward_fraction = forward_overlap / len(dmo_particles)
        if forward_fraction < self.min_overlap_fraction:
            return None
        
        # Get hydro halo particles for reverse match
        hydro_particles = self._get_halo_particles(hydro_idx, is_dmo=False)
        
        if len(hydro_particles) < self.min_particles:
            return None
        
        # Reverse match: Hydro -> DMO
        dmo_reverse_idx, reverse_overlap = self._find_best_match(
            hydro_particles,
            self._dmo_particles,
            self._dmo_offsets,
        )
        
        # Check if bijective
        if dmo_reverse_idx != dmo_idx:
            return None
        
        reverse_fraction = reverse_overlap / len(hydro_particles)
        if reverse_fraction < self.min_overlap_fraction:
            return None
        
        # Use minimum overlap as quality metric
        final_fraction = min(forward_fraction, reverse_fraction)
        
        return (hydro_idx, final_fraction)
    
    def run(self) -> MatchedCatalog:
        """
        Run the bijective matching algorithm.

        Returns
        -------
        matched : MatchedCatalog
            Matched halo catalog.
        """
        t_start = time.time()
        
        # Load data
        self.load_catalogs()
        self.load_particles()
        
        # Get indices to process
        dmo_indices = self._dmo_catalog.indices
        
        # Distribute work across MPI ranks
        my_indices = dmo_indices[self.rank::self.size]
        self._log(f"Processing {len(my_indices)} halos on {self.size} ranks...")
        
        # Match halos
        my_matches = {}
        for i, dmo_idx in enumerate(my_indices):
            if i % 100 == 0:
                print(f"Rank {self.rank}: {i}/{len(my_indices)} halos processed")
            
            result = self._match_single_halo(dmo_idx)
            
            if result is not None:
                hydro_idx, overlap = result
                my_matches[dmo_idx] = (hydro_idx, overlap)
        
        print(f"Rank {self.rank}: Found {len(my_matches)} matches")
        
        # Gather results
        if HAS_MPI and self.size > 1:
            all_matches = self.comm.gather(my_matches, root=0)
            self.comm.Barrier()
            
            if self.is_root:
                # Combine all matches
                bijective_matches = {}
                for matches in all_matches:
                    bijective_matches.update(matches)
            else:
                return None
        else:
            bijective_matches = my_matches
        
        # Build output catalog
        if len(bijective_matches) == 0:
            self._log("WARNING: No bijective matches found!", 'warning')
            raise RuntimeError("No bijective matches found")
        
        dmo_matched = np.array(list(bijective_matches.keys()))
        hydro_matched = np.array([h for h, _ in bijective_matches.values()])
        overlaps = np.array([o for _, o in bijective_matches.values()])
        
        # Get properties for matched halos
        dmo_masses = self._dmo_catalog.masses[dmo_matched]
        hydro_masses = self._hydro_catalog.masses[hydro_matched]
        dmo_positions = self._dmo_catalog.positions[dmo_matched]
        hydro_positions = self._hydro_catalog.positions[hydro_matched]
        dmo_radii = self._dmo_catalog.radii[dmo_matched]
        hydro_radii = self._hydro_catalog.radii[hydro_matched]
        
        # Compute separations
        separations = np.linalg.norm(dmo_positions - hydro_positions, axis=1)
        
        result = MatchedCatalog(
            dmo_indices=dmo_matched,
            hydro_indices=hydro_matched,
            overlap_fractions=overlaps,
            dmo_masses=dmo_masses,
            hydro_masses=hydro_masses,
            dmo_positions=dmo_positions,
            hydro_positions=hydro_positions,
            dmo_radii=dmo_radii,
            hydro_radii=hydro_radii,
            separations=separations,
            n_bound_particles=self.n_bound_particles,
            min_overlap_fraction=self.min_overlap_fraction,
            min_halo_mass=self.min_halo_mass,
        )
        
        t_total = time.time() - t_start
        self._log(f"Matching complete: {result.n_matches} pairs in {t_total/60:.1f} min")
        self._log(f"  Mean overlap: {result.statistics['mean_overlap']:.3f}")
        self._log(f"  Mean mass ratio: {result.statistics['mean_mass_ratio']:.3f}")
        
        return result


# =============================================================================
# Convenience Functions
# =============================================================================

def bijective_halo_matching(
    hydro_catalog: Union[HaloCatalog, pd.DataFrame],
    dmo_catalog: Union[HaloCatalog, pd.DataFrame],
    n_most_bound: int = 100,
    max_distance: float = 0.1,
    min_overlap_fraction: float = 0.5,
    hydro_basePath: Optional[str] = None,
    dmo_basePath: Optional[str] = None,
    snapNum: Optional[int] = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Perform bijective halo matching between hydro and DMO simulations.

    Uses most-bound particle method: for each hydro halo, find the DMO halo
    containing the most bound particles from the hydro halo.

    Parameters
    ----------
    hydro_catalog : HaloCatalog or DataFrame
        Halo catalog from hydro simulation with columns:
        ['halo_id', 'M_200c', 'R_200c', 'x', 'y', 'z']
    dmo_catalog : HaloCatalog or DataFrame
        Halo catalog from DMO simulation (same structure).
    n_most_bound : int
        Number of most bound particles to use for matching.
    max_distance : float
        Maximum allowed center-of-mass distance for valid match (in R_200c).
    min_overlap_fraction : float
        Minimum particle overlap fraction for valid match.
    hydro_basePath : str, optional
        Path to hydro simulation (required if catalogs are DataFrames).
    dmo_basePath : str, optional
        Path to DMO simulation.
    snapNum : int, optional
        Snapshot number.

    Returns
    -------
    matched_catalog : DataFrame
        Catalog with columns: ['hydro_id', 'dmo_id', 'M_hydro', 'M_dmo',
                               'R_hydro', 'R_dmo', 'separation']
    statistics : dict
        Matching statistics: {'success_rate': float, 'mean_mass_ratio': float,
                              'median_separation': float}

    Raises
    ------
    ValueError
        If catalogs are empty or missing required columns.
    RuntimeError
        If matching success rate < 90%.

    Examples
    --------
    >>> hydro_cat = load_halo_catalog('TNG300-1', snapshot=99)
    >>> dmo_cat = load_halo_catalog('TNG300-Dark', snapshot=99)
    >>> matched, stats = bijective_halo_matching(hydro_cat, dmo_cat)
    >>> print(f"Success rate: {stats['success_rate']:.1%}")
    """
    # Validate inputs
    if isinstance(hydro_catalog, HaloCatalog):
        if hydro_basePath is None:
            hydro_basePath = hydro_catalog.basePath
        if snapNum is None:
            snapNum = hydro_catalog.snapNum
    
    if isinstance(dmo_catalog, HaloCatalog):
        if dmo_basePath is None:
            dmo_basePath = dmo_catalog.basePath
    
    if hydro_basePath is None or dmo_basePath is None or snapNum is None:
        raise ValueError("Must provide basePath and snapNum if using DataFrames")
    
    # Use BijectiveMatcher
    matcher = BijectiveMatcher(
        dmo_basePath=dmo_basePath,
        hydro_basePath=hydro_basePath,
        snapNum=snapNum,
        n_bound_particles=n_most_bound,
        min_overlap_fraction=min_overlap_fraction,
    )
    
    matched = matcher.run()
    
    if matched is None:
        raise RuntimeError("Matching failed (possibly not root rank in MPI)")
    
    # Convert to DataFrame format expected by user
    df = matched.to_dataframe()
    
    # Compute success rate
    if isinstance(hydro_catalog, HaloCatalog):
        n_input = hydro_catalog.n_halos
    else:
        n_input = len(hydro_catalog)
    
    success_rate = matched.n_matches / n_input
    
    statistics = {
        'success_rate': success_rate,
        'mean_mass_ratio': matched.statistics['mean_mass_ratio'],
        'median_separation': np.median(matched.separations),
        **matched.statistics,
    }
    
    # Check success rate
    if success_rate < 0.5:
        logger.warning(f"Low matching success rate: {success_rate:.1%}")
    
    return df, statistics
