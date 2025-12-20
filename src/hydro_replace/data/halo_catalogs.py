"""
Halo Catalog Module
===================

Functions for loading and manipulating FoF halo catalogs from simulations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
import pandas as pd

# Try to import illustris_python
try:
    from illustris_python import groupcat
    HAS_ILLUSTRIS = True
except ImportError:
    HAS_ILLUSTRIS = False

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class HaloCatalog:
    """
    Container for halo catalog data.

    This class provides convenient access to halo properties with proper
    unit conversions and filtering capabilities.

    Parameters
    ----------
    data : dict
        Raw halo catalog data from simulation.
    basePath : str
        Base path to the simulation.
    snapNum : int
        Snapshot number.
    mass_unit : float
        Unit conversion for masses (default: 1e10 Msun/h).
    length_unit : float
        Unit conversion for lengths (default: 1e-3 for kpc/h -> Mpc/h).

    Attributes
    ----------
    n_halos : int
        Total number of halos in the catalog.
    masses : ndarray
        Halo masses M_200c in Msun/h.
    radii : ndarray
        Halo radii R_200c in Mpc/h.
    positions : ndarray
        Halo center positions in Mpc/h.

    Examples
    --------
    >>> catalog = load_halo_catalog('/path/to/TNG300', 99)
    >>> print(catalog.n_halos)
    50000
    >>> massive = catalog.filter_by_mass(1e13, 1e14)
    >>> print(len(massive))
    5000
    """
    
    data: Dict[str, np.ndarray]
    basePath: str
    snapNum: int
    mass_unit: float = 1e10
    length_unit: float = 1e-3
    
    # Derived attributes
    _df: Optional[pd.DataFrame] = field(default=None, repr=False)
    
    def __post_init__(self):
        """Validate and process catalog data."""
        required_fields = ['Group_M_Crit200', 'GroupPos']
        for field in required_fields:
            if field not in self.data:
                raise ValueError(f"Required field '{field}' not found in catalog")
        
        logger.info(f"Loaded catalog with {self.n_halos} halos")
    
    @property
    def n_halos(self) -> int:
        """Total number of halos."""
        return len(self.data['Group_M_Crit200'])
    
    @property
    def masses(self) -> np.ndarray:
        """Halo masses M_200c in Msun/h."""
        return self.data['Group_M_Crit200'] * self.mass_unit
    
    @property
    def log_masses(self) -> np.ndarray:
        """Log10 of halo masses."""
        return np.log10(self.masses)
    
    @property
    def radii(self) -> np.ndarray:
        """Halo radii R_200c in Mpc/h."""
        return self.data['Group_R_Crit200'] * self.length_unit
    
    @property
    def positions(self) -> np.ndarray:
        """Halo center positions in Mpc/h."""
        return self.data['GroupPos'] * self.length_unit
    
    @property
    def velocities(self) -> Optional[np.ndarray]:
        """Halo center-of-mass velocities in km/s."""
        if 'GroupVel' in self.data:
            return self.data['GroupVel']
        return None
    
    @property
    def indices(self) -> np.ndarray:
        """Halo indices (0 to n_halos-1)."""
        return np.arange(self.n_halos)
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert catalog to pandas DataFrame.

        Returns
        -------
        df : DataFrame
            Halo catalog as DataFrame with columns:
            ['halo_id', 'mass', 'log_mass', 'radius', 'x', 'y', 'z']
        """
        if self._df is not None:
            return self._df
        
        df = pd.DataFrame({
            'halo_id': self.indices,
            'mass': self.masses,
            'log_mass': self.log_masses,
            'radius': self.radii,
            'x': self.positions[:, 0],
            'y': self.positions[:, 1],
            'z': self.positions[:, 2],
        })
        
        if self.velocities is not None:
            df['vx'] = self.velocities[:, 0]
            df['vy'] = self.velocities[:, 1]
            df['vz'] = self.velocities[:, 2]
        
        # Add additional fields if available
        if 'GroupMassType' in self.data:
            mass_types = self.data['GroupMassType'] * self.mass_unit
            df['mass_gas'] = mass_types[:, 0]
            df['mass_dm'] = mass_types[:, 1]
            df['mass_stars'] = mass_types[:, 4]
            df['mass_bh'] = mass_types[:, 5]
        
        if 'GroupLenType' in self.data:
            len_types = self.data['GroupLenType']
            df['n_gas'] = len_types[:, 0]
            df['n_dm'] = len_types[:, 1]
            df['n_stars'] = len_types[:, 4]
            df['n_bh'] = len_types[:, 5]
        
        self._df = df
        return df
    
    def filter_by_mass(
        self,
        mass_min: float = 0.0,
        mass_max: float = np.inf,
    ) -> 'HaloCatalog':
        """
        Filter halos by mass range.

        Parameters
        ----------
        mass_min : float
            Minimum mass in Msun/h.
        mass_max : float
            Maximum mass in Msun/h.

        Returns
        -------
        filtered : HaloCatalog
            New catalog with only halos in the mass range.
        """
        mask = (self.masses >= mass_min) & (self.masses < mass_max)
        return self._apply_mask(mask)
    
    def filter_by_log_mass(
        self,
        log_mass_min: float = 0.0,
        log_mass_max: float = 20.0,
    ) -> 'HaloCatalog':
        """
        Filter halos by log10 mass range.

        Parameters
        ----------
        log_mass_min : float
            Minimum log10(M/Msun/h).
        log_mass_max : float
            Maximum log10(M/Msun/h).

        Returns
        -------
        filtered : HaloCatalog
            New catalog with only halos in the mass range.
        """
        mask = (self.log_masses >= log_mass_min) & (self.log_masses < log_mass_max)
        return self._apply_mask(mask)
    
    def filter_by_indices(self, indices: np.ndarray) -> 'HaloCatalog':
        """
        Filter to specific halo indices.

        Parameters
        ----------
        indices : ndarray
            Array of halo indices to keep.

        Returns
        -------
        filtered : HaloCatalog
            New catalog with only specified halos.
        """
        mask = np.isin(self.indices, indices)
        return self._apply_mask(mask)
    
    def _apply_mask(self, mask: np.ndarray) -> 'HaloCatalog':
        """Apply boolean mask to all catalog data."""
        filtered_data = {}
        for key, value in self.data.items():
            if isinstance(value, np.ndarray) and len(value) == self.n_halos:
                filtered_data[key] = value[mask]
            else:
                filtered_data[key] = value
        
        return HaloCatalog(
            data=filtered_data,
            basePath=self.basePath,
            snapNum=self.snapNum,
            mass_unit=self.mass_unit,
            length_unit=self.length_unit,
        )
    
    def get_halo(self, halo_id: int) -> Dict[str, Any]:
        """
        Get properties for a single halo.

        Parameters
        ----------
        halo_id : int
            Halo index.

        Returns
        -------
        halo : dict
            Dictionary with halo properties.
        """
        return {
            'halo_id': halo_id,
            'mass': self.masses[halo_id],
            'log_mass': self.log_masses[halo_id],
            'radius': self.radii[halo_id],
            'position': self.positions[halo_id],
            'velocity': self.velocities[halo_id] if self.velocities is not None else None,
        }
    
    def save_hdf5(self, filepath: Union[str, Path]) -> None:
        """
        Save catalog to HDF5 file.

        Parameters
        ----------
        filepath : str or Path
            Output file path.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with h5py.File(filepath, 'w') as f:
            # Metadata
            f.attrs['basePath'] = self.basePath
            f.attrs['snapNum'] = self.snapNum
            f.attrs['n_halos'] = self.n_halos
            f.attrs['mass_unit'] = self.mass_unit
            f.attrs['length_unit'] = self.length_unit
            
            # Data
            for key, value in self.data.items():
                if isinstance(value, np.ndarray):
                    f.create_dataset(key, data=value, compression='gzip', compression_opts=4)
        
        logger.info(f"Saved catalog to {filepath}")
    
    @classmethod
    def load_hdf5(cls, filepath: Union[str, Path]) -> 'HaloCatalog':
        """
        Load catalog from HDF5 file.

        Parameters
        ----------
        filepath : str or Path
            Input file path.

        Returns
        -------
        catalog : HaloCatalog
            Loaded halo catalog.
        """
        with h5py.File(filepath, 'r') as f:
            data = {key: f[key][:] for key in f.keys()}
            basePath = f.attrs['basePath']
            snapNum = f.attrs['snapNum']
            mass_unit = f.attrs.get('mass_unit', 1e10)
            length_unit = f.attrs.get('length_unit', 1e-3)
        
        return cls(
            data=data,
            basePath=basePath,
            snapNum=snapNum,
            mass_unit=mass_unit,
            length_unit=length_unit,
        )


# =============================================================================
# Loading Functions
# =============================================================================

def load_halo_catalog(
    basePath: str,
    snapNum: int,
    fields: Optional[List[str]] = None,
    mass_unit: float = 1e10,
    length_unit: float = 1e-3,
) -> HaloCatalog:
    """
    Load halo catalog from simulation.

    Parameters
    ----------
    basePath : str
        Base path to simulation output.
    snapNum : int
        Snapshot number.
    fields : list of str, optional
        Specific fields to load. If None, loads default fields.
    mass_unit : float
        Unit conversion for masses (default: 1e10 Msun/h).
    length_unit : float
        Unit conversion for lengths (default: 1e-3 for kpc/h -> Mpc/h).

    Returns
    -------
    catalog : HaloCatalog
        Halo catalog object.

    Raises
    ------
    FileNotFoundError
        If catalog files cannot be found.
    RuntimeError
        If illustris_python is not available.

    Examples
    --------
    >>> catalog = load_halo_catalog('/path/to/TNG300', 99)
    >>> print(f"Loaded {catalog.n_halos} halos")
    >>> massive = catalog.filter_by_mass(1e13)
    """
    if fields is None:
        fields = [
            'Group_M_Crit200',
            'Group_R_Crit200',
            'GroupPos',
            'GroupVel',
            'GroupMassType',
            'GroupLenType',
            'GroupFirstSub',
        ]
    
    if HAS_ILLUSTRIS:
        data = groupcat.loadHalos(basePath, snapNum, fields=fields)
    else:
        # Fallback: load directly from HDF5
        data = _load_catalog_hdf5(basePath, snapNum, fields)
    
    return HaloCatalog(
        data=data,
        basePath=basePath,
        snapNum=snapNum,
        mass_unit=mass_unit,
        length_unit=length_unit,
    )


def _load_catalog_hdf5(
    basePath: str,
    snapNum: int,
    fields: List[str],
) -> Dict[str, np.ndarray]:
    """
    Load halo catalog directly from HDF5 (fallback).

    Parameters
    ----------
    basePath : str
        Base path to simulation output.
    snapNum : int
        Snapshot number.
    fields : list of str
        Fields to load.

    Returns
    -------
    data : dict
        Dictionary with catalog data.
    """
    # Try different patterns
    patterns = [
        f"{basePath}/groups_{snapNum:03d}/fof_subhalo_tab_{snapNum:03d}.0.hdf5",
        f"{basePath}/fof_subhalo_tab_{snapNum:03d}.hdf5",
    ]
    
    for pattern in patterns:
        if Path(pattern).exists():
            break
    else:
        raise FileNotFoundError(f"Could not find catalog for snapshot {snapNum}")
    
    data = {}
    with h5py.File(pattern, 'r') as f:
        if 'Group' not in f:
            raise ValueError(f"No 'Group' dataset in {pattern}")
        
        for field in fields:
            if field in f['Group']:
                data[field] = f['Group'][field][:]
            else:
                logger.warning(f"Field '{field}' not found in catalog")
    
    return data


def filter_by_mass(
    catalog: HaloCatalog,
    mass_min: float = 0.0,
    mass_max: float = np.inf,
) -> HaloCatalog:
    """
    Filter halo catalog by mass range.

    This is a convenience function wrapping HaloCatalog.filter_by_mass().

    Parameters
    ----------
    catalog : HaloCatalog
        Input halo catalog.
    mass_min : float
        Minimum mass in Msun/h.
    mass_max : float
        Maximum mass in Msun/h.

    Returns
    -------
    filtered : HaloCatalog
        Filtered catalog.
    """
    return catalog.filter_by_mass(mass_min, mass_max)


def get_mass_bins(
    catalog: HaloCatalog,
    bins: List[Tuple[float, float]],
) -> Dict[str, np.ndarray]:
    """
    Get halo indices for multiple mass bins.

    Parameters
    ----------
    catalog : HaloCatalog
        Halo catalog.
    bins : list of tuples
        List of (min_mass, max_mass) tuples in Msun/h.

    Returns
    -------
    bin_indices : dict
        Dictionary mapping bin labels to arrays of halo indices.
    """
    result = {}
    for i, (mass_min, mass_max) in enumerate(bins):
        mask = (catalog.masses >= mass_min) & (catalog.masses < mass_max)
        label = f"M{np.log10(mass_min):.1f}-{np.log10(mass_max):.1f}"
        result[label] = catalog.indices[mask]
    
    return result


def get_cumulative_mass_bins(
    catalog: HaloCatalog,
    thresholds: List[float],
) -> Dict[str, np.ndarray]:
    """
    Get halo indices for cumulative mass bins (>= threshold).

    Parameters
    ----------
    catalog : HaloCatalog
        Halo catalog.
    thresholds : list of float
        Mass thresholds in Msun/h.

    Returns
    -------
    bin_indices : dict
        Dictionary mapping bin labels to arrays of halo indices.
    """
    result = {}
    for threshold in thresholds:
        mask = catalog.masses >= threshold
        label = f"M_gt{np.log10(threshold):.1f}"
        result[label] = catalog.indices[mask]
    
    return result
